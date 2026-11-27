import base64
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ==============================================================================
# SERVER-LOCKED GROQ CONFIGURATION (Not exposed or overridable in the UI)
# ==============================================================================
LOCKED_ENDPOINT_URL = "https://api.groq.com/openai/v1"
LOCKED_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LOCKED_MODEL = "openai/gpt-oss-20b"
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
USER_ACCESS_KEY = os.getenv("USER_ACCESS_KEY", "")
AUTH_SECRET = os.getenv("AUTH_SECRET", "")
AUTH_TOKEN_TTL_SECONDS = 12 * 60 * 60

RECENT_WINDOW_MESSAGES = int(os.getenv("RECENT_WINDOW_MESSAGES", "4"))
SUMMARY_TRIGGER_MESSAGES = int(os.getenv("SUMMARY_TRIGGER_MESSAGES", "8"))
SUMMARY_MAX_TOKENS = 180
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "650"))
SESSION_MESSAGE_CAP = int(os.getenv("SESSION_MESSAGE_CAP", "50"))
SESSION_TOKEN_CAP = int(os.getenv("SESSION_TOKEN_CAP", "20000"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "180.0"))

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = (
    STATIC_DIR / "index.html"
    if (STATIC_DIR / "index.html").exists()
    else BASE_DIR / "index.html"
)

SYSTEM_PROMPT = (
    "You are OmniGPT, a concise DevOps and SRE assistant. Return direct commands, "
    "minimal diffs, and terse bullet points. Omit introductions and conclusions."
)
ACTIVE_SYSTEM_PROMPT = SYSTEM_PROMPT
ACTIVE_TEMPERATURE = 0.3
ACTIVE_DEFAULT_MAX_TOKENS = DEFAULT_MAX_TOKENS
ACTIVE_SESSION_MESSAGE_CAP = SESSION_MESSAGE_CAP
ACTIVE_SESSION_TOKEN_CAP = SESSION_TOKEN_CAP
SESSION_USAGE: dict[str, dict[str, int]] = {}
USAGE_STATS: dict[str, Any] = {
    "request_count": 0,
    "requests_by_role": {"admin": 0, "user": 0},
    "tokens_total": 0,
    "token_samples": 0,
    "latency_total_ms": 0.0,
    "latency_samples": 0,
}

app = FastAPI(title="OmniGPT", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    summary: str = ""
    stream: bool = True
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    temperature: float | None = None


class LoginRequest(BaseModel):
    username: str = ""
    password: str = ""
    access_key: str = ""


class AdminSettings(BaseModel):
    system_prompt: str = Field(min_length=1, max_length=12000)
    temperature: float = Field(ge=0, le=2)
    default_max_tokens: int = Field(ge=1, le=8192)
    session_message_cap: int | None = Field(default=None, ge=1, le=10000)
    session_token_cap: int | None = Field(default=None, ge=1, le=10000000)


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _secure_compare(value: str, expected: str) -> bool:
    return hmac.compare_digest(value.encode("utf-8"), expected.encode("utf-8"))


def create_token(role: str, username: str) -> str:
    if not AUTH_SECRET:
        raise HTTPException(status_code=503, detail="Authentication is not configured")
    now = int(time.time())
    payload = _b64url_encode(
        json.dumps(
            {"role": role, "username": username, "iat": now, "exp": now + AUTH_TOKEN_TTL_SECONDS},
            separators=(",", ":"),
        ).encode("utf-8")
    )
    signature = _b64url_encode(hmac.new(AUTH_SECRET.encode(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{signature}"


def get_token_claims(request: Request) -> dict[str, Any]:
    if not AUTH_SECRET:
        raise HTTPException(status_code=503, detail="Authentication is not configured")
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")
    try:
        payload, signature = token.split(".", 1)
        expected = _b64url_encode(
            hmac.new(AUTH_SECRET.encode(), payload.encode(), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(signature, expected):
            raise ValueError("Invalid signature")
        claims = json.loads(_b64url_decode(payload))
        if claims.get("role") not in {"user", "admin"} or int(claims.get("exp", 0)) <= int(time.time()):
            raise ValueError("Expired or invalid token")
        return claims
    except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")


def require_admin(request: Request) -> dict[str, Any]:
    claims = get_token_claims(request)
    if claims["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return claims


def normalize_chat_completions_url(raw_url: str | None) -> str:
    candidate = (raw_url or "").strip().rstrip("/")
    if not candidate:
        return "https://api.groq.com/openai/v1/chat/completions"
    if candidate.endswith("/v1/chat/completions"):
        return candidate
    if candidate.endswith("/v1"):
        return f"{candidate}/chat/completions"
    return f"{candidate}/v1/chat/completions"


def build_upstream_headers() -> dict[str, str]:
    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    if LOCKED_API_KEY:
        headers["Authorization"] = f"Bearer {LOCKED_API_KEY}"
    return headers


def build_windowed_payload(req: ChatRequest, stream: bool) -> tuple[dict[str, Any], int]:
    clean_msgs = [
        {"role": m.role, "content": m.content}
        for m in req.messages
        if m.content and m.content.strip()
    ]

    latest_user_index = next(
        (index for index in range(len(clean_msgs) - 1, -1, -1) if clean_msgs[index]["role"] == "user"),
        None,
    )
    window_indices = (
        list(range(max(0, len(clean_msgs) - RECENT_WINDOW_MESSAGES), len(clean_msgs)))
        if RECENT_WINDOW_MESSAGES > 0
        else list(range(len(clean_msgs)))
    )
    if latest_user_index is not None and latest_user_index not in window_indices:
        if RECENT_WINDOW_MESSAGES > 0:
            window_indices = sorted([*window_indices[1:], latest_user_index])
        else:
            window_indices.append(latest_user_index)
    window = [
        {
            "role": clean_msgs[index]["role"],
            "content": clean_msgs[index]["content"]
            if index == latest_user_index
            else clean_msgs[index]["content"][:1200],
        }
        for index in window_indices
    ]
    messages: list[dict[str, str]] = [{"role": "system", "content": ACTIVE_SYSTEM_PROMPT}]

    if req.summary and req.summary.strip():
        messages.append(
            {
                "role": "system",
                "content": f"Earlier conversation summary:\n{req.summary.strip()}",
            }
        )

    messages.extend(window)

    payload: dict[str, Any] = {
        "model": LOCKED_MODEL,
        "messages": messages,
        "stream": stream,
        "temperature": ACTIVE_TEMPERATURE,
        "reasoning_effort": "low",
        "max_tokens": req.max_tokens if req.max_tokens is not None else ACTIVE_DEFAULT_MAX_TOKENS,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}

    older_count = max(0, len(clean_msgs) - len(window))
    estimated_tokens_saved = older_count * 45
    return payload, estimated_tokens_saved


def reserve_session_usage(
    req: ChatRequest,
    role: str,
    payload: dict[str, Any],
) -> tuple[str, int] | None:
    if role != "user":
        return None

    session_key = f"user:{req.session_id or 'unassigned'}"
    state = SESSION_USAGE.setdefault(
        session_key, {"messages": 0, "tokens": 0, "reserved_tokens": 0}
    )
    if state["messages"] >= ACTIVE_SESSION_MESSAGE_CAP:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "session_message_cap",
                "message": (
                    f"This chat reached its {ACTIVE_SESSION_MESSAGE_CAP}-message limit. "
                    "Start a new chat or ask an admin to raise the cap."
                ),
            },
        )

    remaining = ACTIVE_SESSION_TOKEN_CAP - state["tokens"] - state["reserved_tokens"]
    if remaining <= 0:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "session_token_cap",
                "message": (
                    f"This chat reached its {ACTIVE_SESSION_TOKEN_CAP}-token limit. "
                    "Start a new chat or ask an admin to raise the cap."
                ),
            },
        )

    prompt_bytes = sum(
        len(message["content"].encode("utf-8"))
        for message in payload["messages"]
    )
    estimated_prompt_tokens = max(1, (prompt_bytes + 2) // 3)
    summary_reservation = 0
    if len(req.messages) >= SUMMARY_TRIGGER_MESSAGES and RECENT_WINDOW_MESSAGES > 0:
        older_messages = req.messages[:-RECENT_WINDOW_MESSAGES]
        transcript = "\n".join(
            f"{message.role.upper()}: {message.content[:600]}"
            for message in older_messages[-12:]
        )
        summary_prompt = (
            "Update the running technical summary in <= 120 words. "
            "Keep key facts, decisions, names, and open tasks; drop filler.\n\n"
            f"Current summary:\n{req.summary or 'None'}\n\n"
            f"New messages to fold in:\n{transcript}"
        )
        summary_bytes = len(
            "You compress chat history into concise notes.".encode("utf-8")
        ) + len(summary_prompt.encode("utf-8"))
        summary_reservation = max(1, (summary_bytes + 2) // 3) + SUMMARY_MAX_TOKENS

    output_budget = min(
        int(payload["max_tokens"]),
        remaining - estimated_prompt_tokens - summary_reservation,
    )
    if output_budget < 1:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "session_token_cap",
                "message": (
                    "This prompt leaves no token budget in the current chat. "
                    "Start a new chat or ask an admin to raise the cap."
                ),
            },
        )

    payload["max_tokens"] = output_budget
    reservation = estimated_prompt_tokens + output_budget + summary_reservation
    state["messages"] += 1
    state["reserved_tokens"] += reservation
    return session_key, reservation


def settle_session_usage(
    reservation: tuple[str, int] | None,
    usages: list[dict[str, Any] | None],
) -> None:
    if reservation is None:
        return
    session_key, reserved_tokens = reservation
    state = SESSION_USAGE[session_key]
    state["reserved_tokens"] = max(0, state["reserved_tokens"] - reserved_tokens)
    actual_tokens = 0
    has_usage = False
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if (
            isinstance(prompt, int)
            and not isinstance(prompt, bool)
            and isinstance(completion, int)
            and not isinstance(completion, bool)
        ):
            actual_tokens += prompt + completion
            has_usage = True
    state["tokens"] += actual_tokens if has_usage else reserved_tokens


def combine_usage(usages: list[dict[str, Any] | None]) -> dict[str, int] | None:
    prompt_total = 0
    completion_total = 0
    has_usage = False
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if (
            isinstance(prompt, int)
            and not isinstance(prompt, bool)
            and isinstance(completion, int)
            and not isinstance(completion, bool)
        ):
            prompt_total += prompt
            completion_total += completion
            has_usage = True
    return (
        {"prompt_tokens": prompt_total, "completion_tokens": completion_total}
        if has_usage
        else None
    )


def record_request(role: str) -> None:
    USAGE_STATS["request_count"] += 1
    USAGE_STATS["requests_by_role"][role] += 1


def record_request_result(
    usages: list[dict[str, Any] | None],
    latency_ms: float,
) -> None:
    USAGE_STATS["latency_total_ms"] += latency_ms
    USAGE_STATS["latency_samples"] += 1
    request_tokens = 0
    has_usage = False
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if (
            isinstance(prompt, int)
            and not isinstance(prompt, bool)
            and isinstance(completion, int)
            and not isinstance(completion, bool)
        ):
            request_tokens += prompt + completion
            has_usage = True
    if has_usage:
        USAGE_STATS["tokens_total"] += request_tokens
        USAGE_STATS["token_samples"] += 1


async def update_rolling_summary(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    model: str,
    prior_summary: str,
    older_messages: list[dict[str, str]],
) -> tuple[str, dict[str, Any] | None]:
    if not older_messages:
        return prior_summary, None

    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content'][:600]}" for m in older_messages[-12:]
    )
    prompt_content = (
        "Update the running technical summary in <= 120 words. "
        "Keep key facts, decisions, names, and open tasks; drop filler.\n\n"
        f"Current summary:\n{prior_summary or 'None'}\n\n"
        f"New messages to fold in:\n{transcript}"
    )

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You compress chat history into concise notes."},
            {"role": "user", "content": prompt_content},
        ],
        "stream": False,
        "temperature": 0.2,
        "reasoning_effort": "low",
        "max_tokens": SUMMARY_MAX_TOKENS,
    }

    usage: dict[str, Any] | None = None
    try:
        resp = await client.post(url, headers=headers, json=body, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        choices = data.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content", "").strip()
            if content:
                return content, usage
    except Exception:
        pass
    return prior_summary, usage


@app.get("/")
async def serve_index():
    if INDEX_FILE.exists():
        return FileResponse(str(INDEX_FILE))
    return JSONResponse({"status": "ok", "service": "OmniGPT"})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": LOCKED_MODEL,
        "has_api_key": bool(LOCKED_API_KEY),
    }


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    if not AUTH_SECRET:
        raise HTTPException(status_code=503, detail="Authentication is not configured")

    role: str | None = None
    username = req.username.strip()
    if (
        ADMIN_USERNAME
        and ADMIN_PASSWORD
        and _secure_compare(username, ADMIN_USERNAME)
        and _secure_compare(req.password, ADMIN_PASSWORD)
    ):
        role = "admin"
    elif USER_ACCESS_KEY and _secure_compare(
        req.access_key or req.password, USER_ACCESS_KEY
    ):
        role = "user"
        username = username or "user"

    if role is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {
        "access_token": create_token(role, username),
        "token_type": "bearer",
        "role": role,
    }


@app.get("/api/auth/me")
async def auth_me(request: Request):
    claims = get_token_claims(request)
    return {"role": claims["role"], "username": claims.get("username", "")}


@app.get("/api/admin/settings")
async def get_admin_settings(request: Request):
    require_admin(request)
    return {
        "system_prompt": ACTIVE_SYSTEM_PROMPT,
        "temperature": ACTIVE_TEMPERATURE,
        "default_max_tokens": ACTIVE_DEFAULT_MAX_TOKENS,
        "session_message_cap": ACTIVE_SESSION_MESSAGE_CAP,
        "session_token_cap": ACTIVE_SESSION_TOKEN_CAP,
    }


@app.put("/api/admin/settings")
async def update_admin_settings(req: AdminSettings, request: Request):
    require_admin(request)
    prompt = req.system_prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="System prompt cannot be empty")
    global ACTIVE_SYSTEM_PROMPT, ACTIVE_TEMPERATURE, ACTIVE_DEFAULT_MAX_TOKENS
    global ACTIVE_SESSION_MESSAGE_CAP, ACTIVE_SESSION_TOKEN_CAP
    ACTIVE_SYSTEM_PROMPT = prompt
    ACTIVE_TEMPERATURE = req.temperature
    ACTIVE_DEFAULT_MAX_TOKENS = req.default_max_tokens
    if req.session_message_cap is not None:
        ACTIVE_SESSION_MESSAGE_CAP = req.session_message_cap
    if req.session_token_cap is not None:
        ACTIVE_SESSION_TOKEN_CAP = req.session_token_cap
    return {
        "system_prompt": ACTIVE_SYSTEM_PROMPT,
        "temperature": ACTIVE_TEMPERATURE,
        "default_max_tokens": ACTIVE_DEFAULT_MAX_TOKENS,
        "session_message_cap": ACTIVE_SESSION_MESSAGE_CAP,
        "session_token_cap": ACTIVE_SESSION_TOKEN_CAP,
    }


@app.get("/api/admin/health")
async def admin_health(request: Request):
    require_admin(request)
    return await health()


@app.get("/api/admin/usage")
async def admin_usage(request: Request):
    require_admin(request)
    return {
        "request_count": USAGE_STATS["request_count"],
        "requests_by_role": dict(USAGE_STATS["requests_by_role"]),
        "average_tokens": round(
            USAGE_STATS["tokens_total"] / USAGE_STATS["token_samples"], 1
        ) if USAGE_STATS["token_samples"] else 0,
        "average_latency_ms": round(
            USAGE_STATS["latency_total_ms"] / USAGE_STATS["latency_samples"], 1
        ) if USAGE_STATS["latency_samples"] else 0,
        "session_message_cap": ACTIVE_SESSION_MESSAGE_CAP,
        "session_token_cap": ACTIVE_SESSION_TOKEN_CAP,
    }


@app.post("/api/chat/sync")
async def chat_sync(req: ChatRequest, request: Request):
    claims = get_token_claims(request)
    role = claims["role"]
    record_request(role)
    started_at = time.perf_counter()
    url = normalize_chat_completions_url(LOCKED_ENDPOINT_URL)
    headers = build_upstream_headers()
    payload, tokens_saved = build_windowed_payload(req, stream=False)
    try:
        reservation = reserve_session_usage(req, role, payload)
    except HTTPException:
        record_request_result([], (time.perf_counter() - started_at) * 1000)
        raise

    usage: dict[str, Any] | None = None
    summary_usage: dict[str, Any] | None = None
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])
            data = resp.json()
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
            reply = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")

            updated_summary = req.summary
            if len(req.messages) >= SUMMARY_TRIGGER_MESSAGES and RECENT_WINDOW_MESSAGES > 0:
                older = [
                    {"role": m.role, "content": m.content}
                    for m in req.messages[:-RECENT_WINDOW_MESSAGES]
                ]
                updated_summary, summary_usage = await update_rolling_summary(
                    client, url, headers, payload["model"], req.summary, older
                )

        return {
            "reply": reply,
            "summary": updated_summary,
            "tokens_saved_est": tokens_saved,
        }
    finally:
        usages = [usage, summary_usage]
        settle_session_usage(reservation, usages)
        record_request_result(usages, (time.perf_counter() - started_at) * 1000)


@app.post("/api/chat")
async def chat_stream(
    req: ChatRequest,
    request: Request,
):
    claims = get_token_claims(request)
    role = claims["role"]
    record_request(role)
    started_at = time.perf_counter()
    url = normalize_chat_completions_url(LOCKED_ENDPOINT_URL)
    headers = build_upstream_headers()
    payload, tokens_saved = build_windowed_payload(req, stream=True)
    try:
        reservation = reserve_session_usage(req, role, payload)
    except HTTPException:
        record_request_result([], (time.perf_counter() - started_at) * 1000)
        raise

    async def event_generator():
        yield f"data: {json.dumps({'meta': {'tokens_saved_est': tokens_saved}})}\n\n"
        usage: dict[str, Any] | None = None
        summary_usage: dict[str, Any] | None = None
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code >= 400:
                        err_text = (await resp.aread()).decode("utf-8", errors="replace")[:500]
                        yield f"data: {json.dumps({'error': f'Upstream {resp.status_code}: {err_text}'})}\n\n"
                        return

                    async for line in resp.aiter_lines():
                        if await request.is_disconnected():
                            break
                        if not line:
                            continue
                        if line.startswith("data: "):
                            raw_chunk = line[6:].strip()
                            if raw_chunk == "[DONE]":
                                break
                            try:
                                parsed = json.loads(raw_chunk)
                                if isinstance(parsed.get("usage"), dict):
                                    usage = parsed["usage"]
                                delta = (parsed.get("choices") or [{}])[0].get("delta") or {}
                                token = delta.get("content")
                                if token:
                                    yield f"data: {json.dumps({'token': token})}\n\n"
                            except json.JSONDecodeError:
                                continue

                updated_summary = req.summary
                if len(req.messages) >= SUMMARY_TRIGGER_MESSAGES and RECENT_WINDOW_MESSAGES > 0:
                    older = [
                        {"role": m.role, "content": m.content}
                        for m in req.messages[:-RECENT_WINDOW_MESSAGES]
                    ]
                    updated_summary, summary_usage = await update_rolling_summary(
                        client, url, headers, payload["model"], req.summary, older
                    )

            usage_meta = combine_usage([usage, summary_usage]) or {
                "prompt_tokens": None,
                "completion_tokens": None,
            }
            yield f"data: {json.dumps({'done': True, 'summary': updated_summary, 'meta': usage_meta})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            usages = [usage, summary_usage]
            settle_session_usage(reservation, usages)
            record_request_result(usages, (time.perf_counter() - started_at) * 1000)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
