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

RECENT_WINDOW_MESSAGES = int(os.getenv("RECENT_WINDOW_MESSAGES", "6"))
SUMMARY_TRIGGER_MESSAGES = int(os.getenv("SUMMARY_TRIGGER_MESSAGES", "10"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "180.0"))

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = (
    STATIC_DIR / "index.html"
    if (STATIC_DIR / "index.html").exists()
    else BASE_DIR / "index.html"
)

SYSTEM_PROMPT = (
    "You are OmniGPT, a direct, low-token DevOps and SRE technical assistant. "
    "Give concise, accurate, actionable answers with clear commands and zero fluff."
)
ACTIVE_SYSTEM_PROMPT = SYSTEM_PROMPT
ACTIVE_TEMPERATURE = 0.3

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
    summary: str = ""
    stream: bool = True
    max_tokens: int | None = None
    temperature: float | None = None


class LoginRequest(BaseModel):
    username: str = ""
    password: str = ""
    access_key: str = ""


class AdminSettings(BaseModel):
    system_prompt: str = Field(min_length=1, max_length=12000)
    temperature: float = Field(ge=0, le=2)


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

    window = clean_msgs[-RECENT_WINDOW_MESSAGES:] if RECENT_WINDOW_MESSAGES > 0 else clean_msgs
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
    }
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens

    older_count = max(0, len(clean_msgs) - len(window))
    estimated_tokens_saved = older_count * 45
    return payload, estimated_tokens_saved


async def update_rolling_summary(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    model: str,
    prior_summary: str,
    older_messages: list[dict[str, str]],
) -> str:
    if not older_messages:
        return prior_summary

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
    }

    try:
        resp = await client.post(url, headers=headers, json=body, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content", "").strip()
            if content:
                return content
    except Exception:
        pass
    return prior_summary


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
    return {"system_prompt": ACTIVE_SYSTEM_PROMPT, "temperature": ACTIVE_TEMPERATURE}


@app.put("/api/admin/settings")
async def update_admin_settings(req: AdminSettings, request: Request):
    require_admin(request)
    prompt = req.system_prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="System prompt cannot be empty")
    global ACTIVE_SYSTEM_PROMPT, ACTIVE_TEMPERATURE
    ACTIVE_SYSTEM_PROMPT = prompt
    ACTIVE_TEMPERATURE = req.temperature
    return {"system_prompt": ACTIVE_SYSTEM_PROMPT, "temperature": ACTIVE_TEMPERATURE}


@app.get("/api/admin/health")
async def admin_health(request: Request):
    require_admin(request)
    return await health()


@app.post("/api/chat/sync")
async def chat_sync(req: ChatRequest, request: Request):
    get_token_claims(request)
    url = normalize_chat_completions_url(LOCKED_ENDPOINT_URL)
    headers = build_upstream_headers()
    payload, tokens_saved = build_windowed_payload(req, stream=False)

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])
        data = resp.json()
        reply = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")

        updated_summary = req.summary
        if len(req.messages) >= SUMMARY_TRIGGER_MESSAGES and RECENT_WINDOW_MESSAGES > 0:
            older = [
                {"role": m.role, "content": m.content}
                for m in req.messages[:-RECENT_WINDOW_MESSAGES]
            ]
            updated_summary = await update_rolling_summary(
                client, url, headers, payload["model"], req.summary, older
            )

    return {
        "reply": reply,
        "summary": updated_summary,
        "tokens_saved_est": tokens_saved,
    }


@app.post("/api/chat")
async def chat_stream(
    req: ChatRequest,
    request: Request,
):
    get_token_claims(request)
    url = normalize_chat_completions_url(LOCKED_ENDPOINT_URL)
    headers = build_upstream_headers()
    payload, tokens_saved = build_windowed_payload(req, stream=True)

    async def event_generator():
        yield f"data: {json.dumps({'meta': {'tokens_saved_est': tokens_saved}})}\n\n"
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
                    updated_summary = await update_rolling_summary(
                        client, url, headers, payload["model"], req.summary, older
                    )

            yield f"data: {json.dumps({'done': True, 'summary': updated_summary})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
