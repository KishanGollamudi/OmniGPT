import json
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ==============================================================================
# CONFIGURATION (Set secrets & endpoints via Render Environment Variables)
# ==============================================================================
# API key is read from the environment — DO NOT paste raw secrets here.
# On Render -> Environment, set:
#   LLM_API_KEY      = <your Grok/xAI or Groq API key; optional for Kaggle/Ollama>
#   LLM_ENDPOINT_URL = <e.g. https://api.x.ai/v1 or https://atop-remarry-richly.ngrok-free.dev>
#   LLM_MODEL        = <e.g. grok-4.7 or qwen2.5:14b-instruct-q4_K_M>
#   API_PIN          = <optional PIN required via X-API-PIN header>
DEFAULT_ENDPOINT_URL = os.getenv(
    "LLM_ENDPOINT_URL", "https://atop-remarry-richly.ngrok-free.dev"
).strip()
DEFAULT_API_KEY = os.getenv("LLM_API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("LLM_MODEL", "qwen2.5:14b-instruct-q4_K_M").strip()
API_PIN = os.getenv("API_PIN", "").strip()

RECENT_WINDOW_MESSAGES = int(os.getenv("RECENT_WINDOW_MESSAGES", "6"))
SUMMARY_TRIGGER_MESSAGES = int(os.getenv("SUMMARY_TRIGGER_MESSAGES", "10"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "180.0"))

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = (STATIC_DIR / "index.html" if (STATIC_DIR / "index.html").exists() else BASE_DIR / "index.html")

SYSTEM_PROMPT = (
    "You are OmniGPT, a direct, low-token technical assistant. "
    "Give concise, accurate, actionable answers without fluff."
)

app = FastAPI(title="OmniGPT Proxy", version="1.0.0")

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
    base_url: str | None = None
    model: str | None = None
    stream: bool = True
    max_tokens: int | None = None
    temperature: float = 0.7


def verify_pin(x_api_pin: str | None) -> None:
    if API_PIN and (x_api_pin or "").strip() != API_PIN:
        raise HTTPException(status_code=401, detail="Invalid or missing API PIN")


def normalize_chat_completions_url(raw_url: str | None) -> str:
    candidate = (raw_url or DEFAULT_ENDPOINT_URL or "").strip().rstrip("/")
    if not candidate:
        raise HTTPException(status_code=400, detail="No LLM endpoint URL configured")
    if candidate.endswith("/v1/chat/completions"):
        return candidate
    if candidate.endswith("/v1"):
        return f"{candidate}/chat/completions"
    return f"{candidate}/v1/chat/completions"


def build_upstream_headers() -> dict[str, str]:
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }
    # API key is injected here from LLM_API_KEY when set on the server
    if DEFAULT_API_KEY:
        headers["Authorization"] = f"Bearer {DEFAULT_API_KEY}"
    return headers


def build_windowed_payload(req: ChatRequest, stream: bool) -> dict[str, Any]:
    clean_msgs = [
        {"role": m.role, "content": m.content}
        for m in req.messages
        if m.content and m.content.strip()
    ]

    window = clean_msgs[-RECENT_WINDOW_MESSAGES:] if RECENT_WINDOW_MESSAGES > 0 else clean_msgs
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if req.summary and req.summary.strip():
        messages.append(
            {
                "role": "system",
                "content": f"Earlier conversation summary:\n{req.summary.strip()}",
            }
        )

    messages.extend(window)

    payload: dict[str, Any] = {
        "model": (req.model or DEFAULT_MODEL).strip(),
        "messages": messages,
        "stream": stream,
        "temperature": req.temperature,
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
        "default_model": DEFAULT_MODEL,
        "has_api_key": bool(DEFAULT_API_KEY),
    }


@app.post("/api/chat/sync")
async def chat_sync(req: ChatRequest, x_api_pin: str | None = Header(default=None)):
    verify_pin(x_api_pin)
    url = normalize_chat_completions_url(req.base_url)
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
            older = [{"role": m.role, "content": m.content} for m in req.messages[:-RECENT_WINDOW_MESSAGES]]
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
    x_api_pin: str | None = Header(default=None),
):
    verify_pin(x_api_pin)
    url = normalize_chat_completions_url(req.base_url)
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
                                delta = ((parsed.get("choices") or [{}])[0].get("delta") or {})
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
