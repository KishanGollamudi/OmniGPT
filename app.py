import os
import json
import httpx
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

app = FastAPI()

API_PIN = os.getenv("API_PIN", "").strip()
DEFAULT_ENDPOINT_URL = os.getenv("LLM_ENDPOINT_URL", "").strip().rstrip("/")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "qwen2.5:14b-instruct-q4_K_M").strip()
DEFAULT_API_KEY = os.getenv("LLM_API_KEY", "").strip()


def verify_pin(x_access_pin: str | None):
    if API_PIN and x_access_pin != API_PIN:
        raise HTTPException(status_code=401, detail="Invalid Access PIN")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "has_default_endpoint": bool(DEFAULT_ENDPOINT_URL)}


@app.post("/api/chat")
async def proxy_chat(request: Request, x_access_pin: str | None = Header(default=None)):
    verify_pin(x_access_pin)
    body = await request.json()

    endpoint_url = (body.pop("kaggle_url", "") or body.pop("endpoint_url", "") or DEFAULT_ENDPOINT_URL).strip().rstrip("/")
    if not endpoint_url:
        raise HTTPException(
            status_code=400,
            detail="No LLM endpoint configured. Set LLM_ENDPOINT_URL in environment/render.yaml or set an Endpoint URL in the UI.",
        )

    if not body.get("model"):
        body["model"] = DEFAULT_MODEL

    target_url = f"{endpoint_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }
    if DEFAULT_API_KEY:
        headers["Authorization"] = f"Bearer {DEFAULT_API_KEY}"

    if not body.get("stream", True):
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(target_url, json=body, headers=headers)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return resp.json()

    async def event_stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", target_url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    safe_err = json.dumps(err.decode("utf-8", errors="ignore"))
                    yield f"data: {{\"error\": {safe_err}}}\n\n"
                    return
                async for line in resp.aiter_lines():
                    if line:
                        yield f"{line}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
