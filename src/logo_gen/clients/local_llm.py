"""Local/custom LLM client via any OpenAI-compatible API.

Works with Ollama, LM Studio, vLLM, text-generation-webui, or any server
that implements the OpenAI chat completions API.

Configure via settings or .env:
  LOCAL_LLM_BASE_URL=http://localhost:11434/v1   # Ollama
  LOCAL_LLM_BASE_URL=http://localhost:1234/v1    # LM Studio
  LOCAL_LLM_BASE_URL=http://localhost:8000/v1    # vLLM
  LOCAL_LLM_API_KEY=sk-...                       # optional, for authenticated APIs
  LOCAL_LLM_MODEL=qwen3.5:9b                     # model name
"""

from __future__ import annotations

import json

import httpx

from logo_gen.config import settings


def _base_url() -> str:
    return settings.local_llm_base_url


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.local_llm_api_key:
        headers["Authorization"] = f"Bearer {settings.local_llm_api_key}"
    return headers


async def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    response_format: dict | None = None,
) -> str:
    """Send a chat completion request and return the text response."""
    model = model or settings.local_llm_model
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{_base_url()}/chat/completions",
            headers=_headers(),
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    return data["choices"][0]["message"]["content"]


async def chat_stream(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
):
    """Stream chat completion tokens."""
    model = model or settings.local_llm_model
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST",
            f"{_base_url()}/chat/completions",
            headers=_headers(),
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = line[6:]
                if chunk.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                    delta = data["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


async def is_available() -> bool:
    """Check if the configured LLM endpoint is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{_base_url()}/models", headers=_headers())
            return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


async def list_models() -> list[str]:
    """List available models from the endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{_base_url()}/models", headers=_headers())
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        return []
