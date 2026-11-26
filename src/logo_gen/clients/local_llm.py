"""Local LLM through any OpenAI-compatible server (Ollama, LM Studio, vLLM, ...)."""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from logo_gen.config import get_settings


class LocalLLMError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _base_url() -> str:
    return get_settings().local_llm_base_url.rstrip("/")


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = get_settings().local_llm_api_key
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _wrap(e: Exception) -> LocalLLMError:
    if isinstance(e, httpx.ConnectError):
        return LocalLLMError(f"Cannot reach the local LLM server at {_base_url()}. Is it running?")
    if isinstance(e, httpx.HTTPStatusError):
        body = e.response.text[:200] if e.response is not None else ""
        return LocalLLMError(f"Local LLM server returned {e.response.status_code}: {body}")
    return LocalLLMError(f"Local LLM error: {e}")


async def chat(messages: list[dict], model: str | None = None, temperature: float = 0.7) -> str:
    payload = {"model": model or get_settings().local_llm_model, "messages": messages, "temperature": temperature}
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{_base_url()}/chat/completions", headers=_headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        raise _wrap(e) from e
    return data["choices"][0]["message"]["content"] or ""


async def chat_stream(messages: list[dict], model: str | None = None, temperature: float = 0.7) -> AsyncIterator[str]:
    payload = {
        "model": model or get_settings().local_llm_model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", f"{_base_url()}/chat/completions", headers=_headers(), json=payload) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        delta = json.loads(chunk)["choices"][0]["delta"].get("content", "")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if delta:
                        yield delta
    except httpx.HTTPError as e:
        raise _wrap(e) from e


async def list_models() -> list[str]:
    """Models the local server reports. Empty if it is not reachable."""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{_base_url()}/models", headers=_headers())
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
    except (httpx.HTTPError, KeyError, ValueError):
        return []
