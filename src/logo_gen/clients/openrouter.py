"""OpenRouter client: model catalog, chat (streaming) and the Images API.

Docs: https://openrouter.ai/docs/llms.txt
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from logo_gen.config import get_secret

BASE_URL = "https://openrouter.ai/api/v1"
CATALOG_TTL = 600  # seconds
APP_HEADERS = {"HTTP-Referer": "https://github.com/Duoquote/logo-gen", "X-Title": "logo-gen"}


class OpenRouterError(Exception):
    """An error from OpenRouter, with a message fit to show in the UI."""

    def __init__(
        self,
        message: str,
        *,
        status: int = 0,
        kind: str = "error",
        retry_after: float | None = None,
        detail: str = "",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.kind = kind
        self.retry_after = retry_after
        self.detail = detail

    def to_dict(self) -> dict:
        return {
            "message": self.message,
            "status": self.status,
            "kind": self.kind,
            "retry_after": self.retry_after,
        }


def _api_key() -> str:
    return get_secret("OPENROUTER_KEY")


def _headers(key: str | None = None, json_body: bool = True) -> dict[str, str]:
    key = _api_key() if key is None else key
    h = dict(APP_HEADERS)
    if key:
        h["Authorization"] = f"Bearer {key}"
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def _require_key() -> None:
    if not _api_key():
        raise OpenRouterError(
            "No OpenRouter API key. Add one in Settings.", status=401, kind="no_key"
        )


def error_from_payload(status: int, payload: Any, headers: httpx.Headers | None = None) -> OpenRouterError:
    """Turn an OpenRouter error body into a readable OpenRouterError."""
    err = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(err, dict):
        err = {"message": str(payload)[:300] if payload else ""}
    raw_msg = str(err.get("message") or "")
    meta = err.get("metadata") or {}
    code = int(err.get("code") or status or 0)
    status = status or code
    hint = meta.get("remedy_hint") or ""
    limit_source = meta.get("limit_source") or ""
    retry_after = None
    if headers is not None:
        try:
            retry_after = float(headers.get("retry-after", ""))
        except ValueError:
            retry_after = None

    if status == 401:
        return OpenRouterError(
            "OpenRouter rejected the API key. Check it in Settings.",
            status=401, kind="bad_key", detail=raw_msg,
        )
    if status == 402:
        msg = "Not enough OpenRouter credits for this request."
        if limit_source == "openrouter_in_flight_budget":
            msg = "Too many paid requests are in flight for your balance. Wait a moment and retry."
        return OpenRouterError(f"{msg} {hint}".strip(), status=402, kind="credits", retry_after=retry_after, detail=raw_msg)
    if status == 403 and limit_source == "openrouter_key_limit":
        spent, cap = meta.get("spend_usd"), meta.get("limit_usd")
        used = f" (${spent:.2f} of ${cap:.2f} used)" if isinstance(spent, (int, float)) and isinstance(cap, (int, float)) else ""
        return OpenRouterError(
            f"This API key reached its spending limit{used}. Raise the limit at "
            "openrouter.ai/settings/keys or use another key.",
            status=403, kind="key_limit", detail=raw_msg,
        )
    if status == 403:
        return OpenRouterError(
            f"OpenRouter refused the request: {raw_msg or 'forbidden'}", status=403, kind="forbidden", detail=raw_msg
        )
    if status == 429:
        upstream = limit_source == "upstream_provider_shared_pool" or "upstream" in raw_msg
        msg = (
            "The provider behind this model is rate limited right now (common for free models). "
            "Try again shortly or pick another model."
            if upstream
            else "OpenRouter rate limit reached. Try again shortly."
        )
        if retry_after:
            msg += f" Retry after {int(retry_after)}s."
        return OpenRouterError(msg, status=429, kind="rate_limit", retry_after=retry_after, detail=raw_msg)
    if status == 408:
        return OpenRouterError("The model took too long to answer. Try again.", status=408, kind="timeout", detail=raw_msg)
    if status in (502, 503):
        return OpenRouterError(
            f"No provider could serve this model right now ({raw_msg or status}). Try again or pick another model.",
            status=status, kind="unavailable", detail=raw_msg,
        )
    if status == 400:
        return OpenRouterError(f"OpenRouter rejected the request: {raw_msg}", status=400, kind="bad_request", detail=raw_msg)
    return OpenRouterError(raw_msg or f"OpenRouter error {status}", status=status, kind="error", detail=raw_msg)


async def _raise_for(resp: httpx.Response) -> None:
    if resp.status_code < 400:
        return
    try:
        await resp.aread()
        payload = resp.json()
    except Exception:
        payload = {"error": {"message": resp.text[:300] if resp.text else resp.reason_phrase}}
    raise error_from_payload(resp.status_code, payload, resp.headers)


def _network_error(e: Exception) -> OpenRouterError:
    return OpenRouterError(f"Could not reach OpenRouter: {e.__class__.__name__}", kind="network")


# ── Key status ───────────────────────────────────────────────────────


async def key_status(key: str | None = None) -> dict:
    """Validate a key with GET /key (free). Never returns the key itself."""
    key = _api_key() if key is None else key
    if not key:
        return {"valid": False, "error": "No key set"}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{BASE_URL}/key", headers=_headers(key, json_body=False))
            await _raise_for(resp)
            d = resp.json().get("data", {})
    except OpenRouterError as e:
        return {"valid": False, "error": e.message}
    except httpx.HTTPError as e:
        return {"valid": False, "error": _network_error(e).message}
    return {
        "valid": True,
        "label": d.get("label"),
        "usage": d.get("usage"),
        "limit": d.get("limit"),
        "limit_remaining": d.get("limit_remaining"),
        "is_free_tier": d.get("is_free_tier"),
    }


# ── Catalog ──────────────────────────────────────────────────────────

_catalog_cache: dict[str, tuple[float, dict]] = {}
_catalog_lock = asyncio.Lock()


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _provider_of(model: dict) -> str:
    name = model.get("name") or ""
    if ":" in name:
        return name.split(":", 1)[0].strip()
    return (model.get("id") or "").split("/", 1)[0]


def _per_image_estimate(lines: list[dict]) -> tuple[float | None, str]:
    """Estimate the price of one 1024x1024 output from endpoint pricing lines."""
    outs = [ln for ln in lines if ln.get("billable") == "output_image"]
    if not outs:
        return None, ""
    ln = outs[0]
    cost = _num(ln.get("cost_usd"))
    unit = ln.get("unit")
    if cost is None:
        return None, ""
    if unit == "image":
        return cost, "image"
    if unit == "megapixel":
        return round(cost * 1.048576, 6), "megapixel"
    if unit == "token":
        # Roughly 1,300 output tokens for a 1024px image on Gemini and GPT Image (medium).
        return round(cost * 1300, 6), "token"
    return None, ""


async def _image_endpoints(client: httpx.AsyncClient, model_id: str, sem: asyncio.Semaphore) -> dict | None:
    async with sem:
        try:
            r = await client.get(f"{BASE_URL}/images/models/{model_id}/endpoints")
            if r.status_code != 200:
                return None
            eps = r.json().get("endpoints") or []
            return eps[0] if eps else None
        except httpx.HTTPError:
            return None


def _chat_entry(m: dict) -> dict:
    p = m.get("pricing") or {}
    arch = m.get("architecture") or {}
    return {
        "id": m["id"],
        "name": m.get("name") or m["id"],
        "provider": _provider_of(m),
        "description": (m.get("description") or "")[:300],
        "context_length": m.get("context_length"),
        "input_modalities": arch.get("input_modalities") or [],
        "output_modalities": arch.get("output_modalities") or [],
        "prompt_price": _num(p.get("prompt")),
        "completion_price": _num(p.get("completion")),
        "supported_parameters": m.get("supported_parameters") or [],
        "reasoning": m.get("reasoning") or None,
        "created": m.get("created"),
        "is_free": m["id"].endswith(":free"),
    }


def _image_entry(m: dict, im: dict | None, ep: dict | None) -> dict:
    p = m.get("pricing") or {}
    arch = m.get("architecture") or {}
    params = (ep or {}).get("supported_parameters") or (im or {}).get("supported_parameters") or {}
    price, unit = _per_image_estimate((ep or {}).get("pricing") or [])
    if price is None and m["id"].endswith(":free"):
        price, unit = 0.0, "image"
    return {
        "id": m["id"],
        "name": m.get("name") or m["id"],
        "provider": _provider_of(m),
        "description": (m.get("description") or "")[:300],
        "context_length": m.get("context_length"),
        "input_modalities": arch.get("input_modalities") or [],
        "output_modalities": arch.get("output_modalities") or [],
        "price_per_image": price,
        "price_unit": unit,
        "pricing_lines": (ep or {}).get("pricing") or [],
        "image_token_price": _num(p.get("image_output")),
        "params": params,
        "supports_seed": "seed" in params,
        "supports_transparent": "transparent" in ((params.get("background") or {}).get("values") or []),
        "supports_references": bool((params.get("input_references") or {}).get("max")),
        "max_n": int((params.get("n") or {}).get("max") or 1),
        "vector": "svg" in ((params.get("output_format") or {}).get("values") or []),
        "created": m.get("created"),
        "is_free": m["id"].endswith(":free"),
    }


async def _fetch_catalog(key: str) -> dict:
    headers = _headers(key, json_body=False)
    async with httpx.AsyncClient(timeout=30, headers=headers) as client:
        text_r, img_r, imcat_r = await asyncio.gather(
            client.get(f"{BASE_URL}/models"),
            client.get(f"{BASE_URL}/models", params={"output_modalities": "image"}),
            client.get(f"{BASE_URL}/images/models"),
        )
        for r in (text_r, img_r):
            await _raise_for(r)
        text_models = text_r.json().get("data", [])
        image_models = img_r.json().get("data", [])
        imcat = {m["id"]: m for m in (imcat_r.json().get("data", []) if imcat_r.status_code == 200 else [])}

        # Image models: anything whose output modalities include image and that
        # the Images API serves. Routers like openrouter/auto are skipped.
        image_models = [
            m for m in image_models
            if "image" in ((m.get("architecture") or {}).get("output_modalities") or [])
            and "/api/v1/images" in (m.get("endpoints") or ["/api/v1/images"])
        ]
        sem = asyncio.Semaphore(12)
        eps = await asyncio.gather(*[_image_endpoints(client, m["id"], sem) for m in image_models])

    chat = [
        _chat_entry(m) for m in text_models
        if "/api/v1/chat/completions" in (m.get("endpoints") or ["/api/v1/chat/completions"])
        and not m["id"].startswith("~")
        and not m["id"].endswith(":batch")
    ]
    images = [_image_entry(m, imcat.get(m["id"]), ep) for m, ep in zip(image_models, eps)]
    return {"fetched_at": time.time(), "chat": chat, "image": images}


async def catalog(force: bool = False) -> dict:
    """Return the cached model catalog, refreshing it after CATALOG_TTL."""
    key = _api_key()
    cache_key = hashlib.sha256(key.encode()).hexdigest()[:16]
    async with _catalog_lock:
        hit = _catalog_cache.get(cache_key)
        if hit and not force and time.time() - hit[0] < CATALOG_TTL:
            return hit[1]
        try:
            data = await _fetch_catalog(key)
        except httpx.HTTPError as e:
            if hit:
                return hit[1] | {"stale": True}
            raise _network_error(e) from e
        _catalog_cache.clear()
        _catalog_cache[cache_key] = (time.time(), data)
        return data


def cached_model(model_id: str) -> dict | None:
    for _, data in _catalog_cache.values():
        for kind in ("chat", "image"):
            for m in data[kind]:
                if m["id"] == model_id:
                    return m
    return None


async def model_info(model_id: str) -> dict | None:
    m = cached_model(model_id)
    if m is None:
        try:
            await catalog()
        except OpenRouterError:
            return None
        m = cached_model(model_id)
    return m


# ── Chat ─────────────────────────────────────────────────────────────


async def _chat_payload(
    messages: list[dict],
    model: str,
    temperature: float | None,
    reasoning_effort: str | None,
    json_schema: dict | None,
    stream: bool,
) -> dict:
    info = await model_info(model) or {}
    params = set(info.get("supported_parameters") or [])
    payload: dict = {"model": model, "messages": messages, "stream": stream}
    if temperature is not None and (not params or "temperature" in params):
        payload["temperature"] = temperature
    reasoning = info.get("reasoning") or {}
    efforts = reasoning.get("supported_efforts") or []
    if reasoning_effort and "reasoning" in params and efforts:
        effort = reasoning_effort
        if effort not in efforts:
            # Fall back to the lightest effort the model accepts.
            for candidate in ("none", "minimal", "low", "medium"):
                if candidate in efforts:
                    effort = candidate
                    break
        if effort == "none" and reasoning.get("mandatory"):
            effort = next((e for e in ("minimal", "low") if e in efforts), efforts[-1])
        payload["reasoning"] = {"effort": effort, "exclude": not stream}
    if json_schema and ("structured_outputs" in params or "response_format" in params):
        payload["response_format"] = {"type": "json_schema", "json_schema": json_schema}
    return payload


async def chat(
    messages: list[dict],
    model: str,
    temperature: float | None = None,
    reasoning_effort: str | None = "low",
    json_schema: dict | None = None,
) -> tuple[str, dict]:
    """One chat completion. Returns (text, usage)."""
    _require_key()
    payload = await _chat_payload(messages, model, temperature, reasoning_effort, json_schema, stream=False)
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(f"{BASE_URL}/chat/completions", headers=_headers(), json=payload)
            await _raise_for(resp)
            data = resp.json()
    except httpx.HTTPError as e:
        raise _network_error(e) from e
    if data.get("error"):
        raise error_from_payload(int(data["error"].get("code") or 500), data)
    choice = (data.get("choices") or [{}])[0]
    if choice.get("error"):
        raise error_from_payload(int(choice["error"].get("code") or 500), {"error": choice["error"]})
    text = (choice.get("message") or {}).get("content") or ""
    return text, data.get("usage") or {}


async def chat_stream(
    messages: list[dict],
    model: str,
    temperature: float | None = None,
    reasoning_effort: str | None = "low",
) -> AsyncIterator[dict]:
    """Stream a chat completion.

    Yields {"type": "token", "content": str}, {"type": "reasoning", "content": str}
    while the model is thinking, and a final {"type": "usage", ...}.
    """
    _require_key()
    payload = await _chat_payload(messages, model, temperature, reasoning_effort, None, stream=True)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=15)) as client:
            async with client.stream("POST", f"{BASE_URL}/chat/completions", headers=_headers(), json=payload) as resp:
                await _raise_for(resp)
                async for line in resp.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    if data.get("error"):
                        raise error_from_payload(int(data["error"].get("code") or 500), data)
                    choices = data.get("choices") or []
                    if choices:
                        ch = choices[0]
                        if ch.get("error"):
                            raise error_from_payload(int(ch["error"].get("code") or 500), {"error": ch["error"]})
                        delta = ch.get("delta") or {}
                        if delta.get("content"):
                            yield {"type": "token", "content": delta["content"]}
                        elif delta.get("reasoning"):
                            yield {"type": "reasoning", "content": delta["reasoning"]}
                    if data.get("usage"):
                        yield {"type": "usage", **data["usage"]}
    except httpx.HTTPError as e:
        raise _network_error(e) from e


# ── Images ───────────────────────────────────────────────────────────


@dataclass
class ImageResult:
    data: bytes
    media_type: str


def file_to_data_uri(data: bytes, media_type: str) -> str:
    return f"data:{media_type};base64,{base64.b64encode(data).decode()}"


async def generate_images(
    prompt: str,
    model: str,
    n: int = 1,
    seed: int | None = None,
    aspect_ratio: str | None = "1:1",
    transparent: bool = False,
    references: list[tuple[bytes, str]] | None = None,
) -> tuple[list[ImageResult], float]:
    """POST /images. Returns (images, cost_usd). Only sends parameters the model accepts."""
    _require_key()
    info = await model_info(model) or {}
    params = info.get("params") or {}
    payload: dict = {"model": model, "prompt": prompt}

    n_spec = params.get("n") or {}
    max_n = int(n_spec.get("max") or 1)
    if n > 1 and max_n > 1:
        payload["n"] = min(n, max_n)
    ar_values = (params.get("aspect_ratio") or {}).get("values") or []
    if aspect_ratio and (not params or aspect_ratio in ar_values):
        payload["aspect_ratio"] = aspect_ratio
    if seed is not None and "seed" in params:
        payload["seed"] = seed
    if transparent and info.get("supports_transparent"):
        payload["background"] = "transparent"
    ref_spec = params.get("input_references") or {}
    if references and ref_spec.get("max"):
        payload["input_references"] = [
            {"type": "image_url", "image_url": {"url": file_to_data_uri(b, mt)}}
            for b, mt in references[: int(ref_spec["max"])]
        ]

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=15)) as client:
            resp = await client.post(f"{BASE_URL}/images", headers=_headers(), json=payload)
            await _raise_for(resp)
            data = resp.json()
            if data.get("error"):
                raise error_from_payload(int(data["error"].get("code") or 500), data)
            results: list[ImageResult] = []
            for item in data.get("data") or []:
                mt = item.get("media_type") or "image/png"
                if item.get("b64_json"):
                    results.append(ImageResult(base64.b64decode(item["b64_json"]), mt))
                elif item.get("url"):
                    r = await client.get(item["url"])
                    r.raise_for_status()
                    results.append(ImageResult(r.content, r.headers.get("content-type", mt)))
    except httpx.HTTPError as e:
        raise _network_error(e) from e
    if not results:
        raise OpenRouterError("The model returned no image.", kind="empty")
    cost = float((data.get("usage") or {}).get("cost") or 0.0)
    return results, cost
