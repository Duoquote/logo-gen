"""OpenRouter client for LLM chat and image generation."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import httpx

from logo_gen.config import settings

BASE_URL = "https://openrouter.ai/api/v1"


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.openrouter_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "logo-gen",
    }


async def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    response_format: dict | None = None,
) -> str:
    """Send a chat completion request and return the text response."""
    model = model or settings.llm_model
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{BASE_URL}/chat/completions",
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
    model = model or settings.llm_model
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/chat/completions",
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


def _extract_base64_images(text: str) -> list[str]:
    """Extract base64 image data from markdown image tags or raw base64."""
    # Pattern: ![...](data:image/...;base64,DATA)
    pattern = r"data:image/[^;]+;base64,([A-Za-z0-9+/=\s]+)"
    matches = re.findall(pattern, text)
    if matches:
        return [m.replace("\n", "").replace(" ", "") for m in matches]
    return []


def _extract_urls(text: str) -> list[str]:
    """Extract image URLs from markdown or raw URLs."""
    # Markdown images: ![...](url)
    md_pattern = r"!\[[^\]]*\]\(([^)]+)\)"
    urls = re.findall(md_pattern, text)
    if urls:
        return [u for u in urls if not u.startswith("data:")]
    # Raw URLs ending in image extensions
    url_pattern = r"https?://[^\s\)\"']+\.(?:png|jpg|jpeg|webp)[^\s\)\"']*"
    return re.findall(url_pattern, text)


async def generate_image(
    prompt: str,
    model: str,
    seed: int | None = None,
    save_dir: Path | None = None,
) -> list[Path]:
    """Generate an image using an OpenRouter image-capable model.

    Returns list of saved image file paths.
    """
    save_dir = save_dir or Path(settings.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build the message with explicit logo instructions
    full_prompt = (
        f"{prompt}\n\n"
        "Generate this as a single logo design on a clean solid white background. "
        "No text, no letters, no words anywhere in the image. "
        "The design should be centered and suitable as a brand icon."
    )

    messages = [{"role": "user", "content": full_prompt}]

    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": 1.0,
    }

    # Some models support seed
    if seed is not None:
        payload["seed"] = seed

    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            f"{BASE_URL}/chat/completions",
            headers=_headers(),
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    message = data["choices"][0]["message"]
    content = message.get("content") or ""

    saved: list[Path] = []
    model_slug = model.replace("/", "_")
    seed_str = f"_s{seed}" if seed is not None else ""

    # Method 1: Check message.images array (OpenRouter native format)
    images_array = message.get("images", [])
    for i, img_obj in enumerate(images_array):
        b64_data = None
        if isinstance(img_obj, dict):
            url = img_obj.get("image_url", {}).get("url", "")
            if url.startswith("data:image/"):
                b64_data = url.split("base64,", 1)[1] if "base64," in url else None
        elif isinstance(img_obj, str) and img_obj.startswith("data:image/"):
            b64_data = img_obj.split("base64,", 1)[1] if "base64," in img_obj else None

        if b64_data:
            img_bytes = base64.b64decode(b64_data)
            filename = f"{model_slug}{seed_str}_{i}.png"
            path = save_dir / filename
            path.write_bytes(img_bytes)
            saved.append(path)

    # Method 2: Extract base64 images from text content
    if not saved and content:
        b64_images = _extract_base64_images(content)
        for i, b64_data in enumerate(b64_images):
            img_bytes = base64.b64decode(b64_data)
            filename = f"{model_slug}{seed_str}_b{i}.png"
            path = save_dir / filename
            path.write_bytes(img_bytes)
            saved.append(path)

    # Method 3: Extract and download URLs from text content
    if not saved and content:
        urls = _extract_urls(content)
        for i, url in enumerate(urls):
            try:
                async with httpx.AsyncClient(timeout=60) as dl_client:
                    img_resp = await dl_client.get(url)
                    img_resp.raise_for_status()
                filename = f"{model_slug}{seed_str}_url{i}.png"
                path = save_dir / filename
                path.write_bytes(img_resp.content)
                saved.append(path)
            except Exception:
                continue

    return saved
