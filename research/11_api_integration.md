# API Integration Patterns for AI Image Generation

Research compiled for the logo-gen project. Covers LLM access via OpenRouter, image
generation via Together AI / fal.ai / Replicate, abstraction layer design, error
handling, and cost tracking.

---

## 1. OpenRouter API (LLM Chat Completions)

OpenRouter provides a single endpoint that proxies to 200+ LLM providers
(OpenAI, Anthropic, Google, Meta, Mistral, etc.) with an OpenAI-compatible API.

### Base Configuration

| Item | Value |
|------|-------|
| Base URL | `https://openrouter.ai/api/v1` |
| Auth header | `Authorization: Bearer <OPENROUTER_API_KEY>` |
| Endpoint | `POST /chat/completions` |
| SDK compat | Drop-in replacement for `openai` Python SDK |

### Python -- Using the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="<OPENROUTER_API_KEY>",
)

completion = client.chat.completions.create(
    extra_headers={
        "HTTP-Referer": "<YOUR_SITE_URL>",       # optional, for ranking
        "X-OpenRouter-Title": "<YOUR_SITE_NAME>", # optional
    },
    model="openai/gpt-4o",
    messages=[
        {"role": "user", "content": "What is the meaning of life?"}
    ],
)
print(completion.choices[0].message.content)
```

### Python -- Using requests Directly

```python
import requests, json

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer <OPENROUTER_API_KEY>",
        "HTTP-Referer": "<YOUR_SITE_URL>",
        "X-OpenRouter-Title": "<YOUR_SITE_NAME>",
    },
    data=json.dumps({
        "model": "openai/gpt-4o",
        "messages": [
            {"role": "user", "content": "What is the meaning of life?"}
        ],
    }),
)
data = response.json()
print(data["choices"][0]["message"]["content"])
```

### Streaming

```python
completion = client.chat.completions.create(
    model="anthropic/claude-sonnet-4",
    messages=[{"role": "user", "content": "Write a haiku"}],
    stream=True,
)
for chunk in completion:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Model Selection Strategy

OpenRouter exposes models with a `<provider>/<model>` naming convention. Useful
models for a logo generation pipeline:

| Model ID | Use Case |
|----------|----------|
| `anthropic/claude-sonnet-4` | Prompt engineering, logo concept brainstorming |
| `openai/gpt-4o` | Structured output, JSON schema generation |
| `google/gemini-2.5-flash` | Fast/cheap iteration on prompts |
| `meta-llama/llama-4-maverick` | Open-weight alternative |

### Rate Limits

- Free models: capped requests-per-minute and requests-per-day (varies by credit tier).
- Paid models: governed globally per account (creating extra keys does not increase limits).
- Check live limits: `GET /api/v1/key` returns remaining credits and rate info.
- Cloudflare DDoS protection blocks dramatically excessive request rates.

### Error Codes

| HTTP Status | Meaning |
|-------------|---------|
| 400 | Bad request / invalid params |
| 401 | Invalid or expired API key |
| 402 | Insufficient credits |
| 403 | Content moderation block |
| 408 | Request timeout |
| 429 | Rate limited |
| 502 | Upstream provider error |
| 503 | No provider available for routing requirements |

Error response shape:
```json
{
  "error": {
    "code": 429,
    "message": "Rate limit exceeded",
    "metadata": {}
  }
}
```

### Debugging

Pass `debug: {"echo_upstream_body": true}` in streaming requests to inspect what
OpenRouter actually sent upstream. Never use in production.

---

## 2. Together AI API (Flux Image Generation)

Together AI hosts the Black Forest Labs Flux family on serverless GPUs with an
OpenAI-images-compatible endpoint.

### Installation & Auth

```bash
pip install together
```

```python
import os
from together import Together

os.environ["TOGETHER_API_KEY"] = "<YOUR_KEY>"
client = Together()
```

### Basic Image Generation

```python
response = client.images.generate(
    prompt="A minimalist geometric logo for a tech startup, white background",
    model="black-forest-labs/FLUX.1-schnell",
    steps=4,
)
print(f"Image URL: {response.data[0].url}")
```

### Available Models

| Model ID | Speed | Quality | Notes |
|----------|-------|---------|-------|
| `black-forest-labs/FLUX.1-schnell` | Very fast | Good | Free tier available |
| `black-forest-labs/FLUX.1-dev` | Medium | Better | Good balance |
| `black-forest-labs/FLUX.1-pro` | Slower | High | Professional quality |
| `black-forest-labs/FLUX.1-kontext-pro` | Medium | High | Image editing with `image_url` |
| `black-forest-labs/FLUX.2-pro` | Slower | Very high | Best quality, `reference_images` |
| `black-forest-labs/FLUX.2-max` | Slowest | Highest | Maximum fidelity |
| `black-forest-labs/FLUX.2-flex` | Medium | High | Best typography/text rendering |

### Multiple Variations

```python
response = client.images.generate(
    prompt="A modern logo with abstract shapes",
    model="black-forest-labs/FLUX.1-schnell",
    n=4,       # generate 1-4 variations
    steps=4,
    seed=42,   # reproducible output
)
for i, image in enumerate(response.data):
    print(f"Variation {i+1}: {image.url}")
```

### Custom Dimensions

```python
# Square (1:1) -- social media, app icons
response = client.images.generate(
    prompt="logo design", model="black-forest-labs/FLUX.1-schnell",
    width=1024, height=1024, steps=4,
)

# Landscape (16:9) -- banners
response = client.images.generate(
    prompt="logo design", model="black-forest-labs/FLUX.1-schnell",
    width=1344, height=768, steps=4,
)
```

### Base64 Response (No External URL)

```python
response = client.images.generate(
    model="black-forest-labs/FLUX.1-schnell",
    prompt="a minimalist logo",
    response_format="base64",
)
b64_data = response.data[0].b64_json
```

### Image Editing (Kontext)

```python
response = client.images.generate(
    model="black-forest-labs/FLUX.1-kontext-pro",
    width=1024, height=768,
    prompt="Change the color scheme to blue and white",
    image_url="https://example.com/draft_logo.png",
)
```

### Image Editing (FLUX.2 Reference Images)

```python
response = client.images.generate(
    model="black-forest-labs/FLUX.2-pro",
    width=1024, height=768,
    prompt="Create a similar style logo but for a coffee shop",
    reference_images=["https://example.com/reference_logo.png"],
)
```

### Key Parameters

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `prompt` | str | required | Text description |
| `model` | str | required | Model identifier |
| `width` | int | 1024 | Must be multiple of 32 |
| `height` | int | 1024 | Must be multiple of 32 |
| `steps` | int | model-dependent | 1-50; more steps = higher quality |
| `n` | int | 1 | 1-4 images per request |
| `seed` | int | random | For reproducibility |
| `negative_prompt` | str | None | What to exclude |
| `response_format` | str | "url" | "url" or "base64" |
| `disable_safety_checker` | bool | False | Bypass content filter |

---

## 3. fal.ai API (Multi-Model Image Generation)

fal.ai is a marketplace with 1000+ generative models behind a unified API.
Particularly strong for running multiple image models through a single interface.

### Installation & Auth

```bash
pip install fal-client
```

```python
import os
os.environ["FAL_KEY"] = "<YOUR_FAL_KEY>"
```

### Calling Patterns

fal.ai offers three calling patterns:

#### Subscribe (Recommended -- Auto-Polling)

Uses the queue internally but polls automatically until the result is ready.

```python
import fal_client

result = fal_client.subscribe(
    "fal-ai/flux/schnell",
    arguments={
        "prompt": "a minimalist logo, geometric shapes, white background",
        "image_size": "square_hd",  # or "landscape_16_9", "portrait_4_3", etc.
    },
)
print(result["images"][0]["url"])
```

#### Run (Direct Synchronous)

Blocks until the model returns. Good for fast models, but may time out on slow ones.

```python
response = fal_client.run(
    "fal-ai/fast-sdxl",
    arguments={"prompt": "a cute cat, realistic, orange"},
)
print(response["images"][0]["url"])
```

#### Submit (Async Queue)

Returns immediately with a request handle. Best for production workloads.

```python
handler = fal_client.submit(
    "fal-ai/flux/schnell",
    arguments={"prompt": "a sunset over mountains"},
)
print(handler.request_id)

# Later: check status or get result
result = handler.get()  # blocks until done
print(result["images"][0]["url"])
```

### Async Variants

Every method has an `_async` counterpart for use with `asyncio`:

```python
import asyncio
import fal_client

async def generate():
    result = await fal_client.run_async(
        "fal-ai/flux/schnell",
        arguments={"prompt": "abstract logo design"},
    )
    return result["images"][0]["url"]

url = asyncio.run(generate())
```

### Async Queue with Progress Events

```python
import asyncio
import fal_client

async def generate_with_progress():
    response = await fal_client.submit_async(
        "fal-ai/fast-sdxl",
        arguments={"prompt": "a futuristic logo"},
    )
    logs_index = 0
    async for event in response.iter_events(with_logs=True):
        if isinstance(event, fal_client.Queued):
            print(f"Queued. Position: {event.position}")
        elif isinstance(event, (fal_client.InProgress, fal_client.Completed)):
            new_logs = event.logs[logs_index:]
            for log in new_logs:
                print(log["message"])
            logs_index = len(event.logs)
    result = await response.get()
    return result["images"][0]["url"]

asyncio.run(generate_with_progress())
```

### Subscribe with Log Callback

```python
def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(log["message"])

result = fal_client.subscribe(
    "fal-ai/flux/schnell",
    arguments={"prompt": "a sunset over mountains"},
    with_logs=True,
    on_queue_update=on_queue_update,
)
```

### Timeout Control

```python
result = fal_client.subscribe(
    "fal-ai/flux/schnell",
    arguments={"prompt": "logo design"},
    client_timeout=60,  # seconds
)
```

### Available Models (Image Generation)

| Model Endpoint | Family | Notes |
|----------------|--------|-------|
| `fal-ai/flux/schnell` | Flux 1 | Fast, cheapest |
| `fal-ai/flux/dev` | Flux 1 | Higher quality |
| `fal-ai/flux-pro` | Flux 1 | Professional |
| `fal-ai/flux-2-pro` | Flux 2 | Best quality |
| `fal-ai/flux-2-flash` | Flux 2 | Fast + cheap |
| `fal-ai/fast-sdxl` | SDXL | Stable Diffusion XL |
| `fal-ai/recraft-v3` | Recraft | Strong text rendering |
| `fal-ai/imagen-4/preview` | Imagen 4 | Google's model |

### Key Advantage for Logo Gen

Switching models requires only changing the endpoint string -- no restructuring
of request/response handling. This makes fal.ai a natural fit as a multi-model
backend.

---

## 4. Replicate API

Replicate runs open-source models in the cloud with a simple `replicate.run()` API.

### Installation & Auth

```bash
pip install replicate
export REPLICATE_API_TOKEN=r8_...
```

### Basic Image Generation

```python
import replicate

output = replicate.run(
    "black-forest-labs/flux-schnell",
    input={"prompt": "a minimalist logo on white background"},
)

# output is a list of FileOutput objects
with open("logo.png", "wb") as f:
    f.write(output[0].read())
```

### Multiple Outputs

```python
output = replicate.run(
    "black-forest-labs/flux-schnell",
    input={
        "prompt": "abstract geometric logo",
        "num_outputs": 4,
    },
)
for idx, file_output in enumerate(output):
    with open(f"logo_{idx}.png", "wb") as f:
        f.write(file_output.read())
```

### Using a Reference Image

```python
output = replicate.run(
    "black-forest-labs/flux-1.1-pro",
    input={
        "prompt": "redesign this logo in a modern style",
        "image": "https://example.com/old_logo.png",
    },
)
```

### Streaming Text Output (LLM Models)

```python
iterator = replicate.run(
    "meta/meta-llama-3-70b-instruct",
    input={"prompt": "Describe a logo concept for a bakery"},
)
for text in iterator:
    print(text, end="")
```

### Async / Prediction API

```python
import replicate

# Create prediction (returns immediately)
prediction = replicate.predictions.create(
    model="black-forest-labs/flux-schnell",
    input={"prompt": "a logo"},
)
print(prediction.id)  # use to poll later

# Poll for result
prediction = replicate.predictions.get(prediction.id)
if prediction.status == "succeeded":
    print(prediction.output)
```

### Webhook-Based Async

```python
prediction = replicate.predictions.create(
    model="black-forest-labs/flux-schnell",
    input={"prompt": "logo design"},
    webhook="https://your-server.com/replicate-webhook",
    webhook_events_filter=["completed"],
)
```

### Pricing Model

Replicate charges per prediction (not per compute time for official models):
- FLUX.1 schnell: ~$0.003/image
- FLUX.1 pro: ~$0.05/image
- Community models: billed by GPU-second (~$0.000225/sec for CPU, varies by GPU)

---

## 5. Abstraction Layer Design

### Why Abstract?

- Image generation APIs evolve rapidly; new models appear monthly.
- Different providers have different strengths (speed, quality, cost, features).
- A unified interface lets the application swap providers without code changes.
- Enables A/B testing between providers and automatic failover.

### Architecture Pattern

```
Application Code
       |
       v
ImageProvider (abstract interface)
       |
  +---------+---------+---------+
  |         |         |         |
Together   fal.ai  Replicate  (future)
Adapter    Adapter  Adapter   Adapter
```

### Python Implementation

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class ImageSize(Enum):
    SQUARE_512 = (512, 512)
    SQUARE_1024 = (1024, 1024)
    LANDSCAPE_16_9 = (1344, 768)
    PORTRAIT_9_16 = (768, 1344)


@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    num_images: int = 1
    seed: Optional[int] = None
    steps: Optional[int] = None
    model_hint: Optional[str] = None  # e.g. "fast", "quality", "budget"

    @classmethod
    def square(cls, prompt: str, **kwargs) -> "GenerationRequest":
        return cls(prompt=prompt, width=1024, height=1024, **kwargs)


@dataclass
class GeneratedImage:
    url: Optional[str] = None
    b64_data: Optional[str] = None
    width: int = 0
    height: int = 0
    provider: str = ""
    model: str = ""
    generation_time_ms: int = 0
    cost_usd: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    images: list[GeneratedImage]
    total_cost_usd: float = 0.0
    total_time_ms: int = 0
    provider: str = ""
    model: str = ""


class ImageProvider(ABC):
    """Abstract base class for image generation providers."""

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate images from a text prompt."""
        ...

    @abstractmethod
    def estimate_cost(self, request: GenerationRequest) -> float:
        """Estimate cost in USD before generating."""
        ...

    @abstractmethod
    def available_models(self) -> list[str]:
        """List available model identifiers."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging/tracking."""
        ...


class TogetherProvider(ImageProvider):
    """Together AI Flux image generation."""

    MODEL_MAP = {
        "fast": "black-forest-labs/FLUX.1-schnell",
        "quality": "black-forest-labs/FLUX.2-pro",
        "budget": "black-forest-labs/FLUX.1-schnell",
        "edit": "black-forest-labs/FLUX.1-kontext-pro",
        "text": "black-forest-labs/FLUX.2-flex",
    }

    COST_PER_IMAGE = {
        "black-forest-labs/FLUX.1-schnell": 0.0,   # free tier
        "black-forest-labs/FLUX.1-dev": 0.025,
        "black-forest-labs/FLUX.1-pro": 0.05,
        "black-forest-labs/FLUX.2-pro": 0.06,
        "black-forest-labs/FLUX.2-max": 0.10,
    }

    def __init__(self, api_key: str):
        from together import Together
        self._client = Together(api_key=api_key)

    @property
    def name(self) -> str:
        return "together"

    def _resolve_model(self, hint: Optional[str]) -> str:
        if hint and hint in self.MODEL_MAP:
            return self.MODEL_MAP[hint]
        if hint and "/" in hint:
            return hint  # assume it is a full model ID
        return self.MODEL_MAP["fast"]

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        model = self._resolve_model(request.model_hint)
        start = time.monotonic()

        response = self._client.images.generate(
            prompt=request.prompt,
            model=model,
            width=request.width,
            height=request.height,
            n=request.num_images,
            steps=request.steps or 4,
            seed=request.seed,
            response_format="url",
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        cost_per = self.COST_PER_IMAGE.get(model, 0.03)

        images = [
            GeneratedImage(
                url=img.url,
                width=request.width,
                height=request.height,
                provider=self.name,
                model=model,
                generation_time_ms=elapsed_ms,
                cost_usd=cost_per,
            )
            for img in response.data
        ]

        return GenerationResult(
            images=images,
            total_cost_usd=cost_per * len(images),
            total_time_ms=elapsed_ms,
            provider=self.name,
            model=model,
        )

    def estimate_cost(self, request: GenerationRequest) -> float:
        model = self._resolve_model(request.model_hint)
        return self.COST_PER_IMAGE.get(model, 0.03) * request.num_images

    def available_models(self) -> list[str]:
        return list(self.COST_PER_IMAGE.keys())


class FalProvider(ImageProvider):
    """fal.ai multi-model image generation."""

    MODEL_MAP = {
        "fast": "fal-ai/flux/schnell",
        "quality": "fal-ai/flux-2-pro",
        "budget": "fal-ai/flux/schnell",
        "sdxl": "fal-ai/fast-sdxl",
        "recraft": "fal-ai/recraft-v3",
    }

    COST_PER_MP = {
        "fal-ai/flux/schnell": 0.003,
        "fal-ai/flux/dev": 0.025,
        "fal-ai/flux-2-pro": 0.03,
        "fal-ai/fast-sdxl": 0.005,
        "fal-ai/recraft-v3": 0.04,
    }

    def __init__(self, api_key: str):
        import os
        os.environ["FAL_KEY"] = api_key

    @property
    def name(self) -> str:
        return "fal"

    def _resolve_model(self, hint: Optional[str]) -> str:
        if hint and hint in self.MODEL_MAP:
            return self.MODEL_MAP[hint]
        if hint and "/" in hint:
            return hint
        return self.MODEL_MAP["fast"]

    def _megapixels(self, w: int, h: int) -> float:
        import math
        return math.ceil((w * h) / 1_000_000)

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        import fal_client

        model = self._resolve_model(request.model_hint)
        start = time.monotonic()

        result = await fal_client.subscribe_async(
            model,
            arguments={
                "prompt": request.prompt,
                "image_size": {
                    "width": request.width,
                    "height": request.height,
                },
                "num_images": request.num_images,
                "seed": request.seed,
            },
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        mp = self._megapixels(request.width, request.height)
        cost_per = self.COST_PER_MP.get(model, 0.01) * mp

        images = [
            GeneratedImage(
                url=img["url"],
                width=img.get("width", request.width),
                height=img.get("height", request.height),
                provider=self.name,
                model=model,
                generation_time_ms=elapsed_ms,
                cost_usd=cost_per,
            )
            for img in result["images"]
        ]

        return GenerationResult(
            images=images,
            total_cost_usd=cost_per * len(images),
            total_time_ms=elapsed_ms,
            provider=self.name,
            model=model,
        )

    def estimate_cost(self, request: GenerationRequest) -> float:
        model = self._resolve_model(request.model_hint)
        mp = self._megapixels(request.width, request.height)
        return self.COST_PER_MP.get(model, 0.01) * mp * request.num_images

    def available_models(self) -> list[str]:
        return list(self.COST_PER_MP.keys())


class ReplicateProvider(ImageProvider):
    """Replicate image generation."""

    MODEL_MAP = {
        "fast": "black-forest-labs/flux-schnell",
        "quality": "black-forest-labs/flux-1.1-pro",
        "budget": "black-forest-labs/flux-schnell",
    }

    COST_PER_IMAGE = {
        "black-forest-labs/flux-schnell": 0.003,
        "black-forest-labs/flux-1.1-pro": 0.05,
    }

    def __init__(self, api_key: str):
        import os
        os.environ["REPLICATE_API_TOKEN"] = api_key

    @property
    def name(self) -> str:
        return "replicate"

    def _resolve_model(self, hint: Optional[str]) -> str:
        if hint and hint in self.MODEL_MAP:
            return self.MODEL_MAP[hint]
        if hint and "/" in hint:
            return hint
        return self.MODEL_MAP["fast"]

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        import replicate

        model = self._resolve_model(request.model_hint)
        start = time.monotonic()

        output = replicate.run(
            model,
            input={
                "prompt": request.prompt,
                "num_outputs": request.num_images,
                "width": request.width,
                "height": request.height,
                "seed": request.seed,
            },
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        cost_per = self.COST_PER_IMAGE.get(model, 0.01)

        images = []
        for file_output in output:
            images.append(GeneratedImage(
                url=str(file_output),
                width=request.width,
                height=request.height,
                provider=self.name,
                model=model,
                generation_time_ms=elapsed_ms,
                cost_usd=cost_per,
            ))

        return GenerationResult(
            images=images,
            total_cost_usd=cost_per * len(images),
            total_time_ms=elapsed_ms,
            provider=self.name,
            model=model,
        )

    def estimate_cost(self, request: GenerationRequest) -> float:
        model = self._resolve_model(request.model_hint)
        return self.COST_PER_IMAGE.get(model, 0.01) * request.num_images

    def available_models(self) -> list[str]:
        return list(self.COST_PER_IMAGE.keys())
```

### Router / Orchestrator

```python
import random
from typing import Optional


class ImageRouter:
    """Routes generation requests to the best available provider."""

    def __init__(self):
        self._providers: dict[str, ImageProvider] = {}
        self._fallback_order: list[str] = []

    def register(self, provider: ImageProvider, priority: int = 0):
        self._providers[provider.name] = provider
        self._fallback_order.append(provider.name)
        # Sort by priority (lower = preferred)
        self._fallback_order.sort(key=lambda n: priority)

    async def generate(
        self,
        request: GenerationRequest,
        preferred_provider: Optional[str] = None,
        budget_limit_usd: Optional[float] = None,
    ) -> GenerationResult:
        """Generate with automatic fallback across providers."""

        order = list(self._fallback_order)
        if preferred_provider and preferred_provider in self._providers:
            order.remove(preferred_provider)
            order.insert(0, preferred_provider)

        last_error = None
        for provider_name in order:
            provider = self._providers[provider_name]

            # Budget check
            est_cost = provider.estimate_cost(request)
            if budget_limit_usd and est_cost > budget_limit_usd:
                continue

            try:
                return await provider.generate(request)
            except Exception as e:
                last_error = e
                continue  # try next provider

        raise RuntimeError(
            f"All providers failed. Last error: {last_error}"
        )

    def cheapest_provider(self, request: GenerationRequest) -> str:
        """Return the name of the cheapest provider for a request."""
        costs = {
            name: p.estimate_cost(request)
            for name, p in self._providers.items()
        }
        return min(costs, key=costs.get)
```

### Usage Example

```python
import asyncio

async def main():
    router = ImageRouter()
    router.register(TogetherProvider(api_key="..."), priority=0)
    router.register(FalProvider(api_key="..."), priority=1)
    router.register(ReplicateProvider(api_key="..."), priority=2)

    request = GenerationRequest.square(
        prompt="A minimalist mountain logo, flat design, SVG style",
        num_images=4,
        model_hint="fast",
    )

    # Cheapest option
    print(f"Cheapest: {router.cheapest_provider(request)}")

    # Generate with auto-fallback
    result = await router.generate(request, budget_limit_usd=0.50)
    for img in result.images:
        print(f"  {img.url} (${img.cost_usd:.4f})")
    print(f"Total: ${result.total_cost_usd:.4f} in {result.total_time_ms}ms")

asyncio.run(main())
```

---

## 6. Error Handling, Retries, and Rate Limiting

### Retry with Exponential Backoff (using tenacity)

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import httpx


class RateLimitError(Exception):
    pass


class ProviderUnavailableError(Exception):
    pass


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type((RateLimitError, ProviderUnavailableError)),
)
async def generate_with_retry(provider: ImageProvider, request: GenerationRequest):
    try:
        return await provider.generate(request)
    except Exception as e:
        error_str = str(e).lower()
        if "429" in error_str or "rate limit" in error_str:
            raise RateLimitError(str(e)) from e
        if "502" in error_str or "503" in error_str:
            raise ProviderUnavailableError(str(e)) from e
        raise  # non-retryable
```

### Manual Backoff (No Dependencies)

```python
import asyncio
import random


async def generate_with_backoff(
    provider: ImageProvider,
    request: GenerationRequest,
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> GenerationResult:
    for attempt in range(max_retries):
        try:
            return await provider.generate(request)
        except Exception as e:
            error_str = str(e).lower()
            is_retryable = any(
                code in error_str for code in ("429", "502", "503", "rate limit", "timeout")
            )
            if not is_retryable or attempt == max_retries - 1:
                raise

            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}")
            await asyncio.sleep(delay)

    raise RuntimeError("Should not reach here")
```

### Rate Limiter (Token Bucket)

```python
import asyncio
import time


class RateLimiter:
    """Token-bucket rate limiter for API calls."""

    def __init__(self, requests_per_minute: int):
        self._rpm = requests_per_minute
        self._tokens = requests_per_minute
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._rpm,
                self._tokens + elapsed * (self._rpm / 60.0),
            )
            self._last_refill = now

            if self._tokens < 1:
                wait_time = (1 - self._tokens) / (self._rpm / 60.0)
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1


# Usage:
rate_limiter = RateLimiter(requests_per_minute=60)

async def rate_limited_generate(provider, request):
    await rate_limiter.acquire()
    return await provider.generate(request)
```

### Circuit Breaker Pattern

```python
import time


class CircuitBreaker:
    """Prevent calling a failing provider repeatedly."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self._failure_count = 0
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._last_failure_time = 0.0
        self._state = "closed"  # closed, open, half-open

    @property
    def is_open(self) -> bool:
        if self._state == "open":
            if time.monotonic() - self._last_failure_time > self._reset_timeout:
                self._state = "half-open"
                return False
            return True
        return False

    def record_success(self):
        self._failure_count = 0
        self._state = "closed"

    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self._failure_threshold:
            self._state = "open"
```

---

## 7. Cost Tracking and Optimization

### Cost Tracker

```python
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict


@dataclass
class CostEntry:
    timestamp: float
    provider: str
    model: str
    num_images: int
    cost_usd: float
    generation_time_ms: int
    prompt_hash: str = ""


class CostTracker:
    """Track and analyze image generation costs."""

    def __init__(self, log_path: str = "costs.jsonl"):
        self._log_path = Path(log_path)
        self._entries: list[CostEntry] = []
        self._budget_usd: float = float("inf")

    def set_budget(self, daily_usd: float):
        self._budget_usd = daily_usd

    def record(self, result: GenerationResult, prompt: str = ""):
        import hashlib
        entry = CostEntry(
            timestamp=time.time(),
            provider=result.provider,
            model=result.model,
            num_images=len(result.images),
            cost_usd=result.total_cost_usd,
            generation_time_ms=result.total_time_ms,
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:12],
        )
        self._entries.append(entry)

        # Append to JSONL file
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry.__dict__) + "\n")

    def today_spend(self) -> float:
        """Total spend for the current day."""
        import datetime
        today_start = datetime.datetime.now().replace(
            hour=0, minute=0, second=0
        ).timestamp()
        return sum(
            e.cost_usd for e in self._entries
            if e.timestamp >= today_start
        )

    def remaining_budget(self) -> float:
        return max(0, self._budget_usd - self.today_spend())

    def is_within_budget(self, estimated_cost: float) -> bool:
        return self.today_spend() + estimated_cost <= self._budget_usd

    def summary_by_provider(self) -> dict:
        """Aggregate cost and count by provider."""
        summary = defaultdict(lambda: {"count": 0, "cost": 0.0, "images": 0})
        for e in self._entries:
            s = summary[e.provider]
            s["count"] += 1
            s["cost"] += e.cost_usd
            s["images"] += e.num_images
        return dict(summary)

    def summary_by_model(self) -> dict:
        summary = defaultdict(lambda: {"count": 0, "cost": 0.0, "avg_time_ms": 0.0})
        for e in self._entries:
            s = summary[e.model]
            s["count"] += 1
            s["cost"] += e.cost_usd
            s["avg_time_ms"] = (
                (s["avg_time_ms"] * (s["count"] - 1) + e.generation_time_ms)
                / s["count"]
            )
        return dict(summary)
```

### Optimization Strategies

| Strategy | Description | Savings |
|----------|-------------|---------|
| **Tiered generation** | Use fast/cheap model for drafts, expensive model for finals | 60-80% |
| **Prompt caching** | Hash prompts and reuse results for identical requests | Variable |
| **Batch dimensions** | Generate at standard sizes, resize client-side | 20-40% |
| **Provider arbitrage** | Route to cheapest provider that meets quality bar | 30-50% |
| **Step optimization** | Use minimum steps that produce acceptable quality | 20-40% |
| **Free tier first** | Use Together AI FLUX.1 schnell (free) for initial exploration | 100% on drafts |

### Tiered Generation Flow

```python
async def tiered_logo_generation(
    router: ImageRouter,
    prompt: str,
    tracker: CostTracker,
) -> list[GeneratedImage]:
    """
    Stage 1: Generate many cheap drafts.
    Stage 2: Upscale the best candidates with a quality model.
    """
    # Stage 1 -- fast drafts
    draft_request = GenerationRequest(
        prompt=prompt,
        width=512,
        height=512,
        num_images=4,
        model_hint="budget",
    )
    draft_result = await router.generate(draft_request)
    tracker.record(draft_result, prompt)

    # ... user or automated scoring picks the best draft ...
    best_draft_url = draft_result.images[0].url

    # Stage 2 -- high-quality refinement
    refine_request = GenerationRequest(
        prompt=prompt,
        width=1024,
        height=1024,
        num_images=1,
        model_hint="quality",
    )
    final_result = await router.generate(refine_request)
    tracker.record(final_result, prompt)

    return final_result.images
```

---

## 8. Pricing Comparison (as of early 2026)

### Per-Image Cost at 1024x1024 (~1 MP)

| Model | Together AI | fal.ai | Replicate | BFL Direct |
|-------|-----------|--------|-----------|------------|
| FLUX.1 Schnell | Free | $0.003 | $0.003 | -- |
| FLUX.1 Dev | ~$0.025 | $0.025 | ~$0.025 | -- |
| FLUX.1 Pro | ~$0.05 | ~$0.05 | ~$0.05 | $0.05 |
| FLUX.2 Pro | ~$0.06 | $0.03 | -- | $0.06 |
| FLUX.2 Flash/Turbo | -- | $0.008 | -- | -- |
| Recraft V3 | -- | $0.04 | -- | -- |

Notes:
- fal.ai bills by megapixel, rounded up. Smaller images are proportionally cheaper.
- Together AI offers free unlimited FLUX.1 schnell access.
- Replicate charges per prediction for official models.
- Prices change frequently; check provider docs for current rates.

---

## 9. Recommended Stack for logo-gen

| Concern | Recommendation |
|---------|---------------|
| LLM for prompt engineering | OpenRouter with `openai` SDK (model-agnostic) |
| Fast drafts | Together AI FLUX.1 schnell (free) |
| High-quality finals | fal.ai FLUX.2 pro or Together AI FLUX.2 pro |
| Text-in-logo rendering | fal.ai FLUX.2 flex or Recraft V3 |
| Multi-model abstraction | Custom `ImageProvider` ABC + `ImageRouter` (see section 5) |
| Retries | `tenacity` with exponential backoff + jitter |
| Rate limiting | Token-bucket per provider |
| Failover | Circuit breaker + provider fallback chain |
| Cost control | `CostTracker` with daily budget caps |

### Minimal Dependencies

```
# requirements.txt
openai>=1.0         # OpenRouter LLM access
together>=1.0       # Together AI images
fal-client>=0.5     # fal.ai images
replicate>=1.0      # Replicate images
tenacity>=8.0       # Retry logic
httpx>=0.27         # Async HTTP (optional, for custom calls)
```

---

## Sources

- [OpenRouter Quickstart](https://openrouter.ai/docs/quickstart)
- [OpenRouter Error Handling](https://openrouter.ai/docs/api/reference/errors-and-debugging)
- [OpenRouter Rate Limits](https://openrouter.ai/docs/api/reference/limits)
- [Together AI Image Generation](https://docs.together.ai/docs/images-overview)
- [Together AI Flux Tools](https://www.together.ai/blog/flux-tools-models-together-apis-canny-depth-image-generation)
- [fal.ai Model APIs](https://fal.ai/docs/model-apis)
- [fal.ai Python Client](https://fal.ai/docs/clients/python)
- [fal.ai Pricing](https://fal.ai/pricing)
- [Replicate Python Quickstart](https://replicate.com/docs/get-started/python)
- [Replicate Pricing](https://replicate.com/pricing)
- [AI Image API Cost Comparison](https://pricepertoken.com/image)
