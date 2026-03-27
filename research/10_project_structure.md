# Python Project Structure for Logo Generation Tool

## Research Date: March 2026

---

## 1. Project Setup with `uv`

`uv` (by Astral) is the standard Python project manager in 2026, replacing pip, pip-tools, pipenv, and poetry with a single, fast Rust-based tool.

### Initialization

```bash
# Create a new project with src layout (recommended for apps/libraries)
uv init --lib logo-gen
cd logo-gen

# Or initialize in an existing directory
cd logo-gen
uv init --lib

# For a simple application (flat layout, no build system)
uv init logo-gen
```

`uv init --lib` creates the **src layout** automatically:

```
logo-gen/
  .python-version          # pinned Python version
  pyproject.toml           # PEP 621 project metadata
  README.md
  src/
    logo_gen/
      __init__.py
      py.typed              # type annotation marker
```

### Virtual Environment

```bash
# uv creates .venv automatically on first `uv run` or `uv sync`
# Explicit creation if needed:
uv venv                    # creates .venv with pinned Python
uv venv --python 3.12      # specify Python version
```

### Adding Dependencies

```bash
uv add httpx               # add runtime dependency
uv add pydantic-settings   # add another
uv add --dev pytest ruff mypy  # add dev dependencies
uv add --group docs mkdocs     # add to a named group
```

### pyproject.toml Structure

```toml
[project]
name = "logo-gen"
version = "0.1.0"
description = "AI-powered logo generation tool"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.27",
    "pydantic-settings>=2.7",
    "structlog>=25.1",
    "typer>=0.15",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.9",
    "mypy>=1.14",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/logo_gen"]

[tool.ruff]
line-length = 100
src = ["src"]

[tool.mypy]
strict = true

[project.scripts]
logo-gen = "logo_gen.cli:app"
```

### Key Commands

| Command | Purpose |
|---------|---------|
| `uv init --lib` | Create project with src layout |
| `uv add <pkg>` | Add dependency (updates pyproject.toml + uv.lock) |
| `uv remove <pkg>` | Remove dependency |
| `uv sync` | Install all deps from lockfile |
| `uv run <cmd>` | Run command in project environment (auto-syncs) |
| `uv lock` | Regenerate lockfile without installing |
| `uv run pytest` | Run tests |
| `uv run ruff check` | Run linter |

**Sources:**
- [Working on projects - uv docs](https://docs.astral.sh/uv/guides/projects/)
- [Creating projects - uv docs](https://docs.astral.sh/uv/concepts/projects/init/)
- [Structure and files - uv docs](https://docs.astral.sh/uv/concepts/projects/layout/)
- [Managing Python Projects With uv - Real Python](https://realpython.com/python-uv/)

---

## 2. Project Layout: src/ vs Flat

### Flat Layout

```
logo-gen/
  logo_gen/
    __init__.py
    main.py
  tests/
  pyproject.toml
```

### Src Layout

```
logo-gen/
  src/
    logo_gen/
      __init__.py
      main.py
  tests/
  pyproject.toml
```

### Recommendation: Use src/ Layout

The **src layout is the clear winner** for a production tool like our logo generator:

| Aspect | src/ Layout | Flat Layout |
|--------|-------------|-------------|
| Import safety | Tests always import the *installed* package, not local source | Can accidentally import uninstalled source |
| Packaging correctness | Forces proper install before testing | May pass tests locally but fail when distributed |
| Namespace clarity | Clean separation of source, tests, config | Config files can pollute import path |
| Tool support | Recommended by PyPA, Poetry defaults to it | `uv init` default (without `--lib`) |
| Industry trend | Growing adoption, recommended for production | Fine for scripts and quick prototypes |

The Python Packaging Authority (PyPA) explicitly recommends the src layout for distributable packages. Poetry adopted src layout as its default in February 2025. For a multi-module generation pipeline, src layout prevents subtle import bugs.

**Key insight:** `uv init --lib` gives you src layout automatically. Use it.

**Sources:**
- [src layout vs flat layout - Python Packaging User Guide](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
- [Python Package Structure - pyOpenSci](https://www.pyopensci.org/python-package-guide/package-structure-code/python-package-structure.html)
- [Project Layout - Real Python Best Practices](https://realpython.com/ref/best-practices/project-layout/)

---

## 3. Dependency Management with uv

### Lock File (`uv.lock`)

- Cross-platform lockfile containing **exact resolved versions** of all dependencies
- Human-readable TOML format (unlike pip-compile's requirements.txt)
- **Must be committed to version control** for reproducible builds
- `uv sync` reads uv.lock and installs exactly those versions
- `uv lock` regenerates the lockfile from pyproject.toml constraints

### Automatic Sync

uv automatically verifies the lockfile is up-to-date before every `uv run` invocation. No manual `pip install` or `pip freeze` needed. The workflow is:

1. Edit pyproject.toml (or use `uv add`)
2. uv.lock is regenerated automatically
3. `.venv` is synced automatically on next `uv run`

### Dependency Groups

```toml
[project]
dependencies = [
    "httpx>=0.27",
    "pydantic-settings>=2.7",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.9",
    "mypy>=1.14",
    "pytest-asyncio>=0.24",
]
test = [
    "pytest>=8.0",
    "pytest-cov>=6.0",
]
```

```bash
uv sync                    # install all groups
uv sync --no-group dev     # exclude dev group
uv sync --only-group test  # only test group
```

### Expected Dependencies for Logo Generator

**Runtime:**
- `httpx` - async HTTP client for API calls (replaces aiohttp for modern projects)
- `pydantic` + `pydantic-settings` - data validation and configuration
- `structlog` - structured logging
- `typer` - CLI framework
- `Pillow` - image processing
- `cairosvg` or `svglib` - SVG handling (for Recraft V4 vector output)
- `gradio` - web UI (optional)

**Dev:**
- `pytest` + `pytest-asyncio` - testing
- `ruff` - linting + formatting
- `mypy` - type checking
- `pytest-cov` - coverage

**Sources:**
- [Locking and syncing - uv docs](https://docs.astral.sh/uv/concepts/projects/sync/)
- [Managing dependencies - uv docs](https://docs.astral.sh/uv/concepts/projects/dependencies/)
- [How to use a uv lockfile for reproducible Python environments](https://pydevtools.com/handbook/how-to/how-to-use-a-uv-lockfile-for-reproducible-python-environments/)

---

## 4. Configuration Management

### Recommendation: pydantic-settings

`pydantic-settings` (v2.7+) is the best choice for typed, validated configuration. It follows the Twelve-Factor App pattern by reading from environment variables and `.env` files with full Pydantic validation.

### Configuration Priority (highest to lowest)

1. **`__init__` keyword arguments** (constructor overrides)
2. **Environment variables** (always override .env files)
3. **Dotenv file** (`.env`)
4. **Secrets directory** (for Docker secrets)
5. **Default values** (in the class definition)

### Implementation Pattern

```python
# src/logo_gen/config.py
from pydantic import SecretStr, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="LOGOGEN_",        # all vars prefixed: LOGOGEN_RECRAFT_API_KEY
        case_sensitive=False,
    )

    # API Keys (SecretStr hides values in logs/repr)
    recraft_api_key: SecretStr
    ideogram_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    # Generation defaults
    default_model: str = "recraft-v4"
    default_style: str = "flat"
    default_size: str = "1024x1024"
    max_concurrent_requests: int = 5
    request_timeout: int = 60

    # Output
    output_dir: str = "./output"
    save_metadata: bool = True

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "console"


# Singleton pattern
_settings: Settings | None = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

### .env File

```env
# .env (DO NOT commit to git - add to .gitignore)
LOGOGEN_RECRAFT_API_KEY=sk-recraft-xxxxx
LOGOGEN_IDEOGRAM_API_KEY=sk-ideogram-xxxxx
LOGOGEN_LOG_LEVEL=DEBUG
LOGOGEN_MAX_CONCURRENT_REQUESTS=3
```

### .env.example

```env
# .env.example (commit this to git as a template)
LOGOGEN_RECRAFT_API_KEY=
LOGOGEN_IDEOGRAM_API_KEY=
LOGOGEN_OPENAI_API_KEY=
LOGOGEN_DEFAULT_MODEL=recraft-v4
LOGOGEN_LOG_LEVEL=INFO
```

### Key Benefits

- **Type validation at startup** - fail fast if config is wrong
- **SecretStr** - API keys never appear in logs or stack traces
- **Prefix support** - `LOGOGEN_` prefix avoids collisions with system env vars
- **IDE autocomplete** - full typing support for all settings

**Sources:**
- [Settings Management - Pydantic docs](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Settings and Environment Variables - FastAPI](https://fastapi.tiangolo.com/advanced/settings/)
- [Twelve-Factor Python with Pydantic Settings](https://medium.com/datamindedbe/twelve-factor-python-applications-using-pydantic-settings-f74a69906f2f)

---

## 5. Logging Setup for a Generation Pipeline

### Recommendation: structlog

`structlog` is the gold standard for structured logging in Python. It produces machine-parseable JSON logs in production and colorful, human-readable output in development.

### Setup

```python
# src/logo_gen/logging.py
import logging
import sys
import structlog


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """Configure structlog for the application."""

    # Shared processors for both structlog and stdlib
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,      # async-safe context
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "console":
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging to use structlog formatting
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level.upper())

    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
```

### Usage in Generation Pipeline

```python
import structlog

log = structlog.get_logger()

async def generate_logo(prompt: str, model: str) -> Path:
    # Bind context that follows all subsequent log calls
    log_ctx = log.bind(prompt=prompt[:50], model=model)

    await log_ctx.ainfo("generation_started")

    try:
        result = await call_api(prompt, model)
        await log_ctx.ainfo("generation_completed",
                            duration_ms=result.duration,
                            image_size=result.size)
        return result.path
    except APIError as e:
        await log_ctx.aerror("generation_failed",
                             error=str(e),
                             status_code=e.status_code)
        raise
```

### Async Context Isolation

For concurrent generation tasks, use `contextvars` to isolate log context per task:

```python
import structlog

async def process_request(request_id: str, prompt: str):
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    log = structlog.get_logger()
    await log.ainfo("processing_request", prompt=prompt[:50])
    # All subsequent logs in this async context include request_id
```

### JSON Log Output (production)

```json
{"event": "generation_started", "prompt": "minimalist tech startup logo", "model": "recraft-v4", "level": "info", "timestamp": "2026-03-27T10:15:30Z"}
{"event": "generation_completed", "prompt": "minimalist tech startup logo", "model": "recraft-v4", "duration_ms": 3420, "image_size": "1024x1024", "level": "info", "timestamp": "2026-03-27T10:15:33Z"}
```

**Sources:**
- [Structlog Documentation](https://www.structlog.org/en/stable/getting-started.html)
- [Comprehensive Guide to Structlog - Better Stack](https://betterstack.com/community/guides/logging/structlog/)
- [Complete Guide to Logging with StructLog - SigNoz](https://signoz.io/guides/structlog/)

---

## 6. Async Architecture for Parallel API Calls

### Why Async?

Our logo generator needs to call multiple external APIs (Recraft, Ideogram, OpenAI, etc.) concurrently. Async IO avoids blocking on network calls, enabling parallel generation from multiple models simultaneously.

### Core Pattern: httpx + asyncio

`httpx` is the recommended async HTTP client (replaces aiohttp for modern Python). It has a requests-like API, native async/await support, and HTTP/2.

```python
# src/logo_gen/clients/base.py
import httpx
import asyncio
from abc import ABC, abstractmethod


class BaseImageClient(ABC):
    def __init__(self, api_key: str, timeout: int = 60, max_concurrent: int = 5):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=max_concurrent),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self._client.aclose()

    async def generate(self, prompt: str, **kwargs) -> GenerationResult:
        async with self._semaphore:
            return await self._generate(prompt, **kwargs)

    @abstractmethod
    async def _generate(self, prompt: str, **kwargs) -> GenerationResult:
        ...
```

### Parallel Generation Across Models

```python
# src/logo_gen/pipeline.py
import asyncio
import structlog

log = structlog.get_logger()


async def generate_from_multiple_models(
    prompt: str,
    models: list[str],
    clients: dict[str, BaseImageClient],
) -> list[GenerationResult]:
    """Generate logos from multiple models in parallel."""

    tasks = []
    for model in models:
        client = clients[model]
        task = asyncio.create_task(
            client.generate(prompt),
            name=f"generate-{model}",
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = []
    for model, result in zip(models, results):
        if isinstance(result, Exception):
            await log.aerror("model_failed", model=model, error=str(result))
        else:
            successful.append(result)

    return successful
```

### Concurrency Control with Semaphore

```python
# Limit concurrent requests per API provider
semaphore = asyncio.Semaphore(5)

async def rate_limited_call(client, prompt):
    async with semaphore:
        return await client.generate(prompt)
```

### Retry with Exponential Backoff

```python
import asyncio
import httpx

async def retry_with_backoff(
    func,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs,
):
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except (httpx.HTTPStatusError, httpx.TimeoutException) as e:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            await asyncio.sleep(delay)
```

### Application Entry Point

```python
# src/logo_gen/main.py
import asyncio
from logo_gen.config import get_settings
from logo_gen.clients.recraft import RecraftClient
from logo_gen.clients.ideogram import IdeogramClient
from logo_gen.pipeline import generate_from_multiple_models


async def run(prompt: str, models: list[str]) -> list[Path]:
    settings = get_settings()

    async with (
        RecraftClient(settings.recraft_api_key.get_secret_value()) as recraft,
        IdeogramClient(settings.ideogram_api_key.get_secret_value()) as ideogram,
    ):
        clients = {"recraft-v4": recraft, "ideogram-3": ideogram}
        results = await generate_from_multiple_models(prompt, models, clients)
        return [r.save(settings.output_dir) for r in results]
```

**Sources:**
- [Making Fast Parallel Requests with Asyncio](https://proxiesapi.com/articles/making-fast-parallel-requests-with-asyncio)
- [Asynchronous HTTP Requests with aiohttp and asyncio - Twilio](https://www.twilio.com/en-us/blog/asynchronous-http-requests-in-python-with-aiohttp)
- [Python Async Architecture: Real-World Experience](https://xaviercollantes.dev/articles/python-async)

---

## 7. CLI Interface vs Web UI

### CLI: Typer (Recommended for Primary Interface)

Typer builds on Click, uses Python type hints to eliminate boilerplate, and provides automatic shell completion.

```python
# src/logo_gen/cli.py
import typer
import asyncio
from pathlib import Path
from typing import Optional
from logo_gen.config import get_settings
from logo_gen.logging import setup_logging

app = typer.Typer(name="logo-gen", help="AI-powered logo generation tool")


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Logo description"),
    model: str = typer.Option("recraft-v4", "--model", "-m", help="Model to use"),
    style: str = typer.Option("flat", "--style", "-s", help="Logo style"),
    output: Path = typer.Option("./output", "--output", "-o", help="Output directory"),
    count: int = typer.Option(1, "--count", "-n", help="Number of variations"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Generate a logo from a text prompt."""
    settings = get_settings()
    setup_logging(
        log_level="DEBUG" if verbose else settings.log_level,
        log_format="console",
    )
    results = asyncio.run(run_generation(prompt, model, style, output, count))
    for path in results:
        typer.echo(f"Saved: {path}")


@app.command()
def models():
    """List available models."""
    typer.echo("Available models:")
    typer.echo("  recraft-v4     - Best for vector/SVG logos")
    typer.echo("  ideogram-3     - Best for text in logos")
    typer.echo("  flux-2-pro     - Best open-weight model")


@app.command()
def ui():
    """Launch the web UI (Gradio)."""
    from logo_gen.web import launch_ui
    launch_ui()
```

Usage:
```bash
logo-gen generate "minimalist owl logo for tech startup" --model recraft-v4 --count 3
logo-gen models
logo-gen ui
```

### Web UI: Gradio (Recommended for Visual Interface)

Gradio is the best fit for image generation UIs -- it has built-in image display widgets, gallery components, and easy sharing.

```python
# src/logo_gen/web.py
import gradio as gr
import asyncio
from logo_gen.pipeline import generate_from_multiple_models


def launch_ui():
    with gr.Blocks(title="Logo Generator") as demo:
        gr.Markdown("# AI Logo Generator")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Describe your logo", lines=3)
                model = gr.Dropdown(
                    choices=["recraft-v4", "ideogram-3", "flux-2-pro"],
                    value="recraft-v4",
                    label="Model",
                )
                style = gr.Dropdown(
                    choices=["flat", "gradient", "3d", "hand-drawn", "minimal"],
                    value="flat",
                    label="Style",
                )
                count = gr.Slider(1, 8, value=4, step=1, label="Variations")
                btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                gallery = gr.Gallery(label="Generated Logos", columns=2)

        btn.click(fn=generate_handler, inputs=[prompt, model, style, count], outputs=gallery)

    demo.launch()
```

### Comparison Summary

| Aspect | Typer CLI | Gradio Web UI | FastAPI Backend |
|--------|-----------|---------------|-----------------|
| Use case | Scripting, CI/CD, power users | Visual exploration, demos | Production API |
| Setup effort | Low | Low | Medium |
| Image viewing | External viewer | Built-in gallery | Needs frontend |
| Sharing | N/A | `share=True` for public URL | Deploy required |
| Batch ops | Excellent | Possible but clunky | Excellent |
| Async support | via `asyncio.run()` | Built-in | Native |

### Recommendation

Implement **both** Typer CLI and Gradio Web UI:
- **CLI** as the primary interface for scripting, automation, and CI/CD
- **Gradio** as the visual exploration tool, launched via `logo-gen ui`
- **FastAPI** later if a production REST API is needed

**Sources:**
- [Typer - Alternatives and Comparisons](https://typer.tiangolo.com/alternatives/)
- [Python CLI Tools with Click and Typer - DevToolbox](https://devtoolbox.dedyn.io/blog/python-click-typer-cli-guide)
- [Streamlit vs Gradio in 2025 - Squadbase](https://www.squadbase.dev/en/blog/streamlit-vs-gradio-in-2025-a-framework-comparison-for-ai-apps)
- [Streamlit vs Gradio in 2026 - Markaicode](https://markaicode.com/vs/streamlit-vs-gradio-in/)

---

## 8. Recommended Project Structure for Logo Generator

```
logo-gen/
├── .env                          # Local env vars (gitignored)
├── .env.example                  # Template for env vars (committed)
├── .gitignore
├── .python-version               # e.g., 3.12
├── pyproject.toml                # Project metadata, deps, tool config
├── uv.lock                       # Lockfile (committed)
├── README.md
│
├── src/
│   └── logo_gen/
│       ├── __init__.py           # Package version, public API
│       ├── py.typed              # PEP 561 type marker
│       │
│       ├── cli.py                # Typer CLI entry point
│       ├── web.py                # Gradio web UI
│       ├── main.py               # Core orchestration / async entry
│       │
│       ├── config.py             # pydantic-settings configuration
│       ├── logging.py            # structlog setup
│       ├── models.py             # Pydantic data models (GenerationRequest, Result, etc.)
│       │
│       ├── clients/              # External API clients
│       │   ├── __init__.py
│       │   ├── base.py           # Abstract base client (BaseImageClient)
│       │   ├── recraft.py        # Recraft V4 client
│       │   ├── ideogram.py       # Ideogram 3.0 client
│       │   ├── openai.py         # GPT Image 1.5 client
│       │   └── flux.py           # Flux 2 (via fal.ai) client
│       │
│       ├── pipeline/             # Generation pipeline
│       │   ├── __init__.py
│       │   ├── orchestrator.py   # Multi-model parallel generation
│       │   ├── prompt.py         # Prompt engineering / enhancement
│       │   ├── postprocess.py    # Image post-processing (resize, format convert)
│       │   └── vectorize.py      # SVG/vector handling
│       │
│       └── utils/                # Shared utilities
│           ├── __init__.py
│           ├── retry.py          # Retry with backoff
│           └── images.py         # Image I/O helpers
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # Shared fixtures
│   ├── test_config.py
│   ├── test_pipeline/
│   │   ├── __init__.py
│   │   ├── test_orchestrator.py
│   │   └── test_prompt.py
│   └── test_clients/
│       ├── __init__.py
│       └── test_recraft.py
│
├── output/                       # Generated images (gitignored)
│
└── research/                     # Research documents
    └── 10_project_structure.md
```

### Design Rationale

| Directory | Purpose |
|-----------|---------|
| `src/logo_gen/` | Src layout prevents import bugs; `uv init --lib` creates this |
| `clients/` | Each API provider gets its own module inheriting from `BaseImageClient` |
| `pipeline/` | Separates orchestration from API calls; prompt engineering is its own concern |
| `config.py` | Single source of truth for all settings via pydantic-settings |
| `logging.py` | Centralized structlog config, imported once at startup |
| `cli.py` + `web.py` | Two interfaces to the same pipeline -- CLI for automation, Gradio for exploration |
| `models.py` | Pydantic models shared across clients and pipeline (request/response schemas) |
| `tests/` | Mirrors src structure; uses pytest-asyncio for async tests |

### Key Architectural Decisions

1. **src layout** via `uv init --lib` -- prevents import path issues
2. **pydantic-settings** for config -- type-safe, env var driven, fail-fast
3. **structlog** for logging -- structured JSON in prod, pretty console in dev
4. **httpx** for HTTP -- modern async client with HTTP/2 support
5. **Typer** for CLI -- type-hint based, minimal boilerplate
6. **Gradio** for web UI -- built-in image gallery, easy to prototype
7. **asyncio.Semaphore** for rate limiting -- per-provider concurrency control
8. **Abstract base client** pattern -- add new API providers without changing pipeline code

### Quick Start Sequence

```bash
# 1. Initialize
uv init --lib logo-gen
cd logo-gen

# 2. Set Python version
echo "3.12" > .python-version

# 3. Add dependencies
uv add httpx pydantic-settings structlog typer Pillow
uv add --dev pytest pytest-asyncio ruff mypy pytest-cov

# 4. Create structure
mkdir -p src/logo_gen/{clients,pipeline,utils} tests/{test_pipeline,test_clients}

# 5. Copy .env.example to .env and fill in API keys
cp .env.example .env

# 6. Run
uv run logo-gen generate "minimalist owl logo" --model recraft-v4

# 7. Launch web UI
uv run logo-gen ui
```

---

## Summary of Recommendations

| Topic | Recommendation |
|-------|---------------|
| Package manager | **uv** (fast, all-in-one, lockfile) |
| Project layout | **src/ layout** via `uv init --lib` |
| Dependency lock | **uv.lock** committed to git |
| Configuration | **pydantic-settings** with `.env` files |
| Logging | **structlog** with JSON (prod) / console (dev) |
| HTTP client | **httpx** (async, HTTP/2, modern API) |
| Concurrency | **asyncio** + Semaphore for rate limiting |
| CLI framework | **Typer** (type-hint based, built on Click) |
| Web UI | **Gradio** (built-in image components) |
| Testing | **pytest** + **pytest-asyncio** |
| Linting | **ruff** (fast, replaces flake8 + isort + black) |
| Type checking | **mypy** with strict mode |
