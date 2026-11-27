"""FastAPI app: JSON API under /api and the built frontend at /."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from logo_gen import chats, generator, library, paths, postprocess, prompt_engine, rankings, runtime, system, turns, upscaler
from logo_gen.clients import claude_cli, local_llm, openrouter
from logo_gen.clients.local_llm import LocalLLMError
from logo_gen.clients.openrouter import OpenRouterError
from logo_gen.config import (
    SECRET_KEYS,
    get_secret,
    get_settings,
    mask,
    reset_settings,
    secret_source,
    set_secret,
    update_settings,
)
from logo_gen.jobs import jobs
from logo_gen.local_models import manager
from logo_gen.local_models.registry import MODELS
from logo_gen.prompting import preferences, profiles
from logo_gen.prompting.render import render_direction

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    paths.ensure_dirs()
    library.conn()
    turns.recover()
    chats.purge()
    preferences.sync_feedback()  # reactions stored while the prompt layer wasn't reading them
    await run_in_threadpool(library.import_existing)
    background = [asyncio.create_task(manager.refresh_remote()), asyncio.create_task(asyncio.to_thread(_warm_rembg))]
    if get_secret("OPENROUTER_KEY"):
        background.append(asyncio.create_task(_warm_catalog()))
    yield
    for t in background:
        t.cancel()
    await manager.shutdown()


def _warm_rembg() -> None:
    """The first rembg import compiles numba code (about a minute in a fresh venv). Do it before anyone waits on it."""
    try:
        import rembg  # noqa: F401
    except Exception:  # noqa: BLE001
        pass


async def _warm_catalog() -> None:
    try:
        await openrouter.catalog()
    except Exception:  # noqa: BLE001
        pass


app = FastAPI(title="logo-gen", lifespan=lifespan)


# ── Errors ───────────────────────────────────────────────────────────


@app.exception_handler(OpenRouterError)
async def _openrouter_error(_: Request, e: OpenRouterError):
    status = e.status if e.status in (400, 401, 402, 403, 408, 429) else 502
    return JSONResponse({"error": e.message, "kind": e.kind, "retry_after": e.retry_after}, status_code=status)


@app.exception_handler(LocalLLMError)
async def _local_llm_error(_: Request, e: LocalLLMError):
    return JSONResponse({"error": e.message, "kind": "local_llm"}, status_code=502)


@app.exception_handler(ValueError)
async def _value_error(_: Request, e: ValueError):
    return JSONResponse({"error": str(e), "kind": "bad_request"}, status_code=400)


@app.exception_handler(turns.TurnBusy)
async def _turn_busy(_: Request, e: turns.TurnBusy):
    return JSONResponse({"error": e.message, "kind": "busy"}, status_code=409)


def _not_found(what: str) -> HTTPException:
    return HTTPException(404, f"{what} not found")


def _error_text(e: BaseException) -> str:
    return getattr(e, "message", None) or str(e) or e.__class__.__name__


def _refs(rel_paths: list[str] | None) -> list[Path]:
    out = []
    for rel in rel_paths or []:
        try:
            p = paths.resolve_output(rel)
        except ValueError:
            continue
        if p.exists():
            out.append(p)
    return out


# ── System and settings ─────────────────────────────────────────────


@app.get("/api/health")
async def health():
    """Liveness check: {"ok": true}."""
    return {"ok": True}


@app.get("/api/system")
async def system_info():
    """This machine: OS, device (cuda, mps or cpu), GPU and VRAM, free disk, and library totals."""
    return system.info() | {"library": library.stats()}


def _secrets_status() -> dict:
    return {
        name: {"set": bool(get_secret(name)), "source": secret_source(name), "masked": mask(get_secret(name))}
        for name in SECRET_KEYS
    }


@app.get("/api/settings")
async def read_settings():
    """Current settings, plus which secrets are set (never their values)."""
    return {"settings": get_settings().model_dump(), "secrets": _secrets_status()}


@app.put("/api/settings")
async def write_settings(patch: dict[str, Any]):
    """Change any subset of settings, e.g. {"image_models": [...], "images_per_prompt": 2}. Returns the full settings."""
    s = update_settings(patch)
    return {"settings": s.model_dump(), "secrets": _secrets_status()}


@app.post("/api/settings/reset")
async def restore_settings():
    s = reset_settings()
    return {"settings": s.model_dump(), "secrets": _secrets_status()}


class SecretIn(BaseModel):
    name: str
    value: str = ""


@app.put("/api/secrets")
async def write_secret(body: SecretIn):
    """Set or clear OPENROUTER_KEY or HF_TOKEN in .env. An OpenRouter key is checked with OpenRouter first."""
    if body.name not in SECRET_KEYS:
        raise HTTPException(400, "Unknown secret")
    if body.name == "OPENROUTER_KEY" and body.value.strip():
        check = await openrouter.key_status(body.value.strip())
        if not check.get("valid"):
            raise HTTPException(400, check.get("error") or "OpenRouter rejected this key")
    set_secret(body.name, body.value)
    if body.name == "HF_TOKEN":
        manager._access.clear()
    return {"secrets": _secrets_status()}


@app.get("/api/openrouter/key")
async def openrouter_key():
    """Whether the saved OpenRouter key works, what it has spent and its limit."""
    return await openrouter.key_status()


@app.get("/api/openrouter/models")
async def openrouter_models(force: bool = False):
    """The live OpenRouter catalog, split into chat and image models, with prices and capabilities. Cached 10
    minutes; force=true refetches.
    """
    data = await openrouter.catalog(force=force)
    return {"fetched_at": data["fetched_at"], "chat": data["chat"], "image": data["image"], "stale": data.get("stale", False)}


@app.get("/api/models/rankings")
async def model_rankings(refresh: bool = False):
    """Quality (LMArena), value and popularity (OpenRouter usage) for every image model we can use."""
    key = get_secret("OPENROUTER_KEY")
    board, pop, notes = await rankings.data(key or None, refresh=refresh)
    models: list[dict] = []
    if key:
        try:
            cat = await openrouter.catalog()
            models += [{"id": m["id"], "price": m.get("price_per_image")} for m in cat["image"]]
        except OpenRouterError as e:
            notes.append(f"No OpenRouter prices: {e.message}")
    # Ranks count only what you can use now; local models that aren't installed or can't run here get a score but no rank.
    usable = {mid for mid, spec in MODELS.items() if manager.is_installed(spec) and spec.support(system.device_kind()) != "no"}
    models += [{"id": f"{generator.LOCAL_PREFIX}{mid}", "price": 0.0} for mid in usable]
    ranked = rankings.rank_models(models, board, pop)
    for mid in MODELS.keys() - usable:
        ranked |= {k: v | {"rank": None, "value": None, "pareto": False}
                   for k, v in rankings.rank_models([{"id": f"{generator.LOCAL_PREFIX}{mid}", "price": 0.0}], board, None).items()}
    return {
        "source": board["source"] | {"bundled": bool(board.get("bundled")), "fetched_at": board.get("fetched_at")},
        "popularity": pop["source"] if pop else None,
        "notes": notes,
        "formula": {
            "elo_per_doubling": rankings.ELO_PER_DOUBLING,
            "price_ref": rankings.PRICE_REF,
            "price_floor": rankings.PRICE_FLOOR,
        },
        "models": ranked,
        "recent": [r["model"] for r in library._q(
            "SELECT model, MAX(created_at) AS t FROM images WHERE kind = 'generated' AND model != '' GROUP BY model ORDER BY t DESC LIMIT 5"
        )],
    }


@app.get("/api/llm/local")
async def local_llm_models():
    models = await local_llm.list_models()
    return {"reachable": bool(models), "models": models, "base_url": get_settings().local_llm_base_url}


@app.get("/api/llm/claude-cli")
async def claude_cli_status():
    return {"available": await claude_cli.is_available()}


# ── Local models ────────────────────────────────────────────────────

_last_refresh = 0.0


@app.get("/api/local-models")
async def list_local_models():
    """Local models with install state, download progress, device support, batch size and caps."""
    global _last_refresh
    if time.time() - _last_refresh > manager.REMOTE_TTL:
        _last_refresh = time.time()
        asyncio.create_task(manager.refresh_remote())
    s = get_settings()
    models = []
    for m in manager.list_models():
        spec = MODELS[m["id"]]
        chosen = s.local_batch.get(spec.id)
        models.append(m | {
            "batch": chosen or spec.max_batch,
            "batch_source": "calibrated" if chosen else "default",
            "calibration": library.get_calibration(spec.id),
            "cap": s.cap_for(m["ref"]),
        })
    return {"models": models, "system": system.info(), "runtime": runtime.status()}


def _spec(model_id: str):
    spec = MODELS.get(model_id)
    if not spec:
        raise _not_found("Model")
    return spec


@app.get("/api/local-stack")
async def local_stack():
    return runtime.status()


@app.post("/api/local-stack/install")
async def install_local_stack():
    if not runtime.state.running and not system.local_stack_installed():
        async def go():
            try:
                await runtime.ensure_installed()
            except Exception:  # noqa: BLE001
                pass
        asyncio.create_task(go())
        await asyncio.sleep(0.1)
    return runtime.status()


@app.post("/api/local-models/{model_id}/calibrate")
async def calibrate_local_model(model_id: str):
    """Speed test for a local model: times batch sizes and peak VRAM. Returns a job; its result has the options."""
    spec = _spec(model_id)
    if not manager.is_installed(spec):
        raise ValueError(f"Install {spec.name} first.")
    if spec.support(system.device_kind()) == "no":
        raise ValueError(f"{spec.name} cannot run on this machine.")
    for j in jobs.list():
        if j.kind == "calibrate" and j.status == "running":
            raise ValueError("A speed test is already running.")
    max_batch = min(get_settings().cap_for(f"local/{spec.id}"), 16)

    async def fn(job):
        from logo_gen.local_models import diffusion

        def on_status(msg: str) -> None:
            job.message = msg
            if msg.startswith("Testing a batch of "):
                job.done += 1

        job.progress(done=0, total=len([b for b in (1, 2, 3, 4, 6, 8, 12, 16) if b <= max_batch]))
        data = await asyncio.to_thread(diffusion.calibrate, spec.id, max_batch, on_status, lambda: job.cancelled)
        library.save_calibration(spec.id, data)
        job.result.update(data)
        best = data["options"][0] if data["options"] else None
        job.message = f"Fastest: batch of {best['batch']}, {best['per_image']} s per image" if best else "No batch size worked"
        job.done = job.total

    return jobs.start("calibrate", f"Speed test: {spec.name}", fn).to_dict()


@app.post("/api/local-models/{model_id}/install")
async def install_local_model(model_id: str):
    """Start or resume downloading a local model. Progress shows in GET /api/local-models."""
    return manager.start_install(_spec(model_id))


@app.post("/api/local-models/{model_id}/cancel")
async def cancel_local_model(model_id: str):
    """Pause a download. Installing again resumes it."""
    return await manager.cancel_install(_spec(model_id))


@app.delete("/api/local-models/{model_id}")
async def uninstall_local_model(model_id: str):
    """Delete a local model's files."""
    return await manager.uninstall(_spec(model_id))


@app.post("/api/local-models/{model_id}/check-access")
async def check_local_model(model_id: str):
    """For gated models: whether HF_TOKEN can download it."""
    spec = _spec(model_id)
    await manager.access(spec, force=True)
    try:
        await manager.remote_files(spec, force=True)
    except Exception:  # noqa: BLE001
        pass
    return manager.status(spec)


# ── Jobs ────────────────────────────────────────────────────────────


@app.get("/api/jobs")
async def list_jobs():
    """Recent jobs, newest first."""
    return [j.to_dict() for j in jobs.list()]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """One job: status (running, done, error, cancelled), done/total, message, image_ids, cost, byok, notes, errors."""
    job = jobs.get(job_id)
    if not job:
        raise _not_found("Job")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    job = jobs.cancel(job_id)
    if not job:
        raise _not_found("Job")
    return job.to_dict()


# ── Generation ──────────────────────────────────────────────────────


class PromptIn(BaseModel):
    prompt: str = ""
    concept: str = ""
    style: str = ""
    # The design brief this direction came from. When present, each model
    # gets its own prompt rendered from it and `prompt` is only a summary.
    brief: dict | None = None
    # Hand-edited prompts per model id: {"model": {"prompt": ..., "negative_prompt": ...}}
    overrides: dict[str, dict] | None = None
    # Send `prompt` exactly as written to every model.
    verbatim: bool = False
    negative_prompt: str = ""


class GenerateIn(BaseModel):
    prompts: list[PromptIn]
    models: list[str] | None = None
    images_per_prompt: int | None = None
    references: list[str] = Field(default_factory=list)
    chat_id: str | None = None
    message_id: int | None = None  # the designer reply whose directions these are


class QuickIn(BaseModel):
    brief: str
    count: int | None = None
    enhance: bool = True
    models: list[str] | None = None
    images_per_prompt: int | None = None
    references: list[str] = Field(default_factory=list)


def _check_models(models: list[str] | None) -> list[str]:
    models = [m for m in (models or get_settings().image_models) if m]
    if not models:
        raise ValueError("No image models selected.")
    for m in models:
        if generator.is_local(m):
            spec = MODELS.get(m.removeprefix(generator.LOCAL_PREFIX))
            if not spec:
                raise ValueError(f"Unknown local model {m}")
            if not manager.is_installed(spec):
                raise ValueError(f"{spec.name} is not installed. Install it on the Models page first.")
        elif not get_secret("OPENROUTER_KEY"):
            raise ValueError("Add an OpenRouter API key in Settings to use cloud models.")
    return models


def _usable(model: str | None) -> bool:
    if not model:
        return False
    if generator.is_local(model):
        spec = MODELS.get(model.removeprefix(generator.LOCAL_PREFIX))
        return bool(spec and manager.is_installed(spec))
    return "/" in model and bool(get_secret("OPENROUTER_KEY"))


@app.post("/api/generate")
async def generate(body: GenerateIn):
    """Render your own prompts with the given (or default) image models. Returns a job."""
    prompts = [p.model_dump(exclude_none=True) for p in body.prompts if p.prompt.strip() or p.brief]
    if not prompts:
        raise ValueError("No prompts to generate.")
    models = _check_models(body.models)
    refs = _refs(body.references)
    return start_generation(prompts, models, body.images_per_prompt, refs, body.chat_id, body.message_id).to_dict()


def start_generation(prompts: list[dict], models: list[str], per: int | None, refs: list[Path],
                     chat_id: str | None = None, message_id: int | None = None, notes: list[str] | None = None):
    """Start a generate job. With a chat, its images show inline under message_id."""
    title = prompts[0].get("concept") or prompts[0].get("prompt", "")[:40]

    async def fn(job):
        job.result["prompts"] = prompts
        job.notes.extend(notes or [])
        await generator.run(job, prompts, models, per, refs, chat_id=chat_id)

    job = jobs.start("generate", title, fn)
    if chat_id:
        job.chat_id = chat_id
        chats.link_job(job.id, chat_id, message_id, "generate")
    return job


def _money(x: float) -> str:
    return f"${x:.3f}" if x < 0.1 else f"${x:.2f}"


async def auto_round(chat_id: str, message_id: int, prompts: list[dict], history: list[dict]) -> str:
    """Render a designer reply's directions right away, as a server job. Returns a line for the reply.

    Only runs when "Generate automatically" is on (Settings, or the chat's own switch).

    Uses the selected image models and images per direction (Settings, or the composer), within the
    per-model caps. Models that can't run here are skipped with a reason. When the round's estimated
    cloud cost is more than the key has left under its spending limit, the cloud models are skipped too.
    """
    s = get_settings()
    selected = [m for m in s.image_models if m]
    if not selected:
        return "Pick an image model below the message box and press Generate to render these."
    per = max(1, int(s.images_per_prompt or 1))
    usable, skipped = [], []
    for m in selected:
        if _usable(m):
            usable.append(m)
        else:
            skipped.append(f"{m.removeprefix('local/')} (not installed)" if generator.is_local(m) else f"{m} (no OpenRouter key)")
    if usable and any(generator.is_local(m) for m in usable):
        device = system.device_kind()
        for m in [m for m in usable if generator.is_local(m)]:
            spec = MODELS.get(m.removeprefix(generator.LOCAL_PREFIX))
            if spec and spec.support(device) == "no":
                usable.remove(m)
                skipped.append(f"{spec.name} (can't run on this machine)")
    work, cap_notes = generator.plan(usable, prompts, per)
    parts, cloud_cost, unknown = [], 0.0, []
    for m, (ps, n) in work.items():
        count = len(ps) * n
        name = m.removeprefix("local/") if generator.is_local(m) else m.split("/", 1)[-1]
        if generator.is_local(m):
            parts.append(f"{count} on {name} (local, free)")
            continue
        price = (await openrouter.model_info(m) or {}).get("price_per_image")
        if price is None:
            unknown.append(m)
            parts.append(f"{count} on {name} (price unknown)")
        else:
            cloud_cost += price * count
            parts.append(f"{count} on {name} (about {_money(price * count)})")
    cloud = [m for m in work if not generator.is_local(m)]
    if cloud and cloud_cost > 0:
        key = await openrouter.key_status()
        left = key.get("limit_remaining")
        if isinstance(left, (int, float)) and cloud_cost > left:
            for m in cloud:
                work.pop(m)
                skipped.append(f"{m} (about {_money(cloud_cost)} for this round, but the key has {_money(left)} left)")
            parts = [p for p in parts if "(local, free)" in p]
            cloud_cost = 0.0
    note_skipped = f" Skipped: {'; '.join(skipped)}." if skipped else ""
    if not work:
        return f"Not rendering these automatically.{note_skipped} Press Generate when you're ready."
    refs: list[Path] = []
    for m in history:
        refs.extend(_refs(m.get("images") or []))
    total = sum(len(ps) * n for ps, n in work.values())
    cost = f" About {_money(cloud_cost)} in all." if cloud_cost > 0 else ""
    line = f"Rendering {total} image{'s' if total != 1 else ''} now: {', '.join(parts)}.{cost}{note_skipped}"
    if cap_notes:
        line += " Some were trimmed to the per-model caps in Settings."
    start_generation(prompts, list(work), per, refs, chat_id, message_id, notes=[line])
    return line


@app.post("/api/generate/quick")
async def quick_generate(body: QuickIn):
    """Brief in: the chat model writes `count` directions, then each is rendered. Returns a job."""
    brief = body.brief.strip()
    if not brief:
        raise ValueError("Describe what the logo is for.")
    models = _check_models(body.models)
    refs = _refs(body.references)
    count = max(1, min(body.count or get_settings().variations, 8))

    async def fn(job):
        if body.enhance:
            job.progress(message=f"Writing {count} directions with {prompt_engine.backend_label()}")
            prompts, cost = await prompt_engine.directions(brief, count, refs)
            job.add_cost("prompts", cost)
        else:
            prompts = [{"prompt": brief, "concept": brief[:48], "style": ""}]
        job.result["prompts"] = prompts
        job.check()
        await generator.run(job, prompts, models, body.images_per_prompt, refs)

    return jobs.start("quick", brief[:60], fn).to_dict()


class VariationIn(BaseModel):
    count: int = 4
    mode: str = "remix"  # remix: new prompts close to the original; reroll: same prompt, new seeds
    instruction: str = ""
    model: str | None = None


@app.post("/api/images/{image_id}/variations")
async def variations(image_id: str, body: VariationIn):
    """New images close to this one (mode remix writes new prompts; reroll keeps the prompt with new seeds). Returns a job."""
    img = library.get_image(image_id)
    if not img:
        raise _not_found("Image")
    source = {"prompt": img["prompt"], "concept": img["concept"], "style": img["style"], "brief": img.get("brief"),
              "chat_id": img.get("chat_id")}
    if not source["prompt"]:
        raise ValueError("This image has no prompt on record, so there is nothing to vary.")
    model = body.model or img["model"]
    if not _usable(model):
        model = (get_settings().image_models or [""])[0]
    models = _check_models([model])
    count = max(1, min(body.count, 8))
    src_path = paths.resolve_output(img["path"])
    refs: list[Path] = []
    if not generator.is_local(model) and not img["vector"]:
        info = await openrouter.model_info(model) or {}
        if info.get("supports_references"):
            refs = [src_path]

    async def fn(job):
        if body.mode == "reroll":
            # The same rendered prompt, new seeds.
            same = {"concept": img["concept"], "style": img["style"], "prompt": img["prompt"], "verbatim": True,
                    "negative_prompt": img.get("negative_prompt") or "", "brief": img.get("brief")}
            prompts, per = [same], count
        else:
            job.progress(message="Writing variations")
            job.chat_id = img.get("chat_id")
            prompts, cost = await prompt_engine.variations(source, count, body.instruction)
            job.add_cost("prompts", cost)
            per = 1
        job.result["prompts"] = prompts
        job.result["source_id"] = image_id
        await generator.run(job, prompts, models, per, refs, chat_id=img.get("chat_id"), parent_id=image_id)

    job = jobs.start("variations", f"Variations of {img['concept'] or img['model']}", fn)
    if img.get("chat_id"):
        # Variations show up in the chat under the same reply as the image they vary.
        job.chat_id = img["chat_id"]
        chats.link_job(job.id, img["chat_id"], chats.job_message(img.get("job_id")), "variations")
    return job.to_dict()


class RegenerateIn(BaseModel):
    prompt: str
    negative_prompt: str = ""
    count: int = 1
    model: str | None = None


@app.post("/api/images/{image_id}/regenerate")
async def regenerate_image(image_id: str, body: RegenerateIn):
    """Run a hand-edited prompt on this image's model. Results land under it as variations."""
    img = library.get_image(image_id)
    if not img:
        raise _not_found("Image")
    if not body.prompt.strip():
        raise ValueError("The prompt is empty.")
    model = body.model or img["model"]
    models = _check_models([model])
    direction = {"prompt": body.prompt.strip(), "verbatim": True, "negative_prompt": body.negative_prompt,
                 "concept": img["concept"], "style": img["style"], "brief": img.get("brief")}

    async def fn(job):
        job.result["prompts"] = [direction]
        job.result["source_id"] = image_id
        await generator.run(job, [direction], models, max(1, min(body.count, 8)), chat_id=img.get("chat_id"),
                            parent_id=image_id)

    job = jobs.start("variations", f"Edited prompt: {img['concept'] or model}", fn)
    if img.get("chat_id"):
        # Like variations: the results show in the chat under the reply that made the original.
        job.chat_id = img["chat_id"]
        chats.link_job(job.id, img["chat_id"], chats.job_message(img.get("job_id")), "variations")
    return job.to_dict()


# ── Prompts and preferences ─────────────────────────────────────────


@app.get("/api/prompts/profiles")
async def prompt_profiles():
    """The prompt profiles, and which one each selected or local model uses."""
    s = get_settings()
    known = list(dict.fromkeys([*s.image_models, *(f"local/{m}" for m in MODELS)]))
    return {
        "profiles": [p.to_dict() for p in [*profiles.PROFILES, profiles.DEFAULT]],
        "models": {m: profiles.profile_for(m).id for m in known},
    }


class RenderIn(BaseModel):
    prompts: list[PromptIn]
    models: list[str] | None = None
    chat_id: str | None = None


@app.post("/api/prompts/render")
async def render_prompts(body: RenderIn):
    """Preview what each model would be sent for each direction, without generating."""
    s = get_settings()
    models = [m for m in (body.models or s.image_models) if m]
    avoid = preferences.render_hints(body.chat_id)
    out = []
    for i, p in enumerate(body.prompts):
        d = p.model_dump(exclude_none=True)
        for m in models:
            info = None if generator.is_local(m) else await openrouter.model_info(m)
            out.append(render_direction(d, m, info, s.transparent_background, avoid).to_dict() | {"direction": i})
    return {"rendered": out}


@app.get("/api/preferences")
async def get_preferences(chat_id: str | None = None):
    """What the designer learned from reactions and typed feedback: one chat's profile, the cross-chat one,
    and the exact text it adds to the designer's system prompt."""
    return {
        "chat": preferences.profile(preferences.chat_scope(chat_id)) if chat_id else None,
        "global": preferences.profile(preferences.GLOBAL),
        "global_enabled": get_settings().use_global_taste,
        "designer_context": preferences.designer_context(chat_id),
    }


@app.delete("/api/preferences")
async def clear_preferences(chat_id: str | None = None, scope: str | None = None):
    """Forget what the designer learned in one chat (chat_id), or across chats (scope=global).
    The feedback events themselves stay."""
    if scope == "global":
        preferences.clear(preferences.GLOBAL)
    elif chat_id:
        preferences.clear(preferences.chat_scope(chat_id))
    else:
        raise ValueError("Pass chat_id, or scope=global.")
    return await get_preferences(chat_id)


# ── Chats ───────────────────────────────────────────────────────────


class ChatIn(BaseModel):
    title: str | None = None
    pinned: bool | None = None
    ask_first: bool | None = None  # ask questions before proposing; unset follows the setting
    auto_generate: bool | None = None  # render each reply's directions right away; unset follows the setting


class MessageIn(BaseModel):
    content: str
    images: list[str] = Field(default_factory=list)
    # Where the new message goes. Omit to continue the visible branch. To edit an
    # earlier message, pass that message's parent (null for the first message):
    # the edit becomes a sibling and the old branch stays.
    parent_id: int | None = None
    edit_of: int | None = None


class ActiveIn(BaseModel):
    message_id: int


class RestoreIn(BaseModel):
    stamp: float


class FeedbackIn(BaseModel):
    kind: str  # like | dislike | unrate | pick
    image_id: str | None = None
    chat_id: str | None = None
    message_id: int | None = None
    data: dict[str, Any] = Field(default_factory=dict)


def _chat_or_404(chat_id: str) -> dict:
    chat = chats.get(chat_id)
    if not chat:
        raise _not_found("Chat")
    return chat


def _chat_view(chat_id: str) -> dict:
    view = chats.view(chat_id)
    if not view:
        raise _not_found("Chat")
    linked = chats.chat_jobs(chat_id)
    view["jobs"] = [
        j.to_dict() | {"message_id": linked.get(j.id)}
        for j in jobs.list()
        if (j.chat_id == chat_id or j.id in linked) and j.kind in ("generate", "variations")
    ]
    view["spend"] = library.spend(chat_id=chat_id)
    turn = turns.running_in(chat_id)
    view["running"] = turn.message_id if turn else None
    return view


@app.get("/api/chats")
async def list_chats(q: str | None = None):
    """Designer chats with at least one message, newest activity first, with spend and a thumbnail. q searches titles and messages."""
    return chats.list_all(q)


@app.post("/api/chats")
async def create_chat(body: ChatIn):
    """An empty chat."""
    return chats.create(body.title or "")


@app.get("/api/designer/last")
async def last_chat():
    """The chat the designer had open last, from any browser."""
    return {"chat_id": chats.last_opened()}


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str):
    """A chat with the visible branch of messages (each with siblings, cost, status and the images made from it), its
    jobs, spend, and the id of a reply that's still running.
    """
    return _chat_view(chat_id)


@app.post("/api/chats/{chat_id}/open")
async def open_chat(chat_id: str):
    """Remember this as the chat the designer opens next time."""
    _chat_or_404(chat_id)
    chats.mark_opened(chat_id)
    return {"ok": True}


@app.patch("/api/chats/{chat_id}")
async def update_chat(chat_id: str, body: ChatIn):
    """Rename or pin a chat, or set its designer switches: ask questions before proposing (ask_first),
    and render each reply's directions right away (auto_generate)."""
    _chat_or_404(chat_id)
    return chats.update(chat_id, title=body.title, pinned=body.pinned, ask_first=body.ask_first,
                        auto_generate=body.auto_generate)


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat (soft: POST .../restore undoes it for a week)."""
    _chat_or_404(chat_id)
    turn = turns.running_in(chat_id)
    if turn:
        turns.stop(turn.message_id)
    chats.delete(chat_id)
    return {"ok": True}


@app.post("/api/chats/{chat_id}/restore")
async def restore_chat(chat_id: str):
    """Undo a chat delete."""
    chat = chats.restore(chat_id)
    if not chat:
        raise _not_found("Chat")
    return chat


@app.post("/api/chats/{chat_id}/messages")
async def send_message(chat_id: str, body: MessageIn):
    """Add a user message and start the designer's reply on the server. Returns user_id and reply_id; follow the
    reply with the stream endpoint or by polling the chat. edit_of=<message id> sends an edited copy of that
    message as a new branch.
    """
    chat = _chat_or_404(chat_id)
    text = body.content.strip()
    if not text:
        raise ValueError("Empty message")
    if turns.running_in(chat_id):
        raise turns.TurnBusy()
    if body.edit_of is not None:
        original = chats.get_message(body.edit_of)
        if not original or original["chat_id"] != chat_id or original["role"] != "user":
            raise ValueError("Only your own messages in this chat can be edited.")
        parent = original["parent_id"]
    elif "parent_id" in body.model_fields_set:
        parent = body.parent_id
    else:
        parent = chat["active_leaf"]
    if parent is not None:
        p = chats.get_message(parent)
        if not p or p["chat_id"] != chat_id:
            raise ValueError("That message is not in this chat.")
    images = [paths.rel_output(p) for p in _refs(body.images)]
    user_id = chats.add_message(chat_id, "user", text, parent, images=images)
    reply_id = turns.start(chat_id, user_id)
    return {"user_id": user_id, "reply_id": reply_id}


@app.post("/api/chats/{chat_id}/messages/{message_id}/regenerate")
async def regenerate(chat_id: str, message_id: int):
    """A new reply to the same user message. The old reply stays as a sibling."""
    _chat_or_404(chat_id)
    m = chats.get_message(message_id)
    if not m or m["chat_id"] != chat_id:
        raise _not_found("Message")
    parent = m["id"] if m["role"] == "user" else m["parent_id"]
    if parent is None:
        raise ValueError("Nothing to answer.")
    return {"reply_id": turns.start(chat_id, parent)}


@app.post("/api/chats/{chat_id}/messages/{message_id}/stop")
async def stop_message(chat_id: str, message_id: int):
    return {"stopped": turns.stop(message_id)}


@app.get("/api/chats/{chat_id}/messages/{message_id}/stream")
async def follow_message(chat_id: str, message_id: int, request: Request, after: int | None = None):
    """SSE for one reply. Resumes after `after` or the Last-Event-ID header; otherwise starts with a snapshot."""
    cursor = after
    last = request.headers.get("last-event-id")
    if cursor is None and last and last.isdigit():
        cursor = int(last)

    async def stream():
        yield "retry: 1000\n\n"
        async for ev in turns.follow(message_id, cursor):
            head = f"id: {ev['id']}\n" if ev.get("id") else ""
            yield head + f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.put("/api/chats/{chat_id}/active")
async def set_active(chat_id: str, body: ActiveIn):
    """Show another branch: the given message and the newest replies below it."""
    _chat_or_404(chat_id)
    chats.set_active(chat_id, body.message_id)
    return _chat_view(chat_id)


@app.delete("/api/chats/{chat_id}/messages/{message_id}")
async def delete_message(chat_id: str, message_id: int):
    """Delete a message and everything after it. Returns a stamp to undo with."""
    _chat_or_404(chat_id)
    for mid in chats.subtree(chat_id, message_id):
        turns.stop(mid)
    return {"stamp": chats.delete_from(chat_id, message_id)}


@app.post("/api/chats/{chat_id}/messages/{message_id}/restore")
async def restore_message(chat_id: str, message_id: int, body: RestoreIn):
    _chat_or_404(chat_id)
    chats.restore_from(chat_id, message_id, body.stamp)
    return _chat_view(chat_id)


# ── Feedback (read by the prompt layer) ─────────────────────────────


@app.post("/api/feedback")
async def add_feedback(body: FeedbackIn):
    """Record a reaction to an image: like, dislike, unrate (clear the rating) or pick ("use this direction").

    The designer learns from these: see GET /api/preferences.
    """
    event = chats.add_feedback(body.kind, image_id=body.image_id, chat_id=body.chat_id, message_id=body.message_id, data=body.data)
    preferences.sync_feedback()
    return event


@app.get("/api/feedback")
async def list_feedback(since_id: int = 0, chat_id: str | None = None, limit: int = 500):
    return chats.list_feedback(since_id, chat_id, min(limit, 5000))


# ── Library ─────────────────────────────────────────────────────────


@app.get("/api/images")
async def list_images(
    kind: str | None = None,
    favorite: bool = False,
    q: str | None = None,
    chat_id: str | None = None,
    job_id: str | None = None,
    ids: str | None = None,
    limit: int = 200,
    offset: int = 0,
):
    """Library images, newest first. Filter by kind (generated, upscaled, cleaned), favorite, q (text search),
    chat_id, job_id or ids (comma separated).
    """
    id_list = [i for i in ids.split(",") if i] if ids is not None else None
    return library.list_images(kind, favorite, q, chat_id, job_id, id_list, min(limit, 1000), offset)


@app.get("/api/images/{image_id}")
async def get_image(image_id: str):
    """One image with its parent and children (upscales, cleaned copies, variations)."""
    img = library.get_image(image_id)
    if not img:
        raise _not_found("Image")
    parent = library.get_image(img["parent_id"]) if img.get("parent_id") else None
    library.with_primary([img])
    return img | {"children": library.children(image_id), "parent": parent}


class ImagePatch(BaseModel):
    favorite: bool | None = None
    note: str | None = None


class PrimaryIn(BaseModel):
    child_id: str | None = None


@app.put("/api/images/{image_id}/primary")
async def set_primary(image_id: str, body: PrimaryIn):
    """Mark an upscaled or cleaned child as the version to use (child_id null clears it)."""
    if not library.set_primary(image_id, body.child_id):
        raise _not_found("Image")
    return library.with_primary([library.get_image(image_id)])[0]  # type: ignore[list-item]


@app.get("/api/spend")
async def spend(chat_id: str | None = None):
    """What has been spent: OpenRouter credits, BYOK, by kind, image counts. Optionally for one chat."""
    return library.spend(chat_id=chat_id)


@app.patch("/api/images/{image_id}")
async def patch_image(image_id: str, body: ImagePatch):
    """Set favorite or note."""
    img = library.update_image(image_id, **body.model_dump(exclude_none=True))
    if not img:
        raise _not_found("Image")
    return img


@app.delete("/api/images/{image_id}")
async def delete_image(image_id: str):
    """Delete an image, its file and its children."""
    if not library.delete_image(image_id):
        raise _not_found("Image")
    return {"ok": True}


@app.get("/api/tools")
async def tool_options():
    """Upscale methods and scales, and background removal models, available on this machine."""
    return {
        "upscale": {"methods": upscaler.METHODS, "scales": upscaler.SCALES,
                    "ai_available": system.local_stack_installed(), "device": system.device_kind()},
        "background": {"models": postprocess.MODELS, "default": postprocess.DEFAULT_MODEL},
    }


class UpscaleIn(BaseModel):
    method: str = "realesrgan-anime"
    scale: int = 4


def _raster_source(image_id: str) -> tuple[dict, Path]:
    img = library.get_image(image_id)
    if not img:
        raise _not_found("Image")
    if img["vector"]:
        raise ValueError("This is an SVG. It already scales to any size; download it instead.")
    return img, paths.resolve_output(img["path"])


def _derived_meta(img: dict) -> dict:
    keys = ("prompt", "concept", "style", "model", "seed", "chat_id", "brief", "negative_prompt", "params", "profile")
    return {k: img.get(k) for k in keys}


@app.post("/api/images/{image_id}/upscale")
async def upscale(image_id: str, body: UpscaleIn):
    """Upscale an image (methods from GET /api/tools). Returns the new image, kept as a child of the original."""
    img, src = _raster_source(image_id)
    out = await run_in_threadpool(upscaler.upscale_image, src, body.method, body.scale)
    return library.add_image(out, "upscaled", parent_id=image_id, note=f"{body.method} {body.scale}x", **_derived_meta(img))


class BackgroundIn(BaseModel):
    method: str = "ai"
    model: str = postprocess.DEFAULT_MODEL
    alpha_matting: bool = False
    erode_pixels: int = 0
    tolerance: int = 30
    color: str | None = None


@app.post("/api/images/{image_id}/remove-background")
async def remove_background(image_id: str, body: BackgroundIn):
    """Remove the background with an AI model (method ai) or by keying out a color (method color). Returns the new image as a child."""
    img, src = _raster_source(image_id)
    if body.method == "color":
        out = await run_in_threadpool(
            postprocess.remove_background_color, src, body.tolerance, postprocess.parse_hex(body.color), body.erode_pixels
        )
        note = f"color key {body.tolerance}"
    else:
        out = await run_in_threadpool(postprocess.remove_background, src, body.model, body.alpha_matting, body.erode_pixels)
        note = body.model
    return library.add_image(out, "cleaned", parent_id=image_id, note=note, **_derived_meta(img))


# ── Files ───────────────────────────────────────────────────────────


@app.post("/api/uploads")
async def upload(images: list[UploadFile] = File(...)):
    """Upload reference images. Returns their paths for the references or images fields."""
    paths.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in images:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in IMAGE_TYPES:
            continue
        dest = paths.UPLOADS_DIR / f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}{ext}"
        dest.write_bytes(await f.read())
        rel = paths.rel_output(dest)
        saved.append({"path": rel, "url": f"/api/files/{rel}", "name": f.filename})
    if not saved:
        raise ValueError("No PNG, JPEG, WebP or GIF files in the upload.")
    return saved


@app.get("/api/files/{rel:path}")
async def serve_file(rel: str, download: bool = False):
    try:
        p = paths.resolve_output(rel)
    except ValueError:
        raise _not_found("File") from None
    if not p.is_file():
        raise _not_found("File")
    headers = {"Cache-Control": "public, max-age=31536000, immutable"}
    if download:
        return FileResponse(p, filename=p.name, headers=headers)
    return FileResponse(p, headers=headers)


# ── Frontend ────────────────────────────────────────────────────────

_NOT_BUILT = """<!doctype html><meta charset="utf-8"><title>logo-gen</title>
<body style="font-family:system-ui;padding:3rem;max-width:40rem">
<h1>Frontend not built</h1><p>Run <code>./launch.sh</code> (or <code>launch.cmd</code> on Windows) from the project folder. It builds the
frontend and starts the server. For frontend work, add <code>--dev</code>.</p></body>"""


@app.get("/api/{rest:path}")
async def api_404(rest: str):
    raise _not_found(f"/api/{rest}")


@app.get("/{rest:path}")
async def spa(rest: str):
    dist = paths.FRONTEND_DIST
    index = dist / "index.html"
    if not index.exists():
        return HTMLResponse(_NOT_BUILT, status_code=503)
    if rest:
        f = (dist / rest).resolve()
        if f.is_relative_to(dist.resolve()) and f.is_file():
            cache = "public, max-age=31536000, immutable" if rest.startswith("assets/") else "no-cache"
            return FileResponse(f, headers={"Cache-Control": cache})
    return FileResponse(index, headers={"Cache-Control": "no-cache"})
