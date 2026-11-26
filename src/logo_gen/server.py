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

from logo_gen import generator, library, paths, postprocess, prompt_engine, runtime, system, upscaler
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

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    paths.ensure_dirs()
    library.conn()
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
    return {"ok": True}


@app.get("/api/system")
async def system_info():
    return system.info() | {"library": library.stats()}


def _secrets_status() -> dict:
    return {
        name: {"set": bool(get_secret(name)), "source": secret_source(name), "masked": mask(get_secret(name))}
        for name in SECRET_KEYS
    }


@app.get("/api/settings")
async def read_settings():
    return {"settings": get_settings().model_dump(), "secrets": _secrets_status()}


@app.put("/api/settings")
async def write_settings(patch: dict[str, Any]):
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
    return await openrouter.key_status()


@app.get("/api/openrouter/models")
async def openrouter_models(force: bool = False):
    data = await openrouter.catalog(force=force)
    return {"fetched_at": data["fetched_at"], "chat": data["chat"], "image": data["image"], "stale": data.get("stale", False)}


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
    return manager.start_install(_spec(model_id))


@app.post("/api/local-models/{model_id}/cancel")
async def cancel_local_model(model_id: str):
    return await manager.cancel_install(_spec(model_id))


@app.delete("/api/local-models/{model_id}")
async def uninstall_local_model(model_id: str):
    return await manager.uninstall(_spec(model_id))


@app.post("/api/local-models/{model_id}/check-access")
async def check_local_model(model_id: str):
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
    return [j.to_dict() for j in jobs.list()]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise _not_found("Job")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = jobs.cancel(job_id)
    if not job:
        raise _not_found("Job")
    return job.to_dict()


# ── Generation ──────────────────────────────────────────────────────


class PromptIn(BaseModel):
    prompt: str
    concept: str = ""
    style: str = ""


class GenerateIn(BaseModel):
    prompts: list[PromptIn]
    models: list[str] | None = None
    images_per_prompt: int | None = None
    references: list[str] = Field(default_factory=list)
    chat_id: str | None = None


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
    prompts = [p.model_dump() for p in body.prompts if p.prompt.strip()]
    if not prompts:
        raise ValueError("No prompts to generate.")
    models = _check_models(body.models)
    refs = _refs(body.references)
    title = prompts[0]["concept"] or prompts[0]["prompt"][:40]

    async def fn(job):
        job.result["prompts"] = prompts
        await generator.run(job, prompts, models, body.images_per_prompt, refs, chat_id=body.chat_id)

    return jobs.start("generate", title, fn).to_dict()


@app.post("/api/generate/quick")
async def quick_generate(body: QuickIn):
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
    img = library.get_image(image_id)
    if not img:
        raise _not_found("Image")
    source = {"prompt": img["prompt"], "concept": img["concept"], "style": img["style"]}
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
            prompts, per = [source], count
        else:
            job.progress(message="Writing variations")
            job.chat_id = img.get("chat_id")
            prompts, cost = await prompt_engine.variations(source, count, body.instruction)
            job.add_cost("prompts", cost)
            per = 1
        job.result["prompts"] = prompts
        job.result["source_id"] = image_id
        await generator.run(job, prompts, models, per, refs, chat_id=img.get("chat_id"), parent_id=image_id)

    return jobs.start("variations", f"Variations of {img['concept'] or img['model']}", fn).to_dict()


# ── Chats ───────────────────────────────────────────────────────────


class ChatIn(BaseModel):
    title: str = ""


class MessageIn(BaseModel):
    content: str
    images: list[str] = Field(default_factory=list)


@app.get("/api/chats")
async def list_chats():
    return library.list_chats()


@app.post("/api/chats")
async def create_chat(body: ChatIn):
    return library.create_chat(body.title)


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str):
    chat = library.get_chat(chat_id)
    if not chat:
        raise _not_found("Chat")
    return chat | {"spend": library.spend(chat_id=chat_id)}


@app.patch("/api/chats/{chat_id}")
async def rename_chat(chat_id: str, body: ChatIn):
    library.rename_chat(chat_id, body.title)
    return library.get_chat(chat_id)


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    library.delete_chat(chat_id)
    return {"ok": True}


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.post("/api/chats/{chat_id}/messages")
async def send_message(chat_id: str, body: MessageIn):
    chat = library.get_chat(chat_id)
    if not chat:
        raise _not_found("Chat")
    text = body.content.strip()
    if not text:
        raise ValueError("Empty message")
    images = [paths.rel_output(p) for p in _refs(body.images)]
    library.add_message(chat_id, "user", text, images=images)
    history = library.get_chat(chat_id)["messages"]  # type: ignore[index]
    vision = await prompt_engine.chat_supports_images()
    messages = prompt_engine.build_history(history, paths.resolve_output, vision)

    async def stream():
        reply = ""
        saved = False
        try:
            yield _sse({"type": "start", "model": prompt_engine.backend_label()})
            async for ev in prompt_engine.stream_reply(messages):
                if ev["type"] == "token":
                    reply += ev["content"]
                    yield _sse({"type": "token", "content": ev["content"]})
                elif ev["type"] == "reasoning":
                    yield _sse({"type": "reasoning", "content": ev.get("content", "")})
                elif ev["type"] == "usage":
                    rec = prompt_engine.cost_record(ev, get_settings().chat_model)
                    library.add_cost("chat", chat_id=chat_id, **rec)
                    yield _sse({"type": "usage", "credits": rec["credits"], "byok": rec["byok"]})
            prompts = prompt_engine.parse_prompts(reply)
            shown = prompt_engine.strip_prompt_block(reply) if prompts else reply
            mid = library.add_message(chat_id, "assistant", shown, prompts=prompts)
            saved = True
            yield _sse({"type": "done", "message_id": mid, "content": shown, "prompts": prompts})
        except Exception as e:  # noqa: BLE001
            yield _sse({"type": "error", "message": _error_text(e)})
        finally:
            if not saved and reply.strip():
                library.add_message(chat_id, "assistant", reply + "\n\n_(stopped)_")

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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
    id_list = [i for i in ids.split(",") if i] if ids is not None else None
    return library.list_images(kind, favorite, q, chat_id, job_id, id_list, min(limit, 1000), offset)


@app.get("/api/images/{image_id}")
async def get_image(image_id: str):
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
    if not library.set_primary(image_id, body.child_id):
        raise _not_found("Image")
    return library.with_primary([library.get_image(image_id)])[0]  # type: ignore[list-item]


@app.get("/api/spend")
async def spend(chat_id: str | None = None):
    return library.spend(chat_id=chat_id)


@app.patch("/api/images/{image_id}")
async def patch_image(image_id: str, body: ImagePatch):
    img = library.update_image(image_id, **body.model_dump(exclude_none=True))
    if not img:
        raise _not_found("Image")
    return img


@app.delete("/api/images/{image_id}")
async def delete_image(image_id: str):
    if not library.delete_image(image_id):
        raise _not_found("Image")
    return {"ok": True}


@app.get("/api/tools")
async def tool_options():
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
    return {k: img.get(k) for k in ("prompt", "concept", "style", "model", "seed", "chat_id")}


@app.post("/api/images/{image_id}/upscale")
async def upscale(image_id: str, body: UpscaleIn):
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
<h1>Frontend not built</h1><p>Run <code>./logo-gen</code> from the project folder. It builds the
frontend and starts the server. For frontend work, use <code>./logo-gen --dev</code>.</p></body>"""


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
