"""Flask API backend for logo-gen React SPA."""

import asyncio
import json
import queue
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from logo_gen.config import Settings, settings
from logo_gen.generator import generate_logos
from logo_gen.prompt_engine import ChatSession, generate_variations
from logo_gen.postprocess import (
    list_generated_images as list_all_sources,
    list_cleaned_images,
    remove_background,
    remove_background_batch,
    remove_background_color,
    remove_background_color_batch,
    MODELS as BG_MODELS,
    DEFAULT_MODEL as BG_DEFAULT_MODEL,
)
from logo_gen.upscaler import (
    list_generated_images as list_generated_only,
    list_upscaled_images,
    upscale_image,
    upscale_batch,
    METHODS as UP_METHODS,
    SCALES as UP_SCALES,
)


OUTPUT_DIR = Path.cwd() / "output"
UPLOAD_DIR = OUTPUT_DIR / "uploads"


def _norm(p: Path) -> str:
    """Normalize path to forward-slash relative string under output/."""
    s = str(p).replace("\\", "/")
    idx = s.find("output/")
    return s[idx + 7:] if idx >= 0 else s


def _sse(generator_fn):
    """Wrap a generator function as an SSE Response."""
    q: queue.Queue[str | None] = queue.Queue()

    def run():
        try:
            generator_fn(q)
        except Exception as e:
            q.put(json.dumps({"type": "error", "message": str(e)}))
        finally:
            q.put(None)

    threading.Thread(target=run, daemon=True).start()

    def stream():
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {item}\n\n"

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _logo_to_dict(logo) -> dict:
    return {
        "path": _norm(logo.path),
        "prompt": logo.prompt,
        "concept": logo.concept,
        "model": logo.model,
        "seed": logo.seed,
        "generation_time": logo.generation_time,
    }


def create_api() -> Flask:
    app = Flask(__name__, static_folder=None)
    session = ChatSession()

    # ── Upload ─────────────────────────────────────────────────────────

    @app.post("/api/upload")
    def upload_images():
        """Accept image uploads, save to output/uploads/, return paths."""
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        files = request.files.getlist("images")
        if not files:
            return jsonify({"error": "No files uploaded"}), 400

        saved = []
        for f in files:
            if not f.filename:
                continue
            ext = Path(f.filename).suffix.lower()
            if ext not in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
                continue
            name = f"{int(time.time() * 1000)}_{f.filename}"
            dest = UPLOAD_DIR / name
            f.save(dest)
            saved.append({"path": _norm(dest), "name": name})
        return jsonify(saved)

    def _resolve_images(image_paths: list[str] | None) -> list[Path] | None:
        """Resolve relative image paths to absolute paths under output/."""
        if not image_paths:
            return None
        return [OUTPUT_DIR / p for p in image_paths if p]

    # ── Chat ──────────────────────────────────────────────────────────

    @app.post("/api/chat/send")
    def chat_send():
        data = request.get_json() or {}
        message = (data.get("message") or "").strip()
        images = _resolve_images(data.get("images"))
        history = data.get("history") or []
        if not message:
            return jsonify({"error": "Empty message"}), 400

        # If the frontend sent prior turns, build a fresh per-request session
        # seeded with that history. This makes chat survive server restarts —
        # the frontend's IndexedDB is the source of truth.
        if history:
            sess = ChatSession()
            seed = []
            for h in history:
                role = h.get("role")
                if role not in ("user", "assistant"):
                    continue
                content = h.get("content") or ""
                raw_imgs = h.get("images") or []
                resolved = [str(OUTPUT_DIR / p) for p in raw_imgs if p]
                seed.append({"role": role, "content": content, "images": resolved})
            sess.seed_history(seed)
        else:
            sess = session

        def gen(q):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def _stream():
                    async for token in sess.stream(message, images=images):
                        q.put(json.dumps({"type": "token", "content": token}))
                    result = {"type": "done", "has_prompts": sess.has_prompts()}
                    if sess.has_prompts():
                        result["prompts"] = sess.get_prompts()
                    q.put(json.dumps(result))

                loop.run_until_complete(_stream())
            finally:
                loop.close()

        return _sse(gen)

    @app.post("/api/chat/reset")
    def chat_reset():
        session.reset()
        return jsonify({"ok": True})

    @app.get("/api/chat/prompts")
    def chat_prompts():
        return jsonify({
            "has_prompts": session.has_prompts(),
            "prompts": session.get_prompts() if session.has_prompts() else [],
        })

    # ── Generation ────────────────────────────────────────────────────

    def _run_generation(prompts, q, reference_images=None, images_per_model=None):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def _gen():
                def cb(cur, tot, msg):
                    q.put(json.dumps({"type": "progress", "current": cur, "total": tot, "message": msg}))

                result = await generate_logos(
                    prompts,
                    models=settings.image_models,
                    seeds_per_prompt=images_per_model or settings.images_per_model,
                    progress_callback=cb,
                    reference_images=reference_images,
                )
                q.put(json.dumps({
                    "type": "result",
                    "logos": [_logo_to_dict(l) for l in result.logos],
                    "errors": result.errors,
                    "total_time": result.total_time,
                }))

            loop.run_until_complete(_gen())
        finally:
            loop.close()

    @app.post("/api/generate")
    def generate():
        data = request.get_json() or {}
        # Prefer explicit prompts from the request body (per-prompt selection).
        # Fall back to whatever the chat session last parsed.
        requested = data.get("prompts")
        if isinstance(requested, list) and requested:
            prompts = requested
        elif session.has_prompts():
            prompts = session.get_prompts()
        else:
            return jsonify({"error": "No prompts available. Chat first."}), 400

        ref_images = _resolve_images(data.get("images")) or session.get_reference_images() or None

        raw_per_model = data.get("images_per_model")
        try:
            images_per_model = int(raw_per_model) if raw_per_model is not None else None
        except (TypeError, ValueError):
            images_per_model = None
        if images_per_model is not None:
            images_per_model = max(1, min(100, images_per_model))

        return _sse(lambda q: _run_generation(
            prompts, q,
            reference_images=ref_images,
            images_per_model=images_per_model,
        ))

    @app.post("/api/quick-generate")
    def quick_generate():
        data = request.get_json()
        concept = (data or {}).get("concept", "").strip()
        images = _resolve_images((data or {}).get("images"))
        if not concept:
            return jsonify({"error": "Empty concept"}), 400

        def gen(q):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def _gen():
                    q.put(json.dumps({"type": "progress", "message": "Generating prompt variations..."}))
                    variations = await generate_variations(concept, n=4, images=images)
                    q.put(json.dumps({"type": "prompts", "prompts": variations}))

                    def cb(cur, tot, msg):
                        q.put(json.dumps({"type": "progress", "current": cur, "total": tot, "message": msg}))

                    result = await generate_logos(
                        variations,
                        models=settings.image_models,
                        seeds_per_prompt=settings.images_per_model,
                        progress_callback=cb,
                        reference_images=images,
                    )
                    q.put(json.dumps({
                        "type": "result",
                        "logos": [_logo_to_dict(l) for l in result.logos],
                        "errors": result.errors,
                        "total_time": result.total_time,
                    }))

                loop.run_until_complete(_gen())
            finally:
                loop.close()

        return _sse(gen)

    # ── Images ────────────────────────────────────────────────────────

    @app.get("/api/images/generated")
    def images_generated():
        return jsonify([{"path": _norm(p), "name": p.name} for p in list_generated_only()])

    @app.get("/api/images/upscaled")
    def images_upscaled():
        return jsonify([{"path": _norm(p), "name": p.name} for p in list_upscaled_images()])

    @app.get("/api/images/cleaned")
    def images_cleaned():
        return jsonify([{"path": _norm(p), "name": p.name} for p in list_cleaned_images()])

    @app.get("/api/images/all-sources")
    def images_all_sources():
        return jsonify([{"path": _norm(p), "name": p.name} for p in list_all_sources()])

    @app.get("/api/output/<path:filepath>")
    def serve_output(filepath):
        return send_from_directory(str(OUTPUT_DIR), filepath)

    # ── Upscale ───────────────────────────────────────────────────────

    @app.get("/api/upscale/options")
    def upscale_options():
        return jsonify({"methods": UP_METHODS, "scales": UP_SCALES})

    @app.post("/api/upscale")
    def upscale():
        data = request.get_json()
        image_path = (data or {}).get("path", "")
        method = data.get("method", "lanczos")
        scale = data.get("scale", 4)
        if not image_path:
            return jsonify({"error": "No image path"}), 400
        full_path = OUTPUT_DIR / image_path
        try:
            result = upscale_image(full_path, method=method, scale=int(scale))
            return jsonify({"path": _norm(result), "name": result.name})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/api/upscale/batch")
    def upscale_all():
        data = request.get_json() or {}
        method = data.get("method", "lanczos")
        scale = data.get("scale", 4)
        images = list_generated_only()
        if not images:
            return jsonify({"error": "No images to upscale"}), 400

        def gen(q):
            def cb(cur, tot, msg):
                q.put(json.dumps({"type": "progress", "current": cur, "total": tot, "message": msg}))
            results = upscale_batch(images, method=method, scale=int(scale), progress_callback=cb)
            q.put(json.dumps({
                "type": "result",
                "images": [{"path": _norm(p), "name": p.name} for p in results],
            }))

        return _sse(gen)

    # ── Background removal ────────────────────────────────────────────

    @app.get("/api/bg-remove/options")
    def bg_options():
        return jsonify({"models": BG_MODELS, "default": BG_DEFAULT_MODEL})

    def _parse_hex_color(hex_str: str) -> tuple[int, int, int] | None:
        """Parse '#rrggbb' or 'rrggbb' into an RGB tuple. Returns None on failure."""
        if not hex_str:
            return None
        s = hex_str.lstrip("#")
        if len(s) != 6:
            return None
        try:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
        except ValueError:
            return None

    @app.post("/api/bg-remove")
    def bg_remove():
        data = request.get_json() or {}
        image_path = data.get("path", "")
        method = data.get("method", "ai")
        erode_pixels = int(data.get("erode_pixels", 0))
        if not image_path:
            return jsonify({"error": "No image path"}), 400
        full_path = OUTPUT_DIR / image_path
        try:
            if method == "color":
                tolerance = int(data.get("tolerance", 30))
                auto = bool(data.get("auto_corners", True))
                target = None if auto else _parse_hex_color(data.get("color", ""))
                result = remove_background_color(
                    full_path, tolerance=tolerance, target_color=target,
                    erode_pixels=erode_pixels,
                )
            else:
                model = data.get("model", BG_DEFAULT_MODEL)
                alpha_matting = data.get("alpha_matting", False)
                result = remove_background(
                    full_path, model_name=model,
                    alpha_matting=alpha_matting, erode_pixels=erode_pixels,
                )
            return jsonify({"path": _norm(result), "name": result.name})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/api/bg-remove/batch")
    def bg_remove_all():
        data = request.get_json() or {}
        method = data.get("method", "ai")
        erode_pixels = int(data.get("erode_pixels", 0))
        images = list_all_sources()
        if not images:
            return jsonify({"error": "No images to process"}), 400

        def gen(q):
            def cb(cur, tot, msg):
                q.put(json.dumps({"type": "progress", "current": cur, "total": tot, "message": msg}))
            if method == "color":
                tolerance = int(data.get("tolerance", 30))
                auto = bool(data.get("auto_corners", True))
                target = None if auto else _parse_hex_color(data.get("color", ""))
                results = remove_background_color_batch(
                    images, tolerance=tolerance, target_color=target,
                    erode_pixels=erode_pixels, progress_callback=cb,
                )
            else:
                model = data.get("model", BG_DEFAULT_MODEL)
                alpha_matting = data.get("alpha_matting", False)
                results = remove_background_batch(
                    images, model_name=model, alpha_matting=alpha_matting,
                    erode_pixels=erode_pixels, progress_callback=cb,
                )
            q.put(json.dumps({
                "type": "result",
                "images": [{"path": _norm(p), "name": p.name} for p in results],
            }))

        return _sse(gen)

    # ── Settings ──────────────────────────────────────────────────────

    @app.get("/api/settings")
    def get_settings():
        return jsonify({
            "llm_model": settings.llm_model,
            "image_models": settings.image_models,
            "images_per_model": settings.images_per_model,
            "image_size": settings.image_size,
            "use_claude_cli": settings.use_claude_cli,
        })

    @app.get("/api/settings/defaults")
    def get_default_settings():
        defaults = Settings()
        return jsonify({
            "llm_model": defaults.llm_model,
            "image_models": defaults.image_models,
            "images_per_model": defaults.images_per_model,
            "image_size": defaults.image_size,
            "use_claude_cli": defaults.use_claude_cli,
        })

    @app.post("/api/settings")
    def update_settings():
        data = request.get_json() or {}
        if "llm_model" in data:
            settings.llm_model = data["llm_model"]
        if "image_models" in data:
            settings.image_models = data["image_models"]
        if "images_per_model" in data:
            settings.images_per_model = int(data["images_per_model"])
        if "use_claude_cli" in data:
            settings.use_claude_cli = bool(data["use_claude_cli"])
        return jsonify({"ok": True})

    @app.get("/api/claude-cli/status")
    def claude_cli_status():
        """Check whether the claude CLI is available locally."""
        from logo_gen.clients import claude_cli as _cli

        loop = asyncio.new_event_loop()
        try:
            available = loop.run_until_complete(_cli.is_available())
        finally:
            loop.close()
        return jsonify({"available": available})

    # ── SPA fallback ──────────────────────────────────────────────────

    @app.get("/", defaults={"path": ""})
    @app.get("/<path:path>")
    def serve_spa(path):
        dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
        fpath = dist / path
        if path and fpath.exists() and fpath.is_file():
            return send_from_directory(str(dist), path)
        return send_from_directory(str(dist), "index.html")

    return app


def main():
    app = create_api()
    app.run(host="0.0.0.0", port=7860, debug=True, threaded=True)


if __name__ == "__main__":
    main()
