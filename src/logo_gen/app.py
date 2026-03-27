"""Gradio UI for interactive logo generation."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from logo_gen.config import settings
from logo_gen.generator import generate_logos, GenerationResult
from logo_gen.postprocess import (
    MODELS as BG_MODELS,
    list_generated_images,
    list_generated_images_labeled,
    list_cleaned_images,
    remove_background,
    remove_background_batch,
)
from logo_gen.prompt_engine import ChatSession
from logo_gen.upscaler import (
    METHODS as UP_METHODS,
    SCALES as UP_SCALES,
    list_generated_images as list_gen_for_upscale,
    list_upscaled_images,
    upscale_image,
    upscale_batch,
)

CSS = """
.logo-gallery img { border-radius: 12px; }
.prompt-box { font-size: 0.9em; }
"""


def create_app() -> gr.Blocks:
    session = ChatSession()

    async def chat_respond(message: str, history: list):
        """Handle chat messages and stream responses."""
        history = history or []
        history.append({"role": "user", "content": message})

        partial = ""
        async for token in session.stream(message):
            partial += token
            yield history + [{"role": "assistant", "content": partial}], gr.skip(), gr.skip()

        final_history = history + [{"role": "assistant", "content": partial}]

        # Check if prompts were generated
        if session.has_prompts():
            prompts = session.get_prompts()
            prompt_display = "\n\n".join(
                f"**{i+1}. {p.get('concept', 'Concept')}** ({p.get('style', 'auto')})\n{p['prompt']}"
                for i, p in enumerate(prompts)
            )
            yield final_history, gr.update(value=prompt_display, visible=True), gr.update(visible=True)
        else:
            yield final_history, gr.skip(), gr.skip()

    async def do_generate(progress=gr.Progress()):
        """Generate logos from the current prompts."""
        if not session.has_prompts():
            gr.Warning("No prompts ready. Chat with the AI first to develop your logo concept.")
            return [], ""

        prompts = session.get_prompts()

        progress(0, desc="Starting generation...")

        def progress_cb(current, total, msg):
            progress(current / max(total, 1), desc=msg)

        result: GenerationResult = await generate_logos(
            prompts=prompts,
            models=settings.image_models,
            seeds_per_prompt=settings.images_per_model,
            progress_callback=progress_cb,
        )

        images = []
        for logo in result.logos:
            if logo.path.exists():
                label = f"{logo.concept[:40]} | {logo.model.split('/')[-1]}"
                images.append((str(logo.path), label))

        status = f"Generated {len(result.logos)} logos in {result.total_time:.1f}s"
        if result.errors:
            status += f" | {len(result.errors)} errors"

        return images, status

    async def quick_gen(concept: str, progress=gr.Progress()):
        """Quick generate from a concept description."""
        if not concept.strip():
            gr.Warning("Enter a concept description first.")
            return [], ""

        from logo_gen.prompt_engine import generate_variations

        progress(0, desc="Enhancing prompt with AI...")
        prompts = await generate_variations(concept, n=4)

        progress(0.2, desc="Generating logos...")

        def progress_cb(current, total, msg):
            progress(0.2 + 0.8 * current / max(total, 1), desc=msg)

        result = await generate_logos(
            prompts=prompts,
            models=settings.image_models,
            seeds_per_prompt=settings.images_per_model,
            progress_callback=progress_cb,
        )

        images = []
        for logo in result.logos:
            if logo.path.exists():
                label = f"{logo.concept[:40]} | {logo.model.split('/')[-1]}"
                images.append((str(logo.path), label))

        status = f"Generated {len(result.logos)} logos in {result.total_time:.1f}s"
        if result.errors:
            status += f" | {len(result.errors)} errors"

        return images, status

    def reset_chat():
        session.reset()
        return [], gr.update(value="", visible=False), gr.update(visible=False)

    def update_settings(models_text: str, images_per: int, llm_model: str):
        models = [m.strip() for m in models_text.strip().split("\n") if m.strip()]
        settings.image_models = models
        settings.images_per_model = images_per
        settings.llm_model = llm_model
        gr.Info(f"Updated: {len(models)} models, {images_per} imgs each, LLM: {llm_model}")

    # --- Build the UI ---

    with gr.Blocks(title="Logo Generator") as app:
        gr.Markdown("# Logo Generator\nDesign unique logo icons through AI-powered conversation")

        with gr.Tabs():
            # === Tab 1: Interactive Chat ===
            with gr.Tab("Chat Designer"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            height=500,
                            placeholder=(
                                "Start by describing your brand...\n\n"
                                "Example: 'I'm building a cloud computing startup called Nimbus'"
                            ),
                        )
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Describe your brand, or ask for logo ideas...",
                                show_label=False,
                                scale=5,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)

                        with gr.Row():
                            reset_btn = gr.Button("New Session", variant="secondary")
                            generate_btn = gr.Button(
                                "Generate Logos",
                                variant="primary",
                                visible=False,
                            )

                    with gr.Column(scale=2):
                        prompt_display = gr.Markdown(
                            visible=False,
                            elem_classes=["prompt-box"],
                        )

                chat_gallery = gr.Gallery(
                    label="Generated Logos",
                    columns=4,
                    height=400,
                    object_fit="contain",
                    elem_classes=["logo-gallery"],
                )
                chat_status = gr.Markdown("")

                # Wire up events
                send_btn.click(
                    chat_respond,
                    [msg_input, chatbot],
                    [chatbot, prompt_display, generate_btn],
                ).then(lambda: "", outputs=msg_input)

                msg_input.submit(
                    chat_respond,
                    [msg_input, chatbot],
                    [chatbot, prompt_display, generate_btn],
                ).then(lambda: "", outputs=msg_input)

                generate_btn.click(do_generate, outputs=[chat_gallery, chat_status])
                reset_btn.click(reset_chat, outputs=[chatbot, prompt_display, generate_btn])

            # === Tab 2: Quick Generate ===
            with gr.Tab("Quick Generate"):
                gr.Markdown("**Skip the chat** - describe your concept and generate immediately.")

                with gr.Row():
                    quick_input = gr.Textbox(
                        label="Concept",
                        placeholder="e.g., 'modern fintech startup focused on cryptocurrency'",
                        lines=3,
                        scale=4,
                    )
                    quick_btn = gr.Button("Generate", variant="primary", scale=1)

                quick_gallery = gr.Gallery(
                    label="Generated Logos",
                    columns=4,
                    height=500,
                    object_fit="contain",
                    elem_classes=["logo-gallery"],
                )
                quick_status = gr.Markdown("")

                quick_btn.click(quick_gen, [quick_input], [quick_gallery, quick_status])

            # === Tab 3: Upscale ===
            with gr.Tab("Upscale"):
                gr.Markdown(
                    "**Upscale** generated logos to higher resolution. "
                    "Source: `output/generated/` | Output: `output/upscaled/`"
                )

                with gr.Row():
                    up_method_dropdown = gr.Dropdown(
                        label="Upscale Method",
                        choices=[(desc, key) for key, desc in UP_METHODS.items()],
                        value="realesrgan-anime",
                        scale=3,
                    )
                    up_scale_dropdown = gr.Dropdown(
                        label="Scale Factor",
                        choices=[(label, val) for label, val in UP_SCALES.items()],
                        value=4,
                        scale=1,
                    )

                with gr.Row():
                    up_refresh_btn = gr.Button("Refresh", variant="secondary")
                    up_selected_btn = gr.Button("Upscale Selected", variant="primary")
                    up_all_btn = gr.Button("Upscale All", variant="primary")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Generated (click to select)")
                        up_source_gallery = gr.Gallery(
                            label="Source Images",
                            columns=4,
                            height=300,
                            object_fit="contain",
                            elem_classes=["logo-gallery"],
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("### Upscaled")
                        up_output_gallery = gr.Gallery(
                            label="Upscaled Images",
                            columns=4,
                            height=300,
                            object_fit="contain",
                            elem_classes=["logo-gallery"],
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Selected")
                        up_preview_original = gr.Image(
                            label="Original",
                            height=300,
                            type="filepath",
                            interactive=False,
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("#### Result")
                        up_preview_result = gr.Image(
                            label="Upscaled",
                            height=300,
                            type="filepath",
                            interactive=False,
                        )

                up_status = gr.Markdown("")
                up_selected_path = gr.State(value=None)

                def load_up_galleries():
                    source = [(str(p), p.name) for p in list_gen_for_upscale()]
                    upscaled = [(str(p), p.name) for p in list_upscaled_images()]
                    return source, upscaled

                def on_select_up_source(evt: gr.SelectData):
                    images = list_gen_for_upscale()
                    if evt.index < len(images):
                        path = str(images[evt.index])
                        return path, path, None, f"Selected: **{images[evt.index].name}**"
                    return None, None, None, ""

                def do_upscale_selected(
                    selected_path: str | None, method: str, scale: int
                ):
                    if not selected_path:
                        gr.Warning("Click on an image in the source gallery first.")
                        return None, load_up_galleries()[1], "No image selected"
                    path = Path(selected_path)
                    if not path.exists():
                        gr.Warning("Selected image no longer exists.")
                        return None, load_up_galleries()[1], "File not found"
                    out = upscale_image(path, method=method, scale=int(scale))
                    upscaled = [(str(p), p.name) for p in list_upscaled_images()]
                    return str(out), upscaled, f"Upscaled: **{path.name}** -> **{out.name}** ({int(scale)}x {method})"

                def do_upscale_all(method: str, scale: int, progress=gr.Progress()):
                    images = list_gen_for_upscale()
                    if not images:
                        gr.Warning("No generated images found.")
                        return [], "No images to upscale"

                    progress(0, desc="Starting batch upscale...")

                    def progress_cb(current, total, msg):
                        progress(current / max(total, 1), desc=msg)

                    results = upscale_batch(
                        images, method=method, scale=int(scale),
                        progress_callback=progress_cb,
                    )
                    upscaled = [(str(p), p.name) for p in list_upscaled_images()]
                    return upscaled, f"Upscaled **{len(results)}/{len(images)}** images ({int(scale)}x {method})"

                # Wire up events
                up_refresh_btn.click(load_up_galleries, outputs=[up_source_gallery, up_output_gallery])

                up_source_gallery.select(
                    on_select_up_source,
                    outputs=[up_selected_path, up_preview_original, up_preview_result, up_status],
                )

                up_selected_btn.click(
                    do_upscale_selected,
                    [up_selected_path, up_method_dropdown, up_scale_dropdown],
                    [up_preview_result, up_output_gallery, up_status],
                )

                up_all_btn.click(
                    do_upscale_all,
                    [up_method_dropdown, up_scale_dropdown],
                    [up_output_gallery, up_status],
                )

                app.load(load_up_galleries, outputs=[up_source_gallery, up_output_gallery])

            # === Tab 4: Background Removal ===
            with gr.Tab("Background Removal"):
                gr.Markdown(
                    "**Remove backgrounds** from generated logos. "
                    "Click an image to select it, then clean. "
                    "Source: `output/generated/` | Output: `output/cleaned/`"
                )

                with gr.Row():
                    bg_model_dropdown = gr.Dropdown(
                        label="Removal Model",
                        choices=[(desc, key) for key, desc in BG_MODELS.items()],
                        value="birefnet-general",
                        scale=3,
                    )
                    bg_alpha_matting = gr.Checkbox(
                        label="Alpha matting (smoother edges)",
                        value=False,
                        scale=1,
                    )
                    bg_erode_pixels = gr.Slider(
                        label="Edge cleanup (erode N px to remove white fringe)",
                        minimum=0,
                        maximum=20,
                        step=1,
                        value=0,
                        scale=2,
                    )

                with gr.Row():
                    bg_refresh_btn = gr.Button("Refresh", variant="secondary")
                    bg_clean_selected_btn = gr.Button("Clean Selected", variant="primary")
                    bg_clean_all_btn = gr.Button("Clean All", variant="primary")

                with gr.Row():
                    # Left: source gallery
                    with gr.Column(scale=1):
                        gr.Markdown("### Generated (click to select)")
                        bg_source_gallery = gr.Gallery(
                            label="Source Images",
                            columns=4,
                            height=300,
                            object_fit="contain",
                            elem_classes=["logo-gallery"],
                        )

                    # Right: cleaned gallery
                    with gr.Column(scale=1):
                        gr.Markdown("### Cleaned")
                        bg_cleaned_gallery = gr.Gallery(
                            label="Cleaned Images",
                            columns=4,
                            height=300,
                            object_fit="contain",
                            elem_classes=["logo-gallery"],
                        )

                # Preview row: selected original -> cleaned result
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Selected")
                        bg_preview_original = gr.Image(
                            label="Original",
                            height=300,
                            type="filepath",
                            interactive=False,
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("#### Result")
                        bg_preview_cleaned = gr.Image(
                            label="Cleaned",
                            height=300,
                            type="filepath",
                            interactive=False,
                        )

                bg_status = gr.Markdown("")
                bg_selected_path = gr.State(value=None)

                def load_galleries():
                    generated = list_generated_images_labeled()
                    cleaned = [(str(p), p.name) for p in list_cleaned_images()]
                    return generated, cleaned

                def on_select_source(evt: gr.SelectData):
                    """When user clicks an image, show it in the preview."""
                    labeled = list_generated_images_labeled()
                    if evt.index < len(labeled):
                        path = labeled[evt.index][0]
                        label = labeled[evt.index][1]
                        return path, path, None, f"Selected: **{label}**"
                    return None, None, None, ""

                def clean_selected(selected_path: str | None, model: str, alpha: bool, erode: int):
                    if not selected_path:
                        gr.Warning("Click on an image in the source gallery first.")
                        return None, load_galleries()[1], "No image selected"
                    path = Path(selected_path)
                    if not path.exists():
                        gr.Warning("Selected image no longer exists.")
                        return None, load_galleries()[1], "File not found"
                    out = remove_background(path, model_name=model, alpha_matting=alpha, erode_pixels=int(erode))
                    cleaned = [(str(p), p.name) for p in list_cleaned_images()]
                    return str(out), cleaned, f"Cleaned: **{path.name}** -> **{out.name}**"

                def clean_all(model: str, alpha: bool, erode: int, progress=gr.Progress()):
                    images = list_generated_images()
                    if not images:
                        gr.Warning("No generated images found.")
                        return [], "No images to clean"

                    progress(0, desc="Starting batch cleanup...")

                    def progress_cb(current, total, msg):
                        progress(current / max(total, 1), desc=msg)

                    results = remove_background_batch(
                        images, model_name=model, alpha_matting=alpha,
                        erode_pixels=int(erode), progress_callback=progress_cb,
                    )
                    cleaned = [(str(p), p.name) for p in list_cleaned_images()]
                    return cleaned, f"Cleaned **{len(results)}/{len(images)}** images"

                # Wire up events
                bg_refresh_btn.click(load_galleries, outputs=[bg_source_gallery, bg_cleaned_gallery])

                bg_source_gallery.select(
                    on_select_source,
                    outputs=[bg_selected_path, bg_preview_original, bg_preview_cleaned, bg_status],
                )

                bg_clean_selected_btn.click(
                    clean_selected,
                    [bg_selected_path, bg_model_dropdown, bg_alpha_matting, bg_erode_pixels],
                    [bg_preview_cleaned, bg_cleaned_gallery, bg_status],
                )

                bg_clean_all_btn.click(
                    clean_all,
                    [bg_model_dropdown, bg_alpha_matting, bg_erode_pixels],
                    [bg_cleaned_gallery, bg_status],
                )

                # Auto-load galleries on app start
                app.load(load_galleries, outputs=[bg_source_gallery, bg_cleaned_gallery])

            # === Tab 5: Settings ===
            with gr.Tab("Settings"):
                gr.Markdown("### Model Configuration")

                llm_input = gr.Textbox(
                    label="LLM Model (for prompt enhancement)",
                    value=settings.llm_model,
                )
                models_input = gr.Textbox(
                    label="Image Models (one per line)",
                    value="\n".join(settings.image_models),
                    lines=5,
                )
                images_per_input = gr.Slider(
                    label="Images per model per prompt",
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=settings.images_per_model,
                )
                save_settings_btn = gr.Button("Save Settings", variant="primary")
                save_settings_btn.click(
                    update_settings,
                    [models_input, images_per_input, llm_input],
                )

                gr.Markdown(
                    "### Available Image Models (OpenRouter)\n"
                    "- `openai/gpt-5-image` - GPT-5 Image (best quality)\n"
                    "- `openai/gpt-5-image-mini` - GPT-5 Image Mini (cheaper)\n"
                    "- `google/gemini-2.5-flash-image` - Gemini 2.5 Flash Image\n"
                    "- `google/gemini-3.1-flash-image-preview` - Gemini 3.1 Flash Image\n"
                    "- `google/gemini-3-pro-image-preview` - Gemini 3 Pro Image\n"
                    "\n### Available LLM Models\n"
                    "- `anthropic/claude-sonnet-4` - Claude Sonnet 4\n"
                    "- `openai/gpt-4o` - GPT-4o\n"
                    "- `google/gemini-2.5-flash` - Gemini 2.5 Flash\n"
                    "- `anthropic/claude-haiku-3.5` - Claude Haiku\n"
                )

    return app


def main():
    app = create_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860, css=CSS)


if __name__ == "__main__":
    main()
