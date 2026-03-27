"""Gradio UI for interactive logo generation."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from logo_gen.config import settings
from logo_gen.generator import generate_logos, GenerationResult
from logo_gen.prompt_engine import ChatSession

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

            # === Tab 3: Settings ===
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
