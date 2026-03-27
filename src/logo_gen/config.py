from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_key: str = ""

    # LLM model for prompt enhancement / chat
    llm_model: str = "anthropic/claude-sonnet-4"

    # Image generation models (via OpenRouter - must support image output)
    image_models: list[str] = [
        "openai/gpt-5-image-mini",
        "google/gemini-2.5-flash-image",
        "google/gemini-3.1-flash-image-preview",
    ]

    # Generation settings
    images_per_model: int = 2
    image_size: str = "1024x1024"

    # Output
    output_dir: str = "output/generated"
    cleaned_dir: str = "output/cleaned"


settings = Settings()
