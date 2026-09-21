"""Test local LLM prompt generation via Ollama.

Requires Ollama running locally: https://ollama.com
Pull a model first: ollama pull qwen3.5:9b

Usage:
    python scripts/test_local_llm.py
    python scripts/test_local_llm.py --model gemma3:12b
    python scripts/test_local_llm.py --concept "fintech startup for crypto trading"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


async def test_availability(model: str):
    """Check if Ollama is running and the model is available."""
    from logo_gen.clients import local_llm

    print(f"Checking Ollama at {local_llm.DEFAULT_BASE_URL}...")
    available = await local_llm.is_available()
    if not available:
        print("ERROR: Ollama is not running.")
        print("Install from https://ollama.com and start it.")
        return False

    print("Ollama is running.")
    models = await local_llm.list_models()
    print(f"Available models: {models}")

    if model not in models and f"{model}:latest" not in models:
        print(f"\nModel '{model}' not found. Pull it with:")
        print(f"  ollama pull {model}")
        return False

    print(f"Model '{model}' is available.")
    return True


async def test_enhance_prompt(model: str, concept: str):
    """Test single prompt enhancement."""
    from logo_gen.clients import local_llm

    print(f"\n--- Test: enhance_prompt (model: {model}) ---")
    print(f"Concept: {concept}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert at writing prompts for AI image generation models. "
                "Given a logo concept description, create a highly detailed, optimized prompt "
                "that will produce a stunning logo design. "
                "Rules: no text/letters/words in the logo. Include style, colors, composition. "
                "Add quality boosters. Specify centered on solid white background. "
                "Return ONLY the prompt text, nothing else."
            ),
        },
        {"role": "user", "content": f"Create a logo prompt for: {concept}"},
    ]

    start = time.time()
    response = await local_llm.chat(messages, model=model, temperature=0.8)
    elapsed = time.time() - start

    print(f"\nResult ({elapsed:.1f}s):")
    print(response)
    return response


async def test_generate_variations(model: str, concept: str):
    """Test generating multiple prompt variations (same as the app does)."""
    from logo_gen.clients import local_llm

    print(f"\n--- Test: generate_variations (model: {model}) ---")
    print(f"Concept: {concept}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert logo designer and AI image prompt engineer. "
                "Generate diverse logo prompt variations as JSON.\n\n"
                "Rules:\n"
                "- Logos must NEVER contain text, letters, or typography\n"
                "- Focus on symbols, abstract marks, geometric shapes\n"
                "- Each prompt should explore a different visual direction\n\n"
                'Return a JSON code block with format: ```json\n{"prompts": [{"prompt": "...", "concept": "...", "style": "..."}]}\n```'
            ),
        },
        {
            "role": "user",
            "content": (
                f"I want to generate logos for this concept: {concept}\n\n"
                "Generate exactly 4 diverse prompt variations. "
                "Each should explore a completely different visual direction. "
                "Return them in the JSON format specified."
            ),
        },
    ]

    start = time.time()
    response = await local_llm.chat(messages, model=model, temperature=0.9)
    elapsed = time.time() - start

    print(f"\nRaw response ({elapsed:.1f}s):")
    print(response[:500] + "..." if len(response) > 500 else response)

    # Try to parse JSON
    import json

    try:
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()

        data = json.loads(json_str)
        if isinstance(data, dict) and "prompts" in data:
            prompts = data["prompts"]
        elif isinstance(data, list):
            prompts = data
        else:
            prompts = []

        print(f"\nParsed {len(prompts)} prompts:")
        for i, p in enumerate(prompts):
            print(f"  {i+1}. [{p.get('style', '?')}] {p.get('concept', '?')}")
            print(f"     {p.get('prompt', '')[:100]}...")
        return prompts
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        print(f"\nFailed to parse JSON: {e}")
        return []


async def test_streaming(model: str, concept: str):
    """Test streaming response (same as chat designer uses)."""
    from logo_gen.clients import local_llm

    print(f"\n--- Test: streaming (model: {model}) ---")

    messages = [
        {
            "role": "system",
            "content": "You are a logo design expert. Keep responses concise.",
        },
        {
            "role": "user",
            "content": f"Suggest 3 visual directions for a logo for: {concept}. Be brief.",
        },
    ]

    print("Streaming: ", end="", flush=True)
    start = time.time()
    token_count = 0
    async for token in local_llm.chat_stream(messages, model=model, temperature=0.7):
        print(token, end="", flush=True)
        token_count += 1
    elapsed = time.time() - start

    print(f"\n\n({token_count} tokens in {elapsed:.1f}s, {token_count/elapsed:.0f} tok/s)")


async def test_integrated_pipeline(model: str, concept: str):
    """Test through the actual prompt_engine (end-to-end integration)."""
    from logo_gen.config import settings

    # Temporarily set the LLM model to use local
    original_model = settings.llm_model
    settings.llm_model = f"local/{model}"

    print(f"\n--- Test: integrated pipeline (llm_model={settings.llm_model}) ---")

    from logo_gen.prompt_engine import generate_variations

    start = time.time()
    prompts = await generate_variations(concept, n=4)
    elapsed = time.time() - start

    print(f"\nGenerated {len(prompts)} prompts in {elapsed:.1f}s:")
    for i, p in enumerate(prompts):
        print(f"  {i+1}. [{p.get('style', '?')}] {p.get('concept', '?')}")
        print(f"     {p.get('prompt', '')[:120]}...")

    settings.llm_model = original_model
    return prompts


async def main():
    parser = argparse.ArgumentParser(description="Test local LLM for logo prompt generation")
    parser.add_argument("--model", type=str, default="qwen3.5:9b",
                        help="Ollama model name (default: qwen3.5:9b)")
    parser.add_argument("--concept", type=str,
                        default="modern cloud computing startup called Nimbus",
                        help="Logo concept to test with")
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "availability", "enhance", "variations", "streaming", "pipeline"],
                        help="Which test to run")
    args = parser.parse_args()

    if args.test in ("all", "availability"):
        ok = await test_availability(args.model)
        if not ok:
            return

    if args.test in ("all", "enhance"):
        await test_enhance_prompt(args.model, args.concept)

    if args.test in ("all", "variations"):
        await test_generate_variations(args.model, args.concept)

    if args.test in ("all", "streaming"):
        await test_streaming(args.model, args.concept)

    if args.test in ("all", "pipeline"):
        await test_integrated_pipeline(args.model, args.concept)

    print("\n--- Done ---")


if __name__ == "__main__":
    asyncio.run(main())
