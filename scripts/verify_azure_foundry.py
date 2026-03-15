#!/usr/bin/env python3
"""Verify Azure AI Foundry provider can connect, stream, and return a valid response."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kon.core.types import Message, StreamDone, StreamError, TextPart, ThinkPart, UserMessage
from kon.llm.base import ProviderConfig
from kon.llm.providers.azure_ai_foundry import AzureAIFoundryProvider

MODEL = "claude-opus-4.6"


async def verify():
    api_key = os.environ.get("AZURE_AI_FOUNDRY_API_KEY")
    base_url = os.environ.get("AZURE_AI_FOUNDRY_BASE_URL")

    if not api_key:
        print("✗ AZURE_AI_FOUNDRY_API_KEY not set")
        return False
    if not base_url:
        print("✗ AZURE_AI_FOUNDRY_BASE_URL not set")
        return False

    print(f"  Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"  URL: {base_url}")
    print(f"  Model: {MODEL}")
    print()

    config = ProviderConfig(model=MODEL, thinking_level="none")
    try:
        provider = AzureAIFoundryProvider(config)
        print("✓ Provider initialized")
    except Exception as e:
        print(f"✗ Provider init failed: {e}")
        return False

    messages: list[Message] = [UserMessage(content="Say exactly: hello from azure")]
    try:
        stream = await provider.stream(
            messages, system_prompt="You are a helpful assistant. Be very brief."
        )
        print("✓ Stream started")
    except Exception as e:
        print(f"✗ Stream failed: {e}")
        return False

    text_parts: list[str] = []
    think_parts: list[str] = []
    done = False
    error = None

    async for part in stream:
        if isinstance(part, TextPart):
            text_parts.append(part.text)
        elif isinstance(part, ThinkPart):
            think_parts.append(part.think)
        elif isinstance(part, StreamDone):
            done = True
        elif isinstance(part, StreamError):
            error = part.error

    if error:
        print(f"✗ Stream error: {error}")
        return False

    if not done:
        print("✗ Stream never completed (no StreamDone)")
        return False

    full_text = "".join(text_parts)
    if not full_text.strip():
        print("✗ Got empty response")
        return False

    print(f"✓ Response: {full_text.strip()}")

    usage = stream.usage
    if usage:
        print(f"✓ Usage: in={usage.input_tokens} out={usage.output_tokens}")
    else:
        print("⚠ No usage data returned")

    print()
    print("✓ Azure AI Foundry verification passed!")
    return True


if __name__ == "__main__":
    print("Azure AI Foundry Provider Verification")
    print("=" * 40)
    ok = asyncio.run(verify())
    sys.exit(0 if ok else 1)
