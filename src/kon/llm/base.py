import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ..core.types import Message, StreamPart, ToolDefinition, Usage

DEFAULT_THINKING_LEVELS: list[str] = ["none", "low", "medium", "high"]

ENV_API_KEY_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
    "azure-ai-foundry": "AZURE_AI_FOUNDRY_API_KEY",
}


def get_env_api_key(provider: str) -> str | None:
    env_var = ENV_API_KEY_MAP.get(provider)
    return os.environ.get(env_var) if env_var else None


@dataclass
class ProviderConfig:
    api_key: str | None = None
    base_url: str | None = None
    model: str = ""
    max_tokens: int = 8192
    temperature: float | None = None
    thinking_level: str = "medium"
    provider: str | None = None
    session_id: str | None = None


class LLMStream(AsyncIterator["StreamPart"]):
    """
    Async iterator over stream parts with access to final usage/metadata.

    Usage:
        stream = await provider.stream(messages, tools)
        async for part in stream:
            match part:
                case TextPart(text=t):
                    print(t, end="")
                case ThinkPart(think=t):
                    print(f"[thinking] {t}")
                case ToolCallStart(id=id, name=name):
                    print(f"Tool call: {name}")
                ...

        # After iteration, access final stats
        print(f"Usage: {stream.usage}")
    """

    def __init__(self) -> None:
        self._iterator: AsyncIterator[StreamPart] | None = None
        self._usage: Usage | None = None
        self._id: str | None = None

    def set_iterator(self, iterator: AsyncIterator[StreamPart]) -> None:
        self._iterator = iterator

    def __aiter__(self) -> AsyncIterator[StreamPart]:
        return self

    async def __anext__(self) -> StreamPart:
        if self._iterator is None:
            raise StopAsyncIteration
        return await self._iterator.__anext__()

    @property
    def usage(self) -> Usage | None:
        return self._usage

    @property
    def id(self) -> str | None:
        return self._id


class BaseProvider(ABC):
    name: str
    thinking_levels: list[str] = DEFAULT_THINKING_LEVELS

    def __init__(self, config: ProviderConfig):
        self.config = config

    @property
    def thinking_level(self) -> str:
        return self.config.thinking_level

    def set_thinking_level(self, level: str) -> None:
        if level not in self.thinking_levels:
            raise ValueError(
                f"Invalid thinking level '{level}' for {self.name}. "
                f"Valid levels: {self.thinking_levels}"
            )
        self.config.thinking_level = level

    def cycle_thinking_level(self) -> str:
        levels = self.thinking_levels
        current_idx = (
            levels.index(self.config.thinking_level) if self.config.thinking_level in levels else 0
        )
        next_idx = (current_idx + 1) % len(levels)
        new_level = levels[next_idx]
        self.config.thinking_level = new_level
        return new_level

    async def stream(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMStream:
        return await self._stream_impl(
            messages,
            system_prompt=system_prompt,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @abstractmethod
    async def _stream_impl(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMStream: ...

    @abstractmethod
    def should_retry_for_error(self, error: Exception) -> bool: ...
