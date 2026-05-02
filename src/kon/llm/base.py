import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Literal
from urllib.parse import urlparse

from ..core.types import Message, StreamPart, ToolDefinition, Usage

DEFAULT_THINKING_LEVELS: list[str] = ["none", "minimal", "low", "medium", "high", "xhigh"]
LOCAL_API_KEY_PLACEHOLDER = "kon-local"
AuthMode = Literal["auto", "required", "none"]

ENV_API_KEY_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
    "azure-ai-foundry": "AZURE_AI_FOUNDRY_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def get_env_api_key(provider: str) -> str | None:
    env_var = ENV_API_KEY_MAP.get(provider)
    return os.environ.get(env_var) if env_var else None


def is_local_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False

    parsed = urlparse(base_url if "://" in base_url else f"https://{base_url}")
    hostname = parsed.hostname
    if hostname is None:
        return False

    normalized = hostname.lower()
    if normalized in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}:
        return True
    if normalized.endswith(".local"):
        return True

    try:
        addr = ip_address(normalized)
    except ValueError:
        return False

    return addr.is_loopback or addr.is_private or addr.is_link_local


def resolve_api_key(
    explicit_api_key: str | None,
    *,
    env_vars: list[str] | tuple[str, ...] = (),
    base_url: str | None = None,
    auth_mode: AuthMode = "required",
) -> str | None:
    if explicit_api_key:
        return explicit_api_key

    for env_var in env_vars:
        value = os.environ.get(env_var)
        if value:
            return value

    if auth_mode == "none":
        return LOCAL_API_KEY_PLACEHOLDER
    if auth_mode == "auto" and is_local_base_url(base_url):
        return LOCAL_API_KEY_PLACEHOLDER

    return None


@dataclass
class ProviderConfig:
    api_key: str | None = None
    base_url: str | None = None
    model: str = ""
    max_tokens: int = 8192
    temperature: float | None = None
    thinking_level: str = "high"
    provider: str | None = None
    session_id: str | None = None
    openai_compat_auth_mode: AuthMode = "auto"
    anthropic_compat_auth_mode: AuthMode = "auto"


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
