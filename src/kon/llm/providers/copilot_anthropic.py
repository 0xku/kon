"""
GitHub Copilot Anthropic provider.

Uses Anthropic Messages API format with Copilot OAuth authentication.
This enables extended thinking for Claude models via Copilot.
"""

from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import ThinkingConfigEnabledParam

from ...core.types import Message, ToolDefinition
from ..base import BaseProvider, LLMStream, ProviderConfig
from ..oauth import COPILOT_HEADERS, get_base_url_from_token, get_valid_token, load_credentials
from .anthropic import AnthropicProvider, _adjust_max_tokens_for_thinking
from .anthropic_capabilities import lookup_capabilities
from .github_copilot_headers import build_copilot_dynamic_headers


class CopilotAnthropicProvider(AnthropicProvider):
    """
    GitHub Copilot provider for Claude using Anthropic Messages API.

    Uses Copilot OAuth token and /v1/messages endpoint to access
    Claude models with full extended thinking support.
    """

    name = "github-copilot-anthropic"
    thinking_levels: list[str] = ["none", "minimal", "low", "medium", "high", "xhigh"]  # noqa: RUF012

    def __init__(self, config: ProviderConfig):
        # Skip AnthropicProvider.__init__ since we need custom client setup
        BaseProvider.__init__(self, config)
        self._client: AsyncAnthropic | None = None
        self._current_token: str | None = None

    async def _ensure_client(self, messages: list[Message]) -> AsyncAnthropic:
        token = await get_valid_token()
        if not token:
            raise RuntimeError("Not logged in to GitHub Copilot. Use /login to authenticate.")

        self._current_token = token
        creds = load_credentials()
        base_url = get_base_url_from_token(token, creds.enterprise_domain if creds else None)
        dynamic_headers = build_copilot_dynamic_headers(messages)
        self._client = AsyncAnthropic(
            api_key=token,
            base_url=base_url,
            default_headers={
                **COPILOT_HEADERS,
                **dynamic_headers,
                "Authorization": f"Bearer {token}",
                "anthropic-beta": "interleaved-thinking-2025-05-14",
            },
        )

        return self._client

    async def stream(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMStream:
        self._client = await self._ensure_client(messages)

        anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools) if tools else None

        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        create_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "max_tokens": max_tok,
        }

        if system_prompt:
            create_kwargs["system"] = [
                {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
            ]

        temp = temperature if temperature is not None else self.config.temperature
        thinking_level = self.config.thinking_level
        caps = lookup_capabilities(self.config.model)
        thinking_budget = caps.thinking_budgets.get(thinking_level, 0)
        thinking_enabled = thinking_level != "none" and (
            thinking_budget > 0 or caps.adaptive_thinking
        )

        if thinking_enabled:
            if caps.adaptive_thinking:
                create_kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
                effort = caps.effort_map.get(thinking_level, "high")
                create_kwargs["output_config"] = {"effort": effort}
            else:
                adjusted_max, adjusted_budget = _adjust_max_tokens_for_thinking(
                    max_tok, thinking_budget
                )
                create_kwargs["max_tokens"] = adjusted_max
                create_kwargs["thinking"] = ThinkingConfigEnabledParam(
                    type="enabled", budget_tokens=adjusted_budget
                )
                # NB: the interleaved-thinking beta header is already set on
                # the client for all Copilot Anthropic traffic.
        else:
            if caps.adaptive_thinking:
                create_kwargs["thinking"] = {"type": "disabled"}
            if temp is not None:
                create_kwargs["temperature"] = temp

        if anthropic_tools:
            create_kwargs["tools"] = anthropic_tools

        response = await self._client.messages.stream(**create_kwargs).__aenter__()
        llm_stream = LLMStream()
        llm_stream.set_iterator(self._process_stream(response, llm_stream))
        return llm_stream
