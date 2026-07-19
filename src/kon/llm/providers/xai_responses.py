from typing import Any

from openai import AsyncOpenAI

from kon import config as kon_config

from ...core.types import Message, ToolDefinition
from ..base import LLMStream, make_http_client
from ..oauth.xai import get_valid_xai_credentials
from .openai_responses import OpenAIResponsesProvider


class XaiResponsesProvider(OpenAIResponsesProvider):
    name = "xai"
    thinking_levels: list[str] = ["low", "medium", "high"]  # noqa: RUF012

    async def _stream_impl(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMStream:
        credentials = await get_valid_xai_credentials()
        if not credentials:
            raise RuntimeError(
                "Not logged in to xAI. Use /login and choose xAI (Grok/X subscription)."
            )

        headers = {}
        if self.config.session_id:
            headers["session_id"] = self.config.session_id
        self._client = AsyncOpenAI(
            api_key=credentials.access,
            base_url=self.config.base_url,
            default_headers=headers,
            timeout=kon_config.llm.request_timeout_seconds,
            http_client=make_http_client(),
        )
        return await super()._stream_impl(
            messages,
            system_prompt=system_prompt,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _build_params(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        params = super()._build_params(*args, **kwargs)
        reasoning = params.get("reasoning")
        if isinstance(reasoning, dict):
            reasoning.pop("summary", None)
        return params

    def _convert_messages(
        self, messages: list[Message], system_prompt: str | None
    ) -> list[dict[str, Any]]:
        converted = super()._convert_messages(messages, system_prompt)
        if system_prompt and converted:
            converted[0]["role"] = "developer"
        return converted
