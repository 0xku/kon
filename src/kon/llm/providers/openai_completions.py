import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, cast

from openai import APIStatusError, AsyncOpenAI, RateLimitError
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from kon import config as kon_config

from ...core.types import (
    AssistantMessage,
    ImageContent,
    Message,
    StopReason,
    StreamDone,
    StreamError,
    StreamPart,
    TextContent,
    TextPart,
    ThinkingContent,
    ThinkPart,
    ToolCall,
    ToolCallDelta,
    ToolCallStart,
    ToolDefinition,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from ..base import BaseProvider, LLMStream, ProviderConfig, is_local_base_url, resolve_api_key
from .openai_compat import supports_developer_role
from .sanitize import sanitize_surrogates


@dataclass
class OpenAICompletionsCompat:
    supports_store: bool = True
    supports_developer_role: bool = True
    supports_reasoning_effort: bool = True
    max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = "max_completion_tokens"
    thinking_format: Literal["openai", "zai", "qwen", "llama_gemma"] = "openai"


def _detect_compat(provider: str, base_url: str, model: str = "") -> OpenAICompletionsCompat:
    normalized_provider = provider.lower()
    normalized_base_url = base_url.lower()
    normalized_model = model.lower()
    is_zai = (
        normalized_provider == "zai"
        or normalized_provider == "zhipu"
        or "api.z.ai" in normalized_base_url
    )
    is_deepseek = normalized_provider == "deepseek" or "api.deepseek.com" in normalized_base_url

    if is_zai:
        return OpenAICompletionsCompat(
            supports_store=False,
            supports_developer_role=False,
            supports_reasoning_effort=False,
            thinking_format="zai",
        )

    if is_deepseek:
        return OpenAICompletionsCompat(
            supports_store=False, supports_developer_role=False, supports_reasoning_effort=False
        )

    if is_local_base_url(base_url) and "gemma" in normalized_model:
        return OpenAICompletionsCompat(
            supports_developer_role=supports_developer_role(provider, base_url),
            supports_reasoning_effort=False,
            thinking_format="llama_gemma",
        )

    return OpenAICompletionsCompat(
        supports_developer_role=supports_developer_role(provider, base_url)
    )


class OpenAICompletionsProvider(BaseProvider):
    name = "openai"
    thinking_levels: list[str] = ["none", "minimal", "low", "medium", "high", "xhigh"]  # noqa: RUF012
    # Whether to send reasoning_effort param. Some providers (e.g. Copilot)
    # enable thinking server-side and don't accept this parameter.
    supports_reasoning_effort: bool = True
    # Copilot requires assistant content as string, not array.
    # Sending as array causes Claude models to re-answer all previous prompts.
    force_string_assistant_content: bool = False

    def __init__(self, config: ProviderConfig):
        super().__init__(config)

        api_key = resolve_api_key(
            config.api_key,
            env_vars=("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ZAI_API_KEY"),
            base_url=config.base_url,
            auth_mode=config.openai_compat_auth_mode,
        )
        if not api_key:
            raise ValueError(
                f"No API key found for {self.name}. "
                "Set OPENAI_API_KEY, DEEPSEEK_API_KEY, or ZAI_API_KEY environment variable, "
                'or configure llm.auth.openai_compat = "auto"/"none" for local endpoints.'
            )
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=kon_config.llm.request_timeout_seconds,
        )
        self._compat = _detect_compat(
            config.provider or "", config.base_url or "", config.model or ""
        )

    async def _stream_impl(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMStream:
        compat = self._compat
        openai_messages = self._convert_messages(messages, system_prompt, compat)
        openai_tools = self._convert_tools(tools) if tools else None

        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        create_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": openai_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if compat.supports_store:
            create_kwargs["store"] = False

        if temp is not None:
            create_kwargs["temperature"] = temp

        if max_tok is not None:
            if compat.max_tokens_field == "max_tokens":
                create_kwargs["max_tokens"] = max_tok
            else:
                create_kwargs["max_completion_tokens"] = max_tok

        if openai_tools:
            create_kwargs["tools"] = openai_tools

        # Thinking format varies by provider
        thinking_level = self.config.thinking_level
        extra_body: dict[str, Any] = {}
        if compat.thinking_format == "zai":
            if thinking_level and thinking_level != "none":
                extra_body["thinking"] = {"type": "enabled"}
        elif compat.thinking_format in {"qwen", "llama_gemma"}:
            extra_body["enable_thinking"] = bool(thinking_level and thinking_level != "none")
        elif (
            self.supports_reasoning_effort
            and compat.supports_reasoning_effort
            and thinking_level
            and thinking_level != "none"
        ):
            create_kwargs["reasoning_effort"] = thinking_level

        if extra_body:
            create_kwargs["extra_body"] = extra_body

        response = await self._client.chat.completions.create(**create_kwargs)

        llm_stream = LLMStream()
        llm_stream.set_iterator(self._process_stream(response, llm_stream))
        return llm_stream

    async def _process_stream(
        self, response: AsyncIterator[ChatCompletionChunk], llm_stream: LLMStream
    ) -> AsyncIterator[StreamPart]:
        stop_reason: StopReason = StopReason.STOP

        try:
            async for chunk in response:
                if chunk.usage:
                    prompt_details = getattr(chunk.usage, "prompt_tokens_details", None)
                    cached = getattr(prompt_details, "cached_tokens", 0) or 0
                    cache_write = (
                        getattr(prompt_details, "cache_write_tokens", 0)
                        or getattr(prompt_details, "cache_creation_tokens", 0)
                        or getattr(chunk.usage, "cache_write_tokens", 0)
                        or 0
                    )
                    prompt_tokens = chunk.usage.prompt_tokens or 0
                    non_cached_input = max(prompt_tokens - cached, 0)
                    llm_stream._usage = Usage(
                        input_tokens=non_cached_input,
                        output_tokens=chunk.usage.completion_tokens or 0,
                        cache_read_tokens=cached,
                        cache_write_tokens=cache_write,
                    )

                if chunk.id:
                    llm_stream._id = chunk.id

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if choice.finish_reason:
                    stop_reason = self._map_finish_reason(choice.finish_reason)

                # Handle thinking/reasoning content (extended OpenAI format)
                # Providers use "reasoning_content", "reasoning", or "reasoning_text"
                # Store which field was used as signature so we can send it back correctly
                for field_name in ("reasoning_content", "reasoning", "reasoning_text"):
                    reasoning = getattr(delta, field_name, None)
                    if reasoning:
                        yield ThinkPart(think=reasoning, signature=field_name)
                        break

                if delta.content:
                    yield TextPart(text=delta.content)

                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if tool_call.index is None:
                            continue

                        if tool_call.function and tool_call.function.name:
                            yield ToolCallStart(
                                id=tool_call.id or "",
                                name=tool_call.function.name,
                                index=tool_call.index,
                            )

                        if tool_call.function and tool_call.function.arguments:
                            yield ToolCallDelta(
                                index=tool_call.index, arguments_delta=tool_call.function.arguments
                            )

            yield StreamDone(stop_reason=stop_reason)

        except Exception as e:
            yield StreamError(error=str(e))

    def _convert_messages(
        self,
        messages: list[Message],
        system_prompt: str | None,
        compat: OpenAICompletionsCompat | None = None,
    ) -> list[ChatCompletionMessageParam]:
        result: list[ChatCompletionMessageParam] = []

        if system_prompt:
            role = "developer" if (compat and compat.supports_developer_role) else "system"
            prompt_content = sanitize_surrogates(system_prompt)
            if (
                compat
                and compat.thinking_format == "llama_gemma"
                and self.config.thinking_level != "none"
                and not prompt_content.startswith("<|think|>")
            ):
                prompt_content = "<|think|>" + prompt_content
            result.append(
                cast(ChatCompletionMessageParam, {"role": role, "content": prompt_content})
            )

        pending_images: list[ImageContent] = []

        for msg in messages:
            if isinstance(msg, UserMessage):
                if pending_images:
                    result.append(self._create_image_user_message(pending_images))
                    pending_images = []
                result.append(self._convert_user_message(msg))
            elif isinstance(msg, AssistantMessage):
                if pending_images:
                    result.append(self._create_image_user_message(pending_images))
                    pending_images = []
                result.append(self._convert_assistant_message(msg))
            elif isinstance(msg, ToolResultMessage):
                result.append(self._convert_tool_result(msg))
                for item in msg.content:
                    if isinstance(item, ImageContent):
                        pending_images.append(item)

        if pending_images:
            result.append(self._create_image_user_message(pending_images))

        return result

    def _create_image_user_message(self, images: list[ImageContent]) -> ChatCompletionMessageParam:
        parts: list[dict[str, Any]] = [
            {"type": "text", "text": "Attached image(s) from tool result:"}
        ]
        for img in images:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{img.mime_type};base64,{img.data}"},
                }
            )
        return cast(ChatCompletionMessageParam, {"role": "user", "content": parts})

    def _convert_user_message(self, msg: UserMessage) -> ChatCompletionMessageParam:
        if isinstance(msg.content, str):
            return cast(
                ChatCompletionMessageParam,
                {"role": "user", "content": sanitize_surrogates(msg.content)},
            )

        # Multi-part content (text + images)
        parts: list[dict[str, Any]] = []
        for item in msg.content:
            if isinstance(item, TextContent):
                parts.append({"type": "text", "text": sanitize_surrogates(item.text)})
            elif isinstance(item, ImageContent):
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{item.mime_type};base64,{item.data}"},
                    }
                )

        return cast(ChatCompletionMessageParam, {"role": "user", "content": parts})

    def _convert_assistant_message(self, msg: AssistantMessage) -> ChatCompletionMessageParam:
        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        thinking_by_field: dict[str, list[str]] = {}

        for item in msg.content:
            if isinstance(item, TextContent):
                if item.text.strip():
                    content_parts.append(sanitize_surrogates(item.text))
            elif isinstance(item, ThinkingContent):
                if item.thinking.strip():
                    field = item.signature or "reasoning_content"
                    if field not in thinking_by_field:
                        thinking_by_field[field] = []
                    thinking_by_field[field].append(item.thinking)
            elif isinstance(item, ToolCall):
                tool_calls.append(
                    {
                        "id": item.id,
                        "type": "function",
                        "function": {"name": item.name, "arguments": json.dumps(item.arguments)},
                    }
                )

        # Copilot requires assistant content as a string, not an array.
        # Sending as array causes Claude models to re-answer all previous prompts.
        if self.force_string_assistant_content:
            content: Any = "".join(content_parts) if content_parts else None
        else:
            content = (
                [{"type": "text", "text": t} for t in content_parts] if content_parts else None
            )

        result: dict[str, Any] = {"role": "assistant", "content": content}

        for field, thinking_list in thinking_by_field.items():
            result[field] = "\n".join(thinking_list)

        if tool_calls:
            result["tool_calls"] = tool_calls

        # Skip assistant messages with no content and no tool calls
        if not content and not tool_calls:
            return cast(ChatCompletionMessageParam, {"role": "assistant", "content": ""})

        return cast(ChatCompletionMessageParam, result)

    def _convert_tool_result(self, msg: ToolResultMessage) -> ChatCompletionMessageParam:
        text_parts = [item.text for item in msg.content if isinstance(item, TextContent)]
        has_images = any(isinstance(item, ImageContent) for item in msg.content)

        # If there's text, use it; otherwise indicate images are attached
        if text_parts:
            content = "\n".join(text_parts)
        elif has_images:
            content = "(see attached image)"
        else:
            content = "(no output)"

        return cast(
            ChatCompletionMessageParam,
            {"role": "tool", "tool_call_id": msg.tool_call_id, "content": content},
        )

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[ChatCompletionToolParam]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _map_finish_reason(self, reason: str) -> StopReason:
        match reason:
            case "stop":
                return StopReason.STOP
            case "length":
                return StopReason.LENGTH
            case "tool_calls":
                return StopReason.TOOL_USE
            case _:
                return StopReason.STOP

    def should_retry_for_error(self, error: Exception) -> bool:
        if isinstance(error, RateLimitError):
            return True
        if isinstance(error, APIStatusError):
            return error.status_code >= 500
        return False
