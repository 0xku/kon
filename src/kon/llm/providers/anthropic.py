from collections.abc import AsyncIterator
from typing import Any, cast

from anthropic import APIStatusError, AsyncAnthropic, RateLimitError
from anthropic.types import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ImageBlockParam,
    MessageDeltaEvent,
    MessageParam,
    MessageStartEvent,
    MessageStopEvent,
    TextBlockParam,
    ThinkingBlock,
    ThinkingConfigEnabledParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
)

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
from ..base import BaseProvider, LLMStream, ProviderConfig, get_env_api_key
from .sanitize import sanitize_surrogates

THINKING_BUDGET_MAP: dict[str, int] = {
    "none": 0,
    "minimal": 1024,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
}


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    thinking_levels: list[str] = ["none", "minimal", "low", "medium", "high", "xhigh"]  # noqa: RUF012

    def __init__(self, config: ProviderConfig):
        super().__init__(config)

        api_key = config.api_key or get_env_api_key(self.name)
        if not api_key:
            raise ValueError(
                f"No API key found for {self.name}. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key in config."
            )

        self._client = AsyncAnthropic(api_key=api_key, base_url=config.base_url)

    async def _stream_impl(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMStream:
        anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools) if tools else None

        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        create_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "max_tokens": max_tok,
        }

        if system_prompt:
            # Use text block with cache_control for prompt caching
            create_kwargs["system"] = [
                {
                    "type": "text",
                    "text": sanitize_surrogates(system_prompt),
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        # Add temperature if specified and not using thinking
        temp = temperature if temperature is not None else self.config.temperature
        thinking_budget = THINKING_BUDGET_MAP.get(self.config.thinking_level, 0)

        if thinking_budget > 0:
            # Extended thinking - temperature must be 1
            create_kwargs["thinking"] = ThinkingConfigEnabledParam(
                type="enabled", budget_tokens=thinking_budget
            )
            # Don't set temperature when thinking is enabled (defaults to 1)
        elif temp is not None:
            create_kwargs["temperature"] = temp

        if anthropic_tools:
            create_kwargs["tools"] = anthropic_tools

        response = await self._client.messages.stream(**create_kwargs).__aenter__()

        llm_stream = LLMStream()
        llm_stream.set_iterator(self._process_stream(response, llm_stream))
        return llm_stream

    async def _process_stream(
        self, response: Any, llm_stream: LLMStream
    ) -> AsyncIterator[StreamPart]:
        stop_reason: StopReason = StopReason.STOP
        current_tool_index: int = -1
        tool_use_blocks: dict[int, dict[str, Any]] = {}  # index -> {id, name}

        try:
            async for event in response:
                if isinstance(event, MessageStartEvent):
                    if event.message.id:
                        llm_stream._id = event.message.id
                    if event.message.usage:
                        cache_read = (
                            getattr(event.message.usage, "cache_read_input_tokens", 0) or 0
                        )
                        cache_write = (
                            getattr(event.message.usage, "cache_creation_input_tokens", 0) or 0
                        )
                        llm_stream._usage = Usage(
                            input_tokens=event.message.usage.input_tokens,
                            output_tokens=event.message.usage.output_tokens,
                            cache_read_tokens=cache_read,
                            cache_write_tokens=cache_write,
                        )

                elif isinstance(event, ContentBlockStartEvent):
                    block = event.content_block
                    if isinstance(block, ToolUseBlock):
                        current_tool_index += 1
                        tool_use_blocks[event.index] = {"id": block.id, "name": block.name}
                        yield ToolCallStart(id=block.id, name=block.name, index=current_tool_index)
                    elif isinstance(block, ThinkingBlock):
                        # Thinking block start - content comes in deltas
                        pass

                elif isinstance(event, ContentBlockDeltaEvent):
                    delta = event.delta
                    delta_type = delta.type

                    if delta_type == "text_delta":
                        yield TextPart(text=delta.text)  # type: ignore[attr-defined]
                    elif delta_type == "thinking_delta":
                        yield ThinkPart(think=delta.thinking)  # type: ignore[attr-defined]
                    elif delta_type == "signature_delta":
                        # Signature comes after thinking content - emit as ThinkPart
                        yield ThinkPart(think="", signature=delta.signature)  # type: ignore[attr-defined]
                    elif delta_type == "input_json_delta":
                        # Find the tool index for this content block
                        tool_info = tool_use_blocks.get(event.index)
                        if tool_info:
                            # Find the logical tool index
                            logical_index = list(tool_use_blocks.keys()).index(event.index)
                            yield ToolCallDelta(
                                index=logical_index,
                                arguments_delta=delta.partial_json,  # type: ignore[attr-defined]
                            )

                elif isinstance(event, MessageDeltaEvent):
                    if event.delta.stop_reason:
                        stop_reason = self._map_stop_reason(event.delta.stop_reason)
                    if event.usage and llm_stream._usage:
                        cache_read = getattr(event.usage, "cache_read_input_tokens", None)
                        cache_write = getattr(event.usage, "cache_creation_input_tokens", None)
                        llm_stream._usage = Usage(
                            input_tokens=llm_stream._usage.input_tokens,
                            output_tokens=event.usage.output_tokens,
                            cache_read_tokens=cache_read
                            if cache_read is not None
                            else llm_stream._usage.cache_read_tokens,
                            cache_write_tokens=cache_write
                            if cache_write is not None
                            else llm_stream._usage.cache_write_tokens,
                        )

                elif isinstance(event, MessageStopEvent):
                    pass

            yield StreamDone(stop_reason=stop_reason)

        except Exception as e:
            yield StreamError(error=str(e))

    def _convert_messages(self, messages: list[Message]) -> list[MessageParam]:
        result: list[MessageParam] = []

        for msg in messages:
            if isinstance(msg, UserMessage):
                result.append(self._convert_user_message(msg))
            elif isinstance(msg, AssistantMessage):
                assistant_message = self._convert_assistant_message(msg)
                # Anthropic rejects replayed thinking blocks without signatures.
                # Interrupted/malformed historical messages can contain such blocks,
                # so we drop them and skip assistant entries that become empty.
                if assistant_message["content"]:
                    result.append(assistant_message)
            elif isinstance(msg, ToolResultMessage):
                # Anthropic requires tool results as user messages
                result.append(self._convert_tool_result(msg))

        # Add cache_control to the last user message to cache conversation history
        self._add_cache_control_to_last_user_message(result)

        return result

    @staticmethod
    def _add_cache_control_to_last_user_message(messages: list[MessageParam]) -> None:
        if not messages:
            return

        last_message = messages[-1]
        if last_message["role"] != "user":
            return

        content = last_message["content"]
        if isinstance(content, str):
            # Convert string to text block with cache_control
            last_message["content"] = [
                {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
            ]
        elif isinstance(content, list) and content:
            # Add cache_control to the last block
            last_block = content[-1]
            if isinstance(last_block, dict):
                last_block_dict = cast(dict[str, Any], last_block)
                if last_block_dict.get("type") in ("text", "image", "tool_result"):
                    last_block_dict["cache_control"] = {"type": "ephemeral"}  # type: ignore[typeddict-unknown-key]

    def _convert_user_message(self, msg: UserMessage) -> MessageParam:
        if isinstance(msg.content, str):
            return {"role": "user", "content": sanitize_surrogates(msg.content)}

        # Multi-part content
        parts: list[TextBlockParam | ImageBlockParam] = []
        for item in msg.content:
            if isinstance(item, TextContent):
                parts.append({"type": "text", "text": sanitize_surrogates(item.text)})
            elif isinstance(item, ImageContent):
                parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": cast(Any, item.mime_type),  # image/jpeg, image/png, etc.
                            "data": item.data,
                        },
                    }
                )

        return {"role": "user", "content": parts}

    def _convert_assistant_message(self, msg: AssistantMessage) -> MessageParam:
        parts: list[Any] = []

        for item in msg.content:
            if isinstance(item, ThinkingContent):
                # Anthropic requires a signature on replayed thinking blocks.
                # Keep valid signed thinking and drop invalid/partial thinking.
                if not item.signature:
                    continue

                thinking_block: dict[str, Any] = {
                    "type": "thinking",
                    "thinking": sanitize_surrogates(item.thinking),
                    "signature": item.signature,
                }
                parts.append(thinking_block)
            elif isinstance(item, TextContent):
                parts.append({"type": "text", "text": sanitize_surrogates(item.text)})
            elif isinstance(item, ToolCall):
                parts.append(
                    {"type": "tool_use", "id": item.id, "name": item.name, "input": item.arguments}
                )

        return {"role": "assistant", "content": parts}

    def _convert_tool_result(self, msg: ToolResultMessage) -> MessageParam:
        # Convert content to list
        content_parts: list[TextBlockParam | ImageBlockParam] = []
        for item in msg.content:
            if isinstance(item, TextContent):
                content_parts.append({"type": "text", "text": sanitize_surrogates(item.text)})
            elif isinstance(item, ImageContent):
                content_parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": cast(Any, item.mime_type),
                            "data": item.data,
                        },
                    }
                )

        tool_result: ToolResultBlockParam = {
            "type": "tool_result",
            "tool_use_id": msg.tool_call_id,
            "content": content_parts if content_parts else "",
        }

        if msg.is_error:
            tool_result["is_error"] = True

        return {"role": "user", "content": [tool_result]}

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[ToolParam]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": cast(Any, tool.parameters),
            }
            for tool in tools
        ]

    def _map_stop_reason(self, reason: str) -> StopReason:
        match reason:
            case "end_turn":
                return StopReason.STOP
            case "max_tokens":
                return StopReason.LENGTH
            case "tool_use":
                return StopReason.TOOL_USE
            case _:
                return StopReason.STOP

    def should_retry_for_error(self, error: Exception) -> bool:
        if isinstance(error, RateLimitError):
            return True
        if isinstance(error, APIStatusError):
            return error.status_code >= 500
        return False
