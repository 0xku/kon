import json
from collections.abc import AsyncIterator
from typing import Any

from openai import APIStatusError, AsyncOpenAI, RateLimitError

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
from ..base import BaseProvider, LLMStream, ProviderConfig
from .openai_compat import supports_developer_role
from .sanitize import sanitize_surrogates

COPILOT_HEADERS = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}


class OpenAIResponsesProvider(BaseProvider):
    name = "openai-responses"
    thinking_levels: list[str] = ["none", "low", "medium", "high", "xhigh"]  # noqa: RUF012

    def __init__(self, config: ProviderConfig, headers: dict[str, str] | None = None):
        super().__init__(config)
        self._headers = headers or {}
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                default_headers=self._headers,
            )
        return self._client

    async def _stream_impl(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMStream:
        client = self._get_client()
        params = self._build_params(messages, system_prompt, tools, max_tokens)
        response_stream = await client.responses.create(**params)
        llm_stream = LLMStream()
        llm_stream.set_iterator(self._process_stream(response_stream, llm_stream))
        return llm_stream

    async def _process_stream(
        self, response_stream: Any, llm_stream: LLMStream
    ) -> AsyncIterator[StreamPart]:
        current_text = ""
        current_thinking = ""
        current_tool_calls: dict[str, dict[str, Any]] = {}
        call_key_by_item_id: dict[str, str] = {}
        current_tool_call_key: str | None = None
        tool_call_index = 0

        try:
            async for event in response_stream:
                event_type = event.type

                if event_type in (
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_text.delta",
                ):
                    delta = event.delta
                    current_thinking += delta
                    yield ThinkPart(think=delta)

                elif event_type == "response.output_text.delta":
                    delta = event.delta
                    current_text += delta
                    yield TextPart(text=delta)

                elif event_type == "response.output_item.added":
                    item = event.item
                    if item.type == "function_call":
                        item_id = item.id or ""
                        call_id = f"{item.call_id}|{item_id}"
                        current_tool_calls[call_id] = {
                            "id": call_id,
                            "name": item.name,
                            "arguments": item.arguments or "",
                            "index": tool_call_index,
                        }
                        if item_id:
                            call_key_by_item_id[item_id] = call_id
                        current_tool_call_key = call_id
                        yield ToolCallStart(index=tool_call_index, id=call_id, name=item.name)
                        tool_call_index += 1

                elif event_type == "response.function_call_arguments.delta":
                    item_id = getattr(event, "item_id", None)
                    call_key = (
                        call_key_by_item_id.get(item_id) if item_id else current_tool_call_key
                    )
                    if not call_key:
                        continue
                    call_data = current_tool_calls.get(call_key)
                    if not call_data:
                        continue
                    delta = event.delta
                    call_data["arguments"] += delta
                    yield ToolCallDelta(index=call_data["index"], arguments_delta=delta)

                elif event_type == "response.function_call_arguments.done":
                    item_id = getattr(event, "item_id", None)
                    call_key = (
                        call_key_by_item_id.get(item_id) if item_id else current_tool_call_key
                    )
                    if not call_key:
                        continue
                    call_data = current_tool_calls.get(call_key)
                    if not call_data:
                        continue
                    final_args = event.arguments
                    if final_args is None:
                        continue
                    current_args = call_data["arguments"]
                    if final_args.startswith(current_args):
                        missing = final_args[len(current_args) :]
                        if missing:
                            call_data["arguments"] += missing
                            yield ToolCallDelta(index=call_data["index"], arguments_delta=missing)
                    else:
                        call_data["arguments"] = final_args
                        yield ToolCallDelta(index=call_data["index"], arguments_delta=final_args)

                elif event_type == "response.output_item.done":
                    item = event.item
                    if item.type == "function_call":
                        item_id = item.id or ""
                        call_id = f"{item.call_id}|{item_id}"
                        call_key = current_tool_call_key
                        if call_id in current_tool_calls:
                            call_key = call_id
                        elif item_id and item_id in call_key_by_item_id:
                            call_key = call_key_by_item_id[item_id]
                        if (
                            not call_key
                            or call_key not in current_tool_calls
                            or item.arguments is None
                        ):
                            continue
                        call_data = current_tool_calls[call_key]
                        current_args = call_data["arguments"]
                        final_args = item.arguments
                        if final_args.startswith(current_args):
                            missing = final_args[len(current_args) :]
                            if missing:
                                call_data["arguments"] += missing
                                yield ToolCallDelta(
                                    index=call_data["index"], arguments_delta=missing
                                )
                        elif final_args != current_args:
                            call_data["arguments"] = final_args
                            yield ToolCallDelta(
                                index=call_data["index"], arguments_delta=final_args
                            )

                elif event_type in ("response.completed", "response.done"):
                    response = getattr(event, "response", None)
                    if response and response.usage:
                        cached = 0
                        if response.usage.input_tokens_details:
                            cached = response.usage.input_tokens_details.cached_tokens or 0

                        llm_stream._usage = Usage(
                            input_tokens=response.usage.input_tokens or 0,
                            output_tokens=response.usage.output_tokens or 0,
                            cache_read_tokens=cached,
                        )

                    if response and response.id:
                        llm_stream._id = response.id

                    content: list[TextContent | ThinkingContent | ToolCall] = []

                    if current_thinking:
                        content.append(ThinkingContent(thinking=current_thinking))

                    if current_text:
                        content.append(TextContent(text=current_text))

                    for _, call_data in current_tool_calls.items():
                        try:
                            call_args = call_data["arguments"]
                            args = json.loads(call_args) if call_args else {}
                        except json.JSONDecodeError:
                            args = {}
                        content.append(
                            ToolCall(id=call_data["id"], name=call_data["name"], arguments=args)
                        )

                    stop_reason = self._map_stop_reason(
                        response.status if response and hasattr(response, "status") else None
                    )
                    if current_tool_calls and stop_reason == StopReason.STOP:
                        stop_reason = StopReason.TOOL_USE

                    yield StreamDone(stop_reason=stop_reason)
                    return

                elif event_type == "error":
                    yield StreamError(error=f"Error Code {event.code}: {event.message}")
                    return

                elif event_type == "response.failed":
                    yield StreamError(error="Response failed")
                    return

        except Exception as e:
            yield StreamError(error=str(e))

    def _build_params(
        self,
        messages: list[Message],
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
        max_tokens: int | None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        input_messages = self._convert_messages(messages, system_prompt)

        params: dict[str, Any] = {
            "model": self.config.model,
            "input": input_messages,
            "stream": True,
            "store": False,
        }

        # Prompt caching - use session_id as cache key
        if session_id:
            params["prompt_cache_key"] = session_id

        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
        if max_tok:
            params["max_output_tokens"] = max_tok

        if tools:
            params["tools"] = self._convert_tools(tools)

        if self.config.thinking_level and self.config.thinking_level != "none":
            params["reasoning"] = {"effort": self.config.thinking_level, "summary": "auto"}
            params["include"] = ["reasoning.encrypted_content"]

        return params

    def _convert_messages(
        self, messages: list[Message], system_prompt: str | None
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        if system_prompt:
            role = (
                "developer"
                if supports_developer_role(self.config.provider, self.config.base_url)
                else "system"
            )
            result.append({"role": role, "content": sanitize_surrogates(system_prompt)})

        pending_images: list[ImageContent] = []

        for msg in messages:
            if isinstance(msg, UserMessage):
                if pending_images:
                    result.append(self._create_image_user_message(pending_images))
                    pending_images = []

                if isinstance(msg.content, str):
                    result.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": sanitize_surrogates(msg.content)}
                            ],
                        }
                    )
                else:
                    content_parts: list[dict[str, Any]] = []
                    for item in msg.content:
                        if isinstance(item, TextContent):
                            content_parts.append(
                                {"type": "input_text", "text": sanitize_surrogates(item.text)}
                            )
                        elif isinstance(item, ImageContent):
                            content_parts.append(
                                {
                                    "type": "input_image",
                                    "detail": "auto",
                                    "image_url": f"data:{item.mime_type};base64,{item.data}",
                                }
                            )
                    if content_parts:
                        result.append({"role": "user", "content": content_parts})

            elif isinstance(msg, AssistantMessage):
                if pending_images:
                    result.append(self._create_image_user_message(pending_images))
                    pending_images = []

                for block in msg.content:
                    if isinstance(block, ThinkingContent):
                        if block.signature:
                            try:
                                reasoning_item = json.loads(block.signature)
                                result.append(reasoning_item)
                            except json.JSONDecodeError:
                                pass
                    elif isinstance(block, TextContent):
                        content = [{"type": "output_text", "text": block.text, "annotations": []}]
                        result.append(
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": content,
                                "status": "completed",
                            }
                        )
                    elif isinstance(block, ToolCall):
                        if "|" in block.id:
                            call_id, item_id = block.id.split("|")
                        else:
                            call_id, item_id = block.id, None
                        result.append(
                            {
                                "type": "function_call",
                                "id": item_id,
                                "call_id": call_id,
                                "name": block.name,
                                "arguments": json.dumps(block.arguments),
                            }
                        )

            elif isinstance(msg, ToolResultMessage):
                text_parts = [item.text for item in msg.content if isinstance(item, TextContent)]
                has_images = any(isinstance(item, ImageContent) for item in msg.content)

                if "|" in msg.tool_call_id:
                    call_id = msg.tool_call_id.split("|")[0]
                else:
                    call_id = msg.tool_call_id
                text_result = "\n".join(text_parts) if text_parts else "(see attached)"
                result.append(
                    {"type": "function_call_output", "call_id": call_id, "output": text_result}
                )

                if has_images:
                    for item in msg.content:
                        if isinstance(item, ImageContent):
                            pending_images.append(item)

        if pending_images:
            result.append(self._create_image_user_message(pending_images))

        return result

    def _create_image_user_message(self, images: list[ImageContent]) -> dict[str, Any]:
        content_parts: list[dict[str, Any]] = [
            {"type": "input_text", "text": "Attached image(s) from tool result:"}
        ]
        for img in images:
            content_parts.append(
                {
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": f"data:{img.mime_type};base64,{img.data}",
                }
            )
        return {"role": "user", "content": content_parts}

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": False,
            }
            for tool in tools
        ]

    def _map_stop_reason(self, status: str | None) -> StopReason:
        if not status:
            return StopReason.STOP
        match status:
            case "completed":
                return StopReason.STOP
            case "incomplete":
                return StopReason.LENGTH
            case "failed" | "cancelled":
                return StopReason.ERROR
            case _:
                return StopReason.STOP

    def should_retry_for_error(self, error: Exception) -> bool:
        if isinstance(error, RateLimitError):
            return True
        if isinstance(error, APIStatusError):
            return error.status_code >= 500
        return False
