import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import aiohttp

from ...core.types import (
    Message,
    StopReason,
    StreamDone,
    StreamError,
    StreamPart,
    TextPart,
    ThinkPart,
    ToolCallDelta,
    ToolCallStart,
    ToolDefinition,
    Usage,
)
from ..base import BaseProvider, LLMStream, ProviderConfig
from ..oauth.openai import get_valid_openai_token, load_openai_credentials

_MAX_RETRIES = 3
_BASE_DELAY_MS = 1000


def _format_provider_error(error: Exception) -> str:
    message = str(error).strip()
    if message:
        return message
    return f"{error.__class__.__name__}: request failed without an error message"


def _is_retryable_status(status: int) -> bool:
    return status in (429, 500, 502, 503, 504)


class OpenAICodexResponsesProvider(BaseProvider):
    name = "openai-codex"
    thinking_levels: list[str] = ["none", "low", "medium", "high", "xhigh"]  # noqa: RUF012

    def __init__(self, config: ProviderConfig):
        super().__init__(config)

    async def _stream_impl(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMStream:
        token = await get_valid_openai_token()
        creds = load_openai_credentials()
        if not token or not creds:
            raise RuntimeError("Not logged in to OpenAI. Use /login to authenticate.")

        llm_stream = LLMStream()
        llm_stream.set_iterator(
            self._stream_codex(
                token=token,
                account_id=creds.account_id,
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                llm_stream=llm_stream,
            )
        )
        return llm_stream

    def _build_input(
        self, messages: list[Message], system_prompt: str | None
    ) -> list[dict[str, Any]]:
        from .openai_responses import OpenAIResponsesProvider

        helper = OpenAIResponsesProvider(self.config)
        return helper._convert_messages(messages, system_prompt)

    def _build_tools(self, tools: list[ToolDefinition] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
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

    def _resolve_url(self) -> str:
        base = (self.config.base_url or "https://chatgpt.com/backend-api").rstrip("/")
        if base.endswith("/codex/responses"):
            return base
        if base.endswith("/codex"):
            return f"{base}/responses"
        return f"{base}/codex/responses"

    def _build_request_body(
        self,
        messages: list[Message],
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "store": False,
            "stream": True,
            "instructions": system_prompt or "",
            "input": self._build_input(messages, None),
            "include": ["reasoning.encrypted_content"],
            "text": {"verbosity": "medium"},
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }

        if self.config.session_id:
            body["prompt_cache_key"] = self.config.session_id

        tool_payload = self._build_tools(tools)
        if tool_payload:
            body["tools"] = tool_payload

        effort = self.config.thinking_level
        if effort and effort != "none":
            body["reasoning"] = {"effort": effort, "summary": "auto"}

        temp = temperature if temperature is not None else self.config.temperature
        if temp is not None:
            body["temperature"] = temp

        return body

    def _build_headers(self, token: str, account_id: str) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {token}",
            "chatgpt-account-id": account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": "kon",
            "accept": "text/event-stream",
            "content-type": "application/json",
            "User-Agent": "kon",
        }
        if self.config.session_id:
            headers["session_id"] = self.config.session_id
            headers["conversation_id"] = self.config.session_id
        return headers

    async def _stream_codex(
        self,
        *,
        token: str,
        account_id: str,
        messages: list[Message],
        system_prompt: str | None,
        tools: list[ToolDefinition] | None,
        temperature: float | None,
        max_tokens: int | None,
        llm_stream: LLMStream,
    ) -> AsyncIterator[StreamPart]:
        url = self._resolve_url()
        body = self._build_request_body(messages, system_prompt, tools, temperature)
        headers = self._build_headers(token, account_id)

        current_text = ""
        current_thinking = ""
        current_tool_calls: dict[str, dict[str, Any]] = {}
        last_tool_call_id: str | None = None
        tool_call_index = 0

        try:
            last_error: str | None = None
            response: aiohttp.ClientResponse | None = None
            session = aiohttp.ClientSession()

            try:
                for attempt in range(_MAX_RETRIES + 1):
                    response = await session.post(url, headers=headers, json=body)
                    if response.status < 400:
                        break
                    error_text = await response.text()
                    last_error = f"Codex API error ({response.status}): {error_text}"
                    if attempt < _MAX_RETRIES and _is_retryable_status(response.status):
                        delay = _BASE_DELAY_MS * (2**attempt) / 1000
                        await asyncio.sleep(delay)
                        continue
                    yield StreamError(error=last_error)
                    return

                if response is None or response.status >= 400:
                    yield StreamError(error=last_error or "Codex request failed after retries")
                    return

                async for event in self._parse_sse(response):
                    event_type = event.get("type")
                    if not isinstance(event_type, str):
                        continue

                    if event_type == "response.reasoning_summary_text.delta":
                        delta = event.get("delta")
                        if isinstance(delta, str):
                            current_thinking += delta
                            yield ThinkPart(think=delta)

                    elif event_type == "response.output_text.delta":
                        delta = event.get("delta")
                        if isinstance(delta, str):
                            current_text += delta
                            yield TextPart(text=delta)

                    elif event_type == "response.output_item.added":
                        item = event.get("item")
                        if isinstance(item, dict) and item.get("type") == "function_call":
                            call_id = item.get("call_id")
                            item_id = item.get("id")
                            name = item.get("name")
                            if isinstance(call_id, str) and isinstance(name, str):
                                full_id = (
                                    f"{call_id}|{item_id}" if isinstance(item_id, str) else call_id
                                )
                                current_tool_calls[full_id] = {
                                    "id": full_id,
                                    "name": name,
                                    "arguments": "",
                                    "index": tool_call_index,
                                }
                                last_tool_call_id = full_id
                                yield ToolCallStart(index=tool_call_index, id=full_id, name=name)
                                tool_call_index += 1

                    elif event_type == "response.function_call_arguments.delta":
                        delta = event.get("delta")
                        if (
                            isinstance(delta, str)
                            and last_tool_call_id
                            and last_tool_call_id in current_tool_calls
                        ):
                            current_tool_calls[last_tool_call_id]["arguments"] += delta
                            idx = int(current_tool_calls[last_tool_call_id]["index"])
                            yield ToolCallDelta(index=idx, arguments_delta=delta)

                    elif event_type in {"response.completed", "response.done"}:
                        response_obj = event.get("response")
                        if isinstance(response_obj, dict):
                            usage = response_obj.get("usage")
                            if isinstance(usage, dict):
                                input_details = usage.get("input_tokens_details")
                                cached = 0
                                cache_write = int(usage.get("cache_write_tokens") or 0)
                                if isinstance(input_details, dict):
                                    cached = int(input_details.get("cached_tokens") or 0)
                                    cache_write = int(
                                        input_details.get("cache_write_tokens")
                                        or input_details.get("cache_creation_tokens")
                                        or cache_write
                                    )
                                input_tokens = int(usage.get("input_tokens") or 0)
                                non_cached_input = max(input_tokens - cached, 0)
                                llm_stream._usage = Usage(
                                    input_tokens=non_cached_input,
                                    output_tokens=int(usage.get("output_tokens") or 0),
                                    cache_read_tokens=cached,
                                    cache_write_tokens=cache_write,
                                )
                            rid = response_obj.get("id")
                            if isinstance(rid, str):
                                llm_stream._id = rid

                            stop_reason = self._map_stop_reason(response_obj.get("status"))
                            if current_tool_calls and stop_reason == StopReason.STOP:
                                stop_reason = StopReason.TOOL_USE
                            yield StreamDone(stop_reason=stop_reason)
                            return

                    elif event_type in {"response.failed", "error"}:
                        message = event.get("message")
                        if not isinstance(message, str):
                            message = "Codex response failed"
                        yield StreamError(error=message)
                        return
            finally:
                await session.close()
        except Exception as e:
            yield StreamError(error=_format_provider_error(e))

    async def _parse_sse(self, response: aiohttp.ClientResponse) -> AsyncIterator[dict[str, Any]]:
        buffer = ""
        async for raw in response.content.iter_any():
            chunk = raw.decode(errors="ignore")
            buffer += chunk

            while "\n\n" in buffer:
                part, buffer = buffer.split("\n\n", 1)
                lines = [line[5:].strip() for line in part.split("\n") if line.startswith("data:")]
                if not lines:
                    continue
                data = "\n".join(lines).strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    yield parsed

    def _map_stop_reason(self, status: Any) -> StopReason:
        if status == "completed":
            return StopReason.STOP
        if status == "incomplete":
            return StopReason.LENGTH
        if status in {"failed", "cancelled"}:
            return StopReason.ERROR
        return StopReason.STOP

    def should_retry_for_error(self, error: Exception) -> bool:
        return False


def is_openai_logged_in() -> bool:
    return load_openai_credentials() is not None
