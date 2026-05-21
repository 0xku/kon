import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp

from kon import config as kon_config

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
from ..base import BaseProvider, LLMStream
from ..oauth.openai import get_valid_openai_credentials

_MAX_RETRIES = 3
_BASE_DELAY_MS = 1000
_CONNECT_TIMEOUT_SECONDS = 30
_OPENAI_BETA_RESPONSES_WEBSOCKETS = "responses_websockets=2026-02-06"
_WS_FALLBACK_SESSIONS: set[str] = set()


class CodexTransportError(Exception):
    pass


class CodexNonTransportError(Exception):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _format_provider_error(error: Exception) -> str:
    message = str(error).strip()
    if message:
        return message
    return f"{error.__class__.__name__}: request failed without an error message"


def _is_retryable_status(status: int) -> bool:
    return status in (429, 500, 502, 503, 504)


class OpenAICodexResponsesProvider(BaseProvider):
    name = "openai-codex"
    thinking_levels: list[str] = ["none", "minimal", "low", "medium", "high", "xhigh"]  # noqa: RUF012

    async def _stream_impl(
        self,
        messages: list[Message],
        *,
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMStream:
        creds = await get_valid_openai_credentials()
        if not creds:
            raise RuntimeError("Not logged in to OpenAI. Use /login to authenticate.")

        llm_stream = LLMStream()
        llm_stream.set_iterator(
            self._stream_codex(
                token=creds.access,
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

    def _resolve_websocket_url(self) -> str:
        parsed = urlsplit(self._resolve_url())
        scheme = (
            "wss"
            if parsed.scheme == "https"
            else "ws"
            if parsed.scheme == "http"
            else parsed.scheme
        )
        return urlunsplit((scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment))

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
            headers["x-client-request-id"] = self.config.session_id
            headers["conversation_id"] = self.config.session_id
        return headers

    def _build_websocket_headers(self, token: str, account_id: str) -> dict[str, str]:
        request_id = self.config.session_id or str(uuid.uuid4())
        return {
            "Authorization": f"Bearer {token}",
            "chatgpt-account-id": account_id,
            "OpenAI-Beta": _OPENAI_BETA_RESPONSES_WEBSOCKETS,
            "originator": "kon",
            "User-Agent": "kon",
            "x-client-request-id": request_id,
            "session_id": request_id,
        }

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
        body = self._build_request_body(messages, system_prompt, tools, temperature)
        session_id = self.config.session_id

        if session_id not in _WS_FALLBACK_SESSIONS:
            emitted = False
            try:
                websocket_events = self._stream_websocket_events(
                    body, self._build_websocket_headers(token, account_id)
                )
                async for part in self._process_codex_events(websocket_events, llm_stream):
                    emitted = True
                    yield part
                return
            except CodexNonTransportError as e:
                yield StreamError(error=_format_provider_error(e))
                return
            except (CodexTransportError, aiohttp.ClientError, OSError) as e:
                if emitted:
                    yield StreamError(error=_format_provider_error(e))
                    return
                if session_id:
                    _WS_FALLBACK_SESSIONS.add(session_id)

        try:
            sse_events = self._stream_sse_events(body, self._build_headers(token, account_id))
            async for part in self._process_codex_events(sse_events, llm_stream):
                yield part
        except Exception as e:
            yield StreamError(error=_format_provider_error(e))

    async def _process_codex_events(
        self, events: AsyncIterator[dict[str, Any]], llm_stream: LLMStream
    ) -> AsyncIterator[StreamPart]:
        current_tool_calls: dict[str, dict[str, Any]] = {}
        call_key_by_item_id: dict[str, str] = {}
        last_tool_call_id: str | None = None
        tool_call_index = 0

        def _resolve_key(item_id: Any) -> str | None:
            if isinstance(item_id, str) and item_id:
                return call_key_by_item_id.get(item_id)
            return last_tool_call_id

        def _reconcile(call_data: dict[str, Any], final_args: str) -> ToolCallDelta | None:
            current_args = call_data["arguments"]
            if final_args.startswith(current_args):
                missing = final_args[len(current_args) :]
                if not missing:
                    return None
                call_data["arguments"] += missing
                return ToolCallDelta(index=call_data["index"], arguments_delta=missing)
            if final_args == current_args:
                return None
            call_data["arguments"] = final_args
            return ToolCallDelta(
                index=call_data["index"], arguments_delta=final_args, replace=True
            )

        async for event in events:
            event_type = event.get("type")
            if not isinstance(event_type, str):
                continue

            if event_type == "response.reasoning_summary_text.delta":
                delta = event.get("delta")
                if isinstance(delta, str):
                    yield ThinkPart(think=delta)

            elif event_type == "response.output_text.delta":
                delta = event.get("delta")
                if isinstance(delta, str):
                    yield TextPart(text=delta)

            elif event_type == "response.output_item.added":
                item = event.get("item")
                if isinstance(item, dict) and item.get("type") == "function_call":
                    call_id = item.get("call_id")
                    item_id = item.get("id")
                    name = item.get("name")
                    if isinstance(call_id, str) and isinstance(name, str):
                        full_id = f"{call_id}|{item_id}" if isinstance(item_id, str) else call_id
                        initial_args = item.get("arguments")
                        initial_args_text = initial_args if isinstance(initial_args, str) else ""
                        current_tool_calls[full_id] = {
                            "arguments": initial_args_text,
                            "index": tool_call_index,
                        }
                        if isinstance(item_id, str) and item_id:
                            call_key_by_item_id[item_id] = full_id
                        last_tool_call_id = full_id
                        yield ToolCallStart(index=tool_call_index, id=full_id, name=name)
                        if initial_args_text:
                            yield ToolCallDelta(
                                index=tool_call_index, arguments_delta=initial_args_text
                            )
                        tool_call_index += 1

            elif event_type == "response.function_call_arguments.delta":
                delta = event.get("delta")
                if not isinstance(delta, str):
                    continue
                call_key = _resolve_key(event.get("item_id"))
                if not call_key or call_key not in current_tool_calls:
                    continue
                call_data = current_tool_calls[call_key]
                call_data["arguments"] += delta
                yield ToolCallDelta(index=call_data["index"], arguments_delta=delta)

            elif event_type == "response.function_call_arguments.done":
                final_args = event.get("arguments")
                if not isinstance(final_args, str):
                    continue
                call_key = _resolve_key(event.get("item_id"))
                if not call_key or call_key not in current_tool_calls:
                    continue
                call_data = current_tool_calls[call_key]
                delta_part = _reconcile(call_data, final_args)
                if delta_part is not None:
                    yield delta_part

            elif event_type == "response.output_item.done":
                item = event.get("item")
                if not isinstance(item, dict) or item.get("type") != "function_call":
                    continue
                final_args = item.get("arguments")
                if not isinstance(final_args, str):
                    continue
                call_key = _resolve_key(item.get("id"))
                if not call_key or call_key not in current_tool_calls:
                    continue
                call_data = current_tool_calls[call_key]
                delta_part = _reconcile(call_data, final_args)
                if delta_part is not None:
                    yield delta_part

            elif event_type in {"response.completed", "response.done", "response.incomplete"}:
                response_obj = event.get("response")
                if isinstance(response_obj, dict):
                    self._apply_response_metadata(response_obj, llm_stream)
                    stop_reason = self._map_stop_reason(response_obj)
                    if current_tool_calls and stop_reason == StopReason.STOP:
                        stop_reason = StopReason.TOOL_USE
                    yield StreamDone(stop_reason=stop_reason)
                    return

            elif event_type == "error":
                code = event.get("code")
                message = event.get("message")
                if isinstance(message, str) and message:
                    raise CodexNonTransportError(f"Codex error: {message}")
                if isinstance(code, str) and code:
                    raise CodexNonTransportError(f"Codex error: {code}")
                raise CodexNonTransportError(f"Codex error: {json.dumps(event)}")

            elif event_type == "response.failed":
                response_obj = event.get("response")
                msg = None
                if isinstance(response_obj, dict):
                    err = response_obj.get("error")
                    if isinstance(err, dict):
                        msg = err.get("message")
                err_msg = msg if isinstance(msg, str) and msg else "Codex response failed"
                raise CodexNonTransportError(err_msg)

    def _apply_response_metadata(
        self, response_obj: dict[str, Any], llm_stream: LLMStream
    ) -> None:
        usage = response_obj.get("usage")
        if isinstance(usage, dict):
            input_details = usage.get("input_tokens_details")
            cached = 0
            cache_write_value = usage.get("cache_write_tokens")
            if isinstance(input_details, dict):
                cached = int(input_details.get("cached_tokens") or 0)
                if "cache_write_tokens" in input_details:
                    cache_write_value = input_details["cache_write_tokens"]
                elif "cache_creation_tokens" in input_details:
                    cache_write_value = input_details["cache_creation_tokens"]
            cache_write = int(cache_write_value) if cache_write_value is not None else 0
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

    async def _stream_sse_events(
        self, body: dict[str, Any], headers: dict[str, str]
    ) -> AsyncIterator[dict[str, Any]]:
        last_error: str | None = None
        timeout = aiohttp.ClientTimeout(
            sock_connect=_CONNECT_TIMEOUT_SECONDS, sock_read=kon_config.llm.request_timeout_seconds
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            response: aiohttp.ClientResponse | None = None
            for attempt in range(_MAX_RETRIES + 1):
                response = await session.post(self._resolve_url(), headers=headers, json=body)
                if response.status < 400:
                    break
                error_text = await response.text()
                last_error = f"Codex API error ({response.status}): {error_text}"
                if attempt < _MAX_RETRIES and _is_retryable_status(response.status):
                    delay = _BASE_DELAY_MS * (2**attempt) / 1000
                    await asyncio.sleep(delay)
                    continue
                raise CodexNonTransportError(last_error, status=response.status)

            if response is None or response.status >= 400:
                raise CodexNonTransportError(last_error or "Codex request failed after retries")

            async for event in self._parse_sse(response):
                yield event

    async def _stream_websocket_events(
        self, body: dict[str, Any], headers: dict[str, str]
    ) -> AsyncIterator[dict[str, Any]]:
        timeout = aiohttp.ClientTimeout(
            sock_connect=_CONNECT_TIMEOUT_SECONDS, sock_read=kon_config.llm.request_timeout_seconds
        )
        ws_timeout = aiohttp.ClientWSTimeout(ws_receive=kon_config.llm.request_timeout_seconds)  # type: ignore[call-arg]
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.ws_connect(
                self._resolve_websocket_url(), headers=headers, heartbeat=20, timeout=ws_timeout
            ) as ws,
        ):
            await ws.send_json({"type": "response.create", **body})
            saw_completion = False
            async for msg in ws:
                if msg.type in {aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY}:
                    try:
                        raw = (
                            msg.data.decode()
                            if isinstance(msg.data, bytes | bytearray)
                            else msg.data
                        )
                        event = json.loads(raw)
                    except Exception as e:
                        raise CodexNonTransportError(f"Invalid Codex WebSocket JSON: {e}") from e
                    if not isinstance(event, dict):
                        continue
                    event_type = event.get("type")
                    if event_type in {
                        "response.completed",
                        "response.done",
                        "response.incomplete",
                    }:
                        saw_completion = True
                    yield event
                    if saw_completion:
                        return
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    error = ws.exception()
                    raise CodexTransportError(str(error) if error else "WebSocket error")

            if not saw_completion:
                raise CodexTransportError("WebSocket stream closed before response.completed")

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
                except json.JSONDecodeError as e:
                    raise CodexNonTransportError(f"Invalid Codex SSE JSON: {e}") from e
                if isinstance(parsed, dict):
                    yield parsed

    def _map_stop_reason(self, response_obj: dict[str, Any]) -> StopReason:
        status = response_obj.get("status")
        if status == "completed":
            return StopReason.STOP
        if status == "incomplete":
            details = response_obj.get("incomplete_details")
            reason = details.get("reason") if isinstance(details, dict) else None
            if reason == "content_filter":
                return StopReason.ERROR
            return StopReason.LENGTH
        if status in {"failed", "cancelled"}:
            return StopReason.ERROR
        return StopReason.STOP

    def should_retry_for_error(self, error: Exception) -> bool:
        if isinstance(error, CodexTransportError):
            return True
        if isinstance(error, CodexNonTransportError) and error.status is not None:
            return _is_retryable_status(error.status)
        return False
