import asyncio
import contextlib
import json
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
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
from ..base import DEFAULT_THINKING_LEVELS, BaseProvider, LLMStream
from ..models import get_model
from ..oauth.openai import get_valid_openai_credentials

_MAX_RETRIES = 3
_BASE_DELAY_MS = 1000
_CONNECT_TIMEOUT_SECONDS = 30
_OPENAI_BETA_RESPONSES_WEBSOCKETS = "responses_websockets=2026-02-06"
_RESPONSES_LITE_HEADER = "x-openai-internal-codex-responses-lite"
_RESPONSES_LITE_WS_METADATA_KEY = "ws_request_header_x_openai_internal_codex_responses_lite"
# Codex routes Responses Lite models using this protocol compatibility version.
_CODEX_PROTOCOL_VERSION = "0.144.1"
_PROMPT_CACHE_KEY_MAX_LENGTH = 64
_SESSION_WEBSOCKET_CACHE_TTL_SECONDS = 5 * 60
_WS_FALLBACK_SESSIONS: set[str] = set()


@dataclass
class _CachedWebSocketContinuation:
    last_request_body: dict[str, Any]
    last_response_id: str
    last_response_items: list[dict[str, Any]]


@dataclass
class _CachedWebSocketConnection:
    session: aiohttp.ClientSession
    ws: aiohttp.ClientWebSocketResponse
    busy: bool = False
    idle_handle: asyncio.TimerHandle | None = None
    continuation: _CachedWebSocketContinuation | None = None


_WS_SESSION_CACHE: dict[str, _CachedWebSocketConnection] = {}
_WS_CLOSE_TASKS: set[asyncio.Task[None]] = set()


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


def _is_retryable_response_error(status: int, error_text: str) -> bool:
    if _is_retryable_status(status):
        return True
    return (
        re.search(
            r"rate.?limit|overloaded|service.?unavailable|upstream.?connect|connection.?refused",
            error_text,
            re.IGNORECASE,
        )
        is not None
    )


def _clamp_prompt_cache_key(key: str | None) -> str | None:
    if key is None:
        return None
    return "".join(list(key)[:_PROMPT_CACHE_KEY_MAX_LENGTH])


def _retry_delay_seconds(response: aiohttp.ClientResponse, attempt: int) -> float:
    retry_after_ms = response.headers.get("retry-after-ms")
    if retry_after_ms is not None:
        with contextlib.suppress(ValueError):
            return max(0.0, float(retry_after_ms) / 1000)

    retry_after = response.headers.get("retry-after")
    if retry_after:
        with contextlib.suppress(ValueError):
            return max(0.0, float(retry_after))
        with contextlib.suppress(ValueError, TypeError):
            parsed = parsedate_to_datetime(retry_after)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return max(0.0, (parsed - datetime.now(UTC)).total_seconds())

    return _BASE_DELAY_MS * (2**attempt) / 1000


def _request_body_without_input(body: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in body.items() if key not in {"input", "previous_response_id"}
    }


def _get_cached_websocket_input_delta(
    body: dict[str, Any], continuation: _CachedWebSocketContinuation
) -> list[dict[str, Any]] | None:
    if _request_body_without_input(body) != _request_body_without_input(
        continuation.last_request_body
    ):
        return None

    current_input = body.get("input") or []
    if not isinstance(current_input, list):
        return None

    last_input = continuation.last_request_body.get("input") or []
    if not isinstance(last_input, list):
        return None

    baseline = [*last_input, *continuation.last_response_items]
    if len(current_input) < len(baseline):
        return None

    if current_input[: len(baseline)] != baseline:
        return None

    delta = current_input[len(baseline) :]
    return delta if all(isinstance(item, dict) for item in delta) else None


def _build_cached_websocket_request_body(
    entry: _CachedWebSocketConnection, body: dict[str, Any]
) -> dict[str, Any]:
    continuation = entry.continuation
    if continuation is None:
        return body

    delta = _get_cached_websocket_input_delta(body, continuation)
    if delta is None or not continuation.last_response_id:
        entry.continuation = None
        return body

    return {**body, "previous_response_id": continuation.last_response_id, "input": delta}


def _is_websocket_reusable(entry: _CachedWebSocketConnection) -> bool:
    return not entry.ws.closed and not entry.session.closed


def _websocket_connect_headers(headers: dict[str, str]) -> dict[str, str]:
    connect_headers = dict(headers)
    connect_headers.pop("OpenAI-Beta", None)
    connect_headers.pop("openai-beta", None)
    return connect_headers


async def _close_websocket_entry(entry: _CachedWebSocketConnection) -> None:
    if entry.idle_handle:
        entry.idle_handle.cancel()
        entry.idle_handle = None
    with contextlib.suppress(Exception):
        await entry.ws.close(code=1000, message=b"done")
    with contextlib.suppress(Exception):
        await entry.session.close()


def _schedule_websocket_expiry(session_id: str, entry: _CachedWebSocketConnection) -> None:
    if entry.idle_handle:
        entry.idle_handle.cancel()

    loop = asyncio.get_running_loop()

    def _expire() -> None:
        if entry.busy:
            return
        if _WS_SESSION_CACHE.get(session_id) is entry:
            _WS_SESSION_CACHE.pop(session_id, None)
        task = asyncio.create_task(_close_websocket_entry(entry))
        _WS_CLOSE_TASKS.add(task)
        task.add_done_callback(_WS_CLOSE_TASKS.discard)

    entry.idle_handle = loop.call_later(_SESSION_WEBSOCKET_CACHE_TTL_SECONDS, _expire)


class OpenAICodexResponsesProvider(BaseProvider):
    name = "openai-codex"
    thinking_levels: list[str] = DEFAULT_THINKING_LEVELS

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
            raise RuntimeError(
                "Not logged in to OpenAI. Use /login to authenticate. "
                "If you are using an OpenAI-compatible API key instead, select a supported "
                "provider/model such as deepseek/deepseek-v4 and set it as the default in "
                "~/.config/kon/config.toml."
            )

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

    def _build_tools(self, tools: list[ToolDefinition] | None) -> list[dict[str, Any]]:
        if not tools:
            return []
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": None,
            }
            for tool in tools
        ]

    def _uses_responses_lite(self) -> bool:
        model = get_model(self.config.model, self.config.provider or self.name)
        return bool(model and model.uses_responses_lite)

    def _responses_lite_client_metadata(self) -> dict[str, str] | None:
        if not self._uses_responses_lite():
            return None
        return {_RESPONSES_LITE_WS_METADATA_KEY: "true"}

    def _strip_responses_lite_image_details(self, items: list[dict[str, Any]]) -> None:
        for item in items:
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "input_image":
                        part.pop("detail", None)
            output = item.get("output")
            if isinstance(output, list):
                for part in output:
                    if isinstance(part, dict) and part.get("type") == "input_image":
                        part.pop("detail", None)

    def _build_websocket_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = {"type": "response.create", **body}
        client_metadata = self._responses_lite_client_metadata()
        if client_metadata:
            existing_metadata = payload.get("client_metadata")
            if isinstance(existing_metadata, dict):
                payload["client_metadata"] = {**existing_metadata, **client_metadata}
            else:
                payload["client_metadata"] = client_metadata
        return payload

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
        uses_responses_lite = self._uses_responses_lite()
        instructions = system_prompt or "You are a helpful assistant."
        input_items = self._build_input(messages, None)
        tool_payload = self._build_tools(tools)

        if uses_responses_lite:
            self._strip_responses_lite_image_details(input_items)
            prefix: list[dict[str, Any]] = [
                {"type": "additional_tools", "role": "developer", "tools": tool_payload}
            ]
            if instructions:
                prefix.append(
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": instructions}],
                    }
                )
            input_items = [*prefix, *input_items]

        body: dict[str, Any] = {
            "model": self.config.model,
            "store": False,
            "stream": True,
            "input": input_items,
            "include": ["reasoning.encrypted_content"],
            "text": {"verbosity": "low"},
            "tool_choice": "auto",
            "parallel_tool_calls": not uses_responses_lite,
        }
        if not uses_responses_lite:
            body["instructions"] = instructions

        prompt_cache_key = _clamp_prompt_cache_key(self.config.session_id)
        if prompt_cache_key:
            body["prompt_cache_key"] = prompt_cache_key

        if tool_payload and not uses_responses_lite:
            body["tools"] = tool_payload

        effort = self.config.thinking_level
        if effort and effort != "none":
            if effort == "minimal":
                effort = "low"
            elif effort == "ultra":
                effort = "max"
            body["reasoning"] = {"effort": effort, "summary": "auto"}
            if uses_responses_lite:
                body["reasoning"]["context"] = "all_turns"

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
            "version": _CODEX_PROTOCOL_VERSION,
            "accept": "text/event-stream",
            "content-type": "application/json",
            "User-Agent": "kon",
        }
        if self._uses_responses_lite():
            headers[_RESPONSES_LITE_HEADER] = "true"
        if self.config.session_id:
            headers["session-id"] = self.config.session_id
            headers["thread-id"] = self.config.session_id
            headers["x-client-request-id"] = self.config.session_id
        return headers

    def _build_websocket_headers(self, token: str, account_id: str) -> dict[str, str]:
        request_id = self.config.session_id or str(uuid.uuid4())
        return {
            "Authorization": f"Bearer {token}",
            "chatgpt-account-id": account_id,
            "OpenAI-Beta": _OPENAI_BETA_RESPONSES_WEBSOCKETS,
            "originator": "kon",
            "version": _CODEX_PROTOCOL_VERSION,
            "User-Agent": "kon",
            "x-client-request-id": request_id,
            "session-id": request_id,
            "thread-id": request_id,
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
                    body,
                    self._build_websocket_headers(token, account_id),
                    session_id=session_id,
                    on_event=lambda _event: None,
                )
                async for part in self._process_codex_events(websocket_events, llm_stream):
                    emitted = True
                    yield part
                return
            except CodexNonTransportError as e:
                yield StreamError(error=_format_provider_error(e))
                return
            except (CodexTransportError, aiohttp.ClientError, OSError, TimeoutError) as e:
                if session_id:
                    _WS_FALLBACK_SESSIONS.add(session_id)
                if emitted:
                    yield StreamError(error=_format_provider_error(e))
                    return

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
                try:
                    response = await session.post(self._resolve_url(), headers=headers, json=body)
                except (aiohttp.ClientError, OSError, TimeoutError) as e:
                    last_error = _format_provider_error(e)
                    if attempt < _MAX_RETRIES:
                        delay = _BASE_DELAY_MS * (2**attempt) / 1000
                        await asyncio.sleep(delay)
                        continue
                    raise CodexTransportError(last_error) from e

                if response.status < 400:
                    break
                error_text = await response.text()
                last_error = f"Codex API error ({response.status}): {error_text}"
                if attempt < _MAX_RETRIES and _is_retryable_response_error(
                    response.status, error_text
                ):
                    delay = _retry_delay_seconds(response, attempt)
                    await asyncio.sleep(delay)
                    continue
                raise CodexNonTransportError(last_error, status=response.status)

            if response is None or response.status >= 400:
                raise CodexNonTransportError(last_error or "Codex request failed after retries")

            async for event in self._parse_sse(response):
                yield event

    async def _new_websocket_connection(
        self, headers: dict[str, str]
    ) -> _CachedWebSocketConnection:
        timeout = aiohttp.ClientTimeout(
            sock_connect=_CONNECT_TIMEOUT_SECONDS, sock_read=kon_config.llm.request_timeout_seconds
        )
        ws_timeout = aiohttp.ClientWSTimeout(ws_receive=kon_config.llm.request_timeout_seconds)  # type: ignore[call-arg]
        session = aiohttp.ClientSession(timeout=timeout)
        try:
            ws = await session.ws_connect(
                self._resolve_websocket_url(),
                headers=_websocket_connect_headers(headers),
                heartbeat=20,
                timeout=ws_timeout,
            )
        except asyncio.CancelledError:
            await session.close()
            raise
        except Exception:
            await session.close()
            raise
        return _CachedWebSocketConnection(session=session, ws=ws, busy=True)

    async def _acquire_websocket(
        self, headers: dict[str, str], session_id: str | None
    ) -> tuple[_CachedWebSocketConnection, bool, Callable[..., Awaitable[None]]]:
        if not session_id:
            entry = await self._new_websocket_connection(headers)

            async def release(*, keep: bool = False) -> None:
                await _close_websocket_entry(entry)

            return entry, False, release

        cached = _WS_SESSION_CACHE.get(session_id)
        if cached:
            if cached.idle_handle:
                cached.idle_handle.cancel()
                cached.idle_handle = None
            if not cached.busy and _is_websocket_reusable(cached):
                cached.busy = True

                async def release_cached(*, keep: bool = True) -> None:
                    if not keep or not _is_websocket_reusable(cached):
                        if _WS_SESSION_CACHE.get(session_id) is cached:
                            _WS_SESSION_CACHE.pop(session_id, None)
                        await _close_websocket_entry(cached)
                        return
                    cached.busy = False
                    _schedule_websocket_expiry(session_id, cached)

                return cached, True, release_cached

            if cached.busy:
                entry = await self._new_websocket_connection(headers)

                async def release_uncached(*, keep: bool = False) -> None:
                    await _close_websocket_entry(entry)

                return entry, False, release_uncached

            if not _is_websocket_reusable(cached):
                _WS_SESSION_CACHE.pop(session_id, None)
                await _close_websocket_entry(cached)

        entry = await self._new_websocket_connection(headers)
        _WS_SESSION_CACHE[session_id] = entry

        async def release_new(*, keep: bool = True) -> None:
            if not keep or not _is_websocket_reusable(entry):
                if _WS_SESSION_CACHE.get(session_id) is entry:
                    _WS_SESSION_CACHE.pop(session_id, None)
                await _close_websocket_entry(entry)
                return
            entry.busy = False
            _schedule_websocket_expiry(session_id, entry)

        return entry, False, release_new

    def _response_item_for_cache(self, item: dict[str, Any]) -> dict[str, Any] | None:
        item_type = item.get("type")
        if item_type == "message":
            content: list[dict[str, Any]] = []
            raw_content = item.get("content")
            if isinstance(raw_content, list):
                for part in raw_content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "output_text":
                        text = part.get("text")
                        if isinstance(text, str):
                            content.append(
                                {"type": "output_text", "text": text, "annotations": []}
                            )
                    elif part.get("type") == "refusal":
                        refusal = part.get("refusal")
                        if isinstance(refusal, str):
                            content.append({"type": "refusal", "refusal": refusal})
            if not content:
                return None
            return {
                "type": "message",
                "role": "assistant",
                "content": content,
                "status": "completed",
            }

        if item_type == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            if not isinstance(call_id, str) or not isinstance(name, str):
                return None
            arguments = item.get("arguments")
            if isinstance(arguments, str) and arguments:
                with contextlib.suppress(json.JSONDecodeError):
                    arguments = json.dumps(json.loads(arguments))
            return {
                "type": "function_call",
                "id": item.get("id") if isinstance(item.get("id"), str) else None,
                "call_id": call_id,
                "name": name,
                "arguments": arguments if isinstance(arguments, str) else "",
            }

        return None

    async def _stream_websocket_events(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
        session_id: str | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        entry, _, release = await self._acquire_websocket(headers, session_id)
        request_body = _build_cached_websocket_request_body(entry, body)
        keep_connection = True
        response_id: str | None = None
        response_items: list[dict[str, Any]] = []

        try:
            await entry.ws.send_json(self._build_websocket_payload(request_body))
            saw_completion = False
            async for msg in entry.ws:
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
                    if on_event:
                        on_event(event)
                    event_type = event.get("type")
                    if event_type == "response.created":
                        response_obj = event.get("response")
                        if isinstance(response_obj, dict) and isinstance(
                            response_obj.get("id"), str
                        ):
                            response_id = response_obj["id"]
                    elif event_type == "response.output_item.done":
                        item = event.get("item")
                        if isinstance(item, dict):
                            response_item = self._response_item_for_cache(item)
                            if response_item is not None:
                                response_items.append(response_item)
                    if event_type in {
                        "response.completed",
                        "response.done",
                        "response.incomplete",
                    }:
                        saw_completion = True
                        response_obj = event.get("response")
                        if isinstance(response_obj, dict) and isinstance(
                            response_obj.get("id"), str
                        ):
                            response_id = response_obj["id"]
                    yield event
                    if saw_completion:
                        if response_id and response_items:
                            entry.continuation = _CachedWebSocketContinuation(
                                last_request_body=body,
                                last_response_id=response_id,
                                last_response_items=response_items,
                            )
                        return
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    error = entry.ws.exception()
                    raise CodexTransportError(str(error) if error else "WebSocket error")

            if not saw_completion:
                raise CodexTransportError("WebSocket stream closed before response.completed")
        except asyncio.CancelledError:
            entry.continuation = None
            keep_connection = False
            raise
        except Exception:
            entry.continuation = None
            keep_connection = False
            raise
        finally:
            await release(keep=keep_connection)

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
