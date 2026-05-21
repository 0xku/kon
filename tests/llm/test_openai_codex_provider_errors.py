from collections.abc import AsyncIterator
from typing import Any

import pytest

from kon.core.types import (
    StopReason,
    StreamDone,
    StreamError,
    TextPart,
    ToolCallDelta,
    ToolCallStart,
)
from kon.llm.base import LLMStream, ProviderConfig
from kon.llm.oauth.openai import OpenAICredentials
from kon.llm.providers import openai_codex_responses as codex_provider
from kon.llm.providers.openai_codex_responses import (
    _WS_FALLBACK_SESSIONS,
    CodexTransportError,
    OpenAICodexResponsesProvider,
    _format_provider_error,
)


@pytest.fixture(autouse=True)
def _clear_ws_fallback_sessions():
    _WS_FALLBACK_SESSIONS.clear()
    yield
    _WS_FALLBACK_SESSIONS.clear()


async def _async_iter(events: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    for event in events:
        yield event


def test_format_provider_error_preserves_non_empty_message():
    err = RuntimeError("boom")
    assert _format_provider_error(err) == "boom"


def test_format_provider_error_falls_back_for_empty_message():
    err = TimeoutError()
    message = _format_provider_error(err)
    assert "TimeoutError" in message
    assert "without an error message" in message


def test_resolve_websocket_url_uses_ws_scheme_and_codex_responses_path():
    provider = OpenAICodexResponsesProvider(
        ProviderConfig(base_url="https://chatgpt.com/backend-api", model="gpt-5.4")
    )
    assert provider._resolve_websocket_url() == "wss://chatgpt.com/backend-api/codex/responses"


def test_websocket_headers_use_beta_and_request_id():
    provider = OpenAICodexResponsesProvider(
        ProviderConfig(session_id="session-123", model="gpt-5.4")
    )
    headers = provider._build_websocket_headers("token", "account")
    assert headers["OpenAI-Beta"] == "responses_websockets=2026-02-06"
    assert headers["session_id"] == "session-123"
    assert headers["x-client-request-id"] == "session-123"
    assert "accept" not in headers
    assert "content-type" not in headers


@pytest.mark.asyncio
async def test_stream_impl_uses_valid_credentials_for_token_and_account(monkeypatch):
    provider = OpenAICodexResponsesProvider(ProviderConfig(model="gpt-5.5"))
    creds = OpenAICredentials(
        refresh="refresh",
        access="access-token",
        expires=9_999_999_999_999,
        account_id="account-id",
    )
    captured: dict[str, Any] = {}

    async def fake_get_valid_credentials() -> OpenAICredentials:
        return creds

    def fake_stream_codex(**kwargs):
        captured.update(kwargs)
        return _async_iter([])

    monkeypatch.setattr(codex_provider, "get_valid_openai_credentials", fake_get_valid_credentials)
    monkeypatch.setattr(provider, "_stream_codex", fake_stream_codex)

    stream = await provider._stream_impl([])

    assert isinstance(stream, LLMStream)
    assert captured["token"] == "access-token"
    assert captured["account_id"] == "account-id"


@pytest.mark.asyncio
async def test_stream_impl_raises_when_openai_credentials_are_invalid(monkeypatch):
    provider = OpenAICodexResponsesProvider(ProviderConfig(model="gpt-5.5"))

    async def fake_get_valid_credentials() -> None:
        return None

    monkeypatch.setattr(codex_provider, "get_valid_openai_credentials", fake_get_valid_credentials)

    with pytest.raises(RuntimeError, match="Not logged in to OpenAI"):
        await provider._stream_impl([])


@pytest.mark.asyncio
async def test_stream_falls_back_to_sse_when_websocket_fails_before_events(monkeypatch):
    provider = OpenAICodexResponsesProvider(
        ProviderConfig(session_id="session-fallback", model="gpt-5.4")
    )

    async def fail_websocket(*args, **kwargs):
        raise CodexTransportError("websocket unavailable")
        yield

    async def sse_events(*args, **kwargs):
        yield {"type": "response.output_text.delta", "delta": "ok"}
        yield {"type": "response.completed", "response": {"status": "completed"}}

    monkeypatch.setattr(provider, "_stream_websocket_events", fail_websocket)
    monkeypatch.setattr(provider, "_stream_sse_events", sse_events)

    parts = [
        part
        async for part in provider._stream_codex(
            token="token",
            account_id="account",
            messages=[],
            system_prompt=None,
            tools=None,
            temperature=None,
            max_tokens=None,
            llm_stream=LLMStream(),
        )
    ]

    assert isinstance(parts[0], TextPart)
    assert parts[0].text == "ok"
    assert isinstance(parts[1], StreamDone)
    assert "session-fallback" in _WS_FALLBACK_SESSIONS


@pytest.mark.asyncio
async def test_stream_emits_stream_error_and_skips_fallback_on_mid_stream_ws_failure(monkeypatch):
    provider = OpenAICodexResponsesProvider(
        ProviderConfig(session_id="session-mid", model="gpt-5.4")
    )

    async def ws_events(*args, **kwargs):
        yield {"type": "response.output_text.delta", "delta": "hi"}
        raise CodexTransportError("late failure")

    sse_calls: list[int] = []

    async def sse_events(*args, **kwargs):
        sse_calls.append(1)
        if False:
            yield

    monkeypatch.setattr(provider, "_stream_websocket_events", ws_events)
    monkeypatch.setattr(provider, "_stream_sse_events", sse_events)

    parts = [
        part
        async for part in provider._stream_codex(
            token="t",
            account_id="a",
            messages=[],
            system_prompt=None,
            tools=None,
            temperature=None,
            max_tokens=None,
            llm_stream=LLMStream(),
        )
    ]

    assert len(parts) == 2
    assert isinstance(parts[0], TextPart)
    assert parts[0].text == "hi"
    assert isinstance(parts[1], StreamError)
    assert "late failure" in parts[1].error
    assert sse_calls == []
    assert "session-mid" not in _WS_FALLBACK_SESSIONS


@pytest.mark.asyncio
async def test_stream_propagates_non_codex_exception_from_websocket_setup(monkeypatch):
    provider = OpenAICodexResponsesProvider(
        ProviderConfig(session_id="session-bug", model="gpt-5.4")
    )

    async def buggy_websocket(*args, **kwargs):
        raise KeyError("oops")
        yield

    def sse_events(*args, **kwargs):
        pytest.fail("SSE fallback should not be invoked")

    monkeypatch.setattr(provider, "_stream_websocket_events", buggy_websocket)
    monkeypatch.setattr(provider, "_stream_sse_events", sse_events)

    with pytest.raises(KeyError, match="oops"):
        async for _ in provider._stream_codex(
            token="t",
            account_id="a",
            messages=[],
            system_prompt=None,
            tools=None,
            temperature=None,
            max_tokens=None,
            llm_stream=LLMStream(),
        ):
            pass

    assert "session-bug" not in _WS_FALLBACK_SESSIONS


@pytest.mark.asyncio
async def test_process_codex_events_routes_parallel_function_call_deltas_by_item_id():
    events: list[dict[str, Any]] = [
        {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "item_A",
                "call_id": "call_A",
                "name": "tool_a",
            },
        },
        {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "item_B",
                "call_id": "call_B",
                "name": "tool_b",
            },
        },
        {"type": "response.function_call_arguments.delta", "item_id": "item_A", "delta": "{"},
        {"type": "response.function_call_arguments.delta", "item_id": "item_B", "delta": "{"},
        {"type": "response.function_call_arguments.delta", "item_id": "unknown", "delta": "BAD"},
        {"type": "response.function_call_arguments.delta", "item_id": "item_A", "delta": '"x":1}'},
        {"type": "response.function_call_arguments.delta", "item_id": "item_B", "delta": '"y":2}'},
        {"type": "response.completed", "response": {"status": "completed"}},
    ]

    provider = OpenAICodexResponsesProvider(ProviderConfig(model="gpt-5.4"))
    parts = [p async for p in provider._process_codex_events(_async_iter(events), LLMStream())]

    starts = [p for p in parts if isinstance(p, ToolCallStart)]
    deltas = [p for p in parts if isinstance(p, ToolCallDelta)]
    done = [p for p in parts if isinstance(p, StreamDone)]

    assert len(starts) == 2
    assert starts[0].index == 0 and starts[0].name == "tool_a"
    assert starts[1].index == 1 and starts[1].name == "tool_b"

    assert len(deltas) == 4
    a_args = "".join(d.arguments_delta for d in deltas if d.index == 0)
    b_args = "".join(d.arguments_delta for d in deltas if d.index == 1)
    assert a_args == '{"x":1}'
    assert b_args == '{"y":2}'

    assert len(done) == 1
    assert done[0].stop_reason == StopReason.TOOL_USE


@pytest.mark.asyncio
async def test_process_codex_events_done_reconciliation_appends_missing_suffix():
    events: list[dict[str, Any]] = [
        {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "item_A",
                "call_id": "call_A",
                "name": "tool_a",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "item_A",
            "delta": '{"cmd":"ec',
        },
        {
            "type": "response.function_call_arguments.done",
            "item_id": "item_A",
            "arguments": '{"cmd":"echo"}',
        },
        {"type": "response.completed", "response": {"status": "completed"}},
    ]

    provider = OpenAICodexResponsesProvider(ProviderConfig(model="gpt-5.4"))
    parts = [p async for p in provider._process_codex_events(_async_iter(events), LLMStream())]
    deltas = [p for p in parts if isinstance(p, ToolCallDelta)]

    assert len(deltas) == 2
    assert deltas[0].arguments_delta == '{"cmd":"ec'
    assert deltas[1].arguments_delta == 'ho"}'


@pytest.mark.asyncio
async def test_process_codex_events_output_item_done_reconciles_from_initial_arguments():
    events: list[dict[str, Any]] = [
        {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "item_A",
                "call_id": "call_A",
                "name": "tool_a",
                "arguments": '{"path":',
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "id": "item_A",
                "call_id": "call_A",
                "name": "tool_a",
                "arguments": '{"path":"/tmp"}',
            },
        },
        {"type": "response.completed", "response": {"status": "completed"}},
    ]

    provider = OpenAICodexResponsesProvider(ProviderConfig(model="gpt-5.4"))
    parts = [p async for p in provider._process_codex_events(_async_iter(events), LLMStream())]
    deltas = [p for p in parts if isinstance(p, ToolCallDelta)]

    assert len(deltas) == 2
    assert [d.index for d in deltas] == [0, 0]
    assert [d.replace for d in deltas] == [False, False]
    assert "".join(d.arguments_delta for d in deltas) == '{"path":"/tmp"}'


@pytest.mark.asyncio
async def test_process_codex_events_final_arguments_can_replace_partial_arguments():
    events: list[dict[str, Any]] = [
        {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "item_A",
                "call_id": "call_A",
                "name": "tool_a",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "item_A",
            "delta": '{"broken":',
        },
        {
            "type": "response.function_call_arguments.done",
            "item_id": "item_A",
            "arguments": '{"path":"/tmp"}',
        },
        {"type": "response.completed", "response": {"status": "completed"}},
    ]

    provider = OpenAICodexResponsesProvider(ProviderConfig(model="gpt-5.4"))
    parts = [p async for p in provider._process_codex_events(_async_iter(events), LLMStream())]
    deltas = [p for p in parts if isinstance(p, ToolCallDelta)]

    assert len(deltas) == 2
    assert deltas[0].arguments_delta == '{"broken":'
    assert deltas[0].replace is False
    assert deltas[1].arguments_delta == '{"path":"/tmp"}'
    assert deltas[1].replace is True


@pytest.mark.asyncio
async def test_incomplete_response_with_content_filter_maps_to_stop_reason_error():
    events: list[dict[str, Any]] = [
        {
            "type": "response.incomplete",
            "response": {
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
            },
        }
    ]

    provider = OpenAICodexResponsesProvider(ProviderConfig(model="gpt-5.4"))
    parts = [p async for p in provider._process_codex_events(_async_iter(events), LLMStream())]
    done = [p for p in parts if isinstance(p, StreamDone)]

    assert len(done) == 1
    assert done[0].stop_reason == StopReason.ERROR


def test_apply_response_metadata_preserves_zero_cache_write_tokens():
    provider = OpenAICodexResponsesProvider(ProviderConfig(model="gpt-5.4"))
    llm_stream = LLMStream()
    response_obj = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_write_tokens": 7,
            "input_tokens_details": {
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 9,
            },
        }
    }
    provider._apply_response_metadata(response_obj, llm_stream)
    assert llm_stream._usage is not None
    assert llm_stream._usage.cache_write_tokens == 0
