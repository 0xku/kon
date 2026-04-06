import os
from typing import Any, cast
from unittest.mock import patch

import pytest
from anthropic.types import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    InputJSONDelta,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageStopEvent,
    ThinkingConfigEnabledParam,
    ToolUseBlock,
)

from kon.core.types import (
    AssistantMessage,
    StopReason,
    StreamDone,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolCallDelta,
    ToolCallStart,
    UserMessage,
)
from kon.llm.base import LLMStream, ProviderConfig
from kon.llm.providers.anthropic import AnthropicProvider, supports_adaptive_thinking


@pytest.fixture
def anthropic_provider() -> AnthropicProvider:
    # Avoid constructing the real SDK client; conversion helpers don't need it.
    return AnthropicProvider.__new__(AnthropicProvider)


def test_convert_assistant_message_drops_unsigned_thinking(anthropic_provider: AnthropicProvider):
    messages = [
        UserMessage(content="hi"),
        AssistantMessage(content=[ThinkingContent(thinking="partial reasoning", signature=None)]),
        UserMessage(content="next"),
    ]

    converted = anthropic_provider._convert_messages(messages)

    # Assistant message with only unsigned thinking should be dropped entirely.
    assert len(converted) == 2
    assert converted[0]["role"] == "user"
    assert converted[1]["role"] == "user"


def test_convert_assistant_message_keeps_signed_thinking(anthropic_provider: AnthropicProvider):
    messages = [
        UserMessage(content="hi"),
        AssistantMessage(
            content=[
                ThinkingContent(thinking="valid reasoning", signature="sig_123"),
                TextContent(text="result"),
                ToolCall(id="tool_1", name="read", arguments={"path": "a.txt"}),
            ]
        ),
    ]

    converted = anthropic_provider._convert_messages(messages)

    assert len(converted) == 2
    assert converted[1]["role"] == "assistant"
    assistant_content = converted[1]["content"]
    assert isinstance(assistant_content, list)

    assert assistant_content[0] == {
        "type": "thinking",
        "thinking": "valid reasoning",
        "signature": "sig_123",
    }
    assert assistant_content[1] == {"type": "text", "text": "result"}
    assert assistant_content[2] == {
        "type": "tool_use",
        "id": "tool_1",
        "name": "read",
        "input": {"path": "a.txt"},
    }


def test_supports_adaptive_thinking_detection():
    assert supports_adaptive_thinking("claude-opus-4.6")
    assert supports_adaptive_thinking("claude-opus-4-6")
    assert supports_adaptive_thinking("claude-sonnet-4.6")
    assert supports_adaptive_thinking("claude-sonnet-4-6")
    assert not supports_adaptive_thinking("claude-3-7-sonnet")


@pytest.mark.asyncio
async def test_process_stream_uses_tool_use_input_as_initial_arguments():
    provider = AnthropicProvider.__new__(AnthropicProvider)
    llm_stream = LLMStream()

    async def response_iter():
        yield ContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=ToolUseBlock(
                type="tool_use",
                id="tool_1",
                name="write",
                input={"path": "/tmp/test.txt", "content": "hello"},
            ),
        )
        yield MessageDeltaEvent(
            type="message_delta",
            delta=cast(Any, {"stop_reason": "tool_use", "stop_sequence": None}),
            usage=MessageDeltaUsage(output_tokens=1),
        )
        yield MessageStopEvent(type="message_stop")

    parts = [part async for part in provider._process_stream(response_iter(), llm_stream)]

    assert isinstance(parts[0], ToolCallStart)
    assert parts[0].arguments == {"path": "/tmp/test.txt", "content": "hello"}
    assert isinstance(parts[-1], StreamDone)
    assert parts[-1].stop_reason == StopReason.TOOL_USE


@pytest.mark.asyncio
async def test_process_stream_emits_tool_delta_for_input_json_delta():
    provider = AnthropicProvider.__new__(AnthropicProvider)
    llm_stream = LLMStream()

    async def response_iter():
        yield ContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=ToolUseBlock(type="tool_use", id="tool_1", name="write", input={}),
        )
        yield ContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=InputJSONDelta(type="input_json_delta", partial_json='{"path":"/tmp/test.txt"}'),
        )
        yield MessageDeltaEvent(
            type="message_delta",
            delta=cast(Any, {"stop_reason": "tool_use", "stop_sequence": None}),
            usage=MessageDeltaUsage(output_tokens=1),
        )
        yield MessageStopEvent(type="message_stop")

    parts = [part async for part in provider._process_stream(response_iter(), llm_stream)]

    tool_delta = next(part for part in parts if isinstance(part, ToolCallDelta))
    assert tool_delta.arguments_delta == '{"path":"/tmp/test.txt"}'


class _EmptyAsyncIterator:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _DummyStreamContext:
    async def __aenter__(self):
        return _EmptyAsyncIterator()


class _DummyMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return _DummyStreamContext()


@pytest.mark.asyncio
async def test_stream_uses_adaptive_thinking_for_claude_4_6():
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.config = ProviderConfig(model="claude-sonnet-4.6", thinking_level="xhigh")
    dummy_messages = _DummyMessages()
    provider._client = cast(Any, type("DummyClient", (), {"messages": dummy_messages})())

    stream = await provider._stream_impl(messages=[])
    async for _ in stream:
        pass

    kwargs = dummy_messages.calls[0]
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "max"}
    assert "temperature" not in kwargs


@pytest.mark.asyncio
async def test_stream_uses_budget_thinking_for_non_adaptive_models():
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.config = ProviderConfig(model="claude-3-7-sonnet", thinking_level="high")
    dummy_messages = _DummyMessages()
    provider._client = cast(Any, type("DummyClient", (), {"messages": dummy_messages})())

    stream = await provider._stream_impl(messages=[])
    async for _ in stream:
        pass

    kwargs = dummy_messages.calls[0]
    assert kwargs["thinking"] == ThinkingConfigEnabledParam(type="enabled", budget_tokens=8192)


def test_anthropic_provider_uses_placeholder_for_local_auto_auth_mode():
    with patch.dict(os.environ, {}, clear=True):
        provider = AnthropicProvider(
            ProviderConfig(
                model="claude-sonnet-4.6",
                base_url="http://127.0.0.1:8001",
                anthropic_compat_auth_mode="auto",
            )
        )
        assert provider._client.api_key == "kon-local"


def test_anthropic_provider_requires_key_for_remote_required_mode():
    with (
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(ValueError, match="No API key found"),
    ):
        AnthropicProvider(
            ProviderConfig(
                model="claude-sonnet-4.6",
                base_url="https://api.anthropic.com",
                anthropic_compat_auth_mode="required",
            )
        )
