import pytest

from kon.core.types import AssistantMessage, TextContent, ThinkingContent, ToolCall, UserMessage
from kon.llm.providers.anthropic import AnthropicProvider


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
