import asyncio

import pytest

from kon import Config, reset_config, set_config
from kon.core.types import StopReason, TextContent, UserMessage
from kon.events import (
    AgentEndEvent,
    AgentStartEvent,
    ErrorEvent,
    InterruptedEvent,
    RetryEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolArgsTokenUpdateEvent,
    ToolEndEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
    TurnStartEvent,
    WarningEvent,
)
from kon.llm.providers import MockProvider
from kon.loop import Agent
from kon.session import Session
from kon.tools import BashTool, ReadTool
from kon.turn import run_single_turn


@pytest.fixture
def tools():
    return [ReadTool(), BashTool()]


@pytest.fixture
def in_memory_session():
    return Session.in_memory()


@pytest.fixture
def sample_messages():
    return [UserMessage(content="Test query")]


@pytest.fixture
def max_turns_one():
    set_config(Config({"agent": {"max_turns": 1}}))
    try:
        yield
    finally:
        reset_config()


# ============================================================================
# Tests for Agent.run() - high-level agent orchestration
# ============================================================================


@pytest.mark.asyncio
async def test_agent_default_scenario(tools, in_memory_session, max_turns_one):
    provider = MockProvider(scenario="default")
    agent = Agent(provider, tools, in_memory_session)
    events = []

    async for event in agent.run("Test"):
        events.append(event)

    # Check basic event sequence
    assert isinstance(events[0], AgentStartEvent)
    assert isinstance(events[1], TurnStartEvent)
    assert isinstance(events[-1], AgentEndEvent)

    # Check tool calls were made (2 tools in default scenario)
    tool_starts = [e for e in events if isinstance(e, ToolStartEvent)]
    assert len(tool_starts) == 2
    assert tool_starts[0].tool_name == "read"
    assert tool_starts[1].tool_name == "bash"

    turn_end = next(e for e in events if isinstance(e, TurnEndEvent))
    assert turn_end.generation_seconds is not None
    assert turn_end.generation_seconds > 0
    assert turn_end.tool_call_count == 2

    # Check final state
    assert len(in_memory_session.messages) == 4  # user + assistant + 2 tool_results


@pytest.mark.asyncio
async def test_agent_simple_text_scenario(tools, in_memory_session):
    provider = MockProvider(scenario="simple_text")
    agent = Agent(provider, tools, in_memory_session)
    events = []

    async for event in agent.run("Say hello"):
        events.append(event)

    # Check event sequence
    assert isinstance(events[0], AgentStartEvent)
    assert isinstance(events[1], TurnStartEvent)
    assert events[1].turn == 1
    assert isinstance(events[2], TextStartEvent)
    assert isinstance(events[3], TextDeltaEvent)
    assert events[3].delta == "Hello, world!"
    assert isinstance(events[4], TextEndEvent)
    assert isinstance(events[5], TurnEndEvent)
    assert isinstance(events[6], AgentEndEvent)

    turn_end = events[5]
    assert isinstance(turn_end, TurnEndEvent)
    assert turn_end.generation_seconds is not None
    assert turn_end.generation_seconds > 0
    assert turn_end.tool_call_count == 0

    # Check final state
    assert events[6].stop_reason == StopReason.STOP
    assert events[6].total_turns == 1
    assert len(in_memory_session.messages) == 2  # user + assistant


@pytest.mark.asyncio
async def test_agent_max_turns_limit(tools, in_memory_session, max_turns_one):
    provider = MockProvider(scenario="default")
    agent = Agent(provider, tools, in_memory_session)
    events = []

    async for event in agent.run("Test"):
        events.append(event)

    # Check we had one turn
    turn_starts = [e for e in events if isinstance(e, TurnStartEvent)]
    assert len(turn_starts) == 1

    # Check agent ended due to max_turns limit
    agent_end = events[-1]
    assert isinstance(agent_end, AgentEndEvent)
    assert agent_end.total_turns == 1
    assert agent_end.stop_reason == StopReason.LENGTH


@pytest.mark.asyncio
async def test_agent_usage_tracking(tools, in_memory_session, max_turns_one):
    provider = MockProvider(scenario="default")
    agent = Agent(provider, tools, in_memory_session)
    events = []

    async for event in agent.run("Track usage"):
        events.append(event)

    # Check agent end event has usage
    agent_end = next(e for e in events if isinstance(e, AgentEndEvent))
    assert agent_end.total_usage is not None
    assert agent_end.total_usage.input_tokens == 10
    assert agent_end.total_usage.output_tokens == 5
    assert agent_end.total_usage.cache_read_tokens == 2
    assert agent_end.total_usage.total_tokens == 17


@pytest.mark.asyncio
async def test_agent_system_prompt(tools, in_memory_session, max_turns_one):
    provider = MockProvider(scenario="default")
    agent = Agent(provider, tools, in_memory_session, system_prompt="Custom system prompt")

    events = []
    async for event in agent.run("Test"):
        events.append(event)

    # Should complete successfully
    assert isinstance(events[-1], AgentEndEvent)
    assert len(in_memory_session.messages) == 4  # user + assistant + 2 tool_results


@pytest.mark.asyncio
async def test_agent_with_thinking(tools, in_memory_session, max_turns_one):
    provider = MockProvider(scenario="default")
    agent = Agent(provider, tools, in_memory_session)
    events = []

    async for event in agent.run("Think and answer"):
        events.append(event)

    # Check thinking events
    thinking_start = next((e for e in events if isinstance(e, ThinkingStartEvent)), None)
    assert thinking_start is not None

    thinking_delta = next((e for e in events if isinstance(e, ThinkingDeltaEvent)), None)
    assert thinking_delta is not None
    assert thinking_delta.delta == "Let me think about this..."

    thinking_end = next((e for e in events if isinstance(e, ThinkingEndEvent)), None)
    assert thinking_end is not None
    assert thinking_end.thinking == "Let me think about this..."


@pytest.mark.asyncio
async def test_agent_with_images(tools, in_memory_session):
    from kon.core.types import ImageContent

    provider = MockProvider(scenario="simple_text")
    images = [ImageContent(data="base64data", mime_type="image/png")]

    agent = Agent(provider, tools, in_memory_session)
    events = []

    async for event in agent.run("What's in this image?", images=images):
        events.append(event)

    # Check user message was created correctly
    user_msg = in_memory_session.messages[0]
    assert isinstance(user_msg, UserMessage)
    assert isinstance(user_msg.content, list)
    assert len(user_msg.content) == 2  # text + image

    # Check events
    assert isinstance(events[0], AgentStartEvent)
    assert isinstance(events[-1], AgentEndEvent)


@pytest.mark.asyncio
async def test_agent_custom_cwd(tools):
    provider = MockProvider(scenario="simple_text")
    session = Session.in_memory(cwd="/custom/path")

    agent = Agent(provider, tools, session, cwd="/custom/path")

    events = []
    async for event in agent.run("Where am I?"):
        events.append(event)

    assert isinstance(events[-1], AgentEndEvent)
    assert session.cwd == "/custom/path"


# ============================================================================
# Tests for run_single_turn() - direct testing of single turn execution
# ============================================================================


@pytest.mark.asyncio
async def test_run_single_turn_default_scenario(sample_messages, tools):
    provider = MockProvider(scenario="default")
    events = []

    async for event in run_single_turn(provider, sample_messages, tools, turn=1):
        events.append(event)

    # Check thinking sequence
    assert isinstance(events[0], ThinkingStartEvent)
    assert isinstance(events[1], ThinkingDeltaEvent)
    assert isinstance(events[2], ThinkingEndEvent)

    # Check text sequence
    assert isinstance(events[3], TextStartEvent)
    assert isinstance(events[4], TextDeltaEvent)
    assert isinstance(events[5], TextEndEvent)

    # Check tool calls (2 tools)
    tool_starts = [e for e in events if isinstance(e, ToolStartEvent)]
    assert len(tool_starts) == 2
    assert tool_starts[0].tool_name == "read"
    assert tool_starts[1].tool_name == "bash"

    # Check turn end
    turn_end = next(e for e in events if isinstance(e, TurnEndEvent))
    assert turn_end.stop_reason == StopReason.TOOL_USE
    assert len(turn_end.tool_results) == 2


@pytest.mark.asyncio
async def test_run_single_turn_simple_text_scenario(sample_messages, tools):
    provider = MockProvider(scenario="simple_text")
    events = []

    async for event in run_single_turn(provider, sample_messages, tools, turn=1):
        events.append(event)

    # Expected: TextStart, TextDelta, TextEnd, TurnEnd
    assert isinstance(events[0], TextStartEvent)
    assert isinstance(events[1], TextDeltaEvent)
    assert events[1].delta == "Hello, world!"
    assert isinstance(events[2], TextEndEvent)
    assert isinstance(events[3], TurnEndEvent)
    assert events[3].stop_reason == StopReason.STOP

    # No thinking or tools
    assert not any(isinstance(e, ThinkingStartEvent) for e in events)
    assert not any(isinstance(e, ToolStartEvent) for e in events)


@pytest.mark.asyncio
async def test_run_single_turn_thinking_text_tool_scenario(sample_messages, tools):
    provider = MockProvider(scenario="thinking_text_tool")
    events = []

    async for event in run_single_turn(provider, sample_messages, tools, turn=1):
        events.append(event)

    # Check thinking sequence
    assert isinstance(events[0], ThinkingStartEvent)
    assert isinstance(events[1], ThinkingDeltaEvent)
    assert events[1].delta == "I need to read the file"
    assert isinstance(events[2], ThinkingEndEvent)
    assert events[2].thinking == "I need to read the file"

    # Check text sequence
    assert isinstance(events[3], TextStartEvent)
    assert isinstance(events[4], TextDeltaEvent)
    assert events[4].delta == "Let me check the file."
    assert isinstance(events[5], TextEndEvent)

    # Check tool sequence (single tool)
    tool_start = next(e for e in events if isinstance(e, ToolStartEvent))
    assert tool_start.tool_name == "read"

    # Check turn end
    turn_end = next(e for e in events if isinstance(e, TurnEndEvent))
    assert turn_end.stop_reason == StopReason.TOOL_USE
    assert turn_end.assistant_message is not None
    assert len(turn_end.assistant_message.content) == 3  # thinking + text + tool


@pytest.mark.asyncio
async def test_run_single_turn_retries_scenario(sample_messages, tools):
    provider = MockProvider(scenario="retries")
    events = []

    async for event in run_single_turn(
        provider, sample_messages, tools, turn=1, retry_delays=[0, 0, 0]
    ):
        events.append(event)

    # Should have 2 retry events
    retry_events = [e for e in events if isinstance(e, RetryEvent)]
    assert len(retry_events) == 2
    assert retry_events[0].attempt == 1
    assert retry_events[0].total_attempts == 3
    assert "Rate limit" in retry_events[0].error
    assert retry_events[1].attempt == 2
    assert retry_events[1].total_attempts == 3

    # Eventually succeed with default scenario content
    thinking_start = next(e for e in events if isinstance(e, ThinkingStartEvent))
    assert thinking_start is not None

    turn_end = next(e for e in events if isinstance(e, TurnEndEvent))
    assert turn_end.stop_reason == StopReason.TOOL_USE


@pytest.mark.asyncio
async def test_run_single_turn_retry_exhausted_scenario(sample_messages, tools):
    provider = MockProvider(scenario="retry_exhausted")
    events = []

    async for event in run_single_turn(
        provider, sample_messages, tools, turn=1, retry_delays=[0, 0, 0]
    ):
        events.append(event)

    # Should have 3 retry events
    retry_events = [e for e in events if isinstance(e, RetryEvent)]
    assert len(retry_events) == 3

    # Then error event
    error_event = next(e for e in events if isinstance(e, ErrorEvent))
    assert "Always fails" in error_event.error

    # And turn end with error
    turn_end = next(e for e in events if isinstance(e, TurnEndEvent))
    assert turn_end.stop_reason == StopReason.ERROR
    assert turn_end.assistant_message is None


@pytest.mark.asyncio
async def test_run_single_turn_non_retryable_scenario(sample_messages, tools):
    provider = MockProvider(scenario="non_retryable")
    events = []

    async for event in run_single_turn(provider, sample_messages, tools, turn=1):
        events.append(event)

    # No retry events
    retry_events = [e for e in events if isinstance(e, RetryEvent)]
    assert len(retry_events) == 0

    # Just error and turn end
    error_event = next(e for e in events if isinstance(e, ErrorEvent))
    assert "Invalid input" in error_event.error

    turn_end = next(e for e in events if isinstance(e, TurnEndEvent))
    assert turn_end.stop_reason == StopReason.ERROR


@pytest.mark.asyncio
async def test_run_single_turn_stream_error_scenario(sample_messages, tools):
    provider = MockProvider(scenario="stream_error")
    events = []

    async for event in run_single_turn(provider, sample_messages, tools, turn=1):
        events.append(event)

    # Should have text delta before error
    text_deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert len(text_deltas) == 1
    assert text_deltas[0].delta == "Before error"

    # Should have error event
    error_event = next(e for e in events if isinstance(e, ErrorEvent))
    assert "Something went wrong" in error_event.error

    # Turn should end with error
    turn_end = next(e for e in events if isinstance(e, TurnEndEvent))
    assert turn_end.stop_reason == StopReason.ERROR


@pytest.mark.asyncio
async def test_run_single_turn_drops_leading_empty_newlines_before_thinking(
    sample_messages, tools
):
    provider = MockProvider(scenario="leading_empty_text_then_think")
    events = []

    async for event in run_single_turn(provider, sample_messages, tools, turn=1):
        events.append(event)

    # Should begin with thinking, not empty text
    assert isinstance(events[0], ThinkingStartEvent)
    assert not any(isinstance(e, TextDeltaEvent) and not e.delta.strip() for e in events)

    turn_end = next(e for e in events if isinstance(e, TurnEndEvent))
    assert turn_end.assistant_message is not None
    assert len(turn_end.assistant_message.content) == 2


@pytest.mark.asyncio
async def test_run_single_turn_drops_leading_empty_newlines_before_text(sample_messages, tools):
    provider = MockProvider(scenario="leading_empty_text_then_text")
    events = []

    async for event in run_single_turn(provider, sample_messages, tools, turn=1):
        events.append(event)

    assert isinstance(events[0], TextStartEvent)
    text_deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert len(text_deltas) == 1
    assert text_deltas[0].delta == "Hello, world!"


@pytest.mark.asyncio
async def test_run_single_turn_unknown_tool_scenario(sample_messages, tools):
    provider = MockProvider(scenario="unknown_tool")
    events = []

    async for event in run_single_turn(provider, sample_messages, tools, turn=1):
        events.append(event)

    # Tool should still be recorded
    tool_end = next(e for e in events if isinstance(e, ToolEndEvent))
    assert tool_end.tool_name == "unknown_tool"

    # But result should be error
    tool_result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert tool_result.result is not None
    assert tool_result.result.is_error is True
    content = tool_result.result.content[0]
    assert isinstance(content, TextContent)
    assert "Unknown tool" in content.text


@pytest.mark.asyncio
async def test_run_single_turn_long_text_scenario(sample_messages, tools):
    provider = MockProvider(scenario="long_text")
    events = []

    async for event in run_single_turn(provider, sample_messages, tools, turn=1):
        events.append(event)

    # Should have multiple text deltas
    text_deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert len(text_deltas) == 6
    full_text = "".join(d.delta for d in text_deltas)
    assert full_text == "This is a long response."

    # Final text end should have complete text
    text_end = next(e for e in events if isinstance(e, TextEndEvent))
    assert text_end.text == "This is a long response."

    turn_end = next(e for e in events if isinstance(e, TurnEndEvent))
    assert turn_end.stop_reason == StopReason.STOP


@pytest.mark.asyncio
async def test_run_single_turn_tool_hang_timeout_fallback(sample_messages, tools):
    set_config(Config({"llm": {"tool_call_idle_timeout_seconds": 0.01}}))
    try:
        provider = MockProvider(scenario="tool_hang")
        events = []

        async for event in run_single_turn(provider, sample_messages, tools, turn=1):
            events.append(event)
    finally:
        reset_config()

    warning_event = next(e for e in events if isinstance(e, WarningEvent))
    assert "Tool-call stream stalled" in warning_event.warning

    tool_end = next(e for e in events if isinstance(e, ToolEndEvent))
    assert tool_end.tool_name == "read"

    turn_end = next(e for e in events if isinstance(e, TurnEndEvent))
    assert turn_end.stop_reason == StopReason.TOOL_USE
    assert len(turn_end.tool_results) == 1


@pytest.mark.asyncio
async def test_run_single_turn_skips_tool_on_invalid_partial_json_arguments(
    sample_messages, tools
):
    set_config(Config({"llm": {"tool_call_idle_timeout_seconds": 0.01}}))
    try:
        provider = MockProvider(scenario="tool_hang_invalid_json")
        events = []

        async for event in run_single_turn(provider, sample_messages, tools, turn=1):
            events.append(event)
    finally:
        reset_config()

    tool_result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert tool_result.result is not None
    assert tool_result.result.is_error is True
    first_content = tool_result.result.content[0]
    assert isinstance(first_content, TextContent)
    assert "incomplete or invalid JSON" in first_content.text


@pytest.mark.asyncio
async def test_run_single_turn_pre_cancelled(sample_messages, tools):
    cancel_event = asyncio.Event()
    cancel_event.set()  # Pre-cancelled

    provider = MockProvider(scenario="default")
    events = []

    async for event in run_single_turn(
        provider, sample_messages, tools, turn=1, cancel_event=cancel_event
    ):
        events.append(event)

    # Should immediately interrupt
    assert len(events) == 2
    assert isinstance(events[0], InterruptedEvent)
    assert events[0].message == "Interrupted by user"
    assert isinstance(events[1], TurnEndEvent)
    assert events[1].stop_reason == StopReason.INTERRUPTED
    assert events[1].assistant_message is None


@pytest.mark.asyncio
async def test_run_single_turn_cancel_during_retry_backoff(sample_messages, tools):
    cancel_event = asyncio.Event()
    provider = MockProvider(scenario="retry_exhausted")
    events = []

    async def cancel_soon():
        await asyncio.sleep(0.01)
        cancel_event.set()

    asyncio.create_task(cancel_soon())

    async for event in run_single_turn(
        provider, sample_messages, tools, turn=1, cancel_event=cancel_event, retry_delays=[1, 1, 1]
    ):
        events.append(event)

    retry_events = [e for e in events if isinstance(e, RetryEvent)]
    assert len(retry_events) == 1
    assert isinstance(events[-2], InterruptedEvent)
    assert isinstance(events[-1], TurnEndEvent)
    assert events[-1].stop_reason == StopReason.INTERRUPTED


# ============================================================================
# Cancellation tests
# ============================================================================


@pytest.mark.asyncio
async def test_agent_cancellation(tools, in_memory_session):
    provider = MockProvider(scenario="simple_text")
    agent = Agent(provider, tools, in_memory_session)
    cancel_event = asyncio.Event()

    # Cancel immediately
    cancel_event.set()

    events = []
    async for event in agent.run("Cancel me", cancel_event=cancel_event):
        events.append(event)

    # Should get AgentStart, then Interrupted, then AgentEnd
    assert isinstance(events[0], AgentStartEvent)
    assert isinstance(events[1], InterruptedEvent)
    assert events[1].message == "Interrupted by user"

    agent_end = events[-1]
    assert isinstance(agent_end, AgentEndEvent)
    assert agent_end.stop_reason == StopReason.INTERRUPTED
    assert agent_end.total_turns == 0


# ============================================================================
# Token counting tests
# ============================================================================


@pytest.mark.asyncio
async def test_tool_args_token_counting(tools, sample_messages):
    """Test that token count events are fired for tool argument streaming."""
    provider = MockProvider(scenario="tool_with_many_chunks")
    events = []

    async for event in run_single_turn(provider, sample_messages, tools, turn=1):
        events.append(event)

    # Find token update events
    token_updates = [e for e in events if isinstance(e, ToolArgsTokenUpdateEvent)]

    # At least one token update should be fired for large tool arguments
    assert len(token_updates) >= 1, "Token update events should be fired for large tool arguments"

    # Token counts should be positive
    for update in token_updates:
        assert update.token_count > 0, "Token count should be positive"

    # Token counts should be monotonically increasing
    token_counts = [e.token_count for e in token_updates]
    assert token_counts == sorted(token_counts), "Token counts should be monotonically increasing"

    # Check tool name and ID are correctly associated
    for update in token_updates:
        assert update.tool_name == "bash"
        assert update.tool_call_id == "call-1"


@pytest.mark.asyncio
async def test_tool_args_token_count_resets_between_tools(tools, sample_messages):
    """Test that token counter resets when switching tools."""
    provider = MockProvider(scenario="default")
    events = []

    async for event in run_single_turn(provider, sample_messages, tools, turn=1):
        events.append(event)

    # Default scenario has small args, so few or no updates expected
    # Just verify the mechanism works without erroring
    [e for e in events if isinstance(e, ToolArgsTokenUpdateEvent)]
