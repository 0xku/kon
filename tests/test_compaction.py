from types import SimpleNamespace

import pytest

from kon.config import Config
from kon.core.compaction import is_overflow
from kon.core.types import (
    AssistantMessage,
    StopReason,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from kon.llm.providers import MockProvider
from kon.loop import Agent, AgentConfig
from kon.runtime import ConversationRuntime
from kon.session import CompactionEntry, Session
from kon.ui.commands import CommandsMixin

# ---------------------------------------------------------------------------
# is_overflow tests
# ---------------------------------------------------------------------------


class TestIsOverflow:
    def test_below_threshold(self):
        usage = Usage(input_tokens=100_000, output_tokens=5_000)
        assert not is_overflow(
            usage, context_window=200_000, max_output_tokens=16_000, buffer_tokens=20_000
        )

    def test_at_threshold(self):
        # usable = 200_000 - min(20_000, 16_000) = 200_000 - 16_000 = 184_000
        usage = Usage(input_tokens=180_000, output_tokens=4_000)
        assert is_overflow(
            usage, context_window=200_000, max_output_tokens=16_000, buffer_tokens=20_000
        )

    def test_above_threshold(self):
        usage = Usage(input_tokens=190_000, output_tokens=5_000)
        assert is_overflow(
            usage, context_window=200_000, max_output_tokens=16_000, buffer_tokens=20_000
        )

    def test_cache_tokens_counted(self):
        # total = 100k + 5k + 50k + 30k = 185k, usable = 184k -> overflow
        usage = Usage(
            input_tokens=100_000,
            output_tokens=5_000,
            cache_read_tokens=50_000,
            cache_write_tokens=30_000,
        )
        assert is_overflow(
            usage, context_window=200_000, max_output_tokens=16_000, buffer_tokens=20_000
        )

    def test_buffer_smaller_than_max_output(self):
        # reserved = min(10_000, 32_000) = 10_000, usable = 200_000 - 10_000 = 190_000
        usage = Usage(input_tokens=185_000, output_tokens=5_000)
        assert is_overflow(
            usage, context_window=200_000, max_output_tokens=32_000, buffer_tokens=10_000
        )

    def test_buffer_larger_than_max_output(self):
        # reserved = min(20_000, 8_000) = 8_000, usable = 128_000 - 8_000 = 120_000
        usage = Usage(input_tokens=115_000, output_tokens=5_000)
        assert is_overflow(
            usage, context_window=128_000, max_output_tokens=8_000, buffer_tokens=20_000
        )

    def test_zero_usage_no_overflow(self):
        usage = Usage()
        assert not is_overflow(
            usage, context_window=200_000, max_output_tokens=16_000, buffer_tokens=20_000
        )

    def test_exact_boundary(self):
        # usable = 200_000 - min(20_000, 16_000) = 184_000
        # exactly at boundary -> overflow
        usage = Usage(input_tokens=184_000)
        assert is_overflow(
            usage, context_window=200_000, max_output_tokens=16_000, buffer_tokens=20_000
        )

    def test_one_below_boundary(self):
        usage = Usage(input_tokens=183_999)
        assert not is_overflow(
            usage, context_window=200_000, max_output_tokens=16_000, buffer_tokens=20_000
        )


# ---------------------------------------------------------------------------
# session.messages compacted view tests
# ---------------------------------------------------------------------------


class TestSessionCompactedMessages:
    def test_no_compaction_returns_all_messages(self):
        session = Session.in_memory()
        session.append_message(UserMessage(content="Hello"))
        session.append_message(AssistantMessage(content=[TextContent(text="Hi")]))
        session.append_message(UserMessage(content="How are you?"))

        assert len(session.messages) == 3
        assert session.messages[0].role == "user"
        assert session.messages[1].role == "assistant"
        assert session.messages[2].role == "user"

    def test_compaction_filters_old_messages(self):
        session = Session.in_memory()

        # Old conversation
        session.append_message(UserMessage(content="Old question 1"))
        session.append_message(AssistantMessage(content=[TextContent(text="Old answer 1")]))
        session.append_message(UserMessage(content="Old question 2"))
        session.append_message(AssistantMessage(content=[TextContent(text="Old answer 2")]))

        # Compaction
        session.append_compaction(
            summary="User asked two questions and got answers.",
            first_kept_entry_id=session.leaf_id or "",
            tokens_before=50_000,
        )

        # New conversation after compaction
        session.append_message(UserMessage(content="New question"))
        session.append_message(AssistantMessage(content=[TextContent(text="New answer")]))

        messages = session.messages

        # Should be: synthetic user + synthetic assistant (summary) + new user + new assistant
        assert len(messages) == 4
        assert messages[0].role == "user"
        assert messages[0].content == "What did we do so far?"
        assert messages[1].role == "assistant"
        assistant = messages[1]
        assert isinstance(assistant, AssistantMessage)
        assert isinstance(assistant.content[0], TextContent)
        assert assistant.content[0].text == "User asked two questions and got answers."
        assert messages[2].role == "user"
        assert messages[2].content == "New question"
        assert messages[3].role == "assistant"

    def test_all_messages_returns_everything(self):
        session = Session.in_memory()

        session.append_message(UserMessage(content="Old question"))
        session.append_message(AssistantMessage(content=[TextContent(text="Old answer")]))

        session.append_compaction(
            summary="Summary", first_kept_entry_id=session.leaf_id or "", tokens_before=50_000
        )

        session.append_message(UserMessage(content="New question"))

        # all_messages ignores compaction, returns all MessageEntry messages
        assert len(session.all_messages) == 3
        assert session.all_messages[0].content == "Old question"
        assert session.all_messages[2].content == "New question"

    def test_compaction_with_no_messages_after(self):
        session = Session.in_memory()

        session.append_message(UserMessage(content="Question"))
        session.append_message(AssistantMessage(content=[TextContent(text="Answer")]))

        session.append_compaction(
            summary="Had a Q&A.", first_kept_entry_id=session.leaf_id or "", tokens_before=30_000
        )

        messages = session.messages

        # Only synthetic user + assistant summary, no messages after compaction
        assert len(messages) == 2
        assert messages[0].content == "What did we do so far?"
        assistant = messages[1]
        assert isinstance(assistant, AssistantMessage)
        content = assistant.content[0]
        assert isinstance(content, TextContent)
        assert content.text == "Had a Q&A."

    def test_multiple_compactions_uses_last(self):
        session = Session.in_memory()

        session.append_message(UserMessage(content="Q1"))
        session.append_message(AssistantMessage(content=[TextContent(text="A1")]))

        session.append_compaction(
            summary="First summary",
            first_kept_entry_id=session.leaf_id or "",
            tokens_before=30_000,
        )

        session.append_message(UserMessage(content="Q2"))
        session.append_message(AssistantMessage(content=[TextContent(text="A2")]))

        session.append_compaction(
            summary="Second summary (includes first)",
            first_kept_entry_id=session.leaf_id or "",
            tokens_before=60_000,
        )

        session.append_message(UserMessage(content="Q3"))

        messages = session.messages

        # Should use second compaction's summary
        assert len(messages) == 3
        assert messages[0].content == "What did we do so far?"
        assistant = messages[1]
        assert isinstance(assistant, AssistantMessage)
        content = assistant.content[0]
        assert isinstance(content, TextContent)
        assert content.text == "Second summary (includes first)"
        assert messages[2].content == "Q3"

    def test_compaction_preserves_tool_results_after(self):
        session = Session.in_memory()

        session.append_message(UserMessage(content="Old"))
        session.append_message(AssistantMessage(content=[TextContent(text="Old answer")]))

        session.append_compaction(
            summary="Summary", first_kept_entry_id=session.leaf_id or "", tokens_before=40_000
        )

        # New turn with tool calls
        session.append_message(UserMessage(content="Read file.txt"))
        session.append_message(
            AssistantMessage(
                content=[ToolCall(id="t1", name="read", arguments={"path": "file.txt"})]
            )
        )
        session.append_message(
            ToolResultMessage(
                tool_call_id="t1", tool_name="read", content=[TextContent(text="file contents")]
            )
        )

        messages = session.messages

        # synthetic pair + user + assistant (tool call) + tool result
        assert len(messages) == 5
        assert messages[3].role == "assistant"
        assert messages[4].role == "tool_result"


# ---------------------------------------------------------------------------
# Compaction entry persistence tests
# ---------------------------------------------------------------------------


class TestCompactionPersistence:
    def test_compaction_entry_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kon.session.Session.get_sessions_dir", lambda cwd: tmp_path)

        session = Session.create("/test/project")
        session.append_message(UserMessage(content="Hello"))
        session.append_message(AssistantMessage(content=[TextContent(text="Hi")]))

        session.append_compaction(
            summary="Test summary",
            first_kept_entry_id=session.leaf_id or "",
            tokens_before=42_000,
            details={"model": "test"},
        )

        # Need another assistant message after compaction for persistence
        session.append_message(UserMessage(content="Continue"))
        session.append_message(AssistantMessage(content=[TextContent(text="OK")]))

        session_file = session.session_file
        assert session_file is not None
        loaded = Session.load(session_file)

        compaction_entries = [e for e in loaded.entries if isinstance(e, CompactionEntry)]
        assert len(compaction_entries) == 1
        assert compaction_entries[0].summary == "Test summary"
        assert compaction_entries[0].tokens_before == 42_000
        assert compaction_entries[0].details == {"model": "test"}

    def test_loaded_session_messages_are_compacted(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kon.session.Session.get_sessions_dir", lambda cwd: tmp_path)

        session = Session.create("/test/project")
        session.append_message(UserMessage(content="Old"))
        session.append_message(AssistantMessage(content=[TextContent(text="Old reply")]))

        session.append_compaction(
            summary="We discussed old stuff.",
            first_kept_entry_id=session.leaf_id or "",
            tokens_before=50_000,
        )

        session.append_message(UserMessage(content="New"))
        session.append_message(AssistantMessage(content=[TextContent(text="New reply")]))

        session_file = session.session_file
        assert session_file is not None
        loaded = Session.load(session_file)

        # messages should be compacted view
        messages = loaded.messages
        assert len(messages) == 4
        assert messages[0].content == "What did we do so far?"
        assistant = messages[1]
        assert isinstance(assistant, AssistantMessage)
        content = assistant.content[0]
        assert isinstance(content, TextContent)
        assert content.text == "We discussed old stuff."
        assert messages[2].content == "New"

        # all_messages should have everything
        assert len(loaded.all_messages) == 4
        assert loaded.all_messages[0].content == "Old"


# ---------------------------------------------------------------------------
# Regression tests for usage-less latest assistant messages
# ---------------------------------------------------------------------------


class _TestCommandsApp(CommandsMixin):
    def __init__(
        self, session: Session, provider: MockProvider, chat, system_prompt: str = "test"
    ) -> None:
        self._session = session
        self._provider = provider
        self._agent = SimpleNamespace(system_prompt=system_prompt)
        self._is_running = False
        self._chat = chat
        self._runtime = ConversationRuntime(
            cwd=str(session.cwd),
            model="mock-model",
            model_provider="mock",
            api_key=None,
            base_url=None,
            thinking_level="high",
            tools=[],
        )
        self._runtime.provider = self._provider
        self._runtime.session = self._session
        self._runtime.agent = self._agent

    def query_one(self, selector: str, cls):
        assert selector == "#chat-log"
        return self._chat

    def _sync_runtime_state(self) -> None:
        self._provider = self._runtime.provider
        self._session = self._runtime.session
        self._agent = self._runtime.agent


class TestCompactionUsageBacktracking:
    @pytest.mark.asyncio
    async def test_manual_compaction_uses_latest_assistant_with_usage(
        self, monkeypatch, fake_chat
    ):
        session = Session.in_memory()
        session.append_message(UserMessage(content="hi"))
        session.append_message(
            AssistantMessage(
                content=[TextContent(text="usable")],
                usage=Usage(
                    input_tokens=100, output_tokens=50, cache_read_tokens=10, cache_write_tokens=5
                ),
            )
        )
        session.append_message(
            AssistantMessage(
                content=[TextContent(text="interrupted")],
                usage=None,
                stop_reason=StopReason.INTERRUPTED,
            )
        )

        provider = MockProvider()
        app = _TestCommandsApp(session=session, provider=provider, chat=fake_chat)

        async def _fake_summary(*args, **kwargs):
            return "summary"

        monkeypatch.setattr("kon.runtime.generate_summary", _fake_summary)

        await app._do_compact()

        assert fake_chat.errors == []
        assert fake_chat.compaction_tokens == 165
        compaction_entries = [e for e in session.entries if isinstance(e, CompactionEntry)]
        assert len(compaction_entries) == 1
        assert compaction_entries[0].tokens_before == 165

    @pytest.mark.asyncio
    async def test_auto_compaction_uses_latest_assistant_with_usage(self, monkeypatch):
        session = Session.in_memory()
        session.append_message(UserMessage(content="hi"))
        session.append_message(
            AssistantMessage(
                content=[TextContent(text="usable")],
                usage=Usage(
                    input_tokens=3000,
                    output_tokens=500,
                    cache_read_tokens=100,
                    cache_write_tokens=50,
                ),
            )
        )
        session.append_message(
            AssistantMessage(
                content=[TextContent(text="interrupted")],
                usage=None,
                stop_reason=StopReason.INTERRUPTED,
            )
        )

        provider = MockProvider()
        agent = Agent(
            provider=provider,
            tools=[],
            session=session,
            system_prompt="system",
            config=AgentConfig(context_window=1000, max_output_tokens=1),
        )

        async def _fake_summary(*args, **kwargs):
            return "summary"

        monkeypatch.setattr("kon.loop.generate_summary", _fake_summary)

        events = [e async for e in agent._check_compaction(StopReason.STOP, "system", None)]
        assert [e.type for e in events] == ["compaction_start", "compaction_end"]

        end_event = events[1]
        assert end_event.type == "compaction_end"
        assert end_event.tokens_before == 3650

        compaction_entries = [e for e in session.entries if isinstance(e, CompactionEntry)]
        assert len(compaction_entries) == 1
        assert compaction_entries[0].tokens_before == 3650


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestCompactionConfig:
    def test_default_config_values(self):
        cfg = Config({})
        assert cfg.compaction.on_overflow == "continue"
        assert cfg.compaction.buffer_tokens == 20_000
        assert cfg.agent.default_context_window == 200_000

    def test_config_override(self):
        cfg = Config({"compaction": {"on_overflow": "pause", "buffer_tokens": 10_000}})
        assert cfg.compaction.on_overflow == "pause"
        assert cfg.compaction.buffer_tokens == 10_000
        assert cfg.agent.default_context_window == 200_000

    def test_badge_colors_default(self):
        cfg = Config({})
        assert cfg.ui.colors.info == "#fabd2f"
        assert cfg.ui.colors.notice == "#fe8019"
        assert cfg.ui.colors.badge.bg == "#3c3836"
        assert cfg.ui.colors.badge.label == "#d3869b"

    def test_theme_selection_changes_palette(self):
        cfg = Config({"ui": {"theme": "one-light"}})
        assert cfg.ui.theme == "one-light"
        assert cfg.ui.colors.bg == "#fafafa"
        assert cfg.ui.colors.accent == "#4078f2"
