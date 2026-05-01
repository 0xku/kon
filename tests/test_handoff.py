from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest

from kon.core.handoff import HANDOFF_PROMPT_TEMPLATE, generate_handoff_prompt
from kon.core.types import AssistantMessage, StopReason, TextContent, TextPart, UserMessage
from kon.llm.base import LLMStream
from kon.llm.providers.mock import MockProvider
from kon.loop import build_system_prompt
from kon.runtime import ConversationRuntime
from kon.session import CustomMessageEntry, Session
from kon.ui.commands import CommandsMixin


class _FakeInfoBar:
    def __init__(self) -> None:
        self.tokens_calls: list[tuple[int, int, int, int]] = []
        self.file_changes_calls: list[dict[str, tuple[int, int]]] = []
        self.thinking_levels: list[str] = []

    def set_tokens(self, input_t: int, output_t: int, context_t: int, cache_t: int) -> None:
        self.tokens_calls.append((input_t, output_t, context_t, cache_t))

    def set_file_changes(self, file_changes: dict[str, tuple[int, int]]) -> None:
        self.file_changes_calls.append(file_changes)

    def set_thinking_level(self, level: str) -> None:
        self.thinking_levels.append(level)


class _FakeStatusLine:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _TestCommandsApp(CommandsMixin):
    def __init__(
        self,
        session: Session,
        provider: MockProvider,
        chat,
        input_box,
        info_bar: _FakeInfoBar | None = None,
        status_line: _FakeStatusLine | None = None,
    ) -> None:
        self._cwd = "/test/project"
        self._thinking_level = "high"
        self._model = "mock-model"
        self._model_provider = "mock"
        self._api_key = None
        self._provider = provider
        self._session = session
        self._agent = SimpleNamespace(
            system_prompt="system",
            context=SimpleNamespace(agents_files=[], skills=[]),
            reload_context=lambda: None,
        )
        self._is_running = False
        self._tools = []
        self._runtime = ConversationRuntime(
            cwd=self._cwd,
            model=self._model,
            model_provider=self._model_provider,
            api_key=self._api_key,
            base_url=None,
            thinking_level=self._thinking_level,
            tools=self._tools,
        )
        self._runtime.provider = self._provider
        self._runtime.session = self._session
        self._runtime.agent = self._agent
        self._chat = chat
        self._input_box = input_box
        self._info_bar = info_bar or _FakeInfoBar()
        self._status_line = status_line or _FakeStatusLine()

    def query_one(self, selector: str, cls):
        if selector == "#chat-log":
            return self._chat
        if selector == "#info-bar":
            return self._info_bar
        if selector == "#status-line":
            return self._status_line
        if selector == "#input-box":
            return self._input_box
        raise AssertionError(selector)

    def run_worker(self, coro, exclusive=False):
        return coro

    def _sync_runtime_state(self) -> None:
        self._model = self._runtime.model
        self._model_provider = self._runtime.model_provider
        self._thinking_level = self._runtime.thinking_level
        self._provider = self._runtime.provider
        self._session = self._runtime.session
        self._agent = self._runtime.agent

    def _sync_slash_commands(self) -> None:
        return None

    def _render_session_entries(self, session: Session) -> None:
        return None


class _FakeChat:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.infos: list[str] = []
        self.statuses: list[str] = []

    def add_info_message(self, message: str, error: bool = False, warning: bool = False) -> None:
        if error:
            self.errors.append(message)
        else:
            self.infos.append(message)

    def show_status(self, message: str) -> None:
        self.statuses.append(message)

    def show_spinner_status(self, message: str) -> None:
        self.statuses.append(message)

    async def remove_all_children(self) -> None:
        return None

    def add_session_info(self, version: str) -> None:
        return None

    def add_loaded_resources(self, context_paths, skill_paths) -> None:
        return None

    def add_handoff_link_message(
        self, label: str, target_session_id: str, query: str, direction: Literal["back", "forward"]
    ) -> None:
        return None


class _FakeInput:
    def __init__(self) -> None:
        self.cleared = False
        self.inserted = ""
        self.focused = False

    def clear(self) -> None:
        self.cleared = True

    def insert(self, text: str) -> None:
        self.inserted = text

    def focus(self) -> None:
        self.focused = True


@pytest.mark.asyncio
async def test_generate_handoff_prompt_uses_query_and_messages(monkeypatch):
    provider = MockProvider()
    captured = {}

    async def _fake_stream(
        messages, *, system_prompt=None, tools=None, temperature=None, max_tokens=None
    ):
        captured["messages"] = messages
        captured["system_prompt"] = system_prompt

        async def _iter():
            yield TextPart(text="Task: Continue")

        stream = LLMStream()
        stream.set_iterator(_iter())
        return stream

    monkeypatch.setattr(provider, "stream", _fake_stream)

    result = await generate_handoff_prompt(
        [UserMessage(content="we changed auth")],
        provider,
        system_prompt="sys",
        query="ship phase 2",
    )

    assert result == "Task: Continue"
    assert captured["system_prompt"] == "sys"
    assert len(captured["messages"]) == 2
    assert captured["messages"][-1].content == HANDOFF_PROMPT_TEMPLATE.format(query="ship phase 2")


@pytest.mark.asyncio
async def test_do_handoff_creates_link_entries_and_prefills_prompt(monkeypatch):
    session = Session.in_memory("/test/project", provider="mock", model_id="mock-model")
    session.append_message(UserMessage(content="fix bug"))
    session.append_message(
        AssistantMessage(content=[TextContent(text="done")], stop_reason=StopReason.STOP)
    )

    provider = MockProvider()
    chat = _FakeChat()
    input_box = _FakeInput()
    app = _TestCommandsApp(session=session, provider=provider, chat=chat, input_box=input_box)

    async def _fake_handoff(messages, _provider_obj, system_prompt, query):
        return "Task: Implement phase two"

    monkeypatch.setattr("kon.runtime.generate_handoff_prompt", _fake_handoff)

    original_session = app._session
    assert original_session is not None

    await app._do_handoff("implement phase two")

    assert app._session is not None
    assert app._session.id != original_session.id
    assert input_box.cleared is True
    assert input_box.inserted == "Task: Implement phase two"
    assert input_box.focused is True

    new_custom_entries = [e for e in app._session.entries if isinstance(e, CustomMessageEntry)]
    assert any(e.custom_type == app.HANDOFF_BACKLINK_TYPE for e in new_custom_entries)

    original_custom_entries = [
        e for e in original_session.entries if isinstance(e, CustomMessageEntry)
    ]
    assert any(e.custom_type == app.HANDOFF_FORWARD_LINK_TYPE for e in original_custom_entries)


@pytest.mark.asyncio
async def test_new_conversation_resets_file_change_stats():
    session = Session.in_memory("/test/project", provider="mock", model_id="mock-model")
    provider = MockProvider()
    chat = _FakeChat()
    input_box = _FakeInput()
    info_bar = _FakeInfoBar()
    status_line = _FakeStatusLine()
    app = _TestCommandsApp(
        session=session,
        provider=provider,
        chat=chat,
        input_box=input_box,
        info_bar=info_bar,
        status_line=status_line,
    )

    await app._do_new_conversation(cast(Any, chat), info_bar, status_line)

    assert info_bar.file_changes_calls[-1] == {}


def test_clear_conversation_resets_file_change_stats():
    session = Session.in_memory("/test/project", provider="mock", model_id="mock-model")
    provider = MockProvider()
    chat = _FakeChat()
    input_box = _FakeInput()
    info_bar = _FakeInfoBar()
    app = _TestCommandsApp(
        session=session, provider=provider, chat=chat, input_box=input_box, info_bar=info_bar
    )

    app._clear_conversation()

    assert info_bar.file_changes_calls[-1] == {}


def test_clear_conversation_creates_session_with_persisted_system_prompt(monkeypatch):
    captured: dict[str, str | None] = {"system_prompt": None}
    original_create = Session.create

    def _fake_create(
        cwd,
        persist=True,
        provider=None,
        model_id=None,
        thinking_level="high",
        system_prompt=None,
        tools=None,
    ):
        captured["system_prompt"] = system_prompt
        return original_create(
            cwd,
            persist=persist,
            provider=provider,
            model_id=model_id,
            thinking_level=thinking_level,
            system_prompt=system_prompt,
            tools=tools,
        )

    monkeypatch.setattr("kon.runtime.Session.create", _fake_create)

    session = Session.in_memory("/test/project", provider="mock", model_id="mock-model")
    provider = MockProvider()
    chat = _FakeChat()
    input_box = _FakeInput()
    app = _TestCommandsApp(session=session, provider=provider, chat=chat, input_box=input_box)

    app._clear_conversation()

    assert app._session is not None
    assert captured["system_prompt"] == build_system_prompt("/test/project", tools=[])
    assert app._session.system_prompt == captured["system_prompt"]
