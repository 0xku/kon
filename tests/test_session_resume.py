import pytest

from kon.core.types import AssistantMessage, TextContent
from kon.llm.providers.mock import MockProvider
from kon.loop import Agent
from kon.runtime import ConversationRuntime
from kon.session import Session
from kon.ui.session_ui import SessionUIMixin


class _FakeChat:
    def __init__(self) -> None:
        self.infos: list[str] = []

    async def remove_all_children(self) -> None:
        return None

    def add_session_info(self, version: str) -> None:
        return None

    def add_loaded_resources(self, context_paths, skill_paths) -> None:
        return None

    def add_info_message(self, message: str, error: bool = False, warning: bool = False) -> None:
        if not error and not warning:
            self.infos.append(message)


class _FakeInfoBar:
    def __init__(self) -> None:
        self.tokens_calls: list[tuple[int, int, int, int, int]] = []
        self.file_changes_calls: list[dict[str, tuple[int, int]]] = []
        self.models: list[tuple[str, str | None]] = []
        self.thinking_levels: list[str] = []

    def set_tokens(
        self, input_t: int, output_t: int, context_t: int, cache_read_t: int, cache_write_t: int
    ) -> None:
        self.tokens_calls.append((input_t, output_t, context_t, cache_read_t, cache_write_t))

    def set_file_changes(self, file_changes: dict[str, tuple[int, int]]) -> None:
        self.file_changes_calls.append(file_changes)

    def set_model(self, model_id: str, provider: str | None) -> None:
        self.models.append((model_id, provider))

    def set_thinking_level(self, level: str) -> None:
        self.thinking_levels.append(level)


class _FakeStatusLine:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeInputBox:
    def __init__(self) -> None:
        self.focused = False

    def focus(self) -> None:
        self.focused = True


class _TestSessionApp(SessionUIMixin):
    VERSION = "test"

    def __init__(self, session: Session, provider: MockProvider) -> None:
        self._cwd = "/test/project"
        self._session = session
        self._provider = provider
        self._tools = []
        self._model = "mock-model"
        self._model_provider = "mock"
        self._thinking_level = "high"
        self._api_key = None
        self._hide_thinking = False
        self._current_block_type = None
        self._agent = Agent(
            provider=provider, tools=[], session=session, cwd=self._cwd, system_prompt="old prompt"
        )
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
        self._chat = _FakeChat()
        self._info_bar = _FakeInfoBar()
        self._status_line = _FakeStatusLine()
        self._input_box = _FakeInputBox()
        self.rendered_session: Session | None = None
        self.applied_thinking_level: str | None = None

    def query_one(self, selector: str, cls=None):
        if selector == "#chat-log":
            return self._chat
        if selector == "#info-bar":
            return self._info_bar
        if selector == "#status-line":
            return self._status_line
        if selector == "#input-box":
            return self._input_box
        raise AssertionError(selector)

    def _render_session_entries(self, session: Session) -> None:
        self.rendered_session = session

    def _apply_thinking_level_style(self, level: str) -> None:
        self.applied_thinking_level = level

    def _sync_runtime_state(self) -> None:
        self._model = self._runtime.model
        self._model_provider = self._runtime.model_provider
        self._thinking_level = self._runtime.thinking_level
        self._provider = self._runtime.provider
        self._session = self._runtime.session
        self._agent = self._runtime.agent


@pytest.mark.asyncio
async def test_loading_session_rebuilds_agent_with_persisted_system_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr("kon.session.Session.get_sessions_dir", lambda cwd: tmp_path)

    original_session = Session.create(
        "/test/project", provider="mock", model_id="mock-model", system_prompt="old prompt"
    )
    provider = MockProvider()
    app = _TestSessionApp(session=original_session, provider=provider)

    resumed_session = Session.create(
        "/test/project", provider="mock", model_id="mock-model", system_prompt="persisted prompt"
    )
    resumed_session.append_message(AssistantMessage(content=[TextContent(text="hi")]))

    assert resumed_session.session_file is not None

    await app._load_session(resumed_session.session_file)

    assert app._session is not None
    assert app._session.system_prompt == "persisted prompt"
    assert app._agent is not None
    assert app._agent.system_prompt == "persisted prompt"
    assert app._agent.session.id == app._session.id
    assert app.rendered_session is not None
    assert app.rendered_session.id == app._session.id
    assert app._chat.infos[-1] == "Resumed session"
