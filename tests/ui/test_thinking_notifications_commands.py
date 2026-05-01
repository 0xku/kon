from pathlib import Path
from typing import Any, ClassVar, cast

from kon import reset_config
from kon.runtime import ConversationRuntime
from kon.ui.commands import CommandsMixin
from kon.ui.floating_list import ListItem
from kon.ui.selection_mode import SelectionMode


class FakeChat:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.statuses: list[str] = []

    def add_info_message(self, message: str, error: bool = False, warning: bool = False) -> None:
        if error:
            self.errors.append(message)

    def show_status(self, message: str) -> None:
        self.statuses.append(message)


class FakeInfoBar:
    def __init__(self) -> None:
        self.thinking_levels: list[str] = []

    def set_thinking_level(self, level: str) -> None:
        self.thinking_levels.append(level)


class FakeFloatingList:
    def __init__(self) -> None:
        self.items: list[ListItem] = []
        self.searchable: bool | None = None

    def show(self, items: list[ListItem], searchable: bool = False) -> None:
        self.items = items
        self.searchable = searchable


class FakeInputBox:
    def clear(self) -> None:
        pass

    def set_autocomplete_enabled(self, enabled: bool) -> None:
        pass

    def set_completing(self, completing: bool) -> None:
        pass

    def focus(self) -> None:
        pass


class FakeProvider:
    name = "fake"
    thinking_levels: ClassVar[list[str]] = ["none", "minimal", "low", "medium", "high", "xhigh"]

    def __init__(self) -> None:
        self.thinking_level = "low"

    def set_thinking_level(self, level: str) -> None:
        if level not in self.thinking_levels:
            raise ValueError(level)
        self.thinking_level = level


class FakeSession:
    def __init__(self) -> None:
        self.thinking_levels: list[str] = []

    def set_thinking_level(self, level: str) -> None:
        self.thinking_levels.append(level)


class FakeCommands(CommandsMixin):
    def __init__(self) -> None:
        self.chat = FakeChat()
        self.info_bar = FakeInfoBar()
        self.completion_list = FakeFloatingList()
        self.input_box = FakeInputBox()
        self._provider = cast(Any, FakeProvider())
        self._session = cast(Any, FakeSession())
        self._thinking_level = "low"
        self._selection_mode = None
        self._runtime = ConversationRuntime(
            cwd=".",
            model="fake-model",
            model_provider="fake",
            api_key=None,
            base_url=None,
            thinking_level=self._thinking_level,
            tools=[],
        )
        self._runtime.provider = self._provider
        self._runtime.session = self._session
        self.applied_thinking_levels: list[str] = []

    def query_one(self, selector, widget_type):
        if selector == "#chat-log":
            return self.chat
        if selector == "#info-bar":
            return self.info_bar
        if selector == "#completion-list":
            return self.completion_list
        if selector == "#input-box":
            return self.input_box
        raise AssertionError(f"Unexpected selector: {selector}")

    def _apply_thinking_level_style(self, level: str) -> None:
        self.applied_thinking_levels.append(level)

    def _sync_runtime_state(self) -> None:
        self._provider = self._runtime.provider
        self._session = self._runtime.session
        self._thinking_level = self._runtime.thinking_level


def test_thinking_command_with_argument_updates_current_session_only():
    fake = FakeCommands()

    fake._handle_thinking_command("high")

    provider = cast(Any, fake._provider)
    session = cast(Any, fake._session)

    assert provider.thinking_level == "high"
    assert fake._thinking_level == "high"
    assert session.thinking_levels == ["high"]
    assert fake.info_bar.thinking_levels == ["high"]
    assert fake.applied_thinking_levels == ["high"]
    assert fake.chat.statuses == ["Thinking level changed to high"]


def test_thinking_command_without_argument_opens_picker():
    fake = FakeCommands()

    fake._handle_thinking_command("")

    assert fake._selection_mode == SelectionMode.THINKING
    assert fake.completion_list.searchable is True
    assert [(item.value, item.label, item.description) for item in fake.completion_list.items] == [
        ("none", "none", "current session only"),
        ("minimal", "minimal", "current session only"),
        ("low", "low ✓", "current session only"),
        ("medium", "medium", "current session only"),
        ("high", "high", "current session only"),
        ("xhigh", "xhigh", "current session only"),
    ]


def test_notifications_command_with_argument_updates_runtime_and_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    reset_config()
    fake = FakeCommands()

    try:
        fake._handle_notifications_command("on")
    finally:
        reset_config()

    config_file = Path(tmp_path) / ".kon" / "config.toml"
    assert fake.chat.statuses == ["Notifications turned on and saved"]
    assert "enabled = true" in config_file.read_text(encoding="utf-8")


def test_notifications_command_without_argument_opens_picker(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    reset_config()
    fake = FakeCommands()

    try:
        fake._handle_notifications_command("")
    finally:
        reset_config()

    assert fake._selection_mode == SelectionMode.NOTIFICATIONS
    assert fake.completion_list.searchable is True
    assert [(item.value, item.label, item.description) for item in fake.completion_list.items] == [
        ("on", "on", "play notification sounds"),
        ("off", "off ✓", "disable notification sounds"),
    ]
