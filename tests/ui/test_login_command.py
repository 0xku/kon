from contextlib import contextmanager

import pytest

from kon.llm.oauth.openai import OpenAICredentials
from kon.ui import commands
from kon.ui.commands import CommandsMixin
from kon.ui.floating_list import ListItem
from kon.ui.selection_mode import SelectionMode


class FakeChat:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def add_info_message(self, message: str, error: bool = False, warning: bool = False) -> None:
        if error:
            self.errors.append(message)
        elif warning:
            self.warnings.append(message)
        else:
            self.infos.append(message)


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


class FakeCommands(CommandsMixin):
    def __init__(self) -> None:
        self.chat = FakeChat()
        self.completion_list = FakeFloatingList()
        self.input_box = FakeInputBox()
        self._selection_mode = None
        self.workers: list[tuple[object, bool]] = []

    @contextmanager
    def batch_update(self):
        yield

    def query_one(self, selector, widget_type):
        if selector == "#chat-log":
            return self.chat
        if selector == "#completion-list":
            return self.completion_list
        if selector == "#input-box":
            return self.input_box
        raise AssertionError(f"Unexpected selector: {selector}")

    def run_worker(self, coro, exclusive: bool = True):
        self.workers.append((coro, exclusive))
        coro.close()


@pytest.mark.asyncio
async def test_openai_login_flow_skips_oauth_when_credentials_are_valid(monkeypatch):
    fake = FakeCommands()
    creds = OpenAICredentials(
        refresh="refresh", access="access", expires=9_999_999_999_999, account_id="account"
    )
    login_calls: list[int] = []

    async def get_credentials() -> OpenAICredentials:
        return creds

    async def login(**kwargs) -> None:
        login_calls.append(1)

    monkeypatch.setattr(commands, "get_valid_openai_credentials", get_credentials)
    monkeypatch.setattr(commands, "openai_login", login)

    await fake._openai_login_flow()

    assert login_calls == []
    assert fake.chat.infos == ["Already logged in to OpenAI"]


@pytest.mark.asyncio
async def test_openai_login_flow_starts_oauth_for_stale_saved_credentials(monkeypatch):
    fake = FakeCommands()
    login_calls: list[int] = []

    async def get_credentials() -> None:
        return None

    async def login(**kwargs) -> None:
        login_calls.append(1)

    monkeypatch.setattr(commands, "has_saved_openai_credentials", lambda: True)
    monkeypatch.setattr(commands, "get_valid_openai_credentials", get_credentials)
    monkeypatch.setattr(commands, "openai_login", login)

    await fake._openai_login_flow()

    assert login_calls == [1]
    assert fake.chat.warnings == ["Your saved OpenAI session is no longer valid."]
    assert fake.chat.infos == [
        "Successfully logged in to OpenAI!\nYou can now use /model to select openai-codex models."
    ]


@pytest.mark.asyncio
async def test_copilot_login_flow_starts_oauth_for_stale_saved_credentials(monkeypatch):
    fake = FakeCommands()
    login_calls: list[int] = []

    async def get_token() -> None:
        return None

    async def login(**kwargs) -> None:
        login_calls.append(1)

    monkeypatch.setattr(commands, "has_saved_copilot_credentials", lambda: True)
    monkeypatch.setattr(commands, "get_copilot_token", get_token)
    monkeypatch.setattr(commands, "copilot_login", login)

    await fake._copilot_login_flow()

    assert login_calls == [1]
    assert fake.chat.warnings == ["Your saved GitHub Copilot session is no longer valid."]
    assert fake.chat.infos == [
        "Successfully logged in to GitHub Copilot!\n"
        "You can now use /model to select Copilot models."
    ]


def test_select_login_provider_schedules_login_workers():
    fake = FakeCommands()

    fake._select_login_provider("openai")
    fake._select_login_provider("github-copilot")

    assert len(fake.workers) == 2
    assert all(exclusive is False for _, exclusive in fake.workers)
    assert fake.chat.infos == []


def test_login_picker_marks_saved_credentials_without_logged_in_checkmark(monkeypatch):
    fake = FakeCommands()
    monkeypatch.setattr(commands, "has_saved_openai_credentials", lambda: True)
    monkeypatch.setattr(commands, "has_saved_copilot_credentials", lambda: False)

    fake._handle_login_command("")

    assert fake._selection_mode == SelectionMode.LOGIN
    rows = [(item.value, item.label, item.description) for item in fake.completion_list.items]
    assert rows == [
        ("github-copilot", "GitHub Copilot", ""),
        ("openai", "OpenAI (ChatGPT/Codex)", "saved credentials"),
    ]
