from typing import Any, cast

from kon.ui.widgets import StatusLine


class _FakeLabel:
    def __init__(self) -> None:
        self.content = None
        self.layout_values: list[bool] = []

    def update(self, content="", *, layout: bool = True) -> None:
        self.content = content
        self.layout_values.append(layout)


def test_status_line_formats_without_turn_tps(monkeypatch):
    status = StatusLine()
    status._start_time = 100.0
    status._tool_calls = 1

    monkeypatch.setattr("kon.ui.widgets.time.time", lambda: 104.0)

    rendered = status._format_complete_status()
    assert rendered.plain == "4s • 1x"


def test_exit_hint_updates_layout() -> None:
    status = StatusLine()
    label = _FakeLabel()
    status._hint_label = cast(Any, label)

    status.show_exit_hint()

    assert cast(Any, label.content).plain == "ctrl+c again to exit"
    assert label.layout_values == [True]


def test_delete_session_hint_updates_layout() -> None:
    status = StatusLine()
    label = _FakeLabel()
    status._hint_label = cast(Any, label)

    status.show_delete_session_hint()

    assert cast(Any, label.content).plain == "ctrl+d again to delete session"
    assert label.layout_values == [True]
