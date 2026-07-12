from typing import Any, cast

from rich.style import Style

from kon.ui.widgets import StatusLine


class _FakeLabel:
    def __init__(self) -> None:
        self.content = None
        self.layout_values: list[bool] = []

    def update(self, content="", *, layout: bool = True) -> None:
        self.content = content
        self.layout_values.append(layout)


def test_status_line_interrupt_hint_bolds_esc():
    status = StatusLine()
    status._status = "running"

    rendered = status._render_spinner()

    assert rendered.plain.endswith(" Working... (esc to interrupt)")
    esc_span = next(
        span for span in rendered.spans if rendered.plain[span.start : span.end] == "esc"
    )
    esc_style = (
        esc_span.style if isinstance(esc_span.style, Style) else Style.parse(str(esc_span.style))
    )
    assert esc_style.bold is True


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
