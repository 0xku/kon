import inspect
from types import SimpleNamespace

from kon.ui.app import Kon


class _FakeChat:
    def __init__(self) -> None:
        self.statuses: list[str] = []

    def show_status(self, message: str) -> None:
        self.statuses.append(message)


class _FakeEvent:
    def __init__(self, target_session_id: str) -> None:
        self.target_session_id = target_session_id
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _TestKon(Kon):
    def __init__(self) -> None:
        super().__init__(cwd="/tmp")
        self._chat = _FakeChat()
        self.worker_calls: list[tuple[object, bool]] = []

    def query_one(self, selector: str, cls=None):
        if selector == "#chat-log":
            return self._chat
        raise AssertionError(selector)

    def run_worker(self, coro, exclusive=False):
        self.worker_calls.append((coro, exclusive))
        return SimpleNamespace()


def test_handoff_link_interrupts_before_switch_when_running() -> None:
    app = _TestKon()
    app._is_running = True
    app._cancel_event = None
    event = _FakeEvent("session-123")

    app.on_handoff_link_selected(event)

    assert event.stopped is True
    assert app._pending_session_switch_id == "session-123"
    assert app._interrupt_requested is True
    assert app._chat.statuses[-1] == "Interrupting before handoff..."
    assert app.worker_calls == []


def test_handoff_link_switches_immediately_when_idle() -> None:
    app = _TestKon()
    event = _FakeEvent("session-456")

    app.on_handoff_link_selected(event)

    assert event.stopped is True
    assert app._pending_session_switch_id is None
    assert len(app.worker_calls) == 1
    coro, exclusive = app.worker_calls[0]
    assert inspect.iscoroutine(coro)
    coro.close()
    assert exclusive is True
