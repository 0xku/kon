import os
import time

from kon.notify import _bell, notify


def _capture_raw_write(monkeypatch):
    written = bytearray()

    real_open = os.open
    real_write = os.write

    def mock_open(path, flags, *args):
        if isinstance(path, str) and path == "/dev/tty":
            return 99
        return real_open(path, flags, *args)

    def mock_write(fd, data):
        if fd == 99:
            written.extend(data)
            return len(data)
        return real_write(fd, data)

    monkeypatch.setattr(os, "open", mock_open)
    monkeypatch.setattr(os, "write", mock_write)
    return written


def test_notify_emits_bell(monkeypatch):
    written = _capture_raw_write(monkeypatch)
    notify("kon", "Task complete")
    assert written.decode() == "\a"


def test_bell_debounces(monkeypatch):
    import kon.notify as mod

    written = _capture_raw_write(monkeypatch)

    mod._last_bell_time = time.monotonic()
    _bell()
    assert "\a" not in written.decode()

    mod._last_bell_time = 0.0
    _bell()
    assert "\a" in written.decode()
