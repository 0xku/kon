import os
import time

_BELL_DEBOUNCE_S = 0.5
_last_bell_time: float = 0.0


def _raw_write(data: bytes) -> None:
    try:
        fd = os.open("/dev/tty", os.O_WRONLY | os.O_NOCTTY)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
    except OSError:
        os.write(2, data)


def _bell() -> None:
    global _last_bell_time
    now = time.monotonic()
    if now - _last_bell_time < _BELL_DEBOUNCE_S:
        return
    _last_bell_time = now
    _raw_write(b"\a")


def notify(title: str, message: str) -> None:
    del title, message
    _bell()
