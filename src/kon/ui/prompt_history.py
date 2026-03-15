from __future__ import annotations

import json
from pathlib import Path

from kon import CONFIG_DIR_NAME

MAX_HISTORY_ENTRIES = 50


def _history_path() -> Path:
    return Path.home() / CONFIG_DIR_NAME / "prompt-history.jsonl"


class PromptHistory:
    def __init__(self) -> None:
        self._entries: list[str] = []
        self._index: int = 0
        self._draft: str = ""
        self._load()

    def _load(self) -> None:
        path = _history_path()
        if not path.exists():
            return
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return
        lines: list[str] = []
        for line in text.strip().split("\n"):
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(entry, str) and entry:
                lines.append(entry)
        self._entries = lines[-MAX_HISTORY_ENTRIES:]
        if len(lines) > MAX_HISTORY_ENTRIES:
            self._rewrite()

    def _rewrite(self) -> None:
        path = _history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            content = "\n".join(json.dumps(e) for e in self._entries) + "\n"
            path.write_text(content, encoding="utf-8")
        except OSError:
            pass

    def _append_to_file(self, entry: str) -> None:
        path = _history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    def append(self, text: str) -> None:
        if not text:
            return
        if self._entries and self._entries[-1] == text:
            self._reset_index()
            return
        self._entries.append(text)
        trimmed = len(self._entries) > MAX_HISTORY_ENTRIES
        if trimmed:
            self._entries = self._entries[-MAX_HISTORY_ENTRIES:]
            self._rewrite()
        else:
            self._append_to_file(text)
        self._reset_index()

    @property
    def is_browsing(self) -> bool:
        return self._index != 0

    def _reset_index(self) -> None:
        self._index = 0
        self._draft = ""

    def navigate(self, direction: int, current_text: str) -> str | None:
        if not self._entries:
            return None

        if self._index == 0:
            self._draft = current_text

        new_index = self._index + direction

        if new_index > 0 or abs(new_index) > len(self._entries):
            return None

        self._index = new_index

        if self._index == 0:
            return self._draft

        return self._entries[self._index]
