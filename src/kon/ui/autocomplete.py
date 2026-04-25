"""
Autocomplete providers for inline completion.

Providers handle filtering and suggestion generation for different
completion types (slash commands, file paths, sessions, etc.).
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from .floating_list import ListItem


@dataclass
class CompletionResult:
    items: list[ListItem]
    prefix: str  # The text being completed (e.g., "/hel" or "@src")
    replace_start: int  # Column position where replacement starts


class AutocompleteProvider(ABC):
    @property
    @abstractmethod
    def trigger_chars(self) -> set[str]: ...

    @abstractmethod
    def should_trigger(self, text: str, cursor_col: int) -> bool: ...

    @abstractmethod
    def get_suggestions(self, text: str, cursor_col: int) -> CompletionResult | None: ...

    @abstractmethod
    def apply_completion(
        self, text: str, cursor_col: int, item: ListItem, prefix: str
    ) -> tuple[str, int]:
        """
        Apply the selected completion.

        Returns:
            Tuple of (new_text, new_cursor_col)
        """
        ...


class FuzzyMatcher:
    def __init__(self, case_sensitive: bool = False) -> None:
        self.case_sensitive = case_sensitive

    def match(self, query: str, candidate: str) -> tuple[float, Sequence[int]]:
        """
        Match query against candidate.

        Returns:
            Tuple of (score, list of matching indices). (0, []) for no match.
        """
        if not query:
            return (1.0, [])

        if not self.case_sensitive:
            query = query.lower()
            candidate = candidate.lower()

        positions = []
        idx = 0
        for char in query:
            idx = candidate.find(char, idx)
            if idx == -1:
                return (0.0, [])
            positions.append(idx)
            idx += 1

        score = self._score(candidate, positions)
        return (score, positions)

    @classmethod
    @lru_cache(maxsize=1024)
    def get_first_letters(cls, candidate: str) -> frozenset[int]:
        indices = set()
        word_start = True
        for i, char in enumerate(candidate):
            if char.isalnum():
                if word_start:
                    indices.add(i)
                    word_start = False
            else:
                word_start = True
        return frozenset(indices)

    def _score(self, candidate: str, positions: Sequence[int]) -> float:
        if not positions:
            return 0.0

        score = float(len(positions))
        first_letters = self.get_first_letters(candidate)
        first_letter_matches = len(positions) - len(set(positions) - first_letters)
        score += first_letter_matches * 0.5

        groups = 1
        for i in range(1, len(positions)):
            if positions[i] != positions[i - 1] + 1:
                groups += 1

        if len(positions) > 1:
            group_factor = (len(positions) - groups + 1) / len(positions)
            score *= 1 + group_factor

        if positions[0] == 0:
            score *= 1.2

        return score


@dataclass
class SlashCommand:
    name: str
    description: str
    shortcut: str | None = None
    is_skill: bool = False
    submit_on_select: bool = True


class SlashCommandProvider(AutocompleteProvider):
    def __init__(self, commands: list[SlashCommand] | None = None) -> None:
        self._commands = commands or []
        self._matcher = FuzzyMatcher(case_sensitive=False)

    @property
    def commands(self) -> list[SlashCommand]:
        return self._commands

    @commands.setter
    def commands(self, value: list[SlashCommand]) -> None:
        self._commands = value

    @property
    def trigger_chars(self) -> set[str]:
        return {"/"}

    def _extract_token(self, text: str, cursor_col: int) -> tuple[str, int, bool] | None:
        text_before = text[:cursor_col]
        slash_pos = text_before.rfind("/")
        if slash_pos == -1:
            return None

        if slash_pos > 0 and not text_before[slash_pos - 1].isspace():
            return None

        token = text_before[slash_pos:]
        if " " in token or "\n" in token or not token.startswith("/"):
            return None

        is_start_command = not text_before[:slash_pos].strip()
        return token, slash_pos, is_start_command

    def _available_commands(self, is_start_command: bool) -> list[SlashCommand]:
        if is_start_command:
            return self._commands
        return [cmd for cmd in self._commands if cmd.is_skill]

    def should_trigger(self, text: str, cursor_col: int) -> bool:
        extracted = self._extract_token(text, cursor_col)
        if extracted is None:
            return False
        _, _, is_start_command = extracted
        return bool(self._available_commands(is_start_command))

    def get_suggestions(self, text: str, cursor_col: int) -> CompletionResult | None:
        extracted = self._extract_token(text, cursor_col)
        if extracted is None:
            return None

        token, prefix_start, is_start_command = extracted
        available_commands = self._available_commands(is_start_command)
        if not available_commands:
            return None

        # Extract the command prefix (without the /)
        query = token[1:]

        # Filter and score commands
        scored = []
        for cmd in available_commands:
            score, _ = self._matcher.match(query, cmd.name)
            if score > 0 or not query:
                scored.append((score, cmd))

        # Sort by score descending
        scored.sort(key=lambda x: (-x[0], x[1].name))

        items = []
        for _, cmd in scored:
            label = f"/{cmd.name}"
            desc = cmd.description
            if cmd.shortcut:
                desc = f"{desc} ({cmd.shortcut})"
            items.append(ListItem(value=cmd, label=label, description=desc))

        if not items:
            return None

        return CompletionResult(items=items, prefix=token, replace_start=prefix_start)

    def apply_completion(
        self, text: str, cursor_col: int, item: ListItem, prefix: str
    ) -> tuple[str, int]:
        cmd: SlashCommand = item.value
        text_before = text[:cursor_col]
        prefix_start = cursor_col - len(prefix)
        text_after = text[cursor_col:]

        # Replace prefix with command + space
        new_text = text_before[:prefix_start] + f"/{cmd.name} " + text_after
        new_cursor = prefix_start + len(cmd.name) + 2  # +2 for "/" and space

        return (new_text, new_cursor)


class FilePathProvider(AutocompleteProvider):
    def __init__(self, cwd: str = ".", fd_path: str | None = None) -> None:
        self._cwd = cwd
        self._fd_path = fd_path
        self._matcher = FuzzyMatcher(case_sensitive=False)
        self._cached_paths: list[str] = []

    def set_cwd(self, cwd: str) -> None:
        self._cwd = cwd

    def set_fd_path(self, fd_path: str | None) -> None:
        self._fd_path = fd_path

    def set_paths(self, paths: list[str]) -> None:
        self._cached_paths = paths

    @property
    def trigger_chars(self) -> set[str]:
        return {"@"}

    def should_trigger(self, text: str, cursor_col: int) -> bool:
        text_before = text[:cursor_col]
        # Find @ that's at start or after whitespace
        for i in range(len(text_before) - 1, -1, -1):
            if text_before[i] == "@":
                if i == 0 or text_before[i - 1].isspace():
                    return True
                break
            elif text_before[i].isspace():
                break
        return False

    def get_suggestions(self, text: str, cursor_col: int) -> CompletionResult | None:
        text_before = text[:cursor_col]

        # Find the @ and query
        at_pos = -1
        for i in range(len(text_before) - 1, -1, -1):
            if text_before[i] == "@":
                if i == 0 or text_before[i - 1].isspace():
                    at_pos = i
                    break
            elif text_before[i].isspace():
                break

        if at_pos == -1:
            return None

        query = text_before[at_pos + 1 :]  # Text after @
        prefix = text_before[at_pos:]  # Including @

        # Get file suggestions
        paths = self._get_paths(query)

        items = []
        for path in paths[:20]:  # Limit results
            # Format: label = filename (or dirname/), description = parent path
            is_dir = path.endswith("/")
            clean_path = path.rstrip("/")
            basename = os.path.basename(clean_path)
            dirname = os.path.dirname(clean_path)

            label = basename + ("/" if is_dir else "")
            # Show parent directory as description
            description = dirname if dirname else "."

            items.append(ListItem(value=path, label=label, description=description))

        if not items:
            return None

        return CompletionResult(items=items, prefix=prefix, replace_start=at_pos)

    def _get_paths(self, query: str) -> list[str]:
        if self._fd_path:
            return self._query_fd(query)
        return self._fuzzy_filter(query)

    def _query_fd(self, query: str) -> list[str]:
        import subprocess

        try:
            cmd = [
                self._fd_path,
                "--full-path",
                "--color=never",
                "--max-results",
                "50",
                "-t",
                "f",
                "-t",
                "d",
            ]
            if query:
                cmd.append(query)
            else:
                cmd.append(".")

            result = subprocess.run(
                cmd, cwd=self._cwd, capture_output=True, text=True, timeout=0.3
            )

            if result.returncode == 0:
                return [p for p in result.stdout.strip().split("\n") if p]
        except Exception:
            pass

        return self._fuzzy_filter(query)

    def _fuzzy_filter(self, query: str) -> list[str]:
        if not query:
            return self._cached_paths[:50]

        scored = []
        for path in self._cached_paths:
            score, _ = self._matcher.match(query, path)
            if score > 0:
                scored.append((score, path))

        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored[:50]]

    def apply_completion(
        self, text: str, cursor_col: int, item: ListItem, prefix: str
    ) -> tuple[str, int]:
        path: str = item.value
        text_before = text[:cursor_col]
        prefix_start = cursor_col - len(prefix)
        text_after = text[cursor_col:]

        # Replace prefix with @path + space
        is_dir = path.endswith("/")
        suffix = "" if is_dir else " "
        new_text = text_before[:prefix_start] + f"@{path}{suffix}" + text_after
        new_cursor = prefix_start + len(path) + 1 + len(suffix)  # +1 for @

        return (new_text, new_cursor)


# Default slash commands
DEFAULT_COMMANDS = [
    SlashCommand("help", "Show available commands"),
    SlashCommand("quit", "Quit the application", "ctrl+c,c"),
    SlashCommand("clear", "Clear conversation history"),
    SlashCommand("model", "Change model"),
    SlashCommand("themes", "Change UI theme"),
    SlashCommand("permissions", "Change permission mode"),
    SlashCommand("thinking", "Change thinking level"),
    SlashCommand("notifications", "Toggle notifications"),
    SlashCommand("new", "Start new conversation"),
    SlashCommand("handoff", "Start focused handoff in new session", submit_on_select=False),
    SlashCommand("resume", "Resume a session"),
    SlashCommand("session", "Show session info and stats"),
    SlashCommand("login", "Login to a provider"),
    SlashCommand("logout", "Logout from a provider"),
    SlashCommand("export", "Export session to HTML"),
    SlashCommand("copy", "Copy last agent response text"),
    SlashCommand("compact", "Compact current conversation now"),
]
