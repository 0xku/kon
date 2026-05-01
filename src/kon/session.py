"""
Session - persistence layer for agent conversations.

Sessions are stored as append-only JSONL files. Each line is a JSON entry
with a type field. The first line is always the session header.

Structure:
    {"type": "header", "id": "...", "version": 1, "timestamp": "...",
     "cwd": "...", "system_prompt": "...", "tools": ["read", "edit", ...]}
    {"type": "message", "id": "...", "parent_id": "...", "timestamp": "...", "message": {...}}
    {"type": "message", "id": "...", "parent_id": "...", "timestamp": "...", "message": {...}}
    ...
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from kon import CONFIG_DIR_NAME

from .core.types import (
    AssistantMessage,
    Message,
    StopReason,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

CURRENT_VERSION = 1
_SKILL_TRIGGER_HEADER_RE = re.compile(r"^\[([a-z0-9-]+)\]\s*$")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class SessionHeader(BaseModel):
    type: Literal["header"] = "header"
    version: int = CURRENT_VERSION
    id: str
    timestamp: str
    cwd: str
    system_prompt: str | None = None
    tools: list[str] | None = None
    initial_thinking_level: str = "high"


class EntryBase(BaseModel):
    id: str
    parent_id: str | None
    timestamp: str


class MessageEntry(EntryBase):
    type: Literal["message"] = "message"
    message: Message


class ThinkingLevelChangeEntry(EntryBase):
    type: Literal["thinking_level_change"] = "thinking_level_change"
    thinking_level: str


class ModelChangeEntry(EntryBase):
    type: Literal["model_change"] = "model_change"
    provider: str
    model_id: str
    base_url: str | None = None


class CompactionEntry(EntryBase):
    type: Literal["compaction"] = "compaction"
    summary: str
    first_kept_entry_id: str  # Retained for compaction metadata; messages use entry order.
    tokens_before: int
    details: dict[str, Any] | None = None


class CustomMessageEntry(EntryBase):
    type: Literal["custom_message"] = "custom_message"
    custom_type: str
    content: str
    display: bool = True
    details: dict[str, Any] | None = None


class SessionInfoEntry(EntryBase):
    type: Literal["session_info"] = "session_info"
    name: str | None = None


SessionEntry = (
    MessageEntry
    | ThinkingLevelChangeEntry
    | ModelChangeEntry
    | CompactionEntry
    | CustomMessageEntry
    | SessionInfoEntry
)


class SessionInfo(BaseModel):
    id: str
    path: Path
    cwd: str
    name: str | None = None
    created: datetime
    modified: datetime
    message_count: int
    first_message: str


@dataclass(frozen=True)
class SessionTokenTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    context_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


@dataclass(frozen=True)
class SessionMessageCounts:
    user_messages: int = 0
    assistant_messages: int = 0
    tool_calls: int = 0
    tool_results: int = 0

    @property
    def total_messages(self) -> int:
        return self.user_messages + self.assistant_messages


class Session:
    """
    Manages conversation persistence as append-only JSONL.

    Usage:
        # Create new session with initial model/thinking level
        session = Session.create("/path/to/project", provider="openai", model_id="gpt-4")

        # Add messages
        session.append_message(user_message)
        session.append_message(assistant_message)

        # Resume later
        session = Session.load(session_file_path)
        messages = session.messages

        # List sessions
        sessions = Session.list("/path/to/project")
    """

    @staticmethod
    def generate_id() -> str:
        return uuid.uuid4().hex[:8]

    @staticmethod
    def get_sessions_dir(cwd: str) -> Path:
        home = Path.home()
        safe_cwd = cwd.replace("/", "-").replace("\\", "-").strip("-")
        sessions_dir = home / CONFIG_DIR_NAME / "sessions" / safe_cwd
        sessions_dir.mkdir(parents=True, exist_ok=True)
        sessions_dir.chmod(0o700)
        return sessions_dir

    def __init__(
        self,
        session_id: str,
        cwd: str,
        session_file: Path | None = None,
        persist: bool = True,
        initial_provider: str | None = None,
        initial_model_id: str | None = None,
        initial_thinking_level: str = "high",
    ):
        self._id = session_id
        self._cwd = cwd
        self._session_file = session_file
        self._persist = persist

        # In-memory state
        self._header: SessionHeader | None = None
        self._entries: list[SessionEntry] = []
        self._by_id: dict[str, SessionEntry] = {}
        self._leaf_id: str | None = None

        # Initial settings (used as fallback when no entries exist)
        self._initial_provider = initial_provider
        self._initial_model_id = initial_model_id
        self._initial_thinking_level = initial_thinking_level

        # Track disk persistence state
        self._flushed = False
        self._persisted_entries_count = 0

    @property
    def id(self) -> str:
        return self._id

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def session_file(self) -> Path | None:
        return self._session_file

    @property
    def system_prompt(self) -> str | None:
        return self._header.system_prompt if self._header else None

    @property
    def tools(self) -> list[str] | None:
        return self._header.tools if self._header else None

    @property
    def created_at(self) -> str | None:
        return self._header.timestamp if self._header else None

    @property
    def leaf_id(self) -> str | None:
        return self._leaf_id

    def _generate_entry_id(self) -> str:
        for _ in range(100):
            entry_id = self.generate_id()
            if entry_id not in self._by_id:
                return entry_id
        return uuid.uuid4().hex

    def _append_entry(self, entry: SessionEntry) -> None:
        self._entries.append(entry)
        self._by_id[entry.id] = entry
        self._leaf_id = entry.id
        self._persist_entry(entry)

    def _persist_entry(self, entry: SessionEntry) -> None:
        if not self._persist or not self._session_file:
            return

        has_assistant = any(
            isinstance(e, MessageEntry) and e.message.role == "assistant" for e in self._entries
        )
        if not has_assistant:
            return

        # If earlier entries were skipped (e.g., pre-assistant user/custom messages),
        # rewrite to include the full sequence before appending incrementally again.
        if self._persisted_entries_count < len(self._entries) - 1:
            self._write_all()
            self._flushed = True
            self._persisted_entries_count = len(self._entries)
            return

        if not self._flushed:
            self._write_all()
            self._flushed = True
            self._persisted_entries_count = len(self._entries)
        else:
            with open(self._session_file, "a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")
            self._persisted_entries_count += 1

    def _write_all(self) -> None:
        if not self._session_file:
            return

        self._session_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self._session_file, "w", encoding="utf-8") as f:
            if self._header:
                f.write(self._header.model_dump_json() + "\n")
            for entry in self._entries:
                f.write(entry.model_dump_json() + "\n")

    def ensure_persisted(self) -> None:
        if not self._persist or not self._session_file:
            return
        self._write_all()
        self._flushed = True
        self._persisted_entries_count = len(self._entries)

    def append_message(self, message: Message) -> str:
        entry = MessageEntry(
            id=self._generate_entry_id(),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
            message=message,
        )
        self._append_entry(entry)
        return entry.id

    def append_thinking_level_change(self, thinking_level: str) -> str:
        entry = ThinkingLevelChangeEntry(
            id=self._generate_entry_id(),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
            thinking_level=thinking_level,
        )
        self._append_entry(entry)
        return entry.id

    def append_model_change(
        self, provider: str, model_id: str, base_url: str | None = None
    ) -> str:
        entry = ModelChangeEntry(
            id=self._generate_entry_id(),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
            provider=provider,
            model_id=model_id,
            base_url=base_url,
        )
        self._append_entry(entry)
        return entry.id

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: dict[str, Any] | None = None,
    ) -> str:
        entry = CompactionEntry(
            id=self._generate_entry_id(),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
            summary=summary,
            first_kept_entry_id=first_kept_entry_id,
            tokens_before=tokens_before,
            details=details,
        )
        self._append_entry(entry)
        return entry.id

    def append_custom_message(
        self,
        custom_type: str,
        content: str,
        display: bool = True,
        details: dict[str, Any] | None = None,
    ) -> str:
        entry = CustomMessageEntry(
            id=self._generate_entry_id(),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
            custom_type=custom_type,
            content=content,
            display=display,
            details=details,
        )
        self._append_entry(entry)
        return entry.id

    def append_session_info(self, name: str) -> str:
        entry = SessionInfoEntry(
            id=self._generate_entry_id(), parent_id=self._leaf_id, timestamp=_now_iso(), name=name
        )
        self._append_entry(entry)
        return entry.id

    @property
    def entries(self) -> list[SessionEntry]:
        return list(self._entries)

    def get_entry(self, entry_id: str) -> SessionEntry | None:
        return self._by_id.get(entry_id)

    @property
    def messages(self) -> list[Message]:
        """Messages for LLM context. If compaction exists, returns compacted view."""
        last_compaction: CompactionEntry | None = None
        for entry in reversed(self._entries):
            if isinstance(entry, CompactionEntry):
                last_compaction = entry
                break

        if last_compaction is None:
            return [e.message for e in self._entries if isinstance(e, MessageEntry)]

        # Build compacted message list:
        # 1. Synthetic user message asking "what did we do so far?"
        # 2. Assistant message with the compaction summary
        # 3. All MessageEntry entries after the compaction entry
        result: list[Message] = [
            UserMessage(content="What did we do so far?"),
            AssistantMessage(
                content=[TextContent(text=last_compaction.summary)], stop_reason=StopReason.STOP
            ),
        ]

        # Find the compaction entry's position and include messages after it
        past_compaction = False
        for entry in self._entries:
            if isinstance(entry, CompactionEntry) and entry.id == last_compaction.id:
                past_compaction = True
                continue
            if past_compaction and isinstance(entry, MessageEntry):
                result.append(entry.message)

        return result

    @property
    def all_messages(self) -> list[Message]:
        """All messages regardless of compaction (for UI rendering)."""
        return [e.message for e in self._entries if isinstance(e, MessageEntry)]

    def get_last_assistant_text(self) -> str | None:
        for message in reversed(self.messages):
            if not isinstance(message, AssistantMessage):
                continue
            if message.stop_reason == StopReason.INTERRUPTED and not message.content:
                continue

            text = "".join(
                part.text for part in message.content if isinstance(part, TextContent)
            ).strip()
            return text or None

        return None

    def token_totals(self) -> SessionTokenTotals:
        input_tokens = 0
        output_tokens = 0
        cache_read_tokens = 0
        cache_write_tokens = 0
        context_tokens = 0

        for entry in self._entries:
            if isinstance(entry, MessageEntry) and isinstance(entry.message, AssistantMessage):
                usage = entry.message.usage
                if usage is None:
                    continue
                input_tokens += usage.input_tokens
                output_tokens += usage.output_tokens
                cache_read_tokens += usage.cache_read_tokens
                cache_write_tokens += usage.cache_write_tokens
                context_tokens = (
                    usage.input_tokens
                    + usage.output_tokens
                    + usage.cache_read_tokens
                    + usage.cache_write_tokens
                )

        return SessionTokenTotals(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            context_tokens=context_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )

    def file_changes_summary(self) -> dict[str, tuple[int, int]]:
        file_changes: dict[str, tuple[int, int]] = {}
        for entry in self._entries:
            if isinstance(entry, MessageEntry) and isinstance(entry.message, ToolResultMessage):
                fc = entry.message.file_changes
                if fc:
                    prev_added, prev_removed = file_changes.get(fc.path, (0, 0))
                    file_changes[fc.path] = (prev_added + fc.added, prev_removed + fc.removed)
        return file_changes

    def message_counts(self) -> SessionMessageCounts:
        user_messages = 0
        assistant_messages = 0
        tool_calls = 0
        tool_results = 0

        for entry in self._entries:
            if not isinstance(entry, MessageEntry):
                continue
            message = entry.message
            if isinstance(message, UserMessage):
                user_messages += 1
            elif isinstance(message, AssistantMessage):
                assistant_messages += 1
                tool_calls += sum(1 for part in message.content if isinstance(part, ToolCall))
            elif isinstance(message, ToolResultMessage):
                tool_results += 1

        return SessionMessageCounts(
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            tool_calls=tool_calls,
            tool_results=tool_results,
        )

    @property
    def name(self) -> str | None:
        for entry in reversed(self._entries):
            if isinstance(entry, SessionInfoEntry) and entry.name:
                return entry.name
        return None

    @property
    def thinking_level(self) -> str:
        for entry in reversed(self._entries):
            if isinstance(entry, ThinkingLevelChangeEntry):
                return entry.thinking_level
        return self._initial_thinking_level

    @property
    def model(self) -> tuple[str, str, str | None] | None:
        for entry in reversed(self._entries):
            if isinstance(entry, ModelChangeEntry):
                return (entry.provider, entry.model_id, entry.base_url)

        if self._initial_provider and self._initial_model_id:
            return (self._initial_provider, self._initial_model_id, None)
        return None

    def set_model(self, provider: str, model_id: str, base_url: str | None = None) -> None:
        current = self.model
        if (
            current
            and current[0] == provider
            and current[1] == model_id
            and current[2] == base_url
        ):
            return
        self.append_model_change(provider, model_id, base_url)

    def set_thinking_level(self, thinking_level: str) -> None:
        if self.thinking_level == thinking_level:
            return
        self.append_thinking_level_change(thinking_level)

    @classmethod
    def create(
        cls,
        cwd: str,
        persist: bool = True,
        provider: str | None = None,
        model_id: str | None = None,
        thinking_level: str = "high",
        system_prompt: str | None = None,
        tools: list[str] | None = None,
    ) -> Session:
        session_id = str(uuid.uuid4())
        timestamp = _now_iso()

        session = cls(
            session_id=session_id,
            cwd=cwd,
            persist=persist,
            initial_provider=provider,
            initial_model_id=model_id,
            initial_thinking_level=thinking_level,
        )
        session._header = SessionHeader(
            id=session_id,
            timestamp=timestamp,
            cwd=cwd,
            system_prompt=system_prompt,
            tools=tools,
            initial_thinking_level=thinking_level,
        )

        if persist:
            file_timestamp = datetime.fromisoformat(timestamp).strftime("%Y-%m-%dT%H-%M-%S")
            sessions_dir = cls.get_sessions_dir(cwd)
            session._session_file = sessions_dir / f"{file_timestamp}_{session_id}.jsonl"

        return session

    @classmethod
    def load(cls, path: Path | str) -> Session:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")

        header: SessionHeader | None = None
        entries: list[SessionEntry] = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = data.get("type")

                if entry_type == "header":
                    header = SessionHeader.model_validate(data)
                elif entry_type == "message":
                    entries.append(MessageEntry.model_validate(data))
                elif entry_type == "thinking_level_change":
                    entries.append(ThinkingLevelChangeEntry.model_validate(data))
                elif entry_type == "model_change":
                    entries.append(ModelChangeEntry.model_validate(data))
                elif entry_type == "compaction":
                    entries.append(CompactionEntry.model_validate(data))
                elif entry_type == "custom_message":
                    entries.append(CustomMessageEntry.model_validate(data))
                elif entry_type == "session_info":
                    entries.append(SessionInfoEntry.model_validate(data))

        if not header:
            raise ValueError(f"Invalid session file (no header): {path}")

        session = cls(
            session_id=header.id,
            cwd=header.cwd,
            session_file=path,
            persist=True,
            initial_thinking_level=header.initial_thinking_level,
        )
        session._header = header
        session._entries = entries
        session._by_id = {e.id: e for e in entries}
        session._leaf_id = entries[-1].id if entries else None
        session._flushed = True  # Already on disk
        session._persisted_entries_count = len(entries)

        return session

    @classmethod
    def continue_recent(
        cls,
        cwd: str,
        provider: str | None = None,
        model_id: str | None = None,
        thinking_level: str = "high",
        system_prompt: str | None = None,
    ) -> Session:
        sessions_dir = cls.get_sessions_dir(cwd)

        jsonl_files = list(sessions_dir.glob("*.jsonl"))
        if not jsonl_files:
            return cls.create(
                cwd,
                provider=provider,
                model_id=model_id,
                thinking_level=thinking_level,
                system_prompt=system_prompt,
            )

        most_recent = max(jsonl_files, key=lambda p: p.stat().st_mtime)
        return cls.load(most_recent)

    @classmethod
    def continue_by_id(cls, cwd: str, session_id: str) -> Session:
        normalized_id = session_id.strip().lower()
        if not normalized_id:
            raise ValueError("Session ID cannot be empty")

        sessions = cls.list(cwd)
        exact_matches = [s for s in sessions if s.id.lower() == normalized_id]
        if len(exact_matches) == 1:
            return cls.load(exact_matches[0].path)

        prefix_matches = [s for s in sessions if s.id.lower().startswith(normalized_id)]
        if len(prefix_matches) == 1:
            return cls.load(prefix_matches[0].path)
        if len(prefix_matches) > 1:
            raise ValueError(f"Session ID prefix is ambiguous: {session_id}")

        raise FileNotFoundError(f"Session not found: {session_id}")

    @classmethod
    def list(cls, cwd: str) -> list[SessionInfo]:
        sessions_dir = cls.get_sessions_dir(cwd)
        if not sessions_dir.exists():
            return []

        sessions: list[SessionInfo] = []

        for path in sessions_dir.glob("*.jsonl"):
            try:
                info = cls.build_session_info(path)
                if info:
                    sessions.append(info)
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                continue

        sessions.sort(key=lambda s: s.modified, reverse=True)
        return sessions

    @staticmethod
    def _extract_preview_from_user_message(content: str) -> str:
        text = content.strip()
        if not text:
            return ""

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return text

        header_match = _SKILL_TRIGGER_HEADER_RE.match(lines[0])
        if not header_match:
            return text

        skill_name = header_match.group(1)
        query_marker_index = next(
            (i for i, line in enumerate(lines[1:], start=1) if line.lower() == "[query]"), -1
        )
        if query_marker_index == -1:
            return f"/{skill_name}"

        query = " ".join(lines[query_marker_index + 1 :]).strip()
        if not query:
            return f"/{skill_name}"

        return f"/{skill_name} {query}"

    @classmethod
    def build_session_info(cls, path: Path) -> SessionInfo | None:
        header: SessionHeader | None = None
        message_count = 0
        first_message = ""

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "header":
                    header = SessionHeader.model_validate(data)
                elif data.get("type") == "message":
                    message_count += 1
                    msg = data.get("message", {})
                    if msg.get("role") == "user" and not first_message:
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            first_message = cls._extract_preview_from_user_message(content)[:100]
                        elif isinstance(content, list) and content:
                            first_item = content[0]
                            if isinstance(first_item, dict) and first_item.get("type") == "text":
                                first_message = cls._extract_preview_from_user_message(
                                    first_item.get("text", "")
                                )[:100]

        if not header:
            return None

        stat = path.stat()
        return SessionInfo(
            id=header.id,
            path=path,
            cwd=header.cwd,
            created=datetime.fromisoformat(header.timestamp),
            modified=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            message_count=message_count,
            first_message=first_message or "(no messages)",
        )

    @classmethod
    def in_memory(
        cls,
        cwd: str = ".",
        provider: str | None = None,
        model_id: str | None = None,
        thinking_level: str = "high",
        system_prompt: str | None = None,
        tools: list[str] | None = None,
    ) -> Session:
        return cls.create(
            cwd,
            persist=False,
            provider=provider,
            model_id=model_id,
            thinking_level=thinking_level,
            system_prompt=system_prompt,
            tools=tools,
        )
