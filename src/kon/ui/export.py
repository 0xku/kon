import html
import json
import re
from contextlib import suppress
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from ..core.types import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from ..session import (
    CompactionEntry,
    CustomMessageEntry,
    MessageEntry,
    ModelChangeEntry,
    Session,
    SessionEntry,
    ThinkingLevelChangeEntry,
)
from ..tools import tools_by_name

MAX_RESULT_LINES = 20

_CSS = """\
:root {
  --bg0: #282828; --bg1: #3c3836; --bg2: #504945;
  --fg: #ebdbb2; --fg2: #bdae93; --fg3: #a89984; --fg4: #928374;
  --red: #fb4934; --green: #b8bb26; --yellow: #fabd2f;
  --blue: #83a598; --orange: #fe8019;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: var(--bg0);
  color: var(--fg);
  font-family: 'SF Mono', 'JetBrains Mono', 'Menlo', monospace;
  font-size: 12px;
  line-height: 1.5;
  padding: 24px;
  max-width: 960px;
  margin: 0 auto;
}
a { color: var(--blue); }
.header {
  border-bottom: 1px solid var(--bg2);
  padding-bottom: 12px;
  margin-bottom: 14px;
}
.header h1 { font-size: 14px; color: var(--orange); font-weight: 600; }
.header .meta { color: var(--fg4); font-size: 11px; margin-top: 4px; }
.msg-user {
  color: var(--green);
  white-space: pre-wrap;
}
.msg-assistant .text {
  color: var(--fg);
  white-space: pre-wrap;
}
.msg-assistant > * + * {
  margin-top: 6px;
}
.thinking {
  color: var(--fg4);
  font-style: italic;
  font-size: 11px;
  white-space: pre-wrap;
}
.system-msg {
  color: var(--fg4);
  font-style: italic;
  font-size: 11px;
  white-space: pre-wrap;
  margin: 8px 0;
}
.sep {
  border-top: 1px solid var(--bg2);
  margin: 8px 0;
}
.tool-header { color: var(--yellow); font-weight: 600; }
.tool-args { color: var(--fg2); }
.tool-result {
  color: var(--fg3);
  white-space: pre-wrap;
  font-size: 11px;
  overflow-x: auto;
  max-height: 120px;
  overflow-y: auto;
  background: var(--bg1);
  padding: 6px 8px;
  border-radius: 3px;
  margin-top: 4px;
}
.tool-result.error { color: var(--red); }
"""

_RICH_TAG_RE = re.compile(r"\[/?(?:[a-zA-Z0-9#._-]+)\]")


def _strip_rich_markup(text: str) -> str:
    return _RICH_TAG_RE.sub("", text)


def _esc(text: str) -> str:
    return html.escape(_strip_rich_markup(text))


def _format_tool_call_args(tool_call: ToolCall | None) -> str:
    if tool_call is None:
        return ""

    tool = tools_by_name.get(tool_call.name)
    if not tool:
        return json.dumps(tool_call.arguments) if tool_call.arguments else ""

    try:
        params = tool.params(**tool_call.arguments)
        return tool.format_call(params)
    except (TypeError, KeyError, ValueError, ValidationError):
        return json.dumps(tool_call.arguments) if tool_call.arguments else ""


def _truncate(text: str, max_lines: int = MAX_RESULT_LINES) -> str:
    if not text:
        return text

    lines = text.split("\n")
    if len(lines) > max_lines:
        hidden = len(lines) - max_lines
        lines = lines[:max_lines]
        lines.append(f"... ({hidden} lines hidden)")
    return "\n".join(lines)


def _format_name(name: str) -> str:
    return " ".join(word.capitalize() for word in name.split("_"))


class HtmlBuilder:
    def __init__(self) -> None:
        self._parts: list[str] = []
        self._assistant_open = False
        self._last_block_kind: str | None = None

    def _append(self, text: str) -> None:
        self._parts.append(text)

    def _before_chat_block(self) -> None:
        self.close_assistant()
        if self._last_block_kind == "chat":
            self._append('<div class="sep"></div>')
        self._last_block_kind = "chat"

    def open_assistant(self) -> None:
        if not self._assistant_open:
            self._before_chat_block()
            self._append('<div class="msg msg-assistant">')
            self._assistant_open = True

    def close_assistant(self) -> None:
        if self._assistant_open:
            self._append("</div>")
            self._assistant_open = False

    def header(self, version: str, session: Session, model_id: str, provider: str) -> None:
        session_id = session.session_file.stem if session.session_file else session.id
        tokens = session.token_totals()
        token_parts = [f"↑{tokens.input_tokens:,}", f"↓{tokens.output_tokens:,}"]
        if tokens.cache_read_tokens:
            token_parts.append(f"R{tokens.cache_read_tokens:,}")
        if tokens.cache_write_tokens:
            token_parts.append(f"W{tokens.cache_write_tokens:,}")

        model_str = model_id if provider == "unknown" else f"{model_id} ({provider})"
        created = session.created_at or "unknown"
        if "T" in created:
            with suppress(ValueError):
                created = datetime.fromisoformat(created).strftime("%Y-%m-%d %H:%M")

        self._append('<div class="header">')
        self._append(f"<h1>kon {_esc(version)}</h1>")
        self._append(
            f'<div class="meta">session {session_id[:8]}'
            f" · {_esc(created)} · {_esc(model_str)}"
            f" · {' '.join(token_parts)}</div>"
        )
        self._append("</div>")

    def user_message(self, msg: UserMessage) -> None:
        self._before_chat_block()
        parts: list[str] = []
        if isinstance(msg.content, str):
            parts.append(_esc(msg.content))
        else:
            for part in msg.content:
                if isinstance(part, TextContent):
                    parts.append(_esc(part.text))
                elif isinstance(part, ImageContent):
                    parts.append('<span style="color:var(--fg4)">[image]</span>')
        self._append(f'<div class="msg-user">&gt; {"".join(parts)}</div>')

    def assistant_text(self, text: str) -> None:
        self.open_assistant()
        self._append(f'<div class="text">{_esc(text)}</div>')

    def thinking(self, text: str) -> None:
        self.open_assistant()
        self._append(f'<div class="thinking">{_esc(text)}</div>')

    def tool_block(self, name: str, args: str, result_text: str = "", error: bool = False) -> None:
        self.open_assistant()
        self._append('<div class="tool-block">')
        if args:
            self._append(
                f'<div class="tool-header">{_esc(name)}'
                f' <span class="tool-args">{_esc(args)}</span></div>'
            )
        else:
            self._append(f'<div class="tool-header">{_esc(name)}</div>')

        if result_text:
            klass = "tool-result error" if error else "tool-result"
            self._append(f'<div class="{klass}">{_esc(result_text)}</div>')
        self._append("</div>")

    def system_message(self, text: str) -> None:
        self.close_assistant()
        self._last_block_kind = "system"
        self._append(f'<div class="system-msg">{_esc(text)}</div>')

    def build(self) -> str:
        self.close_assistant()
        body = "\n".join(self._parts)
        return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>kon - Export</title>
<style>
{_CSS}
</style>
</head>
<body>
{body}
</body>
</html>"""


class ExportRenderer:
    def __init__(self, builder: HtmlBuilder) -> None:
        self.builder = builder
        self.pending_tool_calls: dict[str, ToolCall] = {}
        self.in_assistant_turn = False

    def _flush_pending_tool_calls(self) -> None:
        for tool_call in self.pending_tool_calls.values():
            self.builder.tool_block(
                _format_name(tool_call.name), _format_tool_call_args(tool_call)
            )
        self.pending_tool_calls.clear()

    def _end_assistant_turn(self) -> None:
        if not self.in_assistant_turn:
            return
        self._flush_pending_tool_calls()
        self.builder.close_assistant()
        self.in_assistant_turn = False

    def render_entry(self, entry: SessionEntry) -> None:
        if isinstance(entry, MessageEntry):
            msg = entry.message

            if isinstance(msg, UserMessage):
                self._end_assistant_turn()
                self.builder.user_message(msg)
                return

            if isinstance(msg, AssistantMessage):
                self.in_assistant_turn = True
                for part in msg.content:
                    if isinstance(part, TextContent) and part.text:
                        self.builder.assistant_text(part.text)
                    elif isinstance(part, ThinkingContent) and part.thinking:
                        self.builder.thinking(part.thinking)
                    elif isinstance(part, ToolCall):
                        self.pending_tool_calls[part.id] = part
                return

            if isinstance(msg, ToolResultMessage):
                self.in_assistant_turn = True
                tool_call = self.pending_tool_calls.pop(msg.tool_call_id, None)
                name = _format_name(tool_call.name) if tool_call else _format_name(msg.tool_name)
                args = _format_tool_call_args(tool_call)

                if msg.is_error:
                    text = "".join(
                        part.text for part in msg.content if isinstance(part, TextContent)
                    ).strip()
                    result = f"-- {text} --" if text else ""
                    self.builder.tool_block(name, args, result_text=result, error=True)
                    return

                if msg.ui_details:
                    result = _truncate(msg.ui_details)
                else:
                    parts = [part.text for part in msg.content if isinstance(part, TextContent)]
                    result = _truncate("".join(parts))
                self.builder.tool_block(name, args, result_text=result)
                return

        self._end_assistant_turn()

        if isinstance(entry, ModelChangeEntry):
            self.builder.system_message(f"Model changed to {entry.model_id} ({entry.provider})")
        elif isinstance(entry, ThinkingLevelChangeEntry):
            self.builder.system_message(f"Thinking level: {entry.thinking_level}")
        elif isinstance(entry, CompactionEntry):
            self.builder.system_message("Context compacted")
        elif isinstance(entry, CustomMessageEntry) and entry.display:
            self.builder.system_message(entry.content)

    def finish(self) -> None:
        self._end_assistant_turn()


def export_session_html(
    session: Session,
    output_dir: str,
    model_id: str = "unknown",
    provider: str = "unknown",
    version: str = "",
) -> Path:
    builder = HtmlBuilder()
    builder.header(version, session, model_id, provider)
    if session.system_prompt:
        builder.system_message(session.system_prompt)

    renderer = ExportRenderer(builder)
    for entry in session.entries:
        renderer.render_entry(entry)
    renderer.finish()

    filename = (
        f"kon-session-{session.session_file.stem if session.session_file else session.id}.html"
    )
    output_path = Path(output_dir) / filename
    output_path.write_text(builder.build(), encoding="utf-8")
    return output_path
