from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel


class StopReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_USE = "tool_use"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    STEER = "steer"


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
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


# =================================================================================================
# Stream Parts - yielded by providers during streaming
# =================================================================================================


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str

    def merge(self, other: TextPart) -> TextPart:
        return TextPart(text=self.text + other.text)


class ThinkPart(BaseModel):
    type: Literal["think"] = "think"
    think: str
    signature: str | None = None

    def merge(self, other: ThinkPart) -> ThinkPart:
        signature = self.signature or other.signature
        return ThinkPart(think=self.think + other.think, signature=signature)


class ToolCallStart(BaseModel):
    type: Literal["tool_call_start"] = "tool_call_start"
    id: str
    name: str
    index: int  # Tool call index for correlating deltas
    arguments: dict[str, Any] | None = None


class ToolCallDelta(BaseModel):
    type: Literal["tool_call_delta"] = "tool_call_delta"
    index: int  # Correlates with ToolCallStart.index
    arguments_delta: str
    replace: bool = False


class StreamDone(BaseModel):
    type: Literal["done"] = "done"
    stop_reason: StopReason


class StreamError(BaseModel):
    type: Literal["error"] = "error"
    error: str


StreamPart = TextPart | ThinkPart | ToolCallStart | ToolCallDelta | StreamDone | StreamError


# =================================================================================================
# Message Types - canonical provider-agnostic conversation interface
# Used for conversation history and cross-provider normalization
# =================================================================================================


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None


class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    data: str  # base64 encoded
    mime_type: str
    display_name: str | None = None


class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any]


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[TextContent | ImageContent]


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[TextContent | ThinkingContent | ToolCall]
    usage: Usage | None = None
    stop_reason: StopReason | None = None


class ToolResultMessage(BaseModel):
    role: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: list[TextContent | ImageContent]
    ui_summary: str | None = None  # One-line UI text rendered on tool header line
    ui_details: str | None = None  # Collapsed multiline UI text rendered below the header
    ui_details_full: str | None = None  # Expanded multiline UI text rendered below the header
    is_error: bool = False
    file_changes: FileChanges | None = None


Message = UserMessage | AssistantMessage | ToolResultMessage


# =================================================================================================
# Tool Definition
# =================================================================================================


class ToolParameter(BaseModel):
    type: str
    description: str | None = None
    enum: list[str] | None = None


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


class FileChanges(BaseModel):
    path: str
    added: int = 0
    removed: int = 0


class ToolResult(BaseModel):
    success: bool
    result: str | None = None  # Raw result (sent to LLM)
    images: list[ImageContent] | None = None  # Images to include in result
    ui_summary: str | None = None  # One-line result text appended to the tool header
    ui_details: str | None = None  # Collapsed multiline result body rendered below the header
    ui_details_full: str | None = None  # Expanded multiline result body rendered below the header
    file_changes: FileChanges | None = None  # Track +/- lines for edit/write tools


# UI rendering model:
#
# format_call is defined for each tool like Read tool and the result they
# return contains further details (along with the resulf for llm) to help paint
# the coomplete picture (or as close to it as possible without polluting) in the ui
#
# - format_call(params): short call text shown on the header line
# - ui_summary: one-line result summary appended to the same header line
# - ui_details: multiline result body shown below the header
#
# Example (read):
#   → Read ~/src/kon/turn.py:150-204 (55 lines)
#   - format_call -> "~/src/kon/turn.py:150-204"
#   - ui_summary  -> "(55 lines)"
#   - ui_details  -> None
#
# Example (edit):
#   + Edit ~/src/kon/tools/base.py +3 -1
#     -12 old line
#     +12 new line
#   - format_call -> "~/src/kon/tools/base.py"
#   - ui_summary  -> "+3 -1"
#   - ui_details  -> formatted diff
