import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal

from .core.types import AssistantMessage, FileChanges, StopReason, ToolResultMessage, Usage
from .permissions import ApprovalResponse

# =================================================================================================
# Agent Lifecycle Events
# =================================================================================================


@dataclass
class AgentStartEvent:
    type: Literal["agent_start"] = "agent_start"


@dataclass
class AgentEndEvent:
    type: Literal["agent_end"] = "agent_end"
    stop_reason: StopReason = StopReason.STOP
    total_turns: int = 0
    total_usage: Usage | None = None


# =================================================================================================
# Turn Lifecycle Events
# =================================================================================================


@dataclass
class TurnStartEvent:
    type: Literal["turn_start"] = "turn_start"
    turn: int = 0


@dataclass
class TurnEndEvent:
    type: Literal["turn_end"] = "turn_end"
    turn: int = 0
    assistant_message: AssistantMessage | None = None
    tool_results: list[ToolResultMessage] = field(default_factory=list)
    stop_reason: StopReason = StopReason.STOP


# =================================================================================================
# Content Streaming Events
# =================================================================================================


@dataclass
class ThinkingStartEvent:
    type: Literal["thinking_start"] = "thinking_start"


@dataclass
class ThinkingDeltaEvent:
    type: Literal["thinking_delta"] = "thinking_delta"
    delta: str = ""


@dataclass
class ThinkingEndEvent:
    type: Literal["thinking_end"] = "thinking_end"
    thinking: str = ""
    signature: str | None = None


@dataclass
class TextStartEvent:
    type: Literal["text_start"] = "text_start"


@dataclass
class TextDeltaEvent:
    type: Literal["text_delta"] = "text_delta"
    delta: str = ""


@dataclass
class TextEndEvent:
    type: Literal["text_end"] = "text_end"
    text: str = ""


# =================================================================================================
# Tool Events
# =================================================================================================


@dataclass
class ToolStartEvent:
    type: Literal["tool_start"] = "tool_start"
    tool_call_id: str = ""
    tool_name: str = ""


@dataclass
class ToolArgsDeltaEvent:
    type: Literal["tool_args_delta"] = "tool_args_delta"
    tool_call_id: str = ""
    delta: str = ""


@dataclass
class ToolArgsTokenUpdateEvent:
    type: Literal["tool_args_token_update"] = "tool_args_token_update"
    tool_call_id: str = ""
    tool_name: str = ""
    token_count: int = 0


@dataclass
class ToolEndEvent:
    type: Literal["tool_end"] = "tool_end"
    tool_call_id: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    display: str = ""  # Formatted display string from tool.format_call()


@dataclass
class ToolResultEvent:
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str = ""
    tool_name: str = ""
    result: ToolResultMessage | None = None
    file_changes: FileChanges | None = None


@dataclass
class ToolApprovalEvent:
    type: Literal["tool_approval"] = "tool_approval"
    tool_call_id: str = ""
    tool_name: str = ""
    future: asyncio.Future[ApprovalResponse] | None = None


# =================================================================================================
# Compaction Events
# =================================================================================================


@dataclass
class CompactionStartEvent:
    type: Literal["compaction_start"] = "compaction_start"


@dataclass
class CompactionEndEvent:
    type: Literal["compaction_end"] = "compaction_end"
    tokens_before: int = 0
    aborted: bool = False


# =================================================================================================
# Other Events
# =================================================================================================


@dataclass
class RetryEvent:
    type: Literal["retry"] = "retry"
    attempt: int = 0
    total_attempts: int = 3
    delay: float = 0.0
    error: str = ""


@dataclass
class ErrorEvent:
    type: Literal["error"] = "error"
    error: str = ""


@dataclass
class WarningEvent:
    type: Literal["warning"] = "warning"
    warning: str = ""


@dataclass
class InterruptedEvent:
    type: Literal["interrupted"] = "interrupted"
    message: str = "Interrupted by user"


# =================================================================================================
# Union Types
# =================================================================================================

# Events yielded by run_single_turn (loop level)
StreamEvent = (
    ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ToolStartEvent
    | ToolArgsDeltaEvent
    | ToolArgsTokenUpdateEvent
    | ToolEndEvent
    | ToolResultEvent
    | ToolApprovalEvent
    | RetryEvent
    | TurnEndEvent
    | ErrorEvent
    | WarningEvent
    | InterruptedEvent
)

# All events yielded by Agent.run()
Event = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | CompactionStartEvent
    | CompactionEndEvent
    | StreamEvent
)
