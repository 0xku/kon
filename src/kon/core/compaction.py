"""
Context compaction for long sessions.

When token usage exceeds the usable context window, send the full conversation
to the LLM with a summarization prompt, then store the summary as a
CompactionEntry. The session.messages property filters to only show messages
after the compaction point.

Overflow formula:
    total_tokens >= context_window - min(buffer_tokens, max_output_tokens)
"""

from ..core.types import Message, TextPart, Usage, UserMessage
from ..llm.base import BaseProvider

SUMMARIZATION_PROMPT = """Provide a detailed prompt for continuing our \
conversation above. Focus on information that would be helpful for \
continuing the conversation, including what we did, what we're doing, \
which files we're working on, and what we're going to do next. \
The summary that you construct will be used so that another agent \
can read it and continue the work.

When constructing the summary, try to stick to this template:
---
## Goal

[What goal(s) is the user trying to accomplish?]

## Instructions

- [What important instructions did the user give you that are relevant]
- [If there is a plan or spec, include information about it
  so next agent can continue using it]

## Discoveries

[What notable things were learned during this conversation that would
be useful for the next agent to know when continuing the work]

## Accomplished

[What work has been completed, what work is still in progress,
and what work is left?]

## Relevant files / directories

[Construct a structured list of relevant files that have been read,
edited, or created that pertain to the task at hand. If all the files
in a directory are relevant, include the path to the directory.]
---"""


# Margin reserved on top of the estimated summary input to absorb the
# summarization prompt and tokenization rounding so the summary request can't
# spill a token over the context window (e.g. a 1-token 400 against vLLM).
_SUMMARY_MARGIN = 256


def summary_max_tokens(
    context_window: int | None, tokens_before: int, provider_max_tokens: int | None
) -> int | None:
    """Clamp the summary request's output budget so input + output fits the window.

    ``tokens_before`` approximates the conversation size that gets re-sent as
    input for summarization. Returns ``None`` (use the provider default) when the
    window is unknown or already exhausted, since nothing sensible can be done
    there without chunked compaction.
    """
    if context_window is None:
        return None
    budget = context_window - tokens_before - _SUMMARY_MARGIN
    if budget <= 0:
        return None
    return min(provider_max_tokens, budget) if provider_max_tokens else budget


def is_overflow(
    usage: Usage, context_window: int, max_output_tokens: int, buffer_tokens: int
) -> bool:
    count = (
        usage.input_tokens
        + usage.output_tokens
        + usage.cache_read_tokens
        + usage.cache_write_tokens
    )
    reserved = min(buffer_tokens, max_output_tokens)
    usable = context_window - reserved
    return count >= usable


def _calculate_context_tokens(usage: Usage) -> int:
    return (
        usage.input_tokens
        + usage.output_tokens
        + usage.cache_read_tokens
        + usage.cache_write_tokens
    )


async def generate_summary(
    messages: list[Message],
    provider: BaseProvider,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Send the full conversation + summarization prompt to the LLM, return summary text."""
    summary_messages: list[Message] = [*messages, UserMessage(content=SUMMARIZATION_PROMPT)]

    stream = await provider.stream(
        summary_messages, system_prompt=system_prompt, tools=None, max_tokens=max_tokens
    )

    text_parts: list[str] = []
    async for part in stream:
        if isinstance(part, TextPart):
            text_parts.append(part.text)

    return "".join(text_parts)
