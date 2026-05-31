import os
import sys
from collections.abc import AsyncIterator
from typing import TextIO

from kon import config, get_config

from .core.types import StopReason, TextContent
from .events import AgentEndEvent, ErrorEvent, Event, ToolApprovalEvent, TurnEndEvent
from .llm.base import AuthMode
from .permissions import ApprovalResponse
from .runtime import ConversationRuntime
from .tools import DEFAULT_TOOLS, EXTRA_TOOLS, get_tools

_EXIT_CODES = {StopReason.STOP: 0, StopReason.ERROR: 1, StopReason.LENGTH: 3}


def _exit_code(stop: StopReason) -> int:
    return _EXIT_CODES.get(stop, 1)


def resolve_prompt(prompt_arg: str, *, stdin: TextIO) -> str:
    if prompt_arg == "-":
        return stdin.read().strip()
    return prompt_arg.strip()


async def render_run(
    events: AsyncIterator[Event], *, out: TextIO | None = None, err: TextIO | None = None
) -> StopReason:
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    final_text = ""
    stop = StopReason.ERROR
    async for event in events:
        match event:
            case TurnEndEvent(assistant_message=msg) if msg is not None:
                text = "".join(p.text for p in msg.content if isinstance(p, TextContent)).strip()
                if text:
                    final_text = text
            case AgentEndEvent(stop_reason=stop_reason):
                stop = stop_reason
            case ErrorEvent(error=error):
                print(f"error: {error}", file=err)
            case ToolApprovalEvent(tool_name=tool_name, future=future) if future is not None:
                future.set_result(ApprovalResponse.DENY)
                print(
                    f"error: {tool_name!r} requires approval, denied (non-interactive mode)",
                    file=err,
                )
            case _:
                pass
    if stop == StopReason.STOP and final_text:
        print(final_text, file=out)
    return stop


async def run_headless(
    *,
    prompt_arg: str,
    model: str | None,
    provider: str | None,
    api_key: str | None,
    base_url: str | None,
    openai_compat_auth_mode: AuthMode | None,
    anthropic_compat_auth_mode: AuthMode | None,
    extra_tools: list[str] | None,
) -> int:
    prompt = resolve_prompt(prompt_arg, stdin=sys.stdin)
    if not prompt:
        print("error: empty prompt", file=sys.stderr)
        return 2

    cfg = get_config()
    previous_permission_mode = cfg.permissions.mode
    # Headless can't show approval prompts; force auto in-memory for this run only.
    cfg.permissions.mode = "auto"

    try:
        initial_model = model or config.llm.default_model
        initial_provider = (
            provider
            if provider is not None
            else (config.llm.default_provider if model is None else None)
        )
        base = base_url or config.llm.default_base_url or None
        thinking = config.llm.default_thinking_level
        openai_auth = openai_compat_auth_mode or config.llm.auth.openai_compat
        anthropic_auth = anthropic_compat_auth_mode or config.llm.auth.anthropic_compat

        merged = list(dict.fromkeys(config.tools.extra + (extra_tools or [])))
        for name in merged:
            if name not in EXTRA_TOOLS:
                print(f"warning: unknown extra tool: {name!r}", file=sys.stderr)
        extras = [n for n in merged if n in EXTRA_TOOLS]
        tools = get_tools(DEFAULT_TOOLS + extras)

        runtime = ConversationRuntime(
            cwd=os.getcwd(),
            model=initial_model,
            model_provider=initial_provider,
            api_key=api_key,
            base_url=base,
            thinking_level=thinking,
            tools=tools,
            openai_compat_auth_mode=openai_auth,
            anthropic_compat_auth_mode=anthropic_auth,
        )

        try:
            init = runtime.initialize()
            if init.provider_error:
                print(f"error: {init.provider_error}", file=sys.stderr)
                return 2

            agent = runtime.prepare_for_run()
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

        if agent is None:
            print("error: agent initialization failed", file=sys.stderr)
            return 2

        return _exit_code(await render_run(agent.run(prompt)))
    finally:
        cfg.permissions.mode = previous_permission_mode
