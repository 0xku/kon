import asyncio
from io import StringIO

import pytest

from kon import get_config
from kon.core.types import AssistantMessage, StopReason, TextContent
from kon.events import AgentEndEvent, ErrorEvent, ToolApprovalEvent, TurnEndEvent
from kon.headless import _exit_code, render_run, resolve_prompt, run_headless
from kon.llm.providers.mock import MockProvider
from kon.loop import Agent
from kon.permissions import ApprovalResponse
from kon.session import Session


async def _emit(events):
    for event in events:
        yield event


async def _run_headless(prompt_arg, *, extra_tools=None):
    return await run_headless(
        prompt_arg=prompt_arg,
        model=None,
        provider=None,
        api_key=None,
        base_url=None,
        openai_compat_auth_mode=None,
        anthropic_compat_auth_mode=None,
        extra_tools=extra_tools,
    )


def test_resolve_prompt_literal_strips():
    assert resolve_prompt("  hello  ", stdin=StringIO("ignored")) == "hello"


def test_resolve_prompt_reads_stdin_on_dash():
    assert resolve_prompt("-", stdin=StringIO("  from stdin  ")) == "from stdin"


def test_resolve_prompt_whitespace_only_is_empty():
    assert resolve_prompt("   ", stdin=StringIO("")) == ""


@pytest.mark.asyncio
async def test_render_run_prints_final_text_on_stop():
    out, err = StringIO(), StringIO()
    events = _emit(
        [
            TurnEndEvent(
                turn=1,
                assistant_message=AssistantMessage(
                    content=[TextContent(text="final answer")], stop_reason=StopReason.STOP
                ),
                tool_results=[],
                stop_reason=StopReason.STOP,
            ),
            AgentEndEvent(stop_reason=StopReason.STOP),
        ]
    )
    stop = await render_run(events, out=out, err=err)
    assert stop == StopReason.STOP
    assert out.getvalue() == "final answer\n"
    assert err.getvalue() == ""


@pytest.mark.asyncio
async def test_render_run_error_goes_to_stderr_only():
    out, err = StringIO(), StringIO()
    events = _emit([ErrorEvent(error="boom"), AgentEndEvent(stop_reason=StopReason.ERROR)])
    stop = await render_run(events, out=out, err=err)
    assert stop == StopReason.ERROR
    assert "error: boom" in err.getvalue()
    assert out.getvalue() == ""


@pytest.mark.asyncio
async def test_render_run_suppresses_text_when_not_stop():
    out, err = StringIO(), StringIO()
    events = _emit(
        [
            TurnEndEvent(
                turn=1,
                assistant_message=AssistantMessage(
                    content=[TextContent(text="partial")], stop_reason=StopReason.ERROR
                ),
                tool_results=[],
                stop_reason=StopReason.ERROR,
            ),
            AgentEndEvent(stop_reason=StopReason.ERROR),
        ]
    )
    stop = await render_run(events, out=out, err=err)
    assert stop == StopReason.ERROR
    assert out.getvalue() == ""


@pytest.mark.asyncio
async def test_render_run_composition_simple_text():
    agent = Agent(MockProvider(scenario="simple_text"), [], Session.in_memory())
    out, err = StringIO(), StringIO()
    stop = await render_run(agent.run("hi"), out=out, err=err)
    assert stop == StopReason.STOP
    assert out.getvalue() == "Hello, world!\n"


@pytest.mark.asyncio
async def test_render_run_composition_stream_error():
    agent = Agent(MockProvider(scenario="stream_error"), [], Session.in_memory())
    out, err = StringIO(), StringIO()
    stop = await render_run(agent.run("hi"), out=out, err=err)
    assert stop == StopReason.ERROR
    assert out.getvalue() == ""
    assert "error" in err.getvalue()


@pytest.mark.asyncio
async def test_run_headless_prints_and_restores_permissions(monkeypatch, capsys):
    get_config().permissions.mode = "prompt"
    monkeypatch.setattr(
        "kon.runtime.create_provider",
        lambda api_type, config: MockProvider(config, scenario="simple_text"),
    )
    code = await _run_headless("hi")
    assert code == 0
    assert get_config().permissions.mode == "prompt"
    assert "Hello, world!" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_run_headless_sets_auto_during_run_and_restores(monkeypatch):
    get_config().permissions.mode = "prompt"
    monkeypatch.setattr(
        "kon.runtime.create_provider",
        lambda api_type, config: MockProvider(config, scenario="simple_text"),
    )

    async def fake_render_run(events):
        assert get_config().permissions.mode == "auto"
        return StopReason.STOP

    monkeypatch.setattr("kon.headless.render_run", fake_render_run)

    code = await _run_headless("hi")
    assert code == 0
    assert get_config().permissions.mode == "prompt"


@pytest.mark.asyncio
async def test_run_headless_warns_on_unknown_extra_tool(monkeypatch, capsys):
    monkeypatch.setattr(
        "kon.runtime.create_provider",
        lambda api_type, config: MockProvider(config, scenario="simple_text"),
    )
    code = await _run_headless("hi", extra_tools=["bogus"])
    assert code == 0
    assert "unknown extra tool: 'bogus'" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_run_headless_empty_prompt_returns_2(capsys):
    code = await _run_headless("   ")
    assert code == 2
    assert "empty prompt" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_run_headless_init_exception_returns_2(capsys):
    code = await run_headless(
        prompt_arg="hi",
        model="definitely-not-a-model",
        provider="bogus",
        api_key=None,
        base_url=None,
        openai_compat_auth_mode=None,
        anthropic_compat_auth_mode=None,
        extra_tools=None,
    )

    assert code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_render_run_denies_tool_approval():
    future = asyncio.get_running_loop().create_future()
    out, err = StringIO(), StringIO()
    events = _emit(
        [
            ToolApprovalEvent(
                tool_call_id="t1", tool_name="bash", display="rm -rf /", future=future
            ),
            AgentEndEvent(stop_reason=StopReason.STOP),
        ]
    )
    stop = await render_run(events, out=out, err=err)
    assert future.result() == ApprovalResponse.DENY
    assert "requires approval" in err.getvalue()
    assert stop == StopReason.STOP


def test_exit_code_mapping():
    assert _exit_code(StopReason.STOP) == 0
    assert _exit_code(StopReason.ERROR) == 1
    assert _exit_code(StopReason.LENGTH) == 3
    assert _exit_code(StopReason.INTERRUPTED) == 1
