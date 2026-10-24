import sys

import pytest

from kon.cli import build_parser, main


def test_prompt_flag_takes_value():
    assert build_parser().parse_args(["-p", "hi"]).prompt == "hi"


def test_bare_prompt_flag_means_stdin():
    assert build_parser().parse_args(["-p"]).prompt == "-"


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
def test_provider_uses_long_form(provider):
    assert build_parser().parse_args(["--provider", provider]).provider == provider


def test_prompt_no_longer_feeds_provider():
    assert build_parser().parse_args(["-p", "x"]).provider is None


def test_option_after_bare_prompt_falls_back_to_stdin():
    args = build_parser().parse_args(["-p", "-m", "x"])
    assert args.prompt == "-"
    assert args.model == "x"


def test_context_window_flag_takes_value():
    assert build_parser().parse_args(["--context-window", "12345"]).context_window == 12345


def test_context_window_defaults_to_none():
    assert build_parser().parse_args([]).context_window is None


def test_context_window_does_not_conflict_with_continue():
    args = build_parser().parse_args(["-c", "--context-window", "1000"])
    assert args.continue_recent is True
    assert args.context_window == 1000


def test_context_window_rejects_non_positive(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["kon", "--context-window", "0"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


@pytest.mark.parametrize("flag", [["-c"], ["-r", "abc123"]])
def test_resume_flags_rejected_with_prompt(monkeypatch, flag):
    monkeypatch.setattr(sys, "argv", ["kon", "-p", "x", *flag])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
