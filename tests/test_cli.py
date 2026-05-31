import sys

import pytest

from kon.cli import build_parser, main


def test_prompt_flag_takes_value():
    assert build_parser().parse_args(["-p", "hi"]).prompt == "hi"


def test_bare_prompt_flag_means_stdin():
    assert build_parser().parse_args(["-p"]).prompt == "-"


def test_provider_uses_long_form():
    assert build_parser().parse_args(["--provider", "openai"]).provider == "openai"


def test_prompt_no_longer_feeds_provider():
    assert build_parser().parse_args(["-p", "x"]).provider is None


def test_option_after_bare_prompt_falls_back_to_stdin():
    args = build_parser().parse_args(["-p", "-m", "x"])
    assert args.prompt == "-"
    assert args.model == "x"


@pytest.mark.parametrize("flag", [["-c"], ["-r", "abc123"]])
def test_resume_flags_rejected_with_prompt(monkeypatch, flag):
    monkeypatch.setattr(sys, "argv", ["kon", "-p", "x", *flag])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
