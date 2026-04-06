import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kon TUI")
    parser.add_argument("--openai-compat-auth", choices=("auto", "required", "none"))
    parser.add_argument("--anthropic-compat-auth", choices=("auto", "required", "none"))
    return parser


def test_cli_auth_flags_accept_valid_values() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["--openai-compat-auth", "none", "--anthropic-compat-auth", "required"]
    )

    assert args.openai_compat_auth == "none"
    assert args.anthropic_compat_auth == "required"


def test_cli_auth_flags_default_to_none_when_omitted() -> None:
    parser = build_parser()
    args = parser.parse_args([])

    assert args.openai_compat_auth is None
    assert args.anthropic_compat_auth is None
