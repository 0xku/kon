from kon.cli import build_parser


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
    assert args.insecure_skip_verify is False


def test_insecure_skip_verify_flag_sets_true() -> None:
    parser = build_parser()
    args = parser.parse_args(["--insecure-skip-verify"])

    assert args.insecure_skip_verify is True
