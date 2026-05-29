import argparse
import asyncio

from kon import config

from .llm import PROVIDER_API_BY_NAME
from .version import VERSION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kon")
    parser.add_argument("--model", "-m", help="Model to use")
    parser.add_argument("--provider", choices=sorted(PROVIDER_API_BY_NAME), help="Provider to use")
    parser.add_argument(
        "--prompt",
        "-p",
        nargs="?",
        const="-",
        default=None,
        help="Run a single prompt non-interactively, then exit "
        "(omit the value or pipe stdin to read the prompt from stdin)",
    )
    parser.add_argument("--api-key", "-k", help="API key")
    parser.add_argument("--base-url", "-u", help="Base URL for API")
    parser.add_argument(
        "--openai-compat-auth",
        choices=("auto", "required", "none"),
        help="Auth mode for OpenAI-compatible endpoints",
    )
    parser.add_argument(
        "--anthropic-compat-auth",
        choices=("auto", "required", "none"),
        help="Auth mode for Anthropic-compatible endpoints",
    )
    parser.add_argument(
        "--insecure-skip-verify",
        action="store_true",
        help="Skip TLS verification (e.g. self-signed certs on local providers)",
    )
    parser.add_argument(
        "--continue",
        "-c",
        action="store_true",
        dest="continue_recent",
        help="Resume the most recent session",
    )
    parser.add_argument(
        "--resume",
        "-r",
        dest="resume_session",
        help="Resume a specific session by ID (full or unique prefix)",
    )
    parser.add_argument("--version", action="version", version=f"kon {VERSION}")
    parser.add_argument(
        "--extra-tools", help="Comma-separated extra tools to enable (e.g. web_search,web_fetch)"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.prompt is not None and (args.continue_recent or args.resume_session):
        parser.error("-c/--continue and -r/--resume are not supported with -p/--prompt")

    if args.insecure_skip_verify:
        config.llm.tls.insecure_skip_verify = True

    extra_tools = (
        [t.strip() for t in args.extra_tools.split(",") if t.strip()] if args.extra_tools else None
    )

    if args.prompt is not None:
        from .headless import run_headless

        raise SystemExit(
            asyncio.run(
                run_headless(
                    prompt_arg=args.prompt,
                    model=args.model,
                    provider=args.provider,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    openai_compat_auth_mode=args.openai_compat_auth,
                    anthropic_compat_auth_mode=args.anthropic_compat_auth,
                    extra_tools=extra_tools,
                )
            )
        )

    from .ui.app import run_tui

    run_tui(args, extra_tools=extra_tools)


if __name__ == "__main__":
    main()
