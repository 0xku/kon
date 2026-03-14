import subprocess


def _run_git_command(cwd: str, args: list[str], timeout: int = 5) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, check=False, capture_output=True, text=True, timeout=timeout
        )
    except Exception:
        return ""

    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def format_git_context_for_prompt(cwd: str) -> str:
    is_git_repo = _run_git_command(cwd, ["rev-parse", "--git-dir"]) != ""
    if not is_git_repo:
        return ""

    current_branch = _run_git_command(cwd, ["branch", "--show-current"])

    main_branch = "main"
    remote_head = _run_git_command(cwd, ["symbolic-ref", "refs/remotes/origin/HEAD"])
    if remote_head.startswith("refs/remotes/origin/"):
        main_branch = remote_head.replace("refs/remotes/origin/", "", 1)
    else:
        remote_branches = _run_git_command(cwd, ["branch", "-r"])
        if "origin/master" in remote_branches:
            main_branch = "master"

    status = _run_git_command(cwd, ["status", "--porcelain"], timeout=10)
    recent_commits = _run_git_command(cwd, ["log", "--oneline", "-5"], timeout=10)

    sections: list[str] = []
    if current_branch:
        sections.append(f"Current branch: {current_branch}")
    if main_branch:
        sections.append(f"Main branch (you will usually use this for PRs): {main_branch}")
    if status:
        sections.append(f"Status:\n{status}")
    if recent_commits:
        sections.append(f"Recent commits:\n{recent_commits}")

    if not sections:
        return ""

    content = "\n\n".join(sections)
    max_chars = 2000
    if len(content) > max_chars:
        content = (
            content[:max_chars] + "\n\n... (truncated because it exceeds 2k characters. "
            'If you need more information, run "git status" using bash)'
        )

    return (
        "# Git Context\n\n"
        "This is the git status at the start of the conversation. "
        "Note that this status is a snapshot in time, and will not update "
        "during the conversation.\n"
        "<git-status>\n"
        f"{content}\n"
        "</git-status>"
    )
