import subprocess

from kon import Config, reset_config, set_config
from kon.context import Context
from kon.loop import build_system_prompt


def _run(cmd: list[str], cwd: str) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def test_system_prompt_git_context_for_git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    _run(["git", "init"], str(repo))
    _run(["git", "config", "user.email", "kon@example.com"], str(repo))
    _run(["git", "config", "user.name", "Kon"], str(repo))
    (repo / "a.txt").write_text("hello\n", encoding="utf-8")
    _run(["git", "add", "a.txt"], str(repo))
    _run(["git", "commit", "-m", "init"], str(repo))

    set_config(Config({"llm": {"system_prompt": {"git_context": True}}}))
    try:
        prompt = build_system_prompt(str(repo), Context(str(repo)))
    finally:
        reset_config()

    assert "<git-status>" in prompt
    assert "Current branch:" in prompt
    assert "Recent commits:" in prompt
