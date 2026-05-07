import subprocess
from pathlib import Path

from kon.git_branch import resolve_git_branch


def test_resolve_git_branch_reads_head_directly(tmp_path: Path):
    repo = tmp_path / "repo"
    nested = repo / "src"
    git_dir = repo / ".git"
    nested.mkdir(parents=True)
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

    assert resolve_git_branch(str(nested)) == "main"


def test_resolve_git_branch_supports_worktrees(tmp_path: Path):
    repo = tmp_path / "repo"
    worktree = tmp_path / "worktree"
    git_dir = repo / ".git" / "worktrees" / "src"
    common_git_dir = repo / ".git"
    git_dir.mkdir(parents=True)
    worktree.mkdir()
    common_git_dir.mkdir(exist_ok=True)
    (worktree / ".git").write_text(f"gitdir: {git_dir}\n")
    (git_dir / "HEAD").write_text("ref: refs/heads/feature\n")
    (git_dir / "commondir").write_text("../..\n")

    assert resolve_git_branch(str(worktree)) == "feature"


def test_resolve_git_branch_falls_back_to_git_for_reftable_invalid_head(
    tmp_path: Path, monkeypatch
):
    repo = tmp_path / "repo"
    git_dir = repo / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/.invalid\n")

    def fake_run(*args, **kwargs):
        assert args[0] == [
            "git",
            "--no-optional-locks",
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
        ]
        assert kwargs["cwd"] == str(repo)
        return subprocess.CompletedProcess(args[0], 0, stdout="main\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert resolve_git_branch(str(repo)) == "main"


def test_resolve_git_branch_treats_unresolved_reftable_head_as_detached(
    tmp_path: Path, monkeypatch
):
    repo = tmp_path / "repo"
    git_dir = repo / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/.invalid\n")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert resolve_git_branch(str(repo)) == "detached"


def test_resolve_git_branch_returns_empty_string_outside_repo(tmp_path: Path):
    assert resolve_git_branch(str(tmp_path)) == ""


def test_resolve_git_branch_reports_detached_head(tmp_path: Path):
    repo = tmp_path / "repo"
    git_dir = repo / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("abc123\n")

    assert resolve_git_branch(str(repo)) == "detached"
