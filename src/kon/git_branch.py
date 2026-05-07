import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GitPaths:
    repo_dir: str
    common_git_dir: str
    head_path: str


def find_git_paths(cwd: str) -> GitPaths | None:
    """Find git metadata paths for regular repos and worktrees."""
    directory = os.path.abspath(cwd)
    while True:
        git_path = os.path.join(directory, ".git")
        if os.path.exists(git_path):
            try:
                if os.path.isfile(git_path):
                    with open(git_path, encoding="utf-8") as f:
                        content = f.read().strip()
                    if content.startswith("gitdir: "):
                        git_dir = os.path.abspath(
                            os.path.join(directory, content.removeprefix("gitdir: ").strip())
                        )
                        head_path = os.path.join(git_dir, "HEAD")
                        if not os.path.exists(head_path):
                            return None
                        common_dir_path = os.path.join(git_dir, "commondir")
                        if os.path.exists(common_dir_path):
                            with open(common_dir_path, encoding="utf-8") as f:
                                common_dir = f.read().strip()
                            common_git_dir = os.path.abspath(os.path.join(git_dir, common_dir))
                        else:
                            common_git_dir = git_dir
                        return GitPaths(directory, common_git_dir, head_path)
                elif os.path.isdir(git_path):
                    head_path = os.path.join(git_path, "HEAD")
                    if not os.path.exists(head_path):
                        return None
                    return GitPaths(directory, git_path, head_path)
            except OSError:
                return None

        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


def _resolve_branch_with_git(repo_dir: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "symbolic-ref", "--quiet", "--short", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None

    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    return branch or None


def resolve_git_branch(cwd: str) -> str:
    """Resolve current git branch, returning an empty string outside git repos."""
    git_paths = find_git_paths(cwd)
    if git_paths is None:
        return ""

    try:
        with open(git_paths.head_path, encoding="utf-8") as f:
            content = f.read().strip()
    except OSError:
        return ""

    prefix = "ref: refs/heads/"
    if content.startswith(prefix):
        branch = content.removeprefix(prefix)
        if branch == ".invalid":
            return _resolve_branch_with_git(git_paths.repo_dir) or "detached"
        return branch

    return "detached"
