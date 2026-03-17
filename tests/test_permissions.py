import pytest

from kon import Config, set_config
from kon.permissions import PermissionDecision, _is_safe_bash_command, check_permission
from kon.tools import BashTool, EditTool, ReadTool, WriteTool


class TestIsSafeBashCommand:
    @pytest.mark.parametrize(
        "command",
        [
            "ls -la",
            "cat src/main.py",
            "pwd",
            "git status",
            "git diff --cached",
            "git log --oneline -10",
            "git -C /tmp status",
            "git --no-pager log",
            "/usr/bin/ls",
            'ls "dir;name"',
            "git log -c",  # -c after subcommand is a log flag (combined diffs), not git -c config
        ],
    )
    def test_allows_safe_commands(self, command):
        assert _is_safe_bash_command(command) is True

    @pytest.mark.parametrize(
        "command,reason",
        [
            ("rm -rf /", "not in safe list"),
            ("python script.py", "not in safe list"),
            ("git push origin main", "unsafe git subcommand"),
            ("git -c core.pager=evil log", "git -c config injection"),
            ("git -C /tmp -c core.pager=evil status", "git -c after -C flag"),
            ("git", "git with no subcommand"),
            ("ls; rm -rf /", "semicolon chaining"),
            ("cat file | sh", "pipe operator"),
            ("echo hello > file", "output redirect"),
            ("ls >& /tmp/evil", "combined redirect operator"),
            ("echo $(rm -rf /)", "command substitution via $()"),
            ("echo `rm -rf /`", "command substitution via backticks"),
            ("cat file\nrm -rf /", "newline command separator"),
            ("", "empty command"),
            ("git --namespace status push", "namespace flag consumes next token"),
            ("git branch -D main", "branch has destructive modes"),
            ("git tag -d v1.0", "tag has destructive modes"),
            ("git remote remove origin", "remote has destructive modes"),
            ("git reflog delete HEAD@{0}", "reflog has destructive modes"),
            ("git --config-env core.pager=SHELL log", "config-env space form injection"),
            ("git --config-env=core.pager=SHELL diff", "config-env equals form injection"),
            ("git diff --output=/tmp/evil HEAD", "git --output writes to file"),
            ("git log --output=/tmp/evil -1", "git log --output writes to file"),
            ("git show --output=/tmp/evil HEAD", "git show --output writes to file"),
            ("git diff --output=/tmp/evil", "git --output= writes to file"),
            ("tree -o /tmp/evil /home", "tree not in safe list"),
            ("cat <(rm -rf /)", "process substitution via <()"),
            ("echo >(rm -rf /)", "process substitution via >()"),
        ],
    )
    def test_rejects_unsafe_commands(self, command, reason):
        assert _is_safe_bash_command(command) is False, reason


class TestCheckPermission:
    @pytest.fixture(autouse=True)
    def _prompt_mode(self):
        set_config(Config({"permissions": {"mode": "prompt"}}))

    def test_auto_mode_allows_everything(self):
        set_config(Config({"permissions": {"mode": "auto"}}))
        assert check_permission(BashTool(), {"command": "rm -rf /"}) == PermissionDecision.ALLOW

    def test_read_only_tool_always_allowed(self):
        assert (
            check_permission(ReadTool(), {"file_path": "/etc/passwd"}) == PermissionDecision.ALLOW
        )

    def test_bash_safe_command_allowed(self):
        assert check_permission(BashTool(), {"command": "ls -la"}) == PermissionDecision.ALLOW

    def test_bash_unsafe_command_prompts(self):
        assert check_permission(BashTool(), {"command": "rm -rf /"}) == PermissionDecision.PROMPT

    def test_mutating_tool_prompts(self):
        assert (
            check_permission(WriteTool(), {"file_path": "x", "content": "y"})
            == PermissionDecision.PROMPT
        )
        assert check_permission(EditTool(), {"file_path": "x"}) == PermissionDecision.PROMPT
