from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kon.core.types import ToolResult
from kon.ui.app import Kon


class TestShellCommand:
    @pytest.fixture
    def mock_kon_app(self):
        """Fixture to create a Kon app instance with mocked dependencies."""
        mock_chat_log_instance = MagicMock()
        mock_status_line_instance = MagicMock()
        mock_session_instance = MagicMock()

        with (
            patch("kon.ui.app.ChatLog", return_value=mock_chat_log_instance),
            patch("kon.ui.app.StatusLine", return_value=mock_status_line_instance),
            patch("kon.ui.app.Session", return_value=mock_session_instance),
        ):
            app = Kon()
            app.query_one = MagicMock(
                side_effect=lambda selector, *args: {
                    "#chat-log": mock_chat_log_instance,
                    "#status-line": mock_status_line_instance,
                }.get(selector)
            )
            app.run_worker = MagicMock()
            app._session = mock_session_instance
            app._session.generate_id.return_value = "test-tool-id"
            app._is_running = False
            yield app, mock_chat_log_instance, mock_status_line_instance

    def test_on_input_submitted_valid_shell_command(self, mock_kon_app):
        app, _, _ = mock_kon_app
        app._handle_command = MagicMock(
            return_value=False
        )  # Ensure it doesn't handle it as a normal command
        app._run_shell_command = MagicMock()

        event = MagicMock()
        event.text = "!ls -l"
        event.shell_cmd = "ls -l"
        event.query_text = None
        event.selected_skill_name = None
        event.selected_skill_query = None
        event.add_to_history = True
        event.steer = False

        app.on_input_submitted(event)

        app.run_worker.assert_called_once()
        app._run_shell_command.assert_called_once_with("ls -l", True)

    def test_on_input_submitted_empty_shell_command(self, mock_kon_app):
        app, mock_chat_log, _ = mock_kon_app
        app._handle_command = MagicMock(return_value=False)
        app._run_shell_command = MagicMock()
        app._run_agent = MagicMock()  # Mock the agent run to avoid actual execution

        event = MagicMock()
        event.text = "!"
        event.shell_cmd = None  # Empty shell commands are not set (falsy empty string)
        event.query_text = "!"  # The actual text that would be processed
        event.selected_skill_name = None
        event.selected_skill_query = None
        event.steer = False

        app.on_input_submitted(event)

        # No error message should be shown for empty shell commands in current implementation
        mock_chat_log.add_info_message.assert_not_called()
        app._run_shell_command.assert_not_called()
        # The agent should be run with the "!" text as a normal query
        app._run_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_shell_command_success(self, mock_kon_app):
        app, mock_chat_log, mock_status_line = mock_kon_app
        with patch("kon.tools.bash.BashTool") as MockBashTool:
            mock_bash_tool_instance = MockBashTool.return_value
            mock_bash_tool_instance.execute = AsyncMock(
                return_value=ToolResult(
                    tool_name="bash",
                    tool_call_id="test-tool-id",
                    result="stdout output",
                    ui_summary="Success",
                    success=True,
                )
            )
            mock_bash_tool_instance.params.return_value = {"command": "echo hello"}

            command = "echo hello"
            await app._run_shell_command(command, add_to_history=True)

            assert app._is_running is False
            mock_status_line.set_status.assert_any_call("working")
            mock_chat_log.add_user_message.assert_called_once_with("! echo hello")
            # Check that start_tool was called with correct parameters except for the dynamic tool_id
            mock_chat_log.start_tool.assert_called_once()
            call_args = mock_chat_log.start_tool.call_args
            assert call_args[1]["name"] == "bash"
            assert call_args[1]["call_msg"] == "Executing: echo hello"
            assert call_args[1]["icon"] == "!"
            assert call_args[1]["tool_id"].startswith("manual-")
            mock_bash_tool_instance.execute.assert_called_once()
            execute_call_args = mock_bash_tool_instance.execute.call_args
            assert execute_call_args[0][0] == {"command": "echo hello"}
            assert "cancel_event" in execute_call_args[1]
            # Check that set_tool_result was called with correct parameters
            mock_chat_log.set_tool_result.assert_called_once()
            result_call_args = mock_chat_log.set_tool_result.call_args
            assert result_call_args[1]["ui_summary"] == "Success"
            assert result_call_args[1]["ui_details"] == "stdout output"
            assert result_call_args[1]["success"] is True
            assert result_call_args[1]["markup"] is False
            # tool_id should match the one generated in start_tool
            assert result_call_args[1]["tool_id"] == call_args[1]["tool_id"]
            mock_status_line.set_status.assert_any_call("idle")

    @pytest.mark.asyncio
    async def test_run_shell_command_failure(self, mock_kon_app):
        app, mock_chat_log, mock_status_line = mock_kon_app

        with patch("kon.tools.bash.BashTool") as MockBashTool:
            mock_bash_tool_instance = MockBashTool.return_value
            mock_bash_tool_instance.execute = AsyncMock(side_effect=Exception("Command failed"))
            mock_bash_tool_instance.params.return_value = {"command": "bad command"}

            command = "bad command"
            await app._run_shell_command(command, add_to_history=True)

            assert app._is_running is False
            mock_status_line.set_status.assert_any_call("working")
            mock_chat_log.add_user_message.assert_called_once_with("! bad command")
            # Check that start_tool was called with correct parameters except for the dynamic tool_id
            mock_chat_log.start_tool.assert_called_once()
            call_args = mock_chat_log.start_tool.call_args
            assert call_args[1]["name"] == "bash"
            assert call_args[1]["call_msg"] == "Executing: bad command"
            assert call_args[1]["icon"] == "!"
            assert call_args[1]["tool_id"].startswith("manual-")
            mock_bash_tool_instance.execute.assert_called_once()
            execute_call_args = mock_bash_tool_instance.execute.call_args
            assert execute_call_args[0][0] == {"command": "bad command"}
            assert "cancel_event" in execute_call_args[1]
            # Check that set_tool_result was called with correct parameters
            mock_chat_log.set_tool_result.assert_called_once()
            result_call_args = mock_chat_log.set_tool_result.call_args
            assert result_call_args[1]["ui_summary"] == "Error"
            assert (
                result_call_args[1]["ui_details"]
                == "Error executing shell command: Command failed"
            )
            assert result_call_args[1]["success"] is False
            assert result_call_args[1]["markup"] is False
            # tool_id should match the one generated in start_tool
            assert result_call_args[1]["tool_id"] == call_args[1]["tool_id"]
            mock_chat_log.add_info_message.assert_called_once_with(
                "Error executing shell command: Command failed", error=True
            )
            mock_status_line.set_status.assert_any_call("idle")
