from unittest.mock import AsyncMock, Mock, patch

import pytest

from kon.tools.bash import BashTool
from kon.ui.app import Kon


def test_handle_shell_command_execution():
    """Test that _handle_shell_command properly handles shell command execution"""
    # Create a mock app instance
    app = Mock()
    app._is_running = False
    app.query_one = Mock()

    # Mock the chat log
    mock_chat = Mock()
    app.query_one.return_value = mock_chat
    app.run_worker = Mock()

    # Test single ! command
    Kon._handle_shell_command(app, "!ls -la", "!ls -la")

    # Verify chat message was added
    mock_chat.add_user_message.assert_called_once_with("!ls -la")

    # Verify app state was set
    assert app._is_running is True

    # Verify run_worker was called with correct arguments
    app.run_worker.assert_called_once()
    call_args = app.run_worker.call_args
    # Check that the first argument is a coroutine function (the method)
    assert callable(call_args[0][0])
    # Check that exclusive=True was passed
    assert call_args[1]["exclusive"] is True


def test_handle_shell_command_history_mode():
    """Test that _handle_shell_command properly handles history mode (!!)"""
    # Create a mock app instance
    app = Mock()
    app._is_running = False
    app.query_one = Mock()

    # Mock the chat log
    mock_chat = Mock()
    app.query_one.return_value = mock_chat
    app.run_worker = Mock()

    # Test double !! command
    Kon._handle_shell_command(app, "!!git status", "!!git status")

    # Verify chat message was added
    mock_chat.add_user_message.assert_called_once_with("!!git status")

    # Verify app state was set
    assert app._is_running is True

    # Verify run_worker was called with correct arguments for history mode
    app.run_worker.assert_called_once()


def test_handle_shell_command_when_running():
    """Test that _handle_shell_command returns early when app is already running"""
    # Create a mock app instance that's already running
    app = Mock()
    app._is_running = True
    app.query_one = Mock()

    # Mock the chat log
    mock_chat = Mock()
    app.query_one.return_value = mock_chat
    app.run_worker = Mock()

    # Call the handler
    Kon._handle_shell_command(app, "!ls -la", "!ls -la")

    # Verify no chat message was added (should return early)
    mock_chat.add_user_message.assert_not_called()

    # Verify app state wasn't changed
    assert app._is_running is True

    # Verify run_worker was not called
    app.run_worker.assert_not_called()


@pytest.mark.asyncio
async def test_execute_shell_command_basic():
    """Test that _execute_shell_command properly executes basic shell commands"""
    # Create a mock app instance
    app = Mock()
    app._is_running = False

    # Mock the chat log and status line
    mock_chat = Mock()
    mock_status = Mock()
    app.query_one.side_effect = lambda id, _: mock_chat if id == "#chat-log" else mock_status

    # Mock bash tool execution
    mock_result = Mock()
    mock_result.success = True
    mock_result.result = "test output"
    mock_result.ui_details = None
    mock_result.ui_summary = None

    with patch.object(BashTool, "execute", new_callable=AsyncMock, return_value=mock_result):
        # Call the method
        await Kon._execute_shell_command(app, "ls -la", False, True)

    # Verify status was set to running and then idle
    mock_status.set_status.assert_any_call("running")
    mock_status.set_status.assert_called_with("idle")

    # Verify chat methods were called
    mock_chat.start_tool.assert_called_once_with("bash", "shell", "$ ls -la")

    # Verify tool block result was set
    tool_block = mock_chat.start_tool.return_value
    tool_block.set_result.assert_called_once_with("test output", None, True, markup=False)

    # Verify app state was reset
    assert app._is_running is False


@pytest.mark.asyncio
async def test_execute_shell_command_with_llm():
    """Test that _execute_shell_command sends output to LLM when using !!"""
    # Create a mock app instance
    app = Mock()
    app._is_running = False

    # Mock the chat log and status line
    mock_chat = Mock()
    mock_status = Mock()
    app.query_one.side_effect = lambda id, _: mock_chat if id == "#chat-log" else mock_status

    # Mock bash tool execution
    mock_result = Mock()
    mock_result.success = True
    mock_result.result = "git status output"
    mock_result.ui_details = None
    mock_result.ui_summary = None

    # Mock the _run_agent method
    app._run_agent = AsyncMock()

    with patch.object(BashTool, "execute", new_callable=AsyncMock, return_value=mock_result):
        # Call the method with send_to_llm=True
        await Kon._execute_shell_command(app, "git status", True, False)

    # Verify _run_agent was called with the correct prompt
    app._run_agent.assert_called_once()
    call_args = app._run_agent.call_args[0][0]
    assert "Shell command output:" in call_args
    assert "git status output" in call_args
    assert "What would you like me to do with this?" in call_args
