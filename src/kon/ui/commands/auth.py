"""/login and /logout commands - provider OAuth flows."""

from __future__ import annotations

from ...llm import (
    clear_openai_credentials,
    clear_xai_credentials,
    copilot_login,
    get_copilot_token,
    get_valid_openai_credentials,
    get_xai_token,
    openai_login,
    xai_login,
)
from ...llm import is_copilot_logged_in as has_saved_copilot_credentials
from ...llm import is_openai_logged_in as has_saved_openai_credentials
from ...llm import is_xai_logged_in as has_saved_xai_credentials
from ..chat import ChatLog
from ..floating_list import ListItem
from ..selection_mode import SelectionMode
from .base import CommandSupport


class AuthCommands(CommandSupport):
    def _handle_login_command(self, args: str) -> None:
        providers = [
            ("github-copilot", "GitHub Copilot", has_saved_copilot_credentials()),
            ("openai", "OpenAI (ChatGPT/Codex)", has_saved_openai_credentials()),
            ("xai", "xAI (Grok/X subscription)", has_saved_xai_credentials()),
        ]

        self._show_selection_picker(
            [
                ListItem(
                    value=provider_id,
                    label=name,
                    description="saved credentials" if has_credentials else "",
                )
                for provider_id, name, has_credentials in providers
            ],
            SelectionMode.LOGIN,
        )

    def _select_login_provider(self, provider_id: str) -> None:
        if provider_id == "github-copilot":
            self.run_worker(self._copilot_login_flow(), exclusive=False)
            return

        if provider_id == "openai":
            self.run_worker(self._openai_login_flow(), exclusive=False)
            return

        if provider_id == "xai":
            self.run_worker(self._xai_login_flow(), exclusive=False)

    async def _copilot_login_flow(self) -> None:
        import webbrowser

        chat = self.query_one("#chat-log", ChatLog)
        had_saved_credentials = has_saved_copilot_credentials()

        def on_user_code(url: str, code: str) -> None:
            webbrowser.open(url)
            self.call_later(
                chat.add_info_message,
                f"Opening browser to: {url}\n"
                f"Enter this code: {code}\n\n"
                "Waiting for authorization...",
            )

        try:
            if await get_copilot_token():
                chat.add_info_message("Already logged in to GitHub Copilot")
                return

            if had_saved_credentials:
                chat.add_info_message(
                    "Your saved GitHub Copilot session is no longer valid.", warning=True
                )
            else:
                chat.add_info_message("Starting GitHub Copilot login...")

            await copilot_login(on_user_code=on_user_code)
            chat.add_info_message(
                "Successfully logged in to GitHub Copilot!\n"
                "You can now use /model to select Copilot models."
            )
        except Exception as e:
            chat.add_info_message(f"Login failed: {e}", error=True)

    async def _openai_login_flow(self) -> None:
        import webbrowser

        chat = self.query_one("#chat-log", ChatLog)
        had_saved_credentials = has_saved_openai_credentials()

        def on_auth_url(url: str) -> None:
            webbrowser.open(url)
            self.call_later(
                chat.add_info_message,
                "Opening browser for OpenAI OAuth...\n"
                f"If browser does not open, visit:\n{url}\n\n"
                "Waiting for authorization callback on http://localhost:1455/auth/callback ...",
            )

        try:
            if await get_valid_openai_credentials():
                chat.add_info_message("Already logged in to OpenAI")
                return

            if had_saved_credentials:
                chat.add_info_message(
                    "Your saved OpenAI session is no longer valid.", warning=True
                )
            else:
                chat.add_info_message("Starting OpenAI login...")

            await openai_login(on_auth_url=on_auth_url)
            chat.add_info_message(
                "Successfully logged in to OpenAI!\n"
                "You can now use /model to select openai-codex models."
            )
        except Exception as e:
            chat.add_info_message(f"Login failed: {e}", error=True)

    async def _xai_login_flow(self) -> None:
        import webbrowser

        chat = self.query_one("#chat-log", ChatLog)
        had_saved_credentials = has_saved_xai_credentials()

        def on_user_code(url: str, code: str) -> None:
            webbrowser.open(url)
            self.call_later(
                chat.add_info_message,
                f"Opening browser to: {url}\n"
                f"Enter this code if prompted: {code}\n\n"
                "Waiting for authorization...",
            )

        try:
            if await get_xai_token():
                chat.add_info_message("Already logged in to xAI")
                return

            if had_saved_credentials:
                chat.add_info_message("Your saved xAI session is no longer valid.", warning=True)
            else:
                chat.add_info_message("Starting xAI login...")

            await xai_login(on_user_code=on_user_code)
            chat.add_info_message(
                "Successfully logged in to xAI!\nYou can now use /model to select Grok models."
            )
        except Exception as e:
            chat.add_info_message(f"Login failed: {e}", error=True)

    def _handle_logout_command(self, args: str) -> None:
        providers = []
        if has_saved_copilot_credentials():
            providers.append(("github-copilot", "GitHub Copilot"))
        if has_saved_openai_credentials():
            providers.append(("openai", "OpenAI (ChatGPT/Codex)"))
        if has_saved_xai_credentials():
            providers.append(("xai", "xAI (Grok/X subscription)"))

        if not providers:
            chat = self.query_one("#chat-log", ChatLog)
            chat.add_info_message("No providers logged in")
            return

        self._show_selection_picker(
            [
                ListItem(value=provider_id, label=name, description="")
                for provider_id, name in providers
            ],
            SelectionMode.LOGOUT,
        )

    def _select_logout_provider(self, provider_id: str) -> None:
        from kon.llm import clear_copilot_credentials

        chat = self.query_one("#chat-log", ChatLog)

        if provider_id == "github-copilot":
            clear_copilot_credentials()
            chat.add_info_message("Logged out of GitHub Copilot")
            return

        if provider_id == "openai":
            clear_openai_credentials()
            chat.add_info_message("Logged out of OpenAI")
            return

        if provider_id == "xai":
            clear_xai_credentials()
            chat.add_info_message("Logged out of xAI")
