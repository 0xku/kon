import asyncio
import ipaddress
from urllib.parse import urlparse

from curl_cffi import AsyncSession, CurlOpt
from html_to_markdown import ConversionOptions, convert
from lxml import html as lxml_html
from pydantic import BaseModel, Field
from readability import Document

from ..core.types import ToolResult
from ._tool_utils import ToolCancelledError, await_task_or_cancel, truncate_text
from .base import BaseTool

MAX_CHARS = 80_000
MAX_CHARS_PER_LINE = 2000
MAX_RESPONSE_BYTES = 20_000_000
REQUEST_TIMEOUT = 15

# Only checked on failure paths, so false positives can't suppress real content.
_CHALLENGE_SIGNATURES = (
    "please wait for verification",  # Reddit 200 JS challenge
    "prove your humanity",  # Reddit 200 reCAPTCHA gate
    "whoa there, pardner",  # Reddit 403 ratelimit page
    "just a moment...",  # Cloudflare
    "checking your browser",  # Cloudflare (legacy)
    "attention required",  # Cloudflare block page
)

# Inline SVG data URIs would otherwise become base64 noise.
_CONVERT_OPTIONS = ConversionOptions(skip_images=True)

# concat(';', ...) anchors to a property boundary to block url() false positives.
_HIDDEN_XPATH = (
    "//script | //style | //noscript | //template"
    " | //*[@hidden or @aria-hidden='true']"
    " | //*[contains(concat(';', translate(@style, ' \t\n', '')), ';display:none')]"
    " | //*[contains(concat(';', translate(@style, ' \t\n', '')), ';visibility:hidden')]"
)


def _looks_like_challenge(html: str) -> bool:
    head = html[:4096].lower()
    return any(sig in head for sig in _CHALLENGE_SIGNATURES)


def _is_link_local(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_link_local
    except ValueError:
        return True  # fail closed


def _extract_markdown(html: str) -> str | None:
    if not html:
        return None
    try:
        tree = lxml_html.fromstring(html)
        for node in tree.xpath(_HIDDEN_XPATH):
            if (parent := node.getparent()) is not None:
                parent.remove(node)
        doc_input = tree
    except Exception:
        doc_input = html  # fall back to readability's own lenient parser
    summary_html = Document(doc_input).summary()
    if not summary_html or len(summary_html) < 50:
        return None
    return convert(summary_html, _CONVERT_OPTIONS).content or None


class WebFetchParams(BaseModel):
    url: str = Field(description="URL of the web page to fetch and extract content from")


class WebFetchTool(BaseTool):
    name = "web_fetch"
    tool_icon = "%"
    mutating = False
    params = WebFetchParams
    description = (
        "Fetch a web page and extract its main content as clean markdown. "
        "Strips navigation, ads, and boilerplate. "
        "Use web_search first to find relevant URLs (if not provided by the user)."
    )

    def format_call(self, params: WebFetchParams) -> str:
        return truncate_text(params.url)

    async def execute(
        self, params: WebFetchParams, cancel_event: asyncio.Event | None = None
    ) -> ToolResult:
        scheme = urlparse(params.url).scheme
        if scheme not in ("http", "https"):
            msg = f"Refused: unsupported scheme {scheme!r}"
            return ToolResult(success=False, result=msg, ui_summary=f"[red]{msg}[/red]")

        try:
            async with AsyncSession(
                impersonate="chrome131",
                allow_redirects="safe",
                curl_options={CurlOpt.MAXFILESIZE_LARGE: MAX_RESPONSE_BYTES},
            ) as session:
                fetch_task = asyncio.create_task(session.get(params.url, timeout=REQUEST_TIMEOUT))
                response = await await_task_or_cancel(fetch_task, cancel_event)
        except ToolCancelledError:
            return ToolResult(success=False, result="Fetch aborted")
        except Exception as e:
            msg = f"Fetch failed: {e}"
            return ToolResult(success=False, result=msg, ui_summary=f"[red]{msg}[/red]")

        if _is_link_local(response.primary_ip):
            msg = f"Refused: link-local address ({response.primary_ip or 'unknown'})"
            return ToolResult(success=False, result=msg, ui_summary=f"[red]{msg}[/red]")

        # Catches decompression bombs (MAXFILESIZE_LARGE only bounds wire bytes).
        if len(response.content) > MAX_RESPONSE_BYTES:
            msg = f"Response too large (>{MAX_RESPONSE_BYTES:,} bytes decoded)"
            return ToolResult(success=False, result=msg, ui_summary=f"[red]{msg}[/red]")

        if not (200 <= response.status_code < 300):
            status = f"HTTP {response.status_code}"
            msg = (
                f"Site appears to block automated fetchers ({status})"
                if _looks_like_challenge(response.text)
                else status
            )
            return ToolResult(success=False, result=msg, ui_summary=f"[red]{msg}[/red]")

        html = response.text

        try:
            extract_task = asyncio.create_task(asyncio.to_thread(_extract_markdown, html))
            content = await await_task_or_cancel(extract_task, cancel_event)
        except ToolCancelledError:
            return ToolResult(success=False, result="Extraction aborted")
        except Exception as e:
            msg = f"Extraction failed: {e}"
            return ToolResult(success=False, result=msg, ui_summary=f"[red]{msg}[/red]")

        if not content:
            msg = (
                "Site appears to block automated fetchers"
                if _looks_like_challenge(html)
                else "Couldn't extract content"
            )
            return ToolResult(success=False, result=msg, ui_summary=f"[red]{msg}[/red]")

        content = "\n".join(line[:MAX_CHARS_PER_LINE] for line in content.split("\n"))

        char_count = len(content)
        truncated = char_count > MAX_CHARS
        if truncated:
            cut = content.rfind("\n", 0, MAX_CHARS)
            content = content[: cut if cut > 0 else MAX_CHARS] + "\n\n[content truncated]"

        ui_summary = f"[dim]({char_count:,} chars)[/dim]"
        if truncated:
            ui_summary += " [yellow](truncated)[/yellow]"

        return ToolResult(success=True, result=content, ui_summary=ui_summary)
