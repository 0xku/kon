import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

from curl_cffi import AsyncSession, CurlOpt
from html_to_markdown import ConversionOptions, convert
from lxml import html as lxml_html
from pydantic import BaseModel, Field
from readability import Document

from ..core.types import ToolResult
from ._tool_utils import ToolCancelledError, await_task_or_cancel
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


class _FetchRefusalError(Exception):
    pass


def _looks_like_challenge(html: str) -> bool:
    head = html[:4096].lower()
    return any(sig in head for sig in _CHALLENGE_SIGNATURES)


_WELL_KNOWN_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")


def _is_public_ip(ip: str) -> bool:
    try:
        parsed = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if not parsed.is_global or parsed.is_multicast:
        return False
    # NAT64 is marked global but can wrap a private IPv4 (e.g., 169.254.169.254).
    if isinstance(parsed, ipaddress.IPv6Address) and parsed in _WELL_KNOWN_NAT64_PREFIX:
        mapped_ipv4 = ipaddress.IPv4Address(int(parsed) & 0xFFFFFFFF)
        return mapped_ipv4.is_global and not mapped_ipv4.is_multicast
    return True


def _parse_fetch_url(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise _FetchRefusalError(f"unsupported scheme {parsed.scheme!r}")
    if not parsed.hostname:
        raise _FetchRefusalError("missing host")
    try:
        port = parsed.port
    except ValueError:
        raise _FetchRefusalError("invalid port") from None
    return parsed.hostname, port or (443 if parsed.scheme == "https" else 80)


async def _resolve_host_addresses(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    results = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(sockaddr[0] for *_, sockaddr in results))


def _curl_resolve_entry(host: str, port: int, addresses: list[str]) -> str:
    formatted_addresses = ",".join(
        f"[{ip}]" if ipaddress.ip_address(ip).version == 6 else ip for ip in addresses
    )
    return f"{host}:{port}:{formatted_addresses}"


async def _prepare_curl_resolve(url: str, cancel_event: asyncio.Event | None = None) -> list[str]:
    host, port = _parse_fetch_url(url)

    try:
        ipaddress.ip_address(host)
    except ValueError:
        resolve_task = asyncio.create_task(_resolve_host_addresses(host, port))
        try:
            addresses = await await_task_or_cancel(resolve_task, cancel_event)
        except ToolCancelledError:
            raise
        except Exception:
            raise _FetchRefusalError("host did not resolve") from None
        public_addresses = [ip for ip in addresses if _is_public_ip(ip)]
        if not public_addresses:
            raise _FetchRefusalError("host did not resolve to a public address") from None
        return [_curl_resolve_entry(host, port, public_addresses)]

    if not _is_public_ip(host):
        raise _FetchRefusalError(f"non-public address ({host})")
    return []


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
        return params.url

    async def execute(
        self, params: WebFetchParams, cancel_event: asyncio.Event | None = None
    ) -> ToolResult:
        try:
            curl_resolve = await _prepare_curl_resolve(params.url, cancel_event)
        except ToolCancelledError:
            return ToolResult(success=False, result="Fetch aborted")
        except _FetchRefusalError as e:
            msg = f"Refused: {e}"
            return ToolResult(success=False, result=msg, ui_summary=f"[red]{msg}[/red]")

        try:
            curl_options = {
                CurlOpt.MAXFILESIZE_LARGE: MAX_RESPONSE_BYTES,
                CurlOpt.PROTOCOLS_STR: "http,https",
                CurlOpt.REDIR_PROTOCOLS_STR: "http,https",
            }
            if curl_resolve:
                curl_options[CurlOpt.RESOLVE] = curl_resolve

            async with AsyncSession(
                impersonate="chrome131", allow_redirects="safe", curl_options=curl_options
            ) as session:
                fetch_task = asyncio.create_task(session.get(params.url, timeout=REQUEST_TIMEOUT))
                response = await await_task_or_cancel(fetch_task, cancel_event)
        except ToolCancelledError:
            return ToolResult(success=False, result="Fetch aborted")
        except Exception as e:
            msg = f"Fetch failed: {e}"
            return ToolResult(success=False, result=msg, ui_summary=f"[red]{msg}[/red]")

        if not _is_public_ip(response.primary_ip):
            msg = f"Refused: non-public address ({response.primary_ip or 'unknown'})"
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
