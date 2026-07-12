import re
from urllib.parse import urlsplit

_URL_RE = re.compile(r"""https?://[^\s<>{}\[\]"']+""", re.IGNORECASE)
# Markdown emphasis markers commonly immediately follow a linked URL.
_TRAILING_PUNCTUATION = ".,;:!?*_"


def extract_urls(texts: list[str]) -> list[str]:
    """Return unique HTTP(S) URLs in order of appearance."""
    urls: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for match in _URL_RE.finditer(text):
            url = _trim_url(match.group())
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or url in seen:
                continue
            seen.add(url)
            urls.append(url)
    return urls


def url_label(url: str) -> str:
    """Create a concise domain-and-endpoint label for a URL picker row."""
    parsed = urlsplit(url)
    domain = parsed.hostname or parsed.netloc
    endpoint = parsed.path.rstrip("/").rpartition("/")[2]
    return f"{domain}/{endpoint}" if endpoint else domain


def _trim_url(url: str) -> str:
    url = url.rstrip(_TRAILING_PUNCTUATION)
    for closing, opening in ((")", "("), ("}", "{")):
        while url.endswith(closing) and url.count(closing) > url.count(opening):
            url = url[:-1]
    return url
