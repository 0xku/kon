from urllib.parse import urlparse

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _hostname(base_url: str | None) -> str | None:
    if not base_url:
        return None

    parsed = urlparse(base_url if "://" in base_url else f"https://{base_url}")
    return parsed.hostname.lower() if parsed.hostname else None


def supports_developer_role(provider: str | None, base_url: str | None) -> bool:
    provider_name = (provider or "").strip().lower()

    if provider_name in {"zai", "zhipu", "github-copilot", "deepseek"}:
        return False

    hostname = _hostname(base_url)
    if hostname is None:
        return True

    if hostname in _LOCAL_HOSTS:
        return False

    return hostname == "api.openai.com"
