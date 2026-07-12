from kon.ui.urls import extract_urls, url_label


def test_extract_urls_returns_unique_urls_in_order() -> None:
    urls = extract_urls(
        [
            "See https://example.com/docs/page, then https://www.python.org/.",
            "Again https://example.com/docs/page and <https://github.com/org/repo>**.",
        ]
    )

    assert urls == [
        "https://example.com/docs/page",
        "https://www.python.org/",
        "https://github.com/org/repo",
    ]


def test_extract_urls_ignores_non_http_urls() -> None:
    assert extract_urls(["mailto:test@example.com and ftp://example.com"]) == []


def test_url_label_uses_domain_and_last_path_part() -> None:
    assert url_label("https://github.com/0xku/kon/issues/42?foo=bar") == "github.com/42"
    assert url_label("https://example.com/") == "example.com"
