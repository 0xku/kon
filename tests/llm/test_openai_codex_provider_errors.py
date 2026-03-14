from kon.llm.providers.openai_codex_responses import _format_provider_error


def test_format_provider_error_preserves_non_empty_message():
    err = RuntimeError("boom")
    assert _format_provider_error(err) == "boom"


def test_format_provider_error_falls_back_for_empty_message():
    err = TimeoutError()
    message = _format_provider_error(err)
    assert "TimeoutError" in message
    assert "without an error message" in message
