from kon.core.errors import format_error


def test_format_error_prefixes_type_name():
    assert format_error(ValueError("bad input")) == "ValueError: bad input"


def test_format_error_handles_empty_message():
    assert format_error(ConnectionError()) == "ConnectionError: failed without an error message"


def test_format_error_does_not_repeat_type_name():
    class APITimeoutError(Exception):
        def __str__(self) -> str:
            return "APITimeoutError: request timed out"

    assert format_error(APITimeoutError()) == "APITimeoutError: request timed out"
