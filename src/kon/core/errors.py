"""Shared error formatting for provider/agent error surfaces.

Errors crossing the event boundary (StreamError, ErrorEvent, compaction
failures) are reduced to strings, so the exception type must be baked into
the message or it is lost to the user.
"""


def format_error(error: BaseException) -> str:
    name = type(error).__name__
    message = str(error).strip()
    if not message:
        return f"{name}: failed without an error message"
    # Some SDK errors already stringify with their type name; don't repeat it.
    if message.startswith(name):
        return message
    return f"{name}: {message}"
