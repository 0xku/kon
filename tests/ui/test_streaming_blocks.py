from rich.text import Text

from kon.ui.blocks import ContentBlock, ThinkingBlock


def _capture_updates(block):
    updates: list[Text] = []
    block._streaming_update_label = updates.append  # type: ignore[method-assign]
    block.call_after_refresh = lambda callback: callback()  # type: ignore[method-assign]
    return updates


def test_content_block_buffers_partial_line_until_newline():
    block = ContentBlock()
    updates = _capture_updates(block)

    block._append_streaming("hello")

    assert updates == []


def test_content_block_commits_completed_lines_and_buffers_tail():
    block = ContentBlock()
    updates = _capture_updates(block)

    block._append_streaming("hello\nwor")

    assert updates
    assert "hello" in updates[-1].plain
    assert not updates[-1].plain.endswith("wor")


def test_content_block_flush_finalizes_display():
    block = ContentBlock()

    block._append_streaming("hello")
    display = block._flush_streaming()

    assert display.plain.rstrip() == "hello"


def test_streaming_update_is_coalesced_until_refresh():
    block = ContentBlock()
    callbacks = []
    updates: list[Text] = []
    block._streaming_update_label = updates.append  # type: ignore[method-assign]
    block.call_after_refresh = callbacks.append  # type: ignore[method-assign]

    block._append_streaming("a\n")
    block._append_streaming("b\n")

    assert len(callbacks) == 1
    assert updates == []

    callbacks[0]()

    assert "a" in updates[-1].plain
    assert "b" in updates[-1].plain


def test_thinking_block_buffers_partial_line_until_newline():
    block = ThinkingBlock()
    updates = _capture_updates(block)

    block._append_streaming("thinking")

    assert updates == []
