from rich.text import Text

from kon.ui.blocks import ContentBlock, ThinkingBlock


def _capture_updates(block):
    updates: list[Text] = []
    block._streaming_update_label = updates.append  # type: ignore[method-assign]
    block.call_after_refresh = lambda callback: callback()  # type: ignore[method-assign]
    return updates


def test_content_block_previews_partial_line_with_cursor():
    block = ContentBlock()
    updates = _capture_updates(block)

    block._append_streaming("hello")

    assert updates
    assert updates[-1].plain == "hello▌"


def test_content_block_commits_completed_lines_and_keeps_tail_live():
    block = ContentBlock()
    updates = _capture_updates(block)

    block._append_streaming("hello\nwor")

    assert updates
    assert "hello" in updates[-1].plain
    assert updates[-1].plain.endswith("wor▌")


def test_content_block_flush_removes_live_cursor():
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

    block._append_streaming("a")
    block._append_streaming("b")

    assert len(callbacks) == 1
    assert updates == []

    callbacks[0]()

    assert updates[-1].plain == "ab▌"


def test_thinking_block_pending_preview_is_dim_and_italic():
    block = ThinkingBlock()
    updates = _capture_updates(block)

    block._append_streaming("thinking")

    assert updates[-1].plain == "thinking▌"
    assert updates[-1].spans[0].style is not None
    assert "italic" in str(updates[-1].spans[0].style)
