from kon.ui.widgets import InfoBar


def test_info_bar_does_not_treat_permission_mode_as_file_changes_click():
    info_bar = InfoBar("/tmp", "model")
    info_bar._file_changes = {"a.txt": (2, 1)}
    label = object()
    info_bar._row2_left = label  # type: ignore[assignment]
    info_bar._format_row2_right()

    assert info_bar._is_file_changes_click(label, 1) is False


def test_info_bar_treats_file_changes_text_as_file_changes_click():
    info_bar = InfoBar("/tmp", "model")
    info_bar._file_changes = {"a.txt": (2, 1)}
    label = object()
    info_bar._row2_left = label  # type: ignore[assignment]
    info_bar._format_row2_right()

    assert info_bar._file_changes_text_start is not None
    assert info_bar._is_file_changes_click(label, info_bar._file_changes_text_start) is True
