from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageGrab

_CLIPBOARD_IMAGE_DIR = Path("/tmp/kon-clipboard")


def save_clipboard_image() -> tuple[Path, bool] | None:
    """Return a clipboard image path and whether the path is a temporary file."""
    clipboard = ImageGrab.grabclipboard()
    if isinstance(clipboard, list):
        for item in clipboard:
            path = Path(item)
            if path.is_file():
                return path, False
        return None
    if not isinstance(clipboard, Image.Image):
        return None

    _CLIPBOARD_IMAGE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = _CLIPBOARD_IMAGE_DIR / f"clipboard-{os.getpid()}-{uuid4().hex[:8]}.png"
    clipboard.save(path, format="PNG")
    return path, True
