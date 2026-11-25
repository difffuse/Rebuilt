"""Custom header icon with registration-scoped preview ownership."""

from pathlib import Path

import bpy.utils.previews


_previews = None


def header_icon() -> int:
    return _previews["rebuilt"].icon_id if _previews is not None else 0


def register() -> None:
    global _previews
    _previews = bpy.utils.previews.new()
    path = Path(__file__).resolve().parents[1] / "resources" / "icons" / "rebuilt.png"
    _previews.load("rebuilt", str(path), "IMAGE")


def unregister() -> None:
    global _previews
    if _previews is not None:
        bpy.utils.previews.remove(_previews)
        _previews = None
