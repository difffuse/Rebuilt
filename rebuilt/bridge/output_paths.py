"""Preflight generated destination names against actual filesystem entries."""
import os
from pathlib import Path
import unicodedata


class OutputNameConflict(ValueError):
    """Another spelling already occupies a generated output destination."""


def require_exact_destination(path):
    """Reject an existing case alias before replacing another asset's output.

    Inspect only the generated leaf, not user-selected parent folder spelling.
    Real directory entries preserve case-sensitive files and distinct hardlinks.
    Canonically equivalent Unicode spelling is accepted for normalizing volumes.
    """
    path = Path(path)
    try:
        with os.scandir(path.parent) as entries:
            names = [entry.name for entry in entries]
    except FileNotFoundError:
        return
    requested = unicodedata.normalize('NFC', path.name)
    if any(unicodedata.normalize('NFC', name) == requested for name in names):
        return
    if not path.exists():
        return
    for name in names:
        try:
            aliases = path.samefile(path.parent/name)
        except FileNotFoundError:
            continue
        if aliases:
            raise OutputNameConflict(
                f"Output name {path.name!r} conflicts with existing {name!r} in {path.parent}. "
                'Choose another project/export folder, or use a distinct source name before '
                'processing; existing files were kept.')
