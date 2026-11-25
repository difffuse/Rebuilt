"""Publish complete asset folders while preserving unrelated artist files."""

import json
import logging
import os
from pathlib import Path
import shutil
import stat
from tempfile import mkdtemp

from .output_paths import require_exact_destination


def _make_temporary_file_writable(path):
    """Clear read-only only on an unshared regular file in our temporary tree."""
    try:
        info = Path(path).lstat()
    except FileNotFoundError:
        return False
    mode = info.st_mode
    # The backup is the renamed original; hard links can still exist outside it.
    if info.st_nlink == 1 and stat.S_ISREG(mode) and not mode & stat.S_IWRITE:
        os.chmod(path, mode | stat.S_IWRITE)
        return True
    return False


def _remove_readonly_temporary_file(function, path, exc_info):
    error = exc_info[1]
    if (function is os.unlink and isinstance(error, PermissionError)
            and _make_temporary_file_writable(path)):
        function(path)
    else:
        raise error


def publish(destination, asset, engine, paths):
    target = Path(destination) / asset
    if Path(asset).name != asset or asset in {".", ".."}:
        raise ValueError("The asset name must be a single folder name")
    require_exact_destination(target)
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError("The asset destination must be a regular folder")
    if target.exists() and any(p.is_symlink() for p in target.rglob("*")):
        raise ValueError("The asset destination contains symbolic links")
    paths = list(dict.fromkeys(Path(p) for p in paths))
    for name in [".rebuilt-export.json", *(p.name for p in paths)]:
        require_exact_destination(target / name)
    temporary = Path(mkdtemp(prefix=".rebuilt-publish-", dir=destination))
    backup = temporary / "previous"
    committed = False
    try:
        staging = temporary / "delivery"
        if target.exists():
            shutil.copytree(target, staging)
        else:
            staging.mkdir()
        manifest = staging / ".rebuilt-export.json"
        owned = json.loads(manifest.read_text()) if manifest.exists() else {}
        names = {p.name for p in paths}
        for name in set(owned.get(engine, [])) - names:
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError("Invalid file name in the export manifest")
            require_exact_destination(staging / name)
            _make_temporary_file_writable(staging / name)
            (staging / name).unlink(missing_ok=True)
        for path in paths:
            _make_temporary_file_writable(staging / path.name)
            shutil.copy2(path, staging / path.name)
        owned[engine] = sorted(names)
        _make_temporary_file_writable(manifest)
        manifest.write_text(json.dumps(owned, indent=2) + "\n")
        if target.exists():
            target.rename(backup)
        try:
            staging.rename(target)
        except OSError:
            if backup.exists():
                try:
                    backup.rename(target)
                except OSError as exc:
                    raise OSError(f"Previous export retained at {backup}: {exc}") from exc
            raise
        committed = True
    finally:
        if committed or not backup.exists():
            try:
                shutil.rmtree(temporary, onerror=_remove_readonly_temporary_file)
            except OSError as exc:
                if not committed:
                    raise
                logging.getLogger(__name__).warning(
                    "Export published at %s, but temporary cleanup failed at %s: %s",
                    target, temporary, exc,
                )
    return target
