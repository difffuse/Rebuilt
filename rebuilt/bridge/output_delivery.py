"""Complete texture delivery shared by preview, finishing and exporters.

Validation happens before writes. A supplied CancelFiles journal remains owned
by its caller; otherwise this service commits or rolls back its own batch.
Relinking is explicit because it changes the image's file/color representation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import bpy

from . import image_io
from .image_refs import image_from_entry
from .output_paths import require_exact_destination
from ..core import engines
from ..core.bake.channels import IMAGE_EXTENSIONS
from ..pipeline.context import ContextError


def map_filename(engine_target, asset, entry, image):
    name = engines.map_filename(engine_target, asset, entry["map"], tuple(image.size))
    fmt = entry.get("image_settings", {}).get("format", "PNG")
    return str(Path(name).with_suffix("." + IMAGE_EXTENSIONS[fmt]))


def required_map_images(maps):
    """Validate the complete requested set before touching files or presentation."""
    images = []
    for entry in maps:
        channel = entry.get('map', '')
        if not entry.get('baked'):
            raise ContextError(f"Required texture {channel!r} was not baked. Bake it before finishing.")
        image = image_from_entry(entry)
        if image is None or min(image.size) <= 0:
            raise ContextError(f"Required texture {channel!r} is missing or empty. Restore it or bake again.")
        images.append((entry, image))
    return images


def save_through_copy(image: bpy.types.Image, path: str, flip_green: bool, options=None) -> None:
    """Write image pixels through a copy without changing the live map."""
    image_copy = image_io.copy_image_pixels(image)
    try:
        if flip_green:
            image_copy.pixels.foreach_set(engines.flip_green(image_io.read_pixels(image_copy).ravel()))
            image_copy.update()
        image_io.save_image(image_copy, path, options)
    finally:
        bpy.data.images.remove(image_copy)


def save_map(
    image: bpy.types.Image, path: str, flip_green: bool = False, relink: bool = True, options=None
) -> None:
    """Save a map and optionally reload it for faithful viewport color."""
    save_through_copy(image, path, flip_green=flip_green, options=options)
    if relink and not flip_green:
        reload_map(image, path)


def reload_map(image, path):
    non_colour = image.colorspace_settings.is_data
    image.filepath_raw = path
    image.source = "FILE"
    image.reload()
    # Reload keeps the old image's color-space tag. Colour float bake targets
    # are linear, while the delivered PNG/JPEG/TIFF buffer is sRGB encoded.
    if not non_colour:
        image.colorspace_settings.name = (
            "Linear Rec.709" if path.lower().endswith(".exr") else "sRGB"
        )


def write_maps(
    maps: list[dict[str, Any]],
    asset: str,
    engine_target: engines.EngineTarget,
    folder: Path | None,
    relink: bool = False,
    cancel_files=None,
) -> list[str]:
    """Write every required map or raise; a borrowed journal stays caller-owned."""
    images = required_map_images(maps)
    if folder is None:
        return []

    planned = []
    destinations = set()
    for entry, image in images:
        path = folder / map_filename(engine_target, asset, entry, image)
        identity = path.resolve()
        if identity in destinations:
            raise ContextError(f"Required textures have the same output destination: {path}")
        destinations.add(identity)
        planned.append((entry, image, path))

    if not planned:
        return []
    for _entry, _image, path in planned:
        require_exact_destination(path)
    if cancel_files is None:
        from ..pipeline.cancel_files import CancelFiles, FileCommitUncertain
        journal = CancelFiles(
            bpy.utils.user_resource('CONFIG', path='rebuilt/recovery', create=True),
            origin={'version': 1, 'kind': 'map_batch', 'blend': bpy.data.filepath,
                    'asset': asset})
        journal.hold()
        try:
            try:
                written = write_maps(maps, asset, engine_target, folder,
                                      relink=relink, cancel_files=journal)
                journal.commit()
            except FileCommitUncertain:
                raise
            except BaseException:
                if journal.phase != 'committed':
                    journal.rollback()
                raise
            journal.cleanup()
            return written
        finally:
            journal.release()

    written: list[str] = []
    for entry, image, path in planned:
        if cancel_files is not None:
            cancel_files.write(path, lambda staging: save_through_copy(
                image, str(staging),
                flip_green=entry["map"] == "NORMAL" and engine_target.flip_normal_green,
                options=entry.get("image_settings")))
            if relink:
                reload_map(image, str(path))
        else:
            folder.mkdir(parents=True, exist_ok=True)
            save_map(
                image, str(path),
                flip_green=entry["map"] == "NORMAL" and engine_target.flip_normal_green,
                relink=relink, options=entry.get("image_settings"))
        written.append(str(path))
        if relink:
            entry["colour_space"] = image.colorspace_settings.name
    return written
