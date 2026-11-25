"""Shared Blender image buffers and explicit texture encoding.

Call these on Blender's main thread. Saving changes the supplied image's file
metadata; callers preserving a live map must save through a pixel copy.
"""
from __future__ import annotations

import bpy
import numpy as np


def read_pixels(image: bpy.types.Image) -> np.ndarray:
    """Image contents as (texels, 4), row-major with v increasing upward."""
    buffer = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(buffer)
    return buffer.reshape(-1, 4)


def copy_image_pixels(image: bpy.types.Image) -> bpy.types.Image:
    """Copy the live buffer and its color representation, including edited pixels."""
    pixels = read_pixels(image)
    if pixels.size == 0:
        raise RuntimeError('The image has no pixel data to copy')
    result = image.copy()
    try:
        # A copied generated image starts at generated_width/height, which can
        # differ from its live buffer after Image.scale(). Size the destination
        # before populating it; keep the copied loaded/float color representation.
        if tuple(result.size) != tuple(image.size):
            result.scale(*image.size)
        result.pixels.foreach_set(pixels.ravel())
        result.update()
        return result
    except BaseException:
        bpy.data.images.remove(result)
        raise


def save_image(image: bpy.types.Image, path: str, options=None) -> None:
    """Save texture data with explicit encoding, independent of scene exposure."""
    from pathlib import Path

    fmt = {
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".tif": "TIFF",
        ".tiff": "TIFF",
        ".exr": "OPEN_EXR",
    }[Path(path).suffix.lower()]
    options = options or {}
    image.filepath_raw = path
    image.file_format = fmt
    scene = bpy.data.scenes.new("Rebuilt texture encoding")
    try:
        settings = scene.render.image_settings
        settings.file_format = fmt
        settings.color_mode = "RGB" if fmt == "JPEG" else "RGBA"
        settings.color_depth = "8" if fmt == "JPEG" else options.get("depth", "16")
        settings.quality = options.get("quality", 85)
        settings.compression = options.get("compression", 15)
        settings.exr_codec = options.get("exr_codec", "ZIP")
        settings.tiff_codec = options.get("tiff_codec", "DEFLATE")
        scene.view_settings.view_transform = (
            "Raw" if image.colorspace_settings.is_data else "Standard"
        )
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        scene.view_settings.use_curve_mapping = False
        image.save_render(path, scene=scene)
    finally:
        bpy.data.scenes.remove(scene)
