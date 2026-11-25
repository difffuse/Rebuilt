"""Unity texture packing and import instructions for native FBX delivery."""

from pathlib import Path
import uuid

import bpy
import numpy as np

from . import image_io
from .image_refs import image_from_entry, map_key
from ..core.file_names import asset_stem


def write_import_files(low, maps, image_paths, asset, folder, pipeline="urp"):
    """Write a linear metallic/occlusion/smoothness texture and a per-asset guide."""
    stem = asset_stem(asset)
    entries = {entry["map"]: entry for entry in maps if map_key(entry) in image_paths}
    inputs = {name: entries[name] for name in ("METALLIC", "AO", "ROUGHNESS") if name in entries}
    shader = (
        next(
            (
                node
                for node in low.active_material.node_tree.nodes
                if node.type == "BSDF_PRINCIPLED"
            ),
            None,
        )
        if (low.active_material and low.active_material.use_nodes)
        else None
    )
    metallic = float(shader.inputs["Metallic"].default_value) if shader else 0.0
    roughness = float(shader.inputs["Roughness"].default_value) if shader else 0.5
    files = []
    packed_path = None
    if inputs:
        sizes = [tuple(image_from_entry(entry).size) for entry in inputs.values()]
        width, height = max(size[0] for size in sizes), max(size[1] for size in sizes)
        pixels = np.empty((width * height, 4), dtype=np.float32)
        pixels[:] = (metallic, 1.0, 0.0, 1.0 - roughness)
        for channel, index in (("METALLIC", 0), ("AO", 1), ("ROUGHNESS", 3)):
            if channel not in inputs:
                continue
            image = bpy.data.images.load(
                str(image_paths[map_key(inputs[channel])]), check_existing=False
            )
            try:
                image.colorspace_settings.name = "Non-Color"
                if tuple(image.size) != (width, height):
                    image.scale(width, height)
                values = image_io.read_pixels(image)[:, 0]
                pixels[:, index] = 1.0 - values if channel == "ROUGHNESS" else values
            finally:
                bpy.data.images.remove(image)
        packed = bpy.data.images.new(
            asset + "_UnityMask", width, height, alpha=True, float_buffer=False
        )
        try:
            packed.alpha_mode = "CHANNEL_PACKED"
            packed.colorspace_settings.name = "Non-Color"
            packed.pixels.foreach_set(np.clip(pixels, 0, 1).ravel())
            packed.update()
            packed_path = Path(folder) / f"{stem}_MetallicOcclusionSmoothness.png"
            # Byte-buffer save preserves RGB when alpha stores smoothness, including zero.
            packed.filepath_raw = str(packed_path)
            packed.file_format = "PNG"
            packed.save()
            files.append(str(packed_path))
        finally:
            bpy.data.images.remove(packed)
    template = Path(__file__).resolve().parents[1] / "resources" / "unity-import.md"
    label = {"urp": "URP", "builtin": "Built-in", "hdrp": "HDRP"}[pipeline]
    guide = template.read_text(encoding="utf-8").format(asset=stem, pipeline=label)
    for channel, entry in entries.items():
        image = image_from_entry(entry)
        path = Path(image_paths[map_key(entry)])
        _write_texture_meta(
            path,
            normal=channel == "NORMAL",
            srgb=not image.colorspace_settings.is_data and path.suffix != ".exr",
            size=max(image.size),
        )
    if packed_path:
        _write_texture_meta(packed_path, normal=False, srgb=False, size=max(width, height))
    path = Path(folder) / f"{stem}_Unity_Import.md"
    path.write_text(guide, encoding="utf-8")
    return files, str(path)


def _write_texture_meta(path, *, normal, srgb, size):
    """Supply import settings for new textures, preserving existing Unity GUIDs and edits."""
    metadata = Path(str(path) + ".meta")
    if metadata.exists():
        return
    max_size = max(32, 1 << (int(size) - 1).bit_length())
    metadata.write_text(
        "fileFormatVersion: 2\n"
        f"guid: {uuid.uuid4().hex}\n"
        "TextureImporter:\n"
        "  serializedVersion: 13\n"
        "  mipmaps:\n"
        "    enableMipMap: 1\n"
        f"    sRGBTexture: {int(srgb)}\n"
        "  bumpmap:\n"
        "    convertToNormalMap: 0\n"
        "  isReadable: 0\n"
        f"  maxTextureSize: {max_size}\n"
        f"  textureType: {1 if normal else 0}\n"
        "  textureShape: 1\n"
        "  alphaSource: 1\n"
        "  alphaIsTransparency: 0\n"
        "  userData: Rebuilt\n",
        encoding="utf-8",
    )
