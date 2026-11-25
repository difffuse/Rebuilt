"""Self-contained glTF assets shared by Unreal and Godot export."""

from pathlib import Path
import tempfile

import bpy
import numpy as np
from mathutils import Matrix

from . import image_io, output_delivery
from .image_refs import image_from_entry, map_key
from .fbx_export import export_copy
from .output_paths import require_exact_destination
from ..core.file_names import asset_stem


def _read(path, size, color=False):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        image.colorspace_settings.name = (
            "sRGB" if color and path.suffix.lower() != ".exr" else "Non-Color"
        )
        if tuple(image.size) != size:
            image.scale(*size)
        values = image_io.read_pixels(image)
        if color and not image.is_float:
            rgb = values[:, :3]
            values[:, :3] = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return values
    finally:
        bpy.data.images.remove(image)


def _png(path, pixels, size, color, depth):
    """Encode straight-alpha pixels without Blender's float-buffer unpremultiplication."""
    import struct
    import zlib

    pixels = pixels.copy()
    if color:
        rgb = np.maximum(pixels[:, :3], 0)
        pixels[:, :3] = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)
    channels = 4 if np.any(pixels[:, 3] != 1) else 3
    values = np.rint(np.clip(pixels[:, :channels], 0, 1) * ((1 << depth) - 1)).astype(
        ">u2" if depth == 16 else "u1"
    )
    rows = (
        values.reshape(size[1], size[0], channels)[::-1].copy().view(np.uint8).reshape(size[1], -1)
    )
    compressor = zlib.compressobj(6)
    compressed = bytearray()
    previous = np.zeros(rows.shape[1], dtype=np.uint8)
    for row in rows:
        compressed.extend(compressor.compress(b"\x02" + (row - previous).tobytes()))
        previous = row
    compressed.extend(compressor.flush())

    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", *size, depth, 6 if channels == 4 else 2, 0, 0, 0))
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


def export_glb(low, asset, maps, destination, *, target_engine="unreal"):
    from ..core import engines

    destination = Path(destination)
    final_path = destination / f"{engines.mesh_name(engines.get(target_engine), asset)}.glb"
    selected = list(bpy.context.selected_objects)
    active = bpy.context.view_layer.objects.active
    loaded = []
    warnings = []
    try:
        with tempfile.TemporaryDirectory(prefix="rebuilt_gltf_") as temporary:
            folder = Path(temporary)
            path = folder / final_path.name
            target = engines.get("generic")
            written = output_delivery.write_maps(maps, asset, target, folder, relink=False)
            entries = {
                e["map"]: e for e in maps if e.get("baked") and image_from_entry(e) is not None
            }
            paths = {
                key: folder / output_delivery.map_filename(target, asset, e, image_from_entry(e))
                for key, e in entries.items()
            }
            if any(str(p) not in written for p in paths.values()):
                raise RuntimeError("One or more GLB textures could not be written")
            replacements = {map_key(e): paths[key] for key, e in entries.items()}
            with export_copy(
                low, replacements, apply_transform=True, ensure_material=True
            ) as copied:
                copied.name = asset
                copied.data.name = asset
                if any(len(face.vertices) > 4 for face in copied.data.polygons):
                    triangulate = copied.modifiers.new("Export tangents", "TRIANGULATE")
                    triangulate.min_vertices = 5
                    triangulate.ngon_method = "CLIP"
                    triangulate.keep_custom_normals = True
                copied.matrix_world = Matrix.Identity(4)
                copied.data.transform(Matrix.Scale(bpy.context.scene.unit_settings.scale_length, 4))
                tree = copied.active_material.node_tree
                shader = next(n for n in tree.nodes if n.type == "BSDF_PRINCIPLED")

                def size_for(channels):
                    sizes = [
                        tuple(image_from_entry(entries[c]).size)
                        for c in channels
                        if c in entries
                    ]
                    return (max(s[0] for s in sizes), max(s[1] for s in sizes)) if sizes else (1, 1)

                def depth_for(channels):
                    return max(
                        int(entries[c].get("image_settings", {}).get("depth", "16"))
                        for c in channels
                        if c in entries
                    )

                def texture(name, pixels, size, color, depth):
                    filename = folder / (asset_stem(asset) + "_" + name + ".png")
                    _png(filename, pixels, size, color, min(depth, 16))
                    image = bpy.data.images.load(str(filename), check_existing=False)
                    loaded.append(image)
                    image.colorspace_settings.name = "sRGB" if color else "Non-Color"
                    node = tree.nodes.new("ShaderNodeTexImage")
                    node.image = image
                    return node

                if "AO" in paths or "OPACITY" in paths:
                    channels = ("BASE_COLOR", "AO", "OPACITY")
                    size = size_for(channels)
                    color = (
                        _read(paths["BASE_COLOR"], size, True)
                        if "BASE_COLOR" in paths
                        else np.tile(
                            tuple(shader.inputs["Base Color"].default_value), (size[0] * size[1], 1)
                        ).astype(np.float32)
                    )
                    if "AO" in paths:
                        color[:, :3] *= _read(paths["AO"], size)[:, :1]
                    color[:, 3] = _read(paths["OPACITY"], size)[:, 0] if "OPACITY" in paths else 1
                    node = texture("BaseColor", color, size, True, depth_for(channels))
                    tree.links.new(node.outputs["Color"], shader.inputs["Base Color"])
                    if "OPACITY" in paths:
                        tree.links.new(node.outputs["Alpha"], shader.inputs["Alpha"])

                if "ROUGHNESS" in paths or "METALLIC" in paths:
                    channels = ("ROUGHNESS", "METALLIC")
                    size = size_for(channels)
                    packed = np.ones((size[0] * size[1], 4), dtype=np.float32)
                    for key, index, socket in (
                        ("ROUGHNESS", 1, "Roughness"),
                        ("METALLIC", 2, "Metallic"),
                    ):
                        packed[:, index] = (
                            _read(paths[key], size)[:, 0]
                            if key in paths
                            else shader.inputs[socket].default_value
                        )
                    node = texture("MetallicRoughness", packed, size, False, depth_for(channels))
                    separate = tree.nodes.new("ShaderNodeSeparateColor")
                    tree.links.new(node.outputs["Color"], separate.inputs[0])
                    tree.links.new(separate.outputs[1], shader.inputs["Roughness"])
                    tree.links.new(separate.outputs[2], shader.inputs["Metallic"])

                if "COAT" in paths or "COAT_ROUGHNESS" in paths:
                    channels = ("COAT", "COAT_ROUGHNESS")
                    size = size_for(channels)
                    packed = np.ones((size[0] * size[1], 4), dtype=np.float32)
                    for key, index, socket in (
                        ("COAT", 0, "Coat Weight"),
                        ("COAT_ROUGHNESS", 1, "Coat Roughness"),
                    ):
                        packed[:, index] = (
                            _read(paths[key], size)[:, 0]
                            if key in paths
                            else shader.inputs[socket].default_value
                        )
                    node = texture("ClearCoat", packed, size, False, depth_for(channels))
                    separate = tree.nodes.new("ShaderNodeSeparateColor")
                    tree.links.new(node.outputs["Color"], separate.inputs[0])
                    tree.links.new(separate.outputs[0], shader.inputs["Coat Weight"])
                    tree.links.new(separate.outputs[1], shader.inputs["Coat Roughness"])

                specular = _prepare_specular(
                    shader, paths, entries, size_for, folder, asset, warnings
                )

                if "EMISSION" in paths or "EMISSION_STRENGTH" in paths:
                    channels = ("EMISSION", "EMISSION_STRENGTH")
                    size = size_for(channels)
                    emission = (
                        _read(paths["EMISSION"], size, True)
                        if "EMISSION" in paths
                        else np.tile(
                            tuple(shader.inputs["Emission Color"].default_value),
                            (size[0] * size[1], 1),
                        ).astype(np.float32)
                    )
                    emission[:, :3] *= (
                        _read(paths["EMISSION_STRENGTH"], size)[:, :1]
                        if "EMISSION_STRENGTH" in paths
                        else shader.inputs["Emission Strength"].default_value
                    )
                    strength = max(1.0, float(np.max(emission[:, :3])))
                    emission[:, :3] /= strength
                    emission[:, 3] = 1
                    node = texture("Emission", emission, size, True, depth_for(channels))
                    tree.links.new(node.outputs["Color"], shader.inputs["Emission Color"])
                    for link in list(shader.inputs["Emission Strength"].links):
                        tree.links.remove(link)
                    shader.inputs["Emission Strength"].default_value = strength

                for obj in selected:
                    obj.select_set(False)
                copied.select_set(True)
                bpy.context.view_layer.objects.active = copied
                bpy.context.view_layer.update()
                result = bpy.ops.export_scene.gltf(
                    filepath=str(path),
                    export_format="GLB",
                    use_selection=True,
                    export_apply=True,
                    export_animations=False,
                    export_cameras=False,
                    export_lights=False,
                    export_extras=False,
                    export_yup=True,
                    export_tangents=True,
                    export_image_format="AUTO",
                    export_original_specular=True,
                )
                if "FINISHED" not in result or not path.is_file():
                    raise RuntimeError("The GLB exporter did not finish")
                _finish_glb(path, asset, specular)
                if "SUBSURFACE" in paths or shader.inputs["Subsurface Weight"].default_value > 0:
                    warnings.append(
                        "glTF does not reproduce Blender's subsurface scattering model."
                    )
                _publish(path, final_path)
                warnings.append(
                    "Copy the GLB into your Godot project; materials and textures are included."
                    if target_engine == "godot" else
                    "Import the GLB in Unreal's Content Browser; materials and textures are included."
                )
    finally:
        for image in loaded:
            bpy.data.images.remove(image)
        for obj in selected:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = active
    return str(final_path), warnings


def _prepare_specular(shader, paths, entries, size_for, folder, asset, warnings):
    channels = ("IOR", "SPECULAR", "SPECULAR_TINT")
    size = size_for(channels)
    ior = _read(paths["IOR"], size)[:, 0] if "IOR" in paths else shader.inputs["IOR"].default_value
    level = (
        _read(paths["SPECULAR"], size)[:, 0]
        if "SPECULAR" in paths
        else shader.inputs["Specular IOR Level"].default_value
    )
    tint = (
        _read(paths["SPECULAR_TINT"], size, True)[:, :3]
        if "SPECULAR_TINT" in paths
        else np.array(shader.inputs["Specular Tint"].default_value[:3])
    )
    if np.any(np.ptp(tint, axis=-1) > 1e-5):
        warnings.append(
            "GLB export uses the luminance of colored specular tint."
        )
    luminance = tint @ np.array([0.2126, 0.7152, 0.0722])
    reflectance = np.clip(
        ((np.maximum(ior, 1) - 1) / (np.maximum(ior, 1) + 1)) ** 2 * 2 * level * luminance, 0, 0.999
    )
    peak = max(0.04, float(np.max(reflectance)))
    effective_ior = (1 + np.sqrt(peak)) / (1 - np.sqrt(peak))
    for socket, value in (
        ("IOR", effective_ior),
        ("Specular IOR Level", 0.5),
        ("Specular Tint", (1, 1, 1, 1)),
    ):
        for link in list(shader.inputs[socket].links):
            shader.id_data.links.remove(link)
        shader.inputs[socket].default_value = value
    factor = reflectance / peak
    if not any(c in paths for c in channels):
        return {"factor": float(factor), "ior": float(effective_ior)}
    pixels = np.ones((size[0] * size[1], 4), dtype=np.float32)
    pixels[:, 3] = factor
    filename = folder / (asset_stem(asset) + "_Specular.png")
    depth = min(
        16,
        max(
            int(entries[c].get("image_settings", {}).get("depth", "16"))
            for c in channels
            if c in entries
        ),
    )
    _png(filename, pixels, size, False, depth)
    return {"path": filename, "ior": float(effective_ior)}


def _finish_glb(path, asset, specular):
    import json
    import struct

    payload = path.read_bytes()
    json_length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20 : 20 + json_length])
    binary = bytearray(
        payload[28 + json_length : 28 + json_length + document["buffers"][0]["byteLength"]]
    )
    material = document["materials"][0]
    material["name"] = "M_" + asset
    for mesh in document.get("meshes", []):
        mesh["name"] = asset + "_LOD0"
    for node in document.get("nodes", []):
        if "mesh" in node:
            node["name"] = asset + "_LOD0"
    extension = {}
    if "path" in specular:
        image_bytes = specular["path"].read_bytes()
        binary.extend(b"\0" * (-len(binary) % 4))
        view = len(document["bufferViews"])
        document["bufferViews"].append(
            {"buffer": 0, "byteOffset": len(binary), "byteLength": len(image_bytes)}
        )
        binary.extend(image_bytes)
        images = document.setdefault("images", [])
        textures = document.setdefault("textures", [])
        images.append({"bufferView": view, "mimeType": "image/png", "name": asset + "_Specular"})
        textures.append({"source": len(images) - 1})
        extension["specularTexture"] = {"index": len(textures) - 1}
    elif abs(specular["factor"] - 1) > 1e-6:
        extension["specularFactor"] = specular["factor"]
    if extension:
        material.setdefault("extensions", {})["KHR_materials_specular"] = extension
        used = document.setdefault("extensionsUsed", [])
        if "KHR_materials_specular" not in used:
            used.append("KHR_materials_specular")
    if abs(specular["ior"] - 1.5) > 1e-6:
        material.setdefault("extensions", {})["KHR_materials_ior"] = {"ior": specular["ior"]}
        used = document.setdefault("extensionsUsed", [])
        if "KHR_materials_ior" not in used:
            used.append("KHR_materials_ior")
    document["buffers"][0]["byteLength"] = len(binary)
    binary.extend(b"\0" * (-len(binary) % 4))
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(binary))
        + struct.pack("<I4s", len(encoded), b"JSON")
        + encoded
        + struct.pack("<I4s", len(binary), b"BIN\0")
        + binary
    )


def _publish(source, destination):
    import os
    import shutil

    require_exact_destination(destination)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=".rebuilt_", suffix=".glb", delete=False
    ) as stream:
        staged = Path(stream.name)
    try:
        shutil.copyfile(source, staged)
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)
