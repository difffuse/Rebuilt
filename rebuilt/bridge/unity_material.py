"""Native Unity materials and FBX remaps for import without manual connections."""

import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import io
import uuid

import bpy
import numpy as np

from . import image_io
from .image_refs import image_from_entry, map_key
from ..core.file_names import asset_stem


URP_LIT = "{fileID: 4800000, guid: 933532a4fcc9baf4fa0491de14d08ed7, type: 3}"
HDRP_LIT = "{fileID: 4800000, guid: 6e4ae4064600d784cac1e41a9e6f2e59, type: 3}"
BUILTIN_STANDARD = "{fileID: 46, guid: 0000000000000000f000000000000000, type: 0}"


def guid(path):
    metadata = Path(str(path) + ".meta")
    if metadata.exists():
        match = re.search(r"^guid: ([0-9a-f]{32})$", metadata.read_text(), re.MULTILINE)
        if match:
            return match[1]
        raise RuntimeError(f"Unity metadata has no valid GUID: {metadata}")
    return uuid.uuid4().hex


def _reference(path, file_id=2800000, kind=3):
    return f"{{fileID: {file_id}, guid: {guid(path)}, type: {kind}}}"


def _shader_value(shader, name, fallback):
    return shader.inputs[name].default_value if shader and name in shader.inputs else fallback


def _read(path, size, *, color=False):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        image.colorspace_settings.name = (
            "sRGB" if color and Path(path).suffix != ".exr" else "Non-Color"
        )
        if tuple(image.size) != size:
            image.scale(*size)
        pixels = image_io.read_pixels(image)
        if color and not image.is_float:
            rgb = pixels[:, :3]
            pixels[:, :3] = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return pixels
    finally:
        bpy.data.images.remove(image)


def _write_color(path, pixels, size, *, hdr=False):
    image = bpy.data.images.new("Unity color", *size, alpha=True, float_buffer=hdr)
    try:
        pixels = pixels.copy()
        if hdr:
            image.colorspace_settings.name = "Linear Rec.709"
        else:
            image.colorspace_settings.name = "sRGB"
            rgb = np.maximum(pixels[:, :3], 0)
            pixels[:, :3] = np.where(
                rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055
            )
            pixels = np.clip(pixels, 0, 1)
        image.pixels.foreach_set(pixels.astype(np.float32).ravel())
        image.update()
        if hdr:
            image_io.save_image(image, str(path), {"format": "OPEN_EXR", "depth": "16"})
        else:
            image.filepath_raw = str(path)
            image.file_format = "PNG"
            image.save()
    finally:
        bpy.data.images.remove(image)


def write_material(
    low,
    maps,
    image_paths,
    asset,
    folder,
    fbx_path,
    material_names,
    packed_path=None,
    pipeline="urp",
):
    from .unity_export import _write_texture_meta

    stem = asset_stem(asset)
    folder = Path(folder)
    entries = {entry["map"]: entry for entry in maps if map_key(entry) in image_paths}
    paths = {channel: Path(image_paths[map_key(entry)]) for channel, entry in entries.items()}
    shader = (
        next(
            (
                node
                for node in low.active_material.node_tree.nodes
                if node.type == "BSDF_PRINCIPLED"
            ),
            None,
        )
        if low.active_material and low.active_material.use_nodes
        else None
    )
    alpha = float(_shader_value(shader, "Alpha", 1.0))
    base = tuple(_shader_value(shader, "Base Color", (0.8, 0.8, 0.8, 1)))
    transparent = alpha < 0.999 or "OPACITY" in paths
    generated = []
    if "BASE_COLOR" not in paths or transparent:
        channels = [entries[c] for c in ("BASE_COLOR", "OPACITY") if c in entries]
        size = (
            (
                max(image_from_entry(e).size[0] for e in channels),
                max(image_from_entry(e).size[1] for e in channels),
            )
            if channels
            else (1, 1)
        )
        pixels = (
            _read(paths["BASE_COLOR"], size, color=True)
            if "BASE_COLOR" in paths
            else np.tile(base, (size[0] * size[1], 1)).astype(np.float32)
        )
        pixels[:, 3] = _read(paths["OPACITY"], size)[:, 0] if "OPACITY" in paths else alpha
        transparent = bool(np.min(pixels[:, 3]) < 0.999)
        paths["BASE_COLOR"] = folder / f"{stem}_UnityBaseColor.png"
        _write_color(paths["BASE_COLOR"], pixels, size)
        _write_texture_meta(paths["BASE_COLOR"], normal=False, srgb=True, size=max(size))
        generated.append(str(paths["BASE_COLOR"]))
    emission = tuple(_shader_value(shader, "Emission Color", (0, 0, 0, 1)))
    strength = float(_shader_value(shader, "Emission Strength", 0))
    emitting = ("EMISSION" in paths or max(emission[:3]) > 0) and (
        "EMISSION_STRENGTH" in paths or strength > 0
    )
    if emitting:
        channels = [entries[c] for c in ("EMISSION", "EMISSION_STRENGTH") if c in entries]
        size = (
            (
                max(image_from_entry(e).size[0] for e in channels),
                max(image_from_entry(e).size[1] for e in channels),
            )
            if channels
            else (1, 1)
        )
        pixels = (
            _read(paths["EMISSION"], size, color=True)
            if "EMISSION" in paths
            else np.tile(emission, (size[0] * size[1], 1)).astype(np.float32)
        )
        factor = (
            _read(paths["EMISSION_STRENGTH"], size)[:, :1]
            if "EMISSION_STRENGTH" in paths
            else strength
        )
        pixels[:, :3] *= factor
        pixels[:, 3] = 1
        paths["EMISSION"] = folder / f"{stem}_UnityEmission.exr"
        _write_color(paths["EMISSION"], pixels, size, hdr=True)
        _write_texture_meta(paths["EMISSION"], normal=False, srgb=False, size=max(size))
        generated.append(str(paths["EMISSION"]))
    specular_path = None
    ior = float(_shader_value(shader, "IOR", 1.5))
    level = float(_shader_value(shader, "Specular IOR Level", 0.5))
    tint = tuple(_shader_value(shader, "Specular Tint", (1, 1, 1, 1)))
    if (
        any(c in paths for c in ("IOR", "SPECULAR", "SPECULAR_TINT"))
        or abs(ior - 1.5) > 1e-5
        or abs(level - 0.5) > 1e-5
        or any(abs(value - 1) > 1e-5 for value in tint[:3])
    ):
        size = (
            tuple(image_from_entry(entries["BASE_COLOR"]).size)
            if "BASE_COLOR" in entries
            else (1, 1)
        )
        if entries:
            size = (
                max(image_from_entry(e).size[0] for e in entries.values()),
                max(image_from_entry(e).size[1] for e in entries.values()),
            )
        reflectance_ior = _read(paths["IOR"], size)[:, :1] if "IOR" in paths else ior
        specular_level = _read(paths["SPECULAR"], size)[:, :1] if "SPECULAR" in paths else level
        specular_tint = (
            _read(paths["SPECULAR_TINT"], size, color=True)[:, :3]
            if "SPECULAR_TINT" in paths
            else np.array(tint[:3])
        )
        metal = (
            _read(paths["METALLIC"], size)[:, :1]
            if "METALLIC" in paths
            else float(_shader_value(shader, "Metallic", 0))
        )
        dielectric = (
            ((reflectance_ior - 1) / (reflectance_ior + 1)) ** 2
            * (2 * specular_level)
            * specular_tint
        )
        pixels = _read(paths["BASE_COLOR"], size, color=True)
        pixels[:, :3] = np.clip((1 - metal) * dielectric + metal * pixels[:, :3], 0, 1)
        pixels[:, 3] = 1 - (
            _read(paths["ROUGHNESS"], size)[:, 0]
            if "ROUGHNESS" in paths
            else float(_shader_value(shader, "Roughness", 0.5))
        )
        specular_path = folder / f"{stem}_UnitySpecularSmoothness.png"
        _write_color(specular_path, pixels, size)
        _write_texture_meta(specular_path, normal=False, srgb=True, size=max(size))
        generated.append(str(specular_path))
    textures = {"_BaseMap": paths["BASE_COLOR"], "_MainTex": paths["BASE_COLOR"]}
    keywords = []
    if "NORMAL" in paths:
        textures["_BumpMap"] = paths["NORMAL"]
        keywords.append("_NORMALMAP")
    if packed_path:
        textures["_MetallicGlossMap"] = Path(packed_path)
        textures["_OcclusionMap"] = Path(packed_path)
        keywords += [
            "_METALLICSPECGLOSSMAP" if pipeline == "urp" else "_METALLICGLOSSMAP",
            "_OCCLUSIONMAP",
        ]
    if emitting:
        textures["_EmissionMap"] = paths["EMISSION"]
        keywords.append("_EMISSION")
    if transparent:
        keywords += (
            ["_SURFACE_TYPE_TRANSPARENT", "_ALPHAPREMULTIPLY_ON"]
            if pipeline == "urp"
            else ["_ALPHABLEND_ON"]
        )
    roughness = float(_shader_value(shader, "Roughness", 0.5))
    floats = {
        "_WorkflowMode": 1,
        "_Metallic": float(_shader_value(shader, "Metallic", 0)),
        "_Smoothness": 1 if packed_path else 1 - roughness,
        "_Glossiness": 1 - roughness,
        "_GlossMapScale": 1,
        "_SmoothnessTextureChannel": 0,
        "_OcclusionStrength": 1,
        "_BumpScale": 1,
        "_Surface": int(transparent),
        "_Blend": 0,
        "_SrcBlend": 1 if pipeline == "urp" or not transparent else 5,
        "_DstBlend": 10 if transparent else 0,
        "_SrcBlendAlpha": 1,
        "_DstBlendAlpha": 10 if transparent else 0,
        "_ZWrite": 0 if transparent else 1,
        "_Cull": 2,
        "_AlphaClip": 0,
        "_Cutoff": 0.5,
        "_Mode": 2 if transparent else 0,
        "_ReceiveShadows": 1,
        "_SpecularHighlights": 1,
        "_EnvironmentReflections": 1,
        "_BlendModePreserveSpecular": 1,
    }
    colors = {
        "_BaseColor": (1, 1, 1, 1),
        "_Color": (1, 1, 1, 1),
        "_EmissionColor": (1, 1, 1, 1) if emitting else (0, 0, 0, 1),
    }
    shader_reference = URP_LIT if pipeline == "urp" else BUILTIN_STANDARD
    if pipeline == "hdrp":
        shader_reference = HDRP_LIT
        textures = {"_BaseColorMap": paths["BASE_COLOR"]}
        keywords = []
        if "NORMAL" in paths:
            textures["_NormalMap"] = paths["NORMAL"]
            keywords += ["_NORMALMAP", "_NORMALMAP_TANGENT_SPACE"]
        if packed_path:
            textures["_MaskMap"] = Path(packed_path)
            keywords.append("_MASKMAP")
        if emitting:
            textures["_EmissiveColorMap"] = paths["EMISSION"]
            keywords.append("_EMISSIVE_COLOR_MAP")
        if transparent:
            keywords += ["_SURFACE_TYPE_TRANSPARENT", "_BLENDMODE_ALPHA"]
        floats.update(
            {
                "_MaterialID": 1,
                "_SurfaceType": int(transparent),
                "_BlendMode": 0,
                "_SrcBlend": 1,
                "_DstBlend": 10 if transparent else 0,
                "_AlphaSrcBlend": 1,
                "_AlphaDstBlend": 10 if transparent else 0,
                "_TransparentZWrite": 0,
                "_CullMode": 2,
                "_CullModeForward": 2,
                "_AlphaCutoffEnable": 0,
                "_AlphaCutoff": 0.5,
                "_NormalMapSpace": 0,
                "_NormalScale": 1,
                "_UVBase": 0,
                "_InvTilingScale": 1,
                "_MetallicRemapMin": 0,
                "_MetallicRemapMax": 1,
                "_SmoothnessRemapMin": 0,
                "_SmoothnessRemapMax": 1,
                "_AORemapMin": 0,
                "_AORemapMax": 1,
                "_AlphaRemapMin": 0,
                "_AlphaRemapMax": 1,
                "_UseEmissiveIntensity": 0,
                "_EmissiveIntensity": 1,
                "_EmissiveExposureWeight": 1,
                "_Ior": float(_shader_value(shader, "IOR", 1.5)),
            }
        )
        colors["_EmissiveColor"] = colors["_EmissionColor"]
        colors["_EmissiveColorLDR"] = colors["_EmissionColor"]
        colors["_UVMappingMask"] = (1, 0, 0, 0)
    if specular_path:
        colors["_SpecColor"] = (1, 1, 1, 1)
        if pipeline == "hdrp":
            textures["_SpecularColorMap"] = specular_path
            colors["_SpecularColor"] = (1, 1, 1, 1)
            floats["_MaterialID"] = 4
            keywords += ["_MATERIAL_FEATURE_SPECULAR_COLOR", "_SPECULARCOLORMAP"]
        else:
            textures["_SpecGlossMap"] = specular_path
            textures.pop("_MetallicGlossMap", None)
            floats["_WorkflowMode"] = 0
            floats["_Smoothness"] = 1
            floats["_GlossMapScale"] = 1
            if pipeline == "urp":
                keywords += ["_SPECULAR_SETUP", "_METALLICSPECGLOSSMAP"]
            else:
                shader_reference = "{fileID: 45, guid: 0000000000000000f000000000000000, type: 0}"
                keywords = [k for k in keywords if k != "_METALLICGLOSSMAP"]
                keywords.append("_SPECGLOSSMAP")
        keywords = list(dict.fromkeys(keywords))
    coat = float(_shader_value(shader, "Coat Weight", 0))
    if pipeline in {"urp", "hdrp"} and ("COAT" in paths or coat > 0):
        if pipeline == "urp":
            shader_reference = "{fileID: 4800000, guid: ee7e4c9a5f6364b688a332c67fc32cca, type: 3}"
            floats.update(
                {
                    "_ClearCoat": 1,
                    "_ClearCoatMask": coat,
                    "_ClearCoatSmoothness": 1
                    - float(_shader_value(shader, "Coat Roughness", 0.03)),
                }
            )
            if "COAT" in paths or "COAT_ROUGHNESS" in paths:
                channels = [entries[c] for c in ("COAT", "COAT_ROUGHNESS") if c in entries]
                size = (
                    max(image_from_entry(e).size[0] for e in channels),
                    max(image_from_entry(e).size[1] for e in channels),
                )
                pixels = np.zeros((size[0] * size[1], 4), dtype=np.float32)
                pixels[:, 0] = _read(paths["COAT"], size)[:, 0] if "COAT" in paths else coat
                pixels[:, 1] = 1 - (
                    _read(paths["COAT_ROUGHNESS"], size)[:, 0]
                    if "COAT_ROUGHNESS" in paths
                    else float(_shader_value(shader, "Coat Roughness", 0.03))
                )
                pixels[:, 3] = 1
                image = bpy.data.images.new("Unity coat", *size, alpha=True, float_buffer=False)
                try:
                    image.colorspace_settings.name = "Non-Color"
                    image.pixels.foreach_set(np.clip(pixels, 0, 1).ravel())
                    image.update()
                    path = folder / f"{stem}_UnityCoat.png"
                    image.filepath_raw = str(path)
                    image.file_format = "PNG"
                    image.save()
                finally:
                    bpy.data.images.remove(image)
                _write_texture_meta(path, normal=False, srgb=False, size=max(size))
                generated.append(str(path))
                textures["_ClearCoatMap"] = path
                floats["_ClearCoatMask"] = floats["_ClearCoatSmoothness"] = 1
                keywords.append("_CLEARCOATMAP")
            else:
                keywords.append("_CLEARCOAT")
        else:
            floats["_CoatMask"] = 1 if "COAT" in paths else coat
            if "COAT" in paths:
                textures["_CoatMaskMap"] = paths["COAT"]
            keywords.append("_MATERIAL_FEATURE_CLEAR_COAT")
    lines = [
        "%YAML 1.1",
        "%TAG !u! tag:unity3d.com,2011:",
        "--- !u!21 &2100000",
        "Material:",
        "  serializedVersion: 8",
        "  m_ObjectHideFlags: 0",
        f"  m_Name: {json.dumps(asset + '_Unity')}",
        f"  m_Shader: {shader_reference}",
        "  m_ValidKeywords:" if keywords else "  m_ValidKeywords: []",
    ]
    lines += [f"  - {keyword}" for keyword in keywords]
    lines += [
        "  m_InvalidKeywords: []",
        f"  m_LightmapFlags: {2 if emitting else 4}",
        "  m_EnableInstancingVariants: 0",
        "  m_DoubleSidedGI: 0",
        f"  m_CustomRenderQueue: {3000 if transparent else -1}",
        "  stringTagMap:",
        f"    RenderType: {'Transparent' if transparent else 'Opaque'}",
        "  disabledShaderPasses: []",
        "  m_SavedProperties:",
        "    serializedVersion: 3",
        "    m_TexEnvs:",
    ]
    for name, path in textures.items():
        lines += [
            f"    - {name}:",
            f"        m_Texture: {_reference(path)}",
            "        m_Scale: {x: 1, y: 1}",
            "        m_Offset: {x: 0, y: 0}",
        ]
    lines += ["    m_Ints: []", "    m_Floats:"]
    lines += [f"    - {name}: {float(value):.9g}" for name, value in floats.items()]
    lines += ["    m_Colors:"]
    for name, value in colors.items():
        lines.append(
            f"    - {name}: {{r: {value[0]}, g: {value[1]}, b: {value[2]}, a: {value[3]}}}"
        )
    material_path = folder / f"{stem}_Unity.mat"
    material_guid = guid(material_path)
    material_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(str(material_path) + ".meta").write_text(
        f"fileFormatVersion: 2\nguid: {material_guid}\nNativeFormatImporter:\n  externalObjects: {{}}\n  mainObjectFileID: 2100000\n  userData: Rebuilt\n",
        encoding="utf-8",
    )
    model_meta = Path(str(fbx_path) + ".meta")
    if model_meta.exists():
        model_text = model_meta.read_text()
    else:
        model_text = f"fileFormatVersion: 2\nguid: {guid(fbx_path)}\nModelImporter:\n  serializedVersion: 24200\n  externalObjects: []\n  materials:\n    materialImportMode: 2\n    materialLocation: 1\n  meshes:\n    globalScale: 1\n    useFileUnits: 1\n    bakeAxisConversion: 0\n  importAnimation: 0\n  userData: Rebuilt\n"
    remaps = ["  externalObjects:"]
    for name in material_names:
        remaps += [
            "  - first:",
            "      type: UnityEngine:Material",
            "      assembly: UnityEngine.CoreModule",
            f"      name: {json.dumps(name)}",
            f"    second: {_reference(material_path, 2100000, 2)}",
        ]
    model_text, count = re.subn(
        r"^  externalObjects:.*?(?=^  [A-Za-z_]|\Z)",
        lambda _match: "\n".join(remaps) + "\n",
        model_text,
        flags=re.MULTILINE | re.DOTALL,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Cannot update this FBX metadata's material remaps")
    model_meta.write_text(model_text, encoding="utf-8")
    return str(material_path), generated


def restore_metadata(package, folder, asset):
    """Reuse package metadata; read legacy sidecars without modifying the old export."""
    stem = asset_stem(asset)
    package, folder = Path(package), Path(folder)
    for path in package.parent.iterdir():
        if (
            path.is_file()
            and path.name.endswith(".meta")
            and (path.name == f"{stem}.fbx.meta" or path.name.startswith(stem + "_"))
        ):
            (folder / path.name).write_bytes(path.read_bytes())
    if not package.exists():
        return
    entries = {}
    try:
        with tarfile.open(package, "r|gz") as archive:
            for member in archive:
                key, _, name = member.name.rpartition("/")
                if name not in {"pathname", "asset.meta"} or not member.isfile():
                    continue
                if member.size > 1024 * 1024:
                    raise ValueError("Existing Unity package metadata is too large")
                entries.setdefault(key, {})[name] = archive.extractfile(member).read()
        root = PurePosixPath("Assets/Rebuilt") / stem
        for entry in entries.values():
            if "pathname" not in entry or "asset.meta" not in entry:
                continue
            path = PurePosixPath(entry["pathname"].decode("utf-8"))
            if path.parent != root or path.name in {".", ".."} or "\\" in path.name:
                continue
            (folder / (path.name + ".meta")).write_bytes(entry["asset.meta"])
    except (tarfile.TarError, UnicodeError, EOFError) as exc:
        raise ValueError(f"Cannot read metadata from the existing Unity package: {exc}") from exc


def referenced_files(paths):
    """Keep model, material and only textures referenced by the material."""
    paths = list(dict.fromkeys(Path(path) for path in paths))
    materials = [path for path in paths if path.suffix == ".mat"]
    if materials:
        referenced = set()
        for material in materials:
            referenced.update(re.findall(r"guid: ([0-9a-f]{32})", material.read_text()))
        paths = [
            path for path in paths if path.suffix in {".fbx", ".mat"} or guid(path) in referenced
        ]
    return paths


def write_package(asset, folder, paths):
    """Build legacy packages for compatibility checks and metadata migration."""
    stem = asset_stem(asset)
    destination = Path(folder) / f"{stem}.unitypackage"
    paths = referenced_files(paths)
    with tarfile.open(destination, "w:gz", format=tarfile.GNU_FORMAT) as archive:

        def directory(name):
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            archive.addfile(info)

        def content(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))

        directory(".")
        for name, identity in (
            ("Assets/Rebuilt", "Rebuilt"),
            (f"Assets/Rebuilt/{stem}", guid(paths[0])),
        ):
            key = uuid.uuid5(uuid.NAMESPACE_URL, "rebuilt/folder/" + identity).hex
            directory("./" + key)
            content(f"./{key}/pathname", name.encode("utf-8"))
            content(
                f"./{key}/asset.meta",
                (
                    f"fileFormatVersion: 2\nguid: {key}\nfolderAsset: yes\nDefaultImporter:\n"
                    "  externalObjects: {}\n  userData: Rebuilt\n"
                ).encode("utf-8"),
            )
        for path in paths:
            key = guid(path)
            directory("./" + key)
            archive.add(path, arcname=f"./{key}/asset", recursive=False)
            archive.add(Path(str(path) + ".meta"), arcname=f"./{key}/asset.meta", recursive=False)
            content(f"./{key}/pathname", f"Assets/Rebuilt/{stem}/{path.name}".encode("utf-8"))
    return str(destination)
