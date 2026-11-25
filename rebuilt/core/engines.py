"""Engine-specific asset naming, axes, scale and normal conventions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .file_names import asset_stem


class EngineError(ValueError):
    """The requested export engine is not registered."""


@dataclass(frozen=True)
class EngineTarget:
    """Coordinate, naming and texture conventions for an export target."""

    identifier: str
    label: str

    suffixes: dict[str, str]

    flip_normal_green: bool = False

    axis_forward: str = "-Z"
    axis_up: str = "Y"

    mesh_prefix: str = ""

    texture_prefix: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)


_GENERIC_SUFFIXES = {
    "BASE_COLOR": "BaseColor",
    "NORMAL": "Normal",
    "AO": "AO",
    "ROUGHNESS": "Roughness",
    "METALLIC": "Metallic",
    "OPACITY": "Opacity",
}
from .bake.channels import EXTRA_CHANNELS

_GENERIC_SUFFIXES.update(
    {name: label.replace(" ", "") for name, (label, *_) in EXTRA_CHANNELS.items()}
)


_UNITY_SUFFIXES = {
    "BASE_COLOR": "BaseColor",
    "NORMAL": "Normal",
    "AO": "Occlusion",
    "ROUGHNESS": "Roughness",
    "METALLIC": "Metallic",
    "OPACITY": "Alpha",
}


_UNREAL_SUFFIXES = {
    "BASE_COLOR": "BC",
    "NORMAL": "N",
    "AO": "AO",
    "ROUGHNESS": "R",
    "METALLIC": "M",
    "OPACITY": "A",
}


ENGINES: dict[str, EngineTarget] = {
    "generic": EngineTarget(
        identifier="generic",
        label="Generic",
        suffixes=_GENERIC_SUFFIXES,
        notes=("Descriptive names and Blender axes for manual import into any destination.",),
    ),
    "unity": EngineTarget(
        identifier="unity",
        label="Unity",
        suffixes=_UNITY_SUFFIXES,
        flip_normal_green=False,
        axis_forward="-Z",
        axis_up="Y",
        notes=(
            "Unity uses the same OpenGL normal convention as Blender; "
            "the green channel is unchanged.",
        ),
    ),
    "unreal": EngineTarget(
        identifier="unreal",
        label="Unreal",
        suffixes=_UNREAL_SUFFIXES,
        flip_normal_green=True,
        axis_forward="X",
        axis_up="Z",
        mesh_prefix="SM_",
        texture_prefix="T_",
        notes=("Unreal uses DirectX normals, so the green channel is flipped during export.",),
    ),
    "godot": EngineTarget(
        identifier="godot",
        label="Godot",
        suffixes=dict(_UNREAL_SUFFIXES),
        mesh_prefix="SM_",
        texture_prefix="T_",
        notes=("Godot GLB export keeps OpenGL normals and embeds materials and textures.",),
    ),
}


def get(identifier: str) -> EngineTarget:
    """Return the conventions for the requested export engine."""
    try:
        return ENGINES[identifier]
    except KeyError:
        raise EngineError(f"unknown target engine {identifier!r}; have {sorted(ENGINES)}") from None


def map_filename(
    engine: EngineTarget, asset: str, map_name: str, size: tuple[int, int] | None = None
) -> str:
    """Name a map using its actual dimensions, never the low object's name."""
    suffix = engine.suffixes.get(map_name.upper(), map_name.lower())
    resolution = ""
    if size is not None:
        width, height = (int(value) for value in size)
        if width == height:
            label = f"{width // 1024}K" if width >= 1024 and width % 1024 == 0 else f"{width}px"
        else:
            label = f"{width}x{height}"
        resolution = f"_{label}"
    return f"{engine.texture_prefix}{asset_stem(asset)}_{suffix}{resolution}.png"


def mesh_name(engine: EngineTarget, asset: str) -> str:
    """Return the mesh name for the export target."""
    return f"{engine.mesh_prefix}{asset_stem(asset)}"


def flip_green(pixels: np.ndarray) -> np.ndarray:
    """Convert RGBA normal pixels between OpenGL and DirectX conventions."""
    flat_pixels = np.asarray(pixels, dtype=np.float32)
    if flat_pixels.ndim != 1 or flat_pixels.size % 4 != 0:
        raise ValueError(f"Expected a flat RGBA buffer; received shape {flat_pixels.shape}")
    flipped = flat_pixels.copy()
    flipped[1::4] = 1.0 - flipped[1::4]
    return flipped
