"""Writing core fields back onto a Blender mesh as attributes."""

from __future__ import annotations

import bpy
import numpy as np


CURVATURE = "rebuilt_curvature"
IMPORTANCE = "rebuilt_importance"
DENSITY = "rebuilt_density"
REGION = "rebuilt_region"
HEATMAP = "rebuilt_heatmap"


def _replace_attribute(
    mesh: bpy.types.Mesh, name: str, data_type: str, domain: str
) -> bpy.types.Attribute:
    """Get or create an attribute, recreating it if the type no longer matches."""
    existing = mesh.attributes.get(name)
    if existing is not None:
        if existing.data_type == data_type and existing.domain == domain:
            return existing
        mesh.attributes.remove(existing)
    return mesh.attributes.new(name, data_type, domain)


def write_float(mesh: bpy.types.Mesh, name: str, values: np.ndarray, domain: str = "POINT") -> None:
    """Store a per-vertex or per-face float field."""
    expected = mesh.attributes.domain_size(domain)
    if values.shape[0] != expected:
        raise ValueError(
            f"{name!r} has {values.shape[0]} values but the {domain} domain has {expected}"
        )

    attribute = _replace_attribute(mesh, name, "FLOAT", domain)
    attribute.data.foreach_set("value", values.astype(np.float32).ravel())


def write_int(mesh: bpy.types.Mesh, name: str, values: np.ndarray, domain: str = "FACE") -> None:
    """Store a per-element integer field, such as region labels."""
    expected = mesh.attributes.domain_size(domain)
    if values.shape[0] != expected:
        raise ValueError(
            f"{name!r} has {values.shape[0]} values but the {domain} domain has {expected}"
        )

    attribute = _replace_attribute(mesh, name, "INT", domain)
    attribute.data.foreach_set("value", values.astype(np.int32).ravel())


_RAMP_SRGB = np.array(
    [
        [0.267, 0.005, 0.329],
        [0.229, 0.322, 0.545],
        [0.128, 0.567, 0.551],
        [0.369, 0.789, 0.383],
        [0.993, 0.906, 0.144],
    ]
)


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    """Convert sRGB components to linear light."""
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


_RAMP = srgb_to_linear(_RAMP_SRGB)


def heatmap_colours(values: np.ndarray) -> np.ndarray:
    """Map a 0-1 field onto linear RGBA colours. Shape (N, 4)."""
    clamped = np.clip(values.astype(np.float64), 0.0, 1.0)
    position = clamped * (len(_RAMP) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, len(_RAMP) - 1)
    blend = (position - lower)[:, None]

    rgb = _RAMP[lower] * (1.0 - blend) + _RAMP[upper] * blend
    return np.concatenate([rgb, np.ones((rgb.shape[0], 1))], axis=1)


def write_heatmap(
    mesh: bpy.types.Mesh, values: np.ndarray, name: str = HEATMAP, make_active: bool = True
) -> None:
    """Store a 0-1 field as a visible colour attribute."""
    expected = mesh.attributes.domain_size("POINT")
    if values.shape[0] != expected:
        raise ValueError(
            f"heatmap has {values.shape[0]} values but the mesh has {expected} vertices"
        )

    attribute = _replace_attribute(mesh, name, "FLOAT_COLOR", "POINT")
    attribute.data.foreach_set("color", heatmap_colours(values).astype(np.float32).ravel())

    if make_active:
        mesh.color_attributes.active_color = mesh.color_attributes.get(name)


def read_float(mesh: bpy.types.Mesh, name: str, domain: str = "POINT") -> np.ndarray:
    """Read a float field back off a mesh."""
    attribute = mesh.attributes.get(name)
    if attribute is None:
        raise KeyError(f"mesh {mesh.name!r} has no attribute {name!r}")

    values = np.empty(mesh.attributes.domain_size(domain), dtype=np.float32)
    attribute.data.foreach_get("value", values)
    return values


def clear_rebuilt_attributes(mesh: bpy.types.Mesh) -> int:
    """Remove every attribute this add-on wrote. Returns how many went."""
    ours = [name for name in mesh.attributes.keys() if name.startswith("rebuilt_")]
    for name in ours:
        attribute = mesh.attributes.get(name)
        if attribute is not None:
            mesh.attributes.remove(attribute)
    return len(ours)
