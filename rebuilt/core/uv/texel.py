"""Texel density: how many pixels of texture cover a unit of surface."""

from __future__ import annotations

import numpy as np

from ..types import MeshArrays
from . import native_analysis


def uv_face_areas(mesh: MeshArrays, uvs: np.ndarray) -> np.ndarray:
    """Area of each face in UV space, unsigned. Shape (F,)."""
    triangles, face_of_tri = surface_triangles(mesh)
    if triangles.shape[0] == 0:
        return np.zeros(mesh.num_faces, dtype=np.float64)

    corners = _corner_of_vertex_in_triangles(mesh)
    a, b, c = (uvs[corners[:, i]] for i in range(3))
    doubled = np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    )
    return np.bincount(face_of_tri, weights=0.5 * doubled, minlength=mesh.num_faces)


def signed_uv_triangle_areas(mesh: MeshArrays, uvs: np.ndarray) -> np.ndarray:
    """Signed UV areas using fixed source-polygon triangulation. Shape (T,)."""
    corners = _corner_of_vertex_in_triangles(mesh)
    if corners.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)

    a, b, c = (uvs[corners[:, i]] for i in range(3))
    return 0.5 * (
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    )


def _corner_of_vertex_in_triangles(mesh: MeshArrays) -> np.ndarray:
    """The native engine's fixed source-polygon triangles. Shape (T, 3)."""
    return native_analysis.analyze(mesh)['triangle_corners']


def surface_triangles(mesh: MeshArrays) -> tuple[np.ndarray, np.ndarray]:
    analysis = native_analysis.analyze(mesh)
    return mesh.corner_verts[analysis['triangle_corners']], analysis['triangle_faces']


def surface_face_areas(mesh: MeshArrays) -> np.ndarray:
    return native_analysis.analyze(mesh)['face_area_3d']


def texel_density(mesh: MeshArrays, uvs: np.ndarray, texture_size: int) -> np.ndarray:
    """Pixels per scene unit for each face. Shape (F,)."""
    surface = surface_face_areas(mesh)
    uv = uv_face_areas(mesh, uvs)

    ratio = np.divide(uv, surface, out=np.zeros_like(uv), where=surface > 0)
    return texture_size * np.sqrt(ratio)


def average_texel_density(mesh: MeshArrays, uvs: np.ndarray, texture_size: int) -> float:
    """Area-weighted mean texel density."""
    surface = surface_face_areas(mesh)
    total = float(surface.sum())
    if total <= 0:
        return 0.0

    density = texel_density(mesh, uvs, texture_size)
    return float((density * surface).sum() / total)


def scale_to_texel_density(
    mesh: MeshArrays,
    uvs: np.ndarray,
    texture_size: int,
    target: float,
) -> tuple[np.ndarray, float]:
    """Rescale UVs so the average density hits `target` pixels per unit."""
    current = average_texel_density(mesh, uvs, texture_size)
    if current <= 0 or target <= 0:
        return uvs.copy(), 1.0

    factor = target / current
    return uvs * factor, float(factor)


def uv_bounds(uvs: np.ndarray) -> tuple[float, float, float, float]:
    """Bounding box of a UV set as `(min_u, min_v, max_u, max_v)`."""
    if uvs.shape[0] == 0:
        return (0.0, 0.0, 0.0, 0.0)
    low = uvs.min(axis=0)
    high = uvs.max(axis=0)
    return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))


def fits_in_tile(uvs: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Whether every UV sits inside the 0-1 tile."""
    min_u, min_v, max_u, max_v = uv_bounds(uvs)
    return (
        min_u >= -tolerance
        and min_v >= -tolerance
        and max_u <= 1.0 + tolerance
        and max_v <= 1.0 + tolerance
    )


def density_evenness(mesh, uvs: np.ndarray, texture_size: int) -> float:
    """How fairly the layout shares the texture out, from 0 to 1."""
    tris, _ = surface_triangles(mesh)
    if tris.size == 0:
        return 0.0

    corners = mesh.positions[tris]
    world = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )

    layout = uvs[_corner_of_vertex_in_triangles(mesh)]
    uv_area = 0.5 * np.abs(np.cross(layout[:, 1] - layout[:, 0], layout[:, 2] - layout[:, 0]))

    usable = (world > 1e-12) & (uv_area > 1e-14)
    if not usable.any():
        return 0.0

    density = np.sqrt(uv_area[usable] / world[usable]) * texture_size
    weight = world[usable] / world[usable].sum()
    order = np.argsort(density)
    cumulative = np.cumsum(weight[order])
    low = float(density[order][np.searchsorted(cumulative, 0.10)])
    high = float(density[order][np.searchsorted(cumulative, 0.90)])
    return low / high if high > 0.0 else 0.0
