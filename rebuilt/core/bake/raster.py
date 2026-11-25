"""Which point of the surface each texel represents."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analysis.mesh_graph import fan_triangles, vertex_normals
from ..types import MeshArrays
from ..uv.texel import _corner_of_vertex_in_triangles


@dataclass(frozen=True)
class TexelMap:
    """Per-texel correspondence back onto the surface."""

    size: int

    covered: np.ndarray

    triangle: np.ndarray

    position: np.ndarray

    normal: np.ndarray | None

    triangles: np.ndarray | None = None

    @property
    def texel_count(self) -> int:
        return self.size * self.size

    @property
    def coverage(self) -> float:
        """Fraction of the tile the islands occupy, counted in texels."""
        return float(self.covered.mean()) if self.covered.size else 0.0


def rasterise(
    mesh: MeshArrays, uvs: np.ndarray, size: int, corners: np.ndarray | None = None,
    *, include_normals: bool = True,
) -> TexelMap:
    """Map texels onto the mesh, optionally omitting unused surface normals."""
    if size <= 0:
        raise ValueError(f"texture size must be positive, got {size}")

    count = size * size
    covered = np.zeros(count, dtype=bool)
    triangle = np.full(count, -1, dtype=np.int64)
    position = np.full((count, 3), np.nan, dtype=np.float64)
    normal = np.zeros((count, 3), dtype=np.float64) if include_normals else None

    if corners is None:
        corners = _corner_of_vertex_in_triangles(mesh)
    triangles = mesh.corner_verts[corners]
    if triangles.shape[0] == 0:
        return TexelMap(size, covered, triangle, position, normal)

    tri_uv = uvs[corners]
    tri_xyz = mesh.positions.astype(np.float64)[triangles]
    tri_normal = vertex_normals(mesh)[triangles] if include_normals else None

    low = np.floor(tri_uv.min(axis=1) * size).astype(np.int64)
    high = np.ceil(tri_uv.max(axis=1) * size).astype(np.int64) + 1
    np.clip(low, 0, size, out=low)
    np.clip(high, 0, size, out=high)

    for index in range(tri_uv.shape[0]):
        x0, y0 = low[index]
        x1, y1 = high[index]
        if x1 <= x0 or y1 <= y0:
            continue

        a, b, c = tri_uv[index]
        edge_1, edge_2 = b - a, c - a
        doubled_area = edge_1[0] * edge_2[1] - edge_2[0] * edge_1[1]
        if abs(doubled_area) < 1e-14:
            continue

        xs = (np.arange(x0, x1) + 0.5) / size
        ys = (np.arange(y0, y1) + 0.5) / size
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        offset = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1) - a

        w1 = (offset[:, 0] * edge_2[1] - edge_2[0] * offset[:, 1]) / doubled_area
        w2 = (edge_1[0] * offset[:, 1] - offset[:, 0] * edge_1[1]) / doubled_area
        w0 = 1.0 - w1 - w2

        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue

        flat = (np.repeat(np.arange(y0, y1), x1 - x0) * size + np.tile(np.arange(x0, x1), y1 - y0))[
            inside
        ]
        weights = np.stack([w0[inside], w1[inside], w2[inside]], axis=1)

        covered[flat] = True
        triangle[flat] = index
        position[flat] = weights @ tri_xyz[index]
        if normal is not None:
            normal[flat] = weights @ tri_normal[index]

    if normal is not None:
        lengths = np.linalg.norm(normal, axis=1)
        np.divide(normal, lengths[:, None], out=normal, where=lengths[:, None] > 1e-12)

    return TexelMap(size, covered, triangle, position, normal, triangles)


def interpolate_vertex_field(mesh: MeshArrays, texels: TexelMap, values: np.ndarray) -> np.ndarray:
    """Sample a per-vertex field at every texel. Shape (size*size,)."""
    if values.shape[0] != mesh.num_verts:
        raise ValueError(f"expected {mesh.num_verts} vertex values, got {values.shape[0]}")

    result = np.full(texels.texel_count, np.inf, dtype=np.float64)
    triangles = texels.triangles
    if triangles is None:
        triangles, _ = fan_triangles(mesh)
    if triangles.shape[0] == 0:
        return result

    if not texels.covered.any():
        return result

    corner_values = values.astype(np.float64)[triangles]
    positions = mesh.positions.astype(np.float64)
    # Per-texel triangle coordinates and barycentric work must not grow with the
    # whole texture. Keep the arithmetic and triangle selection unchanged.
    for start in range(0, texels.texel_count, 65536):
        stop = min(start + 65536, texels.texel_count)
        covered = texels.covered[start:stop]
        if not covered.any():
            continue
        chosen = texels.triangle[start:stop][covered]
        tri_xyz = positions[triangles[chosen]]
        weights = _barycentric(tri_xyz, texels.position[start:stop][covered])
        result[start:stop][covered] = (weights * corner_values[chosen]).sum(axis=1)
    return result


def _barycentric(triangles: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of points inside their triangles. Shape (N, 3)."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    v0, v1, v2 = b - a, c - a, points - a

    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)

    denominator = d00 * d11 - d01 * d01
    safe = np.where(np.abs(denominator) > 1e-20, denominator, 1.0)

    w1 = (d11 * d20 - d01 * d21) / safe
    w2 = (d00 * d21 - d01 * d20) / safe
    return np.stack([1.0 - w1 - w2, w1, w2], axis=1)
