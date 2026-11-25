"""Per-vertex curvature."""

from __future__ import annotations

import numpy as np

from ..types import MeshArrays
from .mesh_graph import face_areas, fan_triangles, manifold_edges


def vertex_areas(mesh: MeshArrays) -> np.ndarray:
    """Barycentric area around each vertex — a third of each incident face."""
    areas = face_areas(mesh)
    corner_verts = mesh.corner_verts.astype(np.int64)
    face_of_corner = np.repeat(
        np.arange(mesh.num_faces, dtype=np.int64), mesh.face_sizes.astype(np.int64)
    )
    share = areas[face_of_corner] / mesh.face_sizes.astype(np.float64)[face_of_corner]
    return np.bincount(corner_verts, weights=share, minlength=mesh.num_verts)


def mean_curvature(mesh: MeshArrays, normals: np.ndarray | None = None) -> np.ndarray:
    """Discrete mean curvature magnitude per vertex. Shape (V,)."""
    edges = manifold_edges(mesh, normals)
    accumulated = np.zeros(mesh.num_verts, dtype=np.float64)

    if edges.count:
        weights = edges.lengths * edges.dihedral
        np.add.at(accumulated, edges.edges[:, 0], weights)
        np.add.at(accumulated, edges.edges[:, 1], weights)

    areas = vertex_areas(mesh)
    return np.divide(accumulated, 4.0 * areas, out=np.zeros_like(accumulated), where=areas > 0)


def gaussian_curvature(mesh: MeshArrays) -> np.ndarray:
    """Discrete Gaussian curvature per vertex, by angle defect. Shape (V,)."""
    triangles, _ = fan_triangles(mesh)
    defect = np.full(mesh.num_verts, 2.0 * np.pi, dtype=np.float64)

    if triangles.shape[0]:
        positions = mesh.positions.astype(np.float64)
        for corner in range(3):
            here = positions[triangles[:, corner]]
            first = positions[triangles[:, (corner + 1) % 3]]
            second = positions[triangles[:, (corner + 2) % 3]]
            angles = _angle_between(first - here, second - here)
            np.add.at(defect, triangles[:, corner], -angles)

    areas = vertex_areas(mesh)
    return np.divide(defect, areas, out=np.zeros_like(defect), where=areas > 0)


def _angle_between(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Angle at the shared origin of two edge vectors, per row."""
    first_length = np.linalg.norm(first, axis=1)
    second_length = np.linalg.norm(second, axis=1)
    denominator = first_length * second_length

    cosines = np.divide(
        np.einsum("ij,ij->i", first, second),
        denominator,
        out=np.zeros(first.shape[0]),
        where=denominator > 0,
    )
    return np.arccos(np.clip(cosines, -1.0, 1.0))


def normalise_curvature(values: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Map curvature onto 0-1, clipped at a high percentile."""
    magnitude = np.abs(values)
    if magnitude.size == 0:
        return magnitude

    ceiling = float(np.percentile(magnitude, percentile))
    if ceiling <= 0:
        return np.zeros_like(magnitude)
    return np.clip(magnitude / ceiling, 0.0, 1.0)


def principal_directions(
    mesh: MeshArrays, normals: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Principal curvature directions and magnitudes per vertex."""
    if mesh.num_verts == 0:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, empty, np.zeros(0), np.zeros(0)

    if normals is None:
        from .mesh_graph import vertex_normals

        normals = vertex_normals(mesh)

    edges = manifold_edges(mesh).edges
    if edges.size == 0:
        fallback = _any_tangent(normals)
        zero = np.zeros(mesh.num_verts, dtype=np.float64)
        return fallback, np.cross(normals, fallback), zero, zero

    positions = mesh.positions.astype(np.float64)
    a, b = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)

    starts = np.concatenate([a, b])
    ends = np.concatenate([b, a])

    direction = positions[ends] - positions[starts]
    length = np.linalg.norm(direction, axis=1)
    usable = length > 1e-12
    starts, ends, direction, length = (
        starts[usable],
        ends[usable],
        direction[usable],
        length[usable],
    )

    normal_at = normals[starts]
    tangent = direction - normal_at * np.einsum("ij,ij->i", direction, normal_at)[:, None]
    tangent_length = np.linalg.norm(tangent, axis=1)
    alive = tangent_length > 1e-12
    starts, ends, direction, length = (starts[alive], ends[alive], direction[alive], length[alive])
    tangent = tangent[alive] / tangent_length[alive][:, None]
    normal_at = normals[starts]

    turn = normals[ends] - normal_at
    kappa = -2.0 * np.einsum("ij,ij->i", direction, turn) / (length**2)

    weight = length
    outer = tangent[:, :, None] * tangent[:, None, :]
    contribution = (weight * kappa)[:, None, None] * outer

    tensor = np.zeros((mesh.num_verts, 3, 3), dtype=np.float64)
    np.add.at(tensor, starts, contribution)
    totals = np.zeros(mesh.num_verts, dtype=np.float64)
    np.add.at(totals, starts, weight)
    tensor /= np.maximum(totals, 1e-20)[:, None, None]

    projector = np.eye(3) - normals[:, :, None] * normals[:, None, :]
    tensor = projector @ tensor @ projector
    tensor = 0.5 * (tensor + np.transpose(tensor, (0, 2, 1)))

    values, vectors = np.linalg.eigh(tensor)

    along_normal = np.abs(np.einsum("ijk,ij->ik", vectors, normals))
    normal_axis = np.argmax(along_normal, axis=1)

    keep = np.argsort(
        np.where(np.arange(3)[None, :] == normal_axis[:, None], np.inf, -np.abs(values)), axis=1
    )[:, :2]

    rows = np.arange(mesh.num_verts)
    k_major = values[rows, keep[:, 0]]
    k_minor = values[rows, keep[:, 1]]
    major = vectors[rows, :, keep[:, 0]]
    minor = vectors[rows, :, keep[:, 1]]

    flat = np.abs(k_major) < 1e-12
    if flat.any():
        major[flat] = _any_tangent(normals[flat])
        minor[flat] = np.cross(normals[flat], major[flat])

    major = _unit(major)
    minor = _unit(np.cross(normals, major))
    return major, minor, k_major, k_minor


def _unit(vectors: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(length, 1e-20)


def _any_tangent(normals: np.ndarray) -> np.ndarray:
    """Some unit vector perpendicular to each normal, chosen without bias."""
    axes = np.eye(3)
    least = np.argmin(np.abs(normals), axis=1)
    return _unit(np.cross(normals, axes[least]))
