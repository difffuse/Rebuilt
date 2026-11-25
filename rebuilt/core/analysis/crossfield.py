"""A four-fold direction field over the surface, aligned to how it bends."""

from __future__ import annotations

import numpy as np

from ..types import MeshArrays
from .curvature import principal_directions
from .mesh_graph import manifold_edges, vertex_normals


DEFAULT_PASSES = 40


DEFAULT_ALIGNMENT = 0.35


def cross_field(
    mesh: MeshArrays,
    normals: np.ndarray | None = None,
    passes: int = DEFAULT_PASSES,
    alignment: float = DEFAULT_ALIGNMENT,
    start: np.ndarray | None = None,
    anchors: np.ndarray | None = None,
    anchor_strength: float = 0.9,
) -> np.ndarray:
    """One representative direction per vertex, smoothed and curvature-aligned."""
    if mesh.num_verts == 0:
        return np.zeros((0, 3), dtype=np.float64)

    if normals is None:
        if mesh.num_faces == 0:
            normals = np.tile(np.array([0.0, 0.0, 1.0]), (mesh.num_verts, 1))
        else:
            normals = vertex_normals(mesh)

    major, _, k_major, k_minor = principal_directions(mesh, normals)
    field = _unit(_project(major, normals))
    if start is not None:
        seeded = _project(start.astype(np.float64), normals)
        lengths = np.linalg.norm(seeded, axis=1)

        usable = lengths > 1e-8
        field = np.where(usable[:, None], _unit(seeded), field)

    pinned = None
    pin_direction = None
    if anchors is not None:
        projected_pins = _project(anchors.astype(np.float64), normals)
        pin_lengths = np.linalg.norm(projected_pins, axis=1)
        pinned = pin_lengths > 1e-8
        if pinned.any():
            pin_direction = _unit(projected_pins)
            field = np.where(pinned[:, None], pin_direction, field)
        else:
            pinned = None

    edges = manifold_edges(mesh).edges
    if edges.size == 0 or passes <= 0:
        return field

    anisotropy = np.abs(k_major - k_minor)
    peak = float(anisotropy.max()) if anisotropy.size else 0.0
    confidence = anisotropy / peak if peak > 0.0 else np.zeros_like(anisotropy)
    hold = (alignment * confidence)[:, None]

    starts = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
    ends = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int64)

    counts = np.zeros(mesh.num_verts, dtype=np.float64)
    np.add.at(counts, starts, 1.0)

    heard = counts > 0.0
    counts = np.maximum(counts, 1.0)[:, None]

    for _ in range(passes):
        neighbour = _nearest_representative(
            _project(field[ends], normals[starts]), field[starts], normals[starts]
        )

        summed = np.zeros_like(field)
        np.add.at(summed, starts, neighbour)
        averaged = summed / counts

        anchored = _nearest_representative(field, averaged, normals)
        updated = averaged * (1.0 - hold) + anchored * hold
        updated = np.where(heard[:, None], updated, field)

        if pinned is not None:
            held = _nearest_representative(pin_direction, updated, normals)
            strong = anchor_strength * pinned[:, None]
            updated = updated * (1.0 - strong) + held * strong

        field = _unit(_project(updated, normals))

    return field


def cross_pair(field: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two perpendicular directions a cross stands for."""
    return field, _unit(np.cross(normals, field))


def field_disagreement(
    mesh: MeshArrays, field: np.ndarray, normals: np.ndarray | None = None
) -> np.ndarray:
    """Angle between each edge's two crosses, in radians, after matching them."""
    if normals is None:
        normals = vertex_normals(mesh)

    edges = manifold_edges(mesh).edges
    if edges.size == 0:
        return np.zeros(0, dtype=np.float64)

    a, b = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    matched = _nearest_representative(_project(field[b], normals[a]), field[a], normals[a])
    dots = np.clip(np.einsum("ij,ij->i", _unit(matched), field[a]), -1.0, 1.0)
    return np.arccos(np.abs(dots))


def _nearest_representative(
    candidate: np.ndarray, towards: np.ndarray, normals: np.ndarray
) -> np.ndarray:
    """Whichever of a cross's four directions sits closest to `towards`."""
    turned = np.cross(normals, candidate)
    direct = np.einsum("ij,ij->i", candidate, towards)
    rotated = np.einsum("ij,ij->i", turned, towards)

    use_turned = np.abs(rotated) > np.abs(direct)
    chosen = np.where(use_turned[:, None], turned, candidate)
    sign = np.where(use_turned, np.sign(rotated), np.sign(direct))
    sign = np.where(sign == 0.0, 1.0, sign)
    return chosen * sign[:, None]


def _project(vectors: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Flatten into the tangent plane, keeping whatever length survives."""
    return vectors - normals * np.einsum("ij,ij->i", vectors, normals)[:, None]


def _unit(vectors: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(length, 1e-20)
