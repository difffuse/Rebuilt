"""The importance and density fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import MeshArrays
from .curvature import vertex_areas
from .mesh_graph import edge_topology, manifold_edges


@dataclass(frozen=True)
class ImportanceWeights:
    """How much each signal contributes to the importance field."""

    curvature: float = 0.6
    features: float = 0.4

    def normalised(self) -> ImportanceWeights:
        total = self.curvature + self.features
        if total <= 0:
            return ImportanceWeights(1.0, 0.0)
        return ImportanceWeights(self.curvature / total, self.features / total)


def importance_field(
    curvature: np.ndarray,
    feature_proximity: np.ndarray,
    weights: ImportanceWeights | None = None,
) -> np.ndarray:
    """Blend the signals into a 0-1 weight per vertex."""
    weights = (weights or ImportanceWeights()).normalised()
    blended = weights.curvature * curvature + weights.features * feature_proximity
    return np.clip(blended, 0.0, 1.0)


def density_field(
    mesh: MeshArrays,
    importance: np.ndarray,
    target_faces: int,
    contrast: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn importance into a face budget spread over the surface."""
    if mesh.num_verts == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty

    weight = 1.0 + max(contrast, 0.0) * np.clip(importance, 0.0, 1.0)
    areas = vertex_areas(mesh)

    weighted_area = float((areas * weight).sum())
    if weighted_area <= 0:
        uniform = np.full(mesh.num_verts, float(target_faces), dtype=np.float64)
        return uniform, np.zeros(mesh.num_verts, dtype=np.float64)

    density = weight * (target_faces / weighted_area)

    edge_length = np.divide(1.0, np.sqrt(density), out=np.zeros_like(density), where=density > 0)
    return density, edge_length


def predicted_face_count(mesh: MeshArrays, density: np.ndarray) -> float:
    """Integrate a density field over the surface."""
    return float((vertex_areas(mesh) * density).sum())


def curvature_sizing(
    mesh: MeshArrays,
    k_major: np.ndarray,
    k_minor: np.ndarray,
    target_faces: int,
    contrast: float = 3.0,
    percentile: float = 95.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A face budget spread by how sharply the surface actually bends."""
    if mesh.num_verts == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty

    sharpest = np.maximum(np.abs(k_major), np.abs(k_minor))
    ceiling = float(np.percentile(sharpest, percentile)) if sharpest.size else 0.0
    if ceiling <= 0.0:
        normalised = np.zeros_like(sharpest)
    else:
        normalised = np.clip(sharpest / ceiling, 0.0, 1.0)

    return density_field(mesh, normalised, target_faces, contrast)


def graded_edge_length(
    mesh: MeshArrays,
    edge_length: np.ndarray,
    limit: float = 0.1,
    keep_budget: bool = True,
) -> np.ndarray:
    """The sizing field with its rate of change capped along the surface."""
    graded = edge_length.astype(np.float64).copy()
    if mesh.num_verts == 0 or graded.size == 0:
        return graded

    manifold = manifold_edges(mesh)
    if manifold.count == 0:
        return graded

    a = manifold.edges[:, 0].astype(np.int64)
    b = manifold.edges[:, 1].astype(np.int64)
    allowance = max(limit, 0.0) * manifold.lengths

    for _ in range(128):
        before = graded.copy()
        np.minimum.at(graded, b, graded[a] + allowance)
        np.minimum.at(graded, a, graded[b] + allowance)
        if np.allclose(graded, before, rtol=1e-9, atol=0.0):
            break

    if keep_budget:
        areas = vertex_areas(mesh)
        spent = float((areas / np.maximum(graded, 1e-12) ** 2).sum())
        granted = float((areas / np.maximum(edge_length.astype(np.float64), 1e-12) ** 2).sum())
        if spent > 0.0 and granted > 0.0:
            graded *= np.sqrt(spent / granted)

    return graded


def sizing_error(mesh: MeshArrays, target_edge_length: np.ndarray) -> np.ndarray:
    """How far each edge is from the length the sizing field asked for."""
    edges = edge_topology(mesh).edges
    if edges.size == 0:
        return np.zeros(0, dtype=np.float64)

    a, b = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    lengths = np.linalg.norm(mesh.positions[a] - mesh.positions[b], axis=1)
    wanted = 0.5 * (target_edge_length[a] + target_edge_length[b])
    return np.divide(lengths, wanted, out=np.ones_like(lengths), where=wanted > 1e-12)
