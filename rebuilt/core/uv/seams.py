"""Deciding where to cut."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analysis.features import Features
from ..analysis.mesh_graph import ManifoldEdges
from ..types import MeshArrays


@dataclass(frozen=True)
class SeamWeights:
    """How the seam score is composed."""

    crease: float = 1.0

    importance: float = 0.8

    material: float = 0.6


def seam_scores(
    mesh: MeshArrays,
    edges: ManifoldEdges,
    importance: np.ndarray | None = None,
    weights: SeamWeights | None = None,
) -> np.ndarray:
    """Score every manifold edge as a seam candidate. Shape (M,)."""
    weights = weights or SeamWeights()
    if edges.count == 0:
        return np.zeros(0, dtype=np.float64)

    crease = np.clip(edges.dihedral / (np.pi / 2.0), 0.0, 1.0)
    score = weights.crease * crease

    if importance is not None and importance.size == mesh.num_verts:
        edge_importance = np.maximum(importance[edges.edges[:, 0]], importance[edges.edges[:, 1]])
        score -= weights.importance * edge_importance

    return score


def select_seams(
    mesh: MeshArrays,
    edges: ManifoldEdges,
    angle_deg: float,
    importance: np.ndarray | None = None,
    weights: SeamWeights | None = None,
    importance_bias: float = 0.5,
) -> np.ndarray:
    """Which manifold edges to mark as seams. Boolean mask, shape (M,)."""
    if edges.count == 0:
        return np.zeros(0, dtype=bool)

    required = np.full(edges.count, np.radians(angle_deg))

    if importance is not None and importance.size == mesh.num_verts:
        edge_importance = np.maximum(importance[edges.edges[:, 0]], importance[edges.edges[:, 1]])
        required = required * (1.0 + importance_bias * edge_importance)

    return edges.dihedral >= required


def boundary_seam_mask(features: Features) -> np.ndarray:
    """Open boundaries are seams whether anyone marks them or not."""
    return features.boundary_edges


def seam_length(mesh: MeshArrays, edges: ManifoldEdges, mask: np.ndarray) -> float:
    """Total length of the selected seams."""
    if edges.count == 0 or not mask.any():
        return 0.0
    return float(edges.lengths[mask].sum())


def island_count(mesh, edges, seams: np.ndarray) -> int:
    """How many separate pieces a seam set cuts the surface into."""
    from ..analysis.mesh_graph import connected_components

    if mesh.num_faces == 0:
        return 0

    pairs = edges.faces[~seams]
    pairs = pairs[(pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)]
    if pairs.size == 0:
        return int(mesh.num_faces)

    labels = connected_components(int(mesh.num_faces), pairs.astype(np.int64))
    return int(np.unique(labels).size)
