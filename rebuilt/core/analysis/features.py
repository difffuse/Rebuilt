"""Sharp edges, boundaries, and how close each vertex is to one."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import MeshArrays
from .mesh_graph import ManifoldEdges, edge_topology, manifold_edges


@dataclass(frozen=True)
class Features:
    """Where the mesh has structure worth preserving."""

    sharp_edges: np.ndarray
    sharp_angles: np.ndarray
    boundary_edges: np.ndarray
    feature_vertices: np.ndarray

    @property
    def sharp_count(self) -> int:
        return int(self.sharp_edges.shape[0])

    @property
    def boundary_count(self) -> int:
        return int(self.boundary_edges.shape[0])


def find_features(
    mesh: MeshArrays,
    hard_edge_angle_deg: float = 35.0,
    normals: np.ndarray | None = None,
    edges: ManifoldEdges | None = None,
) -> Features:
    """Sharp creases and open boundaries."""
    if edges is None:
        edges = manifold_edges(mesh, normals)

    threshold = np.radians(hard_edge_angle_deg)
    sharp = edges.dihedral > threshold

    topology = edge_topology(mesh)
    boundary = topology.edges[topology.boundary]

    feature_vertices = np.unique(
        np.concatenate([edges.edges[sharp].ravel(), boundary.ravel()])
        if sharp.any() or boundary.shape[0]
        else np.empty(0, dtype=np.int64)
    )

    return Features(
        sharp_edges=edges.edges[sharp],
        sharp_angles=edges.dihedral[sharp],
        boundary_edges=boundary,
        feature_vertices=feature_vertices,
    )


def hop_distance(
    num_verts: int, edges: np.ndarray, sources: np.ndarray, max_hops: int = 8
) -> np.ndarray:
    """How many edges away each vertex is from the nearest source."""
    distance = np.full(num_verts, max_hops, dtype=np.int64)
    if sources.size == 0 or edges.shape[0] == 0:
        return distance

    distance[sources] = 0
    a = edges[:, 0].astype(np.int64)
    b = edges[:, 1].astype(np.int64)

    for _ in range(max_hops):
        previous = distance.copy()
        np.minimum.at(distance, a, previous[b] + 1)
        np.minimum.at(distance, b, previous[a] + 1)
        np.clip(distance, 0, max_hops, out=distance)
        if np.array_equal(distance, previous):
            break

    return distance


def feature_proximity(
    mesh: MeshArrays, features: Features, falloff_hops: float = 3.0
) -> np.ndarray:
    """How near each vertex is to a feature, as a 0-1 weight. Shape (V,)."""
    topology = edge_topology(mesh)
    hops = hop_distance(mesh.num_verts, topology.edges, features.feature_vertices)
    return np.exp(-hops.astype(np.float64) / max(falloff_hops, 1e-6))
