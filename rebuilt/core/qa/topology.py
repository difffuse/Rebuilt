"""Is the generated mesh structurally sound?"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..analysis.mesh_graph import edge_keys, edge_topology, face_areas, face_edges
from ..types import MeshArrays


@dataclass(frozen=True)
class TopologyReport:
    """Structural findings about a mesh."""

    vertex_count: int = 0
    face_count: int = 0
    triangle_count: int = 0
    quad_count: int = 0
    ngon_count: int = 0

    non_manifold_edges: int = 0
    boundary_edges: int = 0
    flipped_faces: int = 0
    degenerate_faces: int = 0
    skinny_faces: int = 0

    max_valence: int = 0
    poles: int = 0

    @property
    def quad_ratio(self) -> float:
        total = self.triangle_count + self.quad_count + self.ngon_count
        return float(self.quad_count / total) if total else 0.0

    @property
    def has_errors(self) -> bool:
        """Whether anything here would block the asset going downstream."""
        return bool(self.non_manifold_edges or self.flipped_faces or self.degenerate_faces)

    def errors(self) -> list[str]:
        found: list[str] = []
        if self.non_manifold_edges:
            found.append(f"{self.non_manifold_edges} non-manifold edge(s)")
        if self.flipped_faces:
            found.append(f"{self.flipped_faces} face(s) wound against their neighbours")
        if self.degenerate_faces:
            found.append(f"{self.degenerate_faces} zero-area face(s)")
        return found

    def warnings(self) -> list[str]:
        found: list[str] = []
        if self.skinny_faces:
            found.append(f"{self.skinny_faces} sliver face(s)")
        if self.poles:
            found.append(f"{self.poles} pole(s) of valence 6 or more")
        return found

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quad_ratio"] = self.quad_ratio
        return payload


def flipped_face_mask(mesh: MeshArrays) -> np.ndarray:
    """Faces whose winding disagrees with a neighbour across a shared edge."""
    result = np.zeros(mesh.num_faces, dtype=bool)
    if mesh.num_corners == 0:
        return result

    directed = face_edges(mesh)
    keys = edge_keys(directed, mesh.num_verts)
    face_of_corner = np.repeat(
        np.arange(mesh.num_faces, dtype=np.int64), mesh.face_sizes.astype(np.int64)
    )

    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]

    starts = np.flatnonzero(np.concatenate([[True], sorted_keys[1:] != sorted_keys[:-1]]))
    run_lengths = np.diff(np.concatenate([starts, [sorted_keys.shape[0]]]))
    pairs = starts[run_lengths == 2]
    if pairs.shape[0] == 0:
        return result

    first = order[pairs]
    second = order[pairs + 1]

    disagrees = directed[first, 0] == directed[second, 0]
    result[face_of_corner[second[disagrees]]] = True
    return result


def skinny_face_mask(mesh: MeshArrays, aspect_limit: float = 0.05) -> np.ndarray:
    """Faces far thinner than they are long."""
    result = np.zeros(mesh.num_faces, dtype=bool)
    if mesh.num_faces == 0:
        return result

    directed = face_edges(mesh)
    positions = mesh.positions.astype(np.float64)
    lengths = np.linalg.norm(positions[directed[:, 0]] - positions[directed[:, 1]], axis=1)

    face_of_corner = np.repeat(
        np.arange(mesh.num_faces, dtype=np.int64), mesh.face_sizes.astype(np.int64)
    )
    longest = np.zeros(mesh.num_faces, dtype=np.float64)
    np.maximum.at(longest, face_of_corner, lengths)

    areas = face_areas(mesh)
    scale = longest**2
    ratio = np.divide(areas, scale, out=np.zeros_like(areas), where=scale > 0)
    return (ratio < aspect_limit) & (scale > 0)


def vertex_valence(mesh: MeshArrays) -> np.ndarray:
    """How many edges meet at each vertex. Shape (V,)."""
    topology = edge_topology(mesh)
    if topology.edges.shape[0] == 0:
        return np.zeros(mesh.num_verts, dtype=np.int64)
    return np.bincount(topology.edges.ravel(), minlength=mesh.num_verts)


def inspect_topology(
    mesh: MeshArrays,
    degenerate_area_ratio: float = 1e-8,
    aspect_limit: float = 0.05,
    pole_valence: int = 6,
) -> TopologyReport:
    """Run every structural check over a mesh."""
    if mesh.num_verts == 0:
        return TopologyReport()

    topology = edge_topology(mesh)
    areas = face_areas(mesh)
    valence = vertex_valence(mesh)

    threshold = float(areas.mean()) * degenerate_area_ratio if areas.size else 0.0

    return TopologyReport(
        vertex_count=mesh.num_verts,
        face_count=mesh.num_faces,
        triangle_count=int((mesh.face_sizes == 3).sum()),
        quad_count=int((mesh.face_sizes == 4).sum()),
        ngon_count=int((mesh.face_sizes > 4).sum()),
        non_manifold_edges=int(topology.non_manifold.sum()),
        boundary_edges=int(topology.boundary.sum()),
        flipped_faces=int(flipped_face_mask(mesh).sum()),
        degenerate_faces=int((areas <= threshold).sum()) if areas.size else 0,
        skinny_faces=int(skinny_face_mask(mesh, aspect_limit).sum()),
        max_valence=int(valence.max()) if valence.size else 0,
        poles=int((valence >= pole_valence).sum()),
    )
