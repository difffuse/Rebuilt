"""Where the quad corners want to sit, before any of them exist."""

from __future__ import annotations

import numpy as np

from ..types import MeshArrays
from .crossfield import cross_pair
from .mesh_graph import manifold_edges, vertex_normals


DEFAULT_PASSES = 30


def position_field(
    mesh: MeshArrays,
    field: np.ndarray,
    edge_length: np.ndarray,
    normals: np.ndarray | None = None,
    passes: int = DEFAULT_PASSES,
    start: np.ndarray | None = None,
) -> np.ndarray:
    """A point per vertex, settled onto a lattice its neighbours share."""
    if mesh.num_verts == 0:
        return np.zeros((0, 3), dtype=np.float64)

    if normals is None:
        normals = (
            np.tile(np.array([0.0, 0.0, 1.0]), (mesh.num_verts, 1))
            if mesh.num_faces == 0
            else vertex_normals(mesh)
        )

    positions = mesh.positions.astype(np.float64)
    lattice = positions.copy() if start is None else start.astype(np.float64).copy()

    edges = manifold_edges(mesh).edges
    if edges.size == 0 or passes <= 0:
        return lattice

    first, second = cross_pair(field, normals)
    spacing = np.maximum(edge_length.astype(np.float64), 1e-9)

    starts = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
    ends = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int64)

    counts = np.zeros(mesh.num_verts, dtype=np.float64)
    np.add.at(counts, starts, 1.0)
    heard = counts > 0.0
    counts = np.maximum(counts, 1.0)[:, None]

    for _ in range(passes):
        carried = _nearest_lattice_point(
            lattice[ends],
            lattice[starts],
            first[starts],
            second[starts],
            spacing[starts],
        )

        summed = np.zeros_like(lattice)
        np.add.at(summed, starts, carried)
        averaged = summed / counts

        moved = np.where(heard[:, None], averaged, lattice)
        lattice = _to_surface(moved, positions, normals)

    return lattice


def lattice_disagreement(
    mesh: MeshArrays,
    lattice: np.ndarray,
    field: np.ndarray,
    edge_length: np.ndarray,
    normals: np.ndarray | None = None,
) -> np.ndarray:
    """How far each edge's two lattices are from lining up, in cells."""
    if normals is None:
        normals = vertex_normals(mesh)

    edges = manifold_edges(mesh).edges
    if edges.size == 0:
        return np.zeros(0, dtype=np.float64)

    a, b = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    first, second = cross_pair(field, normals)
    spacing = np.maximum(edge_length.astype(np.float64), 1e-9)

    carried = _nearest_lattice_point(lattice[b], lattice[a], first[a], second[a], spacing[a])
    return np.linalg.norm(carried - lattice[a], axis=1) / spacing[a]


TEAR_THRESHOLD = 0.25


def repair_lattice(
    mesh: MeshArrays,
    sites: np.ndarray,
    field: np.ndarray,
    edge_length: np.ndarray,
    normals: np.ndarray,
    rounds: int = 2,
) -> np.ndarray:
    """Re-phase the torn regions from the settled majority around them."""
    if mesh.num_verts == 0 or rounds <= 0:
        return sites

    edges = manifold_edges(mesh).edges
    if edges.size == 0:
        return sites

    positions = mesh.positions.astype(np.float64)
    first, second = cross_pair(field, normals)
    spacing = np.maximum(edge_length.astype(np.float64), 1e-9)

    starts = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
    ends = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int64)
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    offsets = np.searchsorted(starts, np.arange(mesh.num_verts + 1))

    repaired = sites.astype(np.float64).copy()
    for _ in range(rounds):
        disagreement = lattice_disagreement(mesh, repaired, field, edge_length, normals)
        torn_edge = disagreement > TEAR_THRESHOLD
        incident = np.zeros(mesh.num_verts)
        broken = np.zeros(mesh.num_verts)
        np.add.at(incident, edges[:, 0], 1.0)
        np.add.at(incident, edges[:, 1], 1.0)
        np.add.at(broken, edges[:, 0], torn_edge.astype(np.float64))
        np.add.at(broken, edges[:, 1], torn_edge.astype(np.float64))
        torn = broken > 0.5 * np.maximum(incident, 1.0)
        if not torn.any():
            break

        seen = ~torn
        queue = [
            int(vertex)
            for vertex in np.flatnonzero(~torn)
            if any(
                torn[int(ends[slot])]
                for slot in range(offsets[int(vertex)], offsets[int(vertex) + 1])
            )
        ]
        while queue:
            following: list[int] = []
            for parent in queue:
                base = repaired[parent]
                axis_one, axis_two = first[parent], second[parent]
                step = spacing[parent]
                for slot in range(offsets[parent], offsets[parent + 1]):
                    child = int(ends[slot])
                    if seen[child]:
                        continue
                    seen[child] = True
                    towards = positions[child] - base
                    steps_one = round(float(towards @ axis_one) / step)
                    steps_two = round(float(towards @ axis_two) / step)
                    repaired[child] = base + step * (steps_one * axis_one + steps_two * axis_two)
                    following.append(child)
            queue = following

        repaired = position_field(mesh, field, edge_length, normals, passes=10, start=repaired)

    return repaired


def _nearest_lattice_point(
    point: np.ndarray,
    towards: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    """Slide `point` by whole cells until it sits as close to `towards` as it can."""
    offset = point - towards
    along_first = np.einsum("ij,ij->i", offset, first) / spacing
    along_second = np.einsum("ij,ij->i", offset, second) / spacing

    steps_first = np.round(along_first)
    steps_second = np.round(along_second)

    return point - (
        first * (steps_first * spacing)[:, None] + second * (steps_second * spacing)[:, None]
    )


def _to_surface(points: np.ndarray, anchors: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Flatten any drift along the normal, keeping the tangential movement."""
    offset = points - anchors
    along_normal = np.einsum("ij,ij->i", offset, normals)[:, None]
    return points - normals * along_normal


def lattice_sites(
    mesh: MeshArrays,
    field: np.ndarray,
    edge_length: np.ndarray,
    normals: np.ndarray | None = None,
) -> np.ndarray:
    """Snap every vertex to a lattice site, with one phase shared by all of them."""
    if mesh.num_verts == 0:
        return np.zeros((0, 3), dtype=np.float64)

    if normals is None:
        normals = (
            np.tile(np.array([0.0, 0.0, 1.0]), (mesh.num_verts, 1))
            if mesh.num_faces == 0
            else vertex_normals(mesh)
        )

    positions = mesh.positions.astype(np.float64)
    edges = manifold_edges(mesh).edges
    if edges.size == 0:
        return positions.copy()

    first, second = cross_pair(field, normals)
    spacing = np.maximum(edge_length.astype(np.float64), 1e-9)

    starts = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
    ends = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int64)
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    offsets = np.searchsorted(starts, np.arange(mesh.num_verts + 1))

    sites = positions.copy()
    seen = np.zeros(mesh.num_verts, dtype=bool)

    for seed in range(mesh.num_verts):
        if seen[seed]:
            continue
        seen[seed] = True
        queue = [seed]
        while queue:
            nxt: list[int] = []
            for parent in queue:
                base = sites[parent]
                axis_one, axis_two = first[parent], second[parent]
                step = spacing[parent]
                for slot in range(offsets[parent], offsets[parent + 1]):
                    child = int(ends[slot])
                    if seen[child]:
                        continue
                    seen[child] = True
                    offset = positions[child] - base
                    steps_one = round(float(offset @ axis_one) / step)
                    steps_two = round(float(offset @ axis_two) / step)
                    sites[child] = base + step * (steps_one * axis_one + steps_two * axis_two)
                    nxt.append(child)
            queue = nxt

    return sites
