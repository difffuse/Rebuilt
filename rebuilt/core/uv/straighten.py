"""Straightening seams the way an artist lays them."""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..analysis.mesh_graph import ManifoldEdges, face_normals, manifold_edges
from ..types import MeshArrays


TURN_PENALTY = 2.0


CREST_PENALTY = 1.0


BAND_HOPS = 4


SLIVER_FRACTION = 0.002


TURN_ANGLE_DEG = 30.0


@dataclass
class Straightening:
    labels: np.ndarray
    seams: np.ndarray

    turns_before: int
    turns_after: int

    excess_before: float
    excess_after: float
    chains: int


def straighten_seams(
    mesh: MeshArrays,
    labels: np.ndarray,
    seams: np.ndarray,
    normals: np.ndarray | None = None,
    band_hops: int = BAND_HOPS,
) -> Straightening:
    """Re-route every seam chain along the loops. See docs/development/reference/core.md."""
    edges = manifold_edges(mesh, normals)
    labels = np.asarray(labels, dtype=np.int64)
    seams = np.asarray(seams, dtype=bool)
    if edges.count == 0 or not seams.any():
        return Straightening(labels, seams, 0, 0, 1.0, 1.0, 0)

    positions = mesh.positions.astype(np.float64)
    corners = _corners(mesh, edges, seams)
    chains = _chains(edges, seams, corners)
    if not chains:
        return Straightening(labels, seams, 0, 0, 1.0, 1.0, 0)

    edge_cost = _edge_costs(mesh, edges, normals)
    adjacency = _adjacency(edges, mesh.num_verts)
    turns_before = turns_after = 0
    excess_before = excess_after = 0.0

    current = seams.copy()
    regions = _region_count(mesh, edges, current)
    for chain in chains:
        vertices = _chain_vertices(edges, chain)
        band = _band(adjacency, vertices, band_hops)
        rerouted = _cheapest_path(
            edges, adjacency, positions, edge_cost, band, vertices[0], vertices[-1]
        )
        old_stats = seam_turns(positions, edges, chain, vertices[0], vertices[-1])
        turns_before += old_stats[0]
        excess_before += old_stats[1]
        accepted = chain
        if rerouted and set(rerouted) != set(chain):
            trial = current.copy()
            trial[chain] = False
            trial[rerouted] = True
            if _region_count(mesh, edges, trial) == regions:
                current = trial
                accepted = rerouted
        new_stats = seam_turns(positions, edges, accepted, vertices[0], vertices[-1])
        turns_after += new_stats[0]
        excess_after += new_stats[1]
    new_seams = current

    new_labels = _reassign(mesh, edges, labels, new_seams)
    count = max(len(chains), 1)
    return Straightening(
        new_labels,
        new_seams,
        turns_before,
        turns_after,
        excess_before / count,
        excess_after / count,
        len(chains),
    )


def _corners(mesh: MeshArrays, edges: ManifoldEdges, seams: np.ndarray) -> set[int]:
    """Vertices where the seam network branches or ends, plus every vertex on the mesh's own open boundary, which a seam may end against."""
    from ..analysis.mesh_graph import edge_topology

    degree: dict[int, int] = defaultdict(int)
    for a, b in edges.edges[seams]:
        degree[int(a)] += 1
        degree[int(b)] += 1
    corners = {v for v, d in degree.items() if d != 2}
    topology = edge_topology(mesh)
    for v in np.unique(topology.edges[topology.boundary]):
        if int(v) in degree:
            corners.add(int(v))
    return corners


def _chains(edges: ManifoldEdges, seams: np.ndarray, corners: set[int]) -> list[list[int]]:
    """The seam network cut at its corners: lists of edge indices, in order. A closed loop with no corner gets one made at its first vertex."""
    incident: dict[int, list[int]] = defaultdict(list)
    for index in np.flatnonzero(seams):
        a, b = edges.edges[index]
        incident[int(a)].append(int(index))
        incident[int(b)].append(int(index))
    used = np.zeros(edges.count, dtype=bool)
    chains: list[list[int]] = []

    def walk(start_vertex: int, first_edge: int) -> list[int]:
        chain = [first_edge]
        used[first_edge] = True
        vertex = int(
            edges.edges[first_edge][1]
            if edges.edges[first_edge][0] == start_vertex
            else edges.edges[first_edge][0]
        )
        while vertex not in corners:
            following = [e for e in incident[vertex] if not used[e]]
            if not following:
                break
            nxt = following[0]
            used[nxt] = True
            chain.append(nxt)
            a, b = edges.edges[nxt]
            vertex = int(b if a == vertex else a)
        return chain

    for corner in sorted(corners):
        for edge in incident.get(corner, []):
            if not used[edge]:
                chains.append(walk(corner, edge))

    for index in np.flatnonzero(seams):
        if not used[index]:
            start = int(edges.edges[index][0])
            corners.add(start)
            chains.append(walk(start, int(index)))
    return chains


def _chain_vertices(edges: ManifoldEdges, chain: list[int]) -> list[int]:
    """The chain's vertices in walking order, first to last."""
    if len(chain) == 1:
        a, b = edges.edges[chain[0]]
        return [int(a), int(b)]
    first, second = edges.edges[chain[0]], edges.edges[chain[1]]
    start = int(first[0] if first[0] not in second else first[1])
    vertices = [start]
    vertex = start
    for index in chain:
        a, b = edges.edges[index]
        vertex = int(b if a == vertex else a)
        vertices.append(vertex)
    return vertices


def _adjacency(edges: ManifoldEdges, num_verts: int) -> list[list[tuple[int, int]]]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_verts)]
    for index, (a, b) in enumerate(edges.edges):
        adjacency[int(a)].append((int(b), index))
        adjacency[int(b)].append((int(a), index))
    return adjacency


def _band(adjacency, vertices: list[int], hops: int) -> set[int]:
    """Every vertex within `hops` edges of the chain."""
    band = set(vertices)
    frontier = set(vertices)
    for _ in range(hops):
        following = set()
        for v in frontier:
            for other, _ in adjacency[v]:
                if other not in band:
                    band.add(other)
                    following.add(other)
        frontier = following
    return band


def _edge_costs(mesh: MeshArrays, edges: ManifoldEdges, normals: np.ndarray | None) -> np.ndarray:
    """Length times the crest penalty: a convex fold costs more, a concave one nothing extra."""
    if normals is None:
        normals = face_normals(mesh)
    positions = mesh.positions.astype(np.float64)
    centres = np.zeros((mesh.num_faces, 3))
    for axis in range(3):
        centres[:, axis] = np.bincount(
            np.repeat(np.arange(mesh.num_faces), mesh.face_sizes.astype(np.int64)),
            weights=positions[mesh.corner_verts.astype(np.int64), axis],
            minlength=mesh.num_faces,
        ) / np.maximum(mesh.face_sizes, 1)
    f1, f2 = edges.faces[:, 0], edges.faces[:, 1]

    convex = np.einsum("ij,ij->i", centres[f2] - centres[f1], normals[f1]) < 0.0
    fold = edges.dihedral / (np.pi / 2.0)
    penalty = np.where(convex, 1.0 + CREST_PENALTY * np.clip(fold, 0.0, 2.0), 1.0)
    return edges.lengths * penalty


def _cheapest_path(
    edges, adjacency, positions, edge_cost, band, source: int, target: int
) -> list[int]:
    """Dijkstra over directed edges inside the band, with a turn penalty."""
    if source == target:
        return []
    best: dict[tuple[int, int], float] = {}
    via: dict[tuple[int, int], tuple[int, int] | None] = {}
    heap: list[tuple[float, int, int]] = []
    for other, index in adjacency[source]:
        if other in band:
            state = (other, index)
            best[state] = float(edge_cost[index])
            via[state] = None
            heapq.heappush(heap, (best[state], other, index))
    found = None
    while heap:
        cost, vertex, index = heapq.heappop(heap)
        state = (vertex, index)
        if cost > best.get(state, np.inf):
            continue
        if vertex == target:
            found = state
            break
        a, b = edges.edges[index]
        came_from = int(a if b == vertex else b)
        incoming = positions[vertex] - positions[came_from]
        incoming /= max(float(np.linalg.norm(incoming)), 1e-12)
        for other, next_index in adjacency[vertex]:
            if other not in band or other == came_from:
                continue
            outgoing = positions[other] - positions[vertex]
            outgoing /= max(float(np.linalg.norm(outgoing)), 1e-12)
            turn = 0.5 * (1.0 - float(incoming @ outgoing))
            step = float(edge_cost[next_index]) * (1.0 + TURN_PENALTY * turn * 2.0)
            candidate = cost + step
            next_state = (other, next_index)
            if candidate < best.get(next_state, np.inf):
                best[next_state] = candidate
                via[next_state] = state
                heapq.heappush(heap, (candidate, other, next_index))
    if found is None:
        return []
    path = []
    state = found
    while state is not None:
        path.append(state[1])
        state = via[state]
    return path[::-1]


def _region_count(mesh: MeshArrays, edges: ManifoldEdges, seams: np.ndarray) -> int:
    """How many face regions the seams enclose, slivers aside."""
    from ..analysis.mesh_graph import connected_components, face_areas

    components = connected_components(mesh.num_faces, edges.faces[~seams])
    areas = np.bincount(components, weights=face_areas(mesh))
    return int((areas >= SLIVER_FRACTION * areas.sum()).sum())


def _reassign(
    mesh: MeshArrays, edges: ManifoldEdges, labels: np.ndarray, seams: np.ndarray
) -> np.ndarray:
    """Faces regrouped by the new seams: one chart per connected component across the edges that are not seams."""
    from ..analysis.mesh_graph import connected_components
    from .charts import _Graph, _merge_small

    joins = edges.faces[~seams]
    components = connected_components(mesh.num_faces, joins).astype(np.int64)
    graph = _Graph.build(mesh)
    return _merge_small(graph, components, SLIVER_FRACTION)


def seam_turns(
    positions, edges: ManifoldEdges, chain: list[int], start: int, end: int
) -> tuple[int, float]:
    """(turns, excess): direction changes past `TURN_ANGLE_DEG` along the chain, and its length over the straight-line distance between its ends (1.0 is as short as it could be)."""
    if not chain:
        return 0, 1.0
    vertices = _chain_vertices(edges, chain)
    if vertices[0] != start:
        vertices = vertices[::-1]
    points = positions[vertices]
    steps = np.diff(points, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    keep = lengths > 1e-12
    steps, lengths = steps[keep], lengths[keep]
    turns = 0
    if len(steps) > 1:
        cosines = np.einsum("ij,ij->i", steps[:-1], steps[1:]) / (lengths[:-1] * lengths[1:])
        turns = int((np.arccos(np.clip(cosines, -1.0, 1.0)) > np.radians(TURN_ANGLE_DEG)).sum())
    straight = float(np.linalg.norm(positions[end] - positions[start]))
    excess = float(lengths.sum()) / straight if straight > 1e-12 else 1.0
    return turns, excess
