"""Connectivity primitives: triangles, edges, areas and connected components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import MeshArrays


def fan_triangles(mesh: MeshArrays) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate every face as a fan from its first corner."""
    counts = (mesh.face_sizes - 2).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.int64)

    face_of_tri = np.repeat(np.arange(mesh.num_faces, dtype=np.int64), counts)

    fan_index = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    starts = np.repeat(mesh.face_starts.astype(np.int64), counts)

    corner_verts = mesh.corner_verts.astype(np.int64)
    triangles = np.stack(
        [
            corner_verts[starts],
            corner_verts[starts + fan_index + 1],
            corner_verts[starts + fan_index + 2],
        ],
        axis=1,
    )
    return triangles, face_of_tri


def face_areas(mesh: MeshArrays) -> np.ndarray:
    """Surface area of each face. Shape (F,)."""
    triangles, face_of_tri = fan_triangles(mesh)
    if triangles.shape[0] == 0:
        return np.zeros(mesh.num_faces, dtype=np.float64)

    positions = mesh.positions.astype(np.float64)
    a = positions[triangles[:, 0]]
    b = positions[triangles[:, 1]]
    c = positions[triangles[:, 2]]
    tri_area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return np.bincount(face_of_tri, weights=tri_area, minlength=mesh.num_faces)


def face_edges(mesh: MeshArrays) -> np.ndarray:
    """Every face-edge, as (C, 2) vertex index pairs, unsorted and with repeats."""
    num_corners = mesh.num_corners
    if num_corners == 0:
        return np.empty((0, 2), dtype=np.int64)

    corner_verts = mesh.corner_verts.astype(np.int64)
    next_corner = np.arange(1, num_corners + 1, dtype=np.int64)

    last_corners = mesh.face_starts.astype(np.int64) + mesh.face_sizes.astype(np.int64) - 1
    next_corner[last_corners] = mesh.face_starts.astype(np.int64)

    return np.stack([corner_verts, corner_verts[next_corner]], axis=1)


def edge_keys(edges: np.ndarray, num_verts: int) -> np.ndarray:
    """Order-independent int64 key per edge, so (a, b) and (b, a) collide."""
    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    return lo * np.int64(num_verts) + hi


@dataclass(frozen=True)
class EdgeTopology:
    """Unique edges and how many faces use each one."""

    edges: np.ndarray
    face_counts: np.ndarray

    @property
    def boundary(self) -> np.ndarray:
        """Edges used by exactly one face — the rim of a hole."""
        return self.face_counts == 1

    @property
    def non_manifold(self) -> np.ndarray:
        """Edges used by three or more faces."""
        return self.face_counts > 2

    @property
    def manifold(self) -> np.ndarray:
        return self.face_counts == 2


def edge_topology(mesh: MeshArrays) -> EdgeTopology:
    """Unique edges with their face-use counts."""
    raw = face_edges(mesh)
    if raw.shape[0] == 0:
        return EdgeTopology(np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64))

    keys = edge_keys(raw, mesh.num_verts)
    unique_keys, counts = np.unique(keys, return_counts=True)

    lo = unique_keys // np.int64(mesh.num_verts)
    hi = unique_keys % np.int64(mesh.num_verts)
    return EdgeTopology(edges=np.stack([lo, hi], axis=1), face_counts=counts)


def face_normals(mesh: MeshArrays) -> np.ndarray:
    """Unit normal of each face, by Newell's method. Shape (F, 3)."""
    if mesh.num_faces == 0:
        return np.zeros((0, 3), dtype=np.float64)

    edges = face_edges(mesh)
    face_of_corner = np.repeat(
        np.arange(mesh.num_faces, dtype=np.int64), mesh.face_sizes.astype(np.int64)
    )
    positions = mesh.positions.astype(np.float64)
    current = positions[edges[:, 0]]
    following = positions[edges[:, 1]]

    contribution = np.stack(
        [
            (current[:, 1] - following[:, 1]) * (current[:, 2] + following[:, 2]),
            (current[:, 2] - following[:, 2]) * (current[:, 0] + following[:, 0]),
            (current[:, 0] - following[:, 0]) * (current[:, 1] + following[:, 1]),
        ],
        axis=1,
    )

    normals = np.stack(
        [
            np.bincount(face_of_corner, weights=contribution[:, axis], minlength=mesh.num_faces)
            for axis in range(3)
        ],
        axis=1,
    )
    return _normalise(normals)


def vertex_normals(mesh: MeshArrays) -> np.ndarray:
    """Area-weighted vertex normals. Shape (V, 3)."""
    normals = face_normals(mesh)
    areas = face_areas(mesh)
    weighted = normals * areas[:, None]

    face_of_corner = np.repeat(
        np.arange(mesh.num_faces, dtype=np.int64), mesh.face_sizes.astype(np.int64)
    )
    corner_verts = mesh.corner_verts.astype(np.int64)

    accumulated = np.stack(
        [
            np.bincount(
                corner_verts, weights=weighted[face_of_corner, axis], minlength=mesh.num_verts
            )
            for axis in range(3)
        ],
        axis=1,
    )
    return _normalise(accumulated)


def _normalise(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > 0)


@dataclass(frozen=True)
class ManifoldEdges:
    """The edges with exactly two faces, and which faces those are."""

    edges: np.ndarray
    faces: np.ndarray
    lengths: np.ndarray
    dihedral: np.ndarray

    @property
    def count(self) -> int:
        return int(self.edges.shape[0])


def manifold_edges(mesh: MeshArrays, normals: np.ndarray | None = None) -> ManifoldEdges:
    """Edges shared by exactly two faces, with the angle between those faces."""
    if mesh.num_corners == 0:
        empty = np.empty((0, 2), dtype=np.int64)
        return ManifoldEdges(empty, empty, np.empty(0), np.empty(0))

    raw = face_edges(mesh)
    keys = edge_keys(raw, mesh.num_verts)
    face_of_corner = np.repeat(
        np.arange(mesh.num_faces, dtype=np.int64), mesh.face_sizes.astype(np.int64)
    )

    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_faces = face_of_corner[order]

    starts = np.flatnonzero(np.concatenate([[True], sorted_keys[1:] != sorted_keys[:-1]]))
    run_lengths = np.diff(np.concatenate([starts, [sorted_keys.shape[0]]]))
    pair_starts = starts[run_lengths == 2]

    if pair_starts.shape[0] == 0:
        empty = np.empty((0, 2), dtype=np.int64)
        return ManifoldEdges(empty, empty, np.empty(0), np.empty(0))

    pair_keys = sorted_keys[pair_starts]
    faces = np.stack([sorted_faces[pair_starts], sorted_faces[pair_starts + 1]], axis=1)

    stride = np.int64(mesh.num_verts)
    edges = np.stack([pair_keys // stride, pair_keys % stride], axis=1)

    positions = mesh.positions.astype(np.float64)
    lengths = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)

    if normals is None:
        normals = face_normals(mesh)
    cosines = np.clip(np.einsum("ij,ij->i", normals[faces[:, 0]], normals[faces[:, 1]]), -1.0, 1.0)

    return ManifoldEdges(edges=edges, faces=faces, lengths=lengths, dihedral=np.arccos(cosines))


def connected_components(num_verts: int, edges: np.ndarray) -> np.ndarray:
    """Label each vertex with the id of the component it belongs to."""
    labels = np.arange(num_verts, dtype=np.int64)
    if num_verts == 0 or edges.shape[0] == 0:
        return _compact_labels(labels)

    a = edges[:, 0].astype(np.int64)
    b = edges[:, 1].astype(np.int64)

    while True:
        previous = labels
        label_a = labels[a]
        label_b = labels[b]

        labels = labels.copy()

        np.minimum.at(labels, np.maximum(label_a, label_b), np.minimum(label_a, label_b))

        while True:
            jumped = labels[labels]
            if np.array_equal(jumped, labels):
                break
            labels = jumped

        if np.array_equal(labels, previous):
            break

    return _compact_labels(labels)


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    """Renumber arbitrary labels to a dense 0..k-1 range."""
    _, compacted = np.unique(labels, return_inverse=True)
    return compacted.astype(np.int64).reshape(labels.shape)


@dataclass(frozen=True)
class Component:
    """One connected piece of a mesh."""

    index: int
    vertex_count: int
    face_count: int
    area: float
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]

    @property
    def bbox_diagonal(self) -> float:
        return float(np.linalg.norm(np.asarray(self.bbox_max) - np.asarray(self.bbox_min)))


def components_of(mesh: MeshArrays) -> tuple[np.ndarray, list[Component]]:
    """Split a mesh into connected pieces and measure each one."""
    topology = edge_topology(mesh)
    vertex_labels = connected_components(mesh.num_verts, topology.edges)

    if mesh.num_faces == 0:
        return np.empty(0, dtype=np.int64), []

    face_labels = vertex_labels[mesh.corner_verts[mesh.face_starts].astype(np.int64)]
    areas = face_areas(mesh)

    count = int(vertex_labels.max()) + 1
    vertex_counts = np.bincount(vertex_labels, minlength=count)
    face_counts = np.bincount(face_labels, minlength=count)
    component_areas = np.bincount(face_labels, weights=areas, minlength=count)

    positions = mesh.positions.astype(np.float64)
    bbox_min = np.full((count, 3), np.inf)
    bbox_max = np.full((count, 3), -np.inf)
    np.minimum.at(bbox_min, vertex_labels, positions)
    np.maximum.at(bbox_max, vertex_labels, positions)

    components = [
        Component(
            index=int(index),
            vertex_count=int(vertex_counts[index]),
            face_count=int(face_counts[index]),
            area=float(component_areas[index]),
            bbox_min=tuple(bbox_min[index].tolist()),
            bbox_max=tuple(bbox_max[index].tolist()),
        )
        for index in np.unique(face_labels)
    ]

    components.sort(key=lambda c: c.area, reverse=True)
    return face_labels, components


def boundary_loop_count(topology: EdgeTopology) -> int:
    """How many separate boundary loops (holes and open rims) there are."""
    boundary_edges = topology.edges[topology.boundary]
    if boundary_edges.shape[0] == 0:
        return 0

    vertices = np.unique(boundary_edges)
    remap = np.full(int(vertices.max()) + 1, -1, dtype=np.int64)
    remap[vertices] = np.arange(vertices.shape[0], dtype=np.int64)
    labels = connected_components(vertices.shape[0], remap[boundary_edges])
    return int(labels.max()) + 1 if labels.size else 0


def duplicate_vertex_groups(positions: np.ndarray, tolerance: float) -> int:
    """How many groups of coincident vertices there are, within `tolerance`."""
    if positions.shape[0] == 0 or tolerance <= 0:
        return 0

    quantised = np.round(positions.astype(np.float64) / tolerance).astype(np.int64)
    _, counts = np.unique(quantised, axis=0, return_counts=True)
    return int((counts > 1).sum())


def taubin_smoothed(
    mesh: MeshArrays, passes: int = 6, shrink: float = 0.5, inflate: float = -0.53
) -> np.ndarray:
    """Positions with the scan's noise taken off, and its size left alone."""
    positions = mesh.positions.astype(np.float64).copy()
    if mesh.num_verts == 0 or passes <= 0:
        return positions

    edges = manifold_edges(mesh).edges
    if edges.size == 0:
        return positions

    starts = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
    ends = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int64)

    counts = np.zeros(mesh.num_verts, dtype=np.float64)
    np.add.at(counts, starts, 1.0)
    heard = counts > 0.0
    counts = np.maximum(counts, 1.0)[:, None]

    for _ in range(passes):
        for step in (shrink, inflate):
            summed = np.zeros_like(positions)
            np.add.at(summed, starts, positions[ends])
            towards = summed / counts - positions
            positions = np.where(heard[:, None], positions + step * towards, positions)

    return positions
