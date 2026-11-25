"""Topology-aware checks for user-marked UV cuts."""
import numpy as np
from ..analysis.mesh_graph import edge_keys, face_edges


def cut_corners(mesh, protected_edges):
    """Return matching endpoint corners on both sides of each interior marked edge."""
    marked = np.asarray(protected_edges, dtype=np.int64).reshape(-1, 2)
    if not len(marked):
        return np.empty((0, 2, 2), dtype=np.int64)
    edges = face_edges(mesh)
    keys = edge_keys(edges, mesh.num_verts)
    wanted = edge_keys(marked, mesh.num_verts)
    order = np.argsort(keys, kind="stable")
    starts = np.flatnonzero(np.r_[True, keys[order][1:] != keys[order][:-1]])
    counts = np.diff(np.r_[starts, len(order)])
    selected = np.isin(keys[order[starts]], wanted)
    first = starts[(counts == 2) & selected]
    pairs = np.column_stack((order[first], order[first + 1]))
    if np.any((counts > 2) & selected):
        from itertools import combinations
        extra = [pair for start, count in zip(starts[(counts > 2) & selected], counts[(counts > 2) & selected])
                 for pair in combinations(order[start:start + count], 2)]
        pairs = np.vstack((pairs, np.asarray(extra, dtype=np.int64)))
    following = np.arange(mesh.num_corners) + 1
    following[mesh.face_starts + mesh.face_sizes - 1] = mesh.face_starts
    corners = np.column_stack((np.arange(mesh.num_corners), following))
    reverse = edges[:, 0] > edges[:, 1]
    corners[reverse] = corners[reverse, ::-1]
    return corners[pairs]


def welded_cuts(corners, uvs):
    return np.all(np.abs(uvs[corners[:, 0]] - uvs[corners[:, 1]]) <= 1e-7, axis=(1, 2))


def cuts_preserved(corners, uvs):
    return not bool(welded_cuts(corners, uvs).any())
