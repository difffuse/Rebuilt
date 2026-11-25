"""Compare repeated sharp turns on generated rims with source boundary segments."""

from __future__ import annotations

import numpy as np

from ..analysis.mesh_graph import edge_topology, face_areas


def _segment_distances(points, starts, ends):
    distances = np.full(len(points), np.inf)
    for first in range(0, len(points), 64):
        part = points[first : first + 64]
        best = np.full(len(part), np.inf)
        for offset in range(0, len(starts), 2048):
            a, b = starts[offset : offset + 2048], ends[offset : offset + 2048]
            span = b - a
            length2 = np.einsum("ij,ij->i", span, span)
            delta = part[:, None] - a
            t = np.clip(np.einsum("pij,ij->pi", delta, span) / np.maximum(length2, 1e-300), 0, 1)
            residual = delta - t[:, :, None] * span
            best = np.minimum(best, np.einsum("pij,pij->pi", residual, residual).min(axis=1))
        distances[first : first + len(part)] = np.sqrt(best)
    return distances


def rim_quality(mesh, source, target: int) -> dict:
    report = dict(needs_recovery=False, severity=0.0, unsupported_spikes=0, affected_rims=0)
    topology = edge_topology(mesh)
    edges = topology.edges[topology.boundary]
    neighbours = {}
    for a, b in edges:
        neighbours.setdefault(int(a), []).append(int(b))
        neighbours.setdefault(int(b), []).append(int(a))
    pending = set(neighbours)
    candidates = []
    positions = mesh.positions.astype(np.float64)
    while pending:
        seed = pending.pop()
        component, frontier = [seed], [seed]
        while frontier:
            for vertex in neighbours[frontier.pop()]:
                if vertex in pending:
                    pending.remove(vertex)
                    component.append(vertex)
                    frontier.append(vertex)
        if len(component) <= 8:
            continue
        sharp = []
        for vertex in component:
            adjacent = neighbours[vertex]
            if len(adjacent) != 2:
                continue
            a, b = positions[adjacent] - positions[vertex]
            length = np.linalg.norm(a) * np.linalg.norm(b)
            if length > 0 and np.dot(a, b) > 0.5 * length:
                sharp.append(vertex)
        if len(sharp) >= 6:
            candidates.append((component, sharp))
    if not candidates:
        return report
    source_topology = edge_topology(source)
    boundary = source_topology.edges[source_topology.boundary]
    if not len(boundary):
        return report
    cell = float(np.sqrt(face_areas(source).sum() / max(target, 8)))
    if cell <= 0:
        return report
    source_positions = source.positions.astype(np.float64)
    starts, ends = source_positions[boundary[:, 0]], source_positions[boundary[:, 1]]
    for component, vertices in candidates:
        rim_distances = _segment_distances(positions[component], starts, ends) / cell
        if float((rim_distances < 2).mean()) < 0.6:
            continue
        distances = _segment_distances(positions[vertices], starts, ends) / cell
        unsupported = distances > 0.5
        count = int(unsupported.sum())
        if count >= 6:
            report["affected_rims"] += 1
            report["unsupported_spikes"] += count
            report["severity"] += float(np.minimum(distances[unsupported], 4).sum())
    report["needs_recovery"] = report["affected_rims"] > 0
    return report
