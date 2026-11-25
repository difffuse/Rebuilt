"""Cutting a surface into a few large charts, the way an artist would."""

from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..analysis.mesh_graph import ManifoldEdges, face_areas, face_normals, manifold_edges
from ..types import MeshArrays


@dataclass
class Charting:
    """One partition of the faces into charts, and the seams that cut it."""

    labels: np.ndarray

    proxies: np.ndarray

    seams: np.ndarray

    ring_cuts: int = 0

    error: float = 0.0

    @property
    def chart_count(self) -> int:
        return int(self.proxies.shape[0])


@dataclass
class _Graph:
    """Face adjacency as CSR, built once per mesh."""

    edges: ManifoldEdges
    neighbours: np.ndarray
    offsets: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    neighbour_list: list[int] = field(default_factory=list)
    offset_list: list[int] = field(default_factory=list)

    @classmethod
    def build(cls, mesh: MeshArrays, normals: np.ndarray | None = None) -> _Graph:
        if normals is None:
            normals = face_normals(mesh)
        areas = face_areas(mesh)
        edges = manifold_edges(mesh, normals)
        faces = edges.faces
        starts = np.concatenate([faces[:, 0], faces[:, 1]]).astype(np.int64)
        ends = np.concatenate([faces[:, 1], faces[:, 0]]).astype(np.int64)
        order = np.argsort(starts, kind="stable")
        starts, ends = starts[order], ends[order]
        offsets = np.searchsorted(starts, np.arange(mesh.num_faces + 1))
        return cls(
            edges,
            ends,
            offsets,
            normals.astype(np.float64),
            areas.astype(np.float64),
            ends.tolist(),
            offsets.tolist(),
        )

    def around(self, face: int) -> np.ndarray:
        return self.neighbours[self.offsets[face] : self.offsets[face + 1]]


def _partition(graph: _Graph, seeds: list[int], proxies: np.ndarray) -> np.ndarray:
    """Grow every region from its seed at once, cheapest face first."""
    count = graph.normals.shape[0]
    labels = [-1] * count

    costs = [
        (graph.areas * np.sum((graph.normals - proxy) ** 2, axis=1)).tolist() for proxy in proxies
    ]
    neighbours, offsets = graph.neighbour_list, graph.offset_list
    heap: list[tuple[float, int, int]] = []
    push, pop = heapq.heappush, heapq.heappop
    for region, seed in enumerate(seeds):
        labels[seed] = region
        cost = costs[region]
        for other in neighbours[offsets[seed] : offsets[seed + 1]]:
            push(heap, (cost[other], other, region))
    while heap:
        _, face, region = pop(heap)
        if labels[face] != -1:
            continue
        labels[face] = region
        cost = costs[region]
        for other in neighbours[offsets[face] : offsets[face + 1]]:
            if labels[other] == -1:
                push(heap, (cost[other], other, region))
    labels = np.asarray(labels, dtype=np.int64)

    orphans = np.flatnonzero(labels == -1)
    if orphans.size:
        fit = ((graph.normals[orphans][:, None, :] - proxies[None, :, :]) ** 2).sum(axis=2)
        labels[orphans] = np.argmin(fit, axis=1)
    return labels


def _fit(graph: _Graph, labels: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Refit each proxy to its region; returns (proxies, per-face error)."""
    proxies = np.zeros((count, 3), dtype=np.float64)
    weighted = graph.normals * graph.areas[:, None]
    for axis in range(3):
        proxies[:, axis] = np.bincount(labels, weights=weighted[:, axis], minlength=count)
    lengths = np.linalg.norm(proxies, axis=1, keepdims=True)
    proxies = np.divide(proxies, lengths, out=np.zeros_like(proxies), where=lengths > 1e-20)
    error = graph.areas * np.sum((graph.normals - proxies[labels]) ** 2, axis=1)
    return proxies, error


def _reseed(labels: np.ndarray, error: np.ndarray, count: int) -> list[int]:
    """The best-fitting face of each region becomes its seed."""
    seeds = []
    for region in range(count):
        members = np.flatnonzero(labels == region)
        if members.size == 0:
            seeds.append(int(np.argmax(error)))
            continue
        seeds.append(int(members[np.argmin(error[members])]))
    return seeds


def segment_charts(
    mesh: MeshArrays,
    counts: tuple[int, ...],
    normals: np.ndarray | None = None,
    iterations: int = 6,
    seed_labels: np.ndarray | None = None,
    finish: bool = True,
    check: Callable[[], None] | None = None,
) -> dict[int, Charting]:
    """One charting per requested chart count, from a single incremental run."""
    if mesh.num_faces == 0:
        return {}
    wanted = sorted({int(c) for c in counts if c >= 1})
    if not wanted:
        return {}
    graph = _Graph.build(mesh, normals)
    total = graph.normals.shape[0]
    largest = wanted[-1]

    seeds: list[int] = []
    if seed_labels is not None and seed_labels.shape[0] == total:
        region_area = np.bincount(seed_labels, weights=graph.areas)
        for region in np.argsort(region_area)[::-1]:
            if len(seeds) >= wanted[0]:
                break
            members = np.flatnonzero(seed_labels == region)
            if members.size == 0:
                continue
            seeds.append(int(members[np.argmax(graph.areas[members])]))
    if not seeds:
        seeds.append(int(np.argmax(graph.areas)))

    proxies = graph.normals[seeds].copy()
    labels = _partition(graph, seeds, proxies)
    proxies, error = _fit(graph, labels, len(seeds))
    results: dict[int, Charting] = {}

    while True:
        for round_index in range(iterations):
            if check is not None:
                check()
            seeds = _reseed(labels, error, len(seeds))
            labels = _partition(graph, seeds, proxies)
            proxies, error = _fit(graph, labels, len(seeds))
            if round_index == 0 and len(seeds) >= 3:
                moved = _teleport(graph, labels, proxies, error)
                if moved is not None:
                    labels, proxies, error = moved
        if len(seeds) in wanted:
            if finish:
                results[len(seeds)] = _finish(mesh, graph, labels, proxies, error)
            else:
                seams = labels[graph.edges.faces[:, 0]] != labels[graph.edges.faces[:, 1]]
                results[len(seeds)] = Charting(
                    labels=labels.copy(), proxies=proxies.copy(), seams=seams
                )
        if len(seeds) >= largest:
            break

        region_error = np.bincount(labels, weights=error, minlength=len(seeds))
        region = int(np.argmax(region_error))
        members = np.flatnonzero(labels == region)
        worst = int(members[np.argmax(error[members])])
        seeds = _reseed(labels, error, len(seeds)) + [worst]
        proxies = np.concatenate([proxies, graph.normals[worst][None, :]])
        labels = _partition(graph, seeds, proxies)
        proxies, error = _fit(graph, labels, len(seeds))
    return results


def _teleport(graph: _Graph, labels, proxies, error):
    """The paper's escape from a local minimum: two regions that would cost little to merge give one of them up, and that region is reseeded at the face the partition explains worst."""
    count = proxies.shape[0]
    region_error = np.bincount(labels, weights=error, minlength=count)
    worst = int(np.argmax(region_error))
    faces = graph.edges.faces
    across = labels[faces[:, 0]] != labels[faces[:, 1]]
    if not across.any():
        return None
    pairs = np.unique(
        np.sort(np.stack([labels[faces[across, 0]], labels[faces[across, 1]]], axis=1), axis=1),
        axis=0,
    )
    weighted = graph.normals * graph.areas[:, None]
    region_sum = np.zeros((count, 3))
    for axis in range(3):
        region_sum[:, axis] = np.bincount(labels, weights=weighted[:, axis], minlength=count)
    best, best_delta = None, np.inf
    for one, two in pairs:
        if worst in (one, two):
            continue
        merged = region_sum[one] + region_sum[two]
        norm = float(np.linalg.norm(merged))
        if norm <= 1e-20:
            continue
        merged /= norm
        members = (labels == one) | (labels == two)
        cost = float(
            (graph.areas[members] * np.sum((graph.normals[members] - merged) ** 2, axis=1)).sum()
        )
        delta = cost - float(region_error[one] + region_error[two])
        if delta < best_delta:
            best, best_delta = (int(one), int(two)), delta
    if best is None or best_delta >= 0.5 * float(region_error[worst]):
        return None
    one, two = best
    labels = labels.copy()
    labels[labels == one] = two
    seed = int(np.argmax(error))
    labels[seed] = one
    proxies = proxies.copy()
    proxies[one] = graph.normals[seed]
    seeds = _reseed(labels, error, count)
    seeds[one] = seed
    labels = _partition(graph, seeds, proxies)
    return (labels,) + _fit(graph, labels, count)


SPECK_FACES = 8


def absorb_specks(graph: _Graph, labels: np.ndarray, min_faces: int = SPECK_FACES) -> np.ndarray:
    """Fold every connected piece of a chart under `min_faces` into the chart next to it, smallest first."""
    from ..analysis.mesh_graph import connected_components

    labels = labels.copy()
    faces = graph.edges.faces
    lengths = graph.edges.lengths
    for _ in range(1000):
        same = labels[faces[:, 0]] == labels[faces[:, 1]]
        piece = connected_components(labels.shape[0], faces[same])
        sizes = np.bincount(piece)
        small = np.flatnonzero(sizes < min_faces)
        if small.size == 0:
            break
        a, b = piece[faces[:, 0]], piece[faces[:, 1]]
        across = a != b
        merged = False
        for target in small[np.argsort(sizes[small])]:
            touching = across & ((a == target) | (b == target))
            if not touching.any():
                continue
            other = np.where(a[touching] == target, b[touching], a[touching])
            border = np.bincount(other, weights=lengths[touching], minlength=sizes.size)
            labels[piece == target] = labels[np.flatnonzero(piece == int(np.argmax(border)))[0]]
            merged = True
            break
        if not merged:
            break
    return labels


def _finish(mesh, graph: _Graph, labels, proxies, error) -> Charting:
    labels = _relabel(absorb_specks(graph, labels))
    labels = split_handles(mesh, graph, labels)
    count = int(labels.max()) + 1
    proxies, error = _fit(graph, labels, count)
    seams = labels[graph.edges.faces[:, 0]] != labels[graph.edges.faces[:, 1]]
    seams, cuts = open_rings(mesh, graph.edges, labels, seams)
    area = float(graph.areas.sum())
    return Charting(
        labels=labels,
        proxies=proxies,
        seams=seams,
        ring_cuts=cuts,
        error=float(error.sum() / area) if area > 0 else 0.0,
    )


def _relabel(labels: np.ndarray) -> np.ndarray:
    """Dense labels 0..K-1."""
    _, dense = np.unique(labels, return_inverse=True)
    return dense.reshape(labels.shape).astype(np.int64)


def euler_characteristic(mesh: MeshArrays, graph: _Graph, labels: np.ndarray, chart: int) -> int:
    """V - E + F of one chart's faces, as a surface of its own."""
    members = np.flatnonzero(labels == chart)
    if members.size == 0:
        return 1
    starts = mesh.face_starts[members].astype(np.int64)
    sizes = mesh.face_sizes[members].astype(np.int64)
    corners = np.concatenate(
        [mesh.corner_verts[start : start + size] for start, size in zip(starts, sizes)]
    )
    vertices = int(np.unique(corners).size)
    faces_of = graph.edges.faces
    interior = int(((labels[faces_of[:, 0]] == chart) & (labels[faces_of[:, 1]] == chart)).sum())
    edges = int(sizes.sum()) - interior
    return vertices - edges + int(members.size)


def split_handles(
    mesh: MeshArrays, graph: _Graph, labels: np.ndarray, rounds: int = 4
) -> np.ndarray:
    """Cut every chart that is not a disc — after its rings are opened — in two."""
    from ..mesh_edit import compact_vertices, select_faces

    labels = labels.copy()
    no_seams = np.zeros(graph.edges.count, dtype=bool)
    open_info = _open_edges_with_owner(mesh)
    for _ in range(rounds):
        split_any = False
        for chart in range(int(labels.max()) + 1):
            chi = euler_characteristic(mesh, graph, labels, chart)
            loops = len(_boundary_loops(mesh, graph.edges, labels, no_seams, chart, open_info))
            if 2 - chi - loops < 2:
                continue
            members = np.flatnonzero(labels == chart)
            if members.size < 8:
                continue
            mask = np.zeros(mesh.num_faces, dtype=bool)
            mask[members] = True
            piece, _ = compact_vertices(select_faces(mesh, mask))
            halves = segment_charts(piece, (2,), normals=graph.normals[members], finish=False)
            if 2 not in halves:
                continue
            local = halves[2].labels
            if local.min() == local.max():
                continue
            labels[members[local == 1]] = int(labels.max()) + 1
            split_any = True
        if not split_any:
            break
    return _relabel(labels)


def open_rings(
    mesh: MeshArrays, edges: ManifoldEdges, labels: np.ndarray, seams: np.ndarray
) -> tuple[np.ndarray, int]:
    """Cut every chart with more than one boundary loop until it is a disc."""
    seams = seams.copy()
    cuts = 0
    count = int(labels.max()) + 1 if labels.size else 0
    lengths = edges.lengths
    open_info = _open_edges_with_owner(mesh)

    for chart in range(count):
        for _ in range(8):
            loops = _boundary_loops(mesh, edges, labels, seams, chart, open_info)
            if len(loops) <= 1:
                break
            path = _shortest_path(edges, labels, chart, loops[0], loops[1], lengths)
            if not path:
                break
            seams[path] = True
            cuts += 1
    return seams, cuts


def _open_edges_with_owner(mesh: MeshArrays) -> tuple[np.ndarray, np.ndarray]:
    """(open edges as vertex pairs, owning face per open edge)."""
    from ..analysis.mesh_graph import edge_keys, edge_topology, face_edges

    topology = edge_topology(mesh)
    open_edges = topology.edges[topology.boundary]
    owner = np.zeros(open_edges.shape[0], dtype=np.int64)
    if open_edges.size:
        face_of_corner = np.repeat(
            np.arange(mesh.num_faces, dtype=np.int64), mesh.face_sizes.astype(np.int64)
        )
        keys = edge_keys(face_edges(mesh), mesh.num_verts)
        open_keys = edge_keys(open_edges, mesh.num_verts)
        order = np.argsort(keys, kind="stable")
        position = np.searchsorted(keys[order], open_keys)
        owner = face_of_corner[order[np.clip(position, 0, len(order) - 1)]]
    return open_edges, owner


def _boundary_loops(
    mesh, edges: ManifoldEdges, labels, seams, chart, open_info=None
) -> list[set[int]]:
    """Vertex sets of the chart's boundary loops."""
    from ..analysis.mesh_graph import connected_components

    inside = labels[edges.faces[:, 0]] == chart
    inside |= labels[edges.faces[:, 1]] == chart
    both = (labels[edges.faces[:, 0]] == chart) & (labels[edges.faces[:, 1]] == chart)

    rim = edges.edges[(inside & ~both) | (both & seams)]
    open_edges, owner = open_info if open_info is not None else _open_edges_with_owner(mesh)
    if open_edges.size:
        mine = labels[owner] == chart
        if mine.any():
            rim = np.concatenate([rim, open_edges[mine]])
    if rim.shape[0] == 0:
        return []
    vertices = np.unique(rim)
    remap = np.full(int(vertices.max()) + 1, -1, dtype=np.int64)
    remap[vertices] = np.arange(vertices.shape[0])
    components = connected_components(vertices.shape[0], remap[rim])
    return [set(vertices[components == c].tolist()) for c in range(int(components.max()) + 1)]


def _shortest_path(
    edges: ManifoldEdges, labels, chart, source: set[int], target: set[int], lengths
):
    """Dijkstra over the chart's interior edges from one loop to another. Returns the indices (into `edges`) of the path's edges."""
    both = (labels[edges.faces[:, 0]] == chart) & (labels[edges.faces[:, 1]] == chart)
    interior = np.flatnonzero(both)
    if interior.size == 0:
        return []
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for index in interior:
        a, b = int(edges.edges[index, 0]), int(edges.edges[index, 1])
        adjacency.setdefault(a, []).append((b, int(index)))
        adjacency.setdefault(b, []).append((a, int(index)))
    distance: dict[int, float] = {v: 0.0 for v in source if v in adjacency}
    via: dict[int, tuple[int, int]] = {}
    heap = [(0.0, v) for v in distance]
    heapq.heapify(heap)
    reached = None
    while heap:
        d, vertex = heapq.heappop(heap)
        if d > distance.get(vertex, np.inf):
            continue
        if vertex in target and vertex not in source:
            reached = vertex
            break
        for other, index in adjacency.get(vertex, ()):
            candidate = d + float(lengths[index])
            if candidate < distance.get(other, np.inf):
                distance[other] = candidate
                via[other] = (vertex, index)
                heapq.heappush(heap, (candidate, other))
    if reached is None:
        return []
    path = []
    vertex = reached
    while vertex in via:
        vertex, index = via[vertex]
        path.append(index)
    return path


def panel_charts(
    mesh: MeshArrays,
    angle_deg: float,
    min_fraction: float = 0.01,
    normals: np.ndarray | None = None,
) -> Charting:
    """Charts for a hard-surface scan: its panels, with the seams on its edges."""
    graph = _Graph.build(mesh, normals)
    total = graph.normals.shape[0]
    labels = np.full(total, -1, dtype=np.int64)
    limit = float(np.cos(np.radians(angle_deg)))
    region = 0
    for seed in np.argsort(graph.areas)[::-1]:
        seed = int(seed)
        if labels[seed] != -1:
            continue
        labels[seed] = region
        proxy_sum = graph.normals[seed] * graph.areas[seed]
        proxy = graph.normals[seed]
        heap: list[tuple[float, int]] = []
        for other in graph.around(seed):
            if labels[other] == -1:
                heapq.heappush(heap, (-float(graph.normals[other] @ proxy), int(other)))
        while heap:
            _, face = heapq.heappop(heap)
            if labels[face] != -1:
                continue
            if float(graph.normals[face] @ proxy) < limit:
                continue
            labels[face] = region
            proxy_sum = proxy_sum + graph.normals[face] * graph.areas[face]
            norm = float(np.linalg.norm(proxy_sum))
            if norm > 1e-20:
                proxy = proxy_sum / norm
            for other in graph.around(face):
                if labels[other] == -1:
                    heapq.heappush(heap, (-float(graph.normals[other] @ proxy), int(other)))
        region += 1
    labels = _merge_small(graph, labels, min_fraction)
    labels = _relabel(labels)
    count = int(labels.max()) + 1
    proxies, error = _fit(graph, labels, count)
    return _finish(mesh, graph, labels, proxies, error)


def _merge_small(graph: _Graph, labels: np.ndarray, min_fraction: float) -> np.ndarray:
    """Fold every region under `min_fraction` of the surface into the neighbour it shares the longest border with, smallest first."""
    labels = labels.copy()
    total = float(graph.areas.sum())
    faces = graph.edges.faces
    lengths = graph.edges.lengths
    stuck: set[int] = set()
    for _ in range(10_000):
        count = int(labels.max()) + 1
        area = np.bincount(labels, weights=graph.areas, minlength=count)
        present = np.bincount(labels, minlength=count) > 0
        small = [
            int(r)
            for r in np.flatnonzero((area < min_fraction * total) & present)
            if int(r) not in stuck
        ]
        if not small:
            break
        region = min(small, key=lambda r: area[r])
        a, b = labels[faces[:, 0]], labels[faces[:, 1]]
        touching = ((a == region) & (b != region)) | ((b == region) & (a != region))
        if not touching.any():
            stuck.add(region)
            continue
        other = np.where(a[touching] == region, b[touching], a[touching])
        border = np.bincount(other, weights=lengths[touching], minlength=count)
        labels[labels == region] = int(np.argmax(border))
    return labels


def charting_from_labels(
    mesh: MeshArrays, labels: np.ndarray, normals: np.ndarray | None = None
) -> Charting:
    """A charting for a partition decided elsewhere: handles split, seams on the chart boundaries, rings opened."""
    graph = _Graph.build(mesh, normals)
    labels = _relabel(np.asarray(labels, dtype=np.int64))
    count = int(labels.max()) + 1
    proxies, error = _fit(graph, labels, count)
    return _finish(mesh, graph, labels, proxies, error)


def split_chart(
    mesh: MeshArrays, labels: np.ndarray, chart: int, normals: np.ndarray | None = None
) -> np.ndarray:
    """One more cut: the chart re-partitioned into two by normal, the way the organic charts are grown."""
    from ..mesh_edit import compact_vertices, select_faces

    labels = np.asarray(labels, dtype=np.int64).copy()
    members = np.flatnonzero(labels == chart)
    if members.size < 8:
        return labels
    mask = np.zeros(mesh.num_faces, dtype=bool)
    mask[members] = True
    piece, _ = compact_vertices(select_faces(mesh, mask))
    piece_normals = normals[members] if normals is not None else None
    halves = segment_charts(piece, (2,), normals=piece_normals, finish=False)
    if 2 not in halves:
        return labels
    local = halves[2].labels
    if local.min() == local.max():
        return labels
    labels[members[local == local.max()]] = int(labels.max()) + 1
    return labels


def chart_areas(mesh: MeshArrays, labels: np.ndarray) -> np.ndarray:
    """Surface area per chart, shape (K,)."""
    count = int(labels.max()) + 1 if labels.size else 0
    return np.bincount(labels, weights=face_areas(mesh), minlength=count)
