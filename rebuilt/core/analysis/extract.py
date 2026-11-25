"""Reading a quad mesh off the settled position field."""

from __future__ import annotations

import numpy as np

from ..types import MeshArrays
from .crossfield import cross_pair
from .mesh_graph import manifold_edges
from .positionfield import TEAR_THRESHOLD, lattice_disagreement


MAX_FACE_SIDES = 12


FAN_SPLIT_FROM = 6


def extract_quads(
    mesh: MeshArrays,
    lattice: np.ndarray,
    field: np.ndarray,
    normals: np.ndarray,
    edge_length: np.ndarray,
    method: str = "isoline",
) -> tuple[MeshArrays, dict]:
    """Turn a settled position field into a mesh."""
    empty = MeshArrays(
        positions=np.zeros((0, 3)),
        corner_verts=np.zeros(0, dtype=np.int32),
        face_starts=np.zeros(0, dtype=np.int32),
        face_sizes=np.zeros(0, dtype=np.int32),
    )
    if mesh.num_verts == 0 or mesh.num_faces == 0:
        return empty, {"corners": 0, "faces": 0, "quad_ratio": 0.0}

    label, corners, corner_normals, corner_cell = _merge(mesh, lattice, field, normals, edge_length)
    if method == "graph":
        links, tier = _connect(mesh, label, lattice, field, normals, edge_length)
        links = _arbitrate(links, tier, corners, corner_normals, corner_cell)
        links = _stitch_gaps(links, corners, corner_normals, corner_cell)
        if links.size == 0:
            return empty, {"corners": len(corners), "faces": 0, "quad_ratio": 0.0}
        faces = _trace_faces(links, corners, corner_normals)
        if faces:
            faces, corners = _split_wide_faces(faces, corners)
    else:
        corners = _parametric_positions(mesh, lattice, field, normals, edge_length, label, corners)
        label, corners, corner_normals, corner_cell = _fuse_seam_corners(
            label, corners, corner_normals, corner_cell
        )
        faces = _isoline_faces(
            mesh, lattice, field, normals, edge_length, label, corners, corner_normals
        )
    if not faces:
        return empty, {"corners": len(corners), "faces": 0, "quad_ratio": 0.0}
    faces = _manifold_only(faces)
    if not faces:
        return empty, {"corners": len(corners), "faces": 0, "quad_ratio": 0.0}

    sizes = np.asarray([len(f) for f in faces], dtype=np.int32)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32)
    verts = np.concatenate([np.asarray(f, dtype=np.int32) for f in faces])

    used, verts = np.unique(verts, return_inverse=True)
    verts = verts.astype(np.int32)
    corners = corners[used]

    quads = int((sizes == 4).sum())
    report = {
        "corners": int(len(corners)),
        "faces": int(len(faces)),
        "quad_ratio": quads / max(len(faces), 1),
        "sides": {int(s): int((sizes == s).sum()) for s in np.unique(sizes)},
    }
    return (
        MeshArrays(
            positions=corners,
            corner_verts=verts,
            face_starts=starts,
            face_sizes=sizes,
        ),
        report,
    )


def _merge(
    mesh: MeshArrays,
    lattice: np.ndarray,
    field: np.ndarray,
    normals: np.ndarray,
    edge_length: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fold vertices that landed on the same lattice site into one corner."""
    spacing = np.maximum(edge_length.astype(np.float64), 1e-9)
    count = len(lattice)

    edges = manifold_edges(mesh).edges
    parent = np.arange(count, dtype=np.int64)

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    if edges.size:
        a = edges[:, 0].astype(np.int64)
        b = edges[:, 1].astype(np.int64)
        first, second = cross_pair(field, normals)
        offset = lattice[b] - lattice[a]

        def zero_step(ends: np.ndarray) -> np.ndarray:
            along_first = np.round(np.einsum("ij,ij->i", offset, first[ends]) / spacing[ends])
            along_second = np.round(np.einsum("ij,ij->i", offset, second[ends]) / spacing[ends])
            return (along_first == 0.0) & (along_second == 0.0)

        same = zero_step(a) & zero_step(b)
        for one, two in zip(a[same], b[same]):
            left, right = root(int(one)), root(int(two))
            if left != right:
                parent[left] = right

    roots = np.array([root(i) for i in range(count)], dtype=np.int64)
    _, label = np.unique(roots, return_inverse=True)
    total = int(label.max()) + 1 if count else 0

    corners = np.zeros((total, 3), dtype=np.float64)
    corner_normals = np.zeros((total, 3), dtype=np.float64)
    corner_cell = np.zeros(total, dtype=np.float64)
    weight = np.zeros(total, dtype=np.float64)
    np.add.at(corners, label, lattice)
    np.add.at(corner_normals, label, normals)
    np.add.at(corner_cell, label, spacing)
    np.add.at(weight, label, 1.0)
    corners /= weight[:, None]
    corner_cell /= weight

    lengths = np.linalg.norm(corner_normals, axis=1, keepdims=True)
    corner_normals = corner_normals / np.maximum(lengths, 1e-20)

    return label, corners, corner_normals, corner_cell


def _connect(
    mesh: MeshArrays,
    label: np.ndarray,
    lattice: np.ndarray,
    field: np.ndarray,
    normals: np.ndarray,
    edge_length: np.ndarray,
) -> np.ndarray:
    """Edges between corners exactly one cell apart along the field."""
    edges = manifold_edges(mesh).edges
    if edges.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)

    candidates = _two_ring_pairs(mesh.num_verts, edges)
    a = candidates[:, 0]
    b = candidates[:, 1]
    corner_a, corner_b = label[a], label[b]

    first, second = cross_pair(field, normals)
    spacing = np.maximum(edge_length.astype(np.float64), 1e-9)
    offset = lattice[b] - lattice[a]

    def unit_step(ends: np.ndarray) -> np.ndarray:
        along_first = np.round(np.einsum("ij,ij->i", offset, first[ends]) / spacing[ends])
        along_second = np.round(np.einsum("ij,ij->i", offset, second[ends]) / spacing[ends])
        return np.abs(along_first) + np.abs(along_second) == 1.0

    agrees_a, agrees_b = unit_step(a), unit_step(b)
    length = np.linalg.norm(offset, axis=1) / (0.5 * (spacing[a] + spacing[b]))
    plausible = (length > 1.0 - PLAUSIBLE_STEP_SLACK) & (length < 1.0 + PLAUSIBLE_STEP_SLACK)

    disagreement = lattice_disagreement(mesh, lattice, field, edge_length, normals)
    seam_vertex = np.zeros(mesh.num_verts, dtype=bool)
    torn = disagreement > TEAR_THRESHOLD
    seam_vertex[edges[torn].ravel()] = True
    across_seam = (
        seam_vertex[a]
        & seam_vertex[b]
        & (length > SEAM_STEP_RANGE[0])
        & (length < SEAM_STEP_RANGE[1])
    )

    tier = np.full(len(a), 4, dtype=np.int64)
    tier[across_seam] = 3
    tier[plausible] = 2
    tier[agrees_a ^ agrees_b] = 1
    tier[agrees_a & agrees_b] = 0

    keep = (corner_a != corner_b) & (tier < 4)
    if not keep.any():
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)

    stride = np.int64(label.max()) + 1
    low = np.minimum(corner_a[keep], corner_b[keep])
    high = np.maximum(corner_a[keep], corner_b[keep])
    keys = low * stride + high
    order = np.lexsort((tier[keep], keys))
    keys, tiers = keys[order], tier[keep][order]
    fresh = np.concatenate([[True], keys[1:] != keys[:-1]])

    pairs = np.stack([keys[fresh] // stride, keys[fresh] % stride], axis=1)
    return pairs, tiers[fresh]


def _two_ring_pairs(num_verts: int, edges: np.ndarray) -> np.ndarray:
    """Vertex pairs at most two edges apart, unique and unordered."""
    starts = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
    ends = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int64)
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    offsets = np.searchsorted(starts, np.arange(num_verts + 1))
    degree = np.diff(offsets)

    counts = degree[ends]
    total = int(counts.sum())
    left = np.repeat(starts, counts)
    ramp = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    right = ends[np.repeat(offsets[ends], counts) + ramp]

    keep = left != right
    pairs = np.concatenate([edges.astype(np.int64), np.stack([left[keep], right[keep]], axis=1)])
    keys = np.minimum(pairs[:, 0], pairs[:, 1]) * np.int64(num_verts) + np.maximum(
        pairs[:, 0], pairs[:, 1]
    )
    unique_keys = np.unique(keys)
    return np.stack([unique_keys // np.int64(num_verts), unique_keys % np.int64(num_verts)], axis=1)


PLAUSIBLE_STEP_SLACK = 0.3


SEAM_STEP_RANGE = (0.4, 1.35)


MIN_LINK_SEPARATION_DEG = 45.0


MIN_UNVETTED_SEPARATION_DEG = 60.0


def _arbitrate(
    links: np.ndarray,
    tier: np.ndarray,
    corners: np.ndarray,
    corner_normals: np.ndarray,
    corner_cell: np.ndarray,
) -> np.ndarray:
    """No corner keeps two links in the same direction; best fit wins."""
    if links.size == 0:
        return links

    offset = corners[links[:, 1]] - corners[links[:, 0]]
    cell = np.maximum(0.5 * (corner_cell[links[:, 0]] + corner_cell[links[:, 1]]), 1e-9)
    fit = np.abs(np.linalg.norm(offset, axis=1) / cell - 1.0)

    ordinary = np.cos(np.radians(MIN_LINK_SEPARATION_DEG))
    strict = np.cos(np.radians(MIN_UNVETTED_SEPARATION_DEG))
    accepted: list[list[np.ndarray]] = [[] for _ in range(len(corners))]

    def flattened(vector: np.ndarray, at: int) -> np.ndarray | None:
        normal = corner_normals[at]
        tangent = vector - normal * float(vector @ normal)
        length = float(np.linalg.norm(tangent))
        return None if length < 1e-12 else tangent / length

    keep = np.zeros(len(links), dtype=bool)
    for index in np.lexsort((fit, tier)):
        one, two = int(links[index, 0]), int(links[index, 1])
        step = offset[index]
        outgoing = flattened(step, one)
        incoming = flattened(-step, two)
        if outgoing is None or incoming is None:
            continue
        limit = strict if tier[index] >= 2 else ordinary
        crowded = any(float(outgoing @ other) > limit for other in accepted[one]) or any(
            float(incoming @ other) > limit for other in accepted[two]
        )
        if crowded:
            continue
        accepted[one].append(outgoing)
        accepted[two].append(incoming)
        keep[index] = True

    neighbours: list[set[int]] = [set() for _ in range(len(corners))]
    for index in np.flatnonzero(keep):
        neighbours[int(links[index, 0])].add(int(links[index, 1]))
        neighbours[int(links[index, 1])].add(int(links[index, 0]))
    for index in np.lexsort((fit, tier))[::-1]:
        if not keep[index]:
            continue
        one, two = int(links[index, 0]), int(links[index, 1])
        if neighbours[one] & neighbours[two]:
            neighbours[one].discard(two)
            neighbours[two].discard(one)
            keep[index] = False

    return links[keep]


MAX_STITCH_MISALIGNMENT_DEG = 30.0


def _stitch_gaps(
    links: np.ndarray,
    corners: np.ndarray,
    corner_normals: np.ndarray,
    corner_cell: np.ndarray,
) -> np.ndarray:
    """Give three-linked corners their missing fourth neighbour, if it exists."""
    if links.size == 0:
        return links

    neighbours: list[set[int]] = [set() for _ in range(len(corners))]
    for one, two in links:
        neighbours[int(one)].add(int(two))
        neighbours[int(two)].add(int(one))

    def flattened(vector: np.ndarray, at: int) -> np.ndarray | None:
        normal = corner_normals[at]
        tangent = vector - normal * float(vector @ normal)
        length = float(np.linalg.norm(tangent))
        return None if length < 1e-12 else tangent / length

    aligned = np.cos(np.radians(MAX_STITCH_MISALIGNMENT_DEG))
    separated = np.cos(np.radians(MIN_LINK_SEPARATION_DEG))
    stitched: list[tuple[int, int]] = []

    for _ in range(8):
        added = 0
        for corner in range(len(corners)):
            if len(neighbours[corner]) != 3:
                continue
            directions = []
            for other in neighbours[corner]:
                direction = flattened(corners[other] - corners[corner], corner)
                if direction is not None:
                    directions.append(direction)
            if len(directions) != 3:
                continue
            gap = -np.sum(directions, axis=0)
            gap_length = float(np.linalg.norm(gap))
            if gap_length < 1e-9:
                continue
            gap /= gap_length

            best = None
            best_fit = np.inf
            two_hops = {far for near in neighbours[corner] for far in neighbours[near]}
            reachable = {
                third
                for far in two_hops
                for third in neighbours[far]
                if third != corner and third not in neighbours[corner] and third not in two_hops
            }
            for candidate in reachable:
                if len(neighbours[candidate]) >= 4:
                    continue
                if neighbours[corner] & neighbours[candidate]:
                    continue
                step = corners[candidate] - corners[corner]
                cell = max(0.5 * (corner_cell[corner] + corner_cell[candidate]), 1e-9)
                length = float(np.linalg.norm(step)) / cell
                if not (1.0 - PLAUSIBLE_STEP_SLACK < length < 1.0 + PLAUSIBLE_STEP_SLACK):
                    continue
                outgoing = flattened(step, corner)
                incoming = flattened(-step, candidate)
                if outgoing is None or incoming is None:
                    continue
                if float(outgoing @ gap) < aligned:
                    continue
                crowded = any(
                    float(incoming @ flat) > separated
                    for other in neighbours[candidate]
                    if (flat := flattened(corners[other] - corners[candidate], candidate))
                    is not None
                )
                if crowded:
                    continue
                fit = abs(length - 1.0)
                if fit < best_fit:
                    best_fit = fit
                    best = candidate
            if best is not None:
                neighbours[corner].add(best)
                neighbours[best].add(corner)
                stitched.append((min(corner, best), max(corner, best)))
                added += 1
        if added == 0:
            break

    if not stitched:
        return links
    return np.concatenate([links, np.asarray(stitched, dtype=np.int64)])


SEAM_FUSE_RADIUS = 0.4


def _fuse_seam_corners(
    label: np.ndarray,
    corners: np.ndarray,
    corner_normals: np.ndarray,
    corner_cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge mutually nearest, normal-compatible corners across field seams."""
    count = len(corners)
    if count == 0:
        return label, corners, corner_normals, corner_cell

    cells = np.maximum(corner_cell.astype(np.float64), 1e-9)
    voxel = float(np.median(cells))
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, point in enumerate(corners):
        key = tuple(np.floor(point / voxel).astype(np.int64))
        buckets.setdefault(key, []).append(index)

    def closest_mate(index: int) -> int:
        centre = np.floor(corners[index] / voxel).astype(np.int64)
        best = -1
        best_distance = SEAM_FUSE_RADIUS * cells[index]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (
                        int(centre[0]) + dx,
                        int(centre[1]) + dy,
                        int(centre[2]) + dz,
                    )
                    for other in buckets.get(key, ()):
                        if other == index:
                            continue
                        distance = float(np.linalg.norm(corners[other] - corners[index]))
                        if distance < best_distance and distance < SEAM_FUSE_RADIUS * cells[other]:
                            best_distance = distance
                            best = other
        return best

    mates = np.array([closest_mate(index) for index in range(count)])

    parent = np.arange(count)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    fused_pairs = 0
    for index in range(count):
        mate = int(mates[index])
        if mate <= index:
            continue
        if int(mates[mate]) != index:
            continue
        if corner_normals[index] @ corner_normals[mate] < 0.5:
            continue
        parent[root(index)] = root(mate)
        fused_pairs += 1
    if fused_pairs == 0:
        return label, corners, corner_normals, corner_cell

    roots = np.array([root(index) for index in range(count)])
    representatives, compact = np.unique(roots, return_inverse=True)
    merged = corners.copy()
    merged_normals = corner_normals.copy()
    merged_cell = cells.copy()
    for group in np.flatnonzero(np.bincount(compact) > 1):
        members = np.flatnonzero(compact == group)
        merged[representatives[group]] = corners[members].mean(axis=0)
        pooled = corner_normals[members].sum(axis=0)
        norm = float(np.linalg.norm(pooled))
        if norm > 1e-12:
            merged_normals[representatives[group]] = pooled / norm
        merged_cell[representatives[group]] = cells[members].min()

    return (
        compact[label].astype(label.dtype),
        merged[representatives],
        merged_normals[representatives],
        merged_cell[representatives],
    )


def _isoline_faces(
    mesh: MeshArrays,
    lattice: np.ndarray,
    field: np.ndarray,
    normals: np.ndarray,
    edge_length: np.ndarray,
    label: np.ndarray,
    corners: np.ndarray,
    corner_normals: np.ndarray,
) -> list[list[int]]:
    """Quad faces read directly off the parametrization's integer grid."""
    spacing = np.maximum(edge_length.astype(np.float64), 1e-9)
    positions = mesh.positions.astype(np.float64)
    first, second = cross_pair(field, normals)

    from .mesh_graph import fan_triangles

    triangles, _ = fan_triangles(mesh)
    if triangles.shape[0] == 0:
        return []

    edges_all = manifold_edges(mesh).edges
    neighbours: dict[int, list[int]] = {}
    for one, two in edges_all:
        neighbours.setdefault(int(one), []).append(int(two))
        neighbours.setdefault(int(two), []).append(int(one))

    voxel = float(np.median(spacing))
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, point in enumerate(corners):
        key = tuple(np.floor(point / voxel).astype(np.int64))
        buckets.setdefault(key, []).append(index)

    def corner_near(point: np.ndarray, radius: float) -> int | None:
        centre = np.floor(point / voxel).astype(np.int64)
        best = None
        best_distance = radius
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (int(centre[0]) + dx, int(centre[1]) + dy, int(centre[2]) + dz)
                    for candidate in buckets.get(key, ()):
                        distance = float(np.linalg.norm(corners[candidate] - point))
                        if distance < best_distance:
                            best_distance = distance
                            best = candidate
        return best

    claimed: dict[frozenset, list] = {}

    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        axis_u, axis_v = first[a], second[a]
        cell = spacing[a]

        chart = np.empty((3, 2))
        degenerate = False
        for slot, vertex in enumerate((a, b, c)):
            offset = positions[vertex] - lattice[vertex]
            fraction = np.array([offset @ axis_u, offset @ axis_v]) / cell
            if np.any(np.abs(fraction) > 1.5):
                degenerate = True
                break
            relative = positions[vertex] - lattice[a]
            chart[slot] = np.array([relative @ axis_u, relative @ axis_v]) / cell
        if degenerate:
            continue
        area2 = abs(
            (chart[1, 0] - chart[0, 0]) * (chart[2, 1] - chart[0, 1])
            - (chart[2, 0] - chart[0, 0]) * (chart[1, 1] - chart[0, 1])
        )
        if area2 < 1e-9 or area2 > 50.0:
            continue

        candidates = {a, b, c}
        for vertex in (a, b, c):
            candidates.update(neighbours.get(vertex, ()))
        for _ in range(2):
            for vertex in list(candidates):
                candidates.update(neighbours.get(vertex, ()))
        owners: dict[tuple[int, int], int] = {}
        for vertex in candidates:
            steps = (
                int(np.round((lattice[vertex] - lattice[a]) @ axis_u / cell)),
                int(np.round((lattice[vertex] - lattice[a]) @ axis_v / cell)),
            )
            owners.setdefault(steps, int(label[vertex]))

        lo_u = int(np.floor(chart[:, 0].min()))
        hi_u = int(np.ceil(chart[:, 0].max()))
        lo_v = int(np.floor(chart[:, 1].min()))
        hi_v = int(np.ceil(chart[:, 1].max()))
        for i in range(lo_u, hi_u):
            for j in range(lo_v, hi_v):
                covered = _clip_area(chart, float(i), float(j))
                if covered < 1e-4:
                    continue
                matrix = np.array(
                    [
                        [chart[0, 0] - chart[2, 0], chart[1, 0] - chart[2, 0]],
                        [chart[0, 1] - chart[2, 1], chart[1, 1] - chart[2, 1]],
                    ]
                )
                determinant = float(np.linalg.det(matrix))
                cycle = []
                for ii, jj in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                    owner = owners.get((ii, jj))

                    point = None
                    if abs(determinant) > 1e-12:
                        rhs = np.array([float(ii), float(jj)]) - chart[2]
                        lam01 = np.linalg.solve(matrix, rhs)
                        lam = np.array([lam01[0], lam01[1], 1.0 - lam01[0] - lam01[1]])
                        if np.all(lam >= -0.6) and np.all(lam <= 1.6):
                            point = (
                                lam[0] * positions[a]
                                + lam[1] * positions[b]
                                + lam[2] * positions[c]
                            )

                    if point is not None and (
                        owner is None or np.linalg.norm(corners[owner] - point) > 0.75 * cell
                    ):
                        owner = corner_near(point, 0.45 * cell)
                    if owner is None:
                        cycle = None
                        break
                    cycle.append(owner)
                if cycle is None or len(set(cycle)) != 4:
                    continue
                key = frozenset(cycle)
                entry = claimed.get(key)
                if entry is None:
                    claimed[key] = [covered, cycle]
                else:
                    entry[0] += covered

    ranked = sorted(claimed.values(), key=lambda entry: -entry[0])
    accepted_edges: set[frozenset] = set()
    accepted_diagonals: set[frozenset] = set()
    faces = []
    for covered, cycle in ranked:
        if covered < MIN_SQUARE_COVERAGE:
            continue
        sides = [frozenset((cycle[k], cycle[(k + 1) % 4])) for k in range(4)]
        diagonals = [
            frozenset((cycle[0], cycle[2])),
            frozenset((cycle[1], cycle[3])),
        ]
        if any(
            diagonal in accepted_edges or diagonal in accepted_diagonals for diagonal in diagonals
        ) or any(side in accepted_diagonals for side in sides):
            continue
        accepted_edges.update(sides)
        accepted_diagonals.update(diagonals)
        faces.append(list(cycle))

    faces = _fill_grid_gaps(faces)
    return _oriented(faces, corners, corner_normals)


def _fill_grid_gaps(faces: list[list[int]]) -> list[list[int]]:
    """Claim the grid cells whose four sides already exist."""
    for _ in range(3):
        use: dict[tuple[int, int], int] = {}
        for face in faces:
            for slot in range(len(face)):
                key = (
                    min(face[slot], face[(slot + 1) % len(face)]),
                    max(face[slot], face[(slot + 1) % len(face)]),
                )
                use[key] = use.get(key, 0) + 1

        boundary: dict[int, set[int]] = {}
        for (one, two), count in use.items():
            if count == 1:
                boundary.setdefault(one, set()).add(two)
                boundary.setdefault(two, set()).add(one)

        existing = {frozenset(face) for face in faces}
        added = []
        seen: set[frozenset] = set()
        for start, firsts in boundary.items():
            for second in firsts:
                for third in boundary.get(second, ()):
                    if third == start:
                        continue

                    if start in boundary.get(third, ()):
                        key = frozenset((start, second, third))
                        if len(key) == 3 and key not in existing and key not in seen:
                            seen.add(key)
                            added.append([start, second, third])

                    for fourth in boundary.get(third, ()):
                        if fourth in (start, second):
                            continue
                        if start in boundary.get(fourth, ()):
                            key = frozenset((start, second, third, fourth))
                            if len(key) == 4 and key not in existing and key not in seen:
                                seen.add(key)
                                added.append([start, second, third, fourth])
        if not added:
            break
        faces = faces + added
    return faces


def _oriented(
    faces: list[list[int]], corners: np.ndarray, corner_normals: np.ndarray
) -> list[list[int]]:
    """Windings made consistent by propagation, then pointed outward."""
    if not faces:
        return faces

    edge_faces: dict[tuple[int, int], list[int]] = {}
    for index, face in enumerate(faces):
        for slot in range(len(face)):
            key = (
                min(face[slot], face[(slot + 1) % len(face)]),
                max(face[slot], face[(slot + 1) % len(face)]),
            )
            edge_faces.setdefault(key, []).append(index)

    def traverses_forward(face: list[int], one: int, two: int) -> bool:
        for slot in range(len(face)):
            if face[slot] == one and face[(slot + 1) % len(face)] == two:
                return True
        return False

    oriented = [list(face) for face in faces]
    seen = [False] * len(faces)
    for start in range(len(faces)):
        if seen[start]:
            continue
        component = [start]
        seen[start] = True
        queue = [start]
        while queue:
            current = queue.pop()
            face = oriented[current]
            for slot in range(len(face)):
                one, two = face[slot], face[(slot + 1) % len(face)]
                key = (min(one, two), max(one, two))
                for other in edge_faces[key]:
                    if seen[other]:
                        continue

                    if traverses_forward(oriented[other], one, two):
                        oriented[other] = oriented[other][::-1]
                    seen[other] = True
                    component.append(other)
                    queue.append(other)

        agreement = 0.0
        for index in component:
            cycle = oriented[index]
            pts = corners[cycle]
            newell = np.zeros(3)
            for slot in range(len(cycle)):
                newell += np.cross(pts[slot], pts[(slot + 1) % len(cycle)])
            agreement += float(newell @ corner_normals[cycle].mean(axis=0))
        if agreement < 0.0:
            for index in component:
                oriented[index] = oriented[index][::-1]

    return oriented


def _clip_area(triangle: np.ndarray, i: float, j: float) -> float:
    """Area of a chart triangle clipped to the unit square at (i, j)."""
    polygon = [tuple(point) for point in triangle]
    for edge in range(4):
        if not polygon:
            return 0.0
        clipped = []
        count = len(polygon)
        for index in range(count):
            current = polygon[index]
            following = polygon[(index + 1) % count]
            if edge == 0:
                inside_c, inside_f = current[0] >= i, following[0] >= i
                axis, level = 0, i
            elif edge == 1:
                inside_c, inside_f = current[0] <= i + 1, following[0] <= i + 1
                axis, level = 0, i + 1
            elif edge == 2:
                inside_c, inside_f = current[1] >= j, following[1] >= j
                axis, level = 1, j
            else:
                inside_c, inside_f = current[1] <= j + 1, following[1] <= j + 1
                axis, level = 1, j + 1
            if inside_c:
                clipped.append(current)
            if inside_c != inside_f:
                span = following[axis] - current[axis]
                t = 0.0 if abs(span) < 1e-12 else (level - current[axis]) / span
                clipped.append(
                    (
                        current[0] + t * (following[0] - current[0]),
                        current[1] + t * (following[1] - current[1]),
                    )
                )
        polygon = clipped
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    for index in range(len(polygon)):
        current = polygon[index]
        following = polygon[(index + 1) % len(polygon)]
        total += current[0] * following[1] - following[0] * current[1]
    return abs(total) * 0.5


MIN_SQUARE_COVERAGE = 0.25


def _parametric_positions(
    mesh: MeshArrays,
    lattice: np.ndarray,
    field: np.ndarray,
    normals: np.ndarray,
    edge_length: np.ndarray,
    label: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    """Each corner moved to the exact crossing of its two grid iso-lines."""
    spacing = np.maximum(edge_length.astype(np.float64), 1e-9)
    positions = mesh.positions.astype(np.float64)
    first, second = cross_pair(field, normals)

    from .mesh_graph import fan_triangles

    triangles, _ = fan_triangles(mesh)
    if triangles.shape[0] == 0:
        return corners

    incident: dict[int, list[int]] = {}
    for index, triangle in enumerate(triangles):
        for vertex in triangle:
            incident.setdefault(int(vertex), []).append(index)

    _, anchors = np.unique(label, return_index=True)
    refined = corners.copy()

    for corner_index, anchor in enumerate(anchors):
        anchor = int(anchor)
        axis_u = first[anchor]
        axis_v = second[anchor]
        cell = spacing[anchor]
        target_site = lattice[anchor]

        solutions = []
        for tri_index in incident.get(anchor, ()):
            tri = triangles[tri_index]
            chart = np.empty((3, 2))
            good = True
            for slot, vertex in enumerate(tri):
                vertex = int(vertex)
                whole = np.array(
                    [
                        np.round((lattice[vertex] - target_site) @ axis_u / cell),
                        np.round((lattice[vertex] - target_site) @ axis_v / cell),
                    ]
                )
                offset = positions[vertex] - lattice[vertex]
                fraction = np.array([offset @ axis_u, offset @ axis_v]) / cell

                if np.any(np.abs(fraction) > 1.5):
                    good = False
                    break
                chart[slot] = whole + fraction
            if not good:
                continue

            matrix = np.array(
                [
                    [chart[0, 0] - chart[2, 0], chart[1, 0] - chart[2, 0]],
                    [chart[0, 1] - chart[2, 1], chart[1, 1] - chart[2, 1]],
                ]
            )
            determinant = float(np.linalg.det(matrix))
            if abs(determinant) < 1e-12:
                continue
            rhs = -chart[2]
            lam01 = np.linalg.solve(matrix, rhs)
            lam = np.array([lam01[0], lam01[1], 1.0 - lam01[0] - lam01[1]])
            if np.any(lam < -0.15) or np.any(lam > 1.15):
                continue
            solutions.append(
                lam[0] * positions[int(tri[0])]
                + lam[1] * positions[int(tri[1])]
                + lam[2] * positions[int(tri[2])]
            )

        if solutions:
            candidate = np.mean(solutions, axis=0)

            if float(np.linalg.norm(candidate - refined[corner_index])) <= 0.5 * cell:
                refined[corner_index] = candidate

    return refined


def _manifold_only(faces: list[list[int]]) -> list[list[int]]:
    """Faces that leave every edge with at most two users."""
    from collections import Counter

    seen: set = set()
    survivors = []
    for face in faces:
        key = frozenset(face)
        if key in seen:
            continue
        seen.add(key)
        survivors.append(face)
    for _ in range(4):
        use: Counter = Counter()
        for face in survivors:
            for index, vertex in enumerate(face):
                other = face[(index + 1) % len(face)]
                use[(min(vertex, other), max(vertex, other))] += 1
        crowded = {edge for edge, count in use.items() if count > 2}
        if not crowded:
            break

        def offences(face: list[int]) -> int:
            return sum(
                1
                for index, vertex in enumerate(face)
                if (
                    min(vertex, face[(index + 1) % len(face)]),
                    max(vertex, face[(index + 1) % len(face)]),
                )
                in crowded
            )

        doomed = set()
        for edge in crowded:
            worst = None
            worst_count = 0
            for position, face in enumerate(survivors):
                if position in doomed:
                    continue
                for index, vertex in enumerate(face):
                    if (
                        min(vertex, face[(index + 1) % len(face)]),
                        max(vertex, face[(index + 1) % len(face)]),
                    ) == edge:
                        count = offences(face)
                        if count > worst_count:
                            worst_count = count
                            worst = position
                        break
            if worst is not None:
                doomed.add(worst)
        if not doomed:
            break
        survivors = [face for position, face in enumerate(survivors) if position not in doomed]

    parent = list(range(len(survivors)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    by_edge: dict = {}
    for position, face in enumerate(survivors):
        for index, vertex in enumerate(face):
            other = face[(index + 1) % len(face)]
            edge = (min(vertex, other), max(vertex, other))
            mate = by_edge.setdefault(edge, position)
            if mate != position:
                parent[root(mate)] = root(position)
    sizes: dict = {}
    for position in range(len(survivors)):
        sizes[root(position)] = sizes.get(root(position), 0) + 1
    if sizes and max(sizes.values()) > 2:
        survivors = [face for position, face in enumerate(survivors) if sizes[root(position)] > 2]

    return survivors


def _split_wide_faces(
    faces: list[list[int]], corners: np.ndarray
) -> tuple[list[list[int]], np.ndarray]:
    """Wide faces become fans of quads around a new centre corner."""
    out: list[list[int]] = []
    added: list[np.ndarray] = []
    for face in faces:
        count = len(face)
        if count < FAN_SPLIT_FROM:
            out.append(face)
            continue
        centre = len(corners) + len(added)
        added.append(np.mean(corners[face], axis=0))
        for start in range(0, count - (count % 2), 2):
            out.append([centre, face[start], face[(start + 1) % count], face[(start + 2) % count]])
        if count % 2:
            out.append([centre, face[count - 1], face[0]])
    if added:
        corners = np.concatenate([corners, np.asarray(added)])
    return out, corners


def _trace_faces(links: np.ndarray, corners: np.ndarray, normals: np.ndarray) -> list[list[int]]:
    """Walk the rotation system, taking every face once."""
    count = len(corners)
    directed = np.concatenate([links, links[:, ::-1]], axis=0)

    axes = np.eye(3)
    least = np.argmin(np.abs(normals), axis=1)
    first = np.cross(normals, axes[least])
    first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-20)
    second = np.cross(normals, first)

    origin, target = directed[:, 0], directed[:, 1]
    offset = corners[target] - corners[origin]
    angle = np.arctan2(
        np.einsum("ij,ij->i", offset, second[origin]),
        np.einsum("ij,ij->i", offset, first[origin]),
    )

    order = np.lexsort((angle, origin))
    ordered_origin = origin[order]
    starts = np.searchsorted(ordered_origin, np.arange(count), side="left")
    ends = np.searchsorted(ordered_origin, np.arange(count), side="right")

    slot = np.empty(len(directed), dtype=np.int64)
    slot[order] = np.arange(len(directed))
    lookup = {(int(o), int(t)): i for i, (o, t) in enumerate(directed)}

    used = np.zeros(len(directed), dtype=bool)
    faces: list[list[int]] = []

    for seed in range(len(directed)):
        if used[seed]:
            continue
        face: list[int] = []
        edge = seed
        closed = False
        for _ in range(MAX_FACE_SIDES + 1):
            if used[edge]:
                break
            used[edge] = True
            face.append(int(directed[edge][0]))

            reverse = lookup.get((int(directed[edge][1]), int(directed[edge][0])))
            if reverse is None:
                break
            here = int(directed[reverse][0])
            span = ends[here] - starts[here]
            if span <= 0:
                break
            position = slot[reverse] - starts[here]
            edge = int(order[starts[here] + (position - 1) % span])
            if edge == seed:
                closed = True
                break

        if closed and 3 <= len(face) <= MAX_FACE_SIDES and len(set(face)) == len(face):
            faces.append(face)

    return faces
