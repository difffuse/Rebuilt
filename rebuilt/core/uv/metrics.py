"""Measuring UV quality numerically."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..analysis.mesh_graph import edge_keys, face_edges
from ..types import MeshArrays
from .texel import (_corner_of_vertex_in_triangles, signed_uv_triangle_areas,
                    surface_triangles, surface_face_areas)


@dataclass(frozen=True)
class StretchReport:
    """How distorted the parameterisation is."""

    median: float = 0.0
    p95: float = 0.0
    maximum: float = 0.0

    unmeasurable: int = 0

    def within(self, tolerance: float) -> bool:
        return self.p95 <= tolerance

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OverlapReport:
    """Places where the texture would be applied twice."""

    flipped_triangles: int = 0
    overlapping_pairs: int = 0
    overlapping_area: float = 0.0

    @property
    def is_clean(self) -> bool:
        return self.flipped_triangles == 0 and self.overlapping_pairs == 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def singular_values(mesh: MeshArrays, uvs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Singular values of the UV-to-surface map, per source-interior triangle."""
    triangles, _ = surface_triangles(mesh)
    corners = _corner_of_vertex_in_triangles(mesh)
    if triangles.shape[0] == 0:
        return np.zeros(0), np.zeros(0)

    positions = mesh.positions.astype(np.float64)
    q1, q2, q3 = (positions[triangles[:, i]] for i in range(3))
    p1, p2, p3 = (uvs[corners[:, i]].astype(np.float64) for i in range(3))

    s1, t1 = p1[:, 0], p1[:, 1]
    s2, t2 = p2[:, 0], p2[:, 1]
    s3, t3 = p3[:, 0], p3[:, 1]

    doubled_area = (s2 - s1) * (t3 - t1) - (s3 - s1) * (t2 - t1)
    usable = np.abs(doubled_area) > 1e-12
    safe = np.where(usable, doubled_area, 1.0)

    d_ds = (q1 * (t2 - t3)[:, None] + q2 * (t3 - t1)[:, None] + q3 * (t1 - t2)[:, None]) / safe[
        :, None
    ]
    d_dt = (q1 * (s3 - s2)[:, None] + q2 * (s1 - s3)[:, None] + q3 * (s2 - s1)[:, None]) / safe[
        :, None
    ]

    a = np.einsum("ij,ij->i", d_ds, d_ds)
    b = np.einsum("ij,ij->i", d_ds, d_dt)
    c = np.einsum("ij,ij->i", d_dt, d_dt)

    discriminant = np.sqrt(np.maximum((a - c) ** 2 + 4.0 * b**2, 0.0))
    sigma_max = np.sqrt(np.maximum((a + c + discriminant) * 0.5, 0.0))
    sigma_min = np.sqrt(np.maximum((a + c - discriminant) * 0.5, 0.0))

    return np.where(usable, sigma_max, 0.0), np.where(usable, sigma_min, 0.0)


def stretch_report(
    mesh: MeshArrays, uvs: np.ndarray, faces: np.ndarray | None = None
) -> StretchReport:
    """Angular distortion, as a fraction. Zero means angles are preserved."""
    sigma_max, sigma_min = singular_values(mesh, uvs)
    if sigma_max.size == 0:
        return StretchReport()

    measurable = sigma_min > 1e-12
    if faces is not None:
        _, face_of_tri = surface_triangles(mesh)
        measurable &= np.asarray(faces, dtype=bool)[face_of_tri]
    unmeasurable = int((~measurable).sum())
    if not measurable.any():
        return StretchReport(unmeasurable=unmeasurable)

    ratio = sigma_max[measurable] / sigma_min[measurable] - 1.0

    _, face_of_tri = surface_triangles(mesh)
    weights = surface_face_areas(mesh)[face_of_tri][measurable]

    return StretchReport(
        median=float(_weighted_percentile(ratio, weights, 50.0)),
        p95=float(_weighted_percentile(ratio, weights, 95.0)),
        maximum=float(ratio.max()),
        unmeasurable=unmeasurable,
    )


def lost_coverage_faces(arrays, before, after):
    """New fixed-triangle collapse/fold/nonfinite UVs, including source exemptions.

    Packing may scale or rotate charts, but cannot erase previously nonzero
    coverage. Compare signs without products or an absolute area threshold so
    small, positive triangles remain measurable at their actual stored scale.
    """
    previous = signed_uv_triangle_areas(arrays, before)
    current = signed_uv_triangle_areas(arrays, after)
    had_coverage = np.isfinite(previous) & (previous != 0)
    lost = had_coverage & (~np.isfinite(current) | (current == 0)
                           | ((current > 0) != (previous > 0)))
    damaged = np.zeros(arrays.num_faces, dtype=bool)
    damaged[surface_triangles(arrays)[1][lost]] = True
    return damaged


def introduces_uv_cuts(arrays, before, after, *, precision=1e-7):
    """Optional distortion refinement must not split previously joined UV edges."""
    edges = face_edges(arrays)
    keys = edge_keys(edges, arrays.num_verts)
    order = np.argsort(keys, kind="stable")
    pairs = np.flatnonzero(keys[order][1:] == keys[order][:-1])
    if not len(pairs):
        return False
    following = np.arange(arrays.num_corners) + 1
    following[arrays.face_starts + arrays.face_sizes - 1] = arrays.face_starts
    corners = np.stack([np.arange(arrays.num_corners), following], axis=1)
    reverse = edges[:, 0] > edges[:, 1]
    corners[reverse] = corners[reverse, ::-1]
    left, right = corners[order[pairs]], corners[order[pairs + 1]]
    joined = np.all(np.abs(before[left] - before[right]) <= precision, axis=(1, 2))
    split = np.any(np.abs(after[left] - after[right]) > precision, axis=(1, 2))
    return bool(np.any(joined & split))


def island_count_from_uvs(
    mesh: MeshArrays, uvs: np.ndarray, precision: float = 1e-7, faces: np.ndarray | None = None
) -> int:
    """Count coordinate-connected UV groups for layout QA.

    This measurement groups coincident UV vertices. Native chart identity also
    accounts for source connectivity and explicit cuts; use native chart IDs
    for topology, repair and packing rather than substituting this count.
    """
    if mesh.num_faces == 0:
        return 0
    if faces is not None:
        keep = np.asarray(faces, dtype=bool)
        if not keep.any():
            return 0
        corner_keep = np.repeat(keep, mesh.face_sizes.astype(np.int64))
        sizes = mesh.face_sizes[keep].astype(np.int32)
        sub = MeshArrays(
            positions=mesh.positions,
            corner_verts=mesh.corner_verts[corner_keep],
            face_starts=np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32),
            face_sizes=sizes,
        )
        return island_count_from_uvs(sub, uvs[corner_keep], precision)
    labels = island_labels_from_uvs(mesh, uvs, precision)
    return int(labels.max()) + 1


def island_labels_from_uvs(mesh: MeshArrays, uvs: np.ndarray, precision: float = 1e-7) -> np.ndarray:
    """Label coordinate-connected UV groups, independently of source topology."""
    from ..analysis.mesh_graph import connected_components

    if not mesh.num_faces:
        return np.empty(0, dtype=np.int64)
    keys = np.round(uvs / precision).astype(np.int64)
    _, vertex_of_corner = np.unique(keys, axis=0, return_inverse=True)
    vertex_of_corner = vertex_of_corner.reshape(-1)
    face_of_corner = np.repeat(
        np.arange(mesh.num_faces, dtype=np.int64), mesh.face_sizes.astype(np.int64)
    )

    order = np.argsort(vertex_of_corner, kind="stable")
    sorted_vertices = vertex_of_corner[order]
    first = np.concatenate([[True], sorted_vertices[1:] != sorted_vertices[:-1]])
    representative = np.maximum.accumulate(np.where(first, np.arange(order.size), 0))
    joins = np.stack([face_of_corner[order], face_of_corner[order[representative]]], axis=1)
    joins = joins[joins[:, 0] != joins[:, 1]]
    labels = connected_components(mesh.num_faces, joins)
    return labels


def stretch_by_chart(
    mesh: MeshArrays, uvs: np.ndarray, labels: np.ndarray, percentile: float = 95.0
) -> tuple[np.ndarray, np.ndarray]:
    """Stretch percentile and degenerate-triangle count per chart."""
    count = int(labels.max()) + 1 if labels.size else 0
    p95 = np.full(count, np.inf)
    degenerate = np.zeros(count, dtype=np.int64)
    if count == 0:
        return p95, degenerate
    sigma_max, sigma_min = singular_values(mesh, uvs)
    _, face_of_tri = surface_triangles(mesh)
    if sigma_max.size == 0:
        return p95, degenerate
    areas = surface_face_areas(mesh)[face_of_tri]
    chart_of_tri = labels[face_of_tri]
    measurable = sigma_min > 1e-12
    ratio = np.where(measurable, sigma_max / np.maximum(sigma_min, 1e-12) - 1.0, 0.0)
    degenerate = np.bincount(chart_of_tri[~measurable], minlength=count)
    for chart in range(count):
        mine = (chart_of_tri == chart) & measurable
        if mine.any():
            p95[chart] = _weighted_percentile(ratio[mine], areas[mine], percentile)
    return p95, degenerate


def blown_faces(
    mesh: MeshArrays, uvs: np.ndarray, ratio_limit: float = 20.0, ignore_slivers: bool = True
) -> np.ndarray:
    """Faces the unwrap did not survive: a triangle collapsed to nothing, or stretched past `ratio_limit` (twenty means one direction is twenty times the other — a spike, not a texture)."""
    sigma_max, sigma_min = singular_values(mesh, uvs)
    blown = np.zeros(mesh.num_faces, dtype=bool)
    if sigma_max.size == 0:
        return blown
    triangles, face_of_tri = surface_triangles(mesh)
    degenerate = sigma_min <= 1e-12
    ratio = sigma_max / np.maximum(sigma_min, 1e-12)
    bad = degenerate | (ratio > ratio_limit + 1.0)
    if ignore_slivers:
        positions = mesh.positions.astype(np.float64)
        a, b, c = (positions[triangles[:, i]] for i in range(3))
        area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
        edges2 = ((b - a) ** 2).sum(1) + ((c - b) ** 2).sum(1) + ((a - c) ** 2).sum(1)
        shape = 2.0 * np.sqrt(3.0) * area2 / np.maximum(edges2, 1e-30)
        bad &= shape >= 0.02
    blown[np.unique(face_of_tri[bad])] = True
    return blown


def spiked_faces(mesh: MeshArrays, uvs: np.ndarray, factor: float = 10.0) -> np.ndarray:
    """Faces whose share of the UV area is `factor` times their share of the surface: the spikes the solver shoots across the tile."""
    triangles, face_of_tri = surface_triangles(mesh)
    spiked = np.zeros(mesh.num_faces, dtype=bool)
    if triangles.shape[0] == 0:
        return spiked
    corners = _corner_of_vertex_in_triangles(mesh)
    p = uvs[corners].astype(np.float64)
    uv_area = 0.5 * np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
        - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    )
    positions = mesh.positions.astype(np.float64)
    a, b, c = (positions[triangles[:, i]] for i in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total_uv, total = float(uv_area.sum()), float(area.sum())
    if total_uv <= 0 or total <= 0:
        return spiked
    share_uv = uv_area / total_uv
    share = area / total
    bad = (share_uv > factor * share) & (share_uv > 1e-5)

    extent = np.linalg.norm(p.max(axis=1) - p.min(axis=1), axis=1)
    expected = np.sqrt(np.maximum(share, 1e-12))
    bad |= (extent > factor * expected) & (extent > 0.05)
    spiked[np.unique(face_of_tri[bad])] = True
    return spiked


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    """Percentile of `values` weighted by `weights`."""
    if values.size == 0:
        return 0.0
    total = float(weights.sum())
    if total <= 0:
        return float(np.percentile(values, percentile))

    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order]) / total
    index = int(np.searchsorted(cumulative, percentile / 100.0))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def overlap_report(
    mesh: MeshArrays, uvs: np.ndarray, grid: int = 64, faces: np.ndarray | None = None
) -> OverlapReport:
    """Find UV triangles that cover the same texture twice."""
    signed = signed_uv_triangle_areas(mesh, uvs)
    if signed.size == 0:
        return OverlapReport()

    _, face_of_tri = surface_triangles(mesh)
    keep = np.ones(signed.shape[0], dtype=bool)
    if faces is not None:
        keep = np.asarray(faces, dtype=bool)[face_of_tri]
        signed = signed[keep]
        if signed.size == 0:
            return OverlapReport()

    positive = int((signed > 0).sum())
    negative = int((signed < 0).sum())
    flipped = min(positive, negative)

    corners = _corner_of_vertex_in_triangles(mesh)[keep]
    coords = uvs[corners]
    pairs, area = _overlapping_pairs(coords, face_of_tri[keep], grid)

    return OverlapReport(
        flipped_triangles=flipped,
        overlapping_pairs=pairs,
        overlapping_area=area,
    )


def _overlapping_pairs(coords: np.ndarray, face_of_tri: np.ndarray, grid: int) -> tuple[int, float]:
    overlapping = 0
    area = 0.0
    for first, second, pair_area in _overlap_pairs(coords, face_of_tri, grid):
        overlapping += 1
        area += pair_area
    return overlapping, float(area)


def _overlap_pairs(coords: np.ndarray, face_of_tri: np.ndarray, grid: int):
    """Yield (triangle, triangle, rough area) for every pair that genuinely overlaps."""
    low = coords.min(axis=1)
    high = coords.max(axis=1)

    span = float(max(high.max() - low.min(), 1e-9))
    origin = low.min(axis=0)
    cell = span / max(grid, 1)

    start = np.floor((low - origin) / cell).astype(np.int64)
    end = np.floor((high - origin) / cell).astype(np.int64)

    buckets: dict[tuple[int, int], list[int]] = {}
    for index in range(coords.shape[0]):
        for x in range(start[index, 0], end[index, 0] + 1):
            for y in range(start[index, 1], end[index, 1] + 1):
                buckets.setdefault((x, y), []).append(index)

    seen: set[int] = set()
    triangle_count = len(coords)
    for members in buckets.values():
        members = np.asarray(members, dtype=np.int64)
        count = len(members)
        for offset in range(0, count, 64):
            rows = np.arange(offset, min(offset + 64, count))
            i = np.repeat(rows, count)
            j = np.tile(np.arange(count), len(rows))
            upper = j > i
            first, second = members[i[upper]], members[j[upper]]
            possible = (
                (face_of_tri[first] != face_of_tri[second])
                & (low[first, 0] < high[second, 0] - 1e-9)
                & (low[second, 0] < high[first, 0] - 1e-9)
                & (low[first, 1] < high[second, 1] - 1e-9)
                & (low[second, 1] < high[first, 1] - 1e-9)
            )
            fresh = []
            for first_id, second_id in zip(first[possible], second[possible]):
                key = int(first_id) * triangle_count + int(second_id)
                if key not in seen:
                    seen.add(key)
                    fresh.append((int(first_id), int(second_id)))
            for start_pair in range(0, len(fresh), 512):
                pairs = np.asarray(fresh[start_pair:start_pair + 512], dtype=np.int64)
                overlaps = _triangle_pairs_overlap(coords[pairs[:, 0]], coords[pairs[:, 1]])
                for first_id, second_id in pairs[overlaps]:
                    yield (int(first_id), int(second_id),
                           _box_overlap_area(low[first_id], high[first_id], low[second_id], high[second_id]))


def _triangle_pairs_overlap(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Apply the scalar separating-axis test to independent triangle pairs."""
    edges = np.concatenate((np.roll(first, -1, axis=1) - first,
                            np.roll(second, -1, axis=1) - second), axis=1)
    axes = np.stack((-edges[:, :, 1], edges[:, :, 0]), axis=2)
    lengths = np.hypot(axes[:, :, 0], axes[:, :, 1])
    usable = lengths >= 1e-12
    axes /= np.where(usable, lengths, 1.0)[:, :, None]
    a = np.einsum("bvi,bai->bav", first, axes)
    b = np.einsum("bvi,bai->bav", second, axes)
    separated = ((a.max(axis=2) <= b.min(axis=2) + 1e-9)
                 | (b.max(axis=2) <= a.min(axis=2) + 1e-9)) & usable
    return ~separated.any(axis=1)


def overlapping_faces(
    mesh: MeshArrays, uvs: np.ndarray, grid: int = 64, substantial: float = 0.5, min_region: int = 6,
    face_groups: np.ndarray | None = None,
) -> np.ndarray:
    """The fewest faces whose removal from the layout leaves no overlap."""
    triangles, face_of_tri = surface_triangles(mesh)
    chosen = np.zeros(mesh.num_faces, dtype=bool)
    if triangles.shape[0] == 0:
        return chosen
    corners = _corner_of_vertex_in_triangles(mesh)
    coords = uvs[corners].astype(np.float64)

    tri_area = 0.5 * np.abs(
        (coords[:, 1, 0] - coords[:, 0, 0]) * (coords[:, 2, 1] - coords[:, 0, 1])
        - (coords[:, 2, 0] - coords[:, 0, 0]) * (coords[:, 1, 1] - coords[:, 0, 1])
    )
    pairs = [
        (int(face_of_tri[a]), int(face_of_tri[b]))
        for a, b, area in _overlap_pairs(coords, face_of_tri, grid)
        if area >= substantial * max(min(tri_area[a], tri_area[b]), 1e-12)
        and (face_groups is None or face_groups[face_of_tri[a]] == face_groups[face_of_tri[b]])
    ]
    if not pairs:
        return chosen
    remaining = set(pairs)
    while remaining:
        counts: dict[int, int] = {}
        for a, b in remaining:
            counts[a] = counts.get(a, 0) + 1
            counts[b] = counts.get(b, 0) + 1
        face = max(counts, key=counts.get)
        chosen[face] = True
        remaining = {(a, b) for a, b in remaining if a != face and b != face}

    if min_region > 1 and chosen.any():
        from ..analysis.mesh_graph import connected_components, manifold_edges

        edges = manifold_edges(mesh)
        both = chosen[edges.faces[:, 0]] & chosen[edges.faces[:, 1]]
        region = connected_components(mesh.num_faces, edges.faces[both])
        sizes = np.bincount(region[chosen], minlength=int(region.max()) + 1)
        chosen &= sizes[region] >= min_region
    return chosen


def _boxes_overlap(low_a, high_a, low_b, high_b, tolerance: float = 1e-9) -> bool:
    return bool(
        (low_a[0] < high_b[0] - tolerance)
        and (low_b[0] < high_a[0] - tolerance)
        and (low_a[1] < high_b[1] - tolerance)
        and (low_b[1] < high_a[1] - tolerance)
    )


def _box_overlap_area(low_a, high_a, low_b, high_b) -> float:
    width = max(0.0, float(min(high_a[0], high_b[0]) - max(low_a[0], low_b[0])))
    height = max(0.0, float(min(high_a[1], high_b[1]) - max(low_a[1], low_b[1])))
    return width * height


def _triangles_overlap(first: np.ndarray, second: np.ndarray) -> bool:
    """Separating axis test for two 2D triangles."""
    for triangle in (first, second):
        for i in range(3):
            edge = triangle[(i + 1) % 3] - triangle[i]
            axis = np.array([-edge[1], edge[0]])
            length = np.hypot(axis[0], axis[1])
            if length < 1e-12:
                continue
            axis = axis / length

            a_lo, a_hi = _project(first, axis)
            b_lo, b_hi = _project(second, axis)
            if a_hi <= b_lo + 1e-9 or b_hi <= a_lo + 1e-9:
                return False
    return True


def _project(triangle: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    values = triangle @ axis
    return float(values.min()), float(values.max())


def uv_occupancy(mesh: MeshArrays, uvs: np.ndarray) -> float:
    """Fraction of the unit tile actually covered by islands."""
    from .texel import uv_face_areas

    covered = float(uv_face_areas(mesh, uvs).sum())
    return float(np.clip(covered, 0.0, None))
