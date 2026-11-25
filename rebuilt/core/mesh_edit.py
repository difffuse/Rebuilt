"""Non-destructive mesh edits, in numpy."""

from __future__ import annotations

import numpy as np

from .types import MeshArrays


def guard_surface_projection(mesh: MeshArrays, projected: np.ndarray) -> tuple[np.ndarray, dict]:
    """Back off locally when projection collapses a previously valid face."""
    from .analysis.mesh_graph import face_areas

    if projected.shape != mesh.positions.shape:
        raise ValueError("Projection must preserve the vertex count")
    coordinates = projected.copy()
    trial = MeshArrays(coordinates, mesh.corner_verts, mesh.face_starts, mesh.face_sizes)
    areas = face_areas(trial)
    report = {"prevented_collapses": 0, "limited_vertices": 0, "full_rollback": False}
    if not areas.size or not np.any(areas <= areas.mean() * 1e-8):
        return coordinates, report
    original_areas = face_areas(mesh)
    originally_valid = original_areas > original_areas.mean() * 1e-8
    collapsed = originally_valid & (areas <= areas.mean() * 1e-8)
    report["prevented_collapses"] = int(collapsed.sum())
    affected = np.zeros(mesh.num_verts, dtype=bool)
    for fraction in (0.5, 0.25, 0.125, 0.0):
        if not collapsed.any():
            break
        affected[mesh.corner_verts[np.repeat(collapsed, mesh.face_sizes)]] = True
        coordinates[affected] = (
            mesh.positions[affected]
            + fraction * (projected[affected] - mesh.positions[affected])
        )
        areas = face_areas(trial)
        collapsed = originally_valid & (areas <= areas.mean() * 1e-8)
    if collapsed.any():
        coordinates[:] = mesh.positions
        affected[:] = True
        report["full_rollback"] = True
    report["limited_vertices"] = int(affected.sum())
    return coordinates, report


def _rebuild_faces(
    corner_verts: np.ndarray, face_sizes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute face start offsets from sizes."""
    face_sizes = face_sizes.astype(np.int32)
    face_starts = (np.cumsum(face_sizes) - face_sizes).astype(np.int32)
    return corner_verts.astype(np.int32), face_starts


def _gather_corners(mesh: MeshArrays, face_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Corner indices of the selected faces, flattened in face order."""
    sizes = mesh.face_sizes[face_mask].astype(np.int64)
    starts = mesh.face_starts[face_mask].astype(np.int64)
    total = int(sizes.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), sizes

    within = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    return np.repeat(starts, sizes) + within, sizes


def select_faces(mesh: MeshArrays, face_mask: np.ndarray) -> MeshArrays:
    """A new mesh containing only the faces where `face_mask` is True."""
    if face_mask.dtype != bool or face_mask.shape != (mesh.num_faces,):
        raise ValueError(f"face_mask must be a boolean array of length {mesh.num_faces}")

    corner_indices, sizes = _gather_corners(mesh, face_mask)
    corner_verts, face_starts = _rebuild_faces(mesh.corner_verts[corner_indices], sizes)

    return MeshArrays(
        positions=mesh.positions.copy(),
        corner_verts=corner_verts,
        face_starts=face_starts,
        face_sizes=sizes.astype(np.int32),
    )


def remove_faces(mesh: MeshArrays, face_mask: np.ndarray) -> MeshArrays:
    """A new mesh without the faces where `face_mask` is True."""
    return select_faces(mesh, ~face_mask)


def compact_vertices(mesh: MeshArrays) -> tuple[MeshArrays, np.ndarray]:
    """Drop vertices no face refers to."""
    used = np.zeros(mesh.num_verts, dtype=bool)
    used[mesh.corner_verts.astype(np.int64)] = True
    kept = np.flatnonzero(used)

    if kept.shape[0] == mesh.num_verts:
        return mesh.copy(), kept

    remap = np.full(mesh.num_verts, -1, dtype=np.int64)
    remap[kept] = np.arange(kept.shape[0], dtype=np.int64)

    return (
        MeshArrays(
            positions=mesh.positions[kept].copy(),
            corner_verts=remap[mesh.corner_verts.astype(np.int64)].astype(np.int32),
            face_starts=mesh.face_starts.copy(),
            face_sizes=mesh.face_sizes.copy(),
        ),
        kept,
    )


def merge_by_distance(mesh: MeshArrays, tolerance: float) -> tuple[MeshArrays, int]:
    """Weld coincident vertices together."""
    if tolerance <= 0 or mesh.num_verts == 0:
        return mesh.copy(), 0

    quantised = np.round(mesh.positions.astype(np.float64) / tolerance).astype(np.int64)
    _, first_index, inverse = np.unique(quantised, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)

    merged_count = mesh.num_verts - first_index.shape[0]
    if merged_count == 0:
        return mesh.copy(), 0

    representative = first_index[inverse]
    welded = mesh.corner_verts.astype(np.int64)
    welded = representative[welded]

    result = MeshArrays(
        positions=mesh.positions.copy(),
        corner_verts=welded.astype(np.int32),
        face_starts=mesh.face_starts.copy(),
        face_sizes=mesh.face_sizes.copy(),
    )
    result = remove_collapsed_faces(result)
    result, _ = compact_vertices(result)
    return result, merged_count


def distinct_verts_per_face(mesh: MeshArrays) -> np.ndarray:
    """How many distinct vertices each face refers to. Shape (F,)."""
    if mesh.num_faces == 0:
        return np.zeros(0, dtype=np.int64)
    if mesh.num_corners == 0:
        return np.zeros(mesh.num_faces, dtype=np.int64)

    stride = np.int64(mesh.num_verts + 1)
    face_of_corner = np.repeat(
        np.arange(mesh.num_faces, dtype=np.int64), mesh.face_sizes.astype(np.int64)
    )

    key = face_of_corner * stride + mesh.corner_verts.astype(np.int64)
    key.sort()

    is_first = np.empty(key.shape[0], dtype=bool)
    is_first[0] = True
    np.not_equal(key[1:], key[:-1], out=is_first[1:])

    return np.bincount(key[is_first] // stride, minlength=mesh.num_faces)


def collapsed_face_mask(mesh: MeshArrays) -> np.ndarray:
    """Faces left with fewer than three distinct vertices after a weld."""
    return distinct_verts_per_face(mesh) < 3


def remove_collapsed_faces(mesh: MeshArrays) -> MeshArrays:
    """Drop faces that no longer describe a surface."""
    mask = collapsed_face_mask(mesh)
    return remove_faces(mesh, mask) if mask.any() else mesh


def keep_components(mesh: MeshArrays, face_labels: np.ndarray, keep: set[int]) -> MeshArrays:
    """Keep only the named connected components, then drop orphaned vertices."""
    mask = np.isin(face_labels, list(keep))
    result, _ = compact_vertices(select_faces(mesh, mask))
    return result
