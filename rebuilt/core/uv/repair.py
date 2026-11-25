"""Independent measurements of finite, oriented UV coverage."""

from __future__ import annotations

import numpy as np

from ..types import MeshArrays


def invalid_faces(mesh: MeshArrays, uvs: np.ndarray, *, within_islands=False) -> np.ndarray:
    """Faces that cannot provide unique, finite texture coverage."""
    from .metrics import overlapping_faces, island_labels_from_uvs
    from .texel import signed_uv_triangle_areas, surface_triangles

    finite = np.isfinite(uvs).all(axis=1)
    safe = np.where(np.isfinite(uvs), uvs, 0.0)
    groups = island_labels_from_uvs(mesh, safe) if within_islands else None
    bad = overlapping_faces(mesh, safe, substantial=0.0, min_region=1, face_groups=groups)
    signed = signed_uv_triangle_areas(mesh, safe)
    _, faces = surface_triangles(mesh)
    orientation = 1 if np.count_nonzero(signed > 0) >= np.count_nonzero(signed < 0) else -1
    bad[faces[signed * orientation <= 1e-14]] = True
    corner_faces = np.repeat(np.arange(mesh.num_faces), mesh.face_sizes)
    bad[corner_faces[~finite]] = True
    return bad
