"""Thread-safe polygon tessellation for native remesher input."""

import numpy as np
from mathutils.geometry import tessellate_polygon

from ..core.types import MeshArrays


def remesher_triangles(mesh: MeshArrays) -> np.ndarray:
    """Tessellate polygon interiors without changing source vertices or winding."""
    triangles = np.empty((int((mesh.face_sizes - 2).sum()), 3), dtype=np.int32)
    offset = 0
    for start, size in zip(mesh.face_starts, mesh.face_sizes):
        face = mesh.corner_verts[start : start + size]
        if size == 3:
            triangles[offset] = face
            offset += 1
            continue
        local = tessellate_polygon([mesh.positions[face].tolist()])
        count = len(local)
        if count:
            triangles[offset : offset + count] = face[np.asarray(local, dtype=np.int32)]
            offset += count
    return triangles[:offset]
