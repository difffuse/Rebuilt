"""Moving mesh data between Blender and numpy."""

from __future__ import annotations

import bpy
import numpy as np
from mathutils import Matrix

from ..core.types import MeshArrays


def mesh_to_arrays(mesh: bpy.types.Mesh, matrix: Matrix | None = None) -> MeshArrays:
    """Extract a Blender mesh into flat numpy arrays."""
    num_verts = len(mesh.vertices)
    num_corners = len(mesh.loops)
    num_faces = len(mesh.polygons)

    positions = np.empty(num_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", positions)
    positions = positions.reshape(num_verts, 3)

    corner_verts = np.empty(num_corners, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", corner_verts)

    face_starts = np.empty(num_faces, dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", face_starts)

    face_sizes = np.empty(num_faces, dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", face_sizes)

    if matrix is not None:
        positions = transform_positions(positions, matrix)

    return MeshArrays(
        positions=positions,
        corner_verts=corner_verts,
        face_starts=face_starts,
        face_sizes=face_sizes,
    )


def object_to_arrays(
    obj: bpy.types.Object,
    depsgraph: bpy.types.Depsgraph | None = None,
    *,
    apply_modifiers: bool = False,
    world_space: bool = False,
) -> MeshArrays:
    """Extract a mesh object, optionally as evaluated and/or in world space."""
    if obj.type != "MESH":
        raise TypeError(f"{obj.name!r} is a {obj.type}, not a mesh")

    matrix = obj.matrix_world if world_space else None

    if not apply_modifiers:
        return mesh_to_arrays(obj.data, matrix)

    if depsgraph is None:
        raise ValueError("apply_modifiers needs a depsgraph")

    evaluated = obj.evaluated_get(depsgraph)
    temp_mesh = evaluated.to_mesh()
    try:
        return mesh_to_arrays(temp_mesh, matrix)
    finally:
        evaluated.to_mesh_clear()


def arrays_to_mesh(arrays: MeshArrays, name: str) -> bpy.types.Mesh:
    """Build a new mesh datablock from numpy arrays."""
    arrays.validate()

    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(arrays.num_verts)
    mesh.loops.add(arrays.num_corners)
    mesh.polygons.add(arrays.num_faces)

    mesh.vertices.foreach_set("co", arrays.positions.astype(np.float32).ravel())
    mesh.loops.foreach_set("vertex_index", arrays.corner_verts.astype(np.int32))
    mesh.polygons.foreach_set("loop_start", arrays.face_starts.astype(np.int32))
    mesh.polygons.foreach_set("loop_total", arrays.face_sizes.astype(np.int32))

    mesh.update(calc_edges=True)

    if mesh.validate(verbose=False):
        bpy.data.meshes.remove(mesh)
        raise ValueError(
            f"the arrays for {name!r} did not describe a valid mesh; Blender had to repair it"
        )

    return mesh


def arrays_to_object(arrays: MeshArrays, name: str) -> bpy.types.Object:
    """Build a new, unlinked object from numpy arrays."""
    return bpy.data.objects.new(name, arrays_to_mesh(arrays, name))


def transform_positions(positions: np.ndarray, matrix: Matrix) -> np.ndarray:
    """Apply a 4x4 transform to an (N, 3) array of points."""
    m = np.array(matrix, dtype=np.float32)
    return positions @ m[:3, :3].T + m[:3, 3]


def replace_object_mesh(obj: bpy.types.Object, arrays: MeshArrays) -> None:
    """Swap an object's mesh for one built from `arrays`, freeing the old one."""
    old = obj.data
    new = arrays_to_mesh(arrays, name=old.name)
    obj.data = new
    if old.users == 0:
        bpy.data.meshes.remove(old)
