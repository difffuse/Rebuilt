"""Solve Clean Seams' candidate charts on a disposable Blender mesh."""

import bmesh
import bpy
import math

from ..core.types import MeshArrays
from . import mesh_io, uv_io


def unwrap(payload):
    """Main thread only, with a private scene and dependency graph."""
    arrays = MeshArrays(
        **{
            k: payload[k]
            for k in (
                "positions",
                "corner_verts",
                "face_starts",
                "face_sizes",
            )
        }
    )
    scratch = mesh_io.arrays_to_object(arrays, "__rebuilt_seam_solver")
    mesh = scratch.data
    scene = None
    try:
        scene = bpy.data.scenes.new("__rebuilt_seam_solver")
        scene.collection.objects.link(scratch)
        view_layer = scene.view_layers[0]
        view_layer.update()
        view_layer.objects.active = scratch
        scratch.select_set(True, view_layer=view_layer)
        with bpy.context.temp_override(
            scene=scene,
            view_layer=view_layer,
            active_object=scratch,
            object=scratch,
            selected_objects=[scratch],
            selected_editable_objects=[scratch],
        ):
            uv_io.ensure_uv_layer(mesh)
            with uv_io.edit_only(scratch):
                with bpy.context.temp_override(
                    edit_object=scratch,
                    objects_in_mode=[scratch],
                    objects_in_mode_unique_data=[scratch],
                ):
                    scene.tool_settings.mesh_select_mode = (False, False, True)
                    bpy.ops.mesh.select_all(action="SELECT")
                    if payload["method"] == "SMART_PROJECT":
                        bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.0,
                                                 correct_aspect=False, scale_to_bounds=False)
                    else:
                        bm = bmesh.from_edit_mesh(mesh)
                        bm.faces.ensure_lookup_table()
                        layer = bm.loops.layers.uv.verify()
                        for face, start in zip(bm.faces, arrays.face_starts):
                            for offset, loop in enumerate(face.loops):
                                index = int(start) + offset
                                loop[layer].uv = payload["uvs"][index]
                                loop[layer].pin_uv = bool(payload["pins"][index])
                        bmesh.update_edit_mesh(mesh)
                        bpy.ops.uv.unwrap(method=payload["method"], margin=0.0, correct_aspect=False)
            return uv_io.read_uvs(mesh)
    finally:
        if scene is not None:
            bpy.data.scenes.remove(scene)
        bpy.data.objects.remove(scratch, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
