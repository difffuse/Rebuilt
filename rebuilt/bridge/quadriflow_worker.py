"""Standalone Blender entry point for disposable QuadriFlow attempts."""
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def main():
    root = Path(sys.argv[sys.argv.index("--") + 1])
    options = json.loads((root / "options.json").read_text())
    with np.load(root / "input.npz", allow_pickle=False) as data:
        mesh = bpy.data.meshes.new("QuadriFlow input")
        mesh.vertices.add(len(data["positions"]))
        mesh.loops.add(len(data["corner_verts"]))
        mesh.polygons.add(len(data["face_sizes"]))
        mesh.vertices.foreach_set("co", data["positions"].astype(np.float32).ravel())
        mesh.loops.foreach_set("vertex_index", data["corner_verts"])
        mesh.polygons.foreach_set("loop_start", data["face_starts"])
        mesh.polygons.foreach_set("loop_total", data["face_sizes"])
        mesh.update(calc_edges=True)
        original_faces = len(mesh.polygons)
        repaired = mesh.validate(verbose=False)
        (root / "input_repair.json").write_text(json.dumps({
            "repaired": repaired, "removed_faces": original_faces - len(mesh.polygons),
        }))
        if not mesh.polygons:
            return
    for obj in list(bpy.context.selected_objects):
        obj.select_set(False)
    obj = bpy.data.objects.new("QuadriFlow input", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    request = float(options["target"])
    for attempt in range(options["passes"]):
        obj.data = mesh.copy()
        result = bpy.ops.object.quadriflow_remesh(
            mode="FACES", target_faces=max(round(request), 8),
            use_preserve_sharp=options["preserve_sharp"],
            use_preserve_boundary=options["preserve_boundary"],
            use_mesh_symmetry=False, seed=options["seed"],
        )
        if "FINISHED" not in result:
            return
        achieved = len(obj.data.polygons)
        if achieved <= 0 or (achieved - options["target"]) / options["target"] <= options["tolerance"]:
            break
        request *= options["target"] / achieved
    output = obj.data
    positions = np.empty((len(output.vertices), 3), np.float32)
    corners = np.empty(len(output.loops), np.int32)
    starts = np.empty(len(output.polygons), np.int32)
    sizes = np.empty(len(output.polygons), np.int32)
    output.vertices.foreach_get("co", positions.ravel())
    output.loops.foreach_get("vertex_index", corners)
    output.polygons.foreach_get("loop_start", starts)
    output.polygons.foreach_get("loop_total", sizes)
    np.savez(root / "output.npz", positions=positions, corner_verts=corners,
             face_starts=starts, face_sizes=sizes)


if __name__ == "__main__":
    main()
