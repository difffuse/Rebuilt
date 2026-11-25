"""Reading and writing UV data, and marking seams."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import bpy
import numpy as np

from ..core.analysis.mesh_graph import edge_keys

UV_LAYER = "UVMap"


@contextmanager
def edit_only(obj: bpy.types.Object) -> Iterator[None]:
    """Put exactly one object into edit mode, and put the scene back afterwards."""
    view_layer = bpy.context.view_layer
    previous_active = view_layer.objects.active
    previous_selection = [other for other in view_layer.objects if other.select_get()]

    if previous_active is not None and previous_active.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    mesh = obj.data
    selection_mode = tuple(bpy.context.tool_settings.mesh_select_mode)
    active_face = mesh.polygons.active
    mesh_selection = []
    for items in (mesh.vertices, mesh.edges, mesh.polygons):
        flags = np.empty(len(items), dtype=bool)
        items.foreach_get("select", flags)
        mesh_selection.append(flags)

    for other in view_layer.objects:
        other.select_set(False)
    obj.select_set(True)
    view_layer.objects.active = obj

    try:
        bpy.ops.object.mode_set(mode="EDIT")
        yield
    finally:
        if obj.mode == "EDIT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.context.tool_settings.mesh_select_mode = selection_mode
        items_by_domain = (mesh.vertices, mesh.edges, mesh.polygons)
        same_counts = all(len(items) == len(flags)
                          for items, flags in zip(items_by_domain, mesh_selection))
        for items, flags in zip(items_by_domain, mesh_selection):
            # Conversion fallback creates new topology; old indices no longer apply.
            items.foreach_set("select", flags if same_counts else np.zeros(len(items), dtype=bool))
        mesh.polygons.active = active_face if same_counts else -1
        mesh.update()
        for other in view_layer.objects:
            other.select_set(False)
        for other in previous_selection:
            try:
                other.select_set(True)
            except ReferenceError:
                pass
        try:
            view_layer.objects.active = previous_active
        except ReferenceError:
            view_layer.objects.active = None


def read_uvs(mesh: bpy.types.Mesh, layer_name: str | None = None) -> np.ndarray:
    """UV coordinates per face corner. Shape (C, 2)."""
    layer = mesh.uv_layers.get(layer_name) if layer_name else mesh.uv_layers.active
    if layer is None:
        raise KeyError(f"mesh {mesh.name!r} has no UV layer {layer_name or '(active)'}")

    flat = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    layer.uv.foreach_get("vector", flat)
    return flat.reshape(-1, 2).astype(np.float64)


def write_uvs(mesh: bpy.types.Mesh, uvs: np.ndarray, layer_name: str | None = None) -> None:
    """Overwrite a UV layer from an (C, 2) array."""
    layer = mesh.uv_layers.get(layer_name) if layer_name else mesh.uv_layers.active
    if layer is None:
        raise KeyError(f"mesh {mesh.name!r} has no UV layer {layer_name or '(active)'}")
    if uvs.shape != (len(mesh.loops), 2):
        raise ValueError(f"expected {(len(mesh.loops), 2)} UV coordinates, got {uvs.shape}")

    layer.uv.foreach_set("vector", uvs.astype(np.float32).ravel())
    mesh.update()


def ensure_uv_layer(mesh: bpy.types.Mesh, name: str = UV_LAYER) -> bpy.types.MeshUVLoopLayer:
    """Get or create a UV layer and make it the only one, for rendering too."""
    layer = mesh.uv_layers.get(name)
    if layer is None:
        layer = mesh.uv_layers.new(name=name)

    doomed = [other.name for other in mesh.uv_layers if other.name != layer.name]
    for other in doomed:
        existing = mesh.uv_layers.get(other)
        if existing is not None:
            mesh.uv_layers.remove(existing)

    layer = mesh.uv_layers[name]
    mesh.uv_layers.active = layer
    layer.active_render = True
    return layer


def read_pins(mesh, uv=None):
    """Capture constraints from actual stored coordinates, not just pin flags."""
    uv = read_uvs(mesh) if uv is None else uv
    pins = {i:uv[i].copy() for i, loop in enumerate(mesh.uv_layers.active.data) if loop.pin_uv}
    if any(not np.isfinite(value).all() for value in pins.values()):
        raise ValueError('Pinned UV coordinates must be finite.')
    return pins


def check_pins(mesh, pins):
    uv = read_uvs(mesh)
    for corner, value in pins.items():
        if not mesh.uv_layers.active.data[corner].pin_uv or not np.array_equal(uv[corner], value):
            raise ValueError('UV validation failed: an original pin moved or was removed.')


def clear_seams(mesh: bpy.types.Mesh) -> None:
    """Unmark every seam."""
    if len(mesh.edges) == 0:
        return
    mesh.edges.foreach_set("use_seam", np.zeros(len(mesh.edges), dtype=bool))
    mesh.update()


def mark_seams(mesh: bpy.types.Mesh, edge_pairs: np.ndarray) -> int:
    """Mark the given vertex-index pairs as seams. Returns how many matched."""
    edge_count = len(mesh.edges)
    if edge_count == 0 or edge_pairs.shape[0] == 0:
        return 0

    existing = np.empty(edge_count * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", existing)
    existing = existing.reshape(-1, 2).astype(np.int64)

    vertex_count = len(mesh.vertices)
    lookup = {int(key): index for index, key in enumerate(edge_keys(existing, vertex_count))}

    flags = np.zeros(edge_count, dtype=bool)
    matched = 0
    for key in edge_keys(edge_pairs.astype(np.int64), vertex_count):
        index = lookup.get(int(key))
        if index is not None:
            flags[index] = True
            matched += 1

    mesh.edges.foreach_set("use_seam", flags)
    mesh.update()
    return matched


def unwrap(
    obj: bpy.types.Object,
    method: str = "ANGLE_BASED",
    margin: float = 0.0,
    weight_group: str = "",
) -> None:
    """Unwrap the whole mesh along its current seams."""
    with edit_only(obj):
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.unwrap(
            method=method,
            margin_method="FRACTION",
            margin=margin,
            correct_aspect=True,
            use_weights=bool(weight_group),
            weight_group=weight_group or "uv_importance",
            weight_factor=1.0,
        )


def smart_project(
    obj: bpy.types.Object, angle_limit_deg: float = 66.0, faces: np.ndarray | None = None
) -> None:
    """Blender's own chart generator, used as the fallback."""
    mesh = obj.data
    mask = None
    if faces is not None:
        mask = np.asarray(faces, dtype=bool)
        if mask.shape[0] != len(mesh.polygons):
            raise ValueError("face mask does not match the mesh")
    with edit_only(obj):
        if mask is None:
            bpy.ops.mesh.select_all(action="SELECT")
        else:
            import bmesh

            bpy.context.tool_settings.mesh_select_mode = (False, False, True)
            bm = bmesh.from_edit_mesh(mesh)
            bm.faces.ensure_lookup_table()
            for face in bm.faces:
                face.select = bool(mask[face.index])
            bm.select_flush_mode()
            bmesh.update_edit_mesh(mesh)
        bpy.ops.uv.smart_project(
            angle_limit=np.radians(angle_limit_deg),
            margin_method="FRACTION",
            island_margin=0.0,
            correct_aspect=True,
            scale_to_bounds=False,
        )


def average_islands_scale(obj: bpy.types.Object) -> None:
    """Give every island the same texel density before packing."""
    with edit_only(obj):
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.select_all(action="SELECT")
        bpy.ops.uv.average_islands_scale()





def pack_islands(
    obj: bpy.types.Object,
    padding_px: int,
    texture_size: int,
    rotate_method: str = "ANY",
    shape_method: str = "CONCAVE",
    allow_scaling: bool = True,
    packer: str = "uvpack",
    search_seconds: float = 0.0,
    measurements: dict | None = None,
    *, arrays=None, chart_ids=None, density_target=None, equalize=True,
    deadline=None, cancelled=None,
) -> str:
    """Run exactly the selected packer and certify actual stored footprints."""
    from . import mesh_io
    from .engines.uvpack import pack_object, UVPackClient, uv_fingerprint
    if measurements is not None:
        measurements.clear()
    arrays = arrays if arrays is not None else mesh_io.mesh_to_arrays(obj.data)
    if packer == 'uvpack':
        report = pack_object(obj, arrays, padding_px=padding_px, texture_size=texture_size,
            rotate_method=rotate_method, allow_scaling=allow_scaling, search_seconds=search_seconds,
            chart_ids=chart_ids, density_target=density_target, equalize=equalize,
            deadline=deadline, cancelled=cancelled)
    elif packer == 'blender':
        client = UVPackClient(deadline=deadline, cancelled=cancelled)
        before = read_uvs(obj.data)
        pins = read_pins(obj.data, before)
        extracted = client.extract(arrays, before, chart_ids)
        labels = client.labels(arrays, extracted)
        held = client.fixed_ids(extracted, pins)
        normalization = None
        try:
            if (allow_scaling and equalize) or density_target is not None:
                normalized, labels, normalization = client.normalize(arrays, before,
                    texture_size=texture_size, chart_ids=labels, equalize=equalize and allow_scaling,
                    density_target=density_target, fixed_corners=pins)
                write_uvs(obj.data, normalized)
                if not np.array_equal(read_uvs(obj.data), normalized):
                    raise ValueError('UV normalization stored readback differs from its proposal.')
            if held:
                client.validate(arrays, read_uvs(obj.data), labels, padding_px, texture_size, only_ids=held)
            if len(held) != len(extracted.islands):
                _pack_with_blender(obj, padding_px, texture_size, rotate_method, shape_method, allow_scaling)
            stored = read_uvs(obj.data)
            for item in extracted.islands:
                if item['id'] in held and not np.array_equal(stored[item['corners']], before[item['corners']]):
                    raise ValueError('Blender packing moved a fixed chart.')
            check_pins(obj.data, pins)
            certificate = client.validate(arrays, read_uvs(obj.data), labels, padding_px, texture_size)
            client.check_cancelled()
        except BaseException:
            write_uvs(obj.data, before)
            raise
        report = {'packing_uv_sha256':uv_fingerprint(read_uvs(obj.data)),
                  'packing_validation':certificate, 'packing_runtime':client.runtime.checked.diagnostic(),
                  'normalization':normalization, 'held_fixed_ids':sorted(held)}
    else:
        raise ValueError(f'Unknown UV packer: {packer}')
    if measurements is not None:
        measurements.update(report)
    return packer


def _pack_with_blender(obj, padding_px, texture_size, rotate_method, shape_method, allow_scaling):
    with edit_only(obj):
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.select_all(action="SELECT")
        bpy.ops.uv.pack_islands(
            udim_source="CLOSEST_UDIM",
            rotate=rotate_method != "AXIS_ALIGNED",
            rotate_method=rotate_method,
            scale=allow_scaling,
            pin=True,
            pin_method='LOCKED',
            merge_overlap=False,
            margin_method="FRACTION",
            margin=padding_px / max(texture_size, 1),
            shape_method=shape_method,
        )


def uvpack_binary():
    """Resolve the manifest-owned executable; missing runtimes are errors."""
    from ..runtime.manifest import resolve_engine
    return resolve_engine('uvpack').entrypoints['cli']


def pack_islands_uvpack(obj, padding_px, texture_size, rotate_method="ANY",
                         allow_scaling=True, search_seconds=0.0):
    """Compatibility entrypoint with the same explicit native failure policy."""
    pack_islands(obj, padding_px, texture_size, rotate_method=rotate_method,
                 allow_scaling=allow_scaling, search_seconds=search_seconds, packer='uvpack')
    return True
