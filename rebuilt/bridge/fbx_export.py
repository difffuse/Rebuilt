"""Temporary, FBX-compatible copies of static assets."""

from contextlib import contextmanager
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix
from .image_refs import node_path


def _transform_mesh(mesh, transform):
    """Transport corner normals through scale and mirrored winding on a scratch mesh."""
    normals = np.empty((len(mesh.loops), 3), dtype=np.float32)
    mesh.corner_normals.foreach_get("vector", normals.ravel())
    normals = normals @ np.asarray(transform.to_3x3().inverted().transposed()).T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    layer = mesh.attributes.new("_rebuilt_export_normals", "FLOAT_VECTOR", "CORNER")
    layer_name = layer.name
    try:
        layer.data.foreach_set("vector", normals.ravel())
        mesh.transform(transform)
        if transform.determinant() < 0:
            mesh.flip_normals()
        # A corner attribute follows Blender's winding reversal, including n-gons.
        mesh.attributes[layer_name].data.foreach_get("vector", normals.ravel())
    finally:
        mesh.attributes.remove(mesh.attributes[layer_name])
    mesh.edges.foreach_set("use_edge_sharp", np.ones(len(mesh.edges), dtype=bool))
    mesh.update()
    mesh.normals_split_custom_set(normals)


def _triangle_mesh(source, transform=None):
    """Copy evaluated render triangles and their surface data before axis conversion."""
    source.calc_loop_triangles()
    triangles = list(source.loop_triangles)
    vertex_indices = np.asarray([triangle.vertices[:] for triangle in triangles], dtype=np.int32)
    loop_indices = np.asarray([triangle.loops[:] for triangle in triangles], dtype=np.int32)
    positions = np.asarray([vertex.co[:] for vertex in source.vertices], dtype=np.float32)
    polygons = np.asarray([triangle.polygon_index for triangle in triangles], dtype=np.int32)
    normals = np.empty((len(source.loops), 3), dtype=np.float32)
    source.corner_normals.foreach_get("vector", normals.ravel())
    if transform is not None:
        linear = transform.to_3x3()
        positions = positions @ np.asarray(linear).T
        normals = normals @ np.asarray(linear.inverted().transposed()).T
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
        if linear.determinant() < 0:
            vertex_indices = vertex_indices[:, (0, 2, 1)]
            loop_indices = loop_indices[:, (0, 2, 1)]
    loops = loop_indices.ravel()
    mesh = bpy.data.meshes.new(source.name + "_triangles")
    try:
        mesh.from_pydata(positions, [], vertex_indices)
        for material in source.materials:
            mesh.materials.append(material)
        for name, dtype in (("material_index", np.int32), ("use_smooth", bool)):
            values = np.empty(len(source.polygons), dtype=dtype)
            source.polygons.foreach_get(name, values)
            mesh.polygons.foreach_set(name, values[polygons])
        for layer in source.uv_layers:
            values = np.empty((len(source.loops), 2), dtype=np.float32)
            layer.data.foreach_get("uv", values.ravel())
            copied = mesh.uv_layers.new(name=layer.name)
            copied.data.foreach_set("uv", values[loops].ravel())
            copied.active_render = layer.active_render
        if source.uv_layers:
            mesh.uv_layers.active_index = source.uv_layers.active_index
        for layer in source.color_attributes:
            values = np.empty((len(layer.data), 4), dtype=np.float32)
            layer.data.foreach_get("color", values.ravel())
            copied = mesh.color_attributes.new(layer.name, layer.data_type, layer.domain)
            copied.data.foreach_set("color", (values[loops] if layer.domain == "CORNER" else values).ravel())
        if source.color_attributes:
            mesh.color_attributes.active_color_index = source.color_attributes.active_color_index
            mesh.color_attributes.render_color_index = source.color_attributes.render_color_index
        # Isolate encoding fans; otherwise changed connectivity can average custom normals.
        mesh.edges.foreach_set("use_edge_sharp", np.ones(len(mesh.edges), dtype=bool))
        mesh.update()
        mesh.normals_split_custom_set(normals[loops])
        return mesh
    except Exception:
        bpy.data.meshes.remove(mesh)
        raise


@contextmanager
def export_copy(low, image_paths=None, apply_transform=False, ensure_material=False, triangulate=False):
    """Keep FBX material simplification and transform baking off the live asset."""
    image_paths = image_paths or {}
    objects = bpy.data.objects
    mesh = None
    obj = None
    materials = []
    images = []
    try:
        evaluated = low.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = bpy.data.meshes.new_from_object(
            evaluated,
            preserve_all_data_layers=True,
            depsgraph=bpy.context.evaluated_depsgraph_get(),
        )
        if triangulate:
            triangulated = _triangle_mesh(
                mesh, low.matrix_world.to_3x3().to_4x4() if apply_transform else None
            )
            bpy.data.meshes.remove(mesh)
            mesh = triangulated
        obj = objects.new(low.name + "_export", mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.matrix_world = low.matrix_world.copy()
        if apply_transform:
            transform = obj.matrix_world.to_3x3().to_4x4()
            if not triangulate:
                _transform_mesh(mesh, transform)
            obj.matrix_world = Matrix.Translation(low.matrix_world.translation)
        for index, slot in enumerate(low.material_slots):
            original = slot.material
            if original is None:
                continue
            material = original.copy()
            materials.append(material)
            mesh.materials[index] = material
            if not material.use_nodes:
                continue
            tree = material.node_tree
            for node in tree.nodes:
                if node.type == "TEX_IMAGE" and node.image is not None:
                    path = node_path(image_paths, node)
                    if path:
                        if not Path(path).is_file():
                            raise RuntimeError(f"Export texture is missing: {path}")
                        image = bpy.data.images.load(str(path), check_existing=False)
                        images.append(image)
                        image.colorspace_settings.name = node.image.colorspace_settings.name
                        node.image = image
            for shader in (n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"):
                base = shader.inputs["Base Color"]
                if not base.is_linked:
                    continue
                mix = base.links[0].from_node
                if (
                    mix.type == "MIX"
                    and mix.data_type == "RGBA"
                    and mix.blend_type == "MULTIPLY"
                    and mix.label == "Base Color × Ambient Occlusion"
                ):
                    # FBX only recognizes a texture connected directly to Base Color.
                    tree.links.remove(base.links[0])
                    color = mix.inputs[6]
                    if color.is_linked:
                        tree.links.new(color.links[0].from_socket, base)
                    else:
                        base.default_value = color.default_value
                if base.is_linked and base.links[0].from_node.type == "TEX_IMAGE":
                    base.default_value = (1.0, 1.0, 1.0, 1.0)
        if ensure_material and not any(slot.material for slot in obj.material_slots):
            material = bpy.data.materials.new("Rebuilt Export Material")
            material.use_nodes = True
            materials.append(material)
            mesh.materials.clear()
            mesh.materials.append(material)
        yield obj
    finally:
        if obj is not None:
            objects.remove(obj, do_unlink=True)
        if mesh is not None:
            bpy.data.meshes.remove(mesh)
        for material in materials:
            bpy.data.materials.remove(material)
        for image in images:
            bpy.data.images.remove(image)
