"""Assemble generated presentation materials without stage or bake orchestration.

Operate on Blender's main thread. Preserve source shaders and shared meshes;
use the supplied asset context for image bindings and resource ownership.
"""
from __future__ import annotations

from typing import Any

import bpy

from . import materials
from .image_refs import image_from_entry
from ..core.bake.channels import EXTRA_CHANNELS
from ..pipeline.context import AssetContext, ContextError


MAP_SOCKETS: dict[str, str] = {
    "BASE_COLOR": "Base Color",
    "ROUGHNESS": "Roughness",
    "METALLIC": "Metallic",
    "OPACITY": "Alpha",
}
MAP_SOCKETS.update({name: item[1] for name, item in EXTRA_CHANNELS.items()})


MATERIAL_REVISION = 5


KEY_GENERATED_MATERIAL = 'rebuilt_generated_material'


def local_output_mesh(low, *, ctx=None):
    """Only the output receives delivery transforms and generated material slots."""
    operation = getattr(ctx, '_stage_operation', None)
    if operation is not None and operation.record.get('low') == low:
        operation.require_private_mesh(low)
        return
    if low.data.users > 1 or not low.data.is_editable:
        low.data = low.data.copy()


def assemble_material(
    low: bpy.types.Object, maps: list[dict[str, Any]], *, ctx=None, clear_preview=True
) -> bpy.types.Material | None:
    """Wire the baked maps into a Principled BSDF so the asset looks like itself."""
    if ctx is None:
        ctx = AssetContext.for_object(bpy.context.scene, low)
    if ctx is not None:
        maps = ctx.resolve_maps(maps, required=True)
    usable = [m for m in maps if m.get("baked") and image_from_entry(m)]
    if not usable:
        return None

    if not low.is_editable:
        raise ContextError('The output object is readonly.')
    if ctx is not None:
        ctx.require_owned()
        if ctx.artifact('low') != low:
            raise ContextError('Material assembly requires the owned output object.')

    # A label is not permission to clear an existing shader. Build off-object,
    # then attach the completed shader without changing any shared mesh slots.
    name = f"{low.name}_mat"
    material = (ctx.resources.create('output', bpy.data.materials.new, name)
                if ctx is not None else bpy.data.materials.new(name))
    try:
        _configure_material(material, low, usable, ctx)
        local_output_mesh(low, ctx=ctx)
        low.data.materials.clear()
        low.data.materials.append(material)
        low[KEY_GENERATED_MATERIAL] = material
        from ..pipeline.map_images import bind
        bind(low, maps)
    except Exception:
        if ctx is not None:
            ctx.resources.release(ids={material})
        elif material.users == 0:
            bpy.data.materials.remove(material)
        raise
    if bpy.data.materials.get(name) is None:
        material.name = name
    from . import uv_preview

    if ctx is not None and clear_preview and ctx._stage_operation is None:
        uv_preview.clear(ctx)
    return material


def _configure_material(material, low, usable, ctx):
    material.use_nodes = True
    if low.get("rebuilt_asset"):
        material["rebuilt_asset"] = low["rebuilt_asset"]
    tree = material.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (400, 0)
    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.location = (100, 0)
    from .bake_maps import material_defaults

    if ctx is not None:
        source = ctx.artifact("working") or ctx.source_object
        defaults = material_defaults(source)
        if not defaults:
            defaults = (ctx.report("bake") or {}).get("material_defaults", {})
        for name, value in defaults.items():
            if name in principled.inputs:
                principled.inputs[name].default_value = value
    tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    textures = {}
    for index, entry in enumerate(usable):
        image = image_from_entry(entry)
        texture = tree.nodes.new("ShaderNodeTexImage")
        texture.image = image
        texture['rebuilt_map_channel'] = entry['map']
        if 'image_ref' in entry:
            from ..pipeline.map_images import KEY_NODE_REF
            texture[KEY_NODE_REF] = entry['image_ref']
        texture.location = (-700, -300 * index)
        texture.label = entry["map"].replace("_", " ").title()
        textures[entry["map"]] = texture

        if entry["map"] == "NORMAL":
            normal_map = tree.nodes.new("ShaderNodeNormalMap")
            normal_map.location = (-400, -300 * index)
            tree.links.new(texture.outputs["Color"], normal_map.inputs["Color"])
            tree.links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])
        elif entry["map"] in MAP_SOCKETS:
            socket = principled.inputs.get(MAP_SOCKETS[entry["map"]])
            if socket is not None:
                tree.links.new(texture.outputs["Color"], socket)

    if "AO" in textures:
        multiply = tree.nodes.new("ShaderNodeMix")
        multiply.data_type = "RGBA"
        multiply.blend_type = "MULTIPLY"
        multiply.label = "Base Color × Ambient Occlusion"
        multiply.location = (-180, 160)
        multiply.inputs[0].default_value = 1.0
        if "BASE_COLOR" in textures:
            tree.links.new(textures["BASE_COLOR"].outputs["Color"], multiply.inputs[6])
        else:
            multiply.inputs[6].default_value = principled.inputs["Base Color"].default_value
        tree.links.new(textures["AO"].outputs["Color"], multiply.inputs[7])
        tree.links.new(multiply.outputs[2], principled.inputs["Base Color"])

    if ctx is not None and "EMISSION" in textures and "EMISSION_STRENGTH" not in textures:
        strengths = []
        for slot in ctx.source_object.material_slots:
            color = materials.find_colour_source(slot.material) if slot.material else None
            if color is not None and color.node.type == "EMISSION":
                strength = color.node.inputs["Strength"]
                if not strength.is_linked:
                    strengths.append(strength.default_value)
        if len(strengths) == len(ctx.source_object.material_slots) and len(set(strengths)) == 1:
            principled.inputs["Emission Strength"].default_value = strengths[0]
