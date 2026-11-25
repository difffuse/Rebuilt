"""Temporary Manual UV inspection; never part of the baked material or export."""

from __future__ import annotations

import bpy

PREVIEW_TAG = "rebuilt_uv_preview"


def clear(ctx, *, keep=()) -> None:
    """Detach this asset's exact previews; preserve artist copies and shared IDs."""
    ctx._ownership.ensure()
    rows = ctx.resources.snapshot()
    ids = {row.get('id') for row in rows.values()
           if row['purpose'] == 'preview' and row.get('id') not in keep}
    materials = {value for value in ids if isinstance(value, bpy.types.Material)}
    for obj in ctx._ownership.objects():
        if obj.type != 'MESH' or not any(slot.material in materials for slot in obj.material_slots):
            continue
        if obj.data.users > 1 or obj.data.library is not None:
            obj.data = obj.data.copy()
        for slot in obj.material_slots:
            if slot.material in materials:
                slot.material = None
    ctx.resources.release(ids=ids, purpose='preview')


def show(low: bpy.types.Object, asset: str) -> None:
    """Display a UV grid using the active UV layer, without editing it."""
    if low.data.uv_layers.active is None:
        return
    from ..pipeline.context import AssetContext, ContextError
    ctx = AssetContext.for_object(bpy.context.scene, low)
    if ctx is None or ctx.name != asset or ctx.artifact('low') != low:
        raise ContextError('UV preview requires the owned low mesh.')
    ctx.require_owned()
    owned = {row.get('id') for row in ctx.resources.snapshot().values()
             if row['purpose'] == 'preview' and row.get('id') is not None}
    material = low.active_material
    uv_name = low.data.uv_layers.active.name
    if (material in owned and _matches_preview(material, owned, uv_name)
            and all(slot.material == material for slot in low.material_slots)):
        return
    material, image = _new_preview(ctx, asset, uv_name)
    if low.data.users > 1 or low.data.library is not None:
        low.data = low.data.copy()
    if not low.material_slots:
        low.data.materials.append(material)
    else:
        for slot in low.material_slots:
            slot.material = material
    clear(ctx, keep={material, image})


def _matches_preview(material, owned, uv_name):
    if not material.use_nodes:
        return False
    tree = material.node_tree
    nodes = {node.type: node for node in tree.nodes}
    if len(tree.nodes) != 4 or set(nodes) != {'UVMAP', 'TEX_IMAGE', 'EMISSION', 'OUTPUT_MATERIAL'}:
        return False
    uv, texture, emission, output = (nodes[k] for k in ('UVMAP', 'TEX_IMAGE', 'EMISSION', 'OUTPUT_MATERIAL'))
    image = texture.image
    if (uv.uv_map != uv_name or image not in owned or image.generated_type != 'UV_GRID'
            or tuple(image.size) != (1024, 1024) or any(node.mute for node in tree.nodes)
            or emission.inputs['Strength'].default_value != 1.0):
        return False
    expected = {(uv.outputs['UV'], texture.inputs['Vector']),
                (texture.outputs['Color'], emission.inputs['Color']),
                (emission.outputs[0], output.inputs['Surface'])}
    return {(link.from_socket, link.to_socket) for link in tree.links} == expected


def _new_preview(ctx, asset, uv_name):
    created = []
    try:
        image = ctx.resources.create('preview', bpy.data.images.new, f"{asset}_UV_Grid_Preview", width=1024, height=1024)
        created.append(image)
        image[PREVIEW_TAG] = True
        image['rebuilt_asset'] = asset
        image.generated_type = 'UV_GRID'
        material = ctx.resources.create('preview', bpy.data.materials.new, f"{asset}_UV_Grid_Preview")
        created.append(material)
        _configure_preview(material, image, asset, uv_name)
        return material, image
    except Exception:
        ctx.resources.release(ids=set(created))
        raise


def _configure_preview(material, image, asset, uv_name):
    material[PREVIEW_TAG] = True
    material["rebuilt_asset"] = asset
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    uv = tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = uv_name
    uv.location = (-600, 0)
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.location = (-400, 0)
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.location = (-100, 0)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (100, 0)
    tree.links.new(uv.outputs["UV"], texture.inputs["Vector"])
    tree.links.new(texture.outputs["Color"], emission.inputs["Color"])
    tree.links.new(emission.outputs[0], output.inputs["Surface"])
    tree.nodes.active = texture
