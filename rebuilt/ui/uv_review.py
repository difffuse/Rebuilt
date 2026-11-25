"""Opt-in split UV inspection for the Manual workflow."""

import bpy
from ..bridge.image_refs import image_from_entry


def open_editor(ctx, context):
    from .operators import _show_only_the_asset

    window, screen = context.window, context.screen
    editor = next((a for a in screen.areas if a.type == "IMAGE_EDITOR" and a.ui_type == "UV"), None)
    if editor is None:
        before = {a.as_pointer() for a in screen.areas}
        region = next((r for r in context.area.regions if r.type == "WINDOW"), None)
        with context.temp_override(window=window, area=context.area, region=region):
            outcome = bpy.ops.screen.area_split(direction="VERTICAL", factor=0.58)
        added = [a for a in screen.areas if a.as_pointer() not in before]
        if outcome != {"FINISHED"} or not added:
            raise RuntimeError("Could not split this viewport. Exit maximized view and try again.")
        editor = added[0]
        editor.type = "IMAGE_EDITOR"
        editor.ui_type = "UV"

    low = ctx.artifact("low")
    if context.object is not None and context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    _show_only_the_asset(ctx)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    owned = {row.get('id') for row in ctx.resources.snapshot().values()
             if row['purpose'] == 'preview' and row.get('id') is not None}
    image = next((node.image for slot in low.material_slots
                  if slot.material in owned and slot.material.use_nodes
                  for node in slot.material.node_tree.nodes
                  if node.type == 'TEX_IMAGE' and node.image in owned), None)
    if image is None:
        entry = next(
            (
                m
                for m in (ctx.report("bake") or {}).get("maps", [])
                if m.get("map") == "BASE_COLOR" and m.get("baked")
            ),
            None,
        )
        image = image_from_entry(entry) if entry else None
    editor.spaces.active.image = image

    def frame():
        try:
            if (
                window.screen != screen
                or editor.type != "IMAGE_EDITOR"
                or editor.ui_type != "UV"
                or low.mode != "EDIT"
                or window.view_layer.objects.active != low
            ):
                return None
            region = next((r for r in editor.regions if r.type == "WINDOW"), None)
            if region is None:
                return None
            with bpy.context.temp_override(window=window, area=editor, region=region):
                bpy.ops.uv.select_all(action="SELECT")
                bpy.ops.image.view_selected()
            editor.tag_redraw()
        except (ReferenceError, RuntimeError):
            return None
        return None

    frame()

    if not bpy.app.background:
        bpy.app.timers.register(frame, first_interval=0.1)
    return editor
