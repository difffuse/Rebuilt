"""N-panel entry points for Rebuilt."""

from functools import lru_cache
import json
from pathlib import Path

import bpy

from ..core import profiles
from ..core.analysis.defects import DefectReport
from ..pipeline.context import KEY_ASSET, AssetContext, ContextError
from . import icons

CATEGORY = "Rebuilt"


_THIN_NOTICE_SHARE = 0.10


@lru_cache(maxsize=1)
def help_links():
    path = Path(__file__).resolve().parents[1] / "resources" / "help-links.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _section(layout, settings, prop, title):
    box = layout.box()
    header = box.row(align=True)
    header.prop(
        settings,
        prop,
        text="",
        emboss=False,
        icon="TRIA_DOWN" if getattr(settings, prop) else "TRIA_RIGHT",
    )
    header.prop(settings, prop, text=title, emboss=False)
    return box.column(align=True) if getattr(settings, prop) else None


def _wrap(text: str, width: int = 46) -> list[str]:
    """Wrap text to fit the sidebar."""
    import textwrap

    return textwrap.wrap(text, width=width) or [""]


def _diagnostic_lines(text, context, width=400):
    """Fit warning text to both the sidebar and the home panel's popup width."""
    import blf
    scale = context.preferences.system.ui_scale
    available = max(80, min(context.region.width, width * scale) - 48 * scale)
    blf.size(0, context.preferences.ui_styles[0].widget.points * scale)
    line = ''
    for word in text.split():
        candidate = (line + ' ' + word).lstrip()
        if line and blf.dimensions(0, candidate)[0] > available:
            yield line
            line = ''
        for char in ((' ' if line else '') + word):
            if line and blf.dimensions(0, line + char)[0] > available:
                yield line
                line = ''
            line += char
    if line:
        yield line


def thin_advice(settings, ctx) -> tuple[str, list[str]] | None:
    """Optional thickness guidance from the existing source diagnosis."""
    if ctx is None or settings.topology != "quads":
        return None
    report = ctx.report("diagnosis")
    if not report or not report.get("thickness_quantiles"):
        return None
    from ..bridge.diagnosis import thickness_array
    from ..core.analysis import thickness

    verdict = thickness.judge(
        thickness_array(report), float(report["area"]), int(settings.target_quads)
    )
    if not verdict.warns:
        return None
    return ("INFO", _wrap(thin_message(verdict, int(settings.target_quads)), 44))


def thin_message(verdict, quads: int) -> str:
    """A uniform-cell estimate is a review hint, not a predicted failure."""
    return (
        f"Thickness estimate at {quads:,} quads: about {verdict.share:.0%} of the "
        "surface may need a closer look. This estimate assumes uniform quads; "
        "the remesher can preserve finer features. Review the result first. "
        "Increase the budget only if important details are missing."
    )


def _draw_thin_advice(layout, settings, ctx) -> None:

    if ctx is None or settings.topology != "quads":
        return
    report = ctx.report("diagnosis")
    if not report or not report.get("thickness_quantiles"):
        return
    from ..bridge.diagnosis import thickness_array
    from ..core.analysis import thickness

    verdict = thickness.judge(
        thickness_array(report), float(report["area"]), int(settings.target_quads)
    )
    if verdict.share >= _THIN_NOTICE_SHARE:
        notice = layout.box()
        notice.label(text="Quad budget: thin-detail risk", icon="INFO")
        message = (
            "Thin details are at higher risk. "
            if verdict.severe
            else "Some thin details may lose shape. "
        ) + "Review the result before adding quads."
        for line in _wrap(message, 44):
            notice.label(text=line)


def context_resolution(context, name=None, *, readonly=False):
    """Return the selected context and a diagnostic without throwing in UI draw/poll."""
    ctx = None
    try:
        obj = context.active_object
        if obj is not None and obj.get('rebuilt_flow_guide'):
            obj = obj.parent
        ctx = (AssetContext.find(context.scene, name)
               if name and (obj is None or obj.get(KEY_ASSET) != name)
               else AssetContext.for_object(context.scene, obj))
        if ctx is not None:
            ctx.require_owned(readonly=readonly)
        return ctx, None
    except ContextError as error:
        return ctx, str(error)


def active_context(context: bpy.types.Context) -> AssetContext | None:
    ctx, diagnostic = context_resolution(context)
    return None if diagnostic else ctx


def export_context(context):
    from ..pipeline.export_context import ExportContext

    obj = context.active_object
    name = obj.get(KEY_ASSET) if obj else None
    return ExportContext.find(context.scene, name, obj) if name else None


def _draw_image_settings(layout, settings, *, include_resolution=False):
    header = layout.row(align=True)
    header.prop(
        settings,
        "show_texture_output",
        text="Texture Output",
        emboss=False,
        icon="TRIA_DOWN" if settings.show_texture_output else "TRIA_RIGHT",
    )
    header.operator(
        "rebuilt.texture_output_info", text="", icon="INFO", emboss=False
    ).image_format = settings.bake_image_format
    if not settings.show_texture_output:
        return
    column = layout.column(align=True)
    column.use_property_split = True
    if include_resolution:
        column.prop(settings, "texture_resolution", text="Resolution")
    column.prop(settings, "bake_image_format")
    if settings.bake_image_format == "OPEN_EXR":
        column.prop(settings, "bake_exr_depth")
        column.prop(settings, "bake_exr_codec")
    elif settings.bake_image_format == "JPEG":
        column.prop(settings, "bake_jpeg_quality", slider=True)
    else:
        column.prop(settings, "bake_image_depth")
        column.prop(
            settings,
            "bake_png_compression" if settings.bake_image_format == "PNG" else "bake_tiff_codec",
        )


class REBUILT_PT_home(bpy.types.Panel):
    bl_idname = "REBUILT_PT_home"
    bl_label = "Rebuilt"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY
    bl_ui_units_x = 20

    def draw_header(self, context):
        self.layout.label(text="", icon_value=icons.header_icon())

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout

        layout.use_property_decorate = False
        summary = layout.column(align=True)
        summary.enabled = False
        summary.label(text="Retopo · UV · Bake")
        settings = context.scene.rebuilt
        obj = context.active_object
        if obj is not None and obj.get("rebuilt_flow_guide"):
            obj = obj.parent
        if obj is None or obj.type != "MESH" or not context.active_object.select_get():
            layout.label(text="Select a mesh model to begin", icon="OUTLINER_OB_MESH")
            return
        asset_name = obj.get(KEY_ASSET, "")

        layout.row(align=True).prop(settings, "mode", expand=True)

        ctx, diagnostic = context_resolution(context, readonly=True)
        if diagnostic:
            notice = layout.box()
            notice.label(text='Asset needs review', icon='ERROR')
            for line in _diagnostic_lines(diagnostic, context):
                notice.label(text=line)
            from .operators import REBUILT_OT_recover_asset
            if REBUILT_OT_recover_asset.poll(context):
                notice.operator('rebuilt.recover_asset', text='Recover Interrupted Build')
            else:
                notice.operator('rebuilt.adopt_results', text='Review Existing Results')
                notice.operator('rebuilt.adopt_scan', text='Use Selected Model')
            return

        if context.mode != "OBJECT":
            layout.label(text="Switch to Object Mode to start retopology", icon="INFO")

        if not bpy.data.filepath:
            unsaved = layout.box()
            unsaved.label(text="Save the .blend before generating", icon="ERROR")
            unsaved.label(text="The baked maps are written beside it.")
            unsaved.operator("wm.save_mainfile", text="Save", icon="FILE_TICK")

        if settings.mode == "auto":
            self._draw_auto(layout, settings, asset_name, ctx)
        else:
            self._draw_stages(layout, asset_name, ctx)
            if ctx is not None and ctx.can_finish_here():
                from .operators import REBUILT_OT_finish_here

                if REBUILT_OT_finish_here.poll(context):
                    row = layout.row()
                    row.alignment = "RIGHT"
                    row.operator("rebuilt.finish_here", text="Finish Here")

        self._draw_asset_box(layout, settings, asset_name or obj.name, ctx)

    def _draw_auto(self, layout, settings, asset_name: str, ctx) -> None:
        """Draw model and texture settings and the Rebuild Asset action."""
        section = _section(layout, settings, "show_asset_settings", "Model Settings")
        if section is not None:
            column = section.column(align=True)
            column.use_property_split = True
            column.prop(settings, "profile", text="Type")
            column.prop(settings, "topology", text="Topology")
            column.prop(
                settings,
                "target_quads",
                text="Quads" if settings.topology == "quads" else "Triangles",
            )
            if settings.topology == "quads":
                column.row(align=True).prop(settings, "quad_budget_mode", expand=True)
        _draw_image_settings(layout.box(), settings, include_resolution=True)
        _draw_thin_advice(layout, settings, ctx)

        run = layout.column(align=True)
        run.scale_y = 1.8
        run.enabled = bool(bpy.data.filepath) and bpy.context.mode == "OBJECT"

        generate_operator = run.operator("rebuilt.run_pipeline", text="Rebuild Asset", icon="PLAY")
        generate_operator.asset_name = asset_name
        generate_operator.only_stage = ""
        generate_operator.force = False

    def _draw_asset_box(self, layout, settings, asset_name: str, ctx) -> None:
        """Show the last build and the settings for the next run."""
        box = layout.box()
        content = box.column(align=True)
        heading = content.row()
        heading.label(text=asset_name, icon="OUTLINER_OB_MESH")
        recipe = ctx.recipe if ctx else None
        cached = ctx.cached_keys if ctx else {}
        has_retopo = "retopo" in cached
        completed = "output" in cached and ctx.report("output") is not None
        status = content.row()
        status.enabled = False
        status.label(
            text="Completed" if completed else "In progress" if cached else "Ready to build",
            icon="CHECKMARK" if completed else "TIME" if cached else "DOT",
        )
        quads, texture_size_px = settings.planned_summary()
        planned = (
            settings.profile,
            settings.topology,
            quads,
            settings.quad_budget_mode,
            texture_size_px,
        )
        built = (
            recipe.profile,
            recipe.retopo.topology,
            recipe.retopo.target_faces,
            recipe.retopo.quad_budget_mode,
            recipe.texture_resolution,
        ) if recipe else planned

        def draw_summary(values, report=None):
            profile, topology, count, budget, resolution = values
            unit = "quads" if topology == "quads" else "triangles"
            if report:
                count = report.get(
                    "achieved_quads" if topology == "quads" else "achieved_faces", count
                )
            available = profiles.available()
            label = (
                available[profile].label
                if profile in available
                else profile.replace("_", " ").title()
            )
            rows = content.column(align=True)
            for title, value, icon in (
                ("Asset Type", label, "OBJECT_DATA"),
                ("Geometry", f"{count:,} {unit}", "MESH_GRID"),
                ("Budget", budget.title() if topology == "quads" else "Target", "PREFERENCES"),
                ("Textures", f"{resolution:,} × {resolution:,} px", "IMAGE_DATA"),
            ):
                row = rows.split(factor=0.36)
                row.label(text=title, icon=icon)
                row.label(text=value)

        draw_summary(built if has_retopo else planned, ctx.report("retopo") if has_retopo else None)
        if has_retopo and planned != built:
            content.separator(factor=0.3)
            content.label(text="Next Build", icon="FORWARD")
            draw_summary(planned)
        if ctx is not None:
            content.progress(
                factor=min(len(cached) / 9, 1.0), text=f"{len(cached)} / 9 stages up to date"
            )

        if ctx is not None and ctx.report("output") is not None:
            rerun = box.row(align=True)
            rerun.enabled = bool(bpy.data.filepath)
            rebuild_operator = rerun.operator(
                "rebuilt.run_pipeline", text="Rebuild All", icon="FILE_REFRESH"
            )
            rebuild_operator.asset_name = asset_name
            rebuild_operator.only_stage = ""
            rebuild_operator.force = True

    _STAGE_LABELS = (
        ("retopo", "Retopologise", None),
        ("seams", "Seams", "retopo"),
        ("uv", "Unwrap", "seams"),
        ("bake", "Bake Maps", "uv"),
    )

    def _draw_stages(self, layout, asset_name: str, ctx) -> None:
        """Draw Manual stages in dependency order."""
        cached = ctx.cached_keys if ctx else {}
        has_saved_file = bool(bpy.data.filepath)
        settings = bpy.context.scene.rebuilt
        for name, stage_label, depende_de in self._STAGE_LABELS:
            box = layout.box()
            is_done = name in cached and (name != "bake" or "output" in cached)
            dependencies_ready = depende_de is None or depende_de in cached

            can_run = dependencies_ready and has_saved_file and (
                name != "retopo" or bpy.context.mode == "OBJECT"
            )
            from . import flow_guides
            can_run = can_run and flow_guides._DRAW is None
            expansion_property = f"show_{name}_options"
            has_options = hasattr(settings, expansion_property)
            options_open = has_options and getattr(settings, expansion_property)

            stage_row = box.row(align=True)
            stage_row.label(text="", icon="CHECKMARK" if is_done else "DOT")

            if has_options:
                stage_row.prop(
                    settings,
                    expansion_property,
                    text=stage_label,
                    icon="TRIA_DOWN" if options_open else "TRIA_RIGHT",
                    emboss=False,
                )
                action_row = stage_row.row(align=True)
                action_row.enabled = can_run
                action_row.alignment = "RIGHT"

                operator = action_row.operator(
                    "rebuilt.run_pipeline",
                    text="",
                    icon="FILE_REFRESH" if is_done else "PLAY",
                )
                operator.asset_name = asset_name
                operator.only_stage = name
                operator.force = is_done
            else:
                action_row = stage_row.row(align=True)
                action_row.enabled = can_run
                operator = action_row.operator(
                    "rebuilt.run_pipeline",
                    text=stage_label + (" (redo)" if is_done else ""),
                    icon="FILE_REFRESH" if is_done else "PLAY",
                )
                operator.asset_name = asset_name
                operator.only_stage = name
                operator.force = is_done

            if options_open:
                options_box = box.box()
                self._draw_stage_settings(options_box, name, settings, ctx)
                if name == "uv" and is_done:
                    if (ctx.report("uv") or {}).get("method") == "external":
                        options_box.label(text="Using existing UVs", icon="CHECKMARK")
                    options_box.operator(
                        "rebuilt.show_uvs", text="Show UVs", icon="UV"
                    ).asset_name = asset_name

                footer_row = options_box.row(align=True)
                footer_row.enabled = can_run
                footer_row.scale_y = 1.3
                run_operator = footer_row.operator(
                    "rebuilt.run_pipeline",
                    text=(
                        "Recalculate Seams" if name == "seams" and is_done
                        else ("Redo " if is_done else "") + stage_label
                    ),
                    icon="FILE_REFRESH" if is_done else "PLAY",
                )
                run_operator.asset_name = asset_name
                run_operator.only_stage = name
                run_operator.force = is_done
                if name in {"seams", "uv"}:
                    options_box.separator()
                    self._draw_existing(options_box, name, asset_name, ctx)
        box = layout.column(align=True)
        pending_labels = [e for n, e, _ in self._STAGE_LABELS if n not in cached]
        if pending_labels:
            box.label(text=f"Next: {pending_labels[0]}", icon="FORWARD")
        if "bake" in cached and "output" not in cached:
            box.label(text="Run Bake Maps to complete the asset.", icon="INFO")
        elif "output" in cached:
            box.label(text="Asset ready", icon="CHECKMARK")

    def _draw_existing(self, layout, stage, asset_name, ctx):
        row = layout.row()
        row.enabled = (
            ctx is not None and "retopo" in ctx.cached_keys and ctx.artifact("low") is not None
        )
        op = row.operator(
            "rebuilt.use_existing",
            text="Use Existing Seams" if stage == "seams" else "Use Existing UVs",
            emboss=False,
        )
        op.asset_name = asset_name
        op.stage = stage

    def _draw_stage_settings(self, box, stage: str, settings, ctx=None) -> None:
        """Draw settings for a single stage."""
        if stage == "retopo":
            column = box.column(align=True)
            column.use_property_split = True

            column.prop(settings, "profile", text="Type")
            if not settings.profile_chosen:
                note_column = column.column(align=True)
                note_column.scale_y = 0.7
                note_column.label(text="Proposed by the analysis; pick one to override")
            column.separator()
            column.row(align=True).prop(settings, "topology", expand=True)
            if settings.topology == "quads":
                column.prop(settings, "remesher", text="Engine")
            column.prop(
                settings,
                "target_quads",
                text="Quads" if settings.topology == "quads" else "Triangles",
            )
            if settings.topology == "quads":
                column.row(align=True).prop(settings, "quad_budget_mode", expand=True)

                from .flow_guides import settings as guide_settings, AVAILABLE as guides_available
                flow = guide_settings(bpy.context)
                guided = guides_available and settings.mode == "manual" and settings.remesher == "auto" and flow is not None and flow.enabled
                if guided:
                    column.label(text="Count may vary", icon="INFO")
                else:
                    stage_row = column.row(align=True)
                    stage_row.enabled = settings.remesher in ("field", "instant", "auto")
                    stage_row.prop(settings, "adaptive_size", slider=True)
                    stage_row = column.row(align=True)
                    stage_row.enabled = settings.remesher == "auto"
                    stage_row.prop(settings, "adaptive_shape", slider=True)
                    if settings.remesher == "auto":
                        column.prop(settings, "cloth_folds")
                    column.prop(settings, "keep_open_edges")
            else:
                column.prop(settings, "keep_open_edges")
            _draw_thin_advice(box, settings, ctx)
            if settings.mode == "manual" and settings.topology == "quads" and settings.remesher == "auto":
                from .flow_guides import draw_panel
                draw_panel(box, bpy.context)
        elif stage == "seams":
            column = box.column(align=True)
            column.label(text="UV Islands")
            column.row(align=True).prop(settings, "uv_island_preference", expand=True)
            column.prop(settings, "uv_keep_marked_seams")
            report = ctx.report("seams") if ctx and "seams" in ctx.cached_keys else None
            if report and report.get("method") == "external":
                box.label(
                    text="Provided with existing UVs" if report.get("provided_with_uvs")
                    else "Using existing seams",
                    icon="CHECKMARK",
                )
            elif report:
                marked = len(report.get("protected_edges", []))
                if marked:
                    box.label(text=f"Keeping {marked} marked seams", icon="LOCKED")
                postprocess = report.get("initial_final_validation") or report.get("final_native_report", {})
                reduction = postprocess.get("join_small_islands")
                if reduction is None:
                    reduction = report.get("engine_report", {}).get("fewer_islands")
                if reduction and "before" in reduction and "after" in reduction:
                    if reduction["after"] < reduction["before"]:
                        box.label(text=f"Islands: {reduction['before']} → {reduction['after']}", icon="CHECKMARK")
                    else:
                        box.label(text="No further safe merges found", icon="INFO")
                alternatives = len(report.get("offers", []))
                if alternatives > 1:
                    box.operator(
                        "rebuilt.next_seams", text="Other", icon="LOOP_FORWARDS"
                    ).asset_name = ctx.name
                chart_count = report.get("charts")
                detail = (
                    (f"Cut {int(report.get('choice', 0)) + 1} of {alternatives}: "
                     if alternatives > 1 else "")
                    + (f"{chart_count} charts, " if chart_count else "")
                    + f"{report.get('islands', '?')} islands. Edit red seams, then Unwrap."
                )
                for line in _wrap(detail, 44):
                    box.label(text=line)
        elif stage == "uv":
            column = box.column(align=True)
            column.use_property_split = True
            column.prop(settings, "uv_unwrap_method")
            column.separator()
            column.prop(settings, "uv_packer")
            if settings.uv_packer == "uvpack":
                column.prop(settings, "uv_pack_seconds")
            else:
                column.prop(settings, "uv_shape_method")
            column.prop(settings, "uv_rotate_method")
            column.prop(settings, "uv_padding_px")
        elif stage == "bake":
            column = box.column(align=True)
            column.label(text="Also checks quality and completes the asset.", icon="INFO")
            map_controls = column.grid_flow(columns=2, even_columns=True, align=True)
            for prop in (
                "bake_normal",
                "bake_base_color",
                "bake_ao",
                "bake_roughness",
                "bake_metallic",
                "bake_opacity",
            ):
                map_controls.prop(settings, prop)
            from .properties import _MAP_PROPERTIES
            from ..core.bake.channels import EXTRA_CHANNELS

            column.prop(
                settings,
                "show_material_maps",
                emboss=False,
                icon="TRIA_DOWN" if settings.show_material_maps else "TRIA_RIGHT",
            )
            if settings.show_material_maps:
                extra = column.grid_flow(columns=2, even_columns=True, align=True)
                for name in EXTRA_CHANNELS:
                    extra.prop(settings, _MAP_PROPERTIES[name])
            if not any(getattr(settings, p) for p in _MAP_PROPERTIES.values()):
                column.label(text="No maps selected — Normal will be baked", icon="INFO")
            options_column = column.column(align=True)
            options_column.use_property_split = True
            options_column.prop(settings, "texture_resolution", text="Resolution")
            options_column.prop(settings, "bake_normal_space")
            if settings.bake_ao:
                options_column.prop(settings, "bake_ao_samples")
            _draw_image_settings(column, settings)


class REBUILT_PT_export(bpy.types.Panel):
    bl_idname = "REBUILT_PT_export"
    bl_label = "Export"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 20

    @classmethod
    def poll(cls, context):
        return True

    def draw(self, context):
        layout = self.layout
        layout.use_property_decorate = False
        summary = layout.column(align=True)
        summary.enabled = False
        summary.label(text="Retopo · UV · Bake")
        settings = context.scene.rebuilt
        ctx = export_context(context)
        box = layout.box()
        if ctx is None or not ctx.ready:
            message = box.column(align=True)
            if context.object is None or context.object.type != "MESH":
                message.label(text="Select a model to get started.", icon="INFO")
            else:
                message.label(text="Complete the model workflow", icon="INFO")
                message.label(text="to enable export.")
            row = box.row()
            row.enabled = False
            row.operator("rebuilt.export_asset", text="Export", icon="EXPORT")
            return
        asset_name = ctx.name
        column = box.column(align=True)
        column.use_property_split = True
        column.prop(settings, "target_engine")
        format_row = column.row()
        format_row.label(text="Format")
        format_row.label(
            text="FBX · Unity Folder" if settings.target_engine == "unity" else "glTF (.glb)"
        )
        if settings.target_engine == "unity":
            column.prop(settings, "unity_pipeline")
            column.label(text="Folder with model and textures", icon="INFO")
        if settings.target_engine in {"unreal", "godot"}:
            column.label(text="Folder with a self-contained GLB", icon="INFO")
        stage_row = box.row(align=True)
        stage_row.scale_y = 1.5
        stage_row.operator("rebuilt.export_asset", text="Export", icon="EXPORT").asset_name = asset_name
        stage_report = ctx.report("export")
        if stage_report and stage_report.get("exported"):
            box.label(
                text=stage_report.get("asset_folder")
                or stage_report.get("unity_package")
                or stage_report.get("unreal_package")
                or stage_report["exported"],
                icon="CHECKMARK",
            )


class REBUILT_PT_help(bpy.types.Panel):
    bl_idname = "REBUILT_PT_help"
    bl_label = "Help"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 30

    def draw(self, context):
        description = self.layout.column(align=True)
        description.label(text="Rebuilt · Retopo · UV · Bake")
        for line in _wrap("Turn complex models into clean production assets", 38):
            description.label(text=line)
        self.layout.separator()
        for link in help_links():
            row = self.layout.row()
            row.enabled = bool(link["link"])
            row.operator("wm.url_open", text=link["name"], icon=link["icon"]).url = link["link"]


class REBUILT_PT_report(bpy.types.Panel):
    """Optional processing details, grouped away from the main controls."""

    bl_idname = "REBUILT_PT_report"
    bl_label = "Report"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY
    bl_order = 10
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        ctx = active_context(context)
        return ctx is not None and any(
            bool(ctx.report(stage))
            for stage in (
                "ingest",
                "cleanup",
                "analyze",
                "retopo",
                "uv",
                "bake",
                "qa",
                "output",
            )
        )

    def draw(self, context: bpy.types.Context) -> None:
        self.layout.use_property_decorate = False
        advice = thin_advice(context.scene.rebuilt, active_context(context))
        if advice:
            icon, lines = advice
            column = self.layout.column(align=True)
            for index, line in enumerate(lines):
                column.label(text=line, icon=icon if index == 0 else "NONE")


class REBUILT_PT_scan_report(bpy.types.Panel):
    """What the source model looked like on arrival, and what was done about it."""

    bl_idname = "REBUILT_PT_scan_report"
    bl_label = "Source"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY
    bl_parent_id = "REBUILT_PT_report"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        ctx = active_context(context)
        return ctx is not None and bool(ctx.report("ingest") or ctx.report("cleanup"))

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        ctx = active_context(context)
        if ctx is None:
            return

        ingest = ctx.report("ingest")
        if ingest:
            column = layout.column(align=True)
            column.label(text=f"{ingest['triangle_count']:,} triangles", icon="MESH_DATA")
            column.label(text=f"{ingest['bbox_diagonal']:.3g} units across")
            if ingest.get("proxy"):
                column.label(text="Analysis runs on a proxy", icon="MOD_DECIM")

            checks = ingest.get("preflight") or {}
            for conversion in checks.get("material_protocol", []):
                if conversion.get("status") == "converted":
                    column.label(text="Material: Emission → Principled BSDF", icon="MATERIAL")
            if checks.get("welded_vertices"):
                column.label(
                    text=f"Welded {checks['welded_vertices']:,} duplicate vertices",
                    icon="AUTOMERGE_ON",
                )
            if checks.get("transform_applied"):
                column.label(text="Source transform baked into the copy", icon="CON_SIZELIKE")

            for note in checks.get("notes", ()):
                layout.box().label(text=note, icon="ERROR")

            if ingest.get("scale_warning"):
                layout.box().label(text=ingest["scale_warning"], icon="ERROR")

        payload = ctx.report("cleanup")
        if not payload:
            return

        actions = payload.pop("actions", [])
        findings = DefectReport.from_dict(payload).findings()

        box = layout.box()
        if findings:
            box.label(text="Findings", icon="INFO")
            for line in findings:
                box.label(text=line)
        else:
            box.label(text="Nothing to report", icon="CHECKMARK")

        if actions:
            repaired = layout.box()
            repaired.label(text="Repaired", icon="MODIFIER")
            for line in actions:
                repaired.label(text=line)
        elif findings:
            layout.label(text="Nothing was changed.", icon="DECORATE_LOCKED")


class REBUILT_PT_analysis(bpy.types.Panel):
    """What the system thinks the asset is, and on what evidence."""

    bl_idname = "REBUILT_PT_analysis"
    bl_label = "Analysis"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY
    bl_parent_id = "REBUILT_PT_report"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        ctx = active_context(context)
        return ctx is not None and ctx.report("analyze") is not None

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        ctx = active_context(context)
        report = ctx.report("analyze") if ctx else None
        if not report:
            return

        header = layout.box()
        row = header.row()
        row.label(text=report["asset_class"].replace("_", " ").title(), icon="OBJECT_DATA")
        row.label(text=f"{report['confidence']:.0%}")

        if report.get("overridden"):
            header.label(text=f"Overridden (detected: {report['detected_class']})", icon="LOCKED")
        elif report.get("needs_review"):
            header.label(text="Low confidence — review or override", icon="QUESTION")

        evidence = layout.column(align=True)
        evidence.label(text=f"Flat surface: {report['evidence']['planar_area_ratio']:.0%}")
        evidence.label(text=f"Creased edges: {report['evidence']['sharp_edge_ratio']:.1%}")
        evidence.label(text=f"Axis aligned: {report['evidence']['axis_alignment']:.0%}")
        evidence.label(text=f"Flat regions: {report['region_count']}")

        symmetry = report.get("symmetry")
        layout.label(
            text=(
                f"Symmetry found ({symmetry['confidence']:.0%})"
                if symmetry
                else "No symmetry plane found"
            ),
            icon="MOD_MIRROR",
        )


class REBUILT_PT_retopo(bpy.types.Panel):
    """How the low mesh turned out against what was asked for."""

    bl_idname = "REBUILT_PT_retopo"
    bl_label = "Retopology"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY
    bl_parent_id = "REBUILT_PT_report"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        ctx = active_context(context)
        return ctx is not None and ctx.report("retopo") is not None

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        ctx = active_context(context)
        report = ctx.report("retopo") if ctx else None
        if not report:
            return

        layout.label(text=report["strategy"].replace("_", " ").title(), icon="MOD_REMESH")

        budget = layout.row()
        mode = report.get("budget_mode")
        if mode:
            budget.label(text=f"{report['achieved_quads']:,} quads · {mode.title()}")
            if mode == "strict":
                label = f"Target {report['requested_quads']:,} ({report['quad_budget_error']:+.1%})"
            else:
                label = f"Starting budget {report['requested_quads']:,}"
            layout.label(text=label, icon="CHECKMARK" if report["budget_met"] else "ERROR")
        else:
            budget.label(text=f"{report['achieved_faces']:,} / {report['target_faces']:,} faces")
            budget.label(
                text=f"{report['budget_used']:.0%} of budget",
                icon="ERROR" if report["over_budget"] else "CHECKMARK",
            )

        deviation = layout.row()
        deviation.label(text=f"Deviation P95: {report['deviation']['p95']:.4f}")
        deviation.label(text="", icon="CHECKMARK" if report["within_tolerance"] else "ERROR")

        topology = report["topology"]
        layout.label(text=f"Quads: {topology['quad_ratio']:.0%}")

        review = report.get("finishing_warnings") or report["errors"]
        if review:
            box = layout.box()
            box.label(text="Mesh needs review", icon="ERROR")
            for line in review:
                box.label(text=line)
        else:
            layout.label(text="Topology sound", icon="CHECKMARK")

        for line in report["warnings"]:
            layout.label(text=line, icon="INFO")

        for note in report.get("notes", []):
            box = layout.box()
            for line in _wrap(note):
                box.label(text=line, icon="ERROR" if line is _wrap(note)[0] else "BLANK1")


class REBUILT_PT_uv(bpy.types.Panel):
    """What the layout cost and what it bought."""

    bl_idname = "REBUILT_PT_uv"
    bl_label = "UVs"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY
    bl_parent_id = "REBUILT_PT_report"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        ctx = active_context(context)
        return ctx is not None and ctx.report("uv") is not None

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        ctx = active_context(context)
        report = ctx.report("uv") if ctx else None
        if not report:
            return

        column = layout.column(align=True)
        charts = report.get("charts")
        column.label(
            text=(
                f"Existing UVs, {report.get('islands', '?')} islands"
                if report.get("method") == "external"
                else f"{charts} charts, {report.get('islands', charts)} islands, "
                f"{report['seam_count']} seam edges"
                if charts
                else f"Smart Project, {report.get('islands', '?')} islands"
            ),
            icon="UV_EDGESEL",
        )
        column.label(text=f"{report['texel_density']:.0f} px/unit")
        column.label(
            text="External packing; padding not measured"
            if report.get("method") == "external"
            else f"Padding {report['padding_px']}px of {report['texture_size']}"
        )
        column.label(
            text=f"Occupancy {report['occupancy']:.0%}"
            + (f" ({report['packer']})" if report.get("packer") else "")
        )

        stretch = layout.row()
        stretch.label(text=f"Stretch P95: {report['stretch']['p95']:.3f}")
        stretch.label(text="", icon="CHECKMARK" if report["stretch_met"] else "ERROR")

        for note in report.get("notes", []):
            layout.box().label(text=note, icon="INFO")


class REBUILT_PT_bake(bpy.types.Panel):
    """Whether the detail actually landed where it was supposed to."""

    bl_idname = "REBUILT_PT_bake"
    bl_label = "Bake"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY
    bl_parent_id = "REBUILT_PT_report"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        ctx = active_context(context)
        return ctx is not None and ctx.report("bake") is not None

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        ctx = active_context(context)
        report = ctx.report("bake") if ctx else None
        if not report:
            return

        quality = report["quality"]
        approximate = report.get("approximation", {}).get("texels", 0)
        header = layout.row()
        header.label(text=f"Valid texels: {quality['valid_fraction']:.2%}")
        header.label(
            text="",
            icon="CHECKMARK" if quality["valid_fraction"] >= 0.995 and not approximate else "ERROR",
        )

        if approximate:
            layout.label(text=f"Completed with {approximate:,} approximate texels", icon="INFO")
        detail = layout.column(align=True)
        detail.label(text=f"{quality['missed_texels']:,} missed, {quality['stray_texels']:,} stray")
        detail.label(text=f"Ray distance P95: {quality['ray_distance_p95']:.4f}")
        cage = report.get("cage", {})
        if cage:
            detail.label(
                text=f"Cage {cage['minimum']:.4f}–{cage['maximum']:.4f}",
                icon="MOD_MESHDEFORM",
            )

        if len(report.get("attempts", [])) > 1:
            layout.label(text=f"Retried {len(report['attempts']) - 1}×", icon="FILE_REFRESH")

        maps = layout.box()
        maps.label(text="Maps", icon="TEXTURE")
        for entry in report.get("maps", []):
            row = maps.row()
            row.label(text=entry["map"].replace("_", " ").title())
            row.label(text="", icon="CHECKMARK" if entry.get("baked") else "X")

        for note in report.get("notes", []):
            layout.box().label(text=note, icon="ERROR")


class REBUILT_PT_qa(bpy.types.Panel):
    """The full scorecard and measured issues, available inside Report."""

    bl_idname = "REBUILT_PT_qa"
    bl_label = "Quality"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY
    bl_parent_id = "REBUILT_PT_report"
    bl_options = {"DEFAULT_CLOSED"}

    ICONS = {
        "pass": "CHECKMARK",
        "warn": "ERROR",
        "fail": "CANCEL",
        "unknown": "QUESTION",
    }

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        ctx = active_context(context)
        return ctx is not None and ctx.report("qa") is not None

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        ctx = active_context(context)
        report = ctx.report("qa") if ctx else None
        if not report:
            return

        verdict = report["verdict"]
        header = layout.box().row()
        label = {"pass": "Checks passed", "warn": "Review suggested", "fail": "Needs review"}.get(
            verdict, "Not evaluated"
        )
        header.label(text=label, icon=self.ICONS.get(verdict, "QUESTION"))
        header.label(text=f"{report['score']:.0f}/100")

        for section in report["sections"]:
            row = layout.row(align=True)
            row.label(text="", icon=self.ICONS.get(section["verdict"], "QUESTION"))
            row.label(text=section["name"])
            row.label(text=section["reason"] or f"{section['score']:.0f}")

        issues = report.get("issues", [])
        if issues:
            box = layout.box()
            box.label(text="Needs attention", icon="INFO")
            for issue in issues:
                column = box.column(align=True)
                column.label(text=issue["name"], icon=self.ICONS.get(issue["verdict"], "QUESTION"))
                if issue["value"] is not None and issue["threshold"] is not None:
                    comparison = "at least" if issue["higher_is_better"] else "at most"
                    column.label(
                        text=f"    {issue['value']:.4g} — {comparison} {issue['threshold']:.4g}"
                    )
                if issue["detail"]:
                    column.label(text=f"    {issue['detail']}")

        layout.operator("rebuilt.show_result", icon="HIDE_OFF", text="Show Deviation")


class REBUILT_PT_output(bpy.types.Panel):
    """Where the finished asset ended up."""

    bl_idname = "REBUILT_PT_output"
    bl_label = "Output"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY
    bl_parent_id = "REBUILT_PT_report"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        ctx = active_context(context)
        return ctx is not None and ctx.report("output") is not None

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        ctx = active_context(context)
        report = ctx.report("output") if ctx else None
        if not report:
            return

        column = layout.column(align=True)
        column.label(text=report["object"], icon="MESH_DATA")
        if report["material"]:
            column.label(text=report["material"], icon="MATERIAL")

        written = report.get("maps_on_disk", [])
        if written:
            box = layout.box()
            box.label(text=f"{len(written)} map(s) written", icon="FILE_IMAGE")
        for note in report.get("notes", []):
            layout.label(text=note, icon="INFO")

        export_report = ctx.report("export")
        if export_report:
            box = layout.box()
            box.label(text=f"Exported for {export_report['target_engine']}", icon="EXPORT")
            if export_report.get("exported"):
                box.label(text=export_report["exported"])
            for note in export_report.get("notes", []):
                box.label(text=note, icon="INFO")


_CLASSES = (
    REBUILT_PT_home,
    REBUILT_PT_export,
    REBUILT_PT_help,
    REBUILT_PT_report,
    REBUILT_PT_scan_report,
    REBUILT_PT_analysis,
    REBUILT_PT_retopo,
    REBUILT_PT_uv,
    REBUILT_PT_bake,
    REBUILT_PT_qa,
    REBUILT_PT_output,
)


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
