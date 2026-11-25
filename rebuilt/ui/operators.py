"""Operators. Mostly thin adapters over the pipeline."""

from __future__ import annotations

import time
from functools import wraps

import bpy

from .. import stages as stage_registry
from ..bridge import diagnosis
from ..core.graph import StageGraph
from ..pipeline.context import AssetContext, ContextError
from ..pipeline.job import JobRunner, JobStatus, progress_text
from ..stages.cleanup import current_mesh_object


TIMER_INTERVAL = 0.005
PROGRESS_INTERVAL = 0.1


def _selected_context(context, name=None):
    from .panels import context_resolution
    ctx, diagnostic = context_resolution(context, name)
    if diagnostic:
        raise ContextError(diagnostic)
    return ctx


def _report_context_errors(method):
    @wraps(method)
    def execute(self, context, *args):
        try:
            return method(self, context, *args)
        except ContextError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
    return execute


def _asset_name(context: bpy.types.Context, given: str) -> str:
    if given:
        return given
    obj = context.active_object
    if obj is not None and obj.get("rebuilt_flow_guide"):
        obj = obj.parent
    return obj.get("rebuilt_asset", "") if obj else ""


class REBUILT_OT_texture_output_info(bpy.types.Operator):
    bl_idname = "rebuilt.texture_output_info"
    bl_label = "Texture Output Info"
    bl_options = {"INTERNAL"}

    image_format: bpy.props.StringProperty(options={"HIDDEN", "SKIP_SAVE"})

    _MESSAGES = {
        "PNG": (
            "PNG is lossless. Higher compression can reduce file size but takes longer "
            "to save, without changing image quality. HDR values use OpenEXR."
        ),
        "JPEG": (
            "JPEG compresses color maps. Lower quality reduces file size but can add "
            "visible artifacts. Data maps stay lossless PNG; HDR values use OpenEXR."
        ),
        "TIFF": (
            "TIFF supports 8-bit and 16-bit textures for editing. Deflate and LZW "
            "compression are lossless. HDR values use OpenEXR."
        ),
        "OPEN_EXR": (
            "OpenEXR stores linear HDR textures. 16-bit Float uses less space; "
            "32-bit Float offers more precision. ZIP and PIZ compression are lossless."
        ),
    }

    @classmethod
    def description(cls, context, properties):
        return cls._MESSAGES.get(properties.image_format, "Texture output format information")

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=340)

    def draw(self, context):
        import textwrap

        column = self.layout.column(align=True)
        for line in textwrap.wrap(self.description(context, self), width=48):
            column.label(text=line)

    def execute(self, context):
        return {"FINISHED"}


def _frame_adopted_source(context, source) -> bool:
    """Frame only the invoking viewport; camera and ambiguous quad views stay untouched."""
    area = context.area
    if area is None or area.type != "VIEW_3D" or context.mode != "OBJECT":
        return False
    space = area.spaces.active
    view = space.region_3d
    if space.region_quadviews or view is None or view.view_perspective == "CAMERA":
        return False
    region = next((r for r in area.regions if r.type == "WINDOW"), None)
    if region is None:
        return False
    selected = list(context.selected_objects)
    active = context.view_layer.objects.active
    source_selected = source.select_get()
    try:
        for obj in selected:
            if obj != source:
                obj.select_set(False)
        source.select_set(True)
        context.view_layer.objects.active = source
        with context.temp_override(area=area, region=region):
            return "FINISHED" in bpy.ops.view3d.view_selected(
                "INVOKE_DEFAULT", use_all_regions=False
            )
    except RuntimeError:
        return False
    finally:
        source.select_set(source_selected)
        for obj in selected:
            obj.select_set(True)
        context.view_layer.objects.active = active


def _prepare_source(context, *, reset_choices=False):
    from .flow_guides import resolve_source
    selected = context.active_object
    if selected is not None and selected.get('rebuilt_flow_guide'):
        selected = selected.parent
    source = selected if reset_choices else resolve_source(context) or selected
    ctx = AssetContext.create(context.scene, source)
    settings = context.scene.rebuilt
    if reset_choices:
        settings.profile_chosen = False
    settings.suggest_bake_maps(ctx, reset=reset_choices)
    ctx.recipe = settings.build_recipe(flow_source=ctx.source_object)
    try:
        ctx.set_report("diagnosis", diagnosis.measure_thickness(ctx.source_object))
    except Exception as exc:  # A diagnosis must not prevent starting a model.
        print(f"Rebuilt: thickness diagnosis skipped — {exc}")
    return ctx


class REBUILT_OT_adopt_scan(bpy.types.Operator):
    """Use the selected mesh as the source model for this asset"""

    bl_idname = "rebuilt.adopt_scan"
    bl_label = "Use Selected Model"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            cls.poll_message_set("Select a mesh object. Other object types are not supported.")
            return False
        return True

    def execute(self, context: bpy.types.Context) -> set[str]:
        try:
            ctx = _prepare_source(context, reset_choices=True)
        except ContextError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        self.report({"INFO"}, f"Selected {ctx.name!r} as the source model")
        return {"FINISHED"}


class REBUILT_OT_adopt_results(bpy.types.Operator):
    """Review the exact legacy objects that future rebuilding may replace"""

    bl_idname = 'rebuilt.adopt_results'
    bl_label = 'Review Existing Results'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        from .panels import context_resolution
        ctx, _ = context_resolution(context)
        try:
            return (ctx is not None and ctx._ownership.record() is None
                    and ctx.source_object.get('rebuilt_ownership_version') is None
                    and bool(ctx._ownership.legacy()[0]))
        except ContextError:
            return False

    def _snapshot(self, context):
        obj = context.active_object
        if obj is not None and obj.get('rebuilt_flow_guide'):
            obj = obj.parent
        ctx = AssetContext.for_object(context.scene, obj)
        if ctx is None or ctx._ownership.record() is not None:
            raise ContextError('Select an asset with legacy results to review.')
        objects, collection = ctx._ownership.legacy()
        if not objects:
            raise ContextError('No legacy results are available to adopt.')
        return ctx, (ctx.source_object, tuple((obj, obj['rebuilt_role']) for obj in objects), collection)

    def invoke(self, context, event):
        try:
            _, self._reviewed = self._snapshot(context)
            self._reviewed_labels = tuple(f'{role}: {obj.name}' for obj, role in self._reviewed[1])
        except ContextError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return context.window_manager.invoke_props_dialog(self, width=440)

    def draw(self, context):
        from .panels import _diagnostic_lines
        texts = ('Rebuilding may replace these existing results:', *self._reviewed_labels,
                 'Only accept if these are generated results you want Rebuilt to manage.')
        for text in texts:
            for line in _diagnostic_lines(text, context, width=440):
                self.layout.label(text=line)

    def execute(self, context):
        try:
            ctx, snapshot = self._snapshot(context)
            if getattr(self, '_reviewed', None) != snapshot:
                raise ContextError('The selected results changed. Review them again before adopting.')
            ctx._ownership.ensure(adopt=True)
        except ContextError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}


class REBUILT_OT_recover_asset(bpy.types.Operator):
    """Review and recover an interrupted build, including its texture files"""

    bl_idname = 'rebuilt.recover_asset'
    bl_label = 'Recover Interrupted Build'
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        from ..pipeline.job import active_runners
        if context.mode != 'OBJECT' or active_runners(context.scene):
            return False
        obj = context.active_object
        if obj is not None and obj.get('rebuilt_flow_guide'):
            obj = obj.parent
        try:
            ctx = AssetContext.for_object(context.scene, obj)
            record = ctx._ownership.record() if ctx is not None else None
            return record is not None and any(key in record for key in ('archive', 'file_journal', 'stage_operation', 'auto_completion'))
        except ContextError:
            return False

    @staticmethod
    def _snapshot(context):
        from ..pipeline.auto_archive import AutoArchive
        selected = context.active_object
        obj = selected.parent if selected is not None and selected.get('rebuilt_flow_guide') else selected
        ctx = AssetContext.for_object(context.scene, obj)
        if ctx is None:
            raise ContextError('Select the asset with an interrupted build to review.')
        from ..pipeline.stage_operation import StageOperation
        pending = (ctx._ownership.record() or {}).get('stage_operation')
        recovery = StageOperation if pending is not None and pending['owns_files'] else AutoArchive
        review = recovery.review(ctx)
        review['recovery_signature'] = review['signature']
        review['signature'] = (review['signature'], selected.session_uid)
        return ctx, review

    def invoke(self, context, event):
        try:
            _, self._review = self._snapshot(context)
            self._reviewed = self._review['signature']
        except ContextError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return context.window_manager.invoke_props_dialog(self, width=520)

    def draw(self, context):
        from .panels import _diagnostic_lines
        review = self._review
        lines = [f'Asset: {review["asset"]}']
        lines.append(review.get('action') or ('Restore the previous build:' if review['restoring'] else 'Discard this unfinished first build:'))
        lines.extend(f'{role}: {name}' for role, name in review['restore'])
        lines.extend(f'{review.get("remove_label", "Remove unfinished result")}: {name}' for name in review['remove'])
        lines.extend(f'{action} texture: {name}' for action, name in review['files'])
        lines.append('Recovery rechecks these objects and files before continuing.')
        for text in lines:
            for line in _diagnostic_lines(text, context, width=520):
                self.layout.label(text=line)

    def execute(self, context):
        from ..pipeline.auto_archive import AutoArchive
        try:
            ctx, review = self._snapshot(context)
            if getattr(self, '_reviewed', None) != review['signature']:
                raise ContextError('Review this interrupted build before recovering it.')
            from ..pipeline.stage_operation import StageOperation
            recovery = StageOperation if review.get('kind') == 'stage' else AutoArchive
            recovery.recover(ctx, reviewed_signature=review['recovery_signature'])
        except ContextError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self.report({'INFO'}, 'Interrupted build recovered.')
        return {'FINISHED'}


class REBUILT_OT_run_pipeline(bpy.types.Operator):
    """Run the Rebuilt pipeline on the selected asset"""

    bl_idname = "rebuilt.run_pipeline"
    bl_label = "Rebuild Asset"
    bl_options = {"REGISTER"}

    asset_name: bpy.props.StringProperty(name="Asset", default="", options={"SKIP_SAVE"})

    only_stage: bpy.props.StringProperty(name="Stage", default="", options={"SKIP_SAVE"})

    force: bpy.props.BoolProperty(name="Force", default=False, options={"SKIP_SAVE"})

    confirmed: bpy.props.BoolProperty(name="Confirmed", default=False, options={"SKIP_SAVE"})

    def _make_runner(self, context: bpy.types.Context) -> JobRunner | None:
        try:
            return self._make_owned_runner(context)
        except ContextError as error:
            self.report({'ERROR'}, str(error))
            return None

    def _make_owned_runner(self, context: bpy.types.Context) -> JobRunner | None:
        from . import flow_guides
        from ..pipeline.job import active_runners

        if flow_guides._DRAW is not None or active_runners(context.scene):
            self.report({"ERROR"}, "Finish the current operation before starting processing.")
            return None
        if not bpy.data.filepath:
            self.report(
                {"ERROR"},
                "Save the .blend first. The baked maps are written beside it, "
                "and without a location the asset will not display correctly.",
            )
            return None

        name = _asset_name(context, self.asset_name)
        if context.mode != "OBJECT" and (not name or self.only_stage in {"", "retopo"}):
            self.report({"ERROR"}, "Switch to Object Mode before starting processing.")
            return None
        if name:
            selected = context.active_object
            if selected is not None and selected.get('rebuilt_flow_guide'):
                selected = selected.parent
            ctx = (AssetContext.for_object(context.scene, selected)
                   if selected is not None and selected.get('rebuilt_asset') == name
                   else AssetContext.find(context.scene, name))
        else:
            obj = context.active_object
            if obj is not None and obj.get("rebuilt_flow_guide"):
                obj = obj.parent
            if obj is None or obj.type != "MESH" or not context.active_object.select_get():
                self.report({"ERROR"}, "Select a mesh model to process.")
                return None
            if self.only_stage not in {"", "retopo"}:
                self.report({"ERROR"}, "Run Retopologise before continuing with this model.")
                return None
            try:
                ctx = _prepare_source(context)
            except ContextError as exc:
                self.report({"ERROR"}, str(exc))
                return None
        if ctx is None:
            self.report({"ERROR"}, f"No Rebuilt data for {name!r}")
            return None

        ctx.require_owned()

        settings = context.scene.rebuilt

        auto_run = settings.mode == "auto" and not self.only_stage
        previous_state = None
        if (
            auto_run
            and "output" in ctx.cached_keys
            and ctx.report("output") is not None
            and ctx.artifact("low") is not None
        ):
            from ..pipeline.context import KEY_RECIPE, KEY_REPORTS, KEY_CACHE

            previous_state = {
                key: ctx.source_object[key]
                for key in (KEY_RECIPE, KEY_REPORTS, KEY_CACHE)
                if key in ctx.source_object
            }
        self._review_area = context.area
        self._review_wireframe = None
        self._review_seams = settings.mode == "manual" and self.only_stage == "seams"
        self._review_uv = settings.mode == "manual" and self.only_stage == "uv"
        if settings.mode == "manual":
            if self.only_stage == "retopo":
                self._review_wireframe = True
            elif self.only_stage == "bake":
                self._review_wireframe = False

        settings.suggest_bake_maps(ctx)
        recipe = settings.build_recipe(flow_source=ctx.source_object)
        cached = ctx.cached_keys
        invalidated = frozenset()

        restore_preparation = False
        if cached and current_mesh_object(ctx) is None:
            reusable = (
                self.only_stage in {"seams", "uv", "bake", "qa"}
                and ctx.artifact("low") is not None
                and all(
                    cached.get(name)
                    == stage_registry.stage_named(name).cache_key(recipe)
                    for name in ("ingest", "cleanup", "analyze", "retopo")
                )
            )
            if reusable:
                restore_preparation = True
            else:
                invalidated = frozenset(cached)
                cached = {}
                self.report({"INFO"}, "Rebuilding the asset from the source model.")

        ctx.recipe_for_class = lambda asset_class="": settings.build_recipe(
            asset_class, flow_source=ctx.source_object
        )

        if self.only_stage:
            graph = StageGraph()
            try:
                graph.spec(self.only_stage)
            except Exception as exc:  # noqa: BLE001 — StageGraphError, said to the artist
                self.report({"ERROR"}, str(exc))
                return None

            downstream_stages = graph.downstream(self.only_stage)
            invalidated |= downstream_stages & cached.keys()
            cached = {name: key for name, key in cached.items() if name not in downstream_stages}

            forced = frozenset({self.only_stage}) if self.force else frozenset()

            last_stage = "output" if self.only_stage == "bake" else self.only_stage
            prefix = graph.order[: graph.order.index(last_stage) + 1]
            pipeline = [
                stage
                for stage in stage_registry.build_pipeline(ctx, force=forced, recipe=recipe, cached_keys=cached)
                if stage.name in prefix
            ]
            if not any(stage.name == self.only_stage for stage in pipeline):
                pipeline = [
                    stage
                    for stage in pipeline
                    if stage.name not in graph.downstream(self.only_stage)
                ]
                pipeline.append(stage_registry.stage_named(self.only_stage))
                if self.only_stage == "bake":
                    pipeline.extend(stage_registry.stage_named(name) for name in ("qa", "output"))
            if self.only_stage == "bake":
                for stage in pipeline:
                    if stage.name == "output":
                        stage.retain_working = True
        else:
            forced = (
                frozenset(stage.name for stage in stage_registry.all_stages())
                if self.force
                else frozenset()
            )
            pipeline = stage_registry.build_pipeline(ctx, force=forced, recipe=recipe, cached_keys=cached)

        if restore_preparation:
            pipeline = [
                stage_registry.stage_named(name) for name in ("ingest", "cleanup")
            ] + pipeline

        if not pipeline:
            self.report({"INFO"}, "Everything is already up to date.")
            return None

        if previous_state is not None:
            from ..pipeline.auto_archive import AutoArchive

            ctx.auto_archive = AutoArchive(ctx, previous_state)
        if not self.only_stage or self.only_stage == "retopo":
            obj = context.active_object
            target = (
                obj
                if obj is not None and obj.get("rebuilt_asset") == ctx.name and obj.visible_get()
                else ctx.source_object
            )
            _frame_adopted_source(context, target)
        return JobRunner(ctx, pipeline, reset_on_cancel=auto_run,
                         recipe=recipe, invalidate_stages=invalidated)

    def _report_outcome(self, runner: JobRunner) -> set[str]:
        if runner.status is JobStatus.FAILED:
            self.report({"ERROR"}, f"{runner.current_stage or 'pipeline'} failed: {runner.error}")
            print(runner.formatted_error)
            return {"CANCELLED"}
        if runner.status is JobStatus.CANCELLED and getattr(runner, "reset_on_cancel", False):
            self.report({"WARNING"}, "Auto cancelled. The cancelled attempt was discarded.")
            return {"CANCELLED"}
        if "retopo" in runner.completed or (
            "ingest" in runner.completed and runner.ctx.artifact("low") is not None
        ):
            _show_only_the_asset(runner.ctx)
        if runner.status is JobStatus.DONE:
            settings = runner.ctx.scene.rebuilt
            if settings.mode == "manual":
                for stage in ("retopo", "seams", "uv", "bake"):
                    setattr(settings, f"show_{stage}_options", False)
            if getattr(self, "_review_uv", False) and "uv" in runner.completed:
                _show_uv_review(runner.ctx, getattr(self, "_review_area", None))
            if getattr(self, "_review_seams", False) and "seams" in runner.completed:
                _show_seams_review(runner.ctx, getattr(self, "_review_area", None))
            _update_review_wireframe(
                getattr(self, "_review_area", None),
                getattr(self, "_review_wireframe", None),
                runner.completed,
            )
            retopo = runner.ctx.report("retopo") if "retopo" in runner.completed else None
            if retopo and retopo.get("budget_met") is False:
                self.report(
                    {"WARNING"},
                    f"{retopo['budget_mode'].title()} count not reached: "
                    f"{retopo['achieved_quads']:,} quads for {retopo['requested_quads']:,} requested. "
                    "Review the Retopo report.",
                )
                return {"FINISHED"}
            bake_report = runner.ctx.report("bake") if "bake" in runner.completed else None
            if bake_report and bake_report.get("approximation", {}).get("texels", 0):
                self.report({"WARNING"}, "Finished with approximate texture regions. Review the Bake report.")
                return {"FINISHED"}
            if retopo and retopo.get("finishing_warnings"):
                self.report({"WARNING"}, "Finished with mesh warnings. Review the Retopo report.")
                return {"FINISHED"}
            uv_report = runner.ctx.report("uv") if "uv" in runner.completed else None
            if uv_report and (uv_report.get("unresolved_faces", 0) or uv_report.get("singleton_faces_remaining", 0)):
                self.report({"WARNING"}, "Finished with UV warnings. Review the UV report.")
                return {"FINISHED"}
            seams_report = runner.ctx.report("seams") if "seams" in runner.completed else None
            if seams_report and seams_report.get("singleton_faces_remaining", 0):
                self.report({"WARNING"}, "Finished with single-face UV islands. You can continue.")
                return {"FINISHED"}
            self.report({"INFO"}, f"Finished: {', '.join(runner.completed) or 'nothing to do'}")
            return {"FINISHED"}
        if runner.status is JobStatus.CANCELLED:
            self.report({"WARNING"}, "Cancelled")
            return {"CANCELLED"}
        self.report({"ERROR"}, f"{runner.current_stage or 'pipeline'} failed: {runner.error}")
        print(runner.formatted_error)
        return {"CANCELLED"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        runner = self._make_runner(context)
        if runner is None:
            return {"CANCELLED"}

        runner.start()
        while runner.tick():
            pass
        return self._report_outcome(runner)

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        if bpy.app.background:
            return self.execute(context)

        runner = self._make_runner(context)
        if runner is None:
            return {"CANCELLED"}

        self._runner = runner

        runner.idle_tick_wait = 0.0
        self._next_progress_update = 0.0

        from ..pipeline import service
        self._area = context.area
        self._window_manager = context.window_manager
        self._timer = None
        self._ui_cleaned = False
        self._service_outcome = None
        try:
            self._timer = context.window_manager.event_timer_add(TIMER_INTERVAL, window=context.window)
            context.window_manager.progress_begin(0, 100)
            service.start(runner, window=context.window, view_layer=context.view_layer,
                          on_finish=self._service_finished,
                          on_detach=self._service_detached)
            context.window_manager.modal_handler_add(self)
        except Exception:
            service.detach(runner)
            self._cleanup(context)
            raise
        return {"RUNNING_MODAL"}

    def _service_finished(self):
        # Keep the WM timer until modal() returns a terminal result. Removing it
        # here strands Blender's handler until it consumes the next user event.
        # Even if reporting fails, the handler must still terminate.
        self._service_outcome = {"CANCELLED"}
        self._service_outcome = self._report_outcome(self._runner)

    def _service_detached(self):
        # Normal completion awaits one final modal event. External UI teardown
        # has no such event to await and must release its timer immediately.
        if self._service_outcome is None:
            self._cleanup(bpy.context)

    def modal(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        if self._runner.status.is_finished:
            self._cleanup(context)
            outcome = getattr(self, '_service_outcome', None)
            if outcome is None:
                outcome = self._report_outcome(self._runner)
            return outcome if event.type == "TIMER" else outcome | {"PASS_THROUGH"}
        if event.type in {"ESC"} and event.value == "PRESS":
            self._runner.cancel()
            return {"RUNNING_MODAL"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if getattr(event, "timer", self._timer) != self._timer:
            return {"PASS_THROUGH"}

        if context.window_manager.is_interface_locked:
            return {"RUNNING_MODAL"}

        now = time.monotonic()
        if now >= getattr(self, "_next_progress_update", 0.0):
            self._next_progress_update = now + PROGRESS_INTERVAL
            progress = self._runner.progress
            context.window_manager.progress_update(int(progress.fraction * 100))
            self._set_header(context, progress_text(runner=self._runner))

        return {"RUNNING_MODAL"}

    def cancel(self, context: bpy.types.Context) -> None:
        """Called by Blender if the modal operator is torn down from outside."""
        from ..pipeline import service
        if getattr(self, "_runner", None) is not None:
            service.detach(self._runner)
        self._cleanup(context)

    def _set_header(self, context: bpy.types.Context, text: str | None) -> None:
        area = getattr(self, "_area", None)
        if area is None:
            area = context.area
        if area is not None:
            try:
                area.header_text_set(text)
            except (ReferenceError, RuntimeError):
                pass

    def _cleanup(self, context: bpy.types.Context) -> None:
        if getattr(self, '_ui_cleaned', False):
            return
        self._ui_cleaned = True
        manager = getattr(self, '_window_manager', None) or context.window_manager
        timer = getattr(self, "_timer", None)
        self._timer = None
        if timer is not None:
            try:
                manager.event_timer_remove(timer)
            except (ReferenceError, RuntimeError):
                pass
        try:
            manager.progress_end()
        except (ReferenceError, RuntimeError):
            pass
        try:
            self._set_header(context, None)
        except (ReferenceError, RuntimeError):
            pass


def _severe_thin_warning(context: bpy.types.Context, asset_name: str) -> str | None:
    """The panel's thin-surface advice, when it is severe and quads are asked."""
    from .panels import thin_advice, thin_message  # noqa: F401 — same words as the panel

    name = _asset_name(context, asset_name)
    from .panels import context_resolution
    ctx, diagnostic = context_resolution(context, name)
    if diagnostic:
        return None
    settings = context.scene.rebuilt
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
    if not verdict.severe:
        return None
    return thin_message(verdict, int(settings.target_quads))


class REBUILT_OT_confirm_quads(bpy.types.Operator):
    """Legacy confirmation entry point retained for existing scripts"""

    bl_idname = "rebuilt.confirm_quads"
    bl_label = "Review Thin Features"
    bl_options = {"REGISTER", "INTERNAL"}

    asset_name: bpy.props.StringProperty(options={"SKIP_SAVE"})
    only_stage: bpy.props.StringProperty(options={"SKIP_SAVE"})
    force: bpy.props.BoolProperty(options={"SKIP_SAVE"})
    message: bpy.props.StringProperty(options={"SKIP_SAVE"})

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        return context.window_manager.invoke_confirm(
            self,
            event,
            title=self.bl_label,
            message=self.message,
            confirm_text="Continue with quads",
            icon="WARNING",
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        return bpy.ops.rebuilt.run_pipeline(
            "INVOKE_DEFAULT",
            asset_name=self.asset_name,
            only_stage=self.only_stage,
            force=self.force,
            confirmed=True,
        )


def _update_review_wireframe(area, enabled: bool | None, completed: list[str]) -> None:
    """Review every successful Manual retopo, then show the bake unobstructed."""
    if enabled is None or ("retopo" if enabled else "bake") not in completed:
        return
    try:
        if area is None or area.type != "VIEW_3D":
            return
        overlay = area.spaces.active.overlay
        overlay.show_wireframes = enabled
        if enabled:
            overlay.show_overlays = True
        area.tag_redraw()
    except (ReferenceError, RuntimeError):
        pass


def _show_uv_review(ctx: AssetContext, area) -> None:
    """Inspect packed UVs in Manual; baking replaces and deletes the grid."""
    from ..bridge import uv_preview

    low = ctx.artifact("low")
    if low is None:
        return
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    _show_only_the_asset(ctx)
    uv_preview.show(low, ctx.name)
    try:
        if area is not None and area.type == "VIEW_3D":
            area.spaces.active.shading.type = "MATERIAL"
            area.tag_redraw()
    except (ReferenceError, RuntimeError):
        pass


def _show_seams_review(ctx: AssetContext, area) -> None:
    """Show Manual cuts in Edit Mode without orange selection covering them."""
    try:
        if area is None or area.type != "VIEW_3D":
            return
        overlay = area.spaces.active.overlay
    except (ReferenceError, RuntimeError):
        return
    low = ctx.artifact("low")
    layer = bpy.context.view_layer
    layer.update()
    if low is None or low.name not in layer.objects:
        return
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    _show_only_the_asset(ctx)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="EDGE")
    bpy.ops.mesh.select_all(action="DESELECT")
    overlay.show_overlays = True
    overlay.show_edge_seams = True
    area.tag_redraw()


def _show_only_the_asset(ctx: AssetContext) -> None:
    """Hide the source model and its working copies once a retopology exists."""
    from ..stages.cleanup import ROLE_CLEANED
    from ..stages.ingest import ROLE_PROXY, ROLE_WORKING

    low = ctx.artifact("low")
    view_layer = bpy.context.view_layer

    view_layer.update()
    if low is None or low.name not in view_layer.objects:
        return

    low.hide_set(False)
    for selected in list(bpy.context.selected_objects):
        selected.select_set(False)
    low.select_set(True)
    view_layer.objects.active = low

    for obj in [ctx.source_object] + [
        obj for role in (ROLE_WORKING, ROLE_CLEANED, ROLE_PROXY) for obj in ctx.artifacts(role)
    ]:
        if obj is not None:
            try:
                obj.hide_set(True)
            except RuntimeError:
                pass


class REBUILT_OT_next_seams(bpy.types.Operator):
    """Put the next planned cut on the mesh — another set of seams to look at before unwrapping. Cycles through what the seams stage measured"""

    bl_idname = "rebuilt.next_seams"
    bl_label = "Other Seams"
    bl_options = {"REGISTER", "UNDO"}

    asset_name: bpy.props.StringProperty(name="Asset", default="", options={"SKIP_SAVE"})

    @_report_context_errors
    def execute(self, context: bpy.types.Context) -> set[str]:
        from ..stages.retopo import ROLE_LOW
        from ..stages.seams import apply_cut, JUNK_ATTRIBUTE, LAYOUT_KEY
        from ..bridge.engines.cleanseams import seam_transaction

        name = _asset_name(context, self.asset_name)
        ctx = _selected_context(context, name)
        if ctx is None:
            self.report({"ERROR"}, "No asset selected. Use Selected Model first.")
            return {"CANCELLED"}
        report = ctx.report("seams")
        low = ctx.artifact(ROLE_LOW)
        if not report or not report.get("offers") or low is None:
            self.report({"ERROR"}, "Cut the seams first.")
            return {"CANCELLED"}
        offers = report["offers"]
        if len(offers) < 2:
            self.report({"INFO"}, "Only one cut was planned for this mesh.")
            return {"CANCELLED"}
        if low.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        choice = (int(report.get("choice", 0)) + 1) % len(offers)
        offer = offers[choice]
        with seam_transaction(low, JUNK_ATTRIBUTE, LAYOUT_KEY):
            apply_cut(low, offer["edges"], offer["junk"])
            report["choice"] = choice
            report["charts"] = offer.get("charts")
            report["islands"] = offer.get("islands")
            ctx.set_report("seams", report)

        downstream_stages = StageGraph().downstream("seams")
        if downstream_stages & set(ctx.cached_keys):
            ctx.invalidate_cache(downstream_stages)
        self.report(
            {"INFO"},
            f"Cut {choice + 1} of {len(offers)}: "
            + (f"{offer['charts']} charts, " if offer.get("charts") else "")
            + f"{offer['islands']} islands, stretch {offer['stretch_p95']:.2f}",
        )
        if context.scene.rebuilt.mode == "manual":
            _show_seams_review(ctx, context.area)
        return {"FINISHED"}


class REBUILT_OT_use_existing(bpy.types.Operator):
    """Accept the generated mesh's current seams or active UV map without recalculating them"""

    bl_idname = "rebuilt.use_existing"
    bl_label = "Use Existing"
    bl_options = {"REGISTER", "UNDO"}

    asset_name: bpy.props.StringProperty(options={"SKIP_SAVE"})
    stage: bpy.props.EnumProperty(items=[("seams", "Seams", ""), ("uv", "UVs", "")])

    @classmethod
    def poll(cls, context):
        if context.scene.rebuilt.mode != "manual":
            return False
        return not any(
            getattr(op, "bl_idname", "") in {"REBUILT_OT_run_pipeline", "rebuilt.run_pipeline"}
            for w in context.window_manager.windows
            for op in getattr(w, "modal_operators", ())
        )

    @_report_context_errors
    def execute(self, context):
        from ..pipeline.external import accept_existing

        name = _asset_name(context, self.asset_name)
        ctx = _selected_context(context, name)
        if ctx is None:
            self.report({"ERROR"}, "Select a Rebuilt model first.")
            return {"CANCELLED"}
        try:
            notes = accept_existing(ctx, context.scene.rebuilt.build_recipe(), self.stage)
        except ContextError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report(
            {"WARNING"} if notes else {"INFO"},
            " ".join(notes)
            if notes
            else (
                "Existing seams accepted. Continue with Unwrap."
                if self.stage == "seams"
                else "Existing UVs accepted without changes. Continue with Bake Maps."
            ),
        )
        return {"FINISHED"}


class REBUILT_OT_finish_here(bpy.types.Operator):
    """Keep the current result and remove temporary working objects. Continue later if needed"""

    bl_idname = "rebuilt.finish_here"
    bl_label = "Finish Here"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if context.scene.rebuilt.mode != "manual":
            return False
        if any(
            getattr(op, "bl_idname", "") in {"REBUILT_OT_run_pipeline", "rebuilt.run_pipeline"}
            for window in context.window_manager.windows
            for op in getattr(window, "modal_operators", ())
        ):
            return False
        from .panels import active_context
        ctx = active_context(context)
        return ctx is not None and ctx.can_finish_here()

    @_report_context_errors
    def execute(self, context):
        ctx = _selected_context(context)
        if ctx is None or not ctx.can_finish_here():
            return {"CANCELLED"}

        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        ctx.finish_here()
        _show_only_the_asset(ctx)
        self.report(
            {"INFO"}, "Current result kept. Working objects removed; you can continue later."
        )
        return {"FINISHED"}


class REBUILT_OT_export_asset(bpy.types.Operator):
    """Choose a destination and export the finished game asset"""

    bl_idname = "rebuilt.export_asset"
    bl_label = "Export"
    bl_options = {"REGISTER"}

    asset_name: bpy.props.StringProperty(name="Asset", default="", options={"SKIP_SAVE"})

    directory: bpy.props.StringProperty(
        name="Export Folder", subtype="DIR_PATH", options={"SKIP_SAVE"}
    )
    filter_folder: bpy.props.BoolProperty(default=True, options={"HIDDEN"})

    def invoke(self, context, event):
        from pathlib import Path
        from ..pipeline.export_context import ExportContext
        import os

        self.asset_name = _asset_name(context, self.asset_name)
        ctx = ExportContext.find(context.scene, self.asset_name, context.active_object) if self.asset_name else None
        if ctx is None or ctx.report("output") is None:
            self.report({"ERROR"}, "Finish the asset before exporting it.")
            return {"CANCELLED"}
        previous = (ctx.report("export") or {}).get("destination_folder")
        default = (
            Path(bpy.data.filepath).parent / ctx.recipe.output.export_folder
            if bpy.data.filepath
            else Path.home()
        )
        self.directory = str(previous or default) + os.sep
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        from ..stages.output import export_files
        from ..pipeline.export_context import ExportContext

        name = _asset_name(context, self.asset_name)
        ctx = ExportContext.find(context.scene, name, context.active_object) if name else None
        if ctx is None:
            self.report({"ERROR"}, "Select a finished Rebuilt asset to export.")
            return {"CANCELLED"}
        if ctx.report("output") is None:
            self.report({"ERROR"}, "Finish the asset before exporting it.")
            return {"CANCELLED"}

        settings = context.scene.rebuilt
        try:
            report = export_files(
                ctx,
                settings.target_engine,
                settings.export_format,
                unity_pipeline=settings.unity_pipeline,
                destination_folder=self.directory or None,
            )
        except ContextError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        for note in report["notes"]:
            self.report({"WARNING"} if "failed" in note.lower() else {"INFO"}, note)
        if not report["exported"]:
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Exported {report.get('asset_folder') or report.get('unity_package') or report.get('unreal_package') or report['exported']}",
        )
        return {"FINISHED"}


class REBUILT_OT_show_uvs(bpy.types.Operator):
    """Open a UV Editor beside the 3D view to inspect the unwrapped result"""

    bl_idname = "rebuilt.show_uvs"
    bl_label = "Show UVs"
    bl_options = {"REGISTER"}

    asset_name: bpy.props.StringProperty(name="Asset", default="", options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        if (
            context.window is None
            or context.area is None
            or context.area.type != "VIEW_3D"
            or context.scene.rebuilt.mode != "manual"
        ):
            return False
        if any(
            getattr(op, "bl_idname", "") in {"REBUILT_OT_run_pipeline", "rebuilt.run_pipeline"}
            for window in context.window_manager.windows
            for op in getattr(window, "modal_operators", ())
        ):
            return False
        from .panels import active_context
        ctx = active_context(context)
        low = ctx.artifact("low") if ctx else None
        return bool(low and "uv" in ctx.cached_keys and low.data.uv_layers.active)

    @_report_context_errors
    def execute(self, context):
        name = _asset_name(context, self.asset_name)
        ctx = _selected_context(context, name)
        low = ctx.artifact("low") if ctx else None
        if low is None or "uv" not in ctx.cached_keys or low.data.uv_layers.active is None:
            self.report({"WARNING"}, "Run Unwrap before showing UVs.")
            return {"CANCELLED"}
        from .uv_review import open_editor

        try:
            open_editor(ctx, context)
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class REBUILT_OT_show_result(bpy.types.Operator):
    """Select the generated mesh and show the QA heatmap on it"""

    bl_idname = "rebuilt.show_result"
    bl_label = "Show Result"
    bl_options = {"REGISTER", "UNDO"}

    asset_name: bpy.props.StringProperty(name="Asset", default="")

    field: bpy.props.StringProperty(name="Field", default="")

    @_report_context_errors
    def execute(self, context: bpy.types.Context) -> set[str]:
        from ..stages.qa import focus_on

        name = _asset_name(context, self.asset_name)
        ctx = _selected_context(context, name)
        if ctx is None:
            self.report({"ERROR"}, "No asset selected.")
            return {"CANCELLED"}

        focus_on(ctx, context)
        _show_attribute_colours(context)
        return {"FINISHED"}


def _show_attribute_colours(context: bpy.types.Context) -> None:
    """Switch the viewport to painting the active colour attribute."""
    for area in getattr(context.screen, "areas", ()):
        if area.type != "VIEW_3D":
            continue
        for space in area.spaces:
            if space.type == "VIEW_3D":
                space.shading.type = "SOLID"
                space.shading.color_type = "VERTEX"


_CLASSES = (
    REBUILT_OT_texture_output_info,
    REBUILT_OT_adopt_scan,
    REBUILT_OT_adopt_results,
    REBUILT_OT_recover_asset,
    REBUILT_OT_run_pipeline,
    REBUILT_OT_confirm_quads,
    REBUILT_OT_next_seams,
    REBUILT_OT_use_existing,
    REBUILT_OT_finish_here,
    REBUILT_OT_export_asset,
    REBUILT_OT_show_uvs,
    REBUILT_OT_show_result,
)


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
