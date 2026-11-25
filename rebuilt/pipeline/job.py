"""Running a sequence of stages without freezing Blender."""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
import time
from typing import Any, Iterator, Sequence

from ..core.graph import StageGraph
from ..core.jobs import BackgroundTask, JobState, MainThreadCalls, Progress, TaskState
from ..stages.base import Stage
from .context import AssetContext, ContextError
from ..runtime.manifest import load_runtime_manifest, require
from ..runtime.preflight import preflight
from ..runtime.session import runtime_scope


IDLE_TICK_WAIT = 0.01


def progress_text(runner) -> str:
    """Show elapsed work without treating stage fractions as a time estimate."""
    now = time.monotonic()

    def elapsed(start):
        seconds = int(max(0, now - (start if start is not None else now)))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    progress = runner.progress
    stage = getattr(runner, "current_stage", "")
    prefix = f"{stage.title()} · " if stage else ""
    total = elapsed(getattr(runner, "_started_at", None))
    current = elapsed(getattr(runner, "_stage_started_at", None))
    return (
        f"Rebuilt: {prefix}{progress.message} — {total} elapsed · stage {current} "
        f"· steps {progress.fraction:.0%} (ESC to cancel)"
    )


class JobStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_finished(self) -> bool:
        return self in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)


_RUNNERS = set()


def active_runners(scene=None):
    return [r for r in _RUNNERS if r.status is JobStatus.RUNNING
            and (scene is None or r.ctx.scene == scene)]


class JobRunner:
    """Drives stages one at a time, each on a worker thread."""

    def __init__(
        self,
        ctx: AssetContext,
        stages: Sequence[Stage],
        graph: StageGraph | None = None,
        *,
        write_breadcrumbs: bool = True,
        reset_on_cancel: bool = False,
        recipe=None,
        invalidate_stages: frozenset[str] = frozenset(),
    ) -> None:
        self.ctx = ctx
        self.stages = list(stages)
        self.graph = graph or StageGraph()
        self.write_breadcrumbs = write_breadcrumbs
        self.reset_on_cancel = reset_on_cancel
        self._pending_recipe = recipe
        self._invalidate_stages = invalidate_stages
        self.idle_tick_wait = IDLE_TICK_WAIT

        self.status = JobStatus.IDLE
        self.error: BaseException | None = None
        self.formatted_error = ""
        self.completed: list[str] = []

        self._index = 0
        self._task: BackgroundTask | None = None
        self._state = JobState(asset=ctx.name, stages=[s.name for s in self.stages])
        self._message = ""
        self._apply_progress = 0.0
        self._started_at = None
        self._stage_started_at = None
        self._runtime = None
        self._breadcrumb_path = None
        self._preflighting = False
        self._work_started = False
        self._file_owner = None
        self._cancel_requested = False
        self._aborted = False
        self._first_retopology = (
            any(stage.name == "retopo" for stage in self.stages)
            and ctx.artifact("low") is None
            and ctx.report("retopo") is None
        )

    def start(self) -> None:
        if self.status is not JobStatus.IDLE:
            raise RuntimeError(f"job for {self.ctx.name!r} was already started")
        if self._cancel_requested:
            self._finish(JobStatus.CANCELLED)
            return
        if not self.stages:
            self._finish(JobStatus.DONE)
            return
        try:
            self.ctx.require_owned()
        except ContextError as error:
            self._fail(error)
            return
        self.status = JobStatus.RUNNING
        _RUNNERS.add(self)
        self._started_at = time.monotonic()
        if self.write_breadcrumbs:
            self._breadcrumb_path = self.ctx.job_state_path
        if any(stage.native_engines for stage in self.stages):
            self._preflighting = True
            self._message = "Checking required components"

            def check_runtime(report):
                report.check_cancelled()
                try:
                    return preflight(cancelled=lambda: report.cancelled)
                except Exception:
                    # The process runner has already drained its children.
                    # Translate its cancellation diagnostic to the job state.
                    report.check_cancelled()
                    raise

            self._task = BackgroundTask(check_runtime, name="preflight")
            self._task.start()
        else:
            self._start_work()

    def _start_work(self) -> None:
        try:
            if self.reset_on_cancel:
                import bpy
                from .cancel_files import CancelFiles

                self.ctx.cancel_files = CancelFiles(
                    bpy.utils.user_resource('CONFIG', path='rebuilt/recovery', create=True),
                    before_first_write=self.ctx.auto_archive.files_touched if self.ctx.auto_archive is not None else None)
                self._file_owner = self.ctx.cancel_files
                self._file_owner.hold()
                try:
                    if self.ctx.auto_archive is not None:
                        self.ctx.auto_archive.start()
                except Exception:
                    self.ctx.cancel_files.close()
                    self.ctx.cancel_files = None
                    raise
                self.ctx._ownership.record()['file_journal'] = self.ctx.cancel_files.descriptor
                if self.ctx.auto_archive is not None:
                    self.ctx.auto_archive.snapshot['file_journal'] = self.ctx.cancel_files.descriptor
            self._work_started = True
            if self._pending_recipe is not None:
                self.ctx.recipe = self._pending_recipe
            if self._invalidate_stages:
                self.ctx.invalidate_cache(self._invalidate_stages)
            self._begin_stage()
        except Exception as error:
            self._fail(error)

    def tick(self) -> bool:
        """Advance the job. Returns True while there is still work to do."""
        if self.status is not JobStatus.RUNNING:
            return False

        if not self._cancel_requested:
            try:
                self.ctx.source_object
            except ContextError as error:
                self.abort(error)

        task = self._task
        if task is None:
            self._finish(JobStatus.FAILED if self._aborted else
                         JobStatus.CANCELLED if self._cancel_requested else JobStatus.DONE)
            return False

        if not task.is_finished:
            if self._preflighting or self._cancel_requested:
                task.join(self.idle_tick_wait)
                return True
            stage = self.stages[self._index]
            if stage.main_thread_calls is not None:
                handled = stage.main_thread_calls.poll(
                    self._compute_on_main_thread, timeout=self.idle_tick_wait
                )
                if handled and self.idle_tick_wait == 0.0:
                    deadline = time.monotonic() + 0.003
                    while time.monotonic() < deadline:
                        if not stage.main_thread_calls.poll(
                            self._compute_on_main_thread,
                            timeout=min(0.001, max(0.0, deadline - time.monotonic())),
                        ):
                            break
            else:
                task.join(self.idle_tick_wait)
            return True

        if self._cancel_requested or task.state is TaskState.CANCELLED:
            self._finish(JobStatus.FAILED if self._aborted else JobStatus.CANCELLED)
            return False

        if task.state is TaskState.FAILED:
            self.error = task.error
            self.formatted_error = task.formatted_error
            self._finish(JobStatus.FAILED)
            return False

        if self._preflighting:
            self._runtime = task.result
            self._preflighting = False
            self._task = None
            self._start_work()
            return self.status is JobStatus.RUNNING

        stage = self.stages[self._index]
        try:
            self.ctx.require_owned()
            if self._runtime is not None:
                current = load_runtime_manifest(self._runtime.manifest.root)
                require(current.identity == self._runtime.manifest.identity,
                        'Runtime changed during the job; no result was applied. Restart the job.')
            from .stage_operation import stage_operation
            with stage_operation(self.ctx, stage.name):
                with runtime_scope(self._runtime.manifest if self._runtime else None), self._reporting_apply(stage):
                    stage.apply(self.ctx, task.result)
                from ..stages import cache_keys
                key = cache_keys(self.ctx.recipe, [stage],
                                 runtime=self._runtime.manifest if self._runtime else None,
                                 ctx=self.ctx)[stage.name]
                self.ctx.set_cached_key(stage.name, key)
                self.ctx.resources.collect_unused()
        except BaseException as exc:  # noqa: BLE001 — apply failures must surface
            self._fail(exc)
            return False

        self.completed.append(stage.name)
        self._state.mark_complete(stage.name)
        self._index += 1
        self._task = None

        if self._index >= len(self.stages):
            self._finish(JobStatus.DONE)
            return False

        self._begin_stage()
        return True

    def cancel(self) -> None:
        """Cancel the job, including gaps between worker completion and apply."""
        self._cancel_requested = True
        if self._task is not None:
            self._task.cancel()

    def abort(self, error: BaseException) -> None:
        """Record a driver failure without dropping an in-flight worker."""
        import traceback
        if self.status.is_finished:
            return
        self._aborted = True
        self.error = error
        self.formatted_error = ''.join(traceback.format_exception(error))
        self.cancel()
        if self.status is JobStatus.IDLE:
            self._finish(JobStatus.FAILED)

    @property
    def stopping(self) -> bool:
        return self._cancel_requested

    def worker_drained(self) -> bool:
        """Observe termination without touching any Blender data."""
        return self._task is None or self._task.join(0)

    def drain_cancelled(self) -> bool:
        """Finish cancelled work in the caller's owning scene context."""
        if not self._cancel_requested:
            raise RuntimeError('Cannot drain a job that has not been stopped')
        if not self.worker_drained():
            return True
        self._finish(JobStatus.FAILED if self._aborted else JobStatus.CANCELLED)
        return False

    def finish_without_scene(self) -> bool:
        """Fail after scene deletion, restoring owned files without scene mutation."""
        if not self._aborted:
            raise RuntimeError('Missing-scene finalization requires an aborted job')
        if not self.worker_drained():
            return True
        try:
            if self._work_started and self.ctx.cancel_files is not None:
                self.ctx.cancel_files.rollback()
                self.ctx.cancel_files = None
        except Exception as error:
            import traceback
            self.error = RuntimeError(f'Original scene is unavailable; file restoration failed: {error}')
            self.formatted_error += '\n' + traceback.format_exc()
        finally:
            self._release_file_owner()
        self.status = JobStatus.FAILED
        self._state.finish(self.status.value)
        self._save_breadcrumb()
        _RUNNERS.discard(self)
        return False

    def _release_file_owner(self):
        if self._file_owner is not None:
            owner, self._file_owner = self._file_owner, None
            owner.release()

    @contextmanager
    def _reporting_apply(self, stage: Stage) -> Iterator[None]:
        """Let a stage say what it is doing while it blocks the main thread."""
        self._apply_progress = 0.0

        def report(fraction: float, message: str = "") -> None:
            self._apply_progress = min(max(fraction, 0.0), 1.0)
            if message:
                self._message = message
            self._nudge_interface()

        previous = self.ctx.progress
        self.ctx.progress = report
        try:
            self._message = stage.label
            yield
        finally:
            self.ctx.progress = previous
            self._apply_progress = 0.0

    def _nudge_interface(self) -> None:
        """Push the current fraction and message at Blender, if there is an interface at all — and make it draw them."""
        try:
            import bpy

            if bpy.app.background:
                return
            manager = bpy.context.window_manager
            if manager is not None:
                manager.progress_update(int(self.progress.fraction * 100))
            for window in manager.windows if manager is not None else ():
                for area in window.screen.areas:
                    if area.type == "VIEW_3D":
                        area.header_text_set(progress_text(self))

                        header = next((r for r in area.regions if r.type == "HEADER"), None)
                        if header is not None:
                            with bpy.context.temp_override(window=window, area=area, region=header):
                                bpy.ops.wm.redraw_timer(type="DRAW_SWAP", iterations=1)
        except Exception:  # noqa: BLE001 — reporting is a courtesy, never a failure
            pass

    def _begin_stage(self) -> None:
        self._stage_started_at = time.monotonic()
        stage = self.stages[self._index]
        stage.main_thread_calls = MainThreadCalls()
        self._state.mark_stage(stage.name)
        self._save_breadcrumb()
        self._message = stage.label

        expected = self._runtime.manifest if self._runtime else None
        try:
            self.ctx.require_owned()
            with runtime_scope(expected):
                payload = stage.gather(self.ctx)
        except BaseException as exc:  # noqa: BLE001 — gather failures must surface
            self._fail(exc)
            return

        def compute(report):
            with runtime_scope(expected):
                return stage.compute(payload, report)

        task = BackgroundTask(compute, name=stage.name)
        self._task = task
        task.start()

    def _compute_on_main_thread(self, payload):
        self.ctx.require_owned()
        with runtime_scope(self._runtime.manifest if self._runtime else None):
            return self.stages[self._index].compute_on_main_thread(payload)

    def _fail(self, exc: BaseException) -> None:
        import traceback

        self.error = exc
        self.formatted_error = traceback.format_exc()
        self._task = None
        self._finish(JobStatus.FAILED)

    def _finish(self, status: JobStatus) -> None:
        try:
            self._finish_work(status)
        except Exception as error:
            import traceback
            self.status = JobStatus.FAILED
            self.error = RuntimeError(f'Could not finish job cleanup: {error}')
            self.formatted_error = traceback.format_exc()
            self._state.finish(self.status.value)
            self._save_breadcrumb()
        finally:
            if self.status.is_finished:
                _RUNNERS.discard(self)
                self._release_file_owner()
                # Persistent recovery records survive; a finished runner no longer
                # grants its context authority to mutate a retained transaction.
                if self.reset_on_cancel:
                    self.ctx._ownership.archive = None
                    self.ctx.auto_archive = None

    def _finish_work(self, status: JobStatus) -> None:
        if not self._work_started:
            # Preflight owns no scene changes, archive or output files.
            self.status = status
            self._state.finish(status.value)
            return
        auto_completed = False
        if self.reset_on_cancel and status is JobStatus.DONE:
            from .auto_archive import complete_auto
            from .cancel_files import FileCommitUncertain
            try:
                complete_auto(self.ctx)
                auto_completed = True
            except FileCommitUncertain:
                # Retain both scene/file versions until the decision is known.
                raise
            except Exception as error:
                import traceback
                self.error = error
                self.formatted_error = traceback.format_exc()
                status = JobStatus.FAILED
        restore_auto = self.reset_on_cancel and status in (JobStatus.CANCELLED, JobStatus.FAILED)
        cleanup_failed = False
        if restore_auto:
            try:
                if self.ctx.auto_archive is not None:
                    self.ctx.auto_archive.validate_restore()
                    if self.ctx.cancel_files is not None:
                        self.ctx.cancel_files.rollback()
                    self.ctx.auto_archive.restore()
                else:
                    from .stage_operation import borrowed_objects, discard_borrowed
                    borrowed_objects(self.ctx)
                    if self.ctx.cancel_files is not None:
                        self.ctx.cancel_files.rollback()
                    discard_borrowed(self.ctx)
                    self.ctx.reset_generated()
            except Exception as exc:
                import traceback

                cleanup_failed = True
                status = JobStatus.FAILED
                self.error = RuntimeError(f"Could not fully discard the Auto attempt: {exc}")
                self.formatted_error = traceback.format_exc()
        elif (
            status is JobStatus.CANCELLED
            and self._first_retopology
            and "retopo" not in self.completed
        ):
            self.ctx.discard_preparation()
        if self.reset_on_cancel and not cleanup_failed and not auto_completed and self.ctx.auto_archive is not None:
            self.ctx.auto_archive = None
        if self.reset_on_cancel and not cleanup_failed and not auto_completed and self.ctx.cancel_files is not None:
            self.ctx.cancel_files.close()
            self.ctx.cancel_files = None
            record = self.ctx._ownership.record()
            if record is not None and 'file_journal' in record:
                del record['file_journal']
        self.status = status
        self._state.finish(status.value)
        self._save_breadcrumb()
        if (
            status is JobStatus.DONE or (status is JobStatus.CANCELLED and self.reset_on_cancel)
        ) and self.write_breadcrumbs:
            self._state_path_unlink()

    def _save_breadcrumb(self) -> None:
        if not self.write_breadcrumbs or self._breadcrumb_path is None:
            return
        try:
            self._state.save(self._breadcrumb_path)
        except (OSError, ReferenceError, RuntimeError):
            # The source/scene may have been removed during external teardown.
            pass

    def _state_path_unlink(self) -> None:
        if self._breadcrumb_path is None:
            return
        try:
            self._breadcrumb_path.unlink(missing_ok=True)
        except (OSError, ReferenceError, RuntimeError):
            pass

    @property
    def current_stage(self) -> str:
        if (self.status is not JobStatus.RUNNING or self._preflighting
                or self._index >= len(self.stages)):
            return ""
        return self.stages[self._index].name

    @property
    def progress(self) -> Progress:
        """Overall progress: finished stages plus how far the current one is."""
        total = len(self.stages)
        if total == 0:
            return Progress(1.0, "")

        if self._apply_progress > 0.0:
            within = Progress(0.5 + 0.5 * self._apply_progress, self._message)
        elif self._task is not None:
            task = self._task.progress
            within = Progress(0.5 * task.fraction, task.message)
        else:
            within = Progress(0.0, "")

        fraction = (self._index + within.fraction) / total
        message = within.message or self._message
        return Progress(min(1.0, fraction), message)

    def result_summary(self) -> dict[str, Any]:
        return {
            "asset": self.ctx.name,
            "status": self.status.value,
            "completed": list(self.completed),
            "error": self.formatted_error or None,
        }
