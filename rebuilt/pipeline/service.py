"""Own interactive work until it has drained, independently of modal UI."""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import logging

import bpy

from .job import JobStatus, active_runners

_INTERVAL = .005
_entries = {}
_polling = False
_log = logging.getLogger(__name__)


@dataclass
class _Entry:
    runner: object
    window: object = None
    view_layer: object = None
    on_finish: object = None
    on_detach: object = None


class _SceneUnavailable(RuntimeError):
    pass


def _window_present(window):
    if window is None:
        return True
    try:
        return any(candidate == window for candidate in bpy.context.window_manager.windows)
    except (ReferenceError, RuntimeError):
        return False


@contextmanager
def _owned_context(entry):
    try:
        scene = entry.runner.ctx.scene
        if not any(candidate == scene for candidate in bpy.data.scenes):
            raise _SceneUnavailable('The job scene was removed; stopping without changing the current scene.')
        layers = list(scene.view_layers)
    except ReferenceError as error:
        raise _SceneUnavailable('The job scene was removed; stopping without changing the current scene.') from error
    try:
        valid_layer = entry.view_layer in layers
    except ReferenceError:
        valid_layer = False
    if not valid_layer:
        entry.view_layer = layers[0]
    window_context = bpy.context.temp_override(window=entry.window) if entry.window is not None else nullcontext()
    # A simultaneous window+scene override selects the window's scene in Blender.
    # Nest scene ownership inside window ownership so another selected scene stays intact.
    with window_context:
        with bpy.context.temp_override(scene=scene, view_layer=entry.view_layer):
            yield


def _notify(callback):
    if callback is not None:
        try:
            callback()
        except Exception:
            _log.exception('Could not release or update Rebuilt job UI')


def _release(entry):
    try:
        _notify(entry.on_finish)
    finally:
        _notify(entry.on_detach)
        _entries.pop(entry.runner, None)


def _register():
    if not bpy.app.timers.is_registered(_timer):
        bpy.app.timers.register(_timer, first_interval=_INTERVAL, persistent=True)


def start(runner, *, window=None, view_layer=None, on_finish=None, on_detach=None):
    """Acquire ownership before starting any worker or scene mutation."""
    if runner in _entries or runner.status is not JobStatus.IDLE:
        raise RuntimeError('Job is already owned or started')
    _entries[runner] = _Entry(runner, window, view_layer, on_finish, on_detach)
    try:
        _register()
    except Exception:
        _entries.pop(runner)
        raise
    runner.idle_tick_wait = 0.0
    try:
        runner.start()
    except Exception as error:
        runner.abort(error)
    if runner.status.is_finished:
        _release(_entries[runner])


def detach(runner):
    """Release UI now; retain the cancelled job until worker and cleanup finish."""
    runner.cancel()
    entry = _entries.get(runner)
    if entry is not None:
        callback = entry.on_detach
        entry.on_finish = entry.on_detach = None
        entry.window = None
        _notify(callback)
    if runner.status is JobStatus.IDLE:
        runner.drain_cancelled()  # No worker, archive or scene mutation was started.
    if runner.status.is_finished:
        _entries.pop(runner, None)
        return
    if entry is None:
        _entries[runner] = _Entry(runner)
    _register()


def shutdown():
    """Cancel all work before UI classes/properties disappear; keep draining."""
    for runner in set(active_runners()) | _entries.keys():
        detach(runner)


def _step():
    global _polling
    if _polling:
        return _INTERVAL
    _polling = True
    try:
        for runner, entry in tuple(_entries.items()):
            if runner.status.is_finished:
                _release(entry)
                continue
            if not _window_present(entry.window):
                detach(runner)
            if bpy.context.window_manager.is_interface_locked:
                continue
            try:
                with _owned_context(entry):
                    if runner.stopping:
                        runner.drain_cancelled()
                    else:
                        runner.tick()
                    if runner.status.is_finished:
                        _release(entry)
            except _SceneUnavailable as error:
                runner.abort(error)
                if not runner.finish_without_scene():
                    _release(entry)
            except Exception as error:
                # Never tick outside the owner context or let one driver error
                # unregister the application timer for the other jobs.
                runner.abort(error)
        return _INTERVAL if _entries else None
    finally:
        _polling = False


def _timer():
    return _step()


def poll():
    """Drive the same owner in tests/headless callers without an event loop."""
    result = _step()
    if result is None and bpy.app.timers.is_registered(_timer):
        bpy.app.timers.unregister(_timer)
    return result
