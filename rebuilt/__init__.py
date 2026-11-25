"""Rebuilt — Retopo, UV and Bake for high-poly models."""

from . import ui


def register() -> None:
    # The package builder sets this marker for verified-runtime candidates.
    # Explicit legacy baselines retain their original installation behavior.
    if globals().get('_REBUILT_RUNTIME_REQUIRED', False):
        from .runtime import prepare_runtime
        prepare_runtime()
    ui.register()


def unregister() -> None:
    from .pipeline.service import shutdown
    shutdown()
    ui.unregister()
