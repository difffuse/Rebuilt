"""The verdict, and the heatmaps that show where it came from."""

from __future__ import annotations

from typing import Any, ClassVar

import bpy
import numpy as np

from ..bridge import attributes, bvh, mesh_io
from ..core.jobs import ProgressReporter
from ..core.qa.scorecard import Scorecard, build
from ..core.recipe import AssetRecipe
from ..core.types import MeshArrays
from ..pipeline.context import AssetContext, ContextError
from ..core import profiles
from .base import Stage
from .cleanup import current_mesh_object
from .retopo import ROLE_LOW


HEATMAP_DEVIATION = "rebuilt_qa_deviation"


class QAStage(Stage):
    name: ClassVar[str] = "qa"
    slot: ClassVar[str] = "qa"
    role: ClassVar[str] = "qa"
    label: ClassVar[str] = "Scoring"

    def cache_key(self, recipe: AssetRecipe) -> str:
        return f"{recipe.qa.fingerprint()}:{recipe.profile}"

    def gather(self, ctx: AssetContext) -> dict[str, Any]:
        low = ctx.artifact(ROLE_LOW)
        if low is None:
            raise ContextError("qa needs retopo to have run first")
        high = current_mesh_object(ctx)

        return {
            "low": mesh_io.mesh_to_arrays(low.data, low.matrix_world),
            "high": mesh_io.mesh_to_arrays(high.data, high.matrix_world) if high else None,
            "reports": {
                stage: ctx.report(stage)
                for stage in ("ingest", "cleanup", "analyze", "retopo", "uv", "bake")
            },
            "recipe": ctx.recipe,
        }

    def compute(self, payload: dict[str, Any], report: ProgressReporter) -> dict[str, Any]:
        recipe: AssetRecipe = payload["recipe"]

        report.update(0.2, "Measuring deviation")
        deviation = _per_vertex_deviation(payload["low"], payload["high"])

        report.update(0.7, "Scoring")
        try:
            limits = dict(profiles.get(recipe.profile).limits)
        except profiles.ProfileError:
            limits = {}

        limits["max_deviation_p95"] = recipe.qa.max_deviation_p95
        limits["max_uv_stretch_p95"] = recipe.qa.max_uv_stretch_p95
        limits["min_valid_bake_pixels"] = recipe.qa.min_valid_bake_pixels
        limits["allow_uv_overlap"] = float(recipe.qa.allow_uv_overlap)

        scorecard = build(
            payload["reports"],
            limits=limits,
            warnings_as_errors=recipe.qa.warnings_as_errors,
        )

        report.update(1.0, "Ready")
        return {"scorecard": scorecard, "deviation": deviation}

    def apply(self, ctx: AssetContext, result: dict[str, Any]) -> None:
        low = ctx.artifact(ROLE_LOW)
        scorecard: Scorecard = result["scorecard"]
        deviation: np.ndarray | None = result["deviation"]

        if deviation is not None and deviation.size == len(low.data.vertices):
            attributes.write_float(low.data, HEATMAP_DEVIATION, deviation)
            peak = float(deviation.max())
            attributes.write_heatmap(low.data, deviation / peak if peak > 0 else deviation)

        payload = scorecard.to_dict()
        payload["issues"] = [check.to_dict() for check in scorecard.issues()]
        ctx.set_report(self.name, payload)


def _per_vertex_deviation(low: MeshArrays, high: MeshArrays | None) -> np.ndarray | None:
    """How far each low vertex sits from the scan, relative to the asset's size."""
    if high is None or low.num_verts == 0:
        return None

    diagonal = float(np.linalg.norm(high.positions.max(axis=0) - high.positions.min(axis=0)))
    if diagonal <= 0:
        return None

    distances = bvh.SurfaceQuery(high).nearest_distances(low.positions.astype(np.float64))
    distances = np.where(np.isfinite(distances), distances, 0.0)
    return distances / diagonal


def focus_on(ctx: AssetContext, context: bpy.types.Context) -> None:
    """Select the low mesh and show the heatmap, so an issue can be looked at."""
    low = ctx.artifact(ROLE_LOW)
    if low is None:
        return
    for obj in context.view_layer.objects:
        obj.select_set(False)
    low.select_set(True)
    context.view_layer.objects.active = low
