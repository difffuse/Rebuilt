"""Understand the scan before touching its topology."""

from __future__ import annotations

from typing import Any, ClassVar

from ..bridge import attributes, mesh_io
from ..core.analysis.classify import GENERIC_ORGANIC, UNKNOWN
from ..core.analysis.graph import AnalysisGraph, analyse
from ..core.analysis.fields import ImportanceWeights
from ..core.jobs import ProgressReporter
from ..core.recipe import AssetRecipe
from ..pipeline.context import AssetContext, ContextError
from .base import Stage
from .cleanup import current_mesh_object
from .ingest import ROLE_PROXY

ROLE_ANALYSIS = "analysis"


class AnalyzeStage(Stage):
    name: ClassVar[str] = "analyze"
    slot: ClassVar[str] = "analysis"
    role: ClassVar[str] = ROLE_ANALYSIS
    label: ClassVar[str] = "Analysing model"

    def cache_key(self, recipe: AssetRecipe) -> str:
        return recipe.analyze.fingerprint() + ":uncertain-profile-1"

    def gather(self, ctx: AssetContext) -> dict[str, Any]:

        source = ctx.artifact(ROLE_PROXY) or current_mesh_object(ctx)
        if source is None:
            raise ContextError("analyze needs ingest to have run first")

        return {
            "arrays": mesh_io.mesh_to_arrays(source.data),
            "recipe": ctx.recipe.analyze,
            "source_name": source.name,
        }

    def compute(self, payload: dict[str, Any], report: ProgressReporter) -> dict[str, Any]:
        recipe = payload["recipe"]

        graph = analyse(
            payload["arrays"],
            hard_edge_angle_deg=recipe.hard_edge_angle_deg,
            coplanar_angle_deg=recipe.coplanar_angle_deg,
            curvature_percentile=recipe.curvature_percentile,
            weights=ImportanceWeights(recipe.curvature_weight, recipe.feature_weight),
            detect_symmetry=recipe.detect_symmetry,
            symmetry_extra_directions=recipe.symmetry_extra_directions,
            report=report,
        )
        return {"graph": graph, "override": recipe.asset_class_override}

    def apply(self, ctx: AssetContext, result: dict[str, Any]) -> None:
        graph: AnalysisGraph = result["graph"]
        summary = graph.summary()

        override = result["override"]
        if override:
            summary["detected_class"] = summary["asset_class"]
            summary["asset_class"] = override
            summary["confidence"] = 1.0
            summary["needs_review"] = False
            summary["overridden"] = True
        elif summary.get("needs_review") and summary["asset_class"] != GENERIC_ORGANIC:
            summary["detected_class"] = summary["asset_class"]
            summary["asset_class"] = GENERIC_ORGANIC
            summary["automatic_fallback"] = "Low-confidence classification; using the general organic profile."

        ctx.set_report(self.name, summary)
        self._adopt_detected_profile(ctx, summary.get("asset_class", UNKNOWN))
        self._write_fields(ctx, graph)

    def _adopt_detected_profile(self, ctx: AssetContext, detected: str) -> None:
        """Rebuild the recipe around what the scan turned out to be."""
        if ctx.recipe_for_class is None or not detected or detected == UNKNOWN:
            return
        ctx.recipe = ctx.recipe_for_class(detected)

    def _write_fields(self, ctx: AssetContext, graph: AnalysisGraph) -> None:
        """Put the per-vertex fields on the mesh the analysis actually ran on."""
        target = ctx.artifact(ROLE_PROXY) or current_mesh_object(ctx)
        if target is None or graph.curvature.size == 0:
            return

        mesh = target.data
        if len(mesh.vertices) != graph.curvature.shape[0]:
            return

        attributes.write_float(mesh, attributes.CURVATURE, graph.curvature)
        attributes.write_float(mesh, attributes.IMPORTANCE, graph.importance)
        attributes.write_heatmap(mesh, graph.importance)
        if graph.face_region.shape[0] == len(mesh.polygons):
            attributes.write_int(mesh, attributes.REGION, graph.face_region)


def asset_class_of(ctx: AssetContext) -> str:
    """The asset class decided for this asset, or `unknown`."""
    report = ctx.report("analyze")
    return report.get("asset_class", UNKNOWN) if report else UNKNOWN
