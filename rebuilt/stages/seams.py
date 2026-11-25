"""Generate editable seams and validated corner UVs with Clean Seams."""

from __future__ import annotations

from ..runtime.session import verify_current_runtime

from typing import Any, ClassVar

import bpy
import numpy as np

from ..bridge import attributes, mesh_io, uv_io
from ..core.jobs import ProgressReporter
from ..core.recipe import AssetRecipe
from ..core.uv import cleanseams_native, seam_engine
from ..bridge.engines.cleanseams import seam_transaction
from ..core.uv.protected import cut_corners, cuts_preserved
from ..core.uv.metrics import (
    island_count_from_uvs,
    overlap_report,
    stretch_report,
)
from ..pipeline.context import AssetContext, ContextError
from .analyze import asset_class_of
from .base import Stage
from .retopo import ROLE_LOW

JUNK_ATTRIBUTE = "rebuilt_uv_junk"
LAYOUT_KEY = "rebuilt_seam_layout"

def layout_fingerprint(low, arrays=None) -> str:
    """Identify the precise mesh, cut and UVs whose contour work was validated."""
    arrays = arrays if arrays is not None else mesh_io.mesh_to_arrays(low.data)
    pairs = np.empty((len(low.data.edges), 2), dtype=np.int32)
    flags = np.empty(len(low.data.edges), dtype=bool)
    low.data.edges.foreach_get("vertices", pairs.reshape(-1))
    low.data.edges.foreach_get("use_seam", flags)
    return seam_engine.fingerprint(arrays, pairs[flags], uv_io.read_uvs(low.data))


def marked_edges(mesh):
    pairs = np.empty((len(mesh.edges), 2), dtype=np.int32)
    flags = np.empty(len(mesh.edges), dtype=bool)
    mesh.edges.foreach_get("vertices", pairs.reshape(-1))
    mesh.edges.foreach_get("use_seam", flags)
    return pairs[flags]


class SeamsStage(Stage):
    name: ClassVar[str] = "seams"
    slot: ClassVar[str] = "uv"
    role: ClassVar[str] = "seams"
    label: ClassVar[str] = "Cutting seams"
    native_engines = ('cleanseams',)

    def cache_key(self, recipe: AssetRecipe, *, runtime=None) -> str:

        key = f"{seam_engine.CACHE_VERSION}:{recipe.profile}:{recipe.analyze.fingerprint()}:{recipe.uv.max_stretch}:{recipe.uv.unwrap_method}"
        key += ":fewer-1" if recipe.uv.island_preference == "FEWER" else ""
        key += (":marked-1" if recipe.uv.keep_marked_seams else "") + ":native-apply-1:repair-v1"
        return self._with_runtime(key, recipe, runtime)

    def gather(self, ctx: AssetContext) -> dict[str, Any]:
        low = ctx.artifact(ROLE_LOW)
        if low is None:
            raise ContextError("seams need retopo to have run first")
        if low.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        return {
            "arrays": mesh_io.mesh_to_arrays(low.data),
            "asset_class": asset_class_of(ctx),
            "max_stretch": ctx.recipe.uv.max_stretch,
            "unwrap_method": ctx.recipe.uv.unwrap_method,
            "island_preference": ctx.recipe.uv.island_preference,
            "keep_marked_seams": ctx.recipe.uv.keep_marked_seams,
            "protected_edges": marked_edges(low.data) if ctx.recipe.uv.keep_marked_seams else [],
        }

    def compute(self, payload: dict[str, Any], report: ProgressReporter) -> dict[str, Any]:
        result = seam_engine.plan(
            payload["arrays"],
            payload["asset_class"],
            report,
            main_thread_calls=self.main_thread_calls,
            unwrap_method=payload["unwrap_method"],
            island_preference=payload.get("island_preference", "BALANCED"),
            protected_edges=payload.get("protected_edges", []),
        )
        result["max_stretch"] = payload["max_stretch"]
        result["unwrap_method"] = payload["unwrap_method"]
        if payload.get("keep_marked_seams"):
            result["marked_input_key"] = seam_engine.fingerprint(payload["arrays"], payload["protected_edges"])
        return result

    def compute_on_main_thread(self, payload):
        from ..bridge.seam_solver import unwrap

        return unwrap(payload)

    def apply(self, ctx: AssetContext, result: dict[str, Any]) -> None:
        application = result.pop('_native_application', None)
        if application is None:
            raise ContextError('No validated native UV proposal is available; run Seams again.')
        try:
            self._apply_validated(ctx, result, application)
        finally:
            application.release_application()

    def _apply_validated(self, ctx, result, application):
        low = ctx.artifact(ROLE_LOW)
        if low is None:
            raise ContextError("the low mesh was removed while seams were being planned")
        if low.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        arrays = mesh_io.mesh_to_arrays(low.data)
        if seam_engine.fingerprint(arrays) != result["source_key"]:
            raise ContextError("The mesh changed while seams were being planned; run Seams again.")
        if (result.get("marked_input_key") is not None
                and result["marked_input_key"] != seam_engine.fingerprint(arrays, marked_edges(low.data))):
            raise ContextError("Marked seams changed during calculation; run Seams again.")
        authorized = application.authorize(cleanseams_native.geometry_fingerprint(arrays))
        expected_uvs = np.empty((arrays.num_corners, 2), dtype=np.float64)
        expected_uvs[application.host_loop_indices] = authorized['uv']
        expected_edges = authorized['edge_vertices'][np.asarray(authorized['seams'], dtype=bool)]
        if (not np.array_equal(result['uvs'], expected_uvs)
                or not np.array_equal(result['edges'], expected_edges)
                or result['junk'] != authorized['exempt_faces'].tolist()):
            raise ContextError('The validated UV proposal changed; run Seams again.')
        singleton_count = int(np.count_nonzero(np.bincount(authorized['chart_of_face']) == 1))
        with seam_transaction(low, JUNK_ATTRIBUTE, LAYOUT_KEY):
            apply_cut(low, result["edges"], result["junk"])
            uv_io.ensure_uv_layer(low.data)
            uv_io.write_uvs(low.data, result["uvs"])
            stored = np.ascontiguousarray(uv_io.read_uvs(low.data), dtype=np.float32)
            cleanseams_native.load().check_stored_uv(
                authorized['uv'], application.host_loop_indices, stored)
            if not np.array_equal(stored, expected_uvs.astype(np.float32)):
                raise ContextError('Stored UVs differ from the validated proposal.')
            if not cuts_preserved(cut_corners(arrays, result.get('protected_edges', [])), stored):
                raise ContextError('Stored UVs closed a protected seam.')
            low[LAYOUT_KEY] = layout_fingerprint(low, arrays)

            junk = np.zeros(arrays.num_faces, dtype=bool)
            junk[result["junk"]] = True
            stretch = stretch_report(arrays, result["uvs"], faces=~junk)
            pairs = result["edges"]
            length = float(
                np.linalg.norm(
                    arrays.positions[pairs[:, 0]] - arrays.positions[pairs[:, 1]], axis=1
                ).sum()
            )
            islands = island_count_from_uvs(arrays, result["uvs"])
            offer = {
                "method": seam_engine.ENGINE,
                "edges": pairs.reshape(-1).tolist(),
                "junk": result["junk"],
                "charts": result["charts"],
                "islands": islands,
                "stretch_p95": stretch.p95,
                "holds": True,
            }
            verify_current_runtime()
            ctx.set_report(
                self.name,
                {
                    "method": seam_engine.ENGINE,
                    "engine_version": result["engine_report"].get("runtime_version", seam_engine.VERSION),
                    "solver": result["solver"],
                    "unwrap_calls": result["unwrap_calls"],
                    "unwrap_method": result["unwrap_method"],
                    "charts": result["charts"],
                    "islands": islands,
                    "singleton_faces_remaining": singleton_count,
                    "notes": ([f'{singleton_count} single-face UV island(s) remain; safe joining was not possible.']
                              if singleton_count else []),
                    "seam_count": len(pairs),
                    "seam_length": length,
                    "junk_fraction": float(junk.mean()) if len(junk) else 0.0,
                    "stretch_met": stretch.p95 <= result["max_stretch"],
                    "injective": overlap_report(arrays, result["uvs"]).is_clean,
                    "engine_report": result["engine_report"],
                    "final_native_report": result["final_native_report"],
                    "initial_final_validation": result.get("initial_final_validation"),
                    "island_preference": ctx.recipe.uv.island_preference,
                    "protected_edges": result.get("protected_edges", []),
                    "choice": 0,
                    "offers": [offer],
                },
            )


def apply_cut(low: bpy.types.Object, seam_edges, junk_faces) -> None:
    """Put one candidate cut on the mesh: its seams, and its junk flag."""
    mesh = low.data
    uv_io.clear_seams(mesh)
    edges = np.asarray(seam_edges, dtype=np.int64).reshape(-1, 2)
    if edges.size:
        uv_io.mark_seams(mesh, edges)
    flag = np.zeros(len(mesh.polygons), dtype=np.int64)
    if junk_faces:
        flag[np.asarray(junk_faces, dtype=np.int64)] = 1
    attributes.write_int(mesh, JUNK_ATTRIBUTE, flag, domain="FACE")


def junk_faces_of(mesh: bpy.types.Mesh) -> np.ndarray | None:
    """The junk flag the seams stage left, as a boolean mask, or None."""
    if JUNK_ATTRIBUTE not in mesh.attributes.keys():
        return None
    values = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.attributes[JUNK_ATTRIBUTE].data.foreach_get("value", values)
    return values.astype(bool)
