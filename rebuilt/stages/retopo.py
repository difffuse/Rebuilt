"""Building the low-poly mesh."""

from __future__ import annotations

import re
from dataclasses import replace
import time
from typing import Any, ClassVar

import bpy
import numpy as np

from ..bridge import attributes, bvh, mesh_io, uv_io
from ..core.analysis.classify import ARCHITECTURE, GENERIC_ORGANIC, HARD_SURFACE
from ..core.analysis.fields import density_field
from ..core.analysis.mesh_graph import edge_topology
from ..core.jobs import Cancelled, ProgressReporter
from ..core.qa.shape import budget_error, deviation_report
from ..core.qa.topology import inspect_topology
from ..core.quad_budget import (
    STRICT_PASSES,
    DYNAMIC_PASSES,
    DYNAMIC_MULTIPLIER,
    DYNAMIC_REFINEMENT_SECONDS,
    budget_status,
    prefer_candidate,
)
from ..core.recipe import AssetRecipe, RetopoRecipe
from ..pipeline.context import AssetContext, ContextError
from .analyze import asset_class_of
from .base import Stage
from .cleanup import current_mesh_object, _normals_outward, _orient_component_normals
from .ingest import ROLE_PROXY

ROLE_LOW = "low"

STRATEGY_QUAD = "quad_uniform"
STRATEGY_ADAPTIVE = "adaptive"
STRATEGY_HARD_SURFACE = "hard_surface"


MAX_BUDGET_PASSES = 3
STRICT_NATIVE_PASSES = 4
COUNT_SEARCH_SECONDS = 120.0
RECOVERY_SECONDS = 90.0
RECOVERY_FALLBACK_RESERVE = 30.0
BUDGET_TOLERANCE = 0.02


PRE_DECIMATE_LADDER: tuple[int, ...] = (8, 4, 2)


FIELD_SMOOTHING_PER_CELL = 50


HARD_EDGE_ANGLE = 65.0


MIN_CREASE_EDGES = 6


def _crease_anchors(shaped, creased) -> np.ndarray:
    """Field anchors along the hard-edge chains, one direction per vertex."""
    sharp = creased.dihedral > np.radians(HARD_EDGE_ANGLE)
    anchors = np.zeros((shaped.num_verts, 3), dtype=np.float64)
    if not sharp.any():
        return anchors

    positions = shaped.positions.astype(np.float64)
    edges = creased.edges[sharp]

    from ..core.analysis.mesh_graph import connected_components

    vertices = np.unique(edges)
    remap = np.full(int(vertices.max()) + 1, -1, dtype=np.int64)
    remap[vertices] = np.arange(len(vertices))
    labels = connected_components(len(vertices), remap[edges])

    edge_chain = labels[remap[edges[:, 0]]]
    chain_sizes = np.bincount(edge_chain)
    keep = chain_sizes[edge_chain] >= MIN_CREASE_EDGES
    edges = edges[keep]
    if not len(edges):
        return anchors

    neighbours: dict[int, list[int]] = {}
    for one, two in edges:
        neighbours.setdefault(int(one), []).append(int(two))
        neighbours.setdefault(int(two), []).append(int(one))

    raw: dict[int, np.ndarray] = {}
    for vertex, links in neighbours.items():
        total = np.zeros(3)
        for other in links:
            direction = positions[other] - positions[vertex]
            length = float(np.linalg.norm(direction))
            if length < 1e-12:
                continue
            direction /= length
            total += -direction if total @ direction < 0.0 else direction
        raw[vertex] = total

    for _ in range(3):
        smoothed = {}
        for vertex, links in neighbours.items():
            total = raw[vertex].copy()
            for other in links:
                contribution = raw.get(other)
                if contribution is None:
                    continue
                total += -contribution if total @ contribution < 0.0 else contribution
            smoothed[vertex] = total
        raw = smoothed

    for vertex, links in neighbours.items():
        if len(links) < 2:
            continue
        length = float(np.linalg.norm(raw[vertex]))
        if length > 1e-12:
            anchors[vertex] = raw[vertex] / length
    return anchors


MIN_FIELD_QUADS = 0.8


def choose_strategy(recipe: RetopoRecipe, asset_class: str) -> str:
    """Pick a strategy, honouring an explicit choice over the detected class."""
    if recipe.strategy != "auto":
        return recipe.strategy
    if asset_class in (HARD_SURFACE, ARCHITECTURE):
        return STRATEGY_HARD_SURFACE
    if asset_class == GENERIC_ORGANIC and recipe.adaptive_density:
        return STRATEGY_ADAPTIVE
    return STRATEGY_QUAD


class UnsafeRetopology(ContextError):
    """No usable surface was produced by this attempt."""


def _check_finished_topology(report):
    topology = report["topology"]
    if topology.get("face_count") == 0:
        raise UnsafeRetopology("The remesher returned no surface faces.")
    issues = []
    valence = topology["max_valence"]
    if valence > 32:
        issues.append(
            f"{valence} edges meet at one vertex; inspect this region for a collapsed rim."
        )
    crowded = topology.get("non_manifold_edges", 0)
    if crowded:
        issues.append(
            f"{crowded} edge(s) belong to more than two faces; inspect overlapping sheets."
        )
    for key, label in (
        ("flipped_faces", "inconsistently oriented"),
        ("degenerate_faces", "degenerate"),
    ):
        count = topology.get(key, 0)
        if count:
            issues.append(f"{count} {label} face(s) remain; inspect these regions after baking.")
    report["finishing_warnings"] = issues


def _repair_face_winding(obj):
    import bmesh

    mesh = bmesh.new()
    try:
        mesh.from_mesh(obj.data)
        bmesh.ops.recalc_face_normals(mesh, faces=list(mesh.faces))
        mesh.to_mesh(obj.data)
        obj.data.update()
    finally:
        mesh.free()


class RetopoStage(Stage):
    name: ClassVar[str] = "retopo"
    slot: ClassVar[str] = "low"
    role: ClassVar[str] = ROLE_LOW
    label: ClassVar[str] = "Building low-poly mesh"
    native_engines = ('autoremesher',)

    def runtime_engines(self, recipe: AssetRecipe) -> tuple[str, ...]:
        return self.native_engines if recipe.retopo.remesher == 'auto' and recipe.retopo.topology == 'quads' else ()

    def cache_key(self, recipe: AssetRecipe, *, runtime=None) -> str:
        key = recipe.retopo.fingerprint()
        if recipe.retopo.flow_guides_key:
            from ..core.flow_guides import REVISION
            # Guided output uses the complete ordinary native reference path.
            key += f":flow-guides-{REVISION}:finish-1"
        if recipe.retopo.remesher == "auto":
            key += ":local-singularity-1"
            key += ":managed-finish-2"
            key += ":crease-extent-1:surface-recovery-3:native-surface-preserved-1"
        key += ":practical-acceptance-1:planar-holes-1:source-orientation-1:supported-seams-2"
        key += ":input-tessellation-1:input-orientation-1"
        key += ":rim-repair-4"
        key += ":projection-area-1:quadriflow-process-1"
        if recipe.retopo.topology == "quads" and recipe.retopo.quad_budget_mode == "dynamic":
            key += ":quad-budget-4"
        if recipe.retopo.topology == "quads" and recipe.retopo.quad_budget_mode == "strict":
            key += ":strict-fallback-1"
        return self._with_runtime(key, recipe, runtime)

    def gather(self, ctx: AssetContext) -> dict[str, Any]:
        source = current_mesh_object(ctx)
        if source is None:
            raise ContextError("retopo needs ingest to have run first")

        analysed = ctx.artifact(ROLE_PROXY) or source
        importance = None
        if attributes.IMPORTANCE in analysed.data.attributes.keys():
            importance = attributes.read_float(analysed.data, attributes.IMPORTANCE)

        cleaned_mesh = _duplicate(source, "__rebuilt_sanitize")
        engine_input_triangles = None
        guided_payload = None
        try:
            _sanitize_scan(cleaned_mesh)

            polygon_sides = np.empty(len(cleaned_mesh.data.polygons), dtype=np.int32)
            cleaned_mesh.data.polygons.foreach_get("loop_total", polygon_sides)
            scan_triangles = int(np.maximum(polygon_sides - 2, 0).sum())
            tope = min(
                max(
                    ENGINE_INPUT_TRIANGLES_PER_QUAD * int(ctx.recipe.retopo.target_faces),
                    ENGINE_INPUT_MIN_TRIANGLES,
                ),
                ENGINE_INPUT_MAX_TRIANGLES,
            )
            if scan_triangles > tope:
                bpy.context.scene.collection.objects.link(cleaned_mesh)
                _decimate_to(cleaned_mesh, tope)
                engine_input_triangles = int(_triangle_count(cleaned_mesh))
            sanitized_mesh = mesh_io.mesh_to_arrays(cleaned_mesh.data)
            engine_crease = _crease_within_budget(sanitized_mesh, ctx.recipe.retopo.target_faces)
            if ctx.recipe.retopo.flow_guides_key:
                from ..ui.flow_guides import gather
                guided_payload = gather(ctx, source, prepared_mesh=cleaned_mesh.data)
        finally:
            engine_mesh = cleaned_mesh.data
            bpy.data.objects.remove(cleaned_mesh, do_unlink=True)
            if engine_mesh.users == 0:
                bpy.data.meshes.remove(engine_mesh)

        return {
            "arrays": mesh_io.mesh_to_arrays(source.data),
            "sanitized": sanitized_mesh,
            "engine_input_triangles": engine_input_triangles,
            "engine_crease": engine_crease,
            "importance": importance,
            "importance_matches": analysed is source,
            "recipe": ctx.recipe.retopo,
            "asset_class": asset_class_of(ctx),
            "source_name": source.name,
            "flow_guides": guided_payload,
        }

    def compute(self, payload: dict[str, Any], report: ProgressReporter) -> dict[str, Any]:
        arrays = payload["arrays"]
        recipe: RetopoRecipe = payload["recipe"]

        report.update(0.2, "Planning density")
        strategy = choose_strategy(recipe, payload["asset_class"])

        importance = payload["importance"]
        if importance is None or not payload["importance_matches"]:
            importance = np.zeros(arrays.num_verts, dtype=np.float64)

        density, _ = density_field(arrays, importance, recipe.target_faces, recipe.density_contrast)
        peak = float(density.max()) if density.size else 0.0
        weights = density / peak if peak > 0 else np.ones_like(density)

        report.update(0.6, "Measuring source")

        topology = edge_topology(arrays)
        source_is_closed = bool((topology.face_counts == 1).sum() == 0)
        diagonal = float(
            np.linalg.norm(arrays.positions.max(axis=0) - arrays.positions.min(axis=0))
        )
        source_triangles = int(np.maximum(arrays.face_sizes - 2, 0).sum())

        result = {
            "engine_quads": None,
            "engine_tried": bool(recipe.remesher == "auto" and recipe.topology == "quads"),
            "strategy": strategy,
            "weights": weights,
            "diagonal": diagonal,
            "source_triangles": source_triangles,
            "source_arrays": arrays,
            "source_is_closed": source_is_closed,
            "engine_input_triangles": payload.get("engine_input_triangles"),
            "engine_crease": payload.get("engine_crease"),
            "recipe": recipe,
        }

        if recipe.flow_guides_key:
            from ..bridge.guided_steering import steer, ProgressSlice
            if recipe.topology != "quads" or recipe.remesher != "auto":
                raise ContextError("Flow Guides requires native quad retopology.")
            if self.main_thread_calls is None:
                raise ContextError("Guided retopology must run through the Rebuilt job system.")
            # Run the exact ordinary Auto path, including its count search and
            # finishing. The guide field may move vertices, never rebuild faces.
            baseline = self.compute(dict(payload, recipe=replace(recipe, flow_guides_key="")),
                                    ProgressSlice(report, 0., .55))["finished"]
            if not baseline["report"]["quad_method"].startswith("auto("):
                raise ContextError("AutoRemesher could not provide a reference for these guides. Previous mesh kept.")
            guided, diagnostics = steer(baseline["arrays"], arrays, payload["flow_guides"],
                                        recipe.target_faces, recipe.flow_guides_strength,
                                        ProgressSlice(report, .55, .4))
            diagnostics.update(engine_revision=baseline["report"]["autoremesher_revision"], full_pipeline=True)
            result.update(steered_arrays=guided, reference_report=baseline["report"],
                          flow_guides=diagnostics)
            result["finished"] = self.main_thread_calls.call(result, report)
            report.update(1.0, "Guided retopology ready")
            return result

        if recipe.topology == "quads" and self.main_thread_calls is not None:
            return self._budget_search(payload, result, report)

        if recipe.remesher == "auto" and recipe.topology == "quads":
            result["engine_quads"] = autoremesher_quads(
                payload["sanitized"],
                max(recipe.target_faces, 8),
                payload["engine_crease"] if strategy == STRATEGY_HARD_SURFACE else 0.0,
                bool(recipe.adaptive_density and recipe.density_contrast > 0),
                recipe,
                lambda f, m="": report.update(0.7 + 0.25 * f, m),
                cancelled=lambda: report.cancelled,
            )
        report.update(1.0, "Ready")
        return result

    def _budget_search(self, payload, result, progress):
        """Correct the count of the finished mesh, never the raw engine output."""
        recipe = result["recipe"]
        mode, target = recipe.quad_budget_mode, max(recipe.target_faces, 8)
        attempts = STRICT_PASSES if mode == "strict" else DYNAMIC_PASSES
        scaling = 1.0 if mode == "dynamic" else AUTOREMESHER_FIRST_SCALING
        engine_target = target
        samples, history, rejections = [], [], []
        rejected_attempts = []
        best = None
        search_started = time.perf_counter()
        stop_reason = "pass_limit"
        count_fallback = False
        automatic_recovery = recipe.remesher == "auto"
        native = automatic_recovery
        recovery_started = None
        native_retry = False
        fallback_attempted = False
        recovery_count_passes = 0
        for attempt in range(attempts + 2):
            progress.check_cancelled()
            remaining = None
            if recovery_started is not None:
                remaining = RECOVERY_SECONDS - (time.perf_counter() - recovery_started)
                if remaining <= 0:
                    stop_reason = "recovery_time_budget"
                    break
            candidate = dict(
                result,
                recipe=replace(
                    recipe,
                    target_faces=engine_target,
                    remesher=recipe.remesher if native else "quadriflow",
                    adaptive_density=False if native_retry else recipe.adaptive_density,
                ),
            )
            candidate["engine_crease"] = 0.0 if native_retry else payload.get("engine_crease", 0.0)
            effective_crease = (
                candidate["engine_crease"] if result["strategy"] == STRATEGY_HARD_SURFACE else 0.0
            )
            effective_adaptive = bool(
                candidate["recipe"].adaptive_density
                and recipe.density_contrast > 0
                and (recipe.curvature_adaptivity > 0 or recipe.curvature_anisotropy > 0)
            )
            if recovery_started is not None and not native:
                candidate["recovery_timeout"] = min(
                    remaining, COUNT_SEARCH_SECONDS - (time.perf_counter() - search_started)
                ) if recovery_count_passes else remaining
                if candidate["recovery_timeout"] <= 0:
                    stop_reason = "count_time_budget"
                    break
                fallback_attempted = True
            begin, span = 0.1 + 0.85 * min(attempt, attempts - 1) / attempts, 0.85 / attempts

            def say(fraction, message=""):
                progress.update(
                    begin + span * fraction,
                    (
                        f"Surface recovery: {message}"
                        if recovery_started is not None
                        else f"{mode.title()} quads · pass {attempt + 1}/{attempts}: {message}"
                    ),
                )

            say(0, "Remeshing")
            started = time.perf_counter()
            try:
                if native:
                    options = {}
                    if native_retry:
                        reserve = 0 if recovery_count_passes else RECOVERY_FALLBACK_RESERVE
                        allowance = remaining - reserve
                        if recovery_count_passes:
                            allowance = min(
                                allowance,
                                COUNT_SEARCH_SECONDS - (time.perf_counter() - search_started),
                            )
                            if allowance <= 0:
                                stop_reason = "count_time_budget"
                                break
                        options["timeout"] = max(0.001, allowance)
                    candidate["engine_quads"] = autoremesher_quads(
                        payload["sanitized"],
                        engine_target,
                        effective_crease,
                        effective_adaptive,
                        candidate["recipe"],
                        say,
                        cancelled=lambda: progress.cancelled,
                        single_scaling=scaling,
                        **options,
                    )
                    if candidate["engine_quads"] is None:
                        raise UnsafeRetopology("AutoRemesher did not produce a usable surface.")
                say(0.9, "Checking the finished mesh")
                finished = (self.main_thread_calls.call(candidate, progress) if native
                            else self._finish_quadriflow(candidate, progress))
            except UnsafeRetopology as exc:
                rejections.append(str(exc))
                rejected_attempts.append(
                    {
                        "pass": attempt + 1,
                        "engine": "auto" if native else "quadriflow",
                        "seconds": time.perf_counter() - started,
                        "reason": str(exc),
                    }
                )
                if best is not None and (
                    not automatic_recovery
                    or not best["report"].get("surface_quality", {}).get("needs_recovery")
                ):
                    stop_reason = "usable_result_retained"
                    break
                if native:
                    if recovery_started is None:
                        recovery_started = time.perf_counter()
                    native = False
                    engine_target = target
                    progress.update(begin + span, "Unusable candidate; bounded QuadriFlow recovery")
                    continue
                if best is not None:
                    stop_reason = "recovery_failed_result_retained"
                    break
                raise
            measured = finished["report"]
            quads = measured["topology"]["quad_count"]
            measured.update(budget_status(mode, target, quads))
            history.append(
                {
                    "engine": measured["quad_method"],
                    "surface_quality": measured.get("surface_quality", {}),
                    "recovery": recovery_started is not None,
                    "count_correction": recovery_count_passes > 0,
                    "engine_target": engine_target,
                    "scaling": scaling if native else None,
                    "quads": quads,
                    "faces": measured["achieved_faces"],
                    "shape_error": measured["shape_error"],
                    "seconds": time.perf_counter() - started,
                }
            )
            previous_best = best
            if best is None or prefer_candidate(
                measured, best["report"], mode, target, recover_surface=automatic_recovery
            ):
                best = finished
            quality = measured.get("surface_quality", {})
            if automatic_recovery and (
                quality.get("needs_recovery") or recovery_started is not None
            ):
                retained_over_limit = (
                    mode == "dynamic"
                    and best["report"]["achieved_quads"] > target * DYNAMIC_MULTIPLIER
                )
                if (not best["report"].get("surface_quality", {}).get("needs_recovery")
                        and not (retained_over_limit and quality.get("needs_recovery"))):
                    if recovery_started is not None and native_retry and native:
                        now = time.perf_counter()
                        remaining = RECOVERY_SECONDS - (now - recovery_started)
                        if (
                            not quality.get("needs_recovery")
                            and not best["report"]["budget_met"]
                            and recovery_count_passes < 2
                            and history[-1]["seconds"] < remaining
                            and now - search_started + history[-1]["seconds"] < COUNT_SEARCH_SECONDS
                        ):
                            uniform_samples = [
                                (item["scaling"], item["quads"])
                                for item in history
                                if item["recovery"] and item["scaling"] is not None
                            ]
                            desired = target * 1.02 if mode == "dynamic" else target
                            corrected = _next_scaling(uniform_samples, desired, aim_fraction=1.0)
                            if abs(corrected - scaling) >= 1e-5:
                                scaling = corrected
                                recovery_count_passes += 1
                                progress.update(begin + span, "Surface recovered; correcting quad count")
                                continue
                    if recovery_started is not None and not native:
                        now = time.perf_counter()
                        if (
                            not quality.get("needs_recovery") and quads > 0
                            and not best["report"]["budget_met"]
                            and recovery_count_passes < 2
                            and history[-1]["seconds"] < RECOVERY_SECONDS - (now - recovery_started)
                            and now - search_started + history[-1]["seconds"] < COUNT_SEARCH_SECONDS
                        ):
                            desired = target * 1.02 if mode == "dynamic" else target
                            engine_target = max(8, round(engine_target * desired / quads))
                            recovery_count_passes += 1
                            progress.update(begin + span, "Surface recovered; correcting quad count")
                            continue
                    stop_reason = (
                        "surface_recovered"
                        if recovery_started is not None
                        else "usable_result_retained"
                    )
                    break
                if recovery_started is None:
                    recovery_started = time.perf_counter()
                remaining = RECOVERY_SECONDS - (time.perf_counter() - recovery_started)
                if remaining <= 0:
                    stop_reason = "recovery_time_budget"
                    break
                if not native:
                    stop_reason = "recovery_limit_result_retained"
                    break
                can_change_native = effective_crease > 0 or effective_adaptive
                if (
                    not native_retry
                    and can_change_native
                    and history[-1]["seconds"] < remaining - RECOVERY_FALLBACK_RESERVE
                ):
                    native_retry = True
                    if quads > 0:
                        scaling = _next_scaling([(scaling, quads)], target, aim_fraction=1.0)
                    progress.update(
                        begin + span, "Rim spikes detected; one uniform AutoRemesher retry"
                    )
                else:
                    native = False
                    progress.update(begin + span, "Rim spikes remain; bounded QuadriFlow recovery")
                engine_target = target
                continue

            if quads <= 0:
                stop_reason = "empty_candidate"
                break
            if not best["report"]["budget_met"] and (
                time.perf_counter() - search_started + history[-1]["seconds"]
                >= COUNT_SEARCH_SECONDS
            ):
                stop_reason = "count_time_budget"
                break
            if mode == "strict":
                if best["report"]["budget_met"]:
                    stop_reason = "count_target"
                    break
                if native and attempt + 1 >= STRICT_NATIVE_PASSES:
                    stop_reason = "count_pass_limit"
                    break
                if native and measured["quad_method"].startswith("auto("):
                    samples.append((scaling, quads))
                    corrected = _next_scaling(samples, target, aim_fraction=1.0)
                    if abs(corrected - scaling) < 1e-5:
                        stop_reason = "count_stalled"
                        break
                    scaling = corrected
                else:
                    native = False
                    engine_target = max(8, round(engine_target * target / quads))
            else:
                if (
                    best["report"]["budget_met"]
                    and best["report"]["shape_error"] <= recipe.max_deviation
                ):
                    stop_reason = "shape_target"
                    break
                if best["report"]["budget_met"]:
                    if (
                        time.perf_counter() - search_started + history[-1]["seconds"]
                        >= DYNAMIC_REFINEMENT_SECONDS
                    ):
                        stop_reason = "refinement_time_budget"
                        break
                    if previous_best is best and previous_best["report"]["budget_met"]:
                        stop_reason = "no_quality_gain"
                        break
                if quads > target * DYNAMIC_MULTIPLIER:
                    desired = target * DYNAMIC_MULTIPLIER * 0.9
                else:
                    desired = min(max(target, quads * 1.6), target * DYNAMIC_MULTIPLIER * 0.95)
                if quads >= target * DYNAMIC_MULTIPLIER * 0.95 and measured["budget_met"]:
                    stop_reason = "quad_limit"
                    break
                if native and measured["quad_method"].startswith("auto("):
                    samples.append((scaling, quads))
                    if quads < target:
                        desired = max(desired, target * 1.1)
                    scaling = _next_scaling(samples, desired, aim_fraction=1.0)
                else:
                    native = False
                    engine_target = max(8, round(engine_target * desired / quads))
            if attempt + 1 >= attempts:
                break

        if best is None:
            raise UnsafeRetopology("No safe retopology candidate was produced.")
        measured = best["report"]
        measured["rejected_candidates"] = rejections
        measured["budget_rejected_attempts"] = rejected_attempts
        measured["budget_passes"] = len(history) + len(rejected_attempts)
        measured["target_faces"] = recipe.target_faces
        measured["budget_used"] = measured["achieved_quads"] / target
        measured["budget_error"] = measured["quad_budget_error"]
        measured["over_budget"] = (
            not measured["budget_met"]
            and measured["achieved_quads"] > measured["quad_budget_limit"]
        )
        measured["budget_attempts"] = history
        measured["budget_search_seconds"] = time.perf_counter() - search_started
        measured["budget_stop_reason"] = stop_reason
        measured["budget_count_fallback"] = count_fallback
        measured["surface_recovery"] = {
            "attempted": recovery_started is not None,
            "native_retry": native_retry,
            "fallback_attempted": fallback_attempted,
            "selected_engine": measured["quad_method"],
            "allowance_seconds": RECOVERY_SECONDS,
            "elapsed_seconds": time.perf_counter() - recovery_started
            if recovery_started is not None
            else 0.0,
            "stop_reason": stop_reason,
        }
        warnings = measured.setdefault("finishing_warnings", [])
        if measured.get("surface_quality", {}).get("needs_recovery"):
            warnings.append(
                (
                    "Repeated unsupported rim spikes remain after bounded recovery. "
                    "The least defective available mesh is retained; inspect its open borders."
                    if recovery_started is not None
                    else "Repeated unsupported rim spikes remain; inspect the open borders."
                )
            )
        elif fallback_attempted and measured["quad_method"] == "quadriflow":
            measured["notes"].append(
                "Automatic surface recovery selected QuadriFlow after AutoRemesher "
                "did not provide an acceptable surface."
            )
        measured["notes"] = [
            n
            for n in measured["notes"]
            if not n.startswith("The mesh strays") and not n.startswith("The mesh has lost")
        ]
        if count_fallback:
            measured["notes"].append(
                "AutoRemesher count did not converge after four passes; "
                "QuadriFlow was evaluated with the same count and geometry checks."
            )
        if not measured["budget_met"]:
            measured["notes"].append(
                f"{mode.title()} quad count not reached: {measured['achieved_quads']:,} "
                f"quads for {target:,} requested after {len(history)} passes. "
                "The closest result is retained; inspect it before continuing."
            )
        if stop_reason == "count_time_budget":
            measured["notes"].append(
                "Additional count correction skipped because another pass would exceed the "
                "120-second search allowance. The usable result is retained with its actual count."
            )
        if mode == "dynamic":
            measured["notes"].append(
                f"Dynamic selected {measured['achieved_quads']:,} quads from a {target:,} "
                f"starting budget ({len(history) + len(rejected_attempts)} passes, maximum {target * DYNAMIC_MULTIPLIER:,})."
            )
            if stop_reason == "refinement_time_budget":
                measured["notes"].append(
                    f"Additional shape refinement skipped after the {DYNAMIC_REFINEMENT_SECONDS:g}-second allowance; "
                    "the best mesh meeting the quad budget is retained. Shape tolerance is unchanged."
                )
        if measured["shape_error"] > recipe.max_deviation:
            measured["notes"].append(
                "Some surface detail remains outside the shape tolerance. "
                "Try a higher quad count for a closer match."
            )
        progress.update(1.0, "Quad count checked")
        return dict(result, finished=best)

    def _finish_quadriflow(self, result, progress):
        """Keep native waiting on the worker; request only Blender array operations."""
        from ..bridge.quadriflow_process import remesh

        prepared = self.main_thread_calls.call(
            dict(result, quadriflow_operation="base"), progress)
        base = prepared["arrays"]
        recipe = result["recipe"]
        target = max(recipe.target_faces, 8)
        if result["strategy"] == STRATEGY_HARD_SURFACE:
            recipe = replace(recipe, preserve_sharp=True, preserve_boundary=True)
        recovery = result.get("recovery_timeout")
        deadline = None if recovery is None else time.perf_counter() + recovery
        original_triangles = int(np.maximum(base.face_sizes - 2, 0).sum())
        output = None
        method = "quadriflow"
        for rung in range(1 if recovery is not None else len(PRE_DECIMATE_LADDER)):
            progress.check_cancelled()
            request = dict(result, quadriflow_operation="attempt", quadriflow_base=prepared,
                           quadriflow_triangle_budget=_quadriflow_triangle_budget(
                               target, original_triangles, rung))
            arrays = self.main_thread_calls.call(request, progress)
            remaining = 900.0 if deadline is None else deadline - time.perf_counter()
            if remaining <= 0:
                break
            output = remesh(
                arrays, recipe, target,
                passes=MAX_BUDGET_PASSES if recovery is None else 1,
                tolerance=BUDGET_TOLERANCE, timeout=remaining,
                binary_path=prepared["binary_path"], cancelled=lambda: progress.cancelled)
            progress.check_cancelled()
            if output is not None:
                break
        if output is None:
            if recovery is not None:
                raise UnsafeRetopology("The bounded QuadriFlow recovery did not produce a mesh.")
            output = self.main_thread_calls.call(
                dict(result, quadriflow_operation="conversion", quadriflow_base=prepared), progress)
            method = "conversion"
        return self.main_thread_calls.call(
            dict(result, quadriflow_arrays=output, quadriflow_method=method), progress)

    def compute_on_main_thread(self, result):
        """Finish and measure a candidate on disposable, isolated objects."""
        source = low = None
        selected = list(bpy.context.selected_objects)
        active = bpy.context.view_layer.objects.active
        try:
            source = mesh_io.arrays_to_object(result["source_arrays"], "__rebuilt_budget_source")
            bpy.context.scene.collection.objects.link(source)
            low = _duplicate(source, "__rebuilt_budget_candidate")
            bpy.context.scene.collection.objects.link(low)
            operation = result.get("quadriflow_operation")
            if operation == "base":
                _sanitize_scan(low)
                if (result.get("recovery_timeout") is None
                        and result["strategy"] == STRATEGY_ADAPTIVE):
                    _thin_adaptive(low, result["recipe"], result["weights"],
                                   result["source_triangles"])
                # Polygon arrays alone lose the prepared edge ordering, which
                # affects subsequent Blender decimation. Carry its exact topology.
                edges = np.empty((len(low.data.edges), 2), dtype=np.int32)
                loop_edges = np.empty(len(low.data.loops), dtype=np.int32)
                low.data.edges.foreach_get("vertices", edges.ravel())
                low.data.loops.foreach_get("edge_index", loop_edges)
                return {"arrays": mesh_io.mesh_to_arrays(low.data),
                        "edges": edges, "loop_edges": loop_edges,
                        "binary_path": bpy.app.binary_path}
            if operation in ("attempt", "conversion"):
                _restore_quadriflow_prepared(low, result["quadriflow_base"])
                if operation == "attempt":
                    _decimate_to(low, result["quadriflow_triangle_budget"])
                else:
                    _quads_by_conversion(low, max(result["recipe"].target_faces, 8))
                return mesh_io.mesh_to_arrays(low.data)
            if "steered_arrays" in result:
                # Already checked against the finished reference. Reprojection,
                # welding or hole repair here would invalidate that comparison.
                mesh_io.replace_object_mesh(low, result["steered_arrays"])
                measured = dict(result["reference_report"])
                measured.update(self._measure(low, result, bvh.SurfaceQuery(result["source_arrays"]),
                                              "flow_guides"))
                for key in ("budget_error", "over_budget", "budget_used"):
                    measured[key] = result["reference_report"][key]
                measured["flow_guides"] = result["flow_guides"]
                measured["notes"] = list(result["reference_report"].get("notes", [])) + [
                    "Guides orient the AutoRemesher mesh while preserving its faces and boundaries. "
                    "Protected areas may follow guides less closely."]
                if result["flow_guides"].get("already_aligned"):
                    measured["notes"].append("AutoRemesher already follows these guides; no vertex motion was needed.")
                _check_finished_topology(measured)
            else:
                measured = self._finish_candidate(low, source, result, lambda *args: None)
            return {"arrays": mesh_io.mesh_to_arrays(low.data), "report": measured}
        finally:
            for obj in (low, source):
                if obj is not None:
                    mesh = obj.data
                    bpy.data.objects.remove(obj, do_unlink=True)
                    if mesh.users == 0:
                        bpy.data.meshes.remove(mesh)
            for obj in bpy.context.selected_objects:
                obj.select_set(False)
            for obj in selected:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = active

    def apply(self, ctx: AssetContext, result: dict[str, Any]) -> None:
        if result.get("recipe", ctx.recipe.retopo).flow_guides_key:
            from ..ui.flow_guides import fingerprint
            if fingerprint(ctx.source_object) != result["recipe"].flow_guides_key:
                raise ContextError("Guides changed during generation. Previous mesh kept; run Retopologise again.")
        ctx.clear_artifacts(self.role)

        source = current_mesh_object(ctx)
        low = _duplicate(source, f"{ctx.name}_low")
        ctx.register_artifact(low, self.role, self.slot)
        attributes.clear_rebuilt_attributes(low.data)
        if "finished" in result:
            mesh_io.replace_object_mesh(low, result["finished"]["arrays"])
            _clear_modifiers(low)
            report = result["finished"]["report"]
        else:
            report = self._finish_candidate(low, source, result, ctx.progress)

        # New topology has no artist-authored component selection to retain.
        for items in (low.data.vertices, low.data.edges, low.data.polygons):
            items.foreach_set("select", np.zeros(len(items), dtype=bool))
        low.data.polygons.active = -1
        low.data.update()
        low.data.materials.clear()
        ctx.set_report(self.name, report)
        from ..ui.flow_guides import hide_guides
        hide_guides(ctx.source_object)

    def _finish_candidate(self, low, source, result, progress):
        recipe = result["recipe"]
        strategy = result["strategy"]
        query = bvh.SurfaceQuery(result["source_arrays"])
        welded = 0
        closed = 0
        hygiene: dict = {}

        _sanitize_scan(low)

        try:
            if "quadriflow_arrays" in result:
                mesh_io.replace_object_mesh(low, result["quadriflow_arrays"])
                method = result["quadriflow_method"]
            elif "guided_arrays" in result:
                mesh_io.replace_object_mesh(low, result["guided_arrays"])
                method = "flow_guides"
            elif result.get("recovery_timeout") is not None:
                recovery_deadline = time.perf_counter() + result["recovery_timeout"]
                _decimate_to(low, max(recipe.target_faces, 8) * PRE_DECIMATE_LADDER[0])
                remaining = recovery_deadline - time.perf_counter()
                fallback_recipe = (
                    replace(recipe, preserve_sharp=True, preserve_boundary=True)
                    if strategy == STRATEGY_HARD_SURFACE
                    else recipe
                )
                if remaining <= 0 or not _quadriflow_to_budget(
                    low, fallback_recipe, max(recipe.target_faces, 8), passes=1, timeout=remaining
                ):
                    raise UnsafeRetopology(
                        "The bounded QuadriFlow recovery did not produce a mesh."
                    )
                method = "quadriflow"
            elif recipe.remesher == "auto" and recipe.topology == "quads":
                declined = result.get("engine_quads") is None and result.get("engine_tried", False)
                if declined:
                    progress(0.1, "AutoRemesher unavailable; trying QuadriFlow (Blender)")
                    method = _quad_remesh(low, recipe, progress)
                else:
                    method = _instant_remesh(
                        low,
                        recipe,
                        progress,
                        crease=30.0 if strategy == STRATEGY_HARD_SURFACE else 0.0,
                        engine="auto",
                        prepared=result.get("engine_quads"),
                    )
            elif recipe.topology == "triangles":
                method = _triangulate_to_budget(low, recipe, result["weights"], progress)
            elif strategy == STRATEGY_QUAD:
                method = _quad_remesh(low, recipe, progress)
            elif strategy == STRATEGY_ADAPTIVE:
                method = _adaptive_decimate(
                    low, recipe, result["weights"], result["source_triangles"], progress
                )
            elif strategy == STRATEGY_HARD_SURFACE:
                method = _hard_surface(low, recipe, progress)
            else:
                raise ContextError(f"unknown retopology strategy {strategy!r}")

            native_auto = method.startswith("auto(")
            native_result = native_auto or method == "flow_guides"
            if native_result:
                hygiene = {"native_finishing": "preserved" if native_auto else "minimal"}
            else:
                progress(0.8, "Welding duplicate vertices")
                welded = _weld_coincident(low, result["diagonal"])
                hygiene = _output_hygiene(low, result["diagonal"])

            if not native_result and result["source_is_closed"]:
                progress(0.82, "Closing holes the remesher left")
                closed = _close_remesh_holes(low)
            elif not native_result:
                progress(0.82, "Restoring source openings capped by the remesher")
                hygiene["lids"] = _remove_lids(
                    low, query, result["diagonal"], _boundary_vertices(result["source_arrays"])
                )

            if recipe.project_to_surface and not native_auto:
                progress(0.85, "Projecting onto the source model")
                hygiene["surface_projection"] = _project_onto(low, source)
                if not result["source_is_closed"]:
                    hygiene["boundary_projection"] = _project_open_boundaries(
                        low, result["source_arrays"], recipe.target_faces
                    )
            if native_result and not native_auto:
                from ..bridge.planar_holes import close_supported_holes

                hygiene["source_orientation"] = _orient_native_surface(low, source)
                if method == "flow_guides":
                    from ..bridge.guided_patches import repair_supported_patches
                    hygiene["guided_patches"] = repair_supported_patches(
                        low, result["source_arrays"], query, recipe.target_faces, _closed_loops
                    )
                from ..bridge.supported_seams import repair_supported_seams
                hygiene["supported_seams"] = repair_supported_seams(
                    low, result["source_arrays"], query, recipe.target_faces, _closed_loops
                )
                planar_caps = close_supported_holes(
                    low, result["source_arrays"], query, recipe.target_faces, _closed_loops
                )
                closed = planar_caps + hygiene["supported_seams"]["caps"]
                hygiene["supported_planar_holes"] = planar_caps
            if recipe.topology == "quads" and not native_result:
                hygiene["separated_pinches"] = _split_pinched_vertices(low)
        finally:
            _clear_modifiers(low)

        progress(0.95, "Measuring the result")
        from ..core.analysis.mesh_graph import face_areas

        arrays = mesh_io.mesh_to_arrays(low.data)
        if not np.isfinite(arrays.positions).all() or not np.any(face_areas(arrays) > 0):
            raise UnsafeRetopology("The remesher returned no finite surface area.")
        report = self._measure(low, result, query, method)
        if report["topology"]["flipped_faces"] and not native_auto:
            inconsistent = report["topology"]["flipped_faces"]
            if native_result:
                _normals_outward(low, source, force=True)
            else:
                _repair_face_winding(low)
            report = self._measure(low, result, query, method)
            hygiene["inconsistent_faces_resolved"] = (
                inconsistent - report["topology"]["flipped_faces"]
            )
        if recipe.topology == "quads":
            _check_finished_topology(report)
            boundary = report["topology"]["boundary_edges"]
            if result["source_is_closed"] and boundary:
                report["finishing_warnings"].append(
                    f"{boundary} boundary edge(s) remain on a closed source; inspect the surface openings."
                )
        report["welded_vertices"] = welded
        report["closed_holes"] = closed
        report["hygiene"] = hygiene
        report["engine_input_triangles"] = result.get("engine_input_triangles")
        report["engine_crease"] = result.get("engine_crease")
        if "flow_guides" in result:
            quads = int(np.count_nonzero(mesh_io.mesh_to_arrays(low.data).face_sizes == 4))
            report.update(budget_status(recipe.quad_budget_mode, recipe.target_faces, quads))
            report["over_budget"] = quads > report["quad_budget_limit"] and not report["budget_met"]
            report["flow_guides"] = result["flow_guides"]
            report["notes"].append("Flow guides steer direction; exact edge loops are not guaranteed.")
            if not report["budget_met"]:
                report["notes"].append(
                    f"Guided count: {quads:,} quads for {recipe.target_faces:,} requested. "
                    "The guided result is kept; review it before continuing."
                )
        return report

    def _measure(
        self,
        low: bpy.types.Object,
        result: dict[str, Any],
        query: bvh.SurfaceQuery,
        method: str = "quadriflow",
    ) -> dict[str, Any]:
        """Measure final topology and surface accuracy in both directions."""
        low_arrays = mesh_io.mesh_to_arrays(low.data)
        recipe: RetopoRecipe = result["recipe"]

        distances = query.nearest_distances(bvh.sample_surface(low_arrays))
        deviation = deviation_report(distances, result["diagonal"])

        source_samples = bvh.sample_surface(result["source_arrays"])
        if len(source_samples) > 8192:
            source_samples = source_samples[
                np.linspace(0, len(source_samples) - 1, 8192, dtype=int)
            ]
        reverse = bvh.SurfaceQuery(low_arrays).nearest_distances(source_samples)
        coverage = deviation_report(reverse, result["diagonal"])
        shape_error = max(deviation.p95, coverage.p95)
        if reverse.size and result["diagonal"] > 0:
            shape_error = max(
                shape_error, float(np.percentile(reverse, 99)) / result["diagonal"] * 0.5
            )
        topology = inspect_topology(low_arrays)
        from ..core.qa.rims import rim_quality

        surface_quality = rim_quality(low_arrays, result["source_arrays"], recipe.target_faces)
        error = budget_error(low_arrays.num_faces, recipe.target_faces)

        notes: list[str] = []

        if method == "conversion":
            notes.append(
                "The quad remesher refused this model, so the quads were paired "
                "up from triangles instead. The edge flow will be poor."
            )
        notes.extend(budget_advice(deviation.p95, recipe.max_deviation, recipe.target_faces))

        return {
            "strategy": result["strategy"],
            "quad_method": method,
            "autoremesher_revision": (result.get("flow_guides", {}).get("engine_revision")
                                      if method == "flow_guides" else
                                      _autoremesher_revision() if method.startswith("auto(") else None),
            "cloth_folds": recipe.cloth_folds if method.startswith("auto(") or method == "flow_guides" else None,
            "notes": notes,
            "target_faces": recipe.target_faces,
            "achieved_faces": low_arrays.num_faces,
            "budget_error": error,
            "over_budget": error > BUDGET_TOLERANCE,
            "budget_used": (
                low_arrays.num_faces / recipe.target_faces if recipe.target_faces else 0.0
            ),
            "deviation": deviation.to_dict(),
            "within_tolerance": shape_error <= recipe.max_deviation,
            "coverage": coverage.to_dict(),
            "shape_error": shape_error,
            "fold_fraction": _mesh_health(low)[1],
            "surface_quality": surface_quality,
            "topology": topology.to_dict(),
            "errors": topology.errors(),
            "warnings": topology.warnings(),
        }


def budget_advice(deviation_p95: float, tolerance: float, target_faces: int) -> list[str]:
    """Explain measured shape loss relative to the requested quad budget."""
    if tolerance <= 0 or deviation_p95 <= tolerance:
        return []
    ratio = deviation_p95 / tolerance
    if ratio > 2.0:
        return [
            f"The mesh has lost the source shape: it strays {ratio:.1f}x further than "
            f"the tolerance allows. {target_faces:,} quads is too few for this model — "
            "raise the quad count for a faithful result."
        ]
    return [
        f"The mesh strays from the source more than the tolerance allows "
        f"({deviation_p95:.4f} against {tolerance:.4f}). For a closer match, raise "
        f"the quad count above {target_faces:,}."
    ]


def _duplicate(source: bpy.types.Object, name: str) -> bpy.types.Object:
    copy = source.copy()
    copy.data = source.data.copy()
    copy.name = name
    copy.data.name = name
    return copy


def _clear_modifiers(obj: bpy.types.Object) -> None:
    obj.modifiers.clear()


def _make_active(obj: bpy.types.Object) -> None:
    """Operators read the active object and the selection, so set both."""
    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _replace_mesh(obj: bpy.types.Object, mesh: bpy.types.Mesh) -> None:
    """Swap an object's mesh, freeing the one it was using."""
    old = obj.data
    mesh.name = old.name
    obj.data = mesh
    if old.users == 0:
        bpy.data.meshes.remove(old)


def _apply_modifier(obj: bpy.types.Object, modifier: bpy.types.Modifier) -> None:
    """Bake a modifier into the mesh without going through an operator."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    baked = bpy.data.meshes.new_from_object(evaluated)

    obj.modifiers.remove(modifier)
    _replace_mesh(obj, baked)


def _triangle_count(obj: bpy.types.Object) -> int:
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


def _decimate_to(obj: bpy.types.Object, triangle_budget: int) -> None:
    """Collapse down to a triangle budget, or do nothing if already under it."""
    current = _triangle_count(obj)
    if current <= triangle_budget or triangle_budget <= 0:
        return

    modifier = obj.modifiers.new("rebuilt_pre", "DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = max(triangle_budget / current, 1e-6)
    modifier.use_collapse_triangulate = True
    _apply_modifier(obj, modifier)


def _quadriflow_to_budget(
    obj: bpy.types.Object,
    recipe: RetopoRecipe,
    target: int,
    *,
    passes=MAX_BUDGET_PASSES,
    timeout=900.0,
) -> bool:
    """Run native count correction in a disposable Blender process."""
    from ..bridge.quadriflow_process import remesh

    arrays = remesh(
        mesh_io.mesh_to_arrays(obj.data),
        recipe,
        target,
        passes=passes,
        tolerance=BUDGET_TOLERANCE,
        timeout=timeout,
    )
    if arrays is None:
        return False
    mesh_io.replace_object_mesh(obj, arrays)
    return True


def _quads_by_conversion(obj: bpy.types.Object, target: int) -> None:
    """Last resort: decimate to budget and pair the triangles up."""
    _decimate_to(obj, target * 2)
    with uv_io.edit_only(obj):
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.tris_convert_to_quads(
            face_threshold=np.radians(180.0),
            shape_threshold=np.radians(180.0),
        )


def _sanitize_scan(obj: bpy.types.Object) -> dict:
    """Clean the working copy before remeshing without modifying the source."""
    import bmesh

    mesh = obj.data
    diagonal = max(float(max(obj.dimensions)), 1e-6)
    epsilon = 1e-5 * diagonal

    bm = bmesh.new()
    bm.from_mesh(mesh)
    verts_before = len(bm.verts)
    faces_before = len(bm.faces)

    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=epsilon)
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=epsilon)
    _orient_component_normals(bm, obj)

    report = {
        "welded": verts_before - len(bm.verts),
        "degenerate": faces_before - len(bm.faces),
    }
    bm.to_mesh(mesh)
    mesh.update()
    bm.free()
    if report["welded"] or report["degenerate"]:
        print(
            "Rebuilt: input preparation — "
            f"{report['welded']} vertices welded, "
            f"{report['degenerate']} degenerate faces dissolved, "
            "normals oriented with open sides preserved"
        )
    return report


def _instant_binary():
    """The bundled Instant Meshes executable for this platform, or None."""
    import os
    import platform
    import sys
    from pathlib import Path

    base = Path(__file__).resolve().parent.parent / "bin"
    if sys.platform == "darwin":
        arch = "arm64" if platform.machine() == "arm64" else "x64"
        candidate = base / f"macos-{arch}" / "InstantMeshesHeadless"
    elif sys.platform.startswith("win"):
        candidate = base / "windows-x64" / "InstantMeshesHeadless.exe"
    else:
        candidate = base / "linux-x64" / "InstantMeshesHeadless"
    if not candidate.exists():
        return None
    try:
        os.chmod(candidate, 0o755)
    except OSError:
        pass
    return candidate


def _autoremesher_binary():
    """Resolve the installed locked engine; integrity failures are actionable."""
    from ..runtime.manifest import resolve_engine

    return resolve_engine("autoremesher").entrypoints["cli"]


AUTOREMESHER_PROGRESS = re.compile(r"^PROGRESS\s+(\d+)\s*(.*)$")


def _autoremesher_revision():
    """Report the verified installed engine; cache identity uses the same lock."""
    from ..runtime.manifest import resolve_engine

    return resolve_engine("autoremesher").metadata["commit"]


AUTOREMESHER_TIMEOUT = 900.0


ENGINE_INPUT_TRIANGLES_PER_QUAD = 40
ENGINE_INPUT_MIN_TRIANGLES = 150_000


ENGINE_INPUT_MAX_TRIANGLES = 300_000


HARD_SURFACE_ENGINE_ANGLE = 60.0


ENGINE_SHARP_EDGE_BUDGET = 2500
ENGINE_CREASE_LADDER: tuple[float, ...] = (HARD_SURFACE_ENGINE_ANGLE, 75.0, 90.0)


OPEN_EDGE_TOLERANCE = 8


def _engine_input(obj: bpy.types.Object, target: int):
    """Prepare the same decimated engine input used by gather."""
    tope = min(
        max(ENGINE_INPUT_TRIANGLES_PER_QUAD * int(target), ENGINE_INPUT_MIN_TRIANGLES),
        ENGINE_INPUT_MAX_TRIANGLES,
    )
    if _triangle_count(obj) <= tope:
        return mesh_io.mesh_to_arrays(obj.data)
    arrays = _decimated_arrays(obj, tope)
    return arrays if arrays is not None else mesh_io.mesh_to_arrays(obj.data)


def _crease_has_extent(positions, edges, minimum: float) -> bool:
    from ..core.analysis.mesh_graph import connected_components

    if not len(edges):
        return False
    vertices, inverse = np.unique(edges, return_inverse=True)
    labels = connected_components(len(vertices), inverse.reshape(-1, 2))
    for label in np.unique(labels):
        span = np.ptp(positions[vertices[labels == label]].astype(np.float64), axis=0)
        if np.linalg.norm(span) >= minimum * (1.0 - 1e-6):
            return True
    return False


def _crease_within_budget(scan, target: int = 2500) -> float:
    """Use resolvable crease chains, falling back to the engine's 90-degree default."""
    from ..core.analysis.mesh_graph import face_areas, manifold_edges

    edges = manifold_edges(scan)
    if edges.count == 0:
        return 0.0
    minimum = 2.0 * float(np.sqrt(face_areas(scan).sum() / max(target, 8)))
    for angle in ENGINE_CREASE_LADDER:
        selected = edges.dihedral > np.radians(angle)
        sharp = int(selected.sum())
        if sharp <= ENGINE_SHARP_EDGE_BUDGET:
            if angle < 90.0 and not _crease_has_extent(
                scan.positions, edges.edges[selected], minimum
            ):
                continue
            if angle != HARD_SURFACE_ENGINE_ANGLE:
                print(
                    f"Rebuilt: using {angle:g}° engine creases ({sharp} edges); "
                    "lower angles exceed the edge budget or contain only short chains"
                )
            return float(angle)
    print("Rebuilt: too many creases for the engine at any angle; remeshing as organic")
    return 0.0


AUTOREMESHER_COUNT_PASSES = 4


AUTOREMESHER_COUNT_UNDER = 0.20
AUTOREMESHER_COUNT_OVER = 0.05


AUTOREMESHER_AIM = 0.95


AUTOREMESHER_FIRST_SCALING = 1.6


def _write_scan_obj(path, scan) -> None:
    """Write native engine input without accessing Blender datablocks."""
    from ..bridge.triangulation import remesher_triangles

    triangles = remesher_triangles(scan)
    with open(path, "w") as handle:
        for point in scan.positions:
            handle.write(f"v {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n")
        for a, b, c in triangles:
            handle.write(f"f {a + 1} {b + 1} {c + 1}\n")


def _read_obj_arrays(path):
    """Read engine OBJ positions and faces, or return (None, None)."""
    import os

    if not os.path.exists(path):
        return None, None
    points: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                faces.append([int(token.split("/")[0]) - 1 for token in line.split()[1:]])
    return points, faces


def autoremesher_quads(
    scan,
    target,
    crease,
    adaptive,
    recipe,
    progress=None,
    cancelled=None,
    *,
    single_scaling=None,
    timeout=None,
):
    """Run AutoRemesher on the worker thread and return its mesh arrays."""
    from ..runtime.process import ProcessStartError, run_process
    import tempfile
    from pathlib import Path

    say = progress or (lambda fraction, message="": None)
    binary = _autoremesher_binary()
    if binary is None:
        return None

    with tempfile.TemporaryDirectory(prefix="rebuilt_ar_") as workdir:
        source_path = Path(workdir) / "scan.obj"
        result_path = Path(workdir) / "remesh.obj"
        _write_scan_obj(source_path, scan)

        def run_auto(scaling, span):
            from ..runtime.session import verify_current_runtime
            verify_current_runtime()
            command = [
                str(binary),
                "--input",
                str(source_path),
                "--output",
                str(result_path),
                "--quads",
                str(max(target, 8)),
                "--model-type",
                "hard" if crease > 0 else "organic",
                "--adaptivity",
                f"{recipe.curvature_adaptivity:.4g}" if adaptive else "0.0",
                "--anisotropy",
                f"{recipe.curvature_anisotropy:.4g}" if adaptive else "0.0",
                "--cloth-folds" if recipe.cloth_folds else "--no-cloth-folds",
            ]
            if crease > 0:
                command += ["--sharp-edge-degrees", f"{max(crease, 30.0):g}"]

            if scaling is not None:
                command += ["--scaling", f"{scaling:.6g}"]

            if result_path.exists():
                result_path.unlink()
            begin, end = span
            log: list[str] = []
            pending = {'stdout': b'', 'stderr': b''}
            discarding = {'stdout': False, 'stderr': False}
            line_limit = 4096

            def consume_line(data):
                line = data.decode('utf-8', errors='replace').rstrip('\r')
                match = AUTOREMESHER_PROGRESS.match(line)
                if match is not None:
                    percent = min(int(match.group(1)), 100)
                    step = match.group(2).strip() or 'remeshing'
                    say(begin + (end - begin) * percent / 100.0, step)
                elif line:
                    log.append(line)
                    del log[:-20]

            def output(stream, data):
                # Progress may cross byte chunks; retain bounded whole lines.
                # Overlong diagnostics are discarded through their next LF so
                # their suffix cannot be mistaken for a new progress message.
                if discarding[stream]:
                    _, separator, data = data.partition(b'\n')
                    if not separator:
                        return
                    discarding[stream] = False
                data = pending[stream] + data
                while b'\n' in data:
                    line, data = data.split(b'\n', 1)
                    if len(line) <= line_limit:
                        consume_line(line)
                    else:
                        consume_line(b'[engine diagnostic line exceeded 4096 bytes]')
                if len(data) > line_limit:
                    consume_line(b'[engine diagnostic line exceeded 4096 bytes]')
                    pending[stream] = b''
                    discarding[stream] = True
                else:
                    pending[stream] = data

            try:
                process = run_process(
                    command,
                    timeout=AUTOREMESHER_TIMEOUT if timeout is None else min(timeout, AUTOREMESHER_TIMEOUT),
                    cancelled=cancelled,
                    on_output=output,
                    max_stdout=64*1024,
                    max_stderr=64*1024,
                )
            except ProcessStartError as exc:
                print(f"Rebuilt: AutoRemesher did not start — {exc}")
                return None, None
            if process.cancelled:
                raise Cancelled()
            if process.timed_out:
                print('Rebuilt: AutoRemesher exceeded its time limit')
                return None, None
            for data in pending.values():
                if data:
                    consume_line(data)
            if process.returncode != 0 or not result_path.exists():
                tail = [line for line in log if line.strip()]
                print(
                    "Rebuilt: AutoRemesher declined the model — "
                    + (tail[-1] if tail else f"exit {process.returncode}")
                )
                return None, None
            return _read_obj_arrays(result_path)

        if single_scaling is not None:
            points, faces = run_auto(single_scaling, (0.0, 1.0))
            return (points, faces) if faces else None

        points = faces = None
        best_error = None
        scaling = AUTOREMESHER_FIRST_SCALING
        samples: list[tuple[float, int]] = []
        for attempt in range(AUTOREMESHER_COUNT_PASSES):
            begin = 0.25 + 0.7 * attempt / AUTOREMESHER_COUNT_PASSES
            end = 0.25 + 0.7 * (attempt + 1) / AUTOREMESHER_COUNT_PASSES
            say(begin, f"Remeshing with AutoRemesher (pass {attempt + 1})")
            arranque = time.perf_counter()
            trial_points, trial_faces = run_auto(scaling, (begin, end))
            if trial_faces is None:
                break
            delivered = len(trial_faces)
            signed = delivered / max(target, 1) - 1.0
            print(
                f"Rebuilt: AutoRemesher pass {attempt + 1} — {delivered} faces for "
                f"{target} asked (scaling {scaling or 1.0:.3g}), "
                f"{time.perf_counter() - arranque:.1f} s on {scan.num_faces} input faces"
            )

            in_band = -AUTOREMESHER_COUNT_UNDER <= signed <= AUTOREMESHER_COUNT_OVER
            rank = (
                0 if in_band else 1 if -0.5 < signed < 0 else 2 if signed <= -0.5 else 3,
                abs(signed),
            )
            if best_error is None or rank < best_error:
                points, faces, best_error = trial_points, trial_faces, rank
            if in_band or delivered == 0:
                break
            samples.append((float(scaling or 1.0), delivered))
            corrected = _next_scaling(samples, target)

            if scaling is not None and abs(corrected - scaling) < 1e-3:
                break
            scaling = corrected

    if not faces:
        return None
    return points, faces


def _next_scaling(
    samples: list[tuple[float, int]], target: int, aim_fraction: float = AUTOREMESHER_AIM
) -> float:
    """The edge scale to try next, from what the engine delivered so far."""
    aim = max(target, 1) * aim_fraction
    scaling, delivered = samples[-1]
    over = [(s, c) for s, c in samples if c > aim]
    under = [(s, c) for s, c in samples if 0 < c <= aim]
    if over and under:
        s_over, c_over = min(over, key=lambda sc: sc[1])
        s_under, c_under = max(under, key=lambda sc: sc[1])
        if c_over != c_under and s_over != s_under:
            share = (np.log(aim) - np.log(c_over)) / (np.log(c_under) - np.log(c_over))
            guess = float(np.exp(np.log(s_over) + share * (np.log(s_under) - np.log(s_over))))

            if any(abs(guess - s) < 1e-3 for s, _ in samples):
                guess = float(np.sqrt(s_over * s_under))
            return float(np.clip(guess, 0.25, 4.0))
    exponent = 2.0
    if len(samples) >= 2:
        (s1, c1), (s2, c2) = samples[-2], samples[-1]
        if c1 > 0 and c2 > 0 and abs(np.log(s2 / s1)) > 1e-6:
            exponent = float(np.clip(np.log(c1 / c2) / np.log(s2 / s1), 1.0, 3.0))
    return float(np.clip(scaling * (delivered / aim) ** (1.0 / exponent), 0.25, 4.0))


def _deduplicate_native_faces(faces):
    """Keep one copy of each identical polygon cycle, including reversed winding."""
    unique = []
    seen = set()
    for face in faces:
        cycle = tuple(int(v) for v in face)
        reverse = cycle[::-1]
        key = min(row[i:] + row[:i] for row in (cycle, reverse) for i in range(len(row)))
        if key not in seen:
            seen.add(key)
            unique.append(face)
    return unique, len(faces) - len(unique)


def _instant_remesh(
    obj: bpy.types.Object,
    recipe: RetopoRecipe,
    progress=None,
    crease: float = 0.0,
    engine: str = "instant",
    prepared=None,
) -> str:
    """Quadrangulate with one of the two bundled subprocess engines."""
    import subprocess
    import tempfile
    from pathlib import Path

    say = progress or (lambda fraction, message="": None)
    native = engine == "auto"
    label = "auto" if native else "instant"
    engine_name = "AutoRemesher" if native else "Instant Meshes"
    binary = _autoremesher_binary() if native else _instant_binary()
    if binary is None:
        print(
            f"Rebuilt: {engine_name} is not bundled for this platform — "
            "falling back to QuadriFlow (Blender)"
        )
        return _quad_remesh(obj, recipe, progress)

    target = max(recipe.target_faces, 8)
    say(0.1, "Measuring the source model")
    scan = mesh_io.mesh_to_arrays(obj.data)
    from ..core.analysis.mesh_graph import face_areas

    total_area = float(face_areas(scan).sum())
    cell_estimate = float(np.sqrt(total_area / target)) if total_area > 0 else 1.0

    scan_boundary = _boundary_vertices(scan, cell_estimate)
    close_under = 0.0 if recipe.preserve_boundary else SMALL_OPENING_CELLS

    adaptive = bool(recipe.adaptive_density and recipe.density_contrast > 0)

    native_adaptive = native and adaptive
    sizing = _instant_sizing(obj, target, recipe) if adaptive and not native else None
    base_target = target
    if sizing is not None and total_area > 0:
        _, sizing_spacing = sizing

        flat_cell = float(np.quantile(sizing_spacing, 0.75))
        if flat_cell > 0:
            base_target = int(np.clip(total_area / (flat_cell * flat_cell), target // 3, target))

    def surrender(reason="Invalid remesher output") -> str:

        if native and prepared is not None:
            raise UnsafeRetopology(f"AutoRemesher candidate failed finishing checks: {reason}")

        if sizing is not None or native_adaptive:
            import dataclasses

            say(0.5, "Retrying with uniform density")
            return _instant_remesh(
                obj,
                dataclasses.replace(recipe, adaptive_density=False),
                progress,
                crease=crease,
                engine=engine,
            )
        say(0.5, "Trying QuadriFlow (Blender)")
        return _quad_remesh(obj, recipe, progress)

    say(0.25, f"Handing the model to {engine_name}")
    with tempfile.TemporaryDirectory(prefix="rebuilt_im_") as workdir:
        source_path = Path(workdir) / "scan.obj"
        result_path = Path(workdir) / "remesh.obj"
        _write_scan_obj(source_path, scan)

        def read_result():
            """Read the engine output, or return (None, None) when absent."""
            if not result_path.exists():
                return None, None
            parsed_points: list[list[float]] = []
            parsed_faces: list[list[int]] = []
            with open(result_path) as handle:
                for line in handle:
                    if line.startswith("v "):
                        parts = line.split()
                        parsed_points.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif line.startswith("f "):
                        parsed_faces.append(
                            [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
                        )
            return parsed_points, parsed_faces

        def run_engine(face_argument: int):

            command = [
                str(binary),
                str(source_path),
                "-o",
                str(result_path),
                "-f",
                str(max(face_argument, 8)),
                "-d",
            ]

            if crease > 0:
                command += ["-c", f"{crease:g}"]

            if recipe.preserve_boundary:
                command.append("-b")
            try:
                finished = subprocess.run(command, capture_output=True, text=True, timeout=600)
            except (subprocess.TimeoutExpired, OSError) as exc:
                print(f"Rebuilt: Instant Meshes did not finish — {exc}")
                return None, None
            if finished.returncode != 0 or not result_path.exists():
                tail = (finished.stderr or finished.stdout or "").strip().splitlines()
                print(
                    "Rebuilt: Instant Meshes declined the model — "
                    + (tail[-1] if tail else f"exit {finished.returncode}")
                )
                return None, None
            return read_result()

        if native:
            if prepared is not None:
                points, faces = prepared
            else:
                engine_result = autoremesher_quads(
                    _engine_input(obj, target),
                    target,
                    _crease_within_budget(scan, target) if crease > 0 else 0.0,
                    native_adaptive,
                    recipe,
                    say,
                )
                points, faces = engine_result if engine_result is not None else (None, None)
        else:
            points, faces = run_engine(base_target // 4)
            if faces is not None and len(faces) < 0.25 * base_target:
                say(0.5, "Short result; retrying at the full target")
                more_points, more_faces = run_engine(base_target)
                if more_faces is not None and len(more_faces) > len(faces):
                    points, faces = more_points, more_faces
        if faces is None:
            return surrender()

        say(0.6, "Reading the quads back")

    if not faces:
        print(f"Rebuilt: {engine_name} returned an empty mesh")
        return surrender()

    from ..core.types import MeshArrays

    if native:
        faces, duplicate_faces = _deduplicate_native_faces(faces)
        if duplicate_faces:
            print(f"Rebuilt: omitted {duplicate_faces} exact overlapping native face duplicate(s)")

    sizes = np.asarray([len(face) for face in faces], dtype=np.int32)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32)
    built = MeshArrays(
        positions=np.asarray(points, dtype=np.float64),
        corner_verts=np.concatenate([np.asarray(face, dtype=np.int32) for face in faces]),
        face_starts=starts,
        face_sizes=sizes,
    )
    if not np.isfinite(built.positions).all():
        return surrender("Non-finite vertex coordinates")

    if native:
        try:
            mesh_io.replace_object_mesh(obj, built)
        except ValueError as error:
            raise UnsafeRetopology(
                "AutoRemesher returned invalid mesh connectivity; importing it would "
                "remove or alter faces. The existing mesh was kept."
            ) from error
        if not len(obj.data.polygons):
            return surrender("No surface faces survived input validation")
        ratio = float(np.mean(sizes == 4))
        return f"auto({ratio:.0%} quads, native surface)"

    pristine = obj.data.copy()
    try:
        try:
            mesh_io.replace_object_mesh(obj, built)
        except ValueError:
            repaired = bpy.data.meshes.new(obj.data.name)
            repaired.from_pydata(
                [tuple(point) for point in built.positions],
                [],
                [
                    tuple(int(v) for v in built.corner_verts[start : start + size])
                    for start, size in zip(built.face_starts, built.face_sizes)
                ],
            )
            repaired.validate(verbose=False)
            _replace_mesh(obj, repaired)
        say(0.8, f"Patching the holes {engine_name} left")
        patched, leftover = _close_small_holes(obj, scan_boundary, cell_estimate, close_under)
        if leftover > OPEN_EDGE_TOLERANCE:
            raise ValueError(f"{leftover} open edges survived the patching")
        if leftover > 0:
            print(f"Rebuilt: {leftover} open edge(s) left after the patching; kept")
        _purge_slivers(obj, 5.0 * cell_estimate)
        if sizing is not None:
            say(0.9, "Refining where the curvature asks for smaller quads")
            health_before = _mesh_health(obj)
            refined = _adaptive_refine(
                obj,
                sizing[0],
                sizing[1],
                query=bvh.SurfaceQuery(scan),
                max_faces=int(1.3 * target),
            )
            if refined:
                _requad_cleanup(obj)
                healed, leftover = _close_small_holes(
                    obj, scan_boundary, cell_estimate, close_under
                )
                patched += healed
                if leftover > 0:
                    raise ValueError(f"{leftover} open edges survived the adaptive refinement")

                health_after = _mesh_health(obj)
                if (
                    health_after[0] < health_before[0] - 0.005
                    or health_after[1] > 1.5 * health_before[1] + 0.001
                ):
                    raise ValueError(
                        "adaptive refinement degraded the mesh: "
                        f"quads {health_before[0]:.1%}→{health_after[0]:.1%}, "
                        f"folds {health_before[1]:.2%}→{health_after[1]:.2%}"
                    )

        _purge_slivers(obj, 5.0 * cell_estimate)
        if recipe.preserve_boundary:
            removed_caps = _open_floor(obj, scan)
            if removed_caps:
                print(f"Rebuilt: floor opened — {removed_caps} cap faces removed")
        else:
            _flatten_floor(obj, scan)
        final_faces = len(obj.data.polygons)
        final_quads = sum(1 for p in obj.data.polygons if len(p.vertices) == 4)
        if final_faces < 0.25 * target:
            raise ValueError(f"only {final_faces} faces came back of {target} asked")
        if final_quads < MIN_FIELD_QUADS * max(final_faces, 1):
            raise ValueError(f"only {final_quads / max(final_faces, 1):.0%} quads after patching")

        if sizing is not None and final_faces < 0.7 * target:
            raise ValueError(f"adaptive refinement landed short: {final_faces} of {target}")
    except ValueError as exc:
        print(f"Rebuilt: {engine_name} result rejected — {exc}")
        _replace_mesh(obj, pristine.copy())
        if pristine.users == 0:
            bpy.data.meshes.remove(pristine)
        return surrender(str(exc))
    if pristine.users == 0:
        bpy.data.meshes.remove(pristine)

    _repair_inverted_pockets(obj, cell_estimate)

    ratio = final_quads / max(final_faces, 1)
    return f"{label}({ratio:.0%} quads, {patched} patches)"


def _repair_inverted_pockets(obj, cell: float) -> int:
    """Pinch only small interior defects; never walk into an existing rim."""
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    original_boundary = {e for e in bm.edges if len(e.link_faces) != 2}
    boundary_vertices = {v for e in original_boundary for v in e.verts}
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    inverted = []
    for face in bm.faces:
        if any(v in boundary_vertices for v in face.verts):
            continue
        neighbourhood = {
            other for edge in face.edges for other in edge.link_faces if other is not face
        }
        ring2 = set(neighbourhood)
        for nb in neighbourhood:
            for edge in nb.edges:
                ring2.update(f for f in edge.link_faces if f is not face)
        if len(ring2) < 4:
            continue
        mean = sum((f.normal for f in ring2), face.normal * 0.0)
        if mean.length < 1e-9:
            continue
        if face.normal.dot(mean.normalized()) < -0.2:
            inverted.append(face)
    if inverted and len(inverted) <= max(24, int(0.005 * len(bm.faces))):
        del_verts = {v for surface_face in inverted for v in surface_face.verts}
        bmesh.ops.delete(bm, geom=inverted, context="FACES")
        rims = [
            e
            for e in bm.edges
            if e.is_valid and len(e.link_faces) == 1 and set(e.verts) & del_verts
        ]
        pieces: list[set] = []
        seen: set = set()
        for edge in rims:
            if edge in seen:
                continue
            piece = {edge}
            frontier = [edge]
            while frontier:
                current = frontier.pop()
                for vert in current.verts:
                    for other in vert.link_edges:
                        if (
                            other.is_valid
                            and len(other.link_faces) == 1
                            and other not in original_boundary
                            and other not in piece
                        ):
                            piece.add(other)
                            frontier.append(other)
            seen |= piece
            pieces.append(piece)
        for piece in pieces:
            vertices = {v for e in piece for v in e.verts}
            if (
                len(piece) > 16
                or vertices & boundary_vertices
                or any(sum(v in e.verts for e in piece) != 2 for v in vertices)
                or _piece_extent(list(piece)) > 4 * max(cell, 1e-9)
            ):
                bm.free()
                return 0
        for piece in pieces:
            verts = list({v for e in piece for v in e.verts})
            if not verts:
                continue
            centre = sum((v.co for v in verts), verts[0].co * 0.0) / len(verts)
            for vert in verts:
                vert.co = centre
        bmesh.ops.remove_doubles(
            bm,
            verts=list({v for piece in pieces for e in piece for v in e.verts}),
            dist=1e-6,
        )
        orphans = [v for v in bm.verts if v.is_valid and not v.link_faces]
        if orphans:
            bmesh.ops.delete(bm, geom=orphans, context="VERTS")
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()
    return len(inverted)


def _purge_slivers(obj: bpy.types.Object, limit: float) -> int:
    """Subdivide oversized edges without deleting their surface."""
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    touched = 0
    for _ in range(5):
        long_edges = [e for e in bm.edges if e.is_valid and e.calc_length() > limit]
        if not long_edges:
            break
        touched += len(long_edges)
        bmesh.ops.subdivide_edges(bm, edges=long_edges, cuts=1)
    if touched:
        triangles = [f for f in bm.faces if f.is_valid and len(f.verts) == 3]
        if triangles:
            bmesh.ops.join_triangles(
                bm,
                faces=triangles,
                angle_face_threshold=np.pi,
                angle_shape_threshold=np.pi,
            )
        bm.to_mesh(obj.data)
        obj.data.update()
    bm.free()
    return touched


def _flatten_floor(obj: bpy.types.Object, scan) -> int:
    """Project a planar floor cap onto the source rim plane; return moved vertices."""
    floor_plane = _floor_plane(scan)
    if floor_plane is None:
        return 0
    centroid, normal, diagonal = floor_plane

    mesh = obj.data
    count = len(mesh.vertices)
    coords = np.empty(count * 3)
    mesh.vertices.foreach_get("co", coords)
    coords = coords.reshape(-1, 3)
    depth = (coords - centroid) @ normal
    outside = depth < -1e-9
    if not outside.any():
        return 0
    coords[outside] -= np.outer(depth[outside], normal)
    mesh.vertices.foreach_set("co", coords.reshape(-1))
    mesh.update()
    return int(outside.sum())


def _floor_plane(scan):
    """Find a planar floor opening, or return None."""
    topology = edge_topology(scan)
    boundary = topology.edges[topology.boundary]
    if boundary.size == 0:
        return None
    rim = np.unique(boundary)
    points = scan.positions.astype(np.float64)[rim]
    if len(points) < 8:
        return None
    centroid = points.mean(axis=0)
    diagonal = float(np.linalg.norm(scan.positions.max(axis=0) - scan.positions.min(axis=0)))
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[2]
    offsets = (points - centroid) @ normal
    if float(offsets.std()) > 0.01 * diagonal:
        return None
    inward = scan.positions.astype(np.float64).mean(axis=0) - centroid
    if float(normal @ inward) < 0:
        normal = -normal
    if not _is_cut_plane(scan, centroid, normal, diagonal):
        return None
    return centroid, normal, diagonal


def _is_cut_plane(scan, centroid, normal, diagonal) -> bool:
    """Whether the scan lies (almost) entirely on one side of the plane."""
    depth = (scan.positions.astype(np.float64) - centroid) @ normal
    beyond = float((depth < -FLOOR_CUT_TOLERANCE * diagonal).mean())
    return beyond <= FLOOR_CUT_BEYOND_SHARE


FLOOR_CUT_TOLERANCE = 0.005
FLOOR_CUT_BEYOND_SHARE = 0.02


LID_DISTANCE_CELLS = 1.5


LID_MIN_CELLS = 12.0


LID_RIM_SHARE = 0.5
LID_RIM_CELLS = 2.0


def _boundary_vertices(scan, cell: float = 0.0) -> np.ndarray:
    """Opening rim positions, excluding small dangling crack/fin components."""
    topology = edge_topology(scan)
    edges = topology.edges[topology.boundary]
    if not len(edges):
        return np.zeros((0, 3), dtype=np.float64)
    positions = scan.positions.astype(np.float64)
    if cell <= 0:
        cell = 4 * float(
            np.median(np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1))
        )
    neighbours = {}
    for a, b in edges:
        neighbours.setdefault(int(a), set()).add(int(b))
        neighbours.setdefault(int(b), set()).add(int(a))
    pending = set(neighbours)
    kept = []
    while pending:
        seed = pending.pop()
        component, frontier = [seed], [seed]
        while frontier:
            for v in neighbours[frontier.pop()]:
                if v in pending:
                    pending.remove(v)
                    component.append(v)
                    frontier.append(v)
        points = positions[component]
        closed = all(len(neighbours[v]) == 2 for v in component)
        extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        if closed or extent > cell:
            kept.extend(component)
    return positions[np.asarray(kept, dtype=np.int64)]


def _split_pinched_vertices(obj) -> int:
    """Separate disconnected face fans sharing a vertex, without moving faces."""
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    crowded = {v for e in bm.edges if len(e.link_faces) > 2 for v in e.verts}
    if not crowded:
        bm.free()
        return 0
    arrays = mesh_io.mesh_to_arrays(obj.data)
    positions = list(arrays.positions)
    corners = arrays.corner_verts.copy()
    splits = 0
    for v in sorted(crowded, key=lambda v: v.index):
        pending = set(v.link_faces)
        groups = []
        while pending:
            seed = min(pending, key=lambda f: f.index)
            pending.remove(seed)
            group, frontier = [seed], [seed]
            while frontier:
                for edge in frontier.pop().edges:
                    if v not in edge.verts or len(edge.link_faces) != 2:
                        continue
                    for face in edge.link_faces:
                        if face in pending:
                            pending.remove(face)
                            group.append(face)
                            frontier.append(face)
            groups.append(group)
        for group in groups[1:]:
            index = len(positions)
            positions.append(arrays.positions[v.index])
            splits += 1
            for face in group:
                start = arrays.face_starts[face.index]
                stop = start + arrays.face_sizes[face.index]
                part = corners[start:stop]
                part[part == v.index] = index
    bm.free()
    if splits:
        from ..core.types import MeshArrays

        rebuilt = MeshArrays(np.asarray(positions), corners, arrays.face_starts, arrays.face_sizes)

        if not np.array_equal(
            rebuilt.positions[rebuilt.corner_verts], arrays.positions[arrays.corner_verts]
        ):
            raise ContextError("Pinch separation changed surface geometry")
        smooth = [p.use_smooth for p in obj.data.polygons]
        mesh_io.replace_object_mesh(obj, rebuilt)
        for p, flag in zip(obj.data.polygons, smooth):
            p.use_smooth = flag
    return splits


def _project_open_boundaries(obj, scan, target: int) -> dict:
    """Constrain genuine rims to scan boundary segments, not nearby surfaces."""
    import bmesh
    from ..core.analysis.mesh_graph import face_areas

    cell = float(np.sqrt(face_areas(scan).sum() / max(target, 1)))
    boundary = _boundary_vertices(scan, cell)
    if not len(boundary):
        return {"vertices": 0}
    topology = edge_topology(scan)
    edges = topology.edges[topology.boundary]
    kept = {tuple(p) for p in boundary}
    edges = np.asarray([e for e in edges if all(tuple(scan.positions[v]) in kept for v in e)])
    if not len(edges):
        return {"vertices": 0}
    starts = scan.positions[edges[:, 0]].astype(np.float64)
    spans = scan.positions[edges[:, 1]].astype(np.float64) - starts
    lengths2 = np.einsum("ij,ij->i", spans, spans)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    rims = [e for e in bm.edges if e.is_boundary]
    vertices = {
        v
        for piece, near in _boundary_pieces(rims, boundary, cell)
        if near > 0.6
        for e in piece
        for v in e.verts
    }
    saved = {v: v.co.copy() for v in vertices}
    moved, maximum = 0, 0.0
    for v in vertices:
        t = np.clip(
            np.einsum("ij,ij->i", np.asarray(v.co) - starts, spans) / np.maximum(lengths2, 1e-30),
            0,
            1,
        )
        points = starts + t[:, None] * spans
        distances = np.linalg.norm(points - np.asarray(v.co), axis=1)
        index = int(np.argmin(distances))
        if distances[index] <= REAL_OPENING_REACH * cell:
            v.co = points[index]
            moved += 1
            maximum = max(maximum, float(distances[index]))

    touched = {f for v in vertices for f in v.link_faces}
    bad = set()
    for f in touched:
        before = np.array([saved.get(v, v.co) for v in f.verts], dtype=np.float64)
        after = np.array([v.co for v in f.verts], dtype=np.float64)
        before -= before[0]
        after -= after[0]
        normal = np.cross(before, np.roll(before, -1, axis=0)).sum(axis=0)
        projected_normal = np.cross(after, np.roll(after, -1, axis=0)).sum(axis=0)
        if np.linalg.norm(projected_normal) < 1e-10 or np.dot(normal, projected_normal) <= 0:
            bad.update(v for v in f.verts if v in saved)

    if bad:
        bm.free()
        return {"vertices": 0, "rejected": "Boundary projection would collapse or reverse a face."}
    bm.normal_update()
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()
    return {"vertices": moved, "maximum_distance": maximum}


def _remove_lids(obj: bpy.types.Object, query, diagonal: float, scan_boundary: np.ndarray) -> dict:
    """Delete the big patches of faces that lie off the scan. Returns what went."""
    import bmesh

    from ..core.analysis.mesh_graph import connected_components, face_areas, manifold_edges

    arrays = mesh_io.mesh_to_arrays(obj.data)
    if arrays.num_faces == 0:
        return {"faces": 0, "patches": 0}
    areas = face_areas(arrays)
    cell = float(np.sqrt(areas.sum() / arrays.num_faces))
    starts = arrays.face_starts.astype(np.int64)
    sizes = arrays.face_sizes.astype(np.int64)
    positions = arrays.positions.astype(np.float64)
    centres = np.stack(
        [positions[arrays.corner_verts[s : s + n]].mean(axis=0) for s, n in zip(starts, sizes)]
    )
    off = query.nearest_distances(centres) > LID_DISTANCE_CELLS * cell
    if not off.any():
        return {"faces": 0, "patches": 0}
    edges = manifold_edges(arrays)
    both = off[edges.faces[:, 0]] & off[edges.faces[:, 1]]
    patch = connected_components(arrays.num_faces, edges.faces[both])
    patch_area = np.bincount(patch, weights=areas * off)
    big = np.flatnonzero(patch_area >= LID_MIN_CELLS * cell * cell)

    if scan_boundary.shape[0] == 0 or big.size == 0:
        return {"faces": 0, "patches": 0}
    rim_query = bvh.PointQuery(scan_boundary)
    doomed_patches = []
    for pid in big:
        members = np.flatnonzero(off & (patch == pid))
        inside = np.zeros(arrays.num_faces, dtype=bool)
        inside[members] = True

        corners_in = np.concatenate(
            [arrays.corner_verts[starts[f] : starts[f] + sizes[f]] for f in members]
        )
        outside = np.flatnonzero(~inside)
        corners_out = (
            np.concatenate([arrays.corner_verts[starts[f] : starts[f] + sizes[f]] for f in outside])
            if outside.size
            else np.zeros(0, dtype=np.int64)
        )
        rim = np.intersect1d(corners_in, corners_out)
        if rim.size == 0:
            continue
        near = rim_query.nearest_distances(positions[rim]) < LID_RIM_CELLS * cell
        if float(near.mean()) >= LID_RIM_SHARE:
            doomed_patches.append(int(pid))
    doomed_patches = np.asarray(doomed_patches, dtype=np.int64)
    doomed = off & np.isin(patch, doomed_patches)
    if not doomed.any():
        return {"faces": 0, "patches": 0}
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.faces[i] for i in np.flatnonzero(doomed)], context="FACES")
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    report = {"faces": int(doomed.sum()), "patches": int(len(doomed_patches))}
    print(f"Rebuilt: {report['faces']} lid face(s) in {report['patches']} patch(es) opened")
    return report


def _open_floor(obj: bpy.types.Object, scan) -> int:
    """Remove a planar floor cap and return the number of deleted faces."""
    floor_plane = _floor_plane(scan)
    if floor_plane is None:
        return 0
    centroid, normal, diagonal = floor_plane

    _flatten_floor(obj, scan)

    import bmesh

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    plane_tolerance = 1e-3 * diagonal
    cap_faces = [
        surface_face
        for surface_face in bm.faces
        if all(
            abs((np.asarray(v.co) - centroid) @ normal) < plane_tolerance
            for v in surface_face.verts
        )
    ]
    if not cap_faces or len(cap_faces) > 0.4 * len(bm.faces):
        bm.free()
        return 0
    bmesh.ops.delete(bm, geom=cap_faces, context="FACES")
    orphan_vertices = [v for v in bm.verts if v.is_valid and not v.link_faces]
    if orphan_vertices:
        bmesh.ops.delete(bm, geom=orphan_vertices, context="VERTS")
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()
    return len(cap_faces)


def _mesh_health(obj: bpy.types.Object) -> tuple[float, float]:
    """Measure quad fraction and the fraction of edges folded beyond 137 degrees."""
    import bmesh

    faces = len(obj.data.polygons)
    quads = sum(1 for p in obj.data.polygons if len(p.vertices) == 4)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    folds = 0
    edges = max(len(bm.edges), 1)
    for edge in bm.edges:
        if len(edge.link_faces) != 2:
            continue
        try:
            if edge.calc_face_angle() > 2.4:
                folds += 1
        except ValueError:
            continue
    bm.free()
    return quads / max(faces, 1), folds / edges


def _instant_sizing(
    obj: bpy.types.Object, target: int, recipe: RetopoRecipe
) -> tuple[np.ndarray, np.ndarray] | None:
    """Measure a curvature-driven target edge length on the source surface."""
    from ..core.analysis.curvature import principal_directions
    from ..core.analysis.fields import curvature_sizing, graded_edge_length
    from ..core.analysis.mesh_graph import (
        face_areas,
        manifold_edges,
        taubin_smoothed,
        vertex_normals,
    )
    from ..core.types import MeshArrays

    arrays = mesh_io.mesh_to_arrays(obj.data)
    if arrays.num_verts == 0 or arrays.num_faces == 0:
        return None
    total_area = float(face_areas(arrays).sum())
    if total_area <= 0:
        return None
    cell = float(np.sqrt(total_area / max(target, 8)))
    median_edge = float(np.median(manifold_edges(arrays).lengths)) or 1.0
    if cell / median_edge > FIELD_MAX_CELL_RATIO:
        smaller = _decimated_arrays(obj, int(2.0 * MEASURE_CELL_RATIO**2 * max(target, 8)))
        if smaller is not None:
            arrays = smaller
    shaped = MeshArrays(
        positions=taubin_smoothed(arrays, passes=12),
        corner_verts=arrays.corner_verts,
        face_starts=arrays.face_starts,
        face_sizes=arrays.face_sizes,
    )
    shaped_normals = vertex_normals(shaped)
    _, _, k_major, k_minor = principal_directions(shaped, shaped_normals)
    _, spacing = curvature_sizing(
        shaped,
        k_major,
        k_minor,
        target_faces=max(target, 8),
        contrast=recipe.density_contrast,
    )

    cell = float(np.sqrt(total_area / max(target, 8)))
    ratio = np.maximum(spacing.astype(np.float64), 1e-12) / cell
    gamma = 1.0 + 3.0 * float(np.clip(recipe.density_contrast, 0.0, 1.0))
    spacing = cell * np.power(ratio, gamma)
    density = float(np.mean(1.0 / np.square(spacing)))
    wanted_density = max(target, 8) / total_area
    spacing = spacing * np.sqrt(density / wanted_density)
    return arrays.positions.astype(np.float64), spacing.astype(np.float64)


def _adaptive_refine(
    obj: bpy.types.Object,
    scan_positions: np.ndarray,
    scan_spacing: np.ndarray,
    query=None,
    max_faces: int | None = None,
) -> int:
    """Refine selected quad regions to match local target edge lengths."""
    arrays = mesh_io.mesh_to_arrays(obj.data)
    if arrays.num_faces == 0:
        return 0
    positions = arrays.positions.astype(np.float64)
    faces = [
        [int(v) for v in arrays.corner_verts[start : start + size]]
        for start, size in zip(arrays.face_starts, arrays.face_sizes)
    ]

    def edge_key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    walls: set = set()
    for face in faces:
        if len(face) == 4:
            continue
        for index, vertex in enumerate(face):
            walls.add(edge_key(vertex, face[(index + 1) % len(face)]))

    voxel = float(np.median(scan_spacing)) if scan_spacing.size else 1.0
    buckets: dict = {}
    for index, point in enumerate(scan_positions):
        key = tuple(np.floor(point / voxel).astype(np.int64))
        buckets.setdefault(key, []).append(index)

    def nearest_scan_vertex(point: np.ndarray) -> int | None:
        centre = np.floor(point / voxel).astype(np.int64)
        best, best_distance = None, np.inf
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (int(centre[0]) + dx, int(centre[1]) + dy, int(centre[2]) + dz)
                    for candidate in buckets.get(key, ()):
                        distance = float(np.linalg.norm(scan_positions[candidate] - point))
                        if distance < best_distance:
                            best_distance = distance
                            best = candidate
        return best

    demand: dict[int, float] = {}
    for position, face in enumerate(faces):
        if len(face) != 4:
            continue
        if any(edge_key(face[i], face[(i + 1) % 4]) in walls for i in range(4)):
            continue
        centre = positions[face].mean(axis=0)
        anchor = nearest_scan_vertex(centre)
        if anchor is None:
            continue
        span = float(
            np.mean(
                [
                    np.linalg.norm(positions[face[i]] - positions[face[(i + 1) % 4]])
                    for i in range(4)
                ]
            )
        )
        demand[position] = span / max(float(scan_spacing[anchor]), 1e-12)
    if not demand:
        return 0

    def propagate(threshold: float) -> set:

        seeds: set = set()
        for position, ratio in demand.items():
            if ratio > threshold:
                face = faces[position]
                length_02 = float(
                    np.linalg.norm(positions[face[0]] - positions[face[1]])
                    + np.linalg.norm(positions[face[2]] - positions[face[3]])
                )
                length_13 = float(
                    np.linalg.norm(positions[face[1]] - positions[face[2]])
                    + np.linalg.norm(positions[face[3]] - positions[face[0]])
                )
                pair = (0, 2) if length_02 >= length_13 else (1, 3)
                for i in pair:
                    seeds.add(edge_key(face[i], face[(i + 1) % 4]))
        if not seeds:
            return seeds

        while True:
            added = 0
            for face in faces:
                if len(face) != 4:
                    continue
                edges = [edge_key(face[i], face[(i + 1) % 4]) for i in range(4)]
                flags = [e in seeds for e in edges]
                count = sum(flags)
                if count == 1:
                    opposite = edges[(flags.index(True) + 2) % 4]
                    if opposite not in walls:
                        seeds.add(opposite)
                        added += 1
                elif count == 3:
                    remaining = edges[flags.index(False)]
                    if remaining not in walls:
                        seeds.add(remaining)
                        added += 1
            if not added:
                break
        return seeds

    def predicted_total(seeds: set) -> int:
        total = 0
        for face in faces:
            if len(face) != 4:
                total += 1
                continue
            count = sum(1 for i in range(4) if edge_key(face[i], face[(i + 1) % 4]) in seeds)

            if count == 0:
                total += 1
            elif count == 2:
                flags = [edge_key(face[i], face[(i + 1) % 4]) in seeds for i in range(4)]
                total += 2 if (flags[0] and flags[2]) or (flags[1] and flags[3]) else 3
            else:
                total += 4
        return total

    import os

    trace = bool(os.environ.get("REBUILT_TRACE_REFINE"))
    marked: set = set()
    for threshold in (2.0, 2.4, 2.9, 3.5, 4.5, 6.0):
        candidate = propagate(threshold)
        if trace:
            seeds_count = sum(1 for r in demand.values() if r > threshold)
            print(
                f"rebuilt refine: umbral {threshold} semillas {seeds_count} "
                f"aristas {len(candidate)} prediccion "
                f"{predicted_total(candidate) if candidate else 0} "
                f"limite {max_faces}"
            )
        if not candidate:
            return 0
        if max_faces is None or predicted_total(candidate) <= max_faces:
            marked = candidate
            break
    if not marked:
        return 0

    new_positions = [tuple(p) for p in positions]
    midpoints: dict = {}

    def midpoint(edge: tuple[int, int]) -> int:
        if edge not in midpoints:
            new_positions.append(tuple((positions[edge[0]] + positions[edge[1]]) * 0.5))
            midpoints[edge] = len(new_positions) - 1
        return midpoints[edge]

    refined = 0
    out_faces: list[list[int]] = []
    for face in faces:
        if len(face) != 4:
            out_faces.append(face)
            continue
        edges = [edge_key(face[i], face[(i + 1) % 4]) for i in range(4)]
        flags = [e in marked for e in edges]
        count = sum(flags)
        if count == 0:
            out_faces.append(face)
            continue
        a, b, c, d = face
        refined += 1
        if count == 4:
            centre_point = tuple(positions[face].mean(axis=0))
            new_positions.append(centre_point)
            centre = len(new_positions) - 1
            m0, m1, m2, m3 = (midpoint(e) for e in edges)
            out_faces += [
                [a, m0, centre, m3],
                [m0, b, m1, centre],
                [centre, m1, c, m2],
                [m3, centre, m2, d],
            ]
        elif count == 2 and flags[0] and flags[2]:
            m0, m2 = midpoint(edges[0]), midpoint(edges[2])
            out_faces += [[a, m0, m2, d], [m0, b, c, m2]]
        elif count == 2 and flags[1] and flags[3]:
            m1, m3 = midpoint(edges[1]), midpoint(edges[3])
            out_faces += [[a, b, m1, m3], [m3, m1, c, d]]
        elif count == 2:
            first_marked = next(i for i in range(4) if flags[i] and flags[(i + 1) % 4])
            ring = [face[(first_marked + k) % 4] for k in range(4)]
            m_one = midpoint(edges[first_marked])
            m_two = midpoint(edges[(first_marked + 1) % 4])
            centre_point = tuple(positions[face].mean(axis=0))
            new_positions.append(centre_point)
            centre = len(new_positions) - 1
            out_faces += [
                [m_one, ring[1], m_two, centre],
                [ring[0], m_one, centre, ring[3]],
                [centre, m_two, ring[2], ring[3]],
            ]
        else:
            centre_point = tuple(positions[face].mean(axis=0))
            new_positions.append(centre_point)
            centre = len(new_positions) - 1
            for i in range(4):
                one, two = face[i], face[(i + 1) % 4]
                if flags[i]:
                    m = midpoint(edges[i])
                    out_faces += [[one, m, centre], [m, two, centre]]
                else:
                    out_faces.append([one, two, centre])

    from ..core.types import MeshArrays

    if query is not None and len(new_positions) > len(positions):
        extra = np.asarray(new_positions[len(positions) :], dtype=np.float64)
        projected = query.nearest_points(extra)
        new_positions[len(positions) :] = [tuple(p) for p in projected]

    sizes = np.asarray([len(face) for face in out_faces], dtype=np.int32)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32)
    built = MeshArrays(
        positions=np.asarray(new_positions, dtype=np.float64),
        corner_verts=np.concatenate([np.asarray(face, dtype=np.int32) for face in out_faces]),
        face_starts=starts,
        face_sizes=sizes,
    )
    try:
        mesh_io.replace_object_mesh(obj, built)
    except ValueError:
        repaired = bpy.data.meshes.new(obj.data.name)
        repaired.from_pydata(
            [tuple(point) for point in built.positions],
            [],
            [list(face) for face in out_faces],
        )
        repaired.validate(verbose=False)
        _replace_mesh(obj, repaired)
    return refined


def _field_remesh_best(obj: bpy.types.Object, recipe: RetopoRecipe, progress=None) -> str:
    """Evaluate both field extractors at full budget and retain the better mesh."""
    import dataclasses
    import re as _re

    say = progress or (lambda fraction, message="": None)
    candidates: list[tuple[float, float, bpy.types.Object, str]] = []
    for attempt, name in ((0, "isolines"), (1, "graph")):
        trial = obj.copy()
        trial.data = obj.data.copy()
        bpy.context.scene.collection.objects.link(trial)
        try:
            outcome = _field_remesh(trial, recipe, progress, attempt=attempt, solo=True)
        except ValueError as exc:
            print(f"Rebuilt: extractor {name} failed — {exc}")
            mesh = trial.data
            bpy.data.objects.remove(trial)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
            continue
        faces = len(trial.data.polygons)
        quads = sum(1 for p in trial.data.polygons if len(p.vertices) == 4)
        match = _re.search(r"(\d+) patches", outcome)
        patches = int(match.group(1)) if match else 0
        candidates.append((quads / max(faces, 1), -patches / max(faces, 1), trial, outcome))

    if candidates:
        candidates.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        winner = candidates[0]
        _replace_mesh(obj, winner[2].data.copy())
        for _, _, trial, _ in candidates:
            mesh = trial.data
            bpy.data.objects.remove(trial)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        return winner[3]

    if recipe.target_faces >= 1000:
        say(0.5, "Retrying at half the budget")
        coarser = dataclasses.replace(recipe, target_faces=recipe.target_faces // 2)
        outcome = _field_remesh(obj, coarser, progress, attempt=2)
        if outcome.startswith("field("):
            return outcome.replace("field(", "field(half-budget, ", 1)
        return outcome
    return _quad_remesh(obj, recipe, progress)


def _field_remesh(
    obj: bpy.types.Object,
    recipe: RetopoRecipe,
    progress=None,
    attempt: int = 0,
    solo: bool = False,
) -> str:
    """Quadrangulate with the add-on's own curvature-aligned remesher."""
    from ..core.analysis.crossfield import cross_field
    from ..core.analysis.curvature import principal_directions
    from ..core.analysis.extract import extract_quads
    from ..core.analysis.fields import curvature_sizing, graded_edge_length
    from ..core.analysis.mesh_graph import (
        face_areas,
        manifold_edges,
        taubin_smoothed,
        vertex_normals,
    )
    from ..core.analysis.positionfield import lattice_sites, position_field, repair_lattice
    from ..core.types import MeshArrays

    if attempt == 0 and not solo:
        return _field_remesh_best(obj, recipe, progress)

    say = progress or (lambda fraction, message="": None)
    target = max(recipe.target_faces, 8)

    say(0.1, "Measuring how the surface bends")
    arrays = mesh_io.mesh_to_arrays(obj.data)
    normals = vertex_normals(arrays)

    scan = arrays
    cell_estimate = float(np.sqrt(face_areas(arrays).sum() / max(target, 1)))
    median_edge = float(np.median(manifold_edges(arrays).lengths)) or 1.0
    cell_ratio = cell_estimate / median_edge
    if cell_ratio > FIELD_MAX_CELL_RATIO:
        say(0.15, "Coarsening the measuring mesh")
        triangles = int(2.0 * MEASURE_CELL_RATIO**2 * target)
        arrays = _decimated_arrays(obj, triangles)
        if arrays is None or arrays.num_faces == 0:
            say(0.5, "Budget too coarse for the field engine; using QuadriFlow")
            print("Rebuilt: field remesher stepped aside — decimation failed")
            return _quad_remesh(obj, recipe, progress)
        normals = vertex_normals(arrays)
        median_edge = float(np.median(manifold_edges(arrays).lengths)) or 1.0
        cell_ratio = cell_estimate / median_edge
        if cell_ratio > FIELD_MAX_CELL_RATIO:
            say(0.5, "Budget too coarse for the field engine; using QuadriFlow")
            print(
                f"Rebuilt: field remesher stepped aside — cell/edge {cell_ratio:.2f} "
                f"even after coarsening (max {FIELD_MAX_CELL_RATIO})"
            )
            return _quad_remesh(obj, recipe, progress)

    surface = bvh.SurfaceQuery(scan)

    melt = int(np.clip(FIELD_SMOOTHING_PER_CELL * cell_ratio, 25, 200))
    clean = MeshArrays(
        positions=taubin_smoothed(arrays, passes=melt, shrink=0.5, inflate=-0.1),
        corner_verts=arrays.corner_verts,
        face_starts=arrays.face_starts,
        face_sizes=arrays.face_sizes,
    )
    clean_normals = vertex_normals(clean)

    say(0.25, "Aligning the cross field")

    centered = arrays.positions.astype(np.float64)
    centered = centered - centered.mean(axis=0)
    _, axes = np.linalg.eigh(centered.T @ centered)
    up = axes[:, -1]

    alignment_to_world = np.abs(up)
    nearest_axis = int(np.argmax(alignment_to_world))
    if alignment_to_world[nearest_axis] > np.cos(np.radians(15.0)):
        up = np.eye(3)[nearest_axis]
    field_start = up - clean_normals * (clean_normals @ up)[:, None]

    shaped = MeshArrays(
        positions=taubin_smoothed(arrays, passes=12),
        corner_verts=arrays.corner_verts,
        face_starts=arrays.face_starts,
        face_sizes=arrays.face_sizes,
    )
    creased = manifold_edges(shaped)
    anchors = _crease_anchors(shaped, creased)

    field = cross_field(clean, clean_normals, start=field_start, anchors=anchors)

    say(0.45, f"Sizing cells for {target:,} quads")

    shaped_normals = vertex_normals(shaped)
    _, _, k_major, k_minor = principal_directions(shaped, shaped_normals)

    contrast = recipe.density_contrast if recipe.adaptive_density else 0.0
    _, spacing = curvature_sizing(shaped, k_major, k_minor, target_faces=target, contrast=contrast)

    thickness = surface.inward_spans(
        arrays.positions.astype(np.float64), normals, caps=3.0 * spacing
    )

    manifold = manifold_edges(arrays)
    local_edge = np.full(arrays.num_verts, median_edge, dtype=np.float64)
    if manifold.count:
        accumulated = np.zeros(arrays.num_verts, dtype=np.float64)
        counts = np.zeros(arrays.num_verts, dtype=np.float64)
        for column in (0, 1):
            np.add.at(accumulated, manifold.edges[:, column], manifold.lengths)
            np.add.at(counts, manifold.edges[:, column], 1.0)
        heard = counts > 0
        local_edge[heard] = accumulated[heard] / counts[heard]
    spacing = np.minimum(spacing, np.maximum(0.5 * thickness, 1.1 * local_edge))
    spacing = graded_edge_length(arrays, spacing)

    say(0.6, "Settling the lattice")

    sites = lattice_sites(arrays, field, spacing, normals)
    sites = position_field(arrays, field, spacing, normals, start=sites)
    sites = repair_lattice(arrays, sites, field, spacing, normals)

    say(0.75, "Reading off the quads")

    method = "isoline" if attempt % 2 == 0 else "graph"
    built, report = extract_quads(arrays, sites, field, normals, spacing, method=method)

    scan_boundary = _boundary_vertices(arrays, cell_estimate)
    close_under = 0.0 if recipe.preserve_boundary else SMALL_OPENING_CELLS
    patched = 0
    try:
        if built.num_faces == 0:
            raise ValueError("the field remesher produced no faces")
        if report["quad_ratio"] < MIN_FIELD_QUADS:
            raise ValueError(f"only {report['quad_ratio']:.0%} of the faces came out as quads")

        say(0.85, "Finishing the mesh")
        pristine = obj.data.copy()
        try:
            mesh_io.replace_object_mesh(obj, built)

            say(0.88, "Patching the holes")
            patched, _ = _close_small_holes(obj, scan_boundary, cell_estimate, close_under)
            say(0.9, "Pairing the leftover triangles into quads")
            _requad_cleanup(obj)
            healed, leftover = _close_small_holes(obj, scan_boundary, cell_estimate, close_under)
            patched += healed
            if leftover > 0:
                raise ValueError(f"{leftover} artefact edges survived the patching")

            final_faces = len(obj.data.polygons)
            final_quads = sum(1 for p in obj.data.polygons if len(p.vertices) == 4)

            if final_faces < 0.25 * target:
                raise ValueError(
                    f"patching consumed the mesh: {final_faces} faces left of {target} asked"
                )

            if final_quads < 0.85 * max(final_faces, 1):
                raise ValueError(
                    f"only {final_quads / max(final_faces, 1):.0%} quads after patching"
                )

            say(0.92, "Relaxing the loops onto the scan")

            smooth_scan = MeshArrays(
                positions=taubin_smoothed(scan, passes=12),
                corner_verts=scan.corner_verts,
                face_starts=scan.face_starts,
                face_sizes=scan.face_sizes,
            )
            _relax_onto_surface(obj, bvh.SurfaceQuery(smooth_scan), reach=2.0 * cell_estimate)
        except ValueError:
            _replace_mesh(obj, pristine.copy())
            raise
        finally:
            if pristine.users == 0:
                bpy.data.meshes.remove(pristine)
    except ValueError as exc:
        if solo:
            raise
        say(0.9, "Field remesher declined; falling back")
        print(f"Rebuilt: field remesher fell back — {exc}")

        import dataclasses as _dataclasses

        if attempt < 3 and recipe.target_faces >= 1000:
            next_attempt = attempt + 1
            if next_attempt == 2:
                say(0.5, "Retrying at half the budget")
                coarser = _dataclasses.replace(recipe, target_faces=recipe.target_faces // 2)
            else:
                say(0.5, "Retrying with the graph extractor")
                coarser = recipe
            outcome = _field_remesh(obj, coarser, progress, attempt=next_attempt)
            if outcome.startswith("field(") and next_attempt == 2:
                return outcome.replace("field(", "field(half-budget, ", 1)
            return outcome
        return _quad_remesh(obj, recipe, progress)

    return f"field({report['quad_ratio']:.0%} quads, {patched} patches)"


FIELD_MAX_CELL_RATIO = 2.4


MEASURE_CELL_RATIO = 1.5


def _decimated_arrays(obj: bpy.types.Object, triangles: int):
    """The object's mesh collapsed to a triangle budget, as arrays."""
    temp = obj.copy()
    temp.data = obj.data.copy()
    bpy.context.scene.collection.objects.link(temp)
    try:
        _decimate_to(temp, triangles)
        return mesh_io.mesh_to_arrays(temp.data)
    except RuntimeError as exc:
        print(f"Rebuilt: measuring-mesh decimation failed — {exc}")
        return None
    finally:
        mesh = temp.data
        bpy.data.objects.remove(temp)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _requad_cleanup(obj: bpy.types.Object) -> int:
    """Local topology surgery: absorb stray triangles into the quad flow."""
    import bmesh

    mesh = obj.data
    built = bmesh.new()
    built.from_mesh(mesh)
    reworked = 0

    triangles = [face for face in built.faces if len(face.verts) == 3]
    if triangles:
        joined = bmesh.ops.join_triangles(
            built,
            faces=triangles,
            angle_face_threshold=np.pi,
            angle_shape_threshold=np.pi,
        ).get("faces", [])
        reworked += len(joined)

    for _ in range(5):
        merged_any = False

        for face in list(built.faces):
            if not face.is_valid or len(face.verts) != 3:
                continue
            neighbourhood = {
                len(other.verts)
                for edge in face.edges
                for other in edge.link_faces
                if other is not face
            }
            if neighbourhood != {4}:
                continue
            longest = max(
                (edge for edge in face.edges if len(edge.link_faces) == 2),
                key=lambda edge: (edge.verts[0].co - edge.verts[1].co).length_squared,
                default=None,
            )
            if longest is not None:
                bmesh.ops.dissolve_edges(built, edges=[longest], use_verts=False)
                merged_any = True

        for edge in list(built.edges):
            if not edge.is_valid or len(edge.link_faces) != 2:
                continue
            first, second = edge.link_faces
            sizes = sorted((len(first.verts), len(second.verts)))
            if sizes == [3, 5]:
                region = bmesh.ops.dissolve_edges(built, edges=[edge], use_verts=False)
                for face in region.get("region", []):
                    if face.is_valid and len(face.verts) == 6 and _split_hexagon(built, face):
                        reworked += 2
                        merged_any = True
            elif sizes == [5, 5]:
                region = bmesh.ops.dissolve_edges(built, edges=[edge], use_verts=False)
                for face in region.get("region", []):
                    if face.is_valid and len(face.verts) == 8 and _split_octagon(built, face):
                        reworked += 3
                        merged_any = True
        if not merged_any:
            break

    stubborn = [face for face in built.faces if face.is_valid and len(face.verts) >= 6]
    if stubborn:
        splittable = [face for face in stubborn if len(face.verts) == 6]
        for hexagon in splittable:
            if hexagon.is_valid and _split_hexagon(built, hexagon):
                stubborn.remove(hexagon)
                reworked += 2
        remaining = [face for face in stubborn if face.is_valid]
        if remaining:
            poked = bmesh.ops.poke(built, faces=remaining).get("faces", [])
            if poked:
                bmesh.ops.join_triangles(
                    built,
                    faces=poked,
                    angle_face_threshold=np.pi,
                    angle_shape_threshold=np.pi,
                )
                reworked += len(poked)

    leftovers = [face for face in built.faces if face.is_valid and len(face.verts) == 3]
    if leftovers:
        reworked += len(
            bmesh.ops.join_triangles(
                built,
                faces=leftovers,
                angle_face_threshold=np.pi,
                angle_shape_threshold=np.pi,
            ).get("faces", [])
        )

    built.to_mesh(mesh)
    mesh.update()
    built.free()
    return reworked


def _split_octagon(built, octagon) -> bool:
    """Split an eight-sided face into a fan of three quads from one vertex."""
    import bmesh

    loops = octagon.loops[:]
    if len(loops) != 8:
        return False
    openings = sorted(
        ((index, loops[index].vert, loops[(index + 3) % 8].vert) for index in range(8)),
        key=lambda entry: (entry[1].co - entry[2].co).length_squared,
    )
    for _, start, end in openings:
        try:
            piece, _ = bmesh.utils.face_split(octagon, start, end)
        except ValueError:
            continue
        for face in (octagon, piece):
            if face.is_valid and len(face.verts) == 6:
                _split_hexagon(built, face)
        return True
    return False


def _split_hexagon(built, hexagon) -> bool:
    """Split a six-sided face into two quads along whichever diagonal works."""
    import bmesh

    loops = hexagon.loops[:]
    if len(loops) != 6:
        return False
    diagonals = sorted(
        ((loops[i].vert, loops[i + 3].vert) for i in range(3)),
        key=lambda pair: (pair[0].co - pair[1].co).length_squared,
    )
    for start, end in diagonals:
        try:
            bmesh.utils.face_split(hexagon, start, end)
            return True
        except ValueError:
            continue
    return False


RELAX_PASSES = 10


def _relax_onto_surface(
    obj: bpy.types.Object, query, passes: int = RELAX_PASSES, reach: float | None = None
) -> None:
    """Even out the quads, keeping every vertex on the scanned surface."""
    mesh = obj.data
    count = len(mesh.vertices)
    if count == 0 or len(mesh.edges) == 0:
        return

    flat = np.empty(count * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", flat)
    positions = flat.reshape(-1, 3).astype(np.float64)

    normal_flat = np.empty(count * 3, dtype=np.float32)
    mesh.vertices.foreach_get("normal", normal_flat)
    sides = normal_flat.reshape(-1, 3).astype(np.float64)

    edge_flat = np.empty(len(mesh.edges) * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", edge_flat)
    edges = edge_flat.reshape(-1, 2).astype(np.int64)

    starts = np.concatenate([edges[:, 0], edges[:, 1]])
    ends = np.concatenate([edges[:, 1], edges[:, 0]])
    valence = np.zeros(count, dtype=np.float64)
    np.add.at(valence, starts, 1.0)
    heard = valence > 0.0
    valence = np.maximum(valence, 1.0)[:, None]

    for _ in range(passes):
        summed = np.zeros_like(positions)
        np.add.at(summed, starts, positions[ends])
        averaged = summed / valence
        moved = np.where(heard[:, None], 0.5 * (positions + averaged), positions)
        projected = query.nearest_points(moved, sides=sides)
        if reach is None:
            positions = projected
        else:
            distance = np.linalg.norm(projected - moved, axis=1)
            claimed = distance <= reach
            positions = np.where(claimed[:, None], projected, positions)

    mesh.vertices.foreach_set("co", positions.astype(np.float32).ravel())
    mesh.update()


REAL_OPENING_REACH = 2.0


SMALL_OPENING_CELLS = 6.0


def _close_small_holes(
    obj: bpy.types.Object, scan_boundary: np.ndarray, cell: float, close_under: float = 0.0
) -> tuple[int, int]:
    """Fill the artefact holes, in place. Returns (patched, artefact edges left)."""
    import bmesh

    mesh = obj.data
    built = bmesh.new()
    built.from_mesh(mesh)
    patched = 0

    def artefact_edges() -> list:
        rims = [edge for edge in built.edges if edge.is_valid and len(edge.link_faces) == 1]
        if not rims:
            return []
        return [
            edge
            for edges, near_fraction in _boundary_pieces(rims, scan_boundary, cell)
            if near_fraction <= 0.6 or _piece_extent(edges) < close_under * cell
            for edge in edges
        ]

    for _ in range(3):
        open_edges = artefact_edges()
        if not open_edges:
            break

        fins = {
            face
            for edge in open_edges
            if edge.is_valid
            for face in edge.link_faces
            if any(len(other.link_faces) > 2 for other in face.edges)
        }
        if fins:
            bmesh.ops.delete(built, geom=list(fins), context="FACES")
            open_edges = artefact_edges()
            if not open_edges:
                break

        for piece_edges, _ in _boundary_pieces(open_edges, scan_boundary, cell):
            valid = [edge for edge in piece_edges if edge.is_valid]
            if len(valid) < 4 or len(valid) % 2 != 0:
                continue
            try:
                woven_grid = bmesh.ops.grid_fill(built, edges=valid).get("faces", [])
                patched += len(woven_grid)
            except RuntimeError:
                pass
        open_edges = artefact_edges()
        if not open_edges:
            break

        lids = bmesh.ops.holes_fill(built, edges=open_edges, sides=0).get("faces", [])
        patched += len(lids)

        cracked = artefact_edges()
        if cracked:
            built.normal_update()
            loose = list({vert for edge in cracked for vert in edge.verts})
            from mathutils import kdtree

            finder = kdtree.KDTree(len(loose))
            for index, vert in enumerate(loose):
                finder.insert(vert.co, index)
            finder.balance()
            reach = 0.35 * max(cell, 1e-9)
            targetmap = {}
            claimed = set()
            for vert in loose:
                if vert in targetmap or vert in claimed:
                    continue
                for _, index, _ in finder.find_range(vert.co, reach):
                    other = loose[index]
                    if other is vert or other in targetmap or other in claimed:
                        continue
                    if vert.normal.dot(other.normal) <= 0.0:
                        continue
                    targetmap[vert] = other
                    claimed.add(other)
                    break
            if targetmap:
                bmesh.ops.weld_verts(built, targetmap=targetmap)

        nets = artefact_edges()
        if nets:
            woven = bmesh.ops.triangle_fill(
                built, edges=nets, use_beauty=True, use_dissolve=False
            ).get("geom", [])
            laid = [item for item in woven if isinstance(item, bmesh.types.BMFace)]
            patched += len(laid)

        slits = [
            edge
            for edge in artefact_edges()
            if (edge.verts[0].co - edge.verts[1].co).length < 0.3 * max(cell, 1e-9)
        ]
        if slits:
            bmesh.ops.collapse(built, edges=slits, uvs=False)

    deleted_budget = max(1, int(0.03 * len(built.faces)))
    deleted_total = 0
    previous_count = None
    for _ in range(5):
        tangled = artefact_edges()
        if not tangled:
            break

        if previous_count is not None and len(tangled) >= previous_count and len(tangled) > 8:
            break
        previous_count = len(tangled)
        pieces = _boundary_pieces(tangled, scan_boundary, cell)
        small = [edge for edges, _ in pieces if len(edges) <= 80 for edge in edges]
        if not small:
            break
        touched = {face for edge in small if edge.is_valid for face in edge.link_faces}
        ring = {
            neighbour for face in touched for edge in face.edges for neighbour in edge.link_faces
        }
        doomed = list(touched | ring)
        if deleted_total + len(doomed) > deleted_budget:
            break
        if doomed:
            deleted_total += len(doomed)
            bmesh.ops.delete(built, geom=doomed, context="FACES")
        reopened = artefact_edges()
        if reopened:
            patched += len(bmesh.ops.holes_fill(built, edges=reopened, sides=0).get("faces", []))
        reopened = artefact_edges()
        if reopened:
            woven_cut = bmesh.ops.triangle_fill(
                built, edges=reopened, use_beauty=True, use_dissolve=False
            ).get("geom", [])
            patched += sum(1 for item in woven_cut if isinstance(item, bmesh.types.BMFace))

    last_few = artefact_edges()
    hatch_limit = max(12, int(0.003 * len(built.edges)))
    if 0 < len(last_few) <= hatch_limit:
        loose = list({vert for edge in last_few for vert in edge.verts})

        bmesh.ops.remove_doubles(built, verts=loose, dist=0.55 * max(cell, 1e-9))
        remaining = artefact_edges()
        if remaining:
            filled_last = bmesh.ops.holes_fill(built, edges=remaining, sides=0).get("faces", [])
            patched += len(filled_last)
        remaining = artefact_edges()
        if remaining:
            bmesh.ops.collapse(built, edges=remaining, uvs=False)

    wide = [face for face in built.faces if face.is_valid and len(face.verts) > 5]
    if wide:
        lid_triangles = bmesh.ops.triangulate(built, faces=wide).get("faces", [])
        if lid_triangles:
            bmesh.ops.join_triangles(
                built,
                faces=lid_triangles,
                angle_face_threshold=np.pi,
                angle_shape_threshold=np.pi,
            )

    reopened_late = artefact_edges()
    if reopened_late:
        patched += len(bmesh.ops.holes_fill(built, edges=reopened_late, sides=0).get("faces", []))

    for stubborn in _closed_loops(artefact_edges()):
        if not 3 <= len(stubborn) <= 5:
            continue
        try:
            built.faces.new(stubborn)
            patched += 1
        except ValueError:
            continue

    final_cracks = artefact_edges()
    if 0 < len(final_cracks) <= 4:
        doomed_verts = list({vert for edge in final_cracks for vert in edge.verts})
        bmesh.ops.delete(built, geom=doomed_verts, context="VERTS")
        reopened = artefact_edges()
        if reopened:
            patched += len(bmesh.ops.holes_fill(built, edges=reopened, sides=0).get("faces", []))
        reopened = artefact_edges()
        if reopened:
            woven_last = bmesh.ops.triangle_fill(
                built, edges=reopened, use_beauty=True, use_dissolve=False
            ).get("geom", [])
            patched += sum(1 for item in woven_last if isinstance(item, bmesh.types.BMFace))

    orphans = [vert for vert in built.verts if not vert.link_faces]
    if orphans:
        bmesh.ops.delete(built, geom=orphans, context="VERTS")

    leftover = len(artefact_edges())
    built.to_mesh(mesh)
    mesh.update()
    built.free()
    return patched, leftover


def _closed_loops(edges: list) -> list[list]:
    """The closed cycles among a set of boundary edges, as ordered vertex lists."""
    incident: dict = {}
    for edge in edges:
        for vert in edge.verts:
            incident.setdefault(vert, []).append(edge)
    loops = []
    visited = set()
    for edge in edges:
        if edge in visited:
            continue
        walk = [edge.verts[0], edge.verts[1]]
        trail = [edge]
        broken = False
        while True:
            here = walk[-1]
            options = [e for e in incident.get(here, []) if e is not trail[-1]]
            if len(incident.get(here, [])) != 2 or len(options) != 1:
                broken = True
                break
            trail.append(options[0])
            following = options[0].other_vert(here)
            if following is walk[0]:
                break
            if following in walk:
                broken = True
                break
            walk.append(following)
        visited.update(trail)
        if not broken:
            loops.append(walk)
    return loops


def _piece_extent(edges: list) -> float:
    """Diagonal of the bounding box of a boundary piece."""
    points = np.asarray([vert.co for edge in edges for vert in edge.verts], dtype=np.float64)
    if points.size == 0:
        return 0.0
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def _boundary_pieces(rims: list, scan_boundary: np.ndarray, cell: float):
    """Boundary edges grouped into connected pieces, with how much of each piece hugs the scan's own boundary."""
    piece: dict[int, int] = {}
    parent = list(range(len(rims)))

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for index, edge in enumerate(rims):
        for vert in edge.verts:
            other = piece.setdefault(vert.index, index)
            one, two = root(index), root(other)
            if one != two:
                parent[one] = two

    reach = REAL_OPENING_REACH * max(cell, 1e-9)
    grouped: dict[int, list] = {}
    near: dict[int, int] = {}
    for index, edge in enumerate(rims):
        component = root(index)
        grouped.setdefault(component, []).append(edge)
        if scan_boundary.size:
            middle = 0.5 * (np.asarray(edge.verts[0].co) + np.asarray(edge.verts[1].co))
            if float(np.min(np.linalg.norm(scan_boundary - middle, axis=1))) < reach:
                near[component] = near.get(component, 0) + 1

    return [(edges, near.get(component, 0) / len(edges)) for component, edges in grouped.items()]


def _triangulate_to_budget(
    obj: bpy.types.Object, recipe: RetopoRecipe, weights: np.ndarray | None, progress=None
) -> str:
    """Collapse to the budget and leave it in triangles, because it was asked for."""
    say = progress or (lambda fraction, message="": None)
    say(0.1, f"Decimating to {max(recipe.target_faces, 4):,} triangles")

    group = _density_vertex_group(obj, weights) if weights is not None else None
    source_triangles = _triangle_count(obj)
    _collapse_to_budget(
        obj,
        max(recipe.target_faces, 4),
        source_triangles,
        vertex_group=group,
        weight_factor=1.0,
    )
    return "decimation"


def _restore_quadriflow_prepared(obj, prepared):
    """Recreate private preparation, including the edge ordering used by decimation."""
    arrays = prepared["arrays"]
    mesh = bpy.data.meshes.new(obj.data.name)
    try:
        mesh.vertices.add(arrays.num_verts)
        mesh.edges.add(len(prepared["edges"]))
        mesh.loops.add(arrays.num_corners)
        mesh.polygons.add(arrays.num_faces)
        mesh.vertices.foreach_set("co", arrays.positions.astype(np.float32).ravel())
        mesh.edges.foreach_set("vertices", prepared["edges"].ravel())
        mesh.loops.foreach_set("vertex_index", arrays.corner_verts.astype(np.int32))
        mesh.loops.foreach_set("edge_index", prepared["loop_edges"])
        mesh.polygons.foreach_set("loop_start", arrays.face_starts.astype(np.int32))
        mesh.polygons.foreach_set("loop_total", arrays.face_sizes.astype(np.int32))
        # Preserve the prepared topology, including inputs the existing worker
        # can repair. Regenerating edges or validating here changes that boundary.
        mesh.update()
    except BaseException:
        bpy.data.meshes.remove(mesh)
        raise
    _replace_mesh(obj, mesh)


def _quadriflow_triangle_budget(target, original_triangles, rung):
    budget = target * PRE_DECIMATE_LADDER[rung]
    if rung:
        budget = min(budget, int(original_triangles * 0.5**rung))
    return budget


def _quad_remesh(obj: bpy.types.Object, recipe: RetopoRecipe, progress=None) -> str:
    """Turn whatever `obj` currently is into quads, on budget."""
    target = max(recipe.target_faces, 8)
    original_triangles = _triangle_count(obj)
    say = progress or (lambda fraction, message="": None)

    for rung in range(len(PRE_DECIMATE_LADDER)):
        say(
            0.1 + 0.7 * rung / len(PRE_DECIMATE_LADDER),
            f"Quadrangulating to {target:,} quads"
            + (f" (attempt {rung + 1} of {len(PRE_DECIMATE_LADDER)})" if rung else ""),
        )
        pristine = obj.data.copy()
        try:
            _decimate_to(obj, _quadriflow_triangle_budget(target, original_triangles, rung))
            if _quadriflow_to_budget(obj, recipe, target):
                return "quadriflow"
            _replace_mesh(obj, pristine.copy())
        finally:
            if pristine.users == 0:
                bpy.data.meshes.remove(pristine)

    say(0.8, "Pairing triangles into quads (the remesher refused this mesh)")
    _quads_by_conversion(obj, target)
    return "conversion"


def _collapse_to_budget(
    obj: bpy.types.Object,
    target_faces: int,
    source_triangles: int,
    vertex_group: str | None = None,
    weight_factor: float = 1.0,
) -> None:
    """Collapse-decimate until the face count lands on the budget."""
    if source_triangles <= 0 or target_faces <= 0:
        return

    ratio = min(1.0, target_faces / source_triangles)

    for _ in range(MAX_BUDGET_PASSES):
        modifier = obj.modifiers.new("rebuilt_decimate", "DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = max(min(ratio, 1.0), 1e-6)
        modifier.use_collapse_triangulate = True
        if vertex_group:
            modifier.vertex_group = vertex_group
            modifier.vertex_group_factor = weight_factor

        bpy.context.view_layer.update()
        achieved = modifier.face_count

        _apply_modifier(obj, modifier)

        if achieved <= 0:
            return
        error = budget_error(achieved, target_faces)
        if abs(error) <= BUDGET_TOLERANCE:
            return

        source_triangles = sum(len(p.vertices) - 2 for p in obj.data.polygons)
        if source_triangles <= target_faces:
            return
        ratio = target_faces / source_triangles


WEIGHT_BUCKETS = 32


def _density_vertex_group(obj: bpy.types.Object, weights: np.ndarray) -> str | None:
    """Store the density field as a vertex group."""
    if weights.size != len(obj.data.vertices):
        return None

    clamped = np.clip(weights, 0.0, 1.0)
    if float(clamped.max() - clamped.min()) < 1e-9:
        return None

    name = "rebuilt_density"
    group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)

    buckets = np.clip((clamped * WEIGHT_BUCKETS).astype(np.int64), 0, WEIGHT_BUCKETS - 1)
    for bucket in range(WEIGHT_BUCKETS):
        members = np.flatnonzero(buckets == bucket)
        if members.size:
            group.add(members.tolist(), (bucket + 0.5) / WEIGHT_BUCKETS, "REPLACE")
    return name


def _thin_adaptive(obj, recipe, weights, source_triangles):
    group = _density_vertex_group(obj, weights)

    budget = max(recipe.target_faces, 8) * PRE_DECIMATE_LADDER[0]
    if source_triangles > budget > 0:
        _collapse_to_budget(
            obj,
            budget,
            source_triangles,
            vertex_group=group,
            weight_factor=1.0,
        )


def _adaptive_decimate(
    obj: bpy.types.Object,
    recipe: RetopoRecipe,
    weights: np.ndarray,
    source_triangles: int,
    progress=None,
) -> str:
    """Thin by the density field where the input is too dense, then quadrangulate."""
    say = progress or (lambda fraction, message="": None)
    say(0.05, "Thinning by the density field")
    _thin_adaptive(obj, recipe, weights, source_triangles)
    return _quad_remesh(obj, recipe, progress)


def _hard_surface(obj: bpy.types.Object, recipe: RetopoRecipe, progress=None) -> str:
    """Quadrangulate, holding on to the corners."""
    import dataclasses

    return _quad_remesh(
        obj,
        dataclasses.replace(recipe, preserve_sharp=True, preserve_boundary=True),
        progress,
    )


WELD_TOLERANCE = 1e-6


def _weld_coincident(obj: bpy.types.Object, diagonal: float) -> int:
    """Merge vertices the remesher left sitting on top of each other."""
    import bmesh

    before = len(obj.data.vertices)
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    try:
        bmesh.ops.remove_doubles(
            mesh, verts=mesh.verts[:], dist=max(diagonal, 1e-9) * WELD_TOLERANCE
        )
        mesh.to_mesh(obj.data)
    finally:
        mesh.free()
    obj.data.update()
    return before - len(obj.data.vertices)


STRAY_PIECE_FACES = 32
STRAY_PIECE_FRACTION = 0.005


def _output_hygiene(obj: bpy.types.Object, diagonal: float) -> dict:
    """What no engine may leave in the mesh that ships: zero-area faces and stray pieces. Returns what went, for the report."""
    import bmesh

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    faces_before = len(bm.faces)
    epsilon = 1e-5 * max(diagonal, 1e-9)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges[:], dist=epsilon)

    for _ in range(4):
        flat = [face for face in bm.faces if face.is_valid and face.calc_area() < epsilon * epsilon]
        if not flat:
            break
        for face in flat:
            if not face.is_valid:
                continue

            safe = [
                edge
                for edge in sorted(face.edges, key=lambda edge: edge.calc_length())
                if set(edge.verts[0].link_faces) & set(edge.verts[1].link_faces)
                == set(edge.link_faces)
            ]
            if safe:
                bmesh.ops.collapse(bm, edges=[safe[0]], uvs=False)
            else:
                bmesh.ops.delete(bm, geom=[face], context="FACES_ONLY")
    degenerate = faces_before - len(bm.faces)

    bm.faces.ensure_lookup_table()
    parent = list(range(len(bm.faces)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for edge in bm.edges:
        linked = edge.link_faces
        if len(linked) != 2:
            continue
        a, b = root(linked[0].index), root(linked[1].index)
        if a != b:
            parent[a] = b
    sizes: dict[int, int] = {}
    for face in bm.faces:
        r = root(face.index)
        sizes[r] = sizes.get(r, 0) + 1
    total = len(bm.faces)
    limit = min(STRAY_PIECE_FACES, max(3, int(STRAY_PIECE_FRACTION * total)))
    doomed = [face for face in bm.faces if sizes[root(face.index)] < limit]
    strays = len(doomed)
    if doomed and strays < total:
        bmesh.ops.delete(bm, geom=doomed, context="FACES")
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    mesh.update()
    bm.free()
    report = {"degenerate_faces": degenerate, "stray_faces": strays}
    if degenerate or strays:
        print(
            f"Rebuilt: output hygiene — {degenerate} degenerate face(s) dissolved, "
            f"{strays} stray face(s) removed"
        )
    return report


MAX_REMESH_HOLE = 4


def _close_remesh_holes(obj: bpy.types.Object) -> int:
    """Close the small holes the remesher punched in a closed surface."""
    import bmesh

    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    try:
        boundary = [edge for edge in mesh.edges if len(edge.link_faces) == 1]
        if not boundary:
            return 0
        before = len(mesh.faces)
        bmesh.ops.holes_fill(mesh, edges=boundary, sides=MAX_REMESH_HOLE)
        closed = len(mesh.faces) - before
        mesh.to_mesh(obj.data)
    finally:
        mesh.free()
    obj.data.update()
    return closed


def _orient_native_surface(low, source) -> dict:
    before = np.empty((len(low.data.polygons), 3), dtype=np.float32)
    low.data.polygons.foreach_get("normal", before.ravel())
    _normals_outward(low, source, force=True)
    after = np.empty_like(before)
    low.data.polygons.foreach_get("normal", after.ravel())
    return {"reversed_faces": int((np.einsum("ij,ij->i", before, after) < 0).sum())}


def _project_onto(low: bpy.types.Object, high: bpy.types.Object) -> dict:
    """Pull the low mesh back onto the high surface."""
    from ..core.mesh_edit import guard_surface_projection

    before = mesh_io.mesh_to_arrays(low.data)
    modifier = low.modifiers.new("rebuilt_project", "SHRINKWRAP")
    modifier.target = high
    modifier.wrap_method = "NEAREST_SURFACEPOINT"
    modifier.wrap_mode = "ON_SURFACE"
    _apply_modifier(low, modifier)
    projected = np.empty((len(low.data.vertices), 3), dtype=np.float32)
    low.data.vertices.foreach_get("co", projected.ravel())
    coordinates, report = guard_surface_projection(before, projected)
    if report["limited_vertices"]:
        low.data.vertices.foreach_set("co", coordinates.ravel())
        low.data.update()
    return report
