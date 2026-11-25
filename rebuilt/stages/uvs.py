"""Unwrapping and packing the low mesh, along the seams it carries."""

from __future__ import annotations

from ..runtime.session import verify_current_runtime

from typing import Any, ClassVar
import time
import threading

import bpy
import numpy as np

from ..bridge import mesh_io, uv_io
from ..core.analysis.mesh_graph import face_areas
from ..core.jobs import ProgressReporter
from ..core.recipe import AssetRecipe, UVRecipe
from ..core.uv.metrics import introduces_uv_cuts as _introduces_uv_cuts
from ..core.uv.metrics import island_count_from_uvs, overlap_report, singular_values, stretch_report, uv_occupancy, blown_faces, spiked_faces, lost_coverage_faces
from ..core.uv import cleanseams_native
from ..core.uv.repair import invalid_faces
from ..core.uv.protected import cut_corners, cuts_preserved
from ..core.uv.texel import (
    surface_triangles,
    average_texel_density,
    fits_in_tile,
)
from ..core.types import MeshArrays
from ..pipeline.context import AssetContext, ContextError
from .base import Stage
from .retopo import ROLE_LOW
from .seams import LAYOUT_KEY, junk_faces_of, layout_fingerprint, marked_edges, apply_cut


class UVStage(Stage):
    name: ClassVar[str] = "uv"
    slot: ClassVar[str] = "uv"
    role: ClassVar[str] = "uv"
    label: ClassVar[str] = "Unwrapping"
    native_engines = ('cleanseams', 'uvpack')

    def cache_key(self, recipe: AssetRecipe, *, runtime=None) -> str:
        key = f"{recipe.uv.fingerprint()}:{recipe.texture_resolution}:local-repair-4:consolidate-1:preserve-charts-1:packing-recovery-1:best-effort-singletons-1:native-postprocess-1:native-packing-1:native-normalization-pins-1"
        if recipe.uv.packer == "uvpack":
            key += ":uvpack-float32-1"
        return self._with_runtime(key, recipe, runtime)

    def gather(self, ctx: AssetContext) -> dict[str, Any]:
        low = ctx.artifact(ROLE_LOW)
        if low is None:
            raise ContextError("uv needs retopo to have run first")
        return {
            "recipe": ctx.recipe.uv,
            "texture_size": ctx.recipe.texture_resolution,
            "seams": ctx.report("seams"),
        }

    def compute(self, payload: dict[str, Any], report: ProgressReporter) -> dict[str, Any]:
        report.update(1.0, "Ready")
        return {**payload, "_reporter": report}

    def apply(self, ctx: AssetContext, result: dict[str, Any]) -> None:
        """Commit the complete layout only after its final checks pass."""
        low = ctx.artifact(ROLE_LOW)
        original = low.data
        previous_mode = low.mode
        marker = low.get(LAYOUT_KEY)
        if previous_mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        working = original.copy()
        low.data = working
        try:
            self._apply_layout(ctx, result)
        except BaseException:
            if low.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            low.data = original
            if marker is None:
                if LAYOUT_KEY in low:
                    del low[LAYOUT_KEY]
            else:
                low[LAYOUT_KEY] = marker
            bpy.data.meshes.remove(working)
            if previous_mode != "OBJECT":
                bpy.ops.object.mode_set(mode=previous_mode)
            raise
        else:
            if original.users == 0:
                name = original.name
                bpy.data.meshes.remove(original)
                working.name = name

    def _apply_layout(self, ctx: AssetContext, result: dict[str, Any]) -> None:
        low = ctx.artifact(ROLE_LOW)
        recipe: UVRecipe = result["recipe"]
        texture_size: int = result["texture_size"]

        if low.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        active = low.data.uv_layers.active
        uv_io.ensure_uv_layer(low.data, active.name if active is not None else uv_io.UV_LAYER)
        arrays = mesh_io.mesh_to_arrays(low.data)
        pins = uv_io.read_pins(low.data)

        seams = _seam_count(low.data)
        junk = junk_faces_of(low.data)
        planned = result.get("seams") or {}
        prepared = (
            planned.get("method") == "cleanseams"
            and planned.get("unwrap_method") == recipe.unwrap_method
            and low.get(LAYOUT_KEY) == layout_fingerprint(low, arrays)
        )
        # Supplied seams are the artist's layout constraints. The option controls
        # what the seam generator must retain, not whether UV may erase edits.
        protected = (planned.get("protected_edges", []) if recipe.keep_marked_seams else []) \
            if prepared else marked_edges(low.data)
        protection = {"protected_edges": protected} if len(protected) else {}
        ctx.progress(0.2, "Unwrapping along the seams")
        if prepared:
            method = "cleanseams"
        elif seams == 0 and junk is None:
            selected = np.ones(arrays.num_faces, dtype=bool)
            if pins:
                corner_faces = np.repeat(np.arange(arrays.num_faces), arrays.face_sizes)
                selected[corner_faces[list(pins)]] = False
            if selected.any():
                uv_io.smart_project(low, faces=selected if pins else None)
            method = "smart_project"
        else:
            uv_io.unwrap(low, method=recipe.unwrap_method)
            method = "seams"
        uv_io.check_pins(low.data, pins)

        # Optional native work runs once before packing. A current Seams result
        # already spent its singleton/join allowance; do not start it again.
        reporter = result.get('_reporter') or ProgressReporter(threading.Event())
        deadline = time.monotonic() + 60.0
        already_processed = prepared and bool(planned.get('final_native_report'))
        optional = {} if already_processed else {'region_recovery_seconds': 1.0}
        if not already_processed:
            optional['singleton_seconds'] = 45.0
            if prepared:
                optional.update(join_seconds=2.0 if recipe.island_preference == 'FEWER' else 1.0,
                                join_mode='fewer' if recipe.island_preference == 'FEWER' else 'small')
        selected = np.zeros(arrays.num_faces, dtype=bool) if already_processed or junk is None else junk.copy()
        current = uv_io.read_uvs(low.data)
        initial_islands = island_count_from_uvs(arrays, np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0))
        if not prepared:
            safe = np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)
            selected |= blown_faces(arrays, safe) | spiked_faces(arrays, safe)
        corner_faces = np.repeat(np.arange(arrays.num_faces), arrays.face_sizes)
        selected[corner_faces[~np.isfinite(current).all(axis=1)]] = True
        if selected.any():
            optional['relax_faces_seconds'] = 1.0
        ctx.progress(0.5, 'Validating and repairing the supplied layout')
        post = self._native_postprocess(low, arrays, reporter, protected, deadline,
                                        options=optional, selected=selected, pins=pins)
        before_pack = uv_io.read_uvs(low.data)
        junk = post.exempt.copy()
        changed = set(np.unique(corner_faces[np.any(current != before_pack, axis=1)]).tolist())
        initial_report = post.report.get('initial_final_validation') or post.native_report
        layout_started = time.monotonic()
        ctx.progress(0.7, 'Packing islands')
        outcome = self._lay_out(low, arrays, recipe, texture_size, junk=junk,
            chart_ids=post.chart_of_face, reporter=reporter, deadline=deadline, **protection)
        outcome['repaired_faces'] = len(changed)
        outcome['unresolved_faces'] = 0
        if (outcome['stretch']['p95'] > recipe.max_stretch and not recipe.preserve_texel_density
                and 2 * (time.monotonic() - layout_started) < 10.0):
            outcome = self._improve_stretch(low, arrays, recipe, texture_size, junk, outcome,
                reporter=reporter, deadline=deadline, **protection)
        final = self._native_postprocess(low, arrays, reporter, protected, deadline,
                                         repair_defects=False, pins=pins)
        uv_io.check_pins(low.data, pins)
        if lost_coverage_faces(arrays, before_pack, uv_io.read_uvs(low.data)).any():
            raise ValueError('Stored packing validation failed: previously valid UV coverage was lost.')
        outcome['repaired_faces'] = len(changed)
        outcome['unresolved_faces'] = 0
        outcome['prepack_native_report'] = initial_report
        outcome['final_native_report'] = final.native_report
        outcome['merged_islands'] = max(0, initial_islands - post.n_charts)
        outcome['singleton_repair'] = initial_report.get('repair_singletons', {})
        remaining = int(np.count_nonzero(np.bincount(final.chart_of_face) == 1))
        outcome['singleton_faces_remaining'] = remaining
        outcome['no_single_face_islands'] = remaining == 0
        if remaining:
            outcome['notes'].append(f'{remaining} single-face UV island(s) remain. Safe joining was not possible; processing continued.')
        if len(protected):
            uv_io.mark_seams(low.data, np.vstack((marked_edges(low.data), protected)))
            outcome["protected_seams"] = len(protected)
            outcome["protected_cuts_preserved"] = cuts_preserved(
                cut_corners(arrays, protected), uv_io.read_uvs(low.data)
            )
        if prepared and recipe.island_preference == "FEWER":
            final_islands = island_count_from_uvs(arrays, uv_io.read_uvs(low.data))
            seam_islands = int(planned.get("islands", final_islands))
            outcome["island_preference"] = "FEWER"
            outcome["islands_after_seams"] = seam_islands
            if final_islands > seam_islands:
                outcome["notes"].append(
                    f"UV repairs required {final_islands - seam_islands} additional island(s)."
                )
        if prepared:
            low[LAYOUT_KEY] = layout_fingerprint(low, arrays)
        elif LAYOUT_KEY in low:
            del low[LAYOUT_KEY]
        ctx.progress(1.0, "Layout measured")
        verify_current_runtime()
        ctx.set_report(
            self.name,
            {
                "method": method,
                "seam_count": _seam_count(low.data),
                "charts": planned.get("charts"),
                "islands": island_count_from_uvs(arrays, uv_io.read_uvs(low.data)),
                "stretch_met": outcome["stretch"]["p95"] <= recipe.max_stretch,
                **outcome,
            },
        )

    def _native_postprocess(self, low, arrays, reporter, protected_edges, deadline,
                            *, options=None, selected=None, repair_defects=True, pins=None):
        """Submit existing UVs, authorize native output and verify actual storage."""
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise ValueError('Native UV validation exceeded the remaining layout budget.')
        uv = uv_io.read_uvs(low.data)
        if pins is None:
            pins = {i: uv[i].copy() for i, loop in enumerate(low.data.uv_layers.active.data) if loop.pin_uv}
        proposal = {'uvs': uv, 'edges': marked_edges(low.data), 'protected_edges': protected_edges}
        cleanseams_native.finalize(arrays, proposal, reporter, pins=pins, selected_faces=selected,
            repair_options=options, seconds=seconds, repair_defects=repair_defects)
        application = proposal.pop('_native_application')
        if 'initial_final_validation' in proposal:
            application.report['initial_final_validation'] = proposal['initial_final_validation']
        try:
            current = mesh_io.mesh_to_arrays(low.data)
            authorized = application.authorize(cleanseams_native.geometry_fingerprint(current))
            expected = np.empty_like(uv)
            expected[application.host_loop_indices] = authorized['uv']
            edges = authorized['edge_vertices'][np.asarray(authorized['seams'], dtype=bool)]
            apply_cut(low, edges, authorized['exempt_faces'].tolist())
            uv_io.write_uvs(low.data, expected)
            stored = np.ascontiguousarray(uv_io.read_uvs(low.data), dtype=np.float32)
            cleanseams_native.load().check_stored_uv(authorized['uv'], application.host_loop_indices, stored)
            if not np.array_equal(stored, expected.astype(np.float32)):
                raise ValueError('Native UV validation failed: stored values differ from the authorized proposal.')
            if not cuts_preserved(cut_corners(current, protected_edges), stored):
                raise ValueError('Native UV validation failed: stored UVs closed a protected cut.')
            return application
        finally:
            application.release_application()

    def _improve_stretch(self, low, arrays, recipe, texture_size, junk, original, protected_edges=(),
                         *, reporter=None, deadline=None):
        before = uv_io.read_uvs(low.data)
        maximum, minimum = singular_values(arrays, before)
        selected = np.zeros(arrays.num_faces, dtype=bool)
        selected[surface_triangles(arrays)[1][maximum / np.maximum(minimum, 1e-12) - 1 > recipe.max_stretch]] = True
        if not selected.any():
            return original
        reporter = reporter or ProgressReporter(threading.Event())
        deadline = deadline if deadline is not None else time.monotonic() + 10.0
        previous_edges = marked_edges(low.data)
        previous_junk = junk_faces_of(low.data)
        pins = {i: before[i].copy() for i, loop in enumerate(low.data.uv_layers.active.data) if loop.pin_uv}
        if pins:
            selected[np.repeat(np.arange(arrays.num_faces), arrays.face_sizes)[list(pins)]] = False
        if not selected.any():
            return original
        accepted = False
        selection_mode = tuple(bpy.context.tool_settings.mesh_select_mode)
        try:
            uv_io.smart_project(low, faces=selected)
            current = uv_io.read_uvs(low.data)
            damaged = invalid_faces(arrays, current)
            post = self._native_postprocess(low, arrays, reporter, protected_edges, deadline,
                options={'region_recovery_seconds': 1.0}, selected=damaged,
                repair_defects=False, pins=pins)
            current = uv_io.read_uvs(low.data)
            if (_introduces_uv_cuts(arrays, before, current)
                    or not cuts_preserved(cut_corners(arrays, protected_edges), current)):
                return original
            protection = {"protected_edges": protected_edges} if len(protected_edges) else {}
            candidate = self._lay_out(low, arrays, recipe, texture_size, junk=junk,
                chart_ids=post.chart_of_face, reporter=reporter, deadline=deadline, **protection)
            current = uv_io.read_uvs(low.data)
            if (
                candidate["stretch"]["p95"] < original["stretch"]["p95"]
                and candidate["occupancy"] >= original["occupancy"] * 0.95
                and not _introduces_uv_cuts(arrays, before, current)
                and cuts_preserved(cut_corners(arrays, protected_edges), current)
                and not invalid_faces(arrays, current).any()
            ):
                _mark_chart_boundaries(low.data, current)
                candidate["repaired_faces"] = original["repaired_faces"]
                candidate["unresolved_faces"] = original.get("unresolved_faces", 0)
                candidate["stretch_reprojected_faces"] = int(selected.sum())
                accepted = True
                return candidate
            return original
        finally:
            bpy.context.tool_settings.mesh_select_mode = selection_mode
            if not accepted:
                uv_io.write_uvs(low.data, before)
                apply_cut(low, previous_edges, [] if previous_junk is None else np.flatnonzero(previous_junk).tolist())

    def _lay_out(
        self, low: bpy.types.Object, arrays: MeshArrays, recipe: UVRecipe,
        texture_size: int, junk: np.ndarray | None = None,
        recovery_faces: np.ndarray | None = None, average_scale: bool = True,
        protected_edges=(), *, chart_ids=None, reporter=None, deadline=None,
    ) -> dict[str, Any]:
        """Pack under the selected engine's contract, then measure stored UVs."""
        notes: list[str] = []
        if recovery_faces is not None:
            raise ValueError('Selected-face recovery must finish before packing.')
        measurements = {}
        uv_before_pack = uv_io.read_uvs(low.data)
        if not cuts_preserved(cut_corners(arrays, protected_edges), uv_before_pack):
            raise ValueError('Protected cuts must be repaired in CleanSeams before packing.')
        packer = uv_io.pack_islands(low, padding_px=recipe.padding_px, texture_size=texture_size,
            rotate_method=recipe.rotate_method, shape_method=recipe.shape_method,
            allow_scaling=not recipe.preserve_texel_density, packer=recipe.packer,
            search_seconds=recipe.pack_seconds, measurements=measurements,
            arrays=arrays, chart_ids=chart_ids,
            density_target=recipe.texel_density if recipe.preserve_texel_density and recipe.texel_density > 0 else None,
            equalize=average_scale, deadline=deadline,
            cancelled=(lambda: reporter.cancelled) if reporter is not None else None)
        uvs = uv_io.read_uvs(low.data)
        from ..bridge.engines.uvpack import uv_fingerprint
        if measurements.get('packing_uv_sha256') != uv_fingerprint(uvs):
            uv_io.write_uvs(low.data, uv_before_pack)
            raise ValueError('Stored packing certificate does not describe the current UVs.')
        if (lost_coverage_faces(arrays, uv_before_pack, uvs).any() or
                not cuts_preserved(cut_corners(arrays, protected_edges), uvs)):
            uv_io.write_uvs(low.data, uv_before_pack)
            raise ValueError('Stored packing validation failed: UV coverage or a protected cut was lost.')
        density = average_texel_density(arrays, uvs, texture_size)
        fitted = fits_in_tile(uvs)

        stretch = stretch_report(arrays, uvs, faces=None if junk is None else ~junk)
        overlaps = (
            measurements["overlaps"]
            if "uvs" in measurements and np.array_equal(uvs, measurements["uvs"])
            else overlap_report(arrays, uvs)
        )
        if not overlaps.is_clean:
            notes.append(
                f"{overlaps.overlapping_pairs} overlapping triangle pair(s), "
                f"{overlaps.flipped_triangles} flipped"
            )
        if junk is not None and junk.any():
            areas = face_areas(arrays)
            share = float(areas[junk].sum() / max(areas.sum(), 1e-12))
            notes.append(
                f"{share:.2%} of the surface is excluded from the stretch measurement "
                "because its UVs are degenerate or needed fallback mapping. "
                "Overlap counts still include these faces."
            )

        return {
            "texture_size": texture_size,
            "packer": packer,
            "padding_px": recipe.padding_px,
            "padding_uv": recipe.padding_px / max(texture_size, 1),
            "texel_density": density,
            "texel_density_target": recipe.texel_density,
            "preserve_texel_density": recipe.preserve_texel_density,
            "fits_in_tile": fitted,
            "occupancy": uv_occupancy(arrays, uvs),
            "stretch": stretch.to_dict(),
            "overlaps": overlaps.to_dict(),
            "faces": int(arrays.num_faces),
            "notes": notes,
            **measurements,
        }


def _mark_chart_boundaries(mesh, uvs, *, replace=False):
    if replace:
        for edge in mesh.edges:
            edge.use_seam = False
    seen = {}
    for polygon in mesh.polygons:
        loops = list(polygon.loop_indices)
        for index, corner in enumerate(loops):
            following = loops[(index + 1) % len(loops)]
            first, second = mesh.loops[corner], mesh.loops[following]
            pair = uvs[[corner, following]]
            if first.vertex_index > second.vertex_index:
                pair = pair[::-1]
            previous = seen.setdefault(first.edge_index, pair)
            if not np.array_equal(previous, pair):
                mesh.edges[first.edge_index].use_seam = True


def _seam_count(mesh: bpy.types.Mesh) -> int:
    if len(mesh.edges) == 0:
        return 0
    flags = np.empty(len(mesh.edges), dtype=bool)
    mesh.edges.foreach_get("use_seam", flags)
    return int(flags.sum())
