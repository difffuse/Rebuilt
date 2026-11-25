"""Projecting the scan's detail onto the low mesh."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from typing import Any, ClassVar

import bpy
import numpy as np

from ..bridge import baking, bvh, mesh_io, uv_io
from ..core.bake.metrics import BakeQuality, assess, fault_regions, fault_interior, written_mask
from ..core.bake.raster import TexelMap, interpolate_vertex_field, rasterise
from ..core.bake.raydist import CageField, cage_offsets, cage_positions, envelope
from ..core.analysis.mesh_graph import face_normals, fan_triangles, vertex_normals
from ..core.bake.padding import fill_from_neighbours as _fill_from_neighbours
from ..core.bake.channels import EXTRA_CHANNELS, image_options
from ..core.jobs import ProgressReporter
from ..core import engines
from ..core.recipe import AssetRecipe, BakeRecipe
from ..core.types import MeshArrays
from ..pipeline.context import AssetContext, ContextError
from .base import Stage
from .cleanup import NORMALS_REVISION, bake_source, has_appearance
from .retopo import ROLE_LOW

ROLE_CAGE = "cage"


PROBE_SAMPLES = 1


AO_DISTANCE_FRACTION = 0.02
AO_REVISION = 1
PROJECTION_REVISION = 7


MAP_RECIPES: dict[str, dict[str, Any]] = {
    "NORMAL": {"pass": "NORMAL", "colour": False},
    "AO": {"pass": "AO", "colour": False, "integrates": True},
    "ROUGHNESS": {"pass": "ROUGHNESS", "colour": False},
    "BASE_COLOR": {"pass": "EMIT", "channel": "Base Color", "colour": True},
    "METALLIC": {"pass": "EMIT", "channel": "Metallic", "colour": False},
    "OPACITY": {"pass": "EMIT", "channel": "Alpha", "colour": False},
}
MAP_RECIPES.update(
    {
        name: {"pass": "EMIT", "channel": socket, "colour": color}
        for name, (_, socket, color, _) in EXTRA_CHANNELS.items()
    }
)


class BakeStage(Stage):
    name: ClassVar[str] = "bake"
    slot: ClassVar[str] = "bake"
    role: ClassVar[str] = "bake"
    label: ClassVar[str] = "Baking maps"

    def cache_key(self, recipe: AssetRecipe) -> str:

        return (
            f"{recipe.bake.fingerprint()}:{recipe.uv.fingerprint()}:{recipe.texture_resolution}"
            f":source-normals-{NORMALS_REVISION}:local-ao-{AO_REVISION}"
            f":projection-{PROJECTION_REVISION}:pbr-2:map-references-1:linear-colour-float-1"
        )

    def gather(self, ctx: AssetContext) -> dict[str, Any]:
        low = ctx.artifact(ROLE_LOW)
        if low is None:
            raise ContextError("bake needs retopo to have run first")

        high = bake_source(ctx)
        if high is None:
            raise ContextError("bake needs ingest to have run first")

        from ..bridge.bake_memory import admit_bake
        admit_bake(ctx.recipe.texture_resolution,
                   sum(name in MAP_RECIPES for name in ctx.recipe.bake.maps), low, high)

        if low.mode == "EDIT":
            low.update_from_editmode()
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.context.view_layer.update()

        try:
            uvs = uv_io.read_uvs(low.data)
        except KeyError as exc:
            raise ContextError("bake needs uv to have run first") from exc

        low.data.calc_loop_triangles()
        corners = np.empty(len(low.data.loop_triangles) * 3, dtype=np.int32)
        low.data.loop_triangles.foreach_get("loops", corners)
        return {
            "low": mesh_io.mesh_to_arrays(low.data, low.matrix_world),
            "high": mesh_io.mesh_to_arrays(high.data, high.matrix_world),
            "high_has_appearance": has_appearance(high),
            "uvs": uvs,
            "triangle_corners": corners.reshape(-1, 3),
            "recipe": ctx.recipe.bake,
            "texture_size": ctx.recipe.texture_resolution,
        }

    def compute(self, payload: dict[str, Any], report: ProgressReporter) -> dict[str, Any]:
        """Everything decidable before a single ray is fired."""
        low: MeshArrays = payload["low"]
        high: MeshArrays = payload["high"]
        recipe: BakeRecipe = payload["recipe"]

        report.update(0.1, "Mapping texels")
        texels = rasterise(
            low, payload["uvs"], payload["texture_size"], payload.get("triangle_corners"),
            include_normals=False,
        )

        report.update(0.4, "Measuring clearance")
        normals = vertex_normals(low)
        query = bvh.SurfaceQuery(high)

        diagonal = float(np.linalg.norm(high.positions.max(axis=0) - high.positions.min(axis=0)))
        outward, inward, scale = _measure_clearance(low, normals, query, diagonal)

        report.update(0.8, "Planning cage")
        field = cage_offsets(
            low,
            outward,
            scale,
            diagonal,
            safety=recipe.cage_safety_factor,
        )

        reach = np.maximum(outward, inward)

        report.update(1.0, "Ready")
        return {
            "low": low,
            "texels": texels,
            "triangle_faces": (
                np.searchsorted(low.face_starts, payload["triangle_corners"][:, 0], side="right") - 1
                if "triangle_corners" in payload else fan_triangles(low)[1]
            ),
            "normals": normals,
            "reach": reach,
            "outward": outward,
            "field": field,
            "diagonal": diagonal,
            "recipe": recipe,
            "texture_size": payload["texture_size"],
            "high_has_appearance": payload["high_has_appearance"],
        }

    def apply(self, ctx: AssetContext, result: dict[str, Any]) -> None:
        from ..pipeline.stage_operation import stage_operation
        with stage_operation(ctx, self.name):
            self._apply(ctx, result)

    def _apply(self, ctx: AssetContext, result: dict[str, Any]) -> None:

        low = ctx.artifact(ROLE_LOW)
        high = bake_source(ctx)
        recipe: BakeRecipe = result["recipe"]
        scene = ctx.scene

        arrays: MeshArrays = result["low"]
        texels: TexelMap = result["texels"]
        field: CageField = result["field"]
        offsets = field.offsets.copy()

        ctx.progress(0.05, "Building the cage")
        cage = _build_cage(ctx, low, arrays, result["normals"], offsets)

        placeholder = ctx.resources.create(
            'bake', baking.new_image, f"{ctx.name}_rebuilt_target", result["texture_size"], non_colour=True)
        baking.ensure_bake_material(low, placeholder, f"{ctx.name}_rebuilt_bake", resources=ctx.resources)

        was_hidden = high.hide_get()
        try:
            if was_hidden:
                high.hide_set(False)
            with baking.scene_state(scene):
                baking.prepare(
                    scene,
                    low,
                    high,
                    cage,
                    samples=recipe.samples,
                    margin_px=_dilation_pixels(ctx, result["texture_size"]),
                    normal_space=recipe.normal_space,
                    max_ray_distance=_ray_limit(offsets, result["reach"]),
                )
                ctx.progress(0.15, "Checking where the rays land")
                quality, offsets, attempts, stray = self._bake_within_tolerance(
                    ctx, scene, low, high, cage, result, offsets
                )
                maps = self._bake_maps(
                    ctx, scene, low, high, recipe, result, ctx.progress, faults=stray
                )
        finally:
            result.pop("projection_written", None)
            for plan in result.get("projection_recovery", ()):
                plan.pop("_uv_patch", None)
            _release_target(ctx, low, placeholder)
            high.hide_set(was_hidden)

        from .output import show_baked_maps

        ctx.progress(0.98, "Connecting baked maps")
        preview = show_baked_maps(ctx, maps, clear_preview=False)
        from ..bridge.bake_maps import material_defaults

        ctx.set_report(
            self.name,
            {
                "texture_size": result["texture_size"],
                "maps": maps,
                "preview": preview,
                "material_defaults": material_defaults(high),
                "quality": quality.to_dict(),
                "projection_fallback": result.get("projection_fallback", {}),
                "approximation": {
                    "texels": max((m.get("approximated_texels", 0) for m in maps), default=0),
                    "neutral_texels": max((m.get("neutral_texels", 0) for m in maps), default=0),
                },
                "cage": {**field.to_dict(), "grown": float(offsets.max())},
                "attempts": attempts,
                "coverage": texels.coverage,
                "source_has_appearance": result["high_has_appearance"],
                "source_normals_revision": NORMALS_REVISION,
                "notes": _notes(quality, recipe, result["high_has_appearance"]) + (
                    ["Completed with approximate texture regions. Review the Bake Quality report."]
                    if any(m.get("approximated_texels", 0) for m in maps) else []
                ),
            },
        )

    def _bake_within_tolerance(
        self,
        ctx: AssetContext,
        scene: bpy.types.Scene,
        low: bpy.types.Object,
        high: bpy.types.Object,
        cage: bpy.types.Object,
        result: dict[str, Any],
        offsets: np.ndarray,
    ) -> tuple[BakeQuality, np.ndarray, list[dict[str, Any]], np.ndarray]:
        """Bake a POSITION pass, judge it, and grow the cage where it went wrong."""
        arrays: MeshArrays = result["low"]
        texels: TexelMap = result["texels"]
        recipe: BakeRecipe = result["recipe"]
        size = result["texture_size"]

        probe = ctx.resources.create('bake', baking.new_image, f"{ctx.name}_rebuilt_probe", size, non_colour=True)
        limit = interpolate_vertex_field(arrays, texels, envelope(offsets, result["reach"]))

        attempts: list[dict[str, Any]] = []
        quality = BakeQuality()
        stray = np.zeros(size * size, dtype=bool)
        best = None
        padding = _dilation_pixels(ctx, size)

        try:
            coverage = self._coverage(scene, low, high, probe, texels)

            for attempt in range(recipe.max_retries + 1):
                pixels = baking.run(
                    scene,
                    low,
                    high,
                    baking.BakeRequest(bake_type="POSITION", image=probe, samples=PROBE_SAMPLES),
                )
                written = written_mask(pixels, baking.SENTINEL)

                quality, missed, stray = assess(
                    coverage,
                    written,
                    pixels[:, :3].astype(np.float64),
                    texels.position,
                    limit,
                )
                interior = int(fault_interior(missed | stray, size, padding).sum())
                attempts.append(
                    {
                        "attempt": attempt,
                        "valid_fraction": quality.valid_fraction,
                        "missed": quality.missed_texels,
                        "stray": quality.stray_texels,
                        "cage_max": float(offsets.max()),
                        "interior_fault_texels": interior,
                        "ray_limit": scene.render.bake.max_ray_distance,
                    }
                )
                score = (interior, quality.missed_texels + quality.stray_texels)
                if best is None or score < best[0]:
                    best = (
                        score,
                        quality,
                        offsets.copy(),
                        stray.copy(),
                        scene.render.bake.max_ray_distance,
                        missed.copy(),
                        pixels,
                    )

                if (
                    quality.passes(_minimum_valid(ctx)) and not interior
                ) or attempt == recipe.max_retries:
                    break

                local_limit = float(np.max(envelope(offsets, result["reach"])))
                if attempt == 0 and local_limit > scene.render.bake.max_ray_distance * 1.001:
                    scene.render.bake.max_ray_distance = local_limit
                    continue

                grown = _grow_at_faults(arrays, texels, missed | stray, offsets, result)
                if grown is None:
                    break
                offsets = grown
                _move_cage(cage, arrays, result["normals"], offsets)
                limit = interpolate_vertex_field(arrays, texels, envelope(offsets, result["reach"]))

                scene.render.bake.max_ray_distance = float(
                    np.max(envelope(offsets, result["reach"]))
                )
        finally:
            _release_target(ctx, low, probe)

        _, quality, offsets, stray, ray_limit, missed, pixels = best
        _move_cage(cage, arrays, result["normals"], offsets)
        scene.render.bake.max_ray_distance = ray_limit
        if best[0][0]:
            recovered = self._recover_projection(
                ctx, scene, low, high, cage, result, offsets, coverage,
                pixels, missed | stray, attempts,
            )
            quality, faults = recovered
        else:
            faults = missed | stray
        written = written_mask(pixels, baking.SENTINEL) & np.isfinite(pixels).all(axis=1)
        for plan in result.get("projection_recovery", ()):
            written[plan["texels"]] = True
        result["projection_written"] = written
        interior = int(fault_interior(faults, size, padding).sum())
        if interior or not quality.passes(_minimum_valid(ctx)):
            result["projection_fallback"] = {
                "interior_texels": interior,
                "unreliable_texels": int(faults.sum()),
                "valid_fraction": quality.valid_fraction,
            }
        return quality, offsets, attempts, faults

    def _recover_projection(
        self, ctx, scene, low, high, cage, result, offsets, coverage, pixels, faults, attempts
    ):
        """Recover only failed texels with independently validated projection rays."""
        arrays, texels = result["low"], result["texels"]
        size = result["texture_size"]
        limit = interpolate_vertex_field(arrays, texels, envelope(offsets, result["reach"]))
        merged = pixels.copy()
        remaining = faults.copy()
        plans = []
        probe = ctx.resources.create('bake', baking.new_image, f"{ctx.name}_rebuilt_recovery", size, non_colour=True)
        try:
            for axis, sign in ((2, 1), (2, -1), (0, 1), (0, -1), (1, 1), (1, -1)):
                ctx.progress(0.45, "Recovering projection in concave regions")
                direction = np.zeros(3)
                direction[axis] = sign
                _move_cage(cage, arrays, np.broadcast_to(direction, arrays.positions.shape), offsets)
                patch_texels = np.flatnonzero(remaining)
                with _projection_patch(low, result, patch_texels):
                    candidate = baking.run(
                        scene, low, high,
                        baking.BakeRequest(bake_type="POSITION", image=probe, samples=PROBE_SAMPLES),
                    )
                written = written_mask(candidate, baking.SENTINEL)
                _, missed, stray = assess(coverage, written, candidate[:, :3], texels.position, limit)
                accepted = np.flatnonzero(remaining & ~missed & ~stray)
                if accepted.size:
                    merged[accepted] = candidate[accepted]
                    remaining[accepted] = False
                    plans.append({"direction": direction, "texels": accepted, "patch_texels": patch_texels})
                interior = int(fault_interior(remaining, size, _dilation_pixels(ctx, size)).sum())
                attempts.append({
                    "projection": "local_directional_recovery",
                    "direction": direction.tolist(),
                    "recovered_texels": int(accepted.size),
                    "remaining_fault_texels": int(remaining.sum()),
                    "interior_fault_texels": interior,
                })
                if not interior:
                    break
            quality, missed, stray = assess(
                coverage, written_mask(merged, baking.SENTINEL),
                merged[:, :3], texels.position, limit,
            )
            result["projection_recovery"] = plans
            result["projection_offsets"] = offsets.copy()
            return quality, missed | stray
        finally:
            _move_cage(cage, arrays, result["normals"], offsets)
            _release_target(ctx, low, probe)

    def _coverage(
        self,
        scene: bpy.types.Scene,
        low: bpy.types.Object,
        high: bpy.types.Object,
        probe: bpy.types.Image,
        texels: TexelMap,
    ) -> np.ndarray:
        """Which texels the baker considers inside an island."""
        pixels = baking.run(
            scene,
            low,
            high,
            baking.BakeRequest(
                bake_type="EMIT", image=probe, self_bake=True, samples=PROBE_SAMPLES
            ),
        )
        blender = written_mask(pixels, baking.SENTINEL)
        return blender & texels.covered

    def _bake_maps(
        self,
        ctx: AssetContext,
        scene: bpy.types.Scene,
        low: bpy.types.Object,
        high: bpy.types.Object,
        recipe: BakeRecipe,
        result: dict[str, Any],
        progress=lambda fraction, message="": None,
        faults: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        baked: list[dict[str, Any]] = []

        for index, name in enumerate(recipe.maps):
            progress(
                0.6 + 0.35 * index / max(len(recipe.maps), 1),
                f"Baking {name.replace('_', ' ').lower()} ({index + 1} of {len(recipe.maps)})",
            )
            spec = MAP_RECIPES.get(name)
            if spec is None:
                baked.append({"map": name, "baked": False, "reason": "not supported"})
                continue

            image_name = engines.map_filename(
                engines.get("generic"),
                ctx.name,
                name,
                (result["texture_size"], result["texture_size"]),
            )[:-4]

            image = ctx.resources.create(
                'bake', baking.new_image, image_name, result["texture_size"], not spec["colour"])
            image["rebuilt_asset"] = ctx.name
            channel = spec.get("channel")
            samples = recipe.ao_samples if spec.get("integrates") else recipe.samples
            projection_written = result.get("projection_written")
            constant = (
                baking.constant_scalar_channel(high, channel)
                if name in {"METALLIC", "SPECULAR"} and projection_written is not None
                else None
            )

            if constant is not None:
                pixels = np.tile(baking.SENTINEL, (len(projection_written), 1))
                pixels[projection_written] = (constant, constant, constant, 1.0)
                image.pixels.foreach_set(pixels.ravel())
            elif channel:
                with baking.channel_substitution(high, channel):
                    pixels = _run_projection_passes(
                        scene,
                        low,
                        high,
                        baking.BakeRequest(bake_type=spec["pass"], image=image, samples=samples),
                        result,
                    )
            else:
                pixels = _run_projection_passes(
                    scene,
                    low,
                    high,
                    baking.BakeRequest(
                        bake_type=spec["pass"],
                        image=image,
                        pass_filter=frozenset(spec.get("filter", ())),
                        samples=samples,
                        ao_distance=(
                            result["diagonal"] * AO_DISTANCE_FRACTION if name == "AO" else None
                        ),
                    ),
                    result,
                )

            written = written_mask(pixels, baking.SENTINEL) & np.isfinite(pixels).all(axis=1)

            if faults is not None and faults.shape == written.shape:
                written = written & ~faults

            measured_texels = int(written.sum())
            approximation = {}
            if result.get("projection_fallback") or result.pop("projection_replay_missed", False):
                pixels, repaired, approximation = _repair_map_projection(low, result, pixels, written, name)
                written |= repaired
                image.pixels.foreach_set(pixels.astype(np.float32).ravel())
            _neutralise(
                image, pixels, written, spec["pass"], _dilation_pixels(ctx, result["texture_size"])
            )

            baked.append(
                {
                    "map": name,
                    "baked": True,
                    "image": image.name,
                    "image_ref": ctx.resources.key(image),
                    "colour_space": image.colorspace_settings.name,
                    "written_texels": measured_texels,
                    **approximation,
                    "via": "constant" if constant is not None else "channel" if channel else "pass",
                    "image_settings": image_options(
                        recipe,
                        channel=name,
                        hdr=bool(
                            written.any()
                            and (
                                pixels[written, :3].max() > 1.0001
                                or pixels[written, :3].min() < -0.0001
                            )
                        ),
                    ),
                    **(
                        {
                            "distance": result["diagonal"] * AO_DISTANCE_FRACTION,
                            "distance_fraction": AO_DISTANCE_FRACTION,
                            "ao_revision": AO_REVISION,
                        }
                        if name == "AO"
                        else {}
                    ),
                }
            )

        return baked


def _repair_map_projection(low, result, pixels, written, name):
    from ..core.bake.padding import repair_projection
    from ..core.uv.metrics import island_labels_from_uvs

    if "projection_islands" not in result:
        faces = island_labels_from_uvs(result["low"], uv_io.read_uvs(low.data))
        texels = result["texels"]
        labels = np.full(texels.covered.shape, -1, dtype=np.int32)
        labels[texels.covered] = faces[result["triangle_faces"][texels.triangle[texels.covered]]]
        result["projection_islands"] = labels
    neutral = {"NORMAL": (0.5, 0.5, 1, 1), "AO": (1, 1, 1, 1),
               "BASE_COLOR": (0.5, 0.5, 0.5, 1), "ROUGHNESS": (0.5, 0.5, 0.5, 1),
               "OPACITY": (1, 1, 1, 1)}.get(name, (0, 0, 0, 1))
    return repair_projection(pixels, written, result["projection_islands"],
                             result["texture_size"], neutral, normal=name == "NORMAL")


def _run_projection_passes(scene, low, high, request, result):
    pixels = baking.run(scene, low, high, request)
    plans = result.get("projection_recovery", ())
    if not plans:
        return pixels
    arrays = result["low"]
    offsets = result["projection_offsets"]
    cage = scene.render.bake.cage_object
    try:
        for plan in plans:
            _move_cage(cage, arrays, np.broadcast_to(plan["direction"], arrays.positions.shape), offsets)
            selected = plan["texels"]
            with _projection_patch(low, result, plan.get("patch_texels", selected),
                                   enabled=request.bake_type != "NORMAL", cache=plan):
                candidate = baking.run(scene, low, high, request)
            valid = written_mask(candidate[selected], baking.SENTINEL)
            if not valid.all():
                result["projection_replay_missed"] = True
            pixels[selected[valid]] = candidate[selected[valid]]
            pixels[selected[~valid]] = baking.SENTINEL
        return pixels
    finally:
        _move_cage(cage, arrays, result["normals"], offsets)


@contextmanager
def _projection_patch(low, result, selected, *, enabled=True, cache=None):
    """Restrict extra rays to affected faces; retain the full tangent basis for normals."""
    if not enabled or "triangle_faces" not in result:
        yield
        return
    prepared = cache.get("_uv_patch") if cache is not None else None
    if prepared is None:
        original = uv_io.read_uvs(low.data)
        triangles = result["texels"].triangle[selected]
        faces = np.zeros(result["low"].num_faces, dtype=bool)
        faces[result["triangle_faces"][triangles[triangles >= 0]]] = True
        changed = original.copy()
        changed[~np.repeat(faces, result["low"].face_sizes)] += 2.0 + float(np.abs(original).max())
        original.flags.writeable = changed.flags.writeable = False
        if cache is not None:
            cache["_uv_patch"] = original, changed
    else:
        original, changed = prepared
    try:
        uv_io.write_uvs(low.data, changed)
        yield
    finally:
        uv_io.write_uvs(low.data, original)


def _measure_clearance(
    low: MeshArrays,
    normals: np.ndarray,
    query: bvh.SurfaceQuery,
    diagonal: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """How far the scan sits from the low mesh, per vertex, in each direction."""
    zero = np.zeros(low.num_verts, dtype=np.float64)
    if low.num_verts == 0 or low.num_faces == 0:
        return zero, zero, zero

    samples = bvh.sample_surface(low)
    sample_normals = np.concatenate([normals, face_normals(low)])

    nearest = query.nearest_distances(samples)
    nearest = np.where(np.isfinite(nearest), nearest, 0.0)

    caps = np.maximum(nearest * 4.0, diagonal * 0.01)
    out_samples = query.outward_reach(samples, sample_normals, caps)
    in_samples = query.outward_reach(samples, -sample_normals, caps)

    face_of_corner = np.repeat(
        np.arange(low.num_faces, dtype=np.int64), low.face_sizes.astype(np.int64)
    )
    corner_verts = low.corner_verts.astype(np.int64)

    def fold(values: np.ndarray) -> np.ndarray:
        """Give each vertex the largest value found on it or on its faces."""
        result = values[: low.num_verts].copy()
        np.maximum.at(result, corner_verts, values[low.num_verts :][face_of_corner])
        return result

    return fold(out_samples), fold(in_samples), fold(caps)


def _build_cage(
    ctx: AssetContext,
    low: bpy.types.Object,
    arrays: MeshArrays,
    normals: np.ndarray,
    offsets: np.ndarray,
) -> bpy.types.Object:
    """A copy of the low mesh, displaced by the distance field."""
    cage = low.copy()
    cage.data = low.data.copy()
    cage.data.materials.clear()
    # A cage is geometry only; copied delivery references would retain old maps.
    from ..pipeline.map_images import KEY_MAP_IMAGES
    for key in ('rebuilt_generated_material', KEY_MAP_IMAGES):
        if key in cage:
            del cage[key]
    name = f"{ctx.name}_cage"
    # Blender's bake operator still has a 64-byte cage-name field even when
    # object names allow 255 bytes. Reserve room for Blender's numeric suffix.
    if len(name.encode('utf-8')) > 55:
        name = f"Rebuilt_{hashlib.sha256(ctx.name.encode('utf-8')).hexdigest()[:24]}_cage"
    cage.name = name
    cage.data.name = cage.name
    from mathutils import Matrix

    cage.matrix_world = Matrix.Identity(4)
    cage.hide_render = True
    ctx.register_artifact(cage, ROLE_CAGE, "bake")

    _move_cage(cage, arrays, normals, offsets)
    return cage


def _move_cage(
    cage: bpy.types.Object,
    arrays: MeshArrays,
    normals: np.ndarray,
    offsets: np.ndarray,
) -> None:
    positions = cage_positions(arrays, normals, offsets)
    cage.data.vertices.foreach_set("co", positions.astype(np.float32).ravel())
    cage.data.update()


def _grow_at_faults(
    arrays: MeshArrays,
    texels: TexelMap,
    faults: np.ndarray,
    offsets: np.ndarray,
    result: dict[str, Any],
) -> np.ndarray | None:
    """Push the cage out at the vertices the faulty texels belong to."""
    regions = fault_regions(faults, texels.size)
    if regions.count == 0:
        return None

    meaningful = regions.sizes >= 4
    if not meaningful.any():
        return None

    keep = np.isin(regions.labels, np.flatnonzero(meaningful))
    triangles = texels.triangles
    if triangles is None:
        triangles, _ = fan_triangles(arrays)
    involved = triangles[texels.triangle[keep & (texels.triangle >= 0)]]

    vertex_faults = np.zeros(arrays.num_verts, dtype=bool)
    vertex_faults[involved.ravel()] = True
    if not vertex_faults.any():
        return None

    ceiling = np.maximum(result["outward"] * 3.0, result["diagonal"] * 0.05)
    grown = offsets.copy()
    grown[vertex_faults] = np.minimum(offsets[vertex_faults] * 2.0, ceiling[vertex_faults])
    return grown if not np.array_equal(grown, offsets) else None


_NEUTRAL: dict[str, tuple[float, float, float, float]] = {
    "NORMAL": (0.5, 0.5, 1.0, 1.0),
}


def _neutralise(
    image: bpy.types.Image,
    pixels: np.ndarray,
    written: np.ndarray,
    bake_type: str,
    padding: int = 8,
) -> None:
    """Take the sentinel back out of the map, in every form it survives in."""

    missing = ~written
    if not missing.any():
        return

    filled = pixels.copy()
    neutral = _NEUTRAL.get(bake_type, (0.0, 0.0, 0.0, 1.0))
    filled[missing] = neutral

    size = int(round(float(np.sqrt(pixels.shape[0]))))
    if size * size == pixels.shape[0] and bake_type not in _NEUTRAL:
        filled = _fill_from_neighbours(filled, missing, size, passes=max(padding, 1))

        still = missing & np.all(np.abs(filled[:, :3] - np.asarray(neutral[:3])) < 1e-6, axis=1)
        painted = ~still
        if painted.any() and still.any():
            filled[still, :3] = filled[painted, :3].mean(axis=0)

    image.pixels.foreach_set(filled.astype(np.float32, copy=False).ravel())


def _ray_limit(offsets: np.ndarray, reach: np.ndarray) -> float:
    """How far a ray may travel before it is treated as having found nothing."""
    if offsets.size == 0:
        return 0.0

    allowance = float(np.percentile(reach, 95)) if reach.size else 0.0
    return float((float(np.percentile(offsets, 95)) + max(allowance, 0.0)) * 1.5)


def _dilation_pixels(ctx: AssetContext, texture_size: int) -> int:
    """Bleed the maps outward by the same padding the packer left between islands."""
    return max(int(ctx.recipe.uv.padding_px), 0)


def _minimum_valid(ctx: AssetContext) -> float:
    return float(ctx.recipe.qa.min_valid_bake_pixels)


def _notes(
    quality: BakeQuality, recipe: BakeRecipe, source_has_appearance: bool = True
) -> list[str]:
    notes: list[str] = []
    if not source_has_appearance:
        notes.append(
            "The source model has no texture or no UV layer, so the colour passes have "
            "nothing to read and come back flat."
        )
    if quality.missed_texels:
        notes.append(
            f"{quality.missed_texels} texel(s) had no surface to bake from; "
            "the cage may need to reach further, or the source model has a hole there."
        )
    if quality.stray_texels:
        notes.append(
            f"{quality.stray_texels} texel(s) baked from unrelated geometry. "
            "Two surfaces are close enough that the cage crossed between them."
        )
    return notes


def _release_target(ctx, low, image):
    # The active target belongs to this attempt. Detach only its exact image;
    # unrelated users keep their reference and protect it from retirement.
    for material in low.data.materials:
        if material is not None and material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image == image:
                    node.image = None
    ctx.resources.release(ids={image})
