"""Accept artist-authored cuts or UVs without running their generators."""

import bpy
import numpy as np

from ..bridge import mesh_io, uv_io
from ..core.graph import StageGraph
from ..core.uv.metrics import island_count_from_uvs, overlap_report, stretch_report
from ..core.uv.texel import average_texel_density, signed_uv_triangle_areas
from .context import ContextError


def accept_existing(ctx, recipe, stage):
    """Validate before changing pipeline state; keep coordinates and seam flags."""
    from .. import stages
    from ..stages.seams import JUNK_ATTRIBUTE, LAYOUT_KEY

    if stage not in {"seams", "uv"}:
        raise ContextError("Only seams and UVs can be accepted here.")
    low = ctx.artifact("low")
    if low is None:
        raise ContextError("Run Retopologise first, then edit the generated mesh.")
    current = stages.cache_keys(recipe, ctx=ctx)
    for name in ("ingest", "cleanup", "analyze", "retopo"):
        if ctx.cached_keys.get(name) != current[name]:
            raise ContextError("Restore the settings used for Retopologise, or run it again first.")

    # update_from_editmode alone does not populate Blender's bulk UV reader.
    if low.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    seam_count = sum(edge.use_seam for edge in low.data.edges)
    notes = []
    report = {"method": "external", "seam_count": seam_count, "notes": notes}
    if stage == "seams":
        if not seam_count:
            raise ContextError("Mark seams on the generated mesh before using existing seams.")
    else:
        try:
            uvs = uv_io.read_uvs(low.data)
        except KeyError as exc:
            raise ContextError("Create an active UV map on the generated mesh first.") from exc
        if not len(uvs) or not np.isfinite(uvs).all():
            raise ContextError("The active UV map is empty or contains invalid coordinates.")
        if (uvs < -1e-6).any() or (uvs > 1 + 1e-6).any():
            raise ContextError(
                "Pack UVs inside the 0–1 tile first. Multi-tile baking is not supported."
            )
        arrays = mesh_io.mesh_to_arrays(low.data)
        areas = signed_uv_triangle_areas(arrays, uvs)
        if not np.any(np.abs(areas) > 1e-12):
            raise ContextError("The active UV map has no usable surface area. Unwrap it first.")
        collapsed = int((np.abs(areas) <= 1e-12).sum())
        overlaps = overlap_report(arrays, uvs)
        stretch = stretch_report(arrays, uvs)
        if collapsed:
            notes.append(f"{collapsed} collapsed UV triangles may miss texture detail.")
        if overlaps.overlapping_pairs:
            notes.append(
                f"{overlaps.overlapping_pairs} overlapping triangle pairs may receive conflicting bake colors."
            )
        report.update(
            {
                "uv_layer": low.data.uv_layers.active.name,
                "islands": island_count_from_uvs(arrays, uvs),
                "stretch": stretch.to_dict(),
                "stretch_met": stretch.p95 <= recipe.uv.max_stretch,
                "overlaps": overlaps.to_dict(),
                "faces": int(arrays.num_faces),
                "occupancy": min(1.0, float(np.abs(areas).sum())),
                "fits_in_tile": True,
                "texel_density": average_texel_density(arrays, uvs, recipe.texture_resolution),
                "texture_size": recipe.texture_resolution,
                "padding_px": None,
                "packer": "External",
            }
        )

    if stage == "uv":
        low.data.uv_layers.active.active_render = True
    if LAYOUT_KEY in low:
        del low[LAYOUT_KEY]
    junk = low.data.attributes.get(JUNK_ATTRIBUTE)
    if junk is not None:
        low.data.attributes.remove(junk)
    ctx.recipe = recipe
    if stage == "uv":
        ctx.set_report(
            "seams", {"method": "external", "seam_count": seam_count, "provided_with_uvs": True}
        )
        ctx.set_cached_key("seams", current["seams"])
    ctx.set_report(stage, report)
    ctx.set_cached_key(stage, current[stage])
    ctx.invalidate_cache(StageGraph().downstream(stage))
    return notes
