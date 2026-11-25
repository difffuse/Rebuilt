"""Take in the scan and prepare a working copy."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ..bridge import mesh_io
from ..bridge.materials import MATERIAL_PROTOCOL_REVISION, normalize_scan_materials
from ..core.analysis.defects import assess_scale
from ..core.jobs import ProgressReporter
from ..core.recipe import AssetRecipe
from ..pipeline.context import AssetContext
from .base import Stage

ROLE_WORKING = "working"
ROLE_PROXY = "proxy"


WELD_TOLERANCE = 1e-6


def preflight(source, working) -> dict:
    """Prepare the working copy, and say what the scan will not let us fix."""
    import bmesh

    from ..bridge.materials import find_colour_source

    notes: list[str] = []
    welded = 0

    diagonal = max(float(np.linalg.norm(np.array(source.dimensions))), 1e-9)
    mesh = bmesh.new()
    mesh.from_mesh(working.data)
    try:
        before = len(mesh.verts)
        bmesh.ops.remove_doubles(mesh, verts=mesh.verts[:], dist=diagonal * WELD_TOLERANCE)
        welded = before - len(mesh.verts)
        if welded:
            mesh.to_mesh(working.data)
            working.data.update()
    finally:
        mesh.free()

    layers = working.data.uv_layers
    if len(layers):
        for layer in layers:
            if layer.active_render:
                layers.active = layer
                break

    material_protocol = normalize_scan_materials(working)
    notes.extend(
        item["reason"] for item in material_protocol if item.get("status") == "unsupported"
    )

    if not len(layers):
        notes.append(
            "The source model has no UV layer, so its texture cannot be read. "
            "The colour maps will bake flat."
        )
    elif len(layers) > 1:
        notes.append(
            f"The source model carries {len(layers)} UV layers; "
            f"{layers.active.name!r} is the one being read."
        )

    materials = [slot for slot in working.data.materials if slot is not None]
    if not materials:
        notes.append("The source model has no material, so there is no colour to transfer.")
    elif not any(find_colour_source(material) is not None for material in materials):
        notes.append(
            "Nothing in the source material looks like a colour, so the colour "
            "maps will bake flat. A texture plugged into a Principled, an "
            "Emission or a Diffuse shader is what this reads."
        )

    scale = tuple(round(value, 6) for value in source.scale)
    return {
        "welded_vertices": welded,
        "material_protocol": material_protocol,
        "uv_layers": len(layers),
        "active_uv_layer": layers.active.name if len(layers) else "",
        "materials": len(materials),
        "source_scale": scale,
        "transform_applied": scale != (1.0, 1.0, 1.0),
        "notes": notes,
    }


def _faithful_copy(source, name: str, apply_transform: bool):
    """Duplicate an object with everything on it, transform baked into the mesh."""
    import bpy
    from mathutils import Matrix

    bpy.context.view_layer.update()

    copy = source.copy()
    copy.data = source.data.copy()
    copy.name = name
    copy.data.name = name
    copy.modifiers.clear()

    if apply_transform:
        matrix = source.matrix_world.copy()
        copy.data.transform(matrix)
        if matrix.determinant() < 0.0:
            copy.data.flip_normals()
        copy.matrix_world = Matrix.Identity(4)

    copy.data.update()
    return copy


class IngestStage(Stage):
    name: ClassVar[str] = "ingest"
    slot: ClassVar[str] = "working"
    role: ClassVar[str] = ROLE_WORKING
    label: ClassVar[str] = "Preparing model"

    def cache_key(self, recipe: AssetRecipe) -> str:
        return recipe.ingest.fingerprint() + f":material-protocol-{MATERIAL_PROTOCOL_REVISION}"

    def gather(self, ctx: AssetContext) -> dict[str, Any]:
        import bpy

        source = ctx.source_object
        recipe = ctx.recipe.ingest

        bpy.context.view_layer.update()

        arrays = mesh_io.object_to_arrays(source, world_space=recipe.apply_transforms)
        return {
            "arrays": arrays,
            "source_name": source.name,
            "recipe": recipe,
        }

    def compute(self, payload: dict[str, Any], report: ProgressReporter) -> dict[str, Any]:
        arrays = payload["arrays"]
        recipe = payload["recipe"]

        report.update(0.3, "Measuring model")
        bbox_min = arrays.positions.min(axis=0)
        bbox_max = arrays.positions.max(axis=0)
        diagonal = float(np.linalg.norm(bbox_max - bbox_min))
        triangles = int(np.maximum(arrays.face_sizes - 2, 0).sum())

        report.update(0.8, "Planning")
        needs_proxy = triangles > recipe.proxy_triangle_threshold
        proxy_ratio = (
            min(1.0, recipe.proxy_target_triangles / triangles)
            if needs_proxy and triangles
            else 1.0
        )

        report.update(1.0, "Ready")
        return {
            "arrays": arrays,
            "triangle_count": triangles,
            "vertex_count": arrays.num_verts,
            "bbox_diagonal": diagonal,
            "scale_warning": assess_scale(diagonal),
            "needs_proxy": needs_proxy,
            "proxy_ratio": proxy_ratio,
        }

    def apply(self, ctx: AssetContext, result: dict[str, Any]) -> None:
        ctx.clear_artifacts(self.role)
        ctx.clear_artifacts(ROLE_PROXY)

        source = ctx.source_object
        working = _faithful_copy(
            source,
            f"{ctx.name}_working",
            apply_transform=ctx.recipe.ingest.apply_transforms,
        )
        ctx.register_artifact(working, self.role, self.slot)

        ctx.progress(0.5, "Checking the source model")
        checks = preflight(source, working)

        if result["needs_proxy"]:
            self._build_proxy(ctx, working, result["proxy_ratio"])

        ctx.set_report(
            self.name,
            {
                "triangle_count": result["triangle_count"],
                "vertex_count": result["vertex_count"],
                "bbox_diagonal": result["bbox_diagonal"],
                "scale_warning": result["scale_warning"],
                "proxy": result["needs_proxy"],
                "preflight": checks,
                "notes": checks["notes"],
            },
        )

    def _build_proxy(self, ctx: AssetContext, working, ratio: float) -> None:
        """Decimate a copy down to something analysis can chew through."""
        import bpy

        proxy = working.copy()
        proxy.data = working.data.copy()
        proxy.name = f"{ctx.name}_proxy"

        modifier = proxy.modifiers.new("rebuilt_proxy", "DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = ratio

        ctx.register_artifact(proxy, ROLE_PROXY, "analysis")

        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = proxy.evaluated_get(depsgraph)
        baked = bpy.data.meshes.new_from_object(evaluated)

        old = proxy.data
        proxy.data = baked
        proxy.modifiers.clear()
        if old.users == 0:
            bpy.data.meshes.remove(old)
