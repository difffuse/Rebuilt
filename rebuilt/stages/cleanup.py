"""Find what is wrong with the scan, and repair only what was allowed."""

from __future__ import annotations

from typing import Any, ClassVar

from ..bridge import mesh_io
from ..core.analysis.defects import DefectReport, scan_defects
from ..core.analysis.mesh_graph import components_of
from ..core.jobs import ProgressReporter
from ..core.mesh_edit import (
    keep_components,
    merge_by_distance,
    remove_faces,
)
from ..core.recipe import AssetRecipe, CleanupRecipe
from ..core.types import MeshArrays
from ..pipeline.context import AssetContext, ContextError
from .base import Stage
from .ingest import ROLE_WORKING

ROLE_CLEANED = "cleaned"


def repair(
    mesh: MeshArrays, recipe: CleanupRecipe, report: ProgressReporter | None = None
) -> tuple[MeshArrays, list[str]]:
    """Apply the repairs the recipe permits. Returns the mesh and what was done."""

    def step(fraction: float, message: str) -> None:
        if report is not None:
            report.update(fraction, message)

    actions: list[str] = []
    result = mesh

    if recipe.remove_floaters:
        step(0.4, "Removing loose fragments")
        face_labels, components = components_of(result)
        if len(components) > 1:
            main = max(components, key=lambda c: c.area)
            largest_diagonal = max(c.bbox_diagonal for c in components)
            keep = {
                c.index
                for c in components
                if c.index == main.index
                or (
                    largest_diagonal > 0
                    and c.bbox_diagonal / largest_diagonal >= recipe.floater_relative_size
                )
            }
            if len(keep) < len(components):
                actions.append(f"removed {len(components) - len(keep)} loose fragment(s)")
                result = keep_components(result, face_labels, keep)

    if recipe.merge_doubles:
        step(0.6, "Welding coincident vertices")
        result, merged = merge_by_distance(result, recipe.merge_distance)
        if merged:
            actions.append(f"welded {merged} coincident vertex/vertices")

    if recipe.remove_degenerate:
        step(0.8, "Removing zero-area faces")
        from ..core.analysis.mesh_graph import face_areas

        areas = face_areas(result)
        if areas.size:
            threshold = float(areas.mean()) * recipe.degenerate_area_ratio
            mask = areas <= threshold
            if mask.any():
                actions.append(f"removed {int(mask.sum())} zero-area face(s)")
                result = remove_faces(result, mask)

    return result, actions


class CleanupStage(Stage):
    name: ClassVar[str] = "cleanup"
    slot: ClassVar[str] = "working"
    role: ClassVar[str] = ROLE_CLEANED
    label: ClassVar[str] = "Inspecting model"

    def cache_key(self, recipe: AssetRecipe) -> str:
        return recipe.cleanup.fingerprint()

    def gather(self, ctx: AssetContext) -> dict[str, Any]:
        working = ctx.artifact(ROLE_WORKING)
        if working is None:
            raise ContextError("cleanup needs ingest to have run first")
        return {
            "arrays": mesh_io.mesh_to_arrays(working.data),
            "recipe": ctx.recipe.cleanup,
        }

    def compute(self, payload: dict[str, Any], report: ProgressReporter) -> dict[str, Any]:
        recipe: CleanupRecipe = payload["recipe"]

        report.update(0.05, "Inspecting")
        findings = scan_defects(
            payload["arrays"],
            floater_relative_size=recipe.floater_relative_size,
            merge_distance=recipe.merge_distance,
            degenerate_area_ratio=recipe.degenerate_area_ratio,
        )

        repaired, actions = repair(payload["arrays"], recipe, report)

        report.update(1.0, "Done")
        return {
            "report": findings,
            "arrays": repaired if actions else None,
            "actions": actions,
        }

    def apply(self, ctx: AssetContext, result: dict[str, Any]) -> None:
        findings: DefectReport = result["report"]
        payload = findings.to_dict()
        payload["actions"] = result["actions"]
        ctx.set_report(self.name, payload)

        ctx.clear_artifacts(self.role)
        if result["arrays"] is None:
            return

        cleaned = mesh_io.arrays_to_object(result["arrays"], f"{ctx.name}_cleaned")
        ctx.register_artifact(cleaned, self.role, self.slot)


def current_mesh_object(ctx: AssetContext):
    """The most-processed version of the scan available so far."""
    return ctx.artifact(ROLE_CLEANED) or ctx.artifact(ROLE_WORKING)


def has_appearance(obj) -> bool:
    """Whether an object still carries the scan's texture and its mapping."""
    return bool(obj) and bool(obj.data.uv_layers) and any(obj.data.materials)


def bake_source(ctx: AssetContext):
    """The scan copy the baker should project *from*."""
    cleaned = ctx.artifact(ROLE_CLEANED)
    if has_appearance(cleaned):
        return _normals_outward(cleaned, ctx.source_object)
    working = ctx.artifact(ROLE_WORKING)
    if has_appearance(working):
        return _normals_outward(working, ctx.source_object)
    return _normals_outward(cleaned or working, ctx.source_object)


NORMALS_REVISION = 2


def _normals_outward(obj, reference=None, *, force=False):
    """Repair winding while preserving the photographed side of open scans."""
    if obj is None or (not force and obj.get("rebuilt_normals_outward") == NORMALS_REVISION):
        return obj
    import bmesh

    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        _orient_component_normals(bm, obj, reference)
        bm.to_mesh(obj.data)
        obj.data.update()
    finally:
        bm.free()
    obj["rebuilt_normals_outward"] = NORMALS_REVISION
    return obj


def _orient_component_normals(bm, obj, reference=None):
    """Orient closed components outward and preserve each open component's side."""
    import bmesh
    from mathutils.bvhtree import BVHTree

    bm.normal_update()
    previous = {face: face.normal.copy() for face in bm.faces}
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    remaining = set(bm.faces)
    reference_tree = None
    while remaining:
        first = remaining.pop()
        component, pending, opened = [], [first], False
        while pending:
            face = pending.pop()
            component.append(face)
            for edge in face.edges:
                if len(edge.link_faces) != 2:
                    opened = True
                for neighbor in edge.link_faces:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        pending.append(neighbor)
        if not opened:
            continue
        if reference is not None and reference_tree is None:
            transform = obj.matrix_world.inverted_safe() @ reference.matrix_world
            mirrored = transform.determinant() < 0
            reference_tree = BVHTree.FromPolygons(
                [transform @ v.co for v in reference.data.vertices],
                [
                    tuple(reversed(p.vertices)) if mirrored else tuple(p.vertices)
                    for p in reference.data.polygons
                ],
            )

        score = 0.0
        step = max(1, len(component) // 512)
        for face in component[::step]:
            normal = previous[face]
            if reference_tree is not None:
                _, nearest_normal, _, _ = reference_tree.find_nearest(face.calc_center_median())
                if nearest_normal is not None:
                    normal = nearest_normal
            score += face.calc_area() * face.normal.dot(normal)
        if score < 0:
            bmesh.ops.reverse_faces(bm, faces=component)
