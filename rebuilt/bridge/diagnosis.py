"""Measuring a scan before anything is asked of it: how thick its walls are."""

from __future__ import annotations

import numpy as np

from ..core.analysis.mesh_graph import face_areas, face_normals
from ..core.types import MeshArrays
from . import bvh, mesh_io


MEASURE_TRIANGLES = 150_000


SAMPLES = 3000


REACH = 0.25

QUANTILES = 200


def measure_thickness(obj, seed: int = 0) -> dict:
    """Thickness quantiles and area of `obj`'s surface, as the report stores them."""
    import bpy

    from ..stages.retopo import _decimate_to, _duplicate, _sanitize_scan

    temp = _duplicate(obj, "__rebuilt_diagnosis")
    bpy.context.scene.collection.objects.link(temp)
    try:
        _sanitize_scan(temp)
        _decimate_to(temp, MEASURE_TRIANGLES)
        arrays = mesh_io.mesh_to_arrays(temp.data)
    finally:
        mesh = temp.data
        bpy.data.objects.remove(temp, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    return thickness_of(arrays, seed=seed)


def thickness_of(arrays: MeshArrays, seed: int = 0) -> dict:
    if arrays.num_faces == 0:
        return {"thickness_quantiles": [], "area": 0.0, "samples": 0}
    areas = face_areas(arrays)
    normals = face_normals(arrays)
    starts = arrays.face_starts.astype(np.int64)
    sizes = arrays.face_sizes.astype(np.int64)
    rng = np.random.default_rng(seed)

    weights = areas / max(float(areas.sum()), 1e-30)
    chosen = rng.choice(
        arrays.num_faces, size=min(SAMPLES, arrays.num_faces), replace=True, p=weights
    )
    positions = arrays.positions.astype(np.float64)
    centres = np.stack(
        [
            positions[arrays.corner_verts[starts[f] : starts[f] + sizes[f]]].mean(axis=0)
            for f in chosen
        ]
    )
    diagonal = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    query = bvh.SurfaceQuery(arrays)
    spans = query.inward_spans(centres, normals[chosen], np.full(len(chosen), REACH * diagonal))
    finite = spans[np.isfinite(spans)]
    quantiles = (
        np.quantile(finite, np.linspace(0.0, 1.0, QUANTILES)).tolist() if finite.size else []
    )
    return {
        "thickness_quantiles": [float(q) for q in quantiles],
        "thick_share": float(1.0 - finite.size / max(len(spans), 1)),
        "area": float(areas.sum()),
        "samples": int(len(spans)),
        "diagonal": diagonal,
    }


def thickness_array(report: dict) -> np.ndarray:
    """The distribution back as samples the judgement can take: the finite quantiles, padded with `inf` for the share nothing was measured under."""
    quantiles = np.asarray(report.get("thickness_quantiles", []), dtype=np.float64)
    thick = float(report.get("thick_share", 0.0))
    if quantiles.size == 0:
        return np.zeros(0)
    extra = int(round(quantiles.size * thick / max(1.0 - thick, 1e-6)))
    return np.concatenate([quantiles, np.full(extra, np.inf)])
