"""Native seam generation and the application's remaining layout policies."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from ..jobs import ProgressReporter
from ..types import MeshArrays
from .cleanseams_native import run_available as run, finalize

ENGINE = "cleanseams"
VERSION = "0.2.0"

CACHE_VERSION = f"{ENGINE}-{VERSION}:12-best-effort"
OPTIMIZATION_SECONDS = 45.0
POSTPROCESS_SECONDS = 60.0


class SeamError(ValueError):
    """The engine could not validate a cut; the existing mesh stays intact."""


def fingerprint(arrays: MeshArrays, edges=None, uvs=None) -> str:
    """Fingerprint the geometry, cut and optional corner layout, not edge IDs."""
    digest = hashlib.sha256()
    values = [np.asarray(arrays.positions, dtype="<f4")]
    values.extend(
        np.asarray(a, dtype="<i8")
        for a in (
            arrays.corner_verts,
            arrays.face_starts,
            arrays.face_sizes,
        )
    )
    if edges is not None:
        pairs = np.sort(np.asarray(edges, dtype="<i8").reshape(-1, 2), axis=1)
        if len(pairs):
            pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        values.append(pairs)
    if uvs is not None:
        values.append(np.asarray(uvs, dtype="<f4"))
    for value in values:
        digest.update(str(value.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _json_safe(value):
    """Reports cross the worker boundary as ordinary JSON data, without callbacks."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def plan(
    arrays: MeshArrays,
    asset_class: str,
    report: ProgressReporter,
    *,
    main_thread_calls=None,
    unwrap_method="ANGLE_BASED",
    island_preference="BALANCED",
    protected_edges=(),
) -> dict[str, Any]:
    """Generate one validated cut on a worker, without touching the source arrays."""
    report.update(0.02, "Reading the surface")
    if not arrays.num_faces:
        raise SeamError("Cannot cut seams on an empty mesh")
    source_key = fingerprint(arrays)

    protected = np.unique(np.sort(np.asarray(protected_edges, dtype=np.int64).reshape(-1, 2), axis=1), axis=0)
    hard = asset_class in ("hard_surface", "architecture")
    options = {
        'deadline_s': OPTIMIZATION_SECONDS,
        'layout': {'feature_angle': 60.0 if hard else 0.0, 'prefer_features': not hard},
        'boundary': {'mode': 'auto' if hard else 'organic'},
    }

    class PipelineControl:
        def check(self):
            report.check_cancelled()

    fraction = 0.05
    milestones = {
        "gate": 0.05,
        "generate": 0.15,
        "refine": 0.55,
        "unwrap": 0.65,
        "repair": 0.75,
        "chart_gate": 0.8,
        "contours": 0.9,
        "validate": 0.97,
    }

    def progress(event):
        nonlocal fraction
        fraction = max(fraction, milestones.get(event.get("stage"), fraction))
        labels = {"generate": "Generating seams", "refine": "Refining seams", "unwrap": "Unwrapping charts",
                  "repair": "Repairing charts", "chart_gate": "Checking chart distortion",
                  "contours": "Refining chart boundaries", "validate": "Validating seam layout"}
        message = event.get("msg") or labels.get(event.get("stage"), "Checking seam layout")
        if event.get("total") and event.get("done") is not None:
            message += f" ({event['done']}/{event['total']})"
        report.update(fraction, message)

    backend = None
    if main_thread_calls is not None:
        from .native_solver import NativeHostTransport

        backend = NativeHostTransport(main_thread_calls, report, unwrap_method)
    result = run(
        arrays, protected_edges=protected, opts=options, backend=backend,
        control=PipelineControl(), progress=progress, policy='repair_v1',
    )
    report.check_cancelled()
    fallback = result.report.get("timed_out") or any(
        stage.get("status") == "fallback" for stage in result.report.get("stages", []))
    fragmented = fallback and result.n_charts > max(512, arrays.num_faces // 4)
    details = _json_safe(result.report)
    projected = main_thread_calls is not None and (not result.accepted or fragmented)
    if projected:
        # Whole-object projection remains an explicit application recovery choice.
        # All cut derivation, coverage repair and chart joining belong to native code.
        report.update(.98, "Recovering a practical seam layout")
        payload = {name: getattr(arrays, name) for name in
                   ("positions", "corner_verts", "face_starts", "face_sizes")}
        uvs = np.asarray(main_thread_calls.call(dict(payload, method="SMART_PROJECT"), report),
                         dtype=np.float64)
        edges = np.empty((0, 2), dtype=np.int64)
        details['projection_recovery'] = {
            'reason': 'fragmented_layout' if result.accepted else 'refused_layout',
            'previous_charts': result.n_charts,
        }
    else:
        if not result.accepted:
            raise SeamError(f"Seam layout refused: {result.reason or 'validation failed'}")
        uvs = np.asarray(result.uv, dtype=np.float64).copy()
        if uvs.shape != (arrays.num_corners, 2) or not np.isfinite(uvs).all():
            raise SeamError("Seam layout returned invalid corner UVs")
        groups = [np.flatnonzero(result.chart_of_corner == ch) for ch in range(result.n_charts)]
        span = max((float(np.ptp(uvs[ids], axis=0).max()) for ids in groups if len(ids)), default=1.0)
        uvs /= max(span, 1e-12)
        columns = max(1, int(np.ceil(np.sqrt(len(groups)))))
        for ch, ids in enumerate(groups):
            if len(ids):
                uvs[ids] += np.array([2 * (ch % columns), 2 * (ch // columns)]) - uvs[ids].min(axis=0)
        edges = result.cut_edges
    if uvs.shape != (arrays.num_corners, 2) or not np.isfinite(uvs).all():
        raise SeamError("Projection recovery returned invalid corner UVs")
    planned = {
        'source_key': source_key, 'edges': edges, 'uvs': uvs.astype(np.float32),
        'junk': [], 'charts': result.n_charts, 'engine_report': details,
        'solver': 'SMART_PROJECT' if projected else unwrap_method if backend is not None else 'arap',
        'unwrap_calls': 0, 'protected_edges': protected.tolist(),
    }
    repair_options = {'singleton_seconds': 45.0, 'region_recovery_seconds': 1.0}
    if island_preference == 'FEWER':
        repair_options.update(join_seconds=2.0, join_mode='fewer')
    if backend is not None and result.report.get('deadline_projected_faces', 0) and not projected:
        repair_options['host_proposal_seconds'] = 10.0
    report.update(.99, 'Checking and repairing native UV layout')
    finalize(arrays, planned, report, backend=backend, repair_options=repair_options,
             seconds=POSTPROCESS_SECONDS)
    planned['unwrap_calls'] = (backend.unwrap_calls if backend is not None else 0) + int(projected)
    if projected:
        details['projection_recovery']['charts'] = planned['charts']
    report.update(1.0, 'Seams and contours validated')
    return planned
