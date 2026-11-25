"""Array and main-thread transport for the required locked CleanSeams C++ runtime."""
from copy import deepcopy
import json
import time
import numpy as np
from ..hardware import available_cpu_count
from .contracts import NativeSeamResult

def load():
    """Use the same package-owned native module as the runtime verifier."""
    from ...runtime.native import load_cleanseams

    native = load_cleanseams()
    require_capabilities(native.build_info())
    return native


def require_capabilities(info):
    """Check the full repair contract, beyond the lock's basic feature flags."""
    required = {
        'behaviors': ('repair_v1',),
        'operations': ('pipeline', 'postprocess_layout'),
        'repair_operations': ('repair_regions', 'repair_singletons', 'join_small_islands', 'recover_fragments'),
        'host_capabilities': ('solve', 'region_proposal'),
        'join_modes': ('fewer',),
    }
    missing = []
    for field, names in required.items():
        available = info.get(field)
        missing.extend(name for name in names if not isinstance(available, list) or name not in available)
    features = info.get('features', {})
    missing.extend(name for name in ('mesh_analysis', 'supplied_layout_validation', 'host_method_selection')
                   if not isinstance(features, dict) or features.get(name) is not True)
    if missing:
        raise RuntimeError('CleanSeams runtime is missing required capabilities: ' + ', '.join(missing)
                           + '. Reinstall a compatible Rebuilt package.')


def worker_count(requested=None):
    """Use process-available CPUs within the native engine's 64-thread limit."""
    if requested is not None and requested != 0:
        if not isinstance(requested, int) or not 1 <= requested <= 64:
            raise ValueError('CleanSeams worker count must be an integer between 1 and 64')
        return requested
    return min(64, available_cpu_count())


def _host_buffers(arrays, native, budget):
    """Canonical host buffers shared by submission and fresh apply-time identity."""
    arrays.validate()
    positions = np.array(arrays.positions, dtype=np.float32, order='C', copy=True)
    corners = np.array(arrays.corner_verts, dtype=np.int64, order='C', copy=True)
    offsets = np.concatenate((arrays.face_starts, [arrays.num_corners])).astype(np.uint64)
    analysis = native.analyze_mesh({
        'positions': positions.astype(np.float64),
        'face_offsets': offsets, 'corner_vertices': corners,
    }, memory_budget=budget)
    return (positions, analysis['edge_vertices'].astype(np.int32),
            corners.astype(np.int32), analysis['corner_edges'].astype(np.int32),
            np.ascontiguousarray(arrays.face_starts, dtype=np.int32),
            np.ascontiguousarray(arrays.face_sizes, dtype=np.int32))


def geometry_fingerprint(arrays):
    native = load()
    budget = native.recommended_memory_budget()
    return native.host_geometry_fingerprint(*_host_buffers(arrays, native, budget),
                                            memory_budget=budget)


def run(arrays, seams_in=None, uv_in=None, pins=None, locked=None, anchors=None, opts=None,
        backend=None, control=None, progress=None, *, policy='native_baseline', threads=None, protected_edges=(),
        operation='pipeline', selected_faces=None, cut_edges=(), retain_for_apply=False):
    workers = worker_count(threads)
    native = load()
    # The library owns option defaults and validation. Only application choices
    # are sent here; do not instantiate the retired Python engine's models.
    options = dict(opts or {})
    options['policy'] = policy
    options_json = native.pipeline_options_json(json.dumps(options, allow_nan=False))
    budget = native.recommended_memory_budget()
    buffers = _host_buffers(arrays, native, budget)
    edges = buffers[1]
    seams = (np.zeros(len(edges), dtype=bool) if seams_in is None
             else np.array(seams_in, dtype=bool, copy=True))
    locks = (np.zeros(len(edges), dtype=bool) if locked is None
             else np.array(locked, dtype=bool, copy=True))
    edge_ids = {tuple(edge): i for i, edge in enumerate(edges)}
    for edge in np.sort(np.asarray(cut_edges, dtype=np.int64).reshape(-1, 2), axis=1):
        if tuple(edge) not in edge_ids:
            raise ValueError('A proposed seam no longer belongs to this mesh.')
        seams[edge_ids[tuple(edge)]] = True
    protected = np.asarray(protected_edges, dtype=np.int64).reshape(-1, 2)
    if len(protected):
        for edge in np.sort(protected, axis=1):
            index = edge_ids.get(tuple(edge))
            if index is None:
                raise ValueError('A marked seam no longer belongs to this mesh; run Seams again.')
            seams[index] = locks[index] = True
    mapped = native.read_host_mesh(
        *buffers,
        np.eye(4, dtype=np.float64), np.ones(3, dtype=np.float64),
        memory_budget=budget, attributes={'seams': seams, 'locked': locks})
    payload = {k: mapped[k] for k in ('positions','face_offsets','corner_vertices','geometry_fingerprint')}
    payload['operation'] = operation
    if selected_faces is not None:
        payload['selected_faces'] = np.array(selected_faces, dtype=bool, copy=True)
    payload.update(seam_mask=mapped['seams'].astype(bool), lock_mask=mapped['locked'].astype(bool), options=options_json)
    if uv_in is not None:
        payload['uv'] = np.ascontiguousarray(uv_in, dtype=np.float64)
    if pins:
        payload['pin_corners'] = np.array(list(pins), dtype=np.int64)
        payload['pin_uv'] = np.array(list(pins.values()), dtype=np.float64)
    if anchors is not None:
        payload['anchors'] = np.asarray(anchors, dtype=np.int64)
    session = native.Session(
        backend='host' if backend is not None else 'superlu',
        memory_budget=budget, worker_count=workers,
        host_capabilities=['region_proposal'] if backend is not None else [],
        fragment_unwrap_method=backend.method if backend is not None else 'ANGLE_BASED',
        host_unwrap_method=backend.method if backend is not None else 'MINIMUM_STRETCH')
    try:
        session.start(payload)
        while True:
            if control is not None:
                control.check()
            for event in session.poll_progress():
                if progress is not None:
                    progress(event)
            request = session.next_host_request() if backend is not None else None
            if request is not None:
                session.submit_host_response(backend.respond(request))
            elif session.ready():
                break
            else:
                time.sleep(.001)
        result = session.result()
    finally:
        session.close()
    if result['diagnostics']:
        raise RuntimeError('CleanSeams native: ' + str(result['diagnostics']))
    seams = np.zeros(len(edges),dtype=bool)
    indices = np.asarray(mapped['core_edge_to_host'],dtype=np.int64)
    valid = indices >= 0
    if len(result['seams']):
        seams[indices[valid]] = np.asarray(result['seams'],dtype=bool)[valid]
    uv = None
    if result['uv'] is not None:
        uv = np.full((arrays.num_corners,2),np.nan)
        uv[np.asarray(mapped['host_loop_indices'],dtype=np.int64)] = result['uv']
    exempt = np.zeros(arrays.num_faces,dtype=bool)
    exempt[np.asarray(result['exempt_faces'],dtype=np.int64)] = True
    native_report = json.loads(result['report_json'])
    report = deepcopy(native_report)
    build_info = native.build_info()
    report.update(runtime='cpp', runtime_version=build_info['version'], policy=policy, worker_count=workers,
                  worker_selection='automatic' if threads is None or threads == 0 else 'explicit')
    report['timed_out'] = bool(result['timed_out'])
    # Transfer the binding's owned arrays. Only host-loop ordering needs a
    # scatter; do not infer chart topology again from the returned seams.
    chart_of_corner = result['chart_of_corner']
    if len(chart_of_corner):
        labels = np.empty(arrays.num_corners, dtype=np.int64)
        labels[mapped['host_loop_indices']] = chart_of_corner
        chart_of_corner = labels
    return NativeSeamResult(
        accepted=bool(result['accepted']), reason=result['reason'], seams=seams, uv=uv,
        chart_of_face=result['chart_of_face'], chart_of_corner=chart_of_corner,
        exempt=exempt, invalid_charts=result['invalid_charts'], changed_faces=result['changed_faces'],
        edge_vertices=result['edge_vertices'], host_loop_indices=mapped['host_loop_indices'],
        core_edge_to_host=mapped['core_edge_to_host'], geometry_fingerprint=result['geometry_fingerprint'],
        backend=result['backend'], uv_valid=result['uv_valid'], topology_valid=result['topology_valid'],
        budget_satisfied=result['budget_satisfied'], partial=result['partial'],
        cancelled=result['cancelled'], timed_out=result['timed_out'],
        native_report=native_report, report=report, build_info=build_info,
        _application_session=session if retain_for_apply else None)


def run_available(*args, **kwargs):
    """Compatibility entry point; missing or incompatible native code is an error."""
    return run(*args, **kwargs)


def finalize(arrays, proposal, reporter, *, backend=None, repair_options=None, seconds=10.0,
             pins=None, selected_faces=None, repair_defects=True):
    """Validate the final layout; permit one bounded native defect-region repair.

    Generation and final-layout reports stay separate. The closed Session keeps
    native authorization until apply, without retaining a worker or host request.
    """
    class Control:
        def check(self):
            reporter.check_cancelled()

    deadline = time.monotonic() + seconds
    repair = {'storage_precision': 'float32',
              'require_open_protected_cuts': bool(len(proposal.get('protected_edges', [])))}
    optional = dict(repair_options or {})
    optional.update(repair)
    arguments = dict(backend=backend, uv_in=proposal['uvs'], cut_edges=proposal['edges'],
                     protected_edges=proposal.get('protected_edges', []),
                     policy='repair_v1', operation='postprocess_layout',
                     control=Control(), retain_for_apply=True, pins=pins, selected_faces=selected_faces)
    result = run(arrays, opts={'deadline_s': seconds, 'repair': optional}, **arguments)
    initial_report = result.native_report
    if not (result.accepted and result.uv_valid):
        validation = initial_report.get('stored_validation', {})
        defects = set(validation.get('coverage_defect_faces', []))
        defects.update(validation.get('strict_overlap', {}).get('overlapping_faces', []))
        remaining = deadline - time.monotonic()
        if repair_defects and defects and remaining > 0:
            # Continue from the native incumbent. Restarting from the original
            # input would discard accepted joins outside these defect regions.
            arguments.update(uv_in=result.uv, cut_edges=result.cut_edges)
            result.release_application()
            selected = np.zeros(arrays.num_faces, dtype=bool)
            selected[list(defects)] = True
            repair['region_recovery_seconds'] = min(1.0, remaining)
            arguments['selected_faces'] = selected
            result = run(arrays, opts={'deadline_s': remaining, 'repair': repair}, **arguments)
    if not (result.accepted and result.uv_valid):
        result.release_application()
        raise ValueError(f'CleanSeams final UV validation failed ({result.reason}; '
                         f'{len(result.invalid_charts)} invalid charts). No mesh edits were applied.')
    reporter.check_cancelled()
    proposal['uvs'] = result.uv.copy()
    proposal['edges'] = result.cut_edges
    proposal['junk'] = np.flatnonzero(result.exempt).tolist()
    proposal['charts'] = result.n_charts
    proposal['final_native_report'] = result.native_report
    if initial_report is not result.native_report:
        proposal['initial_final_validation'] = initial_report
    proposal['_native_application'] = result
    return proposal
