"""Locked UVPack transport and failure-atomic application of upstream transforms."""
import hashlib
import json
import os
import time

import numpy as np

from ...core.hardware import available_cpu_count
from ...core.jobs import Cancelled
from ...core.uv import cleanseams_native
from ...core.uv.metrics import lost_coverage_faces, introduces_uv_cuts
from ...core.uv.texel import uv_face_areas, surface_face_areas
from ...runtime.manifest import RuntimeManifestError, _json, load_runtime_manifest
from ...runtime.native import load_uvpack
from ...runtime.process import run_process
from ...runtime.session import verify_current_runtime


def uv_fingerprint(uv):
    """Identity of the actual binary32 corner buffer certified by the packer."""
    return hashlib.sha256(np.ascontiguousarray(uv, dtype='<f4').tobytes()).hexdigest()


class UVPackClient:
    def __init__(self, *, deadline=None, cancelled=None):
        self.deadline = deadline if deadline is not None else time.monotonic() + 300.
        self.cancelled = cancelled
        self.check_cancelled()
        self.runtime = load_uvpack(timeout=self.remaining(), cancelled=cancelled)
        self.companion = self.runtime.companion

    def check_cancelled(self):
        if self.cancelled is not None and self.cancelled():
            raise Cancelled('UVPack cancelled')

    def remaining(self):
        self.check_cancelled()
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError('UVPack exceeded the remaining packing budget.')
        return remaining

    def request(self, payload):
        self.check_cancelled()
        verify_current_runtime()
        binary = self.runtime.checked.manifest.engines['uvpack'].entrypoints['cli']
        env = {k:v for k,v in os.environ.items() if not k.upper().startswith(('LD_', 'DYLD_'))}
        try:
            result = run_process([str(binary), '-'], input=json.dumps(payload, allow_nan=False).encode(),
                                 timeout=self.remaining(), cancelled=self.cancelled, env=env)
        except OSError as error:
            raise ValueError(f'UVPack could not start: {error}') from error
        self.check_cancelled()
        if result.cancelled:
            raise Cancelled('UVPack cancelled')
        if result.timed_out:
            raise ValueError('UVPack exceeded the remaining packing budget.')
        if result.stdout_truncated:
            raise ValueError('UVPack response exceeded the output limit.')
        try:
            report = _json(result.stdout.decode('utf-8'))
        except (UnicodeError, ValueError, RuntimeManifestError) as error:
            raise ValueError('UVPack returned malformed JSON.') from error
        if not isinstance(report, dict) or report.get('dialect') != 'uvpack':
            raise ValueError('UVPack returned an invalid result identity.')
        if report.get('cancelled'):
            raise Cancelled('UVPack cancelled')
        if payload.get('operation') == 'normalize':
            if (result.returncode != 0 or report.get('accepted') is not True or
                    report.get('complete') is not True or report.get('operation') != 'normalize' or
                    report.get('output_precision') != 'float32'):
                raise ValueError(f'UVPack normalize failed: {report.get("error") or report.get("failure") or "invalid normalization result"}')
            if 'valid' in report or 'validation' in report:
                raise ValueError('UVPack normalization returned an unexpected packing certificate.')
            return report
        validation = report.get('validation')
        if report.get('valid') is True and not isinstance(validation, dict):
            raise ValueError('UVPack returned an invalid validation report.')
        # Exit 2 is a documented, structured unsuccessful result, not a reason
        # to discard the diagnostic or invoke a different packer.
        if (result.returncode not in (0, 2) or result.returncode == 2 or
                report.get('valid') is not True or report.get('complete') is not True or
                report.get('output_precision') != 'float32' or
                not isinstance(report.get('validation'), dict) or
                report['validation'].get('ok') is not True):
            detail = report.get('error') or report.get('failure') or (validation.get('messages') if isinstance(validation, dict) else None) or report.get('input')
            raise ValueError(f'UVPack {payload.get("operation", "pack")} failed: {detail or "no complete valid layout"}')
        if payload.get('operation') == 'validate' and report.get('operation') != 'validate':
            raise ValueError('UVPack returned the wrong operation.')
        return report

    def extract(self, arrays, uv, chart_ids=None):
        return self.companion.extract_mesh_islands(
            uv, arrays.face_starts, arrays.face_sizes, arrays.corner_verts, chart_ids=chart_ids,
            face_area_uv=uv_face_areas(arrays, uv), face_area_3d=surface_face_areas(arrays))

    @staticmethod
    def labels(arrays, extracted):
        labels = np.empty(arrays.num_faces, dtype=np.int64)
        for item in extracted.islands:
            labels[item['faces']] = item['id']
        return labels

    @staticmethod
    def fixed_ids(extracted, fixed_corners):
        corners = set(fixed_corners)
        return {item['id'] for item in extracted.islands if corners.intersection(item['corners'])}

    def normalize(self, arrays, before, *, texture_size, chart_ids=None, equalize=True,
                  density_target=None, fixed_corners=()):
        extracted = self.extract(arrays, before, chart_ids)
        held = self.fixed_ids(extracted, fixed_corners)
        for item in extracted.islands:
            item['fixed'] = item['id'] in held
        options = {'output_precision':'float32', 'tex_size':texture_size}
        if density_target is not None:
            options.update(density_mode='uniform_layout', texel_density=density_target)
        elif equalize:
            options['equalize_density'] = True
        else:
            raise ValueError('UV normalization requires a density policy.')
        report = self.request({'operation':'normalize', 'islands':extracted.islands, 'options':options})
        normalized = np.asarray(self.companion.apply_mesh_layout(before, extracted, report, precision='float32'), dtype=np.float64)
        if lost_coverage_faces(arrays, before, normalized).any() or introduces_uv_cuts(arrays, before, normalized):
            raise ValueError('UVPack normalization lost original coverage or chart connections.')
        for item in extracted.islands:
            if item['id'] in held and not np.array_equal(normalized[item['corners']], before[item['corners']]):
                raise ValueError('UVPack normalization moved a fixed chart.')
        if (not isinstance(report.get('fixed_ids'), list) or len(report['fixed_ids']) != len(held)
                or set(report['fixed_ids']) != held):
            raise ValueError('UVPack normalization returned incorrect fixed chart IDs.')
        return normalized, self.labels(arrays, extracted), report

    @staticmethod
    def footprints(arrays, uv, extracted, hull_ids=()):
        by_faces = set(hull_ids) | {item['id'] for item in extracted.issues}
        islands = []
        for item in extracted.islands:
            if item['id'] in by_faces:
                # A chart with a touching boundary has no simple union ring.
                # Submit its original polygons as rigid parts, without inventing
                # an outline. CleanSeams separately checks within-chart coverage.
                parts = [{'outer': uv[int(arrays.face_starts[f]):int(arrays.face_starts[f]+arrays.face_sizes[f])].tolist(),
                          'holes': []} for f in item['faces']]
                islands.append({'id': item['id'], **parts[0], 'parts': parts[1:]})
            else:
                islands.append({key:item[key] for key in ('id', 'outer', 'holes', 'parts') if key in item})
        return islands

    def validate(self, arrays, uv, chart_ids, padding_px, texture_size, *, hull_ids=(), only_ids=None):
        islands = self.footprints(arrays, uv, self.extract(arrays, uv, chart_ids), hull_ids)
        if only_ids is not None:
            islands = [item for item in islands if item['id'] in only_ids]
        return self.request({'operation':'validate', 'islands':islands, 'target':[0,0,1,1],
                             'options':{'tex_size':texture_size, 'pixel_margin':padding_px,
                                        'pixel_border_margin':padding_px, 'output_precision':'float32'}})

    def pack(self, arrays, before, *, chart_ids=None, padding_px, texture_size,
             rotate_method, allow_scaling, search_seconds, density_target=None, equalize=True,
             fixed_corners=()):
        original = before
        extraction = self.extract(arrays, before, chart_ids)
        labels = self.labels(arrays, extraction)
        held = self.fixed_ids(extraction, fixed_corners)
        normalization = None
        if held and ((equalize and allow_scaling) or density_target is not None):
            before, labels, normalization = self.normalize(arrays, before, texture_size=texture_size,
                chart_ids=labels, equalize=equalize and allow_scaling,
                density_target=density_target, fixed_corners=fixed_corners)
            extraction = self.extract(arrays, before, labels)
        movable = [item for item in extraction.islands if item['id'] not in held]
        identities = [{'id':item['id'], 'transform':[1,0,0,0,1,0,0,0,1]}
                      for item in extraction.islands if item['id'] in held]
        obstacles = []
        if held:
            self.validate(arrays, before, labels, padding_px, texture_size, only_ids=held)
            for item in self.footprints(arrays, before, extraction):
                if item['id'] in held:
                    obstacles.append({key:item[key] for key in ('outer','holes') if key in item})
                    obstacles.extend(item.get('parts', []))
        options = {'tex_size':texture_size, 'pixel_margin':padding_px, 'pixel_border_margin':padding_px,
                   'allow_rotation':rotate_method != 'AXIS_ALIGNED', 'rotation_step':90,
                   'prealign':rotate_method == 'ANY', 'threads':available_cpu_count(),
                   'validate':True, 'output_precision':'float32', 'require_complete':True,
                   'hull_policy':'convex_hull', 'equalize_density':bool(equalize and allow_scaling and not held)}
        if search_seconds > 0:
            options['time_budget'] = min(float(search_seconds), 2.)
            if search_seconds > 2:
                options['compaction_time'] = float(search_seconds) - 2.
        if not allow_scaling:
            options.update(scale_mode='fixed', scale=1.)
            if density_target is not None and not held:
                options.update(density_mode='uniform_layout', texel_density=density_target)
        report = (self.request({'operation':'pack', 'islands':movable, 'obstacles':obstacles, 'options':options})
                  if movable else {'operation':'hold_fixed', 'islands':[], 'complete':True})
        if not isinstance(report.get('islands'), list) or not all(isinstance(item, dict) for item in report['islands']):
            raise ValueError('UVPack returned an invalid island table.')
        if report.get('unplaced') or any(item.get('shelved') or item.get('tile', 0) != 0 for item in report['islands']):
            raise ValueError('UVPack did not place every chart in the requested tile.')
        # The native packing report accounts for movable charts only. Identity
        # entries apply held charts; the separate certificate covers both sets.
        application = {**report, 'islands':report['islands'] + identities}
        packed = np.asarray(self.companion.apply_mesh_layout(before, extraction, application, precision='float32'), dtype=np.float64)
        if lost_coverage_faces(arrays, original, packed).any():
            raise ValueError('UVPack proposed loss of original triangle coverage.')
        for item in extraction.islands:
            if item['id'] in held and not np.array_equal(packed[item['corners']], original[item['corners']]):
                raise ValueError('UVPack packing moved a fixed chart.')
        certificate = self.validate(arrays, packed, labels, padding_px, texture_size,
                                    hull_ids=report.get('input', {}).get('hull_ids', ()))
        report['held_fixed_ids'] = sorted(held)
        if normalization is not None:
            report['normalization'] = normalization
        return packed, labels, report, certificate


def pack_object(obj, arrays, *, padding_px, texture_size, rotate_method, allow_scaling,
                search_seconds, density_target=None, chart_ids=None, equalize=True,
                deadline=None, cancelled=None):
    from .. import uv_io, mesh_io
    client = UVPackClient(deadline=deadline, cancelled=cancelled)
    before = uv_io.read_uvs(obj.data)
    pins = uv_io.read_pins(obj.data, before)
    geometry = cleanseams_native.geometry_fingerprint(arrays)
    packed, labels, report, certificate = client.pack(arrays, before, chart_ids=chart_ids,
        padding_px=padding_px, texture_size=texture_size, rotate_method=rotate_method,
        allow_scaling=allow_scaling, search_seconds=search_seconds,
        density_target=density_target, equalize=equalize, fixed_corners=pins)
    client.check_cancelled()
    if (cleanseams_native.geometry_fingerprint(mesh_io.mesh_to_arrays(obj.data)) != geometry or
            not np.array_equal(uv_io.read_uvs(obj.data), before)):
        raise ValueError('UVPack input changed before application.')
    if load_runtime_manifest(client.runtime.checked.manifest.root).identity != client.runtime.checked.manifest.identity:
        raise ValueError('UVPack runtime changed before application.')
    try:
        uv_io.write_uvs(obj.data, packed)
        stored = uv_io.read_uvs(obj.data)
        if not np.array_equal(stored, packed):
            raise ValueError('UVPack stored readback differs from the authorized layout.')
        uv_io.check_pins(obj.data, pins)
        stored_certificate = client.validate(arrays, stored, labels, padding_px, texture_size,
            hull_ids=report.get('input', {}).get('hull_ids', ()))
        client.check_cancelled()
    except BaseException:
        uv_io.write_uvs(obj.data, before)
        raise
    return {'packing_uv_sha256':uv_fingerprint(stored), 'packing_report':report, 'packing_validation':stored_certificate,
            'packing_proposed_validation':certificate,
            'packing_runtime':client.runtime.checked.diagnostic()}
