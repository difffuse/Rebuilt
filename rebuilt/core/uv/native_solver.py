"""Transport native CleanSeams requests to the application's main-thread host."""
from copy import deepcopy

import numpy as np


class NativeHostTransport:
    def __init__(self, calls, report, method):
        self.calls, self.report, self.method = calls, report, method
        self.unwrap_calls = 0

    def respond(self, request):
        """Preserve native selection/identity; return only after cancellation checks."""
        self.report.check_cancelled()
        kind = request['kind']
        if kind not in ('solve', 'region_proposal'):
            raise ValueError(f'Unsupported CleanSeams host request kind: {kind}')
        offsets = np.asarray(request['face_offsets'], dtype=np.int64)
        payload = {
            'positions': request['positions'],
            'corner_verts': request['corner_vertices'],
            'face_starts': offsets[:-1],
            'face_sizes': np.diff(offsets),
            'method': request['method'],
        }
        if kind == 'solve':
            count = len(request['corner_vertices'])
            initial = request['initial_uv']
            initial = (np.zeros((count, 2), dtype=np.float32) if initial is None
                       else np.array(initial, dtype=np.float32, copy=True))
            pinned = np.zeros(count, dtype=bool)
            ids = np.asarray(request['pin_corners'], dtype=np.int64)
            initial[ids] = request['pin_uv']
            pinned[ids] = True
            payload.update(uvs=initial, pins=pinned)
        answer = self.calls.call(payload, self.report)
        self.report.check_cancelled()
        if kind == 'solve':
            self.unwrap_calls += 1
        return {
            'identity': deepcopy(request['identity']), 'kind': kind,
            'success': True, 'method': request['method'],
            'uv': np.array(answer, dtype=np.float64, order='C', copy=True),
        }
