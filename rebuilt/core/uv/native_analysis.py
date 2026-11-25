"""Immutable native geometry analysis cached for one mutable array snapshot."""
import hashlib
from types import MappingProxyType

import numpy as np

from . import cleanseams_native


def analyze(arrays):
    """Reuse canonical triangles only while every source buffer stays identical."""
    arrays.validate()
    expected_starts = np.r_[0, np.cumsum(arrays.face_sizes[:-1])] if arrays.num_faces else []
    if not np.array_equal(arrays.face_starts, expected_starts):
        raise ValueError('Mesh face ranges must be contiguous and cover every corner.')
    digest = hashlib.blake2b(digest_size=20)
    for name in ('positions', 'corner_verts', 'face_starts', 'face_sizes'):
        array = np.ascontiguousarray(getattr(arrays, name))
        digest.update(str((array.dtype.str, array.shape)).encode('ascii'))
        digest.update(array.tobytes())
    key = digest.digest()
    cached = arrays._analysis_cache
    if cached is not None and cached[0] == key:
        return cached[1]
    native = cleanseams_native.load()
    result = native.analyze_mesh({
        'positions': np.array(arrays.positions, dtype=np.float64, order='C', copy=True),
        'face_offsets': np.r_[arrays.face_starts, arrays.num_corners].astype(np.uint64),
        'corner_vertices': np.array(arrays.corner_verts, dtype=np.int64, order='C', copy=True),
    }, memory_budget=native.recommended_memory_budget())
    for value in result.values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    immutable = MappingProxyType(result)
    arrays._analysis_cache = (key, immutable)
    return immutable
