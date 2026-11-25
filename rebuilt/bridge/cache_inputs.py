"""Read-only identities for the low mesh inputs consumed by UV and Bake."""
import hashlib
import struct

import numpy as np


def _digest(layer, *arrays):
    digest = hashlib.sha256(b'rebuilt-scene-uv-1')
    name = layer.encode('utf-8')
    digest.update(struct.pack('<Q', len(name)))
    digest.update(name)
    for array in arrays:
        array = np.ascontiguousarray(array)
        descriptor = f'{array.dtype.str}:{array.shape}'.encode('ascii')
        digest.update(struct.pack('<Q', len(descriptor)))
        digest.update(descriptor)
        digest.update(array.tobytes())
    return digest.hexdigest()


def uv_cache_inputs(low):
    """Snapshot live Edit Mode data without flushing edits or changing selection."""
    if low is None:
        return {'uv': 'scene-uv-1:missing-low', 'bake': 'scene-uv-1:missing-low'}
    mesh = low.data
    if low.mode == 'EDIT':
        import bmesh
        bm = bmesh.from_edit_mesh(mesh)
        indices = {vertex: index for index, vertex in enumerate(bm.verts)}
        seams = [(indices[edge.verts[0]], indices[edge.verts[1]])
                 for edge in bm.edges if edge.seam]
        layer = bm.loops.layers.uv.active
        name = layer.name if layer is not None else ''
        loops = [loop[layer] for face in bm.faces for loop in face.loops] if layer else []
        uv = np.asarray([tuple(loop.uv) for loop in loops], dtype='<f4').reshape(-1, 2)
        pins = np.asarray([loop.pin_uv for loop in loops], dtype=bool)
    else:
        flags = np.empty(len(mesh.edges), dtype=bool)
        pairs = np.empty(len(mesh.edges) * 2, dtype='<i8')
        mesh.edges.foreach_get('use_seam', flags)
        mesh.edges.foreach_get('vertices', pairs)
        seams = pairs.reshape(-1, 2)[flags]
        layer = mesh.uv_layers.active
        name = layer.name if layer is not None else ''
        uv = np.empty((len(mesh.loops) if layer else 0, 2), dtype='<f4')
        pins = np.empty(len(uv), dtype=bool)
        if layer is not None:
            layer.uv.foreach_get('vector', uv.ravel())
            layer.data.foreach_get('pin_uv', pins)
    seams = np.asarray(seams, dtype='<i8').reshape(-1, 2)
    seams.sort(axis=1)
    if len(seams):
        seams = seams[np.lexsort((seams[:, 1], seams[:, 0]))]
    pin_corners = np.flatnonzero(pins).astype('<i8')
    return {
        'uv': 'scene-uv-1:' + _digest(name, seams, pin_corners, uv[pins]),
        'bake': 'scene-uv-1:' + _digest(name, uv),
    }
