"""Admit a bake before texture-sized allocation or mutable host gathering."""
import math
import sys

import bpy

from ..core.bake.memory import GIB, automatic_budget, estimate_bake_memory, require_budget
from .system_memory import installed_memory_bytes


def memory_limit_bytes():
    package = __package__.rsplit('.', 1)[0]
    addon = bpy.context.preferences.addons.get(package)
    setting = addon.preferences.bake_memory_limit_gib if addon is not None else 0.0
    if not math.isfinite(setting) or setting < 0:
        raise ValueError('Set a finite, nonnegative Bake memory limit in Rebuilt addon preferences.')
    if setting > 0:
        return max(1, int(setting * GIB))
    try:
        return automatic_budget(installed_memory_bytes())
    except (OSError, ValueError, AttributeError) as exc:
        raise ValueError(
            'Cannot determine installed RAM. Set an explicit Bake memory limit '
            'in Rebuilt addon preferences before baking.'
        ) from exc


def _mesh_bytes(obj):
    # Use widened array sizes for planning. Edit-mode counts must reflect pending
    # edits without committing them or leaving Edit Mode just to refuse a bake.
    if obj.mode == 'EDIT':
        import bmesh
        mesh = bmesh.from_edit_mesh(obj.data)
        vertices, faces = len(mesh.verts), len(mesh.faces)
        corners = sum(len(face.loops) for face in mesh.faces)
    else:
        mesh = obj.data
        vertices, faces, corners = len(mesh.vertices), len(mesh.polygons), len(mesh.loops)
    return vertices * 24 + corners * 8 + faces * 16


def admit_bake(resolution, map_count, low, high):
    limit = memory_limit_bytes()
    estimate = estimate_bake_memory(
        resolution, map_count, geometry_bytes=_mesh_bytes(low) + _mesh_bytes(high),
        host_bytes=(6 if sys.platform == 'darwin' else 2) * GIB,
    )
    require_budget(estimate, limit)
