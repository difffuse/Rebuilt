"""Rebuild small, scan-supported planar slits without moving the guide layout."""
from functools import lru_cache

import bmesh
import numpy as np
from mathutils import Vector

from ..core.analysis.mesh_graph import edge_topology, face_areas


def _partition(points, normal, cell):
    """Find a convex all-quad partition of a small, possibly concave polygon."""
    @lru_cache(None)
    def solve(indices):
        if len(indices) == 2:
            return 1., ()
        best = (0., ())
        for j in range(2, len(indices)-1, 2):
            for k in range(j+1, len(indices), 2):
                quad = (indices[0], indices[1], indices[j], indices[k])
                q = points[list(quad)]
                edges = np.roll(q, -1, axis=0)-q
                lengths = np.linalg.norm(edges, axis=1)
                if lengths.min() < cell*.05 or lengths.max() > cell*3:
                    continue
                turns = np.cross(edges, np.roll(edges, -1, axis=0)) @ normal
                score = float(np.min(turns/(lengths*np.roll(lengths, -1))))
                area = float(np.cross(q-q.mean(0), np.roll(q, -1, axis=0)-q.mean(0)).sum(0) @ normal)*.5
                if score < .25 or not cell*cell*.05 < area < cell*cell*4:
                    continue
                quads = (quad,)
                for part in (indices[1:j+1], indices[j:k+1], indices[k:]+indices[:1]):
                    value, partition = solve(part)
                    score = min(score, value)
                    quads += partition
                if score > best[0]:
                    best = (score, quads)
        return best

    return solve(tuple(range(len(points))))[1]


def _supported(points, quads, normal, cell, tree, source_boundary):
    nearby = source_boundary[np.all((source_boundary >= points.min(0)-cell*.1)
                                    & (source_boundary <= points.max(0)+cell*.1), axis=1)]
    if len(nearby) >= 3:
        planar = nearby-nearby.mean(0)
        planar -= (planar @ normal)[:, None]*normal
        _, _, axes = np.linalg.svd(planar, full_matrices=False)
        # A genuine source opening is not a remesher defect. Only negligible
        # collinear scan slits can be crossed by this local reconstruction.
        if np.prod(np.ptp(planar @ axes[:2].T, axis=0)) > cell*cell*.002:
            return False
    n = Vector(normal)
    for quad in quads:
        p = points[list(quad)]
        for u in np.linspace(0, 1, 11):
            for v in np.linspace(0, 1, 11):
                point = Vector((1-u)*(1-v)*p[0]+u*(1-v)*p[1]+u*v*p[2]+(1-u)*v*p[3])
                hit, hit_normal, _, _ = tree.ray_cast(point+n*(cell*.2), -n, cell*.4)
                if hit is not None and (hit-point).length <= cell*.08 and hit_normal.dot(n) >= .94:
                    continue
                # Float-sized source cracks may miss a ray; this tolerance is
                # much smaller than a quad and cannot bridge ordinary openings.
                hit, hit_normal, _, distance = tree.find_nearest(point)
                if hit is None or distance > cell*.002 or hit_normal.dot(n) < .94:
                    return False
    return True


def repair_supported_patches(obj, source, query, target_faces, loops_of):
    """Replace a folded slit and its immediate faces with regular convex quads.

    Boundaries and surviving vertices are immutable. Ambiguous, curved, large or
    unsupported regions are left alone; no normals are used to hide a bad patch.
    """
    report = dict(patches=0, removed_faces=0, added_quads=0)
    cell = float(np.sqrt(face_areas(source).sum()/max(target_faces, 1)))
    if cell <= 0:
        return report
    topology = edge_topology(source)
    source_boundary = source.positions[np.unique(topology.edges[topology.boundary])]
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    mesh.normal_update()
    try:
        for loop in loops_of([e for e in mesh.edges if e.is_boundary]):
            if report['patches'] >= 16:
                break
            if not 3 <= len(loop) <= 8 or not all(v.is_valid for v in loop):
                continue
            p = np.array([v.co for v in loop], dtype=float)
            area = np.linalg.norm(np.cross(p-p.mean(0), np.roll(p, -1, axis=0)-p.mean(0)).sum(0))*.5
            if np.linalg.norm(np.ptp(p, axis=0)) > cell*2 or area > cell*cell*.15:
                continue
            region = {f for v in loop for f in v.link_faces}
            if not 3 <= len(region) <= 12:
                continue
            edges = {e for f in region for e in f.edges}
            hole_edges = {mesh.edges.get((a, b)) for a, b in zip(loop, loop[1:]+loop[:1])}
            if {e for e in edges if e.is_boundary} != hole_edges:
                continue
            border = [e for e in edges if sum(f in region for f in e.link_faces) == 1 and not e.is_boundary]
            rings = loops_of(border)
            if len(rings) != 1 or len(border) != len(rings[0]):
                continue
            outer = rings[0]
            if len(outer) not in (6, 8, 10, 12) or any(len(e.link_faces) != 2 for e in border):
                continue
            internal = {v for f in region for v in f.verts}-set(outer)
            if any(any(f not in region for f in v.link_faces) for v in internal):
                continue
            edge = mesh.edges.get((outer[0], outer[1]))
            outside = next(link for link in edge.link_loops if link.face not in region)
            if outside.vert is outer[0]:
                outer = list(reversed(outer))
            if any(next(link for link in mesh.edges.get((a,b)).link_loops if link.face not in region).vert is a
                   for a,b in zip(outer, outer[1:]+outer[:1])):
                continue
            points = np.array([v.co for v in outer], dtype=float)
            centre = points.mean(0)
            normal = np.cross(points-centre, np.roll(points, -1, axis=0)-centre).sum(0)
            length = np.linalg.norm(normal)
            if length < cell*cell*.1:
                continue
            normal /= length
            neighbours = {f for e in border for f in e.link_faces if f not in region}
            if np.max(abs((points-centre) @ normal)) > cell*.08 or any(np.dot(f.normal, normal) < .94 for f in neighbours):
                continue
            quads = _partition(points, normal, cell)
            if not quads or not _supported(points, quads, normal, cell, query._tree, source_boundary):
                continue
            # A convex partition must cover precisely the original outer polygon.
            covered = sum(np.cross(points[list(q)]-centre, np.roll(points[list(q)], -1, axis=0)-centre).sum(0) @ normal for q in quads)
            if abs(covered-length) > length*1e-6:
                continue
            outside_sets = {frozenset(f.verts) for v in outer for f in v.link_faces if f not in region}
            if any(frozenset(outer[i] for i in q) in outside_sets for q in quads):
                continue
            material = next(iter(region)).material_index
            removed = len(region)
            bmesh.ops.delete(mesh, geom=list(region), context='FACES_ONLY')
            for e in edges:
                if e.is_valid and not e.link_faces:
                    bmesh.ops.delete(mesh, geom=[e], context='EDGES')
            for v in internal:
                if v.is_valid and not v.link_edges:
                    bmesh.ops.delete(mesh, geom=[v], context='VERTS')
            for quad in quads:
                mesh.faces.new([outer[i] for i in quad]).material_index = material
            mesh.normal_update()
            report['patches'] += 1
            report['removed_faces'] += removed
            report['added_quads'] += len(quads)
        if report['patches']:
            mesh.to_mesh(obj.data)
            obj.data.update()
        return report
    finally:
        mesh.free()
