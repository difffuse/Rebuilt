"""Conservative native-hole repair where the scan supplies a planar surface."""

import bmesh
import numpy as np
from mathutils.bvhtree import BVHTree

from ..core.analysis.mesh_graph import edge_topology, face_areas
from .triangulation import remesher_triangles


def close_supported_holes(obj, source, query, target_faces, loops_of):
    """Add small supported caps without moving or deleting existing geometry."""
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    mesh.normal_update()
    added = 0
    candidates = []
    cell = float(np.sqrt(face_areas(source).sum() / max(target_faces, 1)))
    try:
        if cell <= 0:
            return 0
        for loop in loops_of([edge for edge in mesh.edges if edge.is_boundary]):
            if not 3 <= len(loop) <= 8:
                continue
            points = np.array([vert.co for vert in loop], dtype=float)
            centre = points.mean(axis=0)
            if np.linalg.norm(np.ptp(points, axis=0)) > 3 * cell:
                continue
            normal = np.cross(points - centre, np.roll(points, -1, axis=0) - centre).sum(axis=0)
            area = np.linalg.norm(normal) * .5
            if not cell * cell * .01 < area < cell * cell * 4:
                continue
            normal /= 2 * area
            edge = mesh.edges.get((loop[0], loop[1]))
            link = edge.link_loops[0]
            if link.vert is loop[0]:
                loop = list(reversed(loop))
                points = points[::-1]
                normal = -normal
            if np.max(np.abs((points - centre) @ normal)) > cell * .03:
                continue
            neighbours = {face for vert in loop for face in vert.link_faces}
            if any(np.dot(face.normal, normal) < .94 for face in neighbours):
                continue
            turns = np.cross(np.roll(points, -1, axis=0) - points,
                             np.roll(points, -2, axis=0) - np.roll(points, -1, axis=0)) @ normal
            if np.min(turns) < -cell * cell * .005:
                continue
            candidates.append((loop, points, centre, normal))
        if not candidates:
            return 0
        tree = query._tree
        if np.any(source.face_sizes != 3):
            tree = BVHTree.FromPolygons(source.positions.tolist(), remesher_triangles(source).tolist(), all_triangles=True)
        topology = edge_topology(source)
        boundary = source.positions[np.unique(topology.edges[topology.boundary])]
        for loop, points, centre, normal in candidates:
            if len(boundary):
                nearby = boundary[np.all((boundary >= points.min(0) - cell * .25)
                                         & (boundary <= points.max(0) + cell * .25), axis=1)]
                if len(nearby) >= 3:
                    _, _, axes = np.linalg.svd(nearby - nearby.mean(0), full_matrices=False)
                    span = np.ptp(nearby @ axes[:2].T, axis=0)
                    if float(np.prod(span)) > cell * cell * .005:
                        continue
            samples = [centre]
            for a, b in zip(points, np.roll(points, -1, axis=0)):
                for i in range(1, 6):
                    for j in range(6 - i):
                        samples.append((centre * (6-i-j) + a * i + b * j) / 6)
            supported = True
            for point in samples:
                location, hit_normal, _, distance = tree.find_nearest(tuple(point))
                if location is None or distance > cell * .06 or np.dot(normal, hit_normal) < .94:
                    supported = False
                    break
            if not supported:
                continue
            # Every shared edge must oppose its existing face's direction.
            if any(mesh.edges.get((a,b)).link_loops[0].vert is a
                   for a,b in zip(loop, loop[1:]+loop[:1])):
                continue
            try:
                mesh.faces.new(loop)
            except ValueError:
                continue
            added += 1
        if added:
            mesh.to_mesh(obj.data)
            obj.data.update()
        return added
    finally:
        mesh.free()
