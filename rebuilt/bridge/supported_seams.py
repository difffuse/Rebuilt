"""Repair tiny folded native rims only where the scan supplies the missing surface.

The operation runs on disposable main-thread candidates, before guide motion.
Each local attempt is transactional; genuine source openings and unrelated faces
are retained. No vertex is projected across a gap to manufacture support.
"""
import bmesh
import numpy as np

from .bvh import SurfaceQuery, sample_surface
from ..core.types import MeshArrays
from ..core.analysis.mesh_graph import edge_topology, face_areas
from ..core.qa.topology import inspect_topology


def _arrays(mesh):
    mesh.verts.index_update()
    positions = np.array([v.co for v in mesh.verts], dtype=np.float32)
    sizes = np.array([len(f.verts) for f in mesh.faces], dtype=np.int32)
    corners = np.array([v.index for f in mesh.faces for v in f.verts], dtype=np.int32)
    return MeshArrays(positions, corners, np.r_[0, np.cumsum(sizes)[:-1]], sizes)


def _supported_loop(mesh, loop, source_boundary, query, cell):
    if not 3 <= len(loop) <= 16:
        return None
    points = np.array([v.co for v in loop], dtype=float)
    centre = points.mean(0)
    normal = np.cross(points-centre, np.roll(points,-1,axis=0)-centre).sum(0)
    area = np.linalg.norm(normal)*.5
    if not .05*cell*cell < area < 2*cell*cell or np.linalg.norm(np.ptp(points,axis=0)) > 3*cell:
        return None
    normal /= 2*area
    edge = mesh.edges.get((loop[0],loop[1]))
    if edge.link_loops[0].vert is loop[0]:
        normal = -normal
    if np.max(abs((points-centre) @ normal)) > cell*.06:
        return None
    nearby = source_boundary[np.all((source_boundary >= points.min(0)-cell*.25)
                                    & (source_boundary <= points.max(0)+cell*.25),axis=1)]
    if len(nearby) >= 3:
        planar = nearby-nearby.mean(0)
        planar -= (planar @ normal)[:,None]*normal
        _,_,axes = np.linalg.svd(planar,full_matrices=False)
        if np.prod(np.ptp(planar @ axes[:2].T,axis=0)) > cell*cell*.005:
            return None
    samples = [centre]
    for a,b in zip(points,np.roll(points,-1,axis=0)):
        for i in range(1,11):
            for j in range(11-i):
                samples.append((centre*(11-i-j)+a*i+b*j)/11)
    for point in samples:
        hit,n,_,distance = query._tree.find_nearest(tuple(point))
        if hit is None or distance > cell*.06 or np.dot(n,normal) < .94:
            return None
    return points,normal


def _stitch(mesh, loop, cell, normal, query, depth=0):
    edges = {e for v in loop for e in v.link_edges}
    before = len(mesh.verts)
    bmesh.ops.remove_doubles(mesh,verts=loop,dist=cell*.03)
    bmesh.ops.dissolve_degenerate(mesh,edges=[e for e in edges if e.is_valid],dist=cell*.03)
    splits = 0
    for _ in range(24):
        found = False
        valid = [v for v in loop if v.is_valid]
        boundary = [e for v in valid for e in v.link_edges if e.is_boundary]
        for v in valid:
            for e in boundary:
                if v in e.verts:
                    continue
                origin = e.verts[0].co
                direction = e.verts[1].co-origin
                if direction.length < cell*.01:
                    continue
                t = (v.co-origin).dot(direction)/direction.length_squared
                if not .001 < t < .999 or (origin+t*direction-v.co).length > cell*.03:
                    continue
                _,new = bmesh.utils.edge_split(e,e.verts[0],t)
                bmesh.ops.weld_verts(mesh,targetmap={new:v})
                splits += 1
                found = True
                break
            if found:
                break
        if not found:
            break
    # Welding retraced rims can expose a minute internal flap. Remove only that
    # flap; a substantial face or an unrelated nonmanifold edge aborts the trial.
    removed = 0
    local = {e for v in loop if v.is_valid for e in v.link_edges}
    for edge in list(local):
        while edge.is_valid and len(edge.link_faces) > 2:
            smallest = min(edge.link_faces,key=lambda f:f.calc_area())
            if smallest.calc_area() > cell*cell*.01:
                return None
            bmesh.ops.delete(mesh,geom=[smallest],context='FACES_ONLY')
            removed += 1
    mesh.normal_update()
    faces = {f for v in loop if v.is_valid for f in v.link_faces}
    folded = []
    for face in faces:
        if face.calc_area() > cell*cell*.01:
            continue
        _,source_normal,_,distance = query._tree.find_nearest(face.calc_center_median())
        backwards = (source_normal is not None and distance < cell*.06
                     and np.dot(source_normal,normal) > .94
                     and face.normal.dot(source_normal) < 0)
        # On a scan seam the nearest triangle may itself be an internal sliver.
        # Require a nearby correctly oriented scan surface before removing the
        # collapsed strip; do not infer support from that sliver's normal.
        collapsed = False
        if np.dot(face.normal,normal) < .5:
            hits = query._tree.find_nearest_range(face.calc_center_median(),cell*.006)
            collapsed = any(np.dot(hit_normal,normal) > .94 for _,hit_normal,_,_ in hits)
        if backwards or collapsed:
            folded.append(face)
    if folded:
        edges = {e for f in folded for e in f.edges}
        vertices = {v for f in folded for v in f.verts}
        bmesh.ops.delete(mesh,geom=folded,context='FACES_ONLY')
        removed += len(folded)
        for edge in edges:
            if edge.is_valid and not edge.link_faces:
                bmesh.ops.delete(mesh,geom=[edge],context='EDGES')
        for vertex in vertices:
            if vertex.is_valid and not vertex.link_edges:
                bmesh.ops.delete(mesh,geom=[vertex],context='VERTS')
        # The removed flap can leave two retraced curved edge chains. Stitch
        # their intermediate vertices too, rather than filling the same folded
        # polygon back in and reproducing its inverted face.
        if depth < 2:
            remaining = list({v for v in [*loop,*vertices] if v.is_valid})
            extra = _stitch(mesh,remaining,cell,normal,query,depth+1)
            if extra is None:
                return None
            splits += extra['stitched_edges']
            removed += extra['removed_flaps']
    return dict(welded_vertices=before-len(mesh.verts),stitched_edges=splits,removed_flaps=removed)


def _closed_flap(face, query, cell):
    """Qualify a minute inverted strip inside an otherwise planar surface."""
    if face.calc_area() > cell*cell*.01 or not all(e.is_manifold for e in face.edges):
        return None
    centre = face.calc_center_median()
    _,normal,_,distance = query._tree.find_nearest(centre)
    if normal is None or distance > cell*.006 or face.normal.dot(normal) >= 0:
        return None
    neighbours = {f for v in face.verts for f in v.link_faces
                  if f is not face and f.calc_area() > cell*cell*.05}
    if not neighbours or any(f.normal.dot(normal) < .94 for f in neighbours):
        return None
    points = np.array([v.co for v in face.verts],dtype=float)
    if np.linalg.norm(np.ptp(points,axis=0)) > 2*cell:
        return None
    # Dense support across the collapsed strip is required, including its ends.
    # A genuine opening, opposite sheet or tall fold cannot qualify.
    for a,b in zip(points,np.roll(points,-1,axis=0)):
        for t in np.linspace(0,1,11):
            hits = query._tree.find_nearest_range(tuple(a*(1-t)+b*t),cell*.006)
            if not any(np.dot(n,normal) > .94 for _,n,_,_ in hits):
                return None
    return points,np.array(normal)


def repair_supported_seams(obj, source, query, target_faces, loops_of):
    report = dict(patches=0,caps=0,welded_vertices=0,stitched_edges=0,removed_flaps=0)
    cell = float(np.sqrt(face_areas(source).sum()/max(target_faces,1)))
    if cell <= 0:
        return report
    topology = edge_topology(source)
    source_boundary = source.positions[np.unique(topology.edges[topology.boundary])]
    current = bmesh.new()
    current.from_mesh(obj.data)
    current.normal_update()
    tried = []
    try:
        for _ in range(16):
            candidate = None
            for loop in loops_of([e for e in current.edges if e.is_boundary]):
                centre = np.array([v.co for v in loop],dtype=float).mean(0)
                if any(np.linalg.norm(centre-p) < cell*.25 for p in tried):
                    continue
                support = _supported_loop(current,loop,source_boundary,query,cell)
                if support is not None:
                    candidate = loop,support,centre,False
                    break
            if candidate is None:
                for face in current.faces:
                    centre = np.array(face.calc_center_median())
                    if any(np.linalg.norm(centre-p) < cell*.25 for p in tried):
                        continue
                    support = _closed_flap(face,query,cell)
                    if support is not None:
                        candidate = list(face.verts),support,centre,True
                        break
            if candidate is None:
                break
            loop,(points,normal),centre,closed = candidate
            tried.append(centre)
            current.verts.index_update()
            indices = [v.index for v in loop]
            before = _arrays(current)
            health = inspect_topology(before)
            trial = current.copy()
            accepted = False
            try:
                trial.verts.ensure_lookup_table()
                changes = _stitch(trial,[trial.verts[i] for i in indices],cell,normal,query)
                if changes is None:
                    continue
                caps = 0
                trial.normal_update()
                for rim in loops_of([e for e in trial.edges if e.is_boundary]):
                    p = np.array([v.co for v in rim],dtype=float)
                    if not np.all((p >= points.min(0)-cell*.5) & (p <= points.max(0)+cell*.5)):
                        continue
                    support = _supported_loop(trial,rim,source_boundary,query,cell)
                    if support is None:
                        continue
                    p,normal = support
                    edge = trial.edges.get((rim[0],rim[1]))
                    if edge.link_loops[0].vert is rim[0]:
                        rim = list(reversed(rim));p = p[::-1]
                    turns = np.cross(np.roll(p,-1,axis=0)-p,np.roll(p,-2,axis=0)-np.roll(p,-1,axis=0)) @ normal
                    if turns.min() < -cell*cell*.005:
                        continue
                    if any(trial.edges.get((a,b)).link_loops[0].vert is a
                           for a,b in zip(rim,rim[1:]+rim[:1])):
                        continue
                    try:
                        trial.faces.new(rim)
                    except ValueError:
                        continue
                    caps += 1
                if not caps and not (closed and changes['removed_flaps']):
                    continue
                trial.normal_update()
                after = _arrays(trial)
                after_health = inspect_topology(after)
                if closed:
                    # Removing a closed folded strip must preserve every actual
                    # boundary, not merely keep the global boundary count equal.
                    def boundary_segments(arrays):
                        topology = edge_topology(arrays)
                        return {tuple(sorted(tuple(arrays.positions[v]) for v in edge))
                                for edge in topology.edges[topology.boundary]}
                    if boundary_segments(before) != boundary_segments(after):
                        continue
                if (after_health.non_manifold_edges > health.non_manifold_edges
                    or after_health.flipped_faces > health.flipped_faces
                    or after_health.degenerate_faces > health.degenerate_faces
                    or after_health.skinny_faces > health.skinny_faces):
                    continue
                # Local stitching must not trade the hole for lost scan coverage.
                samples = sample_surface(source)
                before_distance = SurfaceQuery(before).nearest_distances(samples)
                after_distance = SurfaceQuery(after).nearest_distances(samples)
                if np.max(after_distance-before_distance) > cell*.01:
                    continue
                current.free()
                current = trial
                accepted = True
                report['patches'] += 1
                report['caps'] += caps
                for key,value in changes.items():
                    report[key] += value
            finally:
                if not accepted:
                    trial.free()
        if report['patches']:
            current.to_mesh(obj.data)
            obj.data.update()
        return report
    finally:
        current.free()
