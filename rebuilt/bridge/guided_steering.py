"""Steer an accepted Auto mesh without changing its connectivity or boundaries.

This module owns only arrays and private mathutils trees; it never reads bpy data
and can run on the job worker. All proposed motion is checked against the same
unguided reference, not against the preceding guided attempt.
"""
import time
import numpy as np
from mathutils import Vector

from .bvh import SurfaceQuery, sample_surface
from ..core.types import MeshArrays
from ..core.analysis.mesh_graph import edge_topology, face_areas, face_normals, fan_triangles, manifold_edges
from ..core.qa.topology import skinny_face_mask


class _UnsafeMotion(ValueError):
    def __init__(self, message, vertices):
        super().__init__(message)
        self.vertices = np.unique(vertices)


class ProgressSlice:
    def __init__(self, parent, start, span):
        self.parent, self.start, self.span = parent, start, span

    @property
    def cancelled(self):
        return self.parent.cancelled

    def check_cancelled(self):
        self.parent.check_cancelled()

    def update(self, fraction, message=''):
        self.parent.update(self.start+self.span*fraction, message)


class _MotionGuard:
    def __init__(self, reference, cell):
        self.mesh, self.cell = reference, cell
        self.corners = reference.corner_verts
        self.faces = np.repeat(np.arange(reference.num_faces), reference.face_sizes)
        self.next = np.arange(reference.num_corners)+1
        self.next[reference.face_starts+reference.face_sizes-1] = reference.face_starts
        self.normal = face_normals(reference)
        self.signed = self.corner_areas(reference.positions)
        self.valid = self.signed > cell*cell*.001
        self.manifold = manifold_edges(reference, self.normal)
        self.skinny = skinny_face_mask(reference)
        self.quads = reference.corner_verts[np.repeat(reference.face_sizes == 4, reference.face_sizes)].reshape(-1, 4)
        self.quad_faces = np.flatnonzero(reference.face_sizes == 4)
        self.angle, self.aspect = self.quad_quality(reference.positions)

    def quad_quality(self, positions):
        points = positions[self.quads]
        edges = np.roll(points, -1, axis=1)-points
        lengths = np.linalg.norm(edges, axis=2)
        sine = np.linalg.norm(np.cross(edges, np.roll(edges, -1, axis=1)), axis=2)
        sine /= np.maximum(lengths*np.roll(lengths, -1, axis=1), 1e-30)
        return sine.min(1), lengths.max(1)/np.maximum(lengths.min(1), 1e-30)

    def arrays(self, positions):
        m = self.mesh
        return MeshArrays(positions, m.corner_verts, m.face_starts, m.face_sizes)

    def corner_areas(self, positions):
        a = positions[self.corners[self.next]]-positions[self.corners]
        b = positions[self.corners[self.next[self.next]]]-positions[self.corners[self.next]]
        return np.sum(np.cross(a, b)*self.normal[self.faces], axis=1)

    def bad_faces(self, positions):
        bad = np.zeros(self.mesh.num_faces, bool)
        signs = self.corner_areas(positions)
        bad[np.unique(self.faces[self.valid & (signs < self.signed*.25)])] = True
        trial = self.arrays(positions)
        normals = face_normals(trial)
        pairs = self.manifold.faces
        angles = np.arccos(np.clip(np.sum(normals[pairs[:,0]]*normals[pairs[:,1]], axis=1), -1, 1))
        folded = angles > np.maximum(self.manifold.dihedral+1e-6, 2.4)
        bad[pairs[folded].ravel()] = True
        bad |= skinny_face_mask(trial) & ~self.skinny
        angle, aspect = self.quad_quality(positions)
        distorted = (angle < np.minimum(self.angle*.85, .8)-1e-6)
        distorted |= aspect > np.maximum(self.aspect*1.15, 2.)+1e-6
        bad[self.quad_faces[distorted]] = True
        return bad


def _field(reference, payload, query):
    points = reference.positions.astype(float)
    edges = edge_topology(reference).edges
    normals = np.array([query._tree.find_nearest(Vector(p))[1] for p in points])
    n = normals[edges].mean(1)
    n /= np.maximum(np.linalg.norm(n, axis=1)[:,None], 1e-30)
    u = points[edges[:,1]]-points[edges[:,0]]
    u -= n*np.sum(u*n, axis=1)[:,None]
    u /= np.maximum(np.linalg.norm(u, axis=1)[:,None], 1e-30)
    v = np.cross(n, u)
    samples = np.asarray(payload['samples'], dtype=float).copy()
    if not len(samples):
        raise ValueError('Draw at least one guide, or turn off Flow Guides.')
    samples[:,:3] = samples[:,:3]*payload['scale']+payload['center']
    samples[:,9] *= payload['scale']
    inverse = np.asarray(payload['inverse_matrix'], dtype=float)
    world = np.linalg.inv(inverse)
    mid = points[edges].mean(1) @ world[:3,:3].T+world[:3,3]
    world_normal = n @ inverse[:3,:3]
    world_normal /= np.maximum(np.linalg.norm(world_normal, axis=1)[:,None], 1e-30)
    cx, cy, total = (np.zeros(len(edges)) for _ in range(3))
    for stroke in np.unique(samples[:,-1]):
        rows = samples[samples[:,-1] == stroke]
        for first in range(0, len(edges), 256):
            sl = slice(first, first+256)
            distance = np.linalg.norm(mid[sl,None,:]-rows[None,:,:3], axis=2)/rows[None,:,9]
            distance[world_normal[sl] @ rows[:,6:9].T < .5] = np.inf
            index = distance.argmin(1)
            nearest = distance[np.arange(len(index)), index]
            weight = np.maximum(0, 1-nearest**2)**2
            direction = rows[index,3:6] @ inverse[:3,:3].T
            direction -= n[sl]*np.sum(direction*n[sl], axis=1)[:,None]
            angle = np.arctan2(np.sum(direction*v[sl], axis=1), np.sum(direction*u[sl], axis=1))
            cx[sl] += weight*np.cos(4*angle)
            cy[sl] += weight*np.sin(4*angle)
            total[sl] += weight
    agreement = np.hypot(cx, cy)/np.maximum(total, 1e-30)
    weight = np.minimum(1, total)*agreement
    weight[agreement < .25] = 0
    angle = np.arctan2(cy, cx)/4
    target = u*np.cos(angle)[:,None]+v*np.sin(angle)[:,None]
    return edges, normals, n, target, weight


def _crossings(mesh, changed_faces):
    triangles, face_ids = fan_triangles(mesh)
    positions = mesh.positions.astype(float)
    # Blender's BVH triangle overlap omits coplanar intersections. Use AABB
    # candidates so planar folds and triangles sharing one vertex are tested too.
    corners = positions[triangles]
    minimum, maximum = corners.min(1), corners.max(1)
    order = np.argsort(minimum[:,0])
    sorted_x = minimum[order,0]
    pairs = set()
    for a in np.flatnonzero(changed_faces[face_ids]):
        possible = order[:np.searchsorted(sorted_x,maximum[a,0],side='right')]
        overlap = np.all(maximum[possible]>=minimum[a],axis=1) & np.all(minimum[possible]<=maximum[a],axis=1)
        pairs.update((min(a,b),max(a,b)) for b in possible[overlap] if a!=b)
    result = set()
    for a, b in pairs:
        common = set(triangles[a]) & set(triangles[b])
        # Edge neighbours are covered by the corner and dihedral guards. A
        # common vertex alone does not prevent the remaining edges crossing.
        if len(common) >= 2:
            continue
        one, two = positions[triangles[a]], positions[triangles[b]]
        n = np.cross(one[1]-one[0], one[2]-one[0])
        length = np.linalg.norm(n)
        if length < 1e-20:
            continue
        n /= length
        scale = max(np.ptp(one, axis=0).max(), np.ptp(two, axis=0).max(), 1e-20)
        if np.max(abs((two-one[0]) @ n)) < scale*1e-7:
            axes = [i for i in range(3) if i != int(abs(n).argmax())]
            first, second = one[:,axes], two[:,axes]
            def cross(u, v):
                return u[0]*v[1]-u[1]*v[0]
            epsilon = scale*scale*1e-10
            def inside(point, triangle):
                signs = [cross(q-p,point-p) for p,q in zip(triangle,np.roll(triangle,-1,axis=0))]
                return min(signs)>epsilon or max(signs)<-epsilon
            overlap = any(inside(p,second) for p in [*first,first.mean(0)])
            overlap |= any(inside(p,first) for p in [*second,second.mean(0)])
            if not overlap:
                for p,q in zip(first,np.roll(first,-1,axis=0)):
                    for x,y in zip(second,np.roll(second,-1,axis=0)):
                        left,right=cross(q-p,x-p),cross(q-p,y-p)
                        bottom,top=cross(y-x,p-x),cross(y-x,q-x)
                        if left*right < -epsilon*epsilon and bottom*top < -epsilon*epsilon:
                            overlap=True
            if overlap:
                result.add((a,b))
            continue
        found = False
        # Use float64 in a local unit frame: float32 ray hits on a model far
        # from the origin can turn a shared corner into a false crossing.
        local_one = (one-one[0])/scale
        local_two = (two-one[0])/scale
        local_common = [(positions[i]-one[0])/scale for i in common]
        for source, target in ((local_one,local_two),(local_two,local_one)):
            for start, end in zip(source, np.roll(source, -1, axis=0)):
                direction = end-start
                edge_one, edge_two = target[1]-target[0], target[2]-target[0]
                cross = np.cross(direction,edge_two)
                determinant = float(edge_one @ cross)
                if abs(determinant) < 1e-14:
                    continue
                offset = start-target[0]
                u = float(offset @ cross)/determinant
                perpendicular = np.cross(offset,edge_one)
                v = float(direction @ perpendicular)/determinant
                parameter = float(edge_two @ perpendicular)/determinant
                if u < -1e-10 or v < -1e-10 or u+v > 1+1e-10 or not -1e-10 <= parameter <= 1+1e-10:
                    continue
                hit = start+parameter*direction
                if any(np.linalg.norm(hit-point) < 1e-8 for point in local_common):
                    continue
                found = True
                break
            if found:
                break
        if found:
            result.add((a,b))
    return result


def verify_motion(reference, candidate, source, target_faces, progress=None):
    """Reject structural changes and measured geometric degradation."""
    for field in ('corner_verts','face_starts','face_sizes'):
        if not np.array_equal(getattr(reference,field), getattr(candidate,field)):
            raise ValueError('Guide motion changed mesh connectivity.')
    if candidate.positions.shape != reference.positions.shape or not np.isfinite(candidate.positions).all():
        raise ValueError('Guide motion returned invalid vertices.')
    topology = edge_topology(reference)
    boundary = np.unique(topology.edges[topology.boundary])
    if not np.array_equal(reference.positions[boundary], candidate.positions[boundary]):
        raise ValueError('Guide motion changed a reference boundary.')
    cell = float(np.sqrt(face_areas(source).sum()/max(target_faces,1)))
    guard = _MotionGuard(reference, cell)
    bad = guard.bad_faces(candidate.positions)
    if bad.any():
        raise _UnsafeMotion('Guides introduced a folded, collapsed, distorted or thin face.',
                            candidate.corner_verts[np.repeat(bad, candidate.face_sizes)])
    moved = np.linalg.norm(candidate.positions-reference.positions,axis=1) > cell*1e-7
    changed = np.zeros(reference.num_faces,bool)
    np.logical_or.at(changed, guard.faces, moved[reference.corner_verts])
    if progress:
        progress.check_cancelled()
    crossings = _crossings(candidate, changed)-_crossings(reference, changed)
    if crossings:
        triangles, _ = fan_triangles(candidate)
        raise _UnsafeMotion('Guides introduced a surface intersection.',
                            triangles[np.asarray(list(crossings)).ravel()].ravel())
    # Identical sampling of the source measures lost coverage locally, rather
    # than allowing a global average to conceal a new defect elsewhere.
    samples = sample_surface(source)
    before_query, after_query = SurfaceQuery(reference), SurfaceQuery(candidate)
    maximum = 0.
    unsafe = []
    triangles, _ = fan_triangles(reference)
    tolerance = max(cell*.05, float(np.linalg.norm(np.ptp(source.positions,axis=0)))*1e-6)
    for first in range(0,len(samples),2048):
        if progress:
            progress.check_cancelled()
        part = samples[first:first+2048]
        increase = after_query.nearest_distances(part)-before_query.nearest_distances(part)
        maximum = max(maximum,float(increase.max(initial=0)))
        for point in part[increase > tolerance]:
            for tree in (before_query._tree, after_query._tree):
                unsafe.extend(triangles[tree.find_nearest(Vector(point))[2]])
    if unsafe:
        raise _UnsafeMotion('Guides lost surface coverage compared with AutoRemesher.', unsafe)
    query = SurfaceQuery(source)
    fit_increase = query.nearest_distances(sample_surface(candidate))-query.nearest_distances(sample_surface(reference))
    if float(fit_increase.max(initial=0)) > tolerance:
        indices = np.flatnonzero(fit_increase > tolerance)
        vertices = list(indices[indices < candidate.num_verts])
        for face in indices[indices >= candidate.num_verts]-candidate.num_verts:
            start = candidate.face_starts[face]
            vertices.extend(candidate.corner_verts[start:start+candidate.face_sizes[face]])
        raise _UnsafeMotion('Guides moved the surface away from the AutoRemesher reference.', vertices)
    return dict(reference_topology_preserved=True, boundary_positions_preserved=True,
                quality_checked=True, quad_proportions_preserved=True, new_intersections=0,
                maximum_coverage_increase=maximum, geometric_tolerance=tolerance)


def steer(reference, source, payload, target_faces, strength, progress):
    progress.check_cancelled()
    started = time.monotonic()
    cell = float(np.sqrt(face_areas(source).sum()/max(target_faces,1)))
    query = SurfaceQuery(source)
    edges, normals, edge_normal, target, weight = _field(reference, payload, query)
    active = weight > 1e-8
    if not active.any():
        raise ValueError('No guide reaches editable surface away from protected features.')
    original = reference.positions.astype(float)
    points = original.copy()
    guard = _MotionGuard(reference, cell)
    topology = edge_topology(reference)
    fixed = np.zeros(reference.num_verts,bool)
    fixed[np.unique(topology.edges[topology.boundary])] = True
    protected_faces = (reference.face_sizes != 4) | guard.skinny
    protected_faces[np.unique(guard.faces[~guard.valid])] = True
    fixed[reference.corner_verts[np.repeat(protected_faces, reference.face_sizes)]] = True
    fixed[guard.manifold.edges[guard.manifold.dihedral > np.pi/4].ravel()] = True
    degree = np.maximum(np.bincount(edges.ravel(),minlength=len(points)),1)
    length = np.linalg.norm(points[edges[:,1]]-points[edges[:,0]],axis=1)
    perpendicular = np.cross(edge_normal,target)
    original_tangent = original[edges[:,1]]-original[edges[:,0]]
    original_tangent -= edge_normal*np.sum(original_tangent*edge_normal,axis=1)[:,None]
    original_length = np.linalg.norm(original_tangent,axis=1)
    quads = guard.quads
    diagonals = np.concatenate([quads[:,[0,2]],quads[:,[1,3]]])
    springs = np.concatenate([edges,diagonals])
    rest = np.linalg.norm(original[springs[:,1]]-original[springs[:,0]],axis=1)

    def error(p):
        e = p[edges[:,1]]-p[edges[:,0]]
        e -= edge_normal*np.sum(e*edge_normal,axis=1)[:,None]
        e /= np.maximum(np.linalg.norm(e,axis=1)[:,None],1e-30)
        angle = np.arccos(np.clip(abs(np.sum(e*target,axis=1)),0,1))
        angle = np.minimum(angle,np.pi/2-angle)
        return float(np.average(angle[active],weights=length[active]*weight[active])*180/np.pi)

    initial = previous = error(points)
    if initial <= .1:
        quality = verify_motion(reference, reference, source, target_faces, progress)
        quality.update(adapter='reference-steering-1', guides=payload['guide_count'],
                       alignment_before_degrees=initial, alignment_after_degrees=initial,
                       already_aligned=True, iterations=0, motion_fraction=0.,
                       constrained_triangles=int(active.sum()), elapsed_seconds=time.monotonic()-started)
        progress.update(1., 'AutoRemesher already follows these guides')
        return reference.copy(), quality
    iterations = 0
    for iteration in range(120):
        progress.update(.1+.65*iteration/120,'Orienting quads on the AutoRemesher reference')
        e = points[edges[:,1]]-points[edges[:,0]]
        tangent = e-edge_normal*np.sum(e*edge_normal,axis=1)[:,None]
        choose = abs(np.sum(tangent*target,axis=1)) >= abs(np.sum(tangent*perpendicular,axis=1))
        axis = np.where(choose[:,None],target,perpendicular)
        axis *= np.where(np.sum(tangent*axis,axis=1)<0,-1,1)[:,None]
        force = (axis*original_length[:,None]-tangent)*weight[:,None]*(strength/12.)
        move = np.zeros_like(points)
        np.add.at(move,edges[:,1],force)
        np.add.at(move,edges[:,0],-force)
        # Edge and diagonal rest lengths resist pinching and shearing while the
        # guide field rotates the quads. Using the current length as the target
        # allowed cumulative shrinkage on every iteration.
        current_springs = points[springs[:,1]]-points[springs[:,0]]
        spring_length = np.linalg.norm(current_springs,axis=1)
        spring_force = current_springs*(rest/np.maximum(spring_length,1e-30)-1)[:,None]*.25
        np.add.at(move,springs[:,1],spring_force)
        np.add.at(move,springs[:,0],-spring_force)
        move /= degree[:,None]
        move -= .003*(points-original)
        move -= normals*np.sum(move*normals,axis=1)[:,None]
        move[fixed] = 0
        trial = points+move*.7
        for index in np.flatnonzero(np.linalg.norm(move,axis=1)>cell*1e-7):
            location, normal, _, distance = query._tree.find_nearest(Vector(trial[index]))
            if location is not None and distance < cell*.2 and normal.dot(Vector(normals[index])) > .5:
                trial[index] = location
            else:
                trial[index] = points[index]
        trial[fixed] = original[fixed]
        trial = trial.astype(np.float32).astype(float)
        for attempt in range(24):
            bad = guard.bad_faces(trial)
            if not bad.any():
                break
            vertices = np.unique(reference.corner_verts[np.repeat(bad,reference.face_sizes)])
            trial[vertices] = ((trial[vertices]+points[vertices])*.5 if attempt<12 else points[vertices])
            trial = trial.astype(np.float32).astype(float)
        if bad.any():
            break
        current = error(trial)
        if current > previous+1e-6:
            break
        points, previous = trial,current
        iterations = iteration+1
    progress.update(.8,'Comparing guided surface with AutoRemesher')
    reasons = []
    # Restrict only the unsafe neighbourhood before reducing the entire field.
    # Recheck every time: rolling back one vertex can affect adjacent faces.
    for attempt in range(16):
        candidate = guard.arrays(points.astype(np.float32))
        try:
            verify_motion(reference, candidate, source, target_faces, progress)
            break
        except _UnsafeMotion as exc:
            reasons.append(str(exc))
            points[exc.vertices] = ((points[exc.vertices]+original[exc.vertices])*.5
                                    if attempt < 12 else original[exc.vertices])
    for fraction in (1., .75, .5, .25, .125):
        candidate = guard.arrays((original+fraction*(points-original)).astype(np.float32))
        final = error(candidate.positions.astype(float))
        if final >= initial-min(.1,initial*.02):
            continue
        try:
            quality = verify_motion(reference,candidate,source,target_faces,progress)
        except ValueError as exc:
            reasons.append(str(exc))
            continue
        quality.update(adapter='reference-steering-1',guides=payload['guide_count'],
                       alignment_before_degrees=initial,alignment_after_degrees=final,
                       iterations=iterations,motion_fraction=fraction,
                       constrained_triangles=int(active.sum()),elapsed_seconds=time.monotonic()-started)
        progress.update(1.,'Guided surface checked against AutoRemesher')
        return candidate,quality
    detail = reasons[0] if reasons else 'The guides cannot improve this layout within the surface safeguards.'
    raise ValueError(detail+' Previous mesh kept; adjust the guides and retry.')
