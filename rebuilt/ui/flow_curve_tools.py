"""Surface-constrained stroke processing for the independent guide editor."""
import math
from mathutils import Vector
from mathutils.kdtree import KDTree


def resample(points, spacing, cyclic=False):
    points = [Vector(p) for p in points]
    if not points:
        return []
    if spacing <= 0:
        raise ValueError('Spacing must be positive')
    if cyclic:
        points.append(points[0])
    result = [points[0].copy()]
    carry = 0.0
    for a, b in zip(points, points[1:]):
        delta = b-a
        distance = delta.length
        if distance < 1e-12:
            continue
        offset = spacing-carry
        while offset <= distance:
            result.append(a+delta*(offset/distance))
            offset += spacing
        carry = distance-(offset-spacing)
    if not cyclic:
        if (result[-1]-points[-1]).length > spacing*0.1:
            result.append(points[-1].copy())
        else:
            result[-1]=points[-1].copy()
    if cyclic and len(result)>3 and (result[-1]-result[0]).length < spacing*0.5:
        result.pop()
    return result


def project_points(points, bvh, scale, max_distance=None):
    result = []
    for p in points:
        hit, normal, _, distance = bvh.find_nearest(Vector(p))
        if hit is None or (max_distance is not None and distance > max_distance):
            raise ValueError('The guide cannot reach the source surface')
        result.append(hit+normal*scale*.001)
    return result


def smooth_stroke(points, bvh, scale, amount=.65, cyclic=False):
    points = resample(points, max(scale*.0025,1e-8), cyclic)
    if len(points)<3:
        return project_points(points,bvh,scale)
    amount = max(0.,min(1.,amount))
    if amount:
        sigma=.5+amount*5
        radius=math.ceil(sigma*3)
        kernel=[(j,math.exp(-.5*(j/sigma)**2)) for j in range(-radius,radius+1)]
        output=[]
        for i,p in enumerate(points):
            total=0.;average=Vector((0,0,0))
            for j,weight in kernel:
                index=i+j
                if cyclic:index%=len(points)
                elif index<0 or index>=len(points):continue
                average+=points[index]*weight;total+=weight
            fade=1. if cyclic else min(1.,i/(sigma*2),(len(points)-1-i)/(sigma*2))
            output.append(p.lerp(average/total,fade))
        points=output
        points=project_points(points,bvh,scale,max_distance=scale*.1)
    return project_points(resample(points,max(scale*.004,1e-8),cyclic),bvh,scale)


def surface_line(start, end, raycast, pixel_spacing=6):
    start,end=Vector(start),Vector(end)
    steps=min(256,max(2,math.ceil((end-start).length/pixel_spacing)))
    result=[]
    for i in range(steps+1):
        p=raycast(start.lerp(end,i/steps))
        if p is None:
            raise ValueError('Keep the whole line on the visible surface')
        result.append(Vector(p))
    return result


def arc_lengths(points, cyclic=False):
    lengths=[0.]
    for a,b in zip(points,points[1:]):lengths.append(lengths[-1]+(b-a).length)
    total=lengths[-1]+((points[-1]-points[0]).length if cyclic else 0.)
    return lengths,total


def deform_stroke(points, anchor, delta, radius, bvh, scale, cyclic=False):
    lengths,total=arc_lengths(points,cyclic)
    result=[]
    for p,t in zip(points,lengths):
        distance=abs(t-anchor)
        if cyclic:distance=min(distance,total-distance)
        weight=max(0.,1-(distance/max(radius,1e-9))**2)**2
        result.append(p+delta*weight)
    return project_points(result,bvh,scale,max_distance=scale*.15)


def mirror_stroke(points, matrix, axis, bvh, scale):
    index={'X':0,'Y':1,'Z':2}[axis]
    inverse=matrix.inverted()
    mirrored=[]
    for p in points:
        local=inverse@p
        local[index]*=-1
        mirrored.append(matrix@local)
    return project_points(mirrored,bvh,scale,max_distance=scale*.12)


def same_path(a,b,scale):
    if not a or not b:return False
    def distance(source,target):
        tree=KDTree(len(target))
        for i,p in enumerate(target):tree.insert(p,i)
        tree.balance()
        return max(tree.find(p)[2] for p in source)
    tolerance=max(scale*.0015, min(arc_lengths(a)[1]/len(a),arc_lengths(b)[1]/len(b))*.55)
    return max(distance(a,b),distance(b,a))<tolerance
