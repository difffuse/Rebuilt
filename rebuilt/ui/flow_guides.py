"""Surface guide authoring for Rebuilt Manual retopology. Derived from Flow Guides Lab 0.2.4."""
import bpy
import bmesh
import gpu
import blf
import json
import math
import os
import subprocess
import tempfile
import time
import uuid
import textwrap
from pathlib import Path
from collections import Counter, defaultdict
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from bpy.props import PointerProperty, IntProperty, FloatProperty, BoolProperty, StringProperty, EnumProperty

from .flow_curve_tools import resample, smooth_stroke, surface_line, project_points, mirror_stroke, same_path, arc_lengths, deform_stroke

_DRAW = None
_DRAW_HANDLES = []


def settings(context=None):
    source = resolve_source(context or bpy.context)
    return source.rebuilt_flow if source is not None else None


def source_ok(obj):
    return obj is not None and obj.type == 'MESH' and obj.get('rebuilt_role', 'source') == 'source'


def mesh_snapshot(obj, depsgraph, *, prepared_mesh=None):
    evaluated = obj.evaluated_get(depsgraph) if prepared_mesh is None else None
    mesh = evaluated.to_mesh() if evaluated is not None else prepared_mesh
    try:
        matrix = obj.matrix_world
        vertices = [matrix @ v.co for v in mesh.vertices]
        mesh.calc_loop_triangles()
        faces = [tuple(t.vertices) for t in mesh.loop_triangles]
        if matrix.determinant() < 0:
            faces = [tuple(reversed(f)) for f in faces]
    finally:
        if evaluated is not None:
            evaluated.to_mesh_clear()
    if not vertices or not faces:
        raise ValueError("The source has no surface faces")
    low = Vector(tuple(min(v[k] for v in vertices) for k in range(3)))
    high = Vector(tuple(max(v[k] for v in vertices) for k in range(3)))
    scale = (high-low).length
    if not math.isfinite(scale) or scale < 1e-9:
        raise ValueError("The source has invalid or zero size")
    bvh = BVHTree.FromPolygons(vertices, faces, all_triangles=True)
    return vertices, faces, bvh, (low+high)*0.5, scale


def valid_guide(obj):
    try:
        return (obj is not None and obj.type == 'CURVE' and obj.get('rebuilt_flow_guide')
                and len(obj.data.splines) > 0 and obj.data.splines[0].type == 'POLY'
                and len(obj.data.splines[0].points) >= 3)
    except ReferenceError:
        return False


def guides(s):
    return [o for o in s.collection.objects if valid_guide(o) and o.parent == s.source] if s and s.collection else []


def ensure_collection(s, scene):
    if s.collection is None or any(o.parent != s.source for o in s.collection.objects if valid_guide(o)):
        # Object copies must not share another model's authored curves.
        collection = bpy.data.collections.new("Rebuilt Guides - " + s.source.name)
        scene.collection.children.link(collection)
        collection['rebuilt_flow_owner'] = s.source.name
        s.collection = collection
    return s.collection


def guide_material():
    mat = bpy.data.materials.get("Rebuilt Guides - Yellow")
    if mat is None:
        mat = bpy.data.materials.new("Rebuilt Guides - Yellow")
        mat.diffuse_color = (1.0, 0.65, 0.015, 1.0)
        mat.use_nodes = True
        node = mat.node_tree.nodes.get('Principled BSDF')
        node.inputs['Base Color'].default_value = mat.diffuse_color
        node.inputs['Emission Color'].default_value = mat.diffuse_color
        node.inputs['Emission Strength'].default_value = 0.5
    return mat


def create_guide(s, points, cyclic=False, scene=None, scale=None):
    scene = scene or bpy.context.scene
    if len(points) < 3:
        raise ValueError("Draw a longer guide")
    scale = scale or s.source.dimensions.length
    curve = bpy.data.curves.new("Flow Guide", 'CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = 1
    curve.bevel_depth = max(scale*0.0008, 1e-7)
    curve.bevel_resolution = 1
    spline = curve.splines.new('POLY')
    spline.points.add(len(points)-1)
    inverse = s.source.matrix_world.inverted()
    for p, co in zip(spline.points, points):
        p.co = (*(inverse @ Vector(co)), 1.0)
    spline.use_cyclic_u = cyclic
    obj = bpy.data.objects.new("Flow Guide %02d" % (len(guides(s))+1), curve)
    ensure_collection(s, scene).objects.link(obj)
    obj.parent = s.source
    obj['rebuilt_flow_guide'] = True
    obj.hide_render = True
    obj['rebuilt_flow_id'] = uuid.uuid4().hex
    obj.color = (1.0, 0.65, 0.015, 1)
    curve.materials.append(guide_material())
    s.message = "Guide added. Draw more or generate the mesh."
    s.details='';s.warning=''
    return obj


def guide_points(obj):
    if not valid_guide(obj):raise ValueError('The guide has no usable points')
    return [obj.matrix_world @ p.co.xyz for p in obj.data.splines[0].points]


def replace_points(obj, points, cyclic=None):
    if len(points)<3:raise ValueError('A guide needs at least three points')
    inverse=obj.matrix_world.inverted()
    local=[inverse@Vector(co) for co in points]
    if any(not all(math.isfinite(v) for v in co) for co in local):
        raise ValueError('Guide coordinates must be finite')
    previous=list(obj.data.splines)
    closed=(previous[0].use_cyclic_u if previous else False) if cyclic is None else cyclic
    spline=obj.data.splines.new('POLY')
    try:
        spline.points.add(len(local)-1)
        for p,co in zip(spline.points,local):p.co=(*co,1)
        spline.use_cyclic_u=closed
    except Exception:
        obj.data.splines.remove(spline)
        raise
    for old in previous:obj.data.splines.remove(old)
    obj.data.update_tag()


def paired_guide(s,obj):
    pair=obj.get('rebuilt_flow_pair')
    if not pair:return None
    return next((g for g in guides(s) if g!=obj and g.get('rebuilt_flow_pair')==pair),None)


def sync_mirror(s,obj,bvh,scale):
    if s.symmetry=='OFF':return None
    points=guide_points(obj)
    mirrored=mirror_stroke(points,s.source.matrix_world,s.symmetry,bvh,scale)
    other=paired_guide(s,obj)
    if other and obj.get('rebuilt_flow_mirror_axis')!=s.symmetry:
        del other['rebuilt_flow_pair'];del obj['rebuilt_flow_pair'];other=None
    if other is None and same_path(points,mirrored,scale):return None
    if other is None:
        other=create_guide(s,mirrored,obj.data.splines[0].use_cyclic_u,scale=scale)
        other.name=obj.name+' - Mirror'
        pair=uuid.uuid4().hex;obj['rebuilt_flow_pair']=pair;other['rebuilt_flow_pair']=pair
    else:replace_points(other,mirrored,obj.data.splines[0].use_cyclic_u)
    obj['rebuilt_flow_mirror_axis']=s.symmetry;other['rebuilt_flow_mirror_axis']=s.symmetry
    return other


def delete_guide(s,obj):
    other=paired_guide(s,obj) if s.symmetry!='OFF' else None
    for guide in [g for g in (other,obj) if g is not None]:
        data=guide.data;bpy.data.objects.remove(guide,do_unlink=True)
        if data.users==0:bpy.data.curves.remove(data)


def edit_guide(s, obj, action, bvh, scale):
    """Apply one explicit action to this source's guide, never its mesh."""
    if obj not in guides(s):
        raise ValueError('Choose a guide belonging to this model')
    if action == 'DELETE':
        delete_guide(s, obj)
        select_source(s)
        s.message = 'Guide deleted. Retopologise when ready.'
    elif action == 'SMOOTH':
        if s.edit_smoothness <= 0:
            return
        points = smooth_stroke(guide_points(obj), bvh, scale, s.edit_smoothness,
                               obj.data.splines[0].use_cyclic_u)
        replace_points(obj, points)
        sync_mirror(s, obj, bvh, scale)
        s.message = 'Guide smoothed. Retopologise when ready.'
    else:
        raise ValueError('Unknown guide action')
    s.details = ''; s.warning = ''


def capture_guides(s):
    """Snapshot only authored guide data; never the source or generated mesh."""
    return [{
        'name':obj.name, 'data_name':obj.data.name,
        'properties':dict(obj.items()),
        'basis':tuple(tuple(row) for row in obj.matrix_basis),
        'parent_inverse':tuple(tuple(row) for row in obj.matrix_parent_inverse),
        'points':tuple((tuple(p.co),p.radius,p.tilt) for p in obj.data.splines[0].points),
        'cyclic':obj.data.splines[0].use_cyclic_u,
        'bevel':obj.data.bevel_depth, 'bevel_resolution':obj.data.bevel_resolution,
        'materials':tuple(m.name if m else None for m in obj.data.materials),
        'hidden':obj.hide_get(), 'hide_render':obj.hide_render,
        'color':tuple(obj.color),
    } for obj in sorted(guides(s),key=lambda g:g.get('rebuilt_flow_id',g.name))]


def restore_guides(s, snapshot):
    from mathutils import Matrix
    def identity(record):
        return record['properties'].get('rebuilt_flow_id',record['name'])
    wanted={identity(record):record for record in snapshot}
    current={obj.get('rebuilt_flow_id',obj.name):obj for obj in guides(s)}
    for key,obj in current.items():
        if key not in wanted:
            data=obj.data
            bpy.data.objects.remove(obj,do_unlink=True)
            if data.users==0:bpy.data.curves.remove(data)
    for key,record in wanted.items():
        obj=current.get(key)
        if obj is None:
            curve=bpy.data.curves.new(record['data_name'],'CURVE')
            obj=bpy.data.objects.new(record['name'],curve)
            ensure_collection(s,bpy.context.scene).objects.link(obj)
            obj.parent=s.source
        elif obj.data.users>1:
            obj.data=obj.data.copy()
        obj.name=record['name']
        obj.matrix_parent_inverse=Matrix(record['parent_inverse'])
        obj.matrix_basis=Matrix(record['basis'])
        for prop in list(obj.keys()):
            if prop not in record['properties']:del obj[prop]
        for prop,value in record['properties'].items():obj[prop]=value
        curve=obj.data;curve.dimensions='3D';curve.resolution_u=1
        curve.bevel_depth=record['bevel'];curve.bevel_resolution=record['bevel_resolution']
        curve.splines.clear();spline=curve.splines.new('POLY')
        spline.points.add(len(record['points'])-1)
        for point,(co,radius,tilt) in zip(spline.points,record['points']):
            point.co=co;point.radius=radius;point.tilt=tilt
        spline.use_cyclic_u=record['cyclic']
        curve.materials.clear()
        for name in record['materials']:
            material=bpy.data.materials.get(name) if name else None
            curve.materials.append(material)
        obj.hide_render=record['hide_render'];obj.hide_set(record['hidden']);obj.color=record['color']
        curve.update_tag()
    select_source(s)
    bpy.context.view_layer.update()
    s.details='';s.warning=''
    redraw()


class GuideHistory:
    """Bounded, source-local actions while a modal tool owns Blender input."""
    def __init__(self,s):
        self.settings=s
        self.states=[capture_guides(s)]
        self.cursor=0
    def checkpoint(self):
        state=capture_guides(self.settings)
        if state==self.states[self.cursor]:return
        del self.states[self.cursor+1:]
        self.states.append(state)
        if len(self.states)>65:self.states.pop(0)
        self.cursor=len(self.states)-1
    def undo(self):
        if self.cursor==0:return False
        restore_guides(self.settings,self.states[self.cursor-1]);self.cursor-=1
        return True
    def redo(self):
        if self.cursor+1==len(self.states):return False
        restore_guides(self.settings,self.states[self.cursor+1]);self.cursor+=1
        return True


def history_event(operator,event):
    if event.type!='Z' or not(event.ctrl or event.oskey):return False
    if event.value!='PRESS':return True
    if getattr(operator,'drag',None) is not None:
        operator.restore_drag()
        operator.hover=None
        operator.guide_settings.message='Current move cancelled'
    elif getattr(operator,'dragging',False):
        operator.stroke=[];operator.dragging=False
        operator.last_xy=None;operator.start_xy=None
        operator.guide_settings.message='Current stroke cancelled'
    else:
        if hasattr(operator,'hover'):operator.hover=None
        changed=operator.history.redo() if event.shift else operator.history.undo()
        operator.guide_settings.message=('Guide action redone' if event.shift else 'Guide action undone') if changed else 'No more guide actions in this session'
    redraw()
    return True


def editor_ui_event(operator, event):
    """Sidebar clicks belong to its controls unless an active drag owns release."""
    if operator.drag is not None:
        return False
    return any(r.type == 'UI' and r.width > 1
               and r.x <= event.mouse_x < r.x + r.width
               and r.y <= event.mouse_y < r.y + r.height for r in operator.area.regions)


def sample_guides(s, bvh, center, scale):
    samples = []
    radius = scale*s.radius/100
    for stroke, obj in enumerate(guides(s)):
        for spline in obj.data.splines:
            if spline.type != 'POLY':
                raise ValueError("Use the generated poly curves; other spline types are not supported yet")
            raw = [obj.matrix_world @ p.co.xyz for p in spline.points]
            if len(raw)<3:
                continue
            points = resample(raw, radius*0.25, spline.use_cyclic_u)
            projected = []
            for p in points:
                hit, normal, index, distance = bvh.find_nearest(p)
                if hit is None or distance > scale*0.15:
                    raise ValueError("A guide is too far from the source. Move it back onto the surface")
                projected.append((hit, normal))
            for i, (p,n) in enumerate(projected):
                prev = projected[i-1 if i else (-1 if spline.use_cyclic_u else 0)][0]
                nxt = projected[(i+1)%len(projected) if spline.use_cyclic_u else min(i+1,len(projected)-1)][0]
                tangent = nxt-prev
                tangent -= n*tangent.dot(n)
                if tangent.length<1e-10:
                    continue
                tangent.normalize()
                samples.append([*((p-center)/scale), *tangent, *n, radius/scale, stroke])
    if len(samples)>12000:
        raise ValueError("Too many guide samples. Simplify the guides or increase Influence")
    return samples


def set_view(s, result):
    if s.source:
        s.source.hide_set(bool(result and s.output))
    if s.output:
        s.output.hide_set(not result)
    for obj in guides(s):
        obj.hide_set(bool(result))


def redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type=='VIEW_3D':area.tag_redraw()


def release_guide_mode():
    global _DRAW
    _DRAW=None
    while _DRAW_HANDLES:
        handle=_DRAW_HANDLES.pop()
        try:bpy.types.SpaceView3D.draw_handler_remove(handle,'WINDOW')
        except (ReferenceError,ValueError):pass
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type=='VIEW_3D':area.header_text_set(None)
    redraw()


def add_guide_handler(callback,stage):
    def draw():
        try:callback()
        except ReferenceError:release_guide_mode()
    handle=bpy.types.SpaceView3D.draw_handler_add(draw,(),'WINDOW',stage)
    _DRAW_HANDLES.append(handle)
    return handle


def guard_guide_modal(method):
    def modal(self,context,event):
        try:return method(self,context,event)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            try:self.finish()
            except Exception:release_guide_mode()
            message='Guide editing stopped safely. You can draw again. '+str(exc)
            self.guide_settings.message=message
            self.report({'WARNING'},message)
            return {'CANCELLED'}
    return modal


VIEW_KEYS={'NUMPAD_0','NUMPAD_1','NUMPAD_2','NUMPAD_3','NUMPAD_4','NUMPAD_5',
           'NUMPAD_6','NUMPAD_7','NUMPAD_8','NUMPAD_9','NUMPAD_PERIOD',
           'NUMPAD_PLUS','NUMPAD_MINUS','NUMPAD_SLASH','HOME'}


def navigation_bounds(area,region):
    right=region.x+region.width;top=region.y+region.height
    for other in area.regions:
        if other.width<=1 or other.height<=1:continue
        if other.type=='UI' and other.x>region.x+region.width/2:right=min(right,other.x)
        if other.type in {'HEADER','TOOL_HEADER'} and other.y>region.y+region.height/2:top=min(top,other.y)
    return right,top


def navigation_event(operator,context,event):
    if event.type in VIEW_KEYS or event.type in {'MIDDLEMOUSE','WHEELUPMOUSE','WHEELDOWNMOUSE','NDOF_MOTION','TRACKPADPAN','TRACKPADZOOM'} or event.alt:
        return True
    if getattr(operator,'dragging',False) or getattr(operator,'drag',None) is not None:return False
    space=operator.area.spaces.active
    if not space.show_gizmo or not space.show_gizmo_navigate:return False
    right,top=navigation_bounds(operator.area,operator.region)
    scale=context.preferences.system.ui_scale
    size=context.preferences.view.gizmo_size_navigate_v3d
    x,y=event.mouse_x,event.mouse_y
    axis=context.preferences.view.mini_axis_type=='GIZMO'
    if axis and right-(size+20)*scale<=x<=right and top-(size+20)*scale<=y<=top:return True
    offset=(10+size/2)*2.2 if axis else 22.5
    return right-40*scale<=x<=right and top-(offset+135)*scale<=y<=top


def strength_percent_get(settings):
    return (settings.strength - 1.0) * 100.0 / 29.0


def strength_percent_set(settings, value):
    settings.strength = 1.0 + max(0.0, min(100.0, value)) * 29.0 / 100.0


class REBUILT_FLOW_Settings(bpy.types.PropertyGroup):
    @property
    def source(self):
        return self.id_data

    @property
    def output(self):
        from ..pipeline.context import AssetContext, ContextError
        try:
            ctx = AssetContext.for_object(bpy.context.scene, self.source)
            return ctx.artifact('low') if ctx else None
        except ContextError:
            return None

    enabled: BoolProperty(name="Flow Guides", default=False, description="Draw curves to steer quad direction in Manual retopology, not exact edge loops. Your guides stay with this model")
    show_options: BoolProperty(name="Guide Settings", default=False)
    edit_tool: EnumProperty(name="Edit Tool", items=[
        ('MOVE', 'Move', 'Drag a guide along the surface'),
        ('SMOOTH', 'Smooth', 'Click a guide to smooth it on the surface'),
        ('DELETE', 'Delete', 'Click a guide to delete it; the source mesh is never deleted'),
    ], default='MOVE', options={'SKIP_SAVE'})
    edit_smoothness: FloatProperty(name="Amount", default=.5, min=0, max=1,
        subtype='FACTOR', description="Smoothing applied per click; zero leaves the guide unchanged")
    collection: PointerProperty(type=bpy.types.Collection)
    radius: FloatProperty(name="Influence",default=3.0,min=0.5,max=15,subtype='PERCENTAGE',description="Guide influence radius as a percentage of the source bounding-box diagonal")
    strength: FloatProperty(name="Strength",default=12,min=1,max=30,description="How strongly the field follows the guides; very strong constraints can distort surrounding regions")
    strength_percent: FloatProperty(name="Strength", min=0, max=100, subtype='PERCENTAGE', get=strength_percent_get, set=strength_percent_set, description="Guide weight within the supported range: 0% is the minimum and 100% is the maximum. Higher values favor guide directions more strongly and may distort nearby quads")
    stroke_mode: EnumProperty(name="Stroke",items=[('FREEHAND','Freehand','Draw a smooth freehand stroke'),('LINE','Line','Drag a straight line in the current view, projected onto the visible surface')],default='FREEHAND')
    smoothing: FloatProperty(name="Smoothness",default=.65,min=0,max=1,subtype='FACTOR',description="Remove mouse jitter when a freehand stroke is released; existing guides remain unchanged")
    symmetry: EnumProperty(name="Symmetry",items=[('OFF','Off','Edit guides independently'),('X','X','Mirror guides across local X = 0 through the source origin'),('Y','Y','Mirror guides across local Y = 0 through the source origin'),('Z','Z','Mirror guides across local Z = 0 through the source origin')],default='OFF',description="Mirror new and edited guides in source-local coordinates; this does not force a symmetric retopo")
    edit_falloff: FloatProperty(name="Edit Size",default=35,min=3,max=60,subtype='PERCENTAGE',description="Length of the guide influenced when dragging, as a percentage of its total length")
    close_loop: BoolProperty(name="Close Loop",default=False,description="Close each new stroke; returning near its start also closes it automatically")
    message: StringProperty(default="Draw on the model, then press Esc")
    details: StringProperty()
    warning: StringProperty()


class REBUILT_FLOW_OT_draw(bpy.types.Operator):
    bl_idname='rebuilt.flow_draw';bl_label='Draw Guides';bl_options={'REGISTER','UNDO','BLOCKING'}
    @classmethod
    def poll(cls,context):return can_author(context) and _DRAW is None and context.mode=='OBJECT' and context.area and context.area.type=='VIEW_3D'
    def invoke(self,context,event):
        global _DRAW
        self.guide_settings=settings(context)
        try:
            self.region=next(r for r in context.area.regions if r.type=='WINDOW')
            self.area=context.area;self.rv3d=context.space_data.region_3d
            _,_,self.bvh,_,self.scale=mesh_snapshot(settings(context).source,context.evaluated_depsgraph_get())
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        self.line_mode=False
        self.scene=context.scene;self.stroke=[];self.last_xy=None;self.start_xy=None;self.dragging=False;self.count=0
        set_view(settings(context),False)
        self.history=GuideHistory(self.guide_settings)
        self.handler=add_guide_handler(self.preview,'POST_VIEW')
        self.text_handler=add_guide_handler(self.instructions,'POST_PIXEL')
        _DRAW=self
        context.window_manager.modal_handler_add(self)
        self.area.header_text_set('Flow Guides: LMB draw | Shift-drag: straight line | MMB / wheel: navigate | Enter / Esc: done')
        return {'RUNNING_MODAL'}
    def preview(self):
        if bpy.context.area!=self.area or len(self.stroke)<2:return
        shader=gpu.shader.from_builtin('UNIFORM_COLOR')
        batch=batch_for_shader(shader,'LINE_STRIP',{'pos':self.stroke})
        gpu.state.blend_set('ALPHA');gpu.state.depth_test_set('LESS_EQUAL');gpu.state.line_width_set(3)
        shader.bind();shader.uniform_float('color',(1,.7,.02,1));batch.draw(shader)
        gpu.state.line_width_set(1);gpu.state.depth_test_set('NONE');gpu.state.blend_set('NONE')
    def instructions(self):
        if bpy.context.area!=self.area:return
        blf.position(0,24,50,0);blf.size(0,16);blf.color(0,1,.8,.15,1)
        blf.draw(0,'DRAW GUIDES   LMB: draw  |  Shift-drag: line  |  MMB: orbit  |  Enter / Esc: done')
    def ray_hit(self,xy):
        if not(0<=xy[0]<self.region.width and 0<=xy[1]<self.region.height):return None
        origin=view3d_utils.region_2d_to_origin_3d(self.region,self.rv3d,xy)
        direction=view3d_utils.region_2d_to_vector_3d(self.region,self.rv3d,xy)
        hit,normal,_,_=self.bvh.ray_cast(origin,direction)
        return hit+normal*self.scale*.001 if hit is not None else None
    def hit(self,event):
        xy=(event.mouse_x-self.region.x,event.mouse_y-self.region.y)
        return self.ray_hit(xy),xy
    def commit(self):
        s=self.guide_settings
        if len(self.stroke)>=3:
            close=not self.line_mode and (s.close_loop or (self.last_xy is not None and self.start_xy is not None and (Vector(self.last_xy)-Vector(self.start_xy)).length<24 and len(self.stroke)>8))
            try:
                points=project_points(resample(self.stroke,self.scale*.004,False),self.bvh,self.scale) if self.line_mode else smooth_stroke(self.stroke,self.bvh,self.scale,s.smoothing,close)
                if len(points)>=3:
                    obj=create_guide(s,points,close,self.scene,self.scale)
                    # Parent transforms must be evaluated before building the paired guide.
                    bpy.context.view_layer.update()
                    sync_mirror(s,obj,self.bvh,self.scale)
                    for ob in bpy.context.selected_objects:ob.select_set(False)
                    obj.select_set(True);bpy.context.view_layer.objects.active=obj
                    self.count+=1
                    self.history.checkpoint()
                    s.message='Guide added. Press Esc, then Retopologise.'
            except ValueError as exc:s.message=str(exc)
        self.stroke=[];self.dragging=False;self.last_xy=None;self.start_xy=None
    def finish(self):
        release_guide_mode()
        select_source(self.guide_settings)
    @guard_guide_modal
    def modal(self,context,event):
        if _DRAW is not self:return {'FINISHED'}
        if history_event(self,event):return {'RUNNING_MODAL'}
        if event.type in {'ESC','RET','NUMPAD_ENTER'}:
            self.stroke=[];self.finish();return {'FINISHED'}
        if event.type=='RIGHTMOUSE' and event.value=='PRESS':self.stroke=[];self.dragging=False;redraw();return {'RUNNING_MODAL'}
        if navigation_event(self,context,event):
            if self.dragging and event.value=='PRESS':self.commit()
            return {'PASS_THROUGH'}
        if event.type=='LEFTMOUSE' and event.value=='PRESS':
            p,xy=self.hit(event)
            if p is None:return {'RUNNING_MODAL'}
            self.line_mode=self.guide_settings.stroke_mode=='LINE' or event.shift
            self.dragging=True;self.stroke=[p];self.last_xy=xy;self.start_xy=xy
        elif event.type=='MOUSEMOVE' and self.dragging:
            p,xy=self.hit(event)
            if self.line_mode:
                try:
                    self.stroke=surface_line(self.start_xy,xy,self.ray_hit)
                    if any((b-a).length>self.scale*.12 for a,b in zip(self.stroke,self.stroke[1:])):raise ValueError('The line crosses a gap in the surface')
                    self.guide_settings.message='Release to create the line'
                except ValueError as exc:self.stroke=[];self.guide_settings.message=str(exc)
                self.last_xy=xy
            elif p is not None and (Vector(xy)-Vector(self.last_xy)).length>=4:
                if (p-self.stroke[-1]).length>self.scale*.12:self.commit();self.dragging=True;self.start_xy=xy
                self.stroke.append(p);self.last_xy=xy
        elif event.type=='LEFTMOUSE' and event.value=='RELEASE':self.commit()
        redraw()
        if event.type=='MOUSEMOVE' and not self.dragging:return {'PASS_THROUGH'}
        return {'RUNNING_MODAL'}
    def cancel(self,context):self.finish()


class REBUILT_FLOW_OT_tweak(bpy.types.Operator):
    bl_idname='rebuilt.flow_tweak';bl_label='Edit Guides';bl_options={'REGISTER','UNDO','BLOCKING'}
    bl_description='Move, smooth or delete guides on the surface using the three edit tools'
    @classmethod
    def poll(cls,context):return can_author(context) and _DRAW is None and bool(guides(settings(context))) and context.mode=='OBJECT' and context.area and context.area.type=='VIEW_3D'
    ray_hit=REBUILT_FLOW_OT_draw.ray_hit
    hit=REBUILT_FLOW_OT_draw.hit
    def invoke(self,context,event):
        global _DRAW
        self.guide_settings=settings(context)
        self.guide_settings.edit_tool='MOVE'
        self.area=context.area;self.region=next(r for r in self.area.regions if r.type=='WINDOW');self.rv3d=context.space_data.region_3d
        try:_,_,self.bvh,_,self.scale=mesh_snapshot(settings(context).source,context.evaluated_depsgraph_get())
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        self.scene=context.scene;self.hover=None;self.drag=None
        set_view(settings(context),False)
        self.history=GuideHistory(self.guide_settings)
        self.handler=add_guide_handler(self.preview,'POST_VIEW')
        self.text_handler=add_guide_handler(self.instructions,'POST_PIXEL')
        _DRAW=self;context.window_manager.modal_handler_add(self)
        self.area.header_text_set('Edit Guides: choose Move / Smooth / Delete in Rebuilt | S: smooth | X / Delete: remove | Esc: done')
        return {'RUNNING_MODAL'}
    def pick(self,xy):
        best=None;limit=(14*bpy.context.preferences.system.ui_scale)**2
        for obj in guides(self.guide_settings):
            if obj.hide_get():continue
            points=guide_points(obj);closed=obj.data.splines[0].use_cyclic_u
            lengths,total=arc_lengths(points,closed)
            for i in range(len(points) if closed else len(points)-1):
                a,b=points[i],points[(i+1)%len(points)]
                va=view3d_utils.location_3d_to_region_2d(self.region,self.rv3d,a)
                vb=view3d_utils.location_3d_to_region_2d(self.region,self.rv3d,b)
                if va is None or vb is None:continue
                edge=vb-va
                t=max(0.,min(1.,(xy-va).dot(edge)/max(edge.length_squared,1e-12)))
                nearest=va+edge*t;distance=(xy-nearest).length_squared
                if distance>=limit:continue
                co=a.lerp(b,t);visible=self.ray_hit(nearest)
                if visible is None or (co-visible).length>self.scale*.008:continue
                limit=distance;best={'obj':obj,'co':co,'arc':lengths[i]+(b-a).length*t,'total':total,'xy':nearest}
        return best
    def preview(self):
        if bpy.context.area!=self.area or self.hover is None:return
        shader=gpu.shader.from_builtin('UNIFORM_COLOR')
        batch=batch_for_shader(shader,'POINTS',{'pos':[self.hover['co']]})
        gpu.state.blend_set('ALPHA');gpu.state.depth_test_set('LESS_EQUAL');gpu.state.point_size_set(12*bpy.context.preferences.system.ui_scale)
        shader.bind();shader.uniform_float('color',(1,.15,.07,1));batch.draw(shader)
        gpu.state.point_size_set(1);gpu.state.depth_test_set('NONE');gpu.state.blend_set('NONE')
    def instructions(self):
        if bpy.context.area!=self.area:return
        blf.position(0,24,50,0);blf.size(0,16);blf.color(0,1,.8,.15,1)
        tool=self.guide_settings.edit_tool
        action={'MOVE':'Drag to move','SMOOTH':'Click to smooth','DELETE':'Click to delete'}[tool]
        blf.draw(0,f'EDIT GUIDES   {action}  |  S: smooth  |  X / Delete: remove  |  Enter / Esc: done')
    def apply_edit(self,obj,action):
        self.hover=None
        edit_guide(self.guide_settings,obj,action,self.bvh,self.scale)
        self.history.checkpoint()
        redraw()
    def begin_drag(self,pick,mouse):
        s=self.guide_settings;obj=pick['obj']
        other=paired_guide(s,obj)
        self.drag={'obj':obj,'points':guide_points(obj),'closed':obj.data.splines[0].use_cyclic_u,'anchor':pick['arc'],'total':pick['total'],'mouse':mouse,'other':other,'other_points':guide_points(other) if other else None,'new_pair':None,'pair_id':obj.get('rebuilt_flow_pair'),'axis':obj.get('rebuilt_flow_mirror_axis')}
        for o in bpy.context.selected_objects:o.select_set(False)
        obj.select_set(True);bpy.context.view_layer.objects.active=obj
    def move_drag(self,target):
        d=self.drag;s=self.guide_settings
        radius=max(d['total']*s.edit_falloff/100,self.scale*.006)
        try:
            points=deform_stroke(d['points'],d['anchor'],target-d['mouse'],radius,self.bvh,self.scale,d['closed'])
            replace_points(d['obj'],points,d['closed'])
            other=sync_mirror(s,d['obj'],self.bvh,self.scale)
            if other is not None and other!=d['other']:d['new_pair']=other
            self.hover['co']=target
            s.message='Release to keep the change. Generate when ready.'
        except ValueError as exc:s.message=str(exc)
    def restore_drag(self):
        d=self.drag
        if d is None:return
        replace_points(d['obj'],d['points'],d['closed'])
        if d['other'] is not None:replace_points(d['other'],d['other_points'])
        if d['new_pair'] is not None:
            obj=d['new_pair'];data=obj.data;bpy.data.objects.remove(obj,do_unlink=True)
            if data.users==0:bpy.data.curves.remove(data)
        for key,value in [('rebuilt_flow_pair',d['pair_id']),('rebuilt_flow_mirror_axis',d['axis'])]:
            if value is None:
                if key in d['obj']:del d['obj'][key]
            else:d['obj'][key]=value
            if d['other'] is not None and value is not None:d['other'][key]=value
        self.drag=None
    def finish(self):
        try:self.restore_drag()
        finally:
            release_guide_mode()
            select_source(self.guide_settings)
    @guard_guide_modal
    def modal(self,context,event):
        if _DRAW is not self:return {'FINISHED'}
        if history_event(self,event):return {'RUNNING_MODAL'}
        xy=Vector((event.mouse_x-self.region.x,event.mouse_y-self.region.y))
        if event.type in {'ESC','RET','NUMPAD_ENTER'}:
            if event.type!='ESC':self.drag=None
            self.finish();return {'FINISHED'}
        if editor_ui_event(self,event):
            self.hover=None
            if event.type in {'X','DEL','BACK_SPACE'}:
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}
        if event.type=='RIGHTMOUSE' and event.value=='PRESS':self.restore_drag();redraw();return {'RUNNING_MODAL'}
        if navigation_event(self,context,event):
            if self.drag is not None and event.value=='PRESS':self.drag=None
            self.hover=None
            return {'PASS_THROUGH'}
        if event.type=='MOUSEMOVE':
            if self.drag:
                target=self.ray_hit(xy)
                if target is not None:self.move_drag(target)
            else:self.hover=self.pick(xy)
        elif event.type=='LEFTMOUSE' and event.value=='PRESS':
            self.hover=self.pick(xy);hit=self.ray_hit(xy)
            if self.hover is not None and hit is not None:
                action=self.guide_settings.edit_tool
                if action=='MOVE':
                    self.begin_drag(self.hover,hit)
                else:
                    self.apply_edit(self.hover['obj'],action)
                    self.hover=None
        elif event.type=='LEFTMOUSE' and event.value=='RELEASE':
            if self.drag:
                self.history.checkpoint()
                self.guide_settings.message='Guide updated. Press Esc, then Retopologise.'
                self.guide_settings.details='';self.guide_settings.warning=''
            self.drag=None
        elif event.type in {'X','DEL','BACK_SPACE','S'} and event.value=='PRESS' and self.drag is None:
            self.hover=self.pick(xy)
            if self.hover:
                action='SMOOTH' if event.type=='S' else 'DELETE'
                self.apply_edit(self.hover['obj'],action)
                self.hover=None
        redraw()
        if event.type=='MOUSEMOVE' and self.drag is None:return {'PASS_THROUGH'}
        return {'RUNNING_MODAL'}
    def cancel(self,context):self.finish()


class REBUILT_FLOW_OT_remove(bpy.types.Operator):
    bl_idname='rebuilt.flow_remove';bl_label='Delete Selected Guide';bl_options={'REGISTER','UNDO'}
    @classmethod
    def poll(cls,context):return can_author(context) and _DRAW is None and context.mode=='OBJECT' and context.active_object in guides(settings(context))
    def execute(self,context):
        s=settings(context)
        delete_guide(s,context.active_object)
        select_source(s)
        s.message='Guide deleted. Retopologise to update the mesh.'
        return {'FINISHED'}




def resolve_source(context):
    """Resolve a source from its mesh, a generated result or one of its own guides."""
    obj = context.active_object
    if obj is None:
        return None
    if obj.get('rebuilt_flow_guide'):
        return obj.parent if source_ok(obj.parent) else None
    name = obj.get('rebuilt_asset')
    if name:
        from ..pipeline.context import AssetContext
        ctx = AssetContext.for_object(context.scene, obj)
        return ctx.source_object if ctx else None
    return obj if source_ok(obj) else None


AVAILABLE = False  # Suspended at the user's request; retain authored guide data.


def can_author(context):
    if not AVAILABLE:
        return False
    from ..core.flow_guides import binary_path
    from ..pipeline.job import active_runners
    s = settings(context)
    r = context.scene.rebuilt
    return (s is not None and s.enabled and r.mode == 'manual'
            and r.topology == 'quads' and r.remesher == 'auto'
            and binary_path() is not None and not active_runners(context.scene))


def select_source(s):
    if s and s.source and s.source.name in bpy.context.view_layer.objects:
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        s.source.hide_set(False)
        s.source.select_set(True)
        bpy.context.view_layer.objects.active = s.source


def fingerprint(source):
    """Include actual guide geometry, pairing-independent transforms and solver settings."""
    import hashlib
    if source is None or not source.rebuilt_flow.enabled:
        return ''
    s = source.rebuilt_flow
    state = [s.radius, s.strength, [list(row) for row in source.matrix_world]]
    for obj in sorted(guides(s), key=lambda o: o.get('rebuilt_flow_id', o.name)):
        state.append([[list(p) for p in guide_points(obj)], obj.data.splines[0].use_cyclic_u])
    return hashlib.sha256(json.dumps(state, separators=(',', ':')).encode()).hexdigest()


def recipe_fields(scene_settings, source=None):
    if not AVAILABLE:
        return {}
    if scene_settings.mode != 'manual' or scene_settings.topology != 'quads' or scene_settings.remesher != 'auto':
        return {}
    source = source if source is not None else resolve_source(bpy.context)
    if source is None or not source.rebuilt_flow.enabled:
        return {}
    return {'flow_guides_key': fingerprint(source), 'flow_guides_strength': source.rebuilt_flow.strength}


def gather(ctx, working, *, prepared_mesh=None):
    from ..core.flow_guides import binary_path
    if binary_path() is None:
        raise ValueError('Flow Guides is available in the Apple Silicon build only.')
    s = ctx.source_object.rebuilt_flow
    if not guides(s):
        raise ValueError('Draw at least one guide, or turn off Flow Guides.')
    if fingerprint(ctx.source_object) != ctx.recipe.retopo.flow_guides_key:
        raise ValueError('Guides changed before generation. Run Retopologise again.')
    if not 100 <= ctx.recipe.retopo.target_faces <= 100000:
        raise ValueError('Flow Guides supports 100–100,000 quads. Adjust Quads before generating.')
    vertices, triangles, bvh, center, scale = mesh_snapshot(
        working, bpy.context.evaluated_depsgraph_get(), prepared_mesh=prepared_mesh)
    return {'vertices': [tuple((v-center)/scale) for v in vertices], 'triangles': triangles,
            'samples': sample_guides(s, bvh, center, scale), 'center': tuple(center), 'scale': scale,
            'inverse_matrix': [list(row) for row in working.matrix_world.inverted()],
            'guide_count': len(guides(s))}


def hide_guides(source):
    for obj in guides(source.rebuilt_flow):
        obj.hide_set(True)


class REBUILT_FLOW_OT_view(bpy.types.Operator):
    bl_idname = 'rebuilt.flow_view'
    bl_label = 'Show Retopology'
    bl_description = 'Return to the generated mesh; the guides are kept for later editing'
    @classmethod
    def poll(cls, context):
        from ..pipeline.context import ContextError
        try:
            s = settings(context)
            return _DRAW is None and s is not None and s.output is not None
        except ContextError:
            return False
    def execute(self, context):
        from ..pipeline.context import ContextError
        try:
            s = settings(context)
        except ContextError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        if s is None or s.output is None:
            return {'CANCELLED'}
        set_view(s, True)
        for obj in context.selected_objects:
            obj.select_set(False)
        s.output.select_set(True)
        context.view_layer.objects.active = s.output
        return {'FINISHED'}


class REBUILT_FLOW_OT_delete(bpy.types.Operator):
    bl_idname = 'rebuilt.flow_delete'
    bl_label = 'Delete'
    bl_description = 'Delete the selected guide; with no guide selected, click one to delete it'
    @classmethod
    def poll(cls,context):
        return _DRAW is not None and hasattr(_DRAW,'apply_edit') and _DRAW.drag is None
    def execute(self,context):
        s=_DRAW.guide_settings
        s.edit_tool='DELETE'
        obj=context.active_object
        if obj in guides(s) and obj.select_get():
            _DRAW.apply_edit(obj,'DELETE')
        else:
            s.message='Click a guide to delete it'
        redraw()
        return {'FINISHED'}


class REBUILT_FLOW_OT_done(bpy.types.Operator):
    bl_idname = 'rebuilt.flow_done'
    bl_label = 'Done Editing'
    bl_description = 'Keep the guide edits and return to Retopologise'
    @classmethod
    def poll(cls, context):
        return _DRAW is not None
    def execute(self, context):
        _DRAW.finish()
        return {'FINISHED'}


def draw_panel(layout, context):
    if not AVAILABLE:
        return
    from ..core.flow_guides import binary_path
    from ..pipeline.job import active_runners
    s = settings(context)
    if s is None:
        return
    box = layout.box()
    header = box.row()
    header.enabled = _DRAW is None and not active_runners(context.scene)
    header.prop(s, 'enabled')
    if not s.enabled:
        return
    if binary_path() is None:
        box.label(text='Requires the Apple Silicon build', icon='INFO')
        return
    if _DRAW is not None:
        if isinstance(_DRAW, REBUILT_FLOW_OT_tweak):
            row=box.row(align=True)
            row.prop_enum(s,'edit_tool','MOVE',text='Move')
            row.prop_enum(s,'edit_tool','SMOOTH',text='Smooth')
            row.operator('rebuilt.flow_delete',text='Delete',depress=s.edit_tool=='DELETE')
            if s.edit_tool == 'SMOOTH':
                box.prop(s, 'edit_smoothness', slider=True)
            box.label(text={'MOVE':'Drag a guide to move it', 'SMOOTH':'Click a guide to smooth it',
                            'DELETE':'Click a guide to delete it'}[s.edit_tool])
            box.operator('rebuilt.flow_done', icon='CHECKMARK')
        else:
            box.label(text='Enter / Esc: finish drawing', icon='INFO')
        box.label(text='Ctrl/Cmd+Z: undo · Shift: redo')
        return
    body = box.column(align=True)
    body.enabled = not active_runners(context.scene)
    row = body.row(align=True)
    row.operator('rebuilt.flow_draw', text='Draw', icon='GREASEPENCIL')
    row.operator('rebuilt.flow_tweak', text='Edit', icon='EDITMODE_HLT')
    row.operator('rebuilt.flow_remove', text='', icon='TRASH')
    body.label(text=f'{len(guides(s))} guides · Esc when done')
    row = body.row(align=True)
    row.prop(s, 'show_options', text='Guide Settings', emboss=False,
             icon='TRIA_DOWN' if s.show_options else 'TRIA_RIGHT')
    if s.output:
        row.operator('rebuilt.flow_view', text='', icon='MESH_GRID')
    if s.show_options:
        body.prop(s, 'strength_percent', slider=True)
        body.prop(s, 'radius')
        body.row(align=True).prop(s, 'symmetry', expand=True)
        body.label(text='Shift-drag: straight line', icon='INFO')
        body.label(text='Edit: S to smooth, X to delete')
    body.label(text='Steers quad direction', icon='INFO')


CLASSES = (REBUILT_FLOW_Settings, REBUILT_FLOW_OT_draw, REBUILT_FLOW_OT_tweak,
           REBUILT_FLOW_OT_remove, REBUILT_FLOW_OT_view, REBUILT_FLOW_OT_done, REBUILT_FLOW_OT_delete)


@bpy.app.handlers.persistent
def reset_runtime(_dummy):
    try:
        if _DRAW:
            _DRAW.finish()
    finally:
        release_guide_mode()


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Object.rebuilt_flow = PointerProperty(type=REBUILT_FLOW_Settings)
    bpy.app.handlers.load_pre.append(reset_runtime)
    bpy.app.handlers.undo_pre.append(reset_runtime)


def unregister():
    reset_runtime(None)
    for handlers in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre):
        if reset_runtime in handlers:
            handlers.remove(reset_runtime)
    del bpy.types.Object.rebuilt_flow
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
