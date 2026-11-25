"""Failure-atomic seam/UV writes with isolation of shared Blender meshes."""
from contextlib import contextmanager

import bpy
import numpy as np


def _read(collection, name, dtype, width=1):
    values = np.empty(len(collection) * width, dtype=dtype)
    collection.foreach_get(name, values)
    return values


def _uv_snapshot(mesh):
    layers = []
    for layer in mesh.uv_layers:
        fields = {'uv': _read(layer.data, 'uv', np.float32, 2)}
        properties = layer.data[0].bl_rna.properties if len(layer.data) else {}
        for field in ('pin_uv', 'select', 'select_edge'):
            if field in properties:
                fields[field] = _read(layer.data, field, bool)
        layers.append((layer.name, fields, layer.active_render, layer.active_clone))
    return layers, mesh.uv_layers.active_index


def _attribute_snapshot(attribute):
    if attribute is None:
        return None
    fields = {}
    if len(attribute.data):
        for prop in attribute.data[0].bl_rna.properties:
            if prop.is_readonly or prop.identifier in ('rna_type', 'color_srgb'):
                continue
            name = prop.identifier
            if prop.type == 'STRING':
                fields[name] = [getattr(item, name) for item in attribute.data]
            else:
                dtype = {'FLOAT': np.float32, 'INT': np.int32, 'BOOLEAN': bool}[prop.type]
                fields[name] = _read(attribute.data, name, dtype, prop.array_length or 1)
    return attribute.data_type, attribute.domain, fields


def _restore_marker(low, name, existed, value):
    if existed:
        low[name] = value
    elif name in low:
        del low[name]


@contextmanager
def seam_transaction(low, attribute_name, marker_name):
    """Preserve artist mesh users and restore the original state on failure."""
    mesh = low.data
    had_marker = marker_name in low
    marker = low.get(marker_name)
    if mesh.users > 1 or mesh.library is not None or not mesh.is_editable:
        # Shared/linked input is borrowed. All edits must occur on the owned
        # object's copy; rolling back writes later would still touch bystanders.
        working = mesh.copy()
        try:
            working.use_fake_user = False
            low.data = working
            yield
        except BaseException:
            low.data = mesh
            _restore_marker(low, marker_name, had_marker, marker)
            if working.users == 0:
                bpy.data.meshes.remove(working)
            raise
        return

    seams = _read(mesh.edges, 'use_seam', bool)
    layers, active_index = _uv_snapshot(mesh)
    attribute = _attribute_snapshot(mesh.attributes.get(attribute_name))
    try:
        yield
    except BaseException:
        mesh.edges.foreach_set('use_seam', seams)
        for layer in list(mesh.uv_layers):
            mesh.uv_layers.remove(layer)
        for name, fields, render, clone in layers:
            layer = mesh.uv_layers.new(name=name, do_init=False)
            for field, values in fields.items():
                layer.data.foreach_set(field, values)
            layer.active_render = render
            layer.active_clone = clone
        if layers:
            mesh.uv_layers.active_index = active_index
        current = mesh.attributes.get(attribute_name)
        if current is not None:
            mesh.attributes.remove(current)
        if attribute is not None:
            data_type, domain, fields = attribute
            restored = mesh.attributes.new(attribute_name, data_type, domain)
            for field, values in fields.items():
                if isinstance(values, list):
                    for item, value in zip(restored.data, values):
                        setattr(item, field, value)
                else:
                    restored.data.foreach_set(field, values)
        _restore_marker(low, marker_name, had_marker, marker)
        mesh.update()
        raise
