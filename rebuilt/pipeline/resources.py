"""Exact created-resource references; labels never authorize resource deletion."""
import uuid

import bpy
from idprop.types import IDPropertyGroup

from .context import ContextError


KINDS = {'material': bpy.types.Material, 'image': bpy.types.Image}
PURPOSES = {'bake', 'output', 'preview'}


def validate(rows):
    if not isinstance(rows, (dict, IDPropertyGroup)):
        raise ContextError('Invalid Rebuilt resource ownership record.')
    seen = set()
    for row in rows.values():
        if (not isinstance(row, (dict, IDPropertyGroup))
                or not isinstance(row.get('kind'), str) or row['kind'] not in KINDS
                or not isinstance(row.get('purpose'), str) or row['purpose'] not in PURPOSES):
            raise ContextError('Invalid Rebuilt resource identity.')
        value = row.get('id')
        if value is not None:
            if not isinstance(value, KINDS[row['kind']]) or value in seen:
                raise ContextError('Invalid or duplicate Rebuilt resource reference.')
            seen.add(value)


def copy_rows(rows):
    return {key: dict(row) for key, row in rows.items()}


def retire_rows(rows, *, keys=None):
    """Retire from a Blender-held creation ledger, retaining failed deletion IDs."""
    from .ownership import live
    if not isinstance(rows, IDPropertyGroup):
        raise ContextError('Resource retirement requires a Blender-held creation ledger.')
    for kind, data in (('material', bpy.data.materials), ('image', bpy.data.images)):
        for key in list(rows):
            row = rows[key]
            if row['kind'] != kind or (keys is not None and key not in keys):
                continue
            value = row.get('id')
            # A sole ledger reference authorizes deletion. Other users retain the
            # ID; failure must leave this exact reference available for retry.
            if live(value, data) and value.users == 1:
                data.remove(value)
            del rows[key]


class ResourceLedger:
    def __init__(self, owner):
        self.owner = owner

    def snapshot(self):
        record = self.owner.record()
        return copy_rows(record.get('resources', {})) if record is not None else {}

    def create(self, purpose, factory, *args, **kwargs):
        """Allocate a fresh unlinked ID and register it before any consumer uses it."""
        self.owner.ensure()
        value = factory(*args, **kwargs)
        try:
            return self.add(value, purpose)
        except Exception:
            # This allocation is ours even if registering its ledger row failed.
            # A successful insertion adds a user and retains it for later cleanup.
            for kind, data in (('material', bpy.data.materials), ('image', bpy.data.images)):
                if isinstance(value, KINDS[kind]) and value.users == 0:
                    data.remove(value)
                    break
            raise

    def add(self, value, purpose):
        kind = next((key for key, cls in KINDS.items() if isinstance(value, cls)), None)
        if kind is None or purpose not in PURPOSES or value.library is not None:
            raise ContextError('Only newly created local bake/preview resources can be registered.')
        record = self.owner.ensure()
        # An older addon must refuse a record with new deletion authority.
        self.owner._root()['version'] = max(2, self.owner._root()['version'])
        if 'resources' not in record:
            record['resources'] = {}
        rows = record['resources']
        if not any(row.get('id') == value for row in rows.values()):
            rows[uuid.uuid4().hex] = {'id': value, 'kind': kind, 'purpose': purpose}
        return value

    def key(self, value):
        for key, row in self.snapshot().items():
            if row.get('id') == value:
                return key
        raise ContextError('The newly baked image has no resource reference.')

    def release(self, *, ids=None, purpose=None):
        record = self.owner.ensure()
        rows = record.get('resources')
        if rows is None:
            return
        selected = {key for key, row in rows.items()
                    if (ids is None or row.get('id') in ids)
                    and (purpose is None or row['purpose'] == purpose)}
        retire_rows(rows, keys=selected)

    def collect_unused(self):
        # A sole ledger user means no mesh, node, fake user, artist copy or other
        # scene still needs the ID. Drop materials first to release image users.
        for kind in ('material', 'image'):
            ids = {row.get('id') for row in self.snapshot().values()
                   if row['kind'] == kind and (row.get('id') is None or row['id'].users == 1)}
            if ids:
                self.release(ids=ids)
