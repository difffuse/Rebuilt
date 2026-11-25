"""Scene-held authority for generated objects; copied tags grant no authority."""
import uuid

import bpy
from idprop.types import IDPropertyGroup

from .context import ContextError, KEY_ASSET, KEY_ROLE, KEY_WORK, ROLE_SOURCE

KEY_REGISTRY = 'rebuilt_ownership'
KEY_MARKER = 'rebuilt_ownership_version'
VERSION = 3


def live(value, data):
    try:
        if value is None:
            return False
        library = value.library.filepath if value.library is not None else None
        return data.get((value.name, library)) == value
    except ReferenceError:
        return False


def contains_collection(scene, collection):
    pending = [scene.collection]
    while pending:
        current = pending.pop()
        if current == collection:
            return True
        pending.extend(current.children)
    return False


def _group(value):
    return isinstance(value, (dict, IDPropertyGroup))


class ArtifactOwnership:
    def __init__(self, scene, source, asset):
        self.scene, self.source, self.asset = scene, source, asset
        self.token = None
        self.archive = None
        self.file_recovery = None
        self.operation = None

    def _root(self):
        root = self.scene.get(KEY_REGISTRY)
        if root is None:
            return None
        if (not _group(root) or type(root.get('version')) is not int
                or root['version'] not in (1, 2, VERSION) or not _group(root.get('entries'))):
            raise ContextError('Unsupported Rebuilt ownership data. Keep this file and use a compatible addon version.')
        sources, objects, collections = set(), set(), set()
        holders = {}

        def claim(obj, token):
            if obj is not None:
                if obj in holders and holders[obj] != token:
                    raise ContextError('A Rebuilt artifact or archived original belongs to more than one record.')
                holders[obj] = token

        for token, entry in root['entries'].items():
            if not _group(entry) or not _group(entry.get('objects')):
                raise ContextError('Invalid Rebuilt ownership record.')
            source = entry.get('source')
            collection = entry.get('collection')
            if 'resources' in entry:
                from .resources import validate
                if root['version'] < 2:
                    raise ContextError('Resource ownership requires the updated Rebuilt scene schema.')
                validate(entry['resources'])
            if entry.get('file_journal') is not None:
                from .cancel_files import validate_descriptor, FileRecoveryError
                try:
                    validate_descriptor(entry['file_journal'])
                except FileRecoveryError as exc:
                    raise ContextError(str(exc)) from exc
            if entry.get('auto_completion') is not None:
                from .cancel_files import validate_descriptor, FileRecoveryError
                completion = entry['auto_completion']
                if (root['version'] < 3 or not _group(completion)
                        or type(completion.get('version')) is not int or completion['version'] != 1):
                    raise ContextError('Invalid completed Auto recovery record.')
                try:
                    validate_descriptor(completion.get('journal'))
                except FileRecoveryError as exc:
                    raise ContextError(str(exc)) from exc
                if (entry.get('file_journal') is not None and
                        dict(completion['journal']) != dict(entry['file_journal'])):
                    raise ContextError('The completed Auto and texture journals disagree.')
            if entry.get('stage_operation') is not None:
                from .stage_operation import validate
                if root['version'] < 3:
                    raise ContextError('Stage recovery requires the updated Rebuilt scene schema.')
                operation = entry['stage_operation']
                validate(operation)
                claim(operation.get('low'), token)
                claim(operation.get('previous'), token)
                for row in operation['objects'].values():
                    claim(row.get('object'), token)
                for row in operation.get('rollback', {}).values():
                    claim(row.get('object'), token)
            if source is not None:
                if not isinstance(source, bpy.types.Object) or source in sources:
                    raise ContextError('Invalid or duplicate Rebuilt source ownership.')
                sources.add(source)
            if collection is not None:
                if not isinstance(collection, bpy.types.Collection) or collection in collections:
                    raise ContextError('Invalid or duplicate Rebuilt working-collection reference.')
                collections.add(collection)
            for item in entry['objects'].values():
                if (not _group(item) or not isinstance(item.get('role'), str)
                        or not item['role'] or item['role'] == ROLE_SOURCE):
                    raise ContextError('Invalid Rebuilt artifact role.')
                if 'retiring_mesh' in item and (root['version'] < 3 or
                        (item['retiring_mesh'] is not None and
                         not isinstance(item['retiring_mesh'], bpy.types.Mesh))):
                    raise ContextError('Invalid Rebuilt artifact retirement reference.')
                obj = item.get('object')
                if obj is not None:
                    if not isinstance(obj, bpy.types.Object) or obj in objects or obj == source:
                        raise ContextError('Invalid or duplicate Rebuilt artifact reference.')
                    objects.add(obj)
                    claim(obj, token)
            archive = entry.get('archive')
            if archive is not None:
                if any(key in archive for key in ('rollback_resources', 'rollback_objects')) and root['version'] < 3:
                    raise ContextError('Retryable archive retirement requires the updated Rebuilt scene schema.')
                from .archive_record import validate
                validate(archive)
                for row in archive['objects'].values():
                    claim(row.get('object'), token)
                    claim(row.get('replacement'), token)
                for row in archive.get('rollback_objects', {}).values():
                    claim(row.get('object'), token)
        if sources.intersection(holders):
            raise ContextError('A Rebuilt source is also registered as a generated artifact.')
        return root

    def record(self):
        root = self._root()
        if root is None:
            if self.token is not None:
                raise ContextError('The active Rebuilt ownership record was removed.')
            return None
        entries = root['entries']
        if live(self.source, bpy.data.objects):
            matches = [(key, entry) for key, entry in entries.items() if entry.get('source') == self.source]
            if matches:
                key, entry = matches[0]
                if self.token is not None and key != self.token:
                    raise ContextError('The active Rebuilt ownership record was replaced.')
                self.token = key
                return entry
        elif self.token is not None:
            # Only an already-bound transaction may finish after its source is
            # deleted. A new context cannot obtain authority using a token/name.
            entry = entries.get(self.token)
            if entry is not None and entry.get('source') is None:
                return entry
        if self.token is not None:
            raise ContextError('The active Rebuilt ownership record is unavailable.')
        return None

    def legacy(self):
        if not live(self.source, bpy.data.objects) or self.source.get(KEY_MARKER) is not None:
            return [], None
        objects, by_role = [], {}
        for obj in self.scene.objects:
            role = obj.get(KEY_ROLE)
            if obj.get(KEY_ASSET) != self.asset or role == ROLE_SOURCE:
                continue
            if not isinstance(role, str) or not role:
                raise ContextError(f'Object {obj.name!r} has an invalid Rebuilt role.')
            by_role.setdefault(role, []).append(obj.name)
            objects.append(obj)
        conflicts = [f'{role}: {", ".join(names)}' for role, names in by_role.items() if len(names) > 1]
        if conflicts:
            raise ContextError('Ambiguous legacy results (' + '; '.join(conflicts) + '). Remove Rebuilt tags from independent copies before adopting existing results.')
        collections = [c for c in self.scene.collection.children if c.get(KEY_WORK) == self.asset]
        if len(collections) > 1:
            raise ContextError('Ambiguous legacy working collections: ' + ', '.join(c.name for c in collections))
        return objects, collections[0] if collections else None

    def objects(self, role=None):
        record = self.record()
        if record is None:
            values, _ = self.legacy()
            return [obj for obj in values if role is None or obj.get(KEY_ROLE) == role]
        return [item['object'] for item in record['objects'].values()
                if item.get('object') is not None
                and self.scene.objects.get(item['object'].name) == item['object']
                and (role is None or item['role'] == role)]

    def collection(self):
        record = self.record()
        return record.get('collection') if record is not None else self.legacy()[1]

    def _check_scope(self, objects, collection):
        for obj in objects:
            if (self.scene.objects.get(obj.name) != obj or len(obj.users_scene) != 1
                    or obj.users_scene[0] != self.scene):
                raise ContextError(f'Owned object {obj.name!r} was moved or is shared with another scene. Make a full independent scene copy before processing.')
            if not obj.is_editable:
                raise ContextError(f'Object {obj.name!r} is linked or readonly.')
        if collection is not None:
            scenes = [scene for scene in bpy.data.scenes if contains_collection(scene, collection)]
            if scenes != [self.scene] or not collection.is_editable:
                raise ContextError(f'Working collection {collection.name!r} was moved, shared or made readonly.')

    def validate_source_scope(self):
        if not self.scene.is_editable:
            raise ContextError('The owning scene is readonly.')
        if live(self.source, bpy.data.objects):
            self._check_scope([self.source], None)

    def validate(self, *, readonly=False):
        self.validate_source_scope()
        record = self.record()
        if record is None:
            objects, collection = self.legacy()
            if live(self.source, bpy.data.objects) and self.source.get(KEY_MARKER) is not None:
                raise ContextError('This copied source has no ownership record. Use Selected Model to start an independent asset.')
            if objects or collection is not None:
                raise ContextError('This asset predates ownership tracking. Review and adopt its existing results before rebuilding; copied tags alone cannot authorize cleanup.')
            return
        if record.get('stage_operation') is not None and self.operation is None:
            from .job import active_runners
            live_owner = any(runner.ctx._source == self.source
                             and runner.ctx._ownership.token == self.token
                             and runner.ctx._ownership.operation is not None
                             for runner in active_runners(self.scene))
            if not live_owner:
                raise ContextError('This asset has a retained stage transaction. Recover it before processing this asset.')
            if not readonly:
                raise ContextError('This asset is currently completing a stage. Wait or cancel before changing it.')
        if record.get('archive') is not None and self.archive is None:
            from .job import active_runners
            live_owner = any(
                runner.ctx._source == self.source
                and runner.ctx._ownership.token == self.token
                and runner.ctx.auto_archive is not None
                and runner.ctx._ownership.archive is runner.ctx.auto_archive
                and runner.ctx.auto_archive.started
                for runner in active_runners(self.scene)
            )
            if not live_owner:
                raise ContextError('This asset has a retained Auto backup. Recover the interrupted build before processing this asset.')
            if not readonly:
                raise ContextError('This asset is currently being rebuilt. Wait for completion or cancel the running job before changing it.')
        if ((record.get('file_journal') is not None or record.get('auto_completion') is not None)
                and self.archive is None and self.file_recovery is None):
            from .job import active_runners
            owners = [runner for runner in active_runners(self.scene)
                      if runner.ctx._source == self.source and runner.ctx._ownership.token == self.token]
            if not owners:
                raise ContextError('This asset has an interrupted texture transaction. Recover it before starting another job.')
            if not readonly and all(runner.ctx._ownership is not self for runner in owners):
                raise ContextError('This asset is currently being rebuilt. Wait or cancel before changing it.')
        objects = [item['object'] for item in record['objects'].values() if item.get('object') is not None]
        self._check_scope(objects, record.get('collection'))
        for item in record['objects'].values():
            obj = item.get('object')
            if obj is not None and 'retiring_mesh' in item and obj.data != item['retiring_mesh']:
                raise ContextError('An unfinished artifact mesh was replaced. Recovery data was retained.')

    def ensure(self, *, adopt=False, new=False):
        record = self.record()
        if record is not None:
            self.validate()
            return record
        if not live(self.source, bpy.data.objects):
            raise ContextError('Cannot create ownership without the original source.')
        objects, collection = ([], None) if new else self.legacy()
        if not adopt and not new:
            self.validate()
            objects, collection = [], None
        self._check_scope([self.source, *objects], collection)
        if not self.scene.is_editable:
            raise ContextError('The owning scene is readonly.')
        root = self._root()
        existing = {item['object'] for entry in root['entries'].values()
                    for item in entry['objects'].values() if item.get('object') is not None} if root is not None else set()
        if existing.intersection(objects):
            raise ContextError('An existing result already belongs to another source.')
        if self.source in existing:
            raise ContextError('This is an owned result. Duplicate it before using it as an independent source.')
        created = root is None
        if created:
            self.scene[KEY_REGISTRY] = {'version': VERSION, 'entries': {}}
            root = self.scene[KEY_REGISTRY]
        token = uuid.uuid4().hex
        entry = {'source': self.source, 'objects': {
            str(i): {'object': obj, 'role': obj[KEY_ROLE]} for i, obj in enumerate(objects)}}
        if collection is not None:
            entry['collection'] = collection
        try:
            root['entries'][token] = entry
            self.source[KEY_MARKER] = VERSION
        except Exception:
            if token in root['entries']:
                del root['entries'][token]
            if created:
                del self.scene[KEY_REGISTRY]
            raise
        self.token = token
        return self.record()

    def check_registration(self, obj, role):
        self.ensure()
        if not isinstance(role, str) or not role or role == ROLE_SOURCE:
            raise ContextError('Invalid generated artifact role.')
        if obj == self.source or not obj.is_editable:
            raise ContextError('The source or a readonly object cannot be registered as a generated artifact.')
        if obj.users_scene and tuple(obj.users_scene) != (self.scene,):
            raise ContextError('An artifact must belong exclusively to this scene.')
        root = self._root()
        for token, other in root['entries'].items():
            if other.get('source') == obj:
                raise ContextError('Another source cannot be registered as a generated artifact.')
            archive = other.get('archive')
            if archive is not None and any(row.get('object') == obj for row in archive['objects'].values()):
                raise ContextError('An archived original cannot be registered as a new generated artifact.')
            for item in other['objects'].values():
                if item.get('object') == obj and token != self.token:
                    raise ContextError('This object already belongs to another asset.')

    def add(self, obj, role):
        self.check_registration(obj, role)
        entry = self.record()
        self._check_scope([obj], entry.get('collection'))
        for item in entry['objects'].values():
            if item.get('object') == obj:
                item['role'] = role
                return
        entry['objects'][uuid.uuid4().hex] = {'object': obj, 'role': role}

    def forget(self, objects):
        entry = self.ensure()
        for key, item in list(entry['objects'].items()):
            if (item.get('object') in objects or
                    (item.get('object') is None and item.get('retiring_mesh') is None)):
                del entry['objects'][key]

    def retire(self, roles, *, objects=None, pending_only=False):
        """Keep exact object/mesh authority through failed deletion and reopen."""
        entry = self.ensure()
        removed = 0
        for key, item in list(entry['objects'].items()):
            obj = item.get('object')
            if pending_only and obj is not None and 'retiring_mesh' not in item:
                continue
            if item['role'] not in roles or (objects is not None and obj is not None and obj not in objects):
                continue
            if live(obj, bpy.data.objects):
                if obj.type == 'MESH' and 'retiring_mesh' not in item:
                    self._root()['version'] = max(3, self._root()['version'])
                    item['retiring_mesh'] = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                removed += 1
            mesh = item.get('retiring_mesh')
            if live(mesh, bpy.data.meshes) and mesh.users == 1:
                bpy.data.meshes.remove(mesh)
            del entry['objects'][key]
        return removed

    def set_collection(self, collection):
        entry = self.ensure()
        self._check_scope([], collection)
        if collection is None:
            if 'collection' in entry:
                del entry['collection']
        else:
            entry['collection'] = collection

    def replace(self, original, replacement):
        entry = self.record()
        if entry is None:
            raise ContextError('Cannot replace an unowned artifact.')
        for item in entry['objects'].values():
            if item.get('object') == original:
                item['object'] = replacement
                return
        raise ContextError('The original artifact is no longer owned.')
