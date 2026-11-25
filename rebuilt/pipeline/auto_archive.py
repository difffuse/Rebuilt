"""Keep a completed asset intact while Auto builds its replacement."""
import bpy
import uuid
from contextlib import contextmanager

from .context import ContextError
from .archive_record import capture, validate_state, STATE_KEYS
from .ownership import live
from .resources import copy_rows, retire_rows
from ..bridge.image_refs import image_from_entry


def _in_layer(obj, layer):
    try:
        return obj is not None and layer.objects.get(obj.name) == obj
    except ReferenceError:
        return False


def _included_collections(layer_collection):
    if layer_collection.exclude:
        return set()
    included = {layer_collection.collection}
    for child in layer_collection.children:
        included.update(_included_collections(child))
    return included


def _validate_name(data, original, name, kind):
    if original.name == name:
        return
    library = original.library.filepath if original.library is not None else None
    occupying = data.get((name, library))
    if occupying is not None and occupying != original:
        raise ContextError(f'{kind} name {name!r} now belongs to other data. Rename it before recovering this build.')


def _freeze(value):
    if isinstance(value, bpy.types.ID):
        return (value.bl_rna.identifier, value.session_uid, value.name)
    if hasattr(value, 'items'):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _unique_recovery(ctx, record, descriptor):
    """Scene copies may duplicate recovery references, never external authority."""
    from .ownership import KEY_REGISTRY
    originals = {row.get('object') for row in (record.get('archive') or {}).get('objects', {}).values()}
    originals.discard(None)
    for scene in bpy.data.scenes:
        root = scene.get(KEY_REGISTRY)
        if not hasattr(root, 'get') or not hasattr(root.get('entries'), 'values'):
            continue
        for other in root['entries'].values():
            if not hasattr(other, 'get') or (scene == ctx.scene and other.get('source') == ctx._source):
                continue
            snapshot = other.get('archive')
            if not hasattr(snapshot, 'get'):
                snapshot = {}
            operation = other.get('stage_operation')
            if not hasattr(operation, 'get'):
                operation = {}
            completion = other.get('auto_completion')
            if not hasattr(completion, 'get'):
                completion = {}
            journal = (other.get('file_journal') or snapshot.get('file_journal')
                       or operation.get('journal') or completion.get('journal'))
            if descriptor is not None and hasattr(journal, 'get') and journal.get('token') == descriptor['token']:
                raise ContextError('This texture recovery journal is shared by multiple scene assets. Keep the intended pending scene before recovering.')
            rows = snapshot.get('objects')
            if hasattr(rows, 'values') and any(hasattr(row, 'get') and row.get('object') in originals for row in rows.values()):
                raise ContextError('Archived originals are shared with another scene. Keep the intended pending scene before recovering.')


class AutoArchive:
    def __init__(self, ctx, state=None):
        self.ctx = ctx
        # Keep the old argument for callers; capture authoritative state at start.
        self.started = False

    @property
    def snapshot(self):
        record = self.ctx._ownership.record()
        if record is None or record.get('archive') is None:
            raise ContextError('The Auto recovery snapshot is unavailable.')
        return record['archive']

    def start(self):
        owner = self.ctx._ownership
        record = owner.ensure()
        if self.started or record.get('archive') is not None:
            raise ContextError('An earlier Auto backup must be recovered first.')
        self.ctx.finish_pending_retirement()
        originals = [obj for role in ('low', 'working', 'cleaned', 'proxy', 'cage')
                     for obj in owner.objects(role)]
        for obj in originals:
            for collection in obj.users_collection:
                if collection != self.ctx.scene.collection:
                    owner._check_scope([], collection)
        materials = []
        for obj in owner.objects('low'):
            for material in obj.data.materials:
                if material is not None and all(material != item[0] for item in materials):
                    materials.append((material, material.name))
        images = []
        for entry in (self.ctx.report('bake') or {}).get('maps', []):
            image = image_from_entry(entry)
            if image is not None and image not in images:
                images.append(image)
        record['archive'] = capture(self.ctx, originals, materials, images)
        owner._root()['version'] = max(2, owner._root()['version'])
        record['resources'] = {}
        owner.archive = self
        try:
            for obj in originals:
                self._stage_object(obj)
        except Exception:
            # The job does not own cleanup yet. Undo only this transaction.
            self.restore()
            raise
        self.snapshot['phase'] = 'running'
        self.started = True

    def _stage_object(self, obj):
        snapshot = self.snapshot
        row = next(row for row in snapshot['objects'].values() if row.get('object') == obj)
        replacement = obj.copy()
        row['replacement'] = replacement
        if row['role'] == 'low':
            replacement.data = obj.data.copy()
        obj.name = row['name'] + '_previous'
        replacement.name = row['name']
        collections = self._destinations(row)
        for collection in collections:
            collection.objects.link(replacement)
        self.ctx._ownership.replace(obj, replacement)
        for collection in collections:
            collection.objects.unlink(obj)
        layer = bpy.context.view_layer
        layer.update()
        replacement.hide_set(row['hidden'])
        if obj in snapshot['selected'].values():
            replacement.select_set(True)
        if obj == snapshot.get('active'):
            layer.objects.active = replacement

    def _destinations(self, row):
        return ([self.ctx.scene.collection] if row['scene_root'] else []) + list(row['collections'].values())

    def keeps_collection(self, collection):
        return any(collection in self._destinations(row) for row in self.snapshot['objects'].values())

    def files_touched(self):
        """Persist the file dependency before the attempt writes its first byte."""
        self.snapshot['files_touched'] = True

    def validate_restore(self):
        """Check all inputs before deleting, relinking or renaming any object."""
        owner = self.ctx._ownership
        owner.validate_source_scope()
        layer = bpy.context.view_layer
        if layer not in self.ctx.scene.view_layers[:]:
            raise ContextError('Switch to the owning scene before recovering this build.')
        included = _included_collections(layer.layer_collection)
        record, snapshot = owner.record(), self.snapshot
        rows = list(snapshot['objects'].values())
        originals = {row.get('object') for row in rows}
        attempts = {item['object'] for item in record['objects'].values()
                    if item.get('object') is not None and item['object'] not in originals}
        replacements = {row['replacement'] for row in rows if row.get('replacement') is not None}
        attempts.update(replacements)
        for row in snapshot.get('rollback_objects', {}).values():
            obj = row.get('object')
            if obj is not None:
                if obj.type == 'MESH' and obj.data != row.get('mesh'):
                    raise ContextError('An unfinished Auto mesh was replaced. Recovery data was retained.')
                attempts.add(obj)
        from .stage_operation import borrowed_objects
        attempts.update(borrowed_objects(self.ctx) - originals)
        owner._check_scope([], record.get('collection'))
        if 'work_collection' in snapshot:
            collection = snapshot['work_collection']
            if not live(collection, bpy.data.collections):
                raise ContextError('The original working collection was deleted. Backup data was retained.')
            owner._check_scope([], collection)
        for row in rows:
            obj = row.get('object')
            if not live(obj, bpy.data.objects) or not obj.is_editable:
                raise ContextError('An archived original was deleted or made readonly. Backup data was retained.')
            if obj.users_scene and tuple(obj.users_scene) != (self.ctx.scene,):
                raise ContextError('An archived original was moved to another scene.')
            if row['mesh_name'] and (not live(row.get('mesh'), bpy.data.meshes) or obj.data != row['mesh']):
                raise ContextError('An archived mesh was deleted or replaced. Backup data was retained.')
            if row['mesh_name'] and obj.data.name != row['mesh_name'] and not obj.data.is_editable:
                raise ContextError('An archived mesh name cannot be restored on readonly data. Backup data was retained.')
            if row['mesh_name']:
                _validate_name(bpy.data.meshes, obj.data, row['mesh_name'], 'Mesh')
            destinations = self._destinations(row)
            if not destinations:
                raise ContextError('An Auto restore destination is missing. Backup data was retained.')
            if not any(collection in included for collection in destinations):
                raise ContextError(f'Restore collections for {row["name"]!r} are excluded from the current view layer. Include a destination collection before recovering this build.')
            for collection in destinations:
                if collection != self.ctx.scene.collection:
                    if not live(collection, bpy.data.collections):
                        raise ContextError('An Auto restore collection was deleted. Backup data was retained.')
                    owner._check_scope([], collection)
            occupying = bpy.data.objects.get(row['name'])
            if occupying is not None and occupying != obj and occupying not in attempts:
                raise ContextError(f'Object name {row["name"]!r} now belongs to another object. Rename it before recovering this build.')
        for obj in attempts:
            if not (snapshot['phase'] == 'starting' and obj in replacements and not obj.users_scene):
                owner._check_scope([obj], None)
        for row in snapshot['materials'].values():
            if not live(row.get('material'), bpy.data.materials):
                raise ContextError('An archived material was deleted. Backup data was retained.')
            if row['material'].name != row['name'] and not row['material'].is_editable:
                raise ContextError('An archived material name cannot be restored on readonly data. Backup data was retained.')
            _validate_name(bpy.data.materials, row['material'], row['name'], 'Material')
        if any(not live(image, bpy.data.images) for image in snapshot['images'].values()):
            raise ContextError('An archived image was deleted. Backup data was retained.')
        return attempts

    @classmethod
    @contextmanager
    def _recovery_inputs(cls, ctx):
        from .job import active_runners
        from .cancel_files import CancelFiles, FileRecoveryError
        ctx.source_object
        if active_runners(ctx.scene):
            raise ContextError('Finish or cancel the running job before recovering an interrupted build.')
        owner = ctx._ownership
        record = owner.record()
        if record is None or not any(record.get(key) is not None
                                     for key in ('archive', 'file_journal', 'auto_completion')):
            raise ContextError('This asset has no interrupted build to recover.')
        archive = cls(ctx) if record.get('archive') is not None else None
        completion = record.get('auto_completion')
        descriptor = (completion['journal'] if completion is not None else
                      archive.snapshot.get('file_journal') if archive is not None else record.get('file_journal'))
        if archive is not None:
            if archive.snapshot['files_touched'] and descriptor is None:
                raise ContextError('This build changed texture files but has no durable file recovery journal. Backup data was retained.')
            if descriptor is not None and record.get('file_journal') is not None and dict(descriptor) != dict(record['file_journal']):
                raise ContextError('The scene and file recovery journals do not match. Backup data was retained.')
        _unique_recovery(ctx, record, descriptor)
        if completion is not None:
            owner.validate_source_scope()
            owner._check_scope(owner.objects(), record.get('collection'))
            if archive is not None:
                archive.validate_commit()
            attempts = set()
        elif archive is not None:
            attempts = archive.validate_restore()
        else:
            owner.validate_source_scope()
            source = ctx.source_object
            validate_state({key: source[key] for key in STATE_KEYS if key in source})
            attempts = {row['object'] for row in record['objects'].values() if row.get('object') is not None}
            from .stage_operation import borrowed_objects
            attempts.update(borrowed_objects(ctx))
            owner._check_scope(attempts, record.get('collection'))
        files, file_review = None, {'files': [], 'signature': ''}
        try:
            if descriptor is not None:
                try:
                    files = CancelFiles.reopen(descriptor)
                    files.hold()
                    if completion is not None:
                        if files.phase != 'committed':
                            raise ContextError('The saved scene and Auto file decisions disagree. Backups were retained.')
                        file_review = {'files': [], 'signature': 'committed'}
                    else:
                        file_review = files.review()
                except (FileRecoveryError, OSError) as exc:
                    raise ContextError(str(exc)) from exc
            yield archive, files, record, attempts, file_review
        finally:
            if files is not None:
                files.release()

    @classmethod
    def review(cls, ctx):
        with cls._recovery_inputs(ctx) as (archive, files, record, attempts, file_review):
            completed = record.get('auto_completion') is not None
            restored = sorted((row['role'], row['name']) for row in archive.snapshot['objects'].values()) if archive is not None and not completed else []
            return {
                'asset': ctx.source_object.name,
                'action': 'Finish completed Auto cleanup:' if completed else '',
                'restoring': archive is not None and not completed,
                'restore': restored,
                'remove_label': 'Retire previous build object' if completed else 'Remove unfinished result',
                'remove': (sorted(row['object'].name for row in archive.snapshot['objects'].values()
                                  if row.get('object') is not None)
                           if completed and archive is not None else sorted(obj.name for obj in attempts)),
                'files': file_review['files'],
                'signature': (_freeze(ctx.scene), _freeze(record), file_review['signature']),
            }

    @classmethod
    def recover(cls, ctx, *, reviewed_signature=None):
        from .cancel_files import FileRecoveryError
        with cls._recovery_inputs(ctx) as (archive, files, record, attempts, file_review):
            signature = (_freeze(ctx.scene), _freeze(record), file_review['signature'])
            if reviewed_signature is not None and signature != reviewed_signature:
                raise ContextError('The recovery inputs changed. Review this build again before recovering.')
            owner = ctx._ownership
            if archive is not None:
                owner.archive = ctx.auto_archive = archive
                archive.started = True
            else:
                owner.file_recovery = files
            try:
                if record.get('auto_completion') is not None:
                    cleanup_completed(ctx, files, archive)
                    return
                if files is not None:
                    try:
                        files.rollback()
                    except (FileRecoveryError, OSError) as exc:
                        raise ContextError(str(exc)) from exc
                if archive is not None:
                    archive.restore()
                else:
                    from .stage_operation import discard_borrowed
                    discard_borrowed(ctx)
                    ctx.reset_generated()
                record = owner.record()
                if 'file_journal' in record:
                    del record['file_journal']
            finally:
                owner.file_recovery = None
                owner.archive = None
                ctx.auto_archive = None

    def restore(self):
        attempts = self.validate_restore()
        owner = self.ctx._ownership
        record, snapshot = owner.record(), self.snapshot
        owner._root()['version'] = max(3, owner._root()['version'])
        if 'rollback_objects' not in snapshot:
            snapshot['rollback_objects'] = {}
        captured = {row.get('object') for row in snapshot['rollback_objects'].values()}
        for obj in attempts - captured:
            row = {'object': obj}
            if obj.type == 'MESH':
                row['mesh'] = obj.data
            snapshot['rollback_objects'][uuid.uuid4().hex] = row
        # Parent-held rows acquire the IDs before the child drops its snapshot.
        from .stage_operation import discard_borrowed
        discard_borrowed(self.ctx)
        snapshot['phase'] = 'restoring'
        if 'rollback_resources' not in snapshot:
            owner._root()['version'] = max(3, owner._root()['version'])
            snapshot['rollback_resources'] = self.ctx.resources.snapshot()
        meshes = {}
        for row in snapshot['rollback_objects'].values():
            obj, mesh = row.get('object'), row.get('mesh')
            if live(obj, bpy.data.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
            if live(mesh, bpy.data.meshes):
                meshes[mesh] = meshes.get(mesh, 0) + 1
        for mesh, references in meshes.items():
            if mesh.users == references:
                bpy.data.meshes.remove(mesh)
        for key, item in list(record['objects'].items()):
            if item.get('object') is None or item['object'] in attempts:
                del record['objects'][key]
        rows = list(snapshot['objects'].values())
        for row in rows:
            obj = row['object']
            obj.name = row['name']
            if row['mesh_name'] and obj.data.name != row['mesh_name']:
                obj.data.name = row['mesh_name']
            for collection in self._destinations(row):
                if collection.objects.get(obj.name) != obj:
                    collection.objects.link(obj)
            if not any(item.get('object') == obj for item in record['objects'].values()):
                record['objects'][str(obj.session_uid)] = {'object': obj, 'role': row['role']}
        for row in snapshot['materials'].values():
            if row['material'].name != row['name']:
                row['material'].name = row['name']
        source = self.ctx.source_object if self.ctx.exists() else None
        if source is not None:
            for key in STATE_KEYS:
                if key in snapshot['state']:
                    source[key] = snapshot['state'][key]
                elif key in source:
                    del source[key]
        layer = bpy.context.view_layer
        layer.update()
        if _in_layer(source, layer):
            source.hide_set(snapshot['source_hidden'])
        for row in rows:
            row['object'].hide_set(row['hidden'])
        for obj in list(bpy.context.selected_objects):
            obj.select_set(False)
        for obj in snapshot['selected'].values():
            if _in_layer(obj, layer):
                obj.select_set(True)
        if _in_layer(snapshot.get('active'), layer):
            layer.objects.active = snapshot['active']
        old_collection = snapshot.get('work_collection')
        current_collection = record.get('collection')
        if current_collection != old_collection:
            if current_collection is not None and not current_collection.objects and not current_collection.children:
                bpy.data.collections.remove(current_collection)
            if old_collection is not None:
                record['collection'] = old_collection
            elif 'collection' in record:
                del record['collection']
        owner._root()['version'] = max(2, owner._root()['version'])
        record['resources'] = copy_rows(snapshot.get('resources', {}))
        retire_rows(snapshot['rollback_resources'])
        self._release()

    def _release(self):
        owner = self.ctx._ownership
        record = owner.record()
        del record['archive']
        owner.archive = None
        self.started = False

    def validate_commit(self):
        """Preflight scene retirement while the previous result is still intact."""
        owner = self.ctx._ownership
        owner.validate_source_scope()
        owner._check_scope(owner.objects(), owner.record().get('collection'))
        snapshot = self.snapshot
        for row in snapshot['objects'].values():
            obj = row.get('object')
            if obj is None and snapshot['phase'] == 'committing':
                continue
            if not live(obj, bpy.data.objects) or obj.users_scene or not obj.is_editable:
                raise ContextError('An archived original changed ownership. Backup data was retained.')
            if row['mesh_name'] and obj.data != row.get('mesh'):
                raise ContextError('An archived original mesh was replaced. Backup data was retained.')

    def commit(self):
        """Retire old scene IDs only after the enclosing file decision is known."""
        self.validate_commit()
        snapshot = self.snapshot
        descriptor = snapshot.get('file_journal')
        completion = self.ctx._ownership.record().get('auto_completion')
        if descriptor is not None and (completion is None or
                dict(completion['journal']) != dict(descriptor)):
            raise ContextError('The Auto file decision must commit before retiring the previous scene.')
        snapshot['phase'] = 'committing'
        meshes = {}
        for row in snapshot['objects'].values():
            obj, mesh = row.get('object'), row.get('mesh')
            if live(obj, bpy.data.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
            if live(mesh, bpy.data.meshes):
                meshes[mesh] = meshes.get(mesh, 0) + 1
        for mesh, references in meshes.items():
            if mesh.users == references:
                bpy.data.meshes.remove(mesh)
        # Release incidental snapshot references; creation rows remain durable
        # until each deletion succeeds or an artist's remaining users protect it.
        snapshot['materials'] = {}
        snapshot['images'] = {}
        if 'resources' in snapshot:
            # Cached stages can retain the previous shader/images on the new
            # result. Transfer their creation authority before dropping any
            # archive references; collect only IDs with no remaining consumer.
            record = self.ctx._ownership.record()
            resources = copy_rows(snapshot['resources'])
            resources.update(self.ctx.resources.snapshot())
            record['resources'] = resources
            snapshot['resources'] = {}
            self.ctx.resources.collect_unused()
        self._release()


def cleanup_completed(ctx, files, archive=None):
    """Retry retirement without undoing a completed scene or texture set."""
    record = ctx._ownership.record()
    completion = record.get('auto_completion')
    if completion is None or dict(completion['journal']) != files.descriptor:
        raise ContextError('The completed Auto scene has no matching file decision.')
    if files.phase != 'committed':
        raise ContextError('The scene and Auto file decisions disagree. Backups were retained.')
    # A reopened visible marker must also cross the durability barrier before
    # destructive scene retirement. commit() confirms an existing terminal marker.
    files.commit()
    if archive is not None:
        archive.commit()
    files.cleanup()
    if 'file_journal' in record:
        del record['file_journal']
    del record['auto_completion']
    ctx.auto_archive = None
    ctx.cancel_files = None


def complete_auto(ctx):
    """Commit files first, then perform retryable scene and backup retirement."""
    files, archive = ctx.cancel_files, ctx.auto_archive
    if files is None:
        raise ContextError('The Auto file transaction is missing; previous scene data was retained.')
    if archive is not None:
        archive.validate_commit()
    files.commit()
    try:
        owner = ctx._ownership
        owner._root()['version'] = max(3, owner._root()['version'])
        owner.record()['auto_completion'] = {'version': 1, 'journal': files.descriptor}
        cleanup_completed(ctx, files, archive)
    except Exception as error:
        # All requested work and the file decision succeeded. A retirement error
        # cannot justify reverting the scene against the committed texture set.
        ctx.operation_cleanup_error = str(error)
        ctx._ownership.archive = None
        ctx.auto_archive = None
