"""Keep a stage's prior Blender result until its file decision is durable."""
from contextlib import contextmanager
import uuid

import bpy
from idprop.types import IDPropertyGroup

from .context import ContextError, KEY_CACHE, KEY_EXPORT_READY, KEY_RECIPE, KEY_REPORTS
from .cancel_files import CancelFiles, FileCommitUncertain
from .ownership import live
from .resources import copy_rows


KEY_OPERATION = 'stage_operation'
STATE_KEYS = (KEY_CACHE, KEY_EXPORT_READY, KEY_RECIPE, KEY_REPORTS,
              'rebuilt_generated_material', 'rebuilt_map_images')


def _copy(value):
    if isinstance(value, (dict, IDPropertyGroup)):
        return {key: _copy(item) for key, item in value.items()}
    return value


def validate(record):
    """Validate structure; deleted references remain recognizable, never inferred."""
    def require(condition):
        if not condition:
            raise ContextError('Invalid stage recovery record. Backup data was retained.')
    group = lambda value: isinstance(value, (dict, IDPropertyGroup))
    require(group(record))
    require(type(record.get('version')) is int and record['version'] == 1)
    require(record.get('stage') in ('bake', 'output', 'preview'))
    require(record.get('phase') in ('active', 'restoring', 'rolled_back', 'committed'))
    require(isinstance(record.get('token'), str) and len(record['token']) == 32)
    require(isinstance(record.get('name'), str))
    require(type(record.get('owns_files')) in (bool, int) and record['owns_files'] in (0, 1))
    for key in ('low', 'previous', 'active'):
        require(record.get(key) is None or isinstance(record[key], bpy.types.Object))
    require(record.get('candidate') is None or isinstance(record['candidate'], bpy.types.Mesh))
    require(record.get('previous_mesh') is None or isinstance(record['previous_mesh'], bpy.types.Mesh))
    require(group(record.get('objects')) and group(record.get('retire')))
    require(group(record.get('source_state')) and group(record.get('visibility')))
    from .archive_record import validate_state
    validate_state(record['source_state'])
    from .resources import validate as validate_resources
    validate_resources(record.get('resources'))
    seen = set()
    for row in record['objects'].values():
        require(group(row) and isinstance(row.get('role'), str) and row['role'] != 'source')
        obj = row.get('object')
        require(obj is None or (isinstance(obj, bpy.types.Object) and obj not in seen))
        if obj is not None:
            seen.add(obj)
    for row in record['retire'].values():
        require(group(row))
        require(row.get('object') is None or row['object'] in seen)
        require(row.get('mesh') is None or isinstance(row['mesh'], bpy.types.Mesh))
    require(group(record.get('rollback', {})))
    for row in record.get('rollback', {}).values():
        require(group(row) and isinstance(row.get('role'), str) and row['role'] != 'source')
        obj = row.get('object')
        require(obj is None or (isinstance(obj, bpy.types.Object) and obj not in seen
                               and obj != record.get('previous')))
        if obj is not None:
            seen.add(obj)
        require(row.get('mesh') is None or isinstance(row['mesh'], bpy.types.Mesh))
    for row in record['visibility'].values():
        require(group(row) and (row.get('object') is None or row['object'] in seen))
        for key in ('hidden', 'render', 'selected'):
            require(type(row.get(key)) in (bool, int) and row[key] in (0, 1))
    from .cancel_files import validate_descriptor
    validate_descriptor(record.get('journal'))


def borrowed_objects(ctx):
    """Objects retained by a stage belong to the enclosing Auto attempt as well."""
    owned = ctx._ownership.record()
    record = owned.get(KEY_OPERATION) if owned is not None else None
    if record is None:
        return set()
    if record['owns_files'] or dict(record['journal']) != dict(owned.get('file_journal', {})):
        raise ContextError('The stage and Auto recovery journals disagree. Backups were retained.')
    previous = record.get('previous')
    if live(previous, bpy.data.objects) and previous.data != record.get('previous_mesh'):
        raise ContextError('The stage backup mesh was replaced. Recovery data was retained.')
    rows = [*record['objects'].values(), *record.get('rollback', {}).values()]
    for row in record.get('rollback', {}).values():
        obj = row.get('object')
        if obj is not None and obj.type == 'MESH' and obj.data != row.get('mesh'):
            raise ContextError('An unfinished stage mesh was replaced. Recovery data was retained.')
    objects = {row['object'] for row in rows if row.get('object') is not None}
    ctx._ownership.validate_source_scope()
    ctx._ownership._check_scope(objects, owned.get('collection'))
    return objects


def discard_borrowed(ctx):
    """Transfer retained stage IDs back to Auto before its outer rollback."""
    objects = borrowed_objects(ctx)
    owned = ctx._ownership.record()
    if owned is None or KEY_OPERATION not in owned:
        return
    record = owned[KEY_OPERATION]
    present = {row.get('object') for row in owned['objects'].values()}
    for row in [*record['objects'].values(), *record.get('rollback', {}).values()]:
        if row.get('object') in objects - present:
            owned['objects'][uuid.uuid4().hex] = {'object': row['object'], 'role': row['role']}
    resources = copy_rows(record['resources'])
    resources.update(ctx.resources.snapshot())
    owned['resources'] = resources
    StageOperation(ctx, record['stage'])._release_snapshot()


class StageOperation:
    def __init__(self, ctx, stage):
        self.ctx, self.stage = ctx, stage
        self.files = None
        self.borrowed = ctx.cancel_files

    @property
    def record(self):
        record = self.ctx._ownership.record()
        if record is None or KEY_OPERATION not in record:
            raise ContextError('The stage recovery record is unavailable.')
        return record[KEY_OPERATION]

    def start(self):
        ctx, owner = self.ctx, self.ctx._ownership
        owned = owner.ensure()
        if KEY_OPERATION in owned:
            raise ContextError('Recover the earlier stage operation before starting another.')
        ctx.finish_pending_retirement()
        low = ctx.artifact('low')
        if low is None:
            raise ContextError(f'{self.stage} needs retopo to have run first')
        token = uuid.uuid4().hex
        self.token = token
        if self.borrowed is None:
            self.files = CancelFiles(
                bpy.utils.user_resource('CONFIG', path='rebuilt/recovery', create=True),
                origin={'version': 1, 'kind': 'stage', 'operation': token,
                        'ownership': owner.token, 'stage': self.stage,
                        'blend': bpy.data.filepath})
            self.files.hold()
        else:
            self.files = self.borrowed
        previous = low.copy()
        previous.use_fake_user = False
        try:
            record = {
                'version': 1, 'token': token, 'stage': self.stage, 'phase': 'active',
                'owns_files': self.borrowed is None, 'journal': self.files.descriptor,
                'low': low, 'previous': previous, 'previous_mesh': low.data, 'name': low.name,
                'objects': _copy(owned['objects']), 'retire': {}, 'rollback': {},
                'resources': ctx.resources.snapshot(),
                'source_state': {key: ctx.source_object[key]
                                 for key in (KEY_RECIPE, KEY_REPORTS, KEY_CACHE)
                                 if key in ctx.source_object},
                'visibility': {key: {'object': row['object'],
                                    'hidden': row['object'].hide_get(),
                                    'render': row['object'].hide_render,
                                    'selected': row['object'].select_get()}
                               for key, row in owned['objects'].items()
                               if row.get('object') is not None},
            }
            if bpy.context.view_layer.objects.active is not None:
                record['active'] = bpy.context.view_layer.objects.active
            validate(record)
            owner._root()['version'] = 3
            owned[KEY_OPERATION] = record
        except BaseException:
            bpy.data.objects.remove(previous)
            if self.borrowed is None:
                self.files.release()
            raise
        owner.operation = self
        ctx._stage_operation = self
        ctx.cancel_files = self.files
        candidate = low.data.copy()
        candidate.use_fake_user = False
        self.record['candidate'] = candidate
        low.data = candidate
        for slot in low.material_slots:
            slot.link = 'DATA'
        if self.stage == 'bake':
            old = {row['object'] for row in self.record['objects'].values()
                   if row['role'] in ('bake', 'cage') and row.get('object') is not None}
            self.retire(old)
            owner.forget(old)

    def retire(self, objects):
        for obj in objects:
            rows = self.record['retire']
            if any(row.get('object') == obj for row in rows.values()):
                continue
            row = {'object': obj}
            if obj.type == 'MESH':
                row['mesh'] = obj.data
            rows[uuid.uuid4().hex] = row

    def require_private_mesh(self, obj):
        candidate = self.record.get('candidate')
        if (self.record.get('low') != obj or candidate != obj.data
                or candidate.users > 2 + int(candidate.use_fake_user)):
            raise ContextError('The private stage mesh was replaced or shared during application.')

    def preflight(self, *, restore=False):
        ctx, owner, record = self.ctx, self.ctx._ownership, self.record
        owner.validate_source_scope()
        low, previous = record.get('low'), record.get('previous')
        if live(previous, bpy.data.objects) and (
                not live(record.get('previous_mesh'), bpy.data.meshes)
                or previous.data != record['previous_mesh']):
            raise ContextError('The stage backup mesh was replaced. Recovery data was retained.')
        if record['phase'] in ('active', 'restoring'):
            if not live(low, bpy.data.objects) or not live(previous, bpy.data.objects):
                raise ContextError('The stage output or its backup was removed. Recovery data was retained.')
            if low.data not in (record.get('candidate'), previous.data):
                raise ContextError('The stage output mesh was replaced. Recovery data was retained.')
            originals = [row.get('object') for row in record['objects'].values()]
            if any(not live(obj, bpy.data.objects) for obj in originals):
                raise ContextError('A previous stage object was removed. Recovery data was retained.')
            owner._check_scope(originals, owner.record().get('collection'))
            if restore:
                from .auto_archive import _validate_name
                _validate_name(bpy.data.objects, low, record['name'], 'Object')
        owner._check_scope(owner.objects(), owner.record().get('collection'))
        owner._check_scope([row['object'] for row in record['retire'].values()
                            if row.get('object') is not None], None)
        for row in record.get('rollback', {}).values():
            obj = row.get('object')
            if obj is not None:
                owner._check_scope([obj], None)
                if obj.type == 'MESH' and obj.data != row.get('mesh'):
                    raise ContextError('An unfinished stage mesh was replaced. Recovery data was retained.')

    def _release_rollback_meshes(self):
        """Retire captured mesh IDs after their attempt objects release them."""
        meshes = {}
        for row in self.record.get('rollback', {}).values():
            mesh = row.get('mesh')
            if live(mesh, bpy.data.meshes):
                meshes[mesh] = meshes.get(mesh, 0) + 1
        for mesh, references in meshes.items():
            if mesh.users == references:
                bpy.data.meshes.remove(mesh)
        self.record['rollback'] = {}

    def rollback(self):
        self.preflight(restore=True)
        record, ctx = self.record, self.ctx
        if record['phase'] == 'committed':
            raise ContextError('The stage already committed; only cleanup may be retried.')
        if self.borrowed is None:
            self.files.rollback()
        record['phase'] = 'restoring'
        low, previous = record['low'], record['previous']
        low.data = record['previous_mesh']
        low.name = record['name']
        low.matrix_world = previous.matrix_world.copy()
        for slot, old in zip(low.material_slots, previous.material_slots):
            slot.link = old.link
            if old.link == 'OBJECT':
                slot.material = old.material
        for key in STATE_KEYS:
            if key in previous:
                low[key] = _copy(previous[key])
            elif key in low:
                del low[key]
        original = {row.get('object') for row in record['objects'].values()}
        if 'rollback' not in record:
            record['rollback'] = {}
        captured = {row.get('object') for row in record['rollback'].values()}
        for row in ctx._ownership.record()['objects'].values():
            obj = row.get('object')
            if obj is not None and obj not in original and obj not in captured:
                pending = {'object': obj, 'role': row['role']}
                if obj.type == 'MESH':
                    pending['mesh'] = obj.data
                record['rollback'][uuid.uuid4().hex] = pending
        for row in record['rollback'].values():
            if row.get('object') is not None:
                bpy.data.objects.remove(row['object'], do_unlink=True)
        self._release_rollback_meshes()
        ctx._ownership.record()['objects'] = _copy(record['objects'])
        if live(ctx._source, bpy.data.objects):
            for key in (KEY_RECIPE, KEY_REPORTS, KEY_CACHE):
                if key in record['source_state']:
                    ctx._source[key] = record['source_state'][key]
                elif key in ctx._source:
                    del ctx._source[key]
        for row in record['visibility'].values():
            obj = row['object']
            obj.hide_set(row['hidden'])
            obj.hide_render = row['render']
            obj.select_set(row['selected'])
        active = record.get('active')
        if live(active, bpy.data.objects) and bpy.context.view_layer.objects.get(active.name) == active:
            bpy.context.view_layer.objects.active = active
        candidate = record.get('candidate')
        if live(candidate, bpy.data.meshes) and candidate.users == 1:
            bpy.data.meshes.remove(candidate)
        if 'candidate' in record:
            del record['candidate']
        created = {row.get('id') for key, row in ctx.resources.snapshot().items()
                   if key not in record['resources']}
        ctx.resources.release(ids=created)
        # Snapshot references kept the old resources alive even if an intermediate
        # presentation step dropped their live ledger rows.
        ctx._ownership.record()['resources'] = copy_rows(record['resources'])
        record['phase'] = 'rolled_back'
        self.cleanup()

    def _drop_snapshot(self):
        record = self.record
        self._release_rollback_meshes()
        candidate = record.get('candidate')
        if live(candidate, bpy.data.meshes) and candidate.users == 1:
            bpy.data.meshes.remove(candidate)
        previous = record.get('previous')
        if live(previous, bpy.data.objects):
            if previous.data != record.get('previous_mesh'):
                raise ContextError('The stage backup mesh was replaced. Recovery data was retained.')
            if previous.users == 1:
                bpy.data.objects.remove(previous, do_unlink=True)
        if 'previous' in record:
            del record['previous']
        mesh = record.get('previous_mesh')
        if live(mesh, bpy.data.meshes) and mesh.users == 1:
            bpy.data.meshes.remove(mesh)
        for key in ('previous_mesh', 'candidate'):
            if key in record:
                del record[key]
        for key in ('objects', 'visibility', 'resources', 'retire'):
            record[key] = {}

    def _release_snapshot(self):
        self._drop_snapshot()
        del self.ctx._ownership.record()[KEY_OPERATION]

    def cleanup(self):
        self.preflight()
        record, ctx = self.record, self.ctx
        if record['phase'] not in ('committed', 'rolled_back'):
            raise ContextError('The stage has no committed decision; its backup was retained.')
        committed = record['phase'] == 'committed'
        for key in list(record['retire']) if committed else ():
            row = record['retire'][key]
            obj, mesh = row.get('object'), row.get('mesh')
            if live(obj, bpy.data.objects):
                ctx._ownership.forget({obj})
                bpy.data.objects.remove(obj, do_unlink=True)
            # The row is the final owned reference when no artist still uses it.
            # Keep it until deletion succeeds so a failed retirement is retryable.
            if live(mesh, bpy.data.meshes) and mesh.users == 1:
                bpy.data.meshes.remove(mesh)
            del record['retire'][key]
        if committed:
            collection = ctx._ownership.collection()
            if collection is not None and not collection.all_objects and not collection.children:
                bpy.data.collections.remove(collection)
                ctx._ownership.set_collection(None)
        if self.borrowed is None:
            self.files.cleanup()
        self._drop_snapshot()
        if committed:
            # Once the old mesh releases its preview users, the live creation
            # ledger can retire them, or preserve remaining artist users.
            from ..bridge import uv_preview
            uv_preview.clear(ctx)
            ctx.resources.collect_unused()
        del ctx._ownership.record()[KEY_OPERATION]

    def finish(self):
        self.preflight()
        if self.borrowed is None:
            self.files.commit()
        self.record['phase'] = 'committed'
        try:
            self.cleanup()
        except Exception as error:
            # A durable success cannot become a rollback because retirement failed.
            record = self.ctx._ownership.record()
            if record is not None and KEY_OPERATION in record:
                record[KEY_OPERATION]['cleanup_error'] = str(error)
            self.ctx.operation_cleanup_error = str(error)
            if self.borrowed is not None:
                raise ContextError('Auto stage retirement failed; the outer attempt must be recovered.') from error

    def release(self):
        self.ctx.cancel_files = self.borrowed
        self.ctx._stage_operation = None
        self.ctx._ownership.operation = None
        if self.borrowed is None and self.files is not None:
            self.files.release()

    @classmethod
    @contextmanager
    def _recovery_inputs(cls, ctx):
        from .job import active_runners
        from .auto_archive import _unique_recovery
        from .cancel_files import FileRecoveryError
        if active_runners(ctx.scene):
            raise ContextError('Finish or cancel the running job before recovering a stage.')
        owner = ctx._ownership
        owned = owner.record()
        record = owned.get(KEY_OPERATION) if owned is not None else None
        if record is None or not record['owns_files']:
            raise ContextError('This asset has no independent stage transaction to recover.')
        _unique_recovery(ctx, owned, record['journal'])
        operation = cls(ctx, record['stage'])
        operation.token = record['token']
        try:
            operation.files = CancelFiles.reopen(record['journal'])
            operation.files.hold()
            origin = operation.files.origin
            if any(origin.get(key) != value for key, value in {
                    'kind': 'stage', 'operation': record['token'],
                    'ownership': owner.token, 'stage': record['stage']}.items()):
                raise ContextError('The stage and file journal origins disagree. Backups were retained.')
            phase = operation.files.phase
            terminal = record['phase'] in ('committed', 'rolled_back')
            if terminal:
                if phase != record['phase']:
                    raise ContextError('The saved scene and file decisions disagree. Backups were retained.')
                file_review = {'files': [], 'signature': phase}
            else:
                # A scene saved before durable completion cannot be treated as
                # that completed scene merely because its journal is committed.
                file_review = operation.files.review()
            operation.preflight(restore=not terminal)
            owner.operation = ctx._stage_operation = operation
            ctx.cancel_files = operation.files
            yield operation, file_review
        except (FileRecoveryError, OSError) as error:
            raise ContextError(str(error)) from error
        finally:
            operation.release()

    @classmethod
    def review(cls, ctx):
        from .auto_archive import _freeze
        with cls._recovery_inputs(ctx) as (operation, files):
            record = operation.record
            terminal = record['phase'] in ('committed', 'rolled_back')
            original = {row.get('object') for row in record['objects'].values()}
            return {
                'kind': 'stage', 'asset': ctx.source_object.name,
                'action': ('Finish retained stage cleanup:' if terminal
                           else 'Restore the previous stage result and texture files:'),
                'restoring': not terminal,
                'restore': [] if terminal else [('low', record['name'])],
                'remove_label': 'Retire previous stage object' if terminal else 'Remove unfinished result',
                'remove': (sorted(row['object'].name for row in record['retire'].values()
                                  if row.get('object') is not None)
                           if record['phase'] == 'committed' else
                           sorted(obj.name for obj in ctx._ownership.objects() if obj not in original)
                           if not terminal else []),
                'files': files['files'],
                'signature': (_freeze(ctx.scene), _freeze(ctx._ownership.record()), files['signature']),
            }

    @classmethod
    def recover(cls, ctx, *, reviewed_signature=None):
        from .auto_archive import _freeze
        with cls._recovery_inputs(ctx) as (operation, files):
            signature = (_freeze(ctx.scene), _freeze(ctx._ownership.record()), files['signature'])
            if reviewed_signature is not None and signature != reviewed_signature:
                raise ContextError('The recovery inputs changed. Review the stage again before recovering.')
            if operation.record['phase'] in ('committed', 'rolled_back'):
                operation.cleanup()
            else:
                operation.rollback()


@contextmanager
def stage_operation(ctx, stage):
    """JobRunner owns apply plus cache; direct stage calls use the same boundary."""
    if stage not in ('bake', 'output', 'preview'):
        yield None
        return
    existing = getattr(ctx, '_stage_operation', None)
    if existing is not None:
        if stage not in (existing.stage, 'preview'):
            raise ContextError('A different stage operation is already active.')
        yield existing
        return
    operation = StageOperation(ctx, stage)
    try:
        try:
            operation.start()
            yield operation
            operation.finish()
        except FileCommitUncertain:
            # Keep both versions: no decision is safe until durability is known.
            raise
        except BaseException:
            owned = ctx._ownership.record()
            if (owned is not None and KEY_OPERATION in owned
                    and owned[KEY_OPERATION]['token'] == getattr(operation, 'token', None)
                    and operation.record['phase'] != 'committed'):
                operation.rollback()
            raise
    finally:
        operation.release()
