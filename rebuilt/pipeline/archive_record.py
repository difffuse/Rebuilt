"""Versioned Blender-held snapshot for restoring an interrupted Auto build."""
import bpy
import json
from idprop.types import IDPropertyGroup

from .context import ContextError, KEY_RECIPE, KEY_REPORTS, KEY_CACHE, ROLE_SOURCE

VERSION = 2
STATE_KEYS = (KEY_RECIPE, KEY_REPORTS, KEY_CACHE)


def _group(value):
    return isinstance(value, (dict, IDPropertyGroup))


def _require(condition, message):
    if not condition:
        raise ContextError('Invalid Auto recovery snapshot: ' + message + '. Backup data was retained.')


def _flag(value):
    return type(value) in (bool, int) and value in (0, 1)


def validate_state(state):
    """Validate source settings before a transaction consumes or restores them."""
    _require(all(key in STATE_KEYS and isinstance(value, str) for key, value in state.items()), 'invalid source state')
    from ..core.recipe import AssetRecipe, RecipeMigrationError
    for key, value in state.items():
        try:
            payload = json.loads(value)
            _require(isinstance(payload, dict), key + ' must contain an object')
            if key == KEY_RECIPE:
                AssetRecipe.from_dict(payload)
            elif key == KEY_CACHE:
                _require(all(isinstance(value, str) for value in payload.values()), 'invalid cached fingerprints')
            elif key == KEY_REPORTS:
                _require(all(isinstance(value, dict) for value in payload.values()), 'invalid stage reports')
        except (ValueError, RecipeMigrationError) as exc:
            raise ContextError('Invalid Auto recovery source state: ' + str(exc) + '. Backup data was retained.') from exc


def validate(snapshot):
    """Validate structure without mistaking deleted ID references for other IDs."""
    _require(_group(snapshot), 'record is not a group')
    _require(type(snapshot.get('version')) is int and snapshot['version'] in (1, VERSION), 'unsupported version')
    if snapshot['version'] >= 2 or 'resources' in snapshot:
        from .resources import validate as validate_resources
        validate_resources(snapshot.get('resources'))
    if 'rollback_resources' in snapshot:
        from .resources import validate as validate_resources
        validate_resources(snapshot['rollback_resources'])
    _require(snapshot.get('phase') in ('starting', 'running', 'restoring', 'committing'), 'unknown phase')
    for key in ('objects', 'materials', 'images', 'state', 'selected'):
        _require(_group(snapshot.get(key)), key + ' is not a group')
    _require(_flag(snapshot.get('source_hidden')) and _flag(snapshot.get('files_touched')), 'invalid flags')
    if snapshot.get('file_journal') is not None:
        from .cancel_files import validate_descriptor, FileRecoveryError
        try:
            validate_descriptor(snapshot['file_journal'])
        except FileRecoveryError as exc:
            raise ContextError(str(exc)) from exc
    validate_state(snapshot['state'])
    _require(snapshot.get('active') is None or isinstance(snapshot['active'], bpy.types.Object), 'invalid active object')
    _require(snapshot.get('work_collection') is None or isinstance(snapshot['work_collection'], bpy.types.Collection), 'invalid working collection')
    _require(all(value is None or isinstance(value, bpy.types.Object) for value in snapshot['selected'].values()), 'invalid selection')
    originals, replacements = set(), set()
    for row in snapshot['objects'].values():
        _require(_group(row), 'invalid object row')
        for key in ('name', 'role', 'mesh_name'):
            _require(isinstance(row.get(key), str), 'invalid object ' + key)
        _require(bool(row['name']) and bool(row['role']) and row['role'] != ROLE_SOURCE, 'invalid object identity')
        _require(_flag(row.get('hidden')) and _flag(row.get('scene_root')), 'invalid object flags')
        _require(_group(row.get('collections')), 'invalid destination collections')
        _require(all(value is None or isinstance(value, bpy.types.Collection) for value in row['collections'].values()), 'invalid collection reference')
        for key, seen in (('object', originals), ('replacement', replacements)):
            obj = row.get(key)
            if obj is not None:
                _require(isinstance(obj, bpy.types.Object) and obj not in seen, 'invalid or duplicate ' + key)
                seen.add(obj)
        _require(row.get('mesh') is None or isinstance(row['mesh'], bpy.types.Mesh), 'invalid original mesh')
    _require(not originals.intersection(replacements), 'original is also a replacement')
    if 'rollback_objects' in snapshot:
        _require(_group(snapshot['rollback_objects']), 'invalid rollback objects')
        pending = set()
        for row in snapshot['rollback_objects'].values():
            _require(_group(row), 'invalid rollback object row')
            obj = row.get('object')
            _require(obj is None or (isinstance(obj, bpy.types.Object)
                     and obj not in originals and obj not in pending), 'invalid rollback object')
            if obj is not None:
                pending.add(obj)
            _require(row.get('mesh') is None or isinstance(row['mesh'], bpy.types.Mesh), 'invalid rollback mesh')
    materials = set()
    for row in snapshot['materials'].values():
        _require(_group(row) and isinstance(row.get('name'), str), 'invalid material row')
        material = row.get('material')
        if material is not None:
            _require(isinstance(material, bpy.types.Material) and material not in materials, 'invalid or duplicate material')
            materials.add(material)
    _require(all(value is None or isinstance(value, bpy.types.Image) for value in snapshot['images'].values()), 'invalid image reference')


def capture(ctx, originals, materials, images):
    """Capture all restoration inputs before renaming or unlinking anything."""
    source = ctx.source_object
    roles = {item['object']: item['role'] for item in ctx._ownership.record()['objects'].values()
             if item.get('object') is not None}
    snapshot = {
        'version': VERSION, 'phase': 'starting', 'files_touched': False,
        'source_hidden': source.hide_get(),
        'state': {key: source[key] for key in STATE_KEYS if key in source},
        'resources': ctx.resources.snapshot(),
        'selected': {str(i): obj for i, obj in enumerate(bpy.context.selected_objects)},
        'objects': {},
        'materials': {str(i): {'material': material, 'name': name}
                      for i, (material, name) in enumerate(materials)},
        'images': {str(i): image for i, image in enumerate(images)},
    }
    active = bpy.context.view_layer.objects.active
    if active is not None:
        snapshot['active'] = active
    collection = ctx._ownership.collection()
    if collection is not None:
        snapshot['work_collection'] = collection
    for index, obj in enumerate(originals):
        snapshot['objects'][str(index)] = {
            'object': obj, 'name': obj.name, 'role': roles[obj],
            'mesh_name': obj.data.name if obj.type == 'MESH' else '',
            'hidden': obj.hide_get(),
            'scene_root': ctx.scene.collection in obj.users_collection[:],
            'collections': {str(i): collection for i, collection in enumerate(obj.users_collection)
                            if collection != ctx.scene.collection},
        }
        if obj.type == 'MESH':
            snapshot['objects'][str(index)]['mesh'] = obj.data
    validate(snapshot)
    return snapshot
