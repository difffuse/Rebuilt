"""Qualified image lookups and export keys, including Blender library collisions."""
import bpy


def image_from_entry(entry):
    name = entry.get('image', '')
    return bpy.data.images.get((name, entry.get('image_library'))) if name else None


def map_key(entry):
    # One image can serve several roles with different files/import settings.
    role = ('map', entry['image_ref']) if 'image_ref' in entry else ('channel', entry['map'])
    return (*role, (entry.get('image', ''), entry.get('image_library')))


def node_key(node):
    reference = node.get('rebuilt_map_ref')
    image = node.image
    identity = (image.name, image.library.filepath if image.library else None)
    if reference:
        return ('map', reference, identity)
    channel = node.get('rebuilt_map_channel') or node.label.replace(' ', '_').upper()
    return ('channel', channel, identity)


def describe(entry, image):
    entry['image'] = image.name if image is not None else ''
    if image is not None and image.library is not None:
        entry['image_library'] = image.library.filepath
    else:
        entry.pop('image_library', None)


def node_path(paths, node):
    key = node_key(node)
    if key in paths:
        return paths[key]
    if not node.get('rebuilt_map_ref'):
        # Old generated nodes lack semantic keys. A unique exact image binding
        # retains their previous behavior even if the artist changed the label.
        candidates = {path for role, channel, identity in paths if identity == key[2]
                      for path in (paths[(role, channel, identity)],)}
        if len(candidates) == 1:
            return candidates.pop()
        if len(candidates) > 1:
            raise RuntimeError('A legacy texture has ambiguous map roles. Restore its channel label or rebake before exporting.')
    return None
