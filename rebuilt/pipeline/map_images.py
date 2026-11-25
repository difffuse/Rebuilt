"""Read-only map identity, carried by the delivered object independently of ownership."""
import bpy
from idprop.types import IDPropertyGroup

from .context import ContextError
from ..bridge.image_refs import image_from_entry, describe

KEY_MAP_IMAGES = 'rebuilt_map_images'
VERSION = 1
KEY_NODE_REF = 'rebuilt_map_ref'


def references(low):
    record = low.get(KEY_MAP_IMAGES) if low is not None else None
    if record is None:
        return {}
    if (not isinstance(record, (dict, IDPropertyGroup))
            or type(record.get('version')) is not int or record['version'] != VERSION
            or not isinstance(record.get('images'), (dict, IDPropertyGroup))):
        raise ContextError('Invalid baked-image reference record.')
    images = dict(record['images'])
    if any(not isinstance(key, str) or not key
           or (image is not None and not isinstance(image, bpy.types.Image))
           for key, image in images.items()):
        raise ContextError('Invalid baked-image reference.')
    return images


def resolve(maps, images, *, required=False):
    """Normalize current names for adapters; missing exact IDs never use labels."""
    result = []
    for original in maps:
        entry = dict(original)
        if entry.get('baked'):
            if 'image_ref' in entry:
                key = entry['image_ref']
                if not isinstance(key, str) or not key:
                    raise ContextError('Invalid baked-image reference key.')
                image = images.get(key)
                describe(entry, image)
            else:
                # Legacy names remain read compatibility, not creation authority.
                image = image_from_entry(entry)
            if required and image is None:
                raise ContextError(f"Baked texture {entry.get('map', '')!r} is missing. Restore it or bake again.")
        result.append(entry)
    return result


def bind(low, maps):
    """Retain exact new-bake references on the object that can be delivered alone."""
    images = {}
    for entry in maps:
        if entry.get('baked') and 'image_ref' in entry:
            image = image_from_entry(entry)
            if image is None:
                raise ContextError('Cannot retain a missing baked texture.')
            images[entry['image_ref']] = image
    low[KEY_MAP_IMAGES] = {'version': VERSION, 'images': images}


def for_export(low, maps):
    """Read deliberate relinks from stable generated-node keys, without editing IDs."""
    result = []
    for original in maps:
        entry = dict(original)
        if entry.get('baked') and 'image_ref' in entry:
            matches = set()
            for slot in low.material_slots:
                material = slot.material
                if material is None or not material.use_nodes:
                    continue
                tree = material.node_tree
                reachable = set()
                pending = [node for node in tree.nodes
                           if node.type == 'OUTPUT_MATERIAL' and node.is_active_output]
                while pending:
                    node = pending.pop()
                    if node in reachable:
                        continue
                    reachable.add(node)
                    pending.extend(link.from_node for socket in node.inputs for link in socket.links)
                matches.update(node.image for node in reachable
                               if node.type == 'TEX_IMAGE' and node.get(KEY_NODE_REF) == entry['image_ref'])
            if not matches or matches == {None}:
                raise ContextError(f"Baked texture {entry.get('map', '')!r} is missing or disconnected. Restore its material connection before exporting.")
            if len(matches) != 1 or None in matches:
                raise ContextError(f"Baked texture {entry.get('map', '')!r} has conflicting material connections.")
            describe(entry, matches.pop())
        if entry.get('baked') and image_from_entry(entry) is None:
            raise ContextError(f"Baked texture {entry.get('map', '')!r} is missing.")
        result.append(entry)
    return result
