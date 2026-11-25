"""Export completed deliveries without adopting them as processing sources."""
from .context import (
    AssetContext, ContextError, KEY_ASSET, KEY_ROLE, KEY_RECIPE, KEY_REPORTS, KEY_EXPORT_READY,
    ROLE_DELIVERABLE, ROLE_SOURCE,
    _find_source,
)

class ExportContext(AssetContext):
    def __init__(self, scene, name, owner, low):
        super().__init__(scene, name, source=owner)
        self._owner = owner
        self._low = low

    @classmethod
    def find(cls, scene, name, selected=None):
        if not name:
            return None
        try:
            source = _find_source(scene, name)
        except ContextError:
            # A selected completed delivery has its own serialized export state.
            # Ambiguous source labels never authorize inheriting source state.
            source = None
        if selected is not None and selected.get(KEY_ASSET) != name:
            return None
        if selected is not None and selected.get(KEY_ROLE) == ROLE_DELIVERABLE:
            low = selected if selected.type == 'MESH' else None
        else:
            lows = [obj for obj in scene.objects if obj.type == 'MESH'
                    and obj.get(KEY_ASSET) == name and obj.get(KEY_ROLE) == ROLE_DELIVERABLE]
            low = lows[0] if len(lows) == 1 else None
        if low is not None:
            from .ownership import ArtifactOwnership, KEY_MARKER
            root = ArtifactOwnership(scene, None, name)._root()
            record = next((entry for entry in root['entries'].values()
                           if any(row.get('object') == low for row in entry['objects'].values())), None) if root else None
            if record is not None:
                source = record.get('source')
            elif source is not None and source.get(KEY_MARKER) is not None:
                # An independent delivery carries its own report and image IDs.
                # A same-tag processing source cannot replace that saved state.
                source = None
        owner = source if source is not None else low
        if owner is None:
            return None
        ctx = cls(scene, name, owner, low)
        return ctx if source is not None or ctx.ready else None

    @property
    def _state(self):
        return self._owner

    def artifact(self, role):
        return self._low if role == ROLE_DELIVERABLE else None

    @property
    def ready(self):
        report = self.report('output')
        if self._low is None or report is None:
            return False
        if self._owner.get(KEY_ROLE) == ROLE_SOURCE:
            return 'output' in self.cached_keys
        return bool(self._low.get(KEY_EXPORT_READY) or report.get('object') == self._low.name)

    def resolve_maps(self, maps, *, required=False):
        from .map_images import references, resolve
        return resolve(maps, references(self._low), required=required)

    def report(self, stage):
        report = super().report(stage)
        if stage != 'bake' or not report or self._low is None:
            return report
        for entry in report.get('maps', []):
            if 'image_ref' in entry:
                continue
            label = entry.get('map', '').replace('_', ' ').title()
            images = {node.image for material in self._low.data.materials
                      if material and material.use_nodes for node in material.node_tree.nodes
                      if node.type == 'TEX_IMAGE' and node.image and node.label == label}
            if len(images) == 1:
                from ..bridge.image_refs import describe
                describe(entry, images.pop())
        return report


def preserve_export_state(ctx):
    low = ctx.artifact(ROLE_DELIVERABLE)
    if low is None or ctx.report('output') is None:
        return
    low[KEY_RECIPE] = ctx.recipe.to_json()
    low[KEY_REPORTS] = ctx._state.get(KEY_REPORTS, '{}')
    low[KEY_EXPORT_READY] = True
