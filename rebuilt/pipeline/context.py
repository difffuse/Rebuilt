"""Where an asset's data lives in the scene, and how stages find each other's work."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import bpy

from ..core.recipe import AssetRecipe


WORK_SUFFIX = "_rebuilt_work"


KEY_ASSET = "rebuilt_asset"
KEY_SLOT = "rebuilt_slot"
KEY_ROLE = "rebuilt_role"
KEY_RECIPE = "rebuilt_recipe"
KEY_CACHE = "rebuilt_cache"
KEY_REPORTS = "rebuilt_reports"
KEY_WORK = "rebuilt_work"
KEY_EXPORT_READY = "rebuilt_export_ready"

ROLE_SOURCE = "source"


ROLE_DELIVERABLE = "low"


class ContextError(RuntimeError):
    """The scene is not in a state this operation can work with."""


class AssetContext:
    """Handle onto one asset's objects, recipe and reports."""

    def __init__(self, scene: bpy.types.Scene, name: str, *, source=None) -> None:
        self.scene = scene
        self.name = name
        self._source = source if source is not None else _find_source(scene, name)
        from .ownership import ArtifactOwnership
        self._ownership = ArtifactOwnership(scene, self._source, name)
        from .resources import ResourceLedger
        self.resources = ResourceLedger(self._ownership)
        self.cancel_files = None
        self.auto_archive = None
        self._stage_operation = None
        self.operation_cleanup_error = ''

        self.progress: Callable[[float, str], None] = lambda fraction, message: None

        self.recipe_for_class: Callable[[str], AssetRecipe] | None = None

    @classmethod
    def create(
        cls,
        scene: bpy.types.Scene,
        source: bpy.types.Object,
        name: str | None = None,
    ) -> AssetContext:
        """Adopt `source` as the scan for a new asset."""
        if source.type != "MESH":
            raise ContextError(f"{source.name!r} is a {source.type}, not a mesh")

        if not source.is_editable or len(source.users_scene) != 1 or source.users_scene[0] != scene:
            raise ContextError('Use a local source that belongs to this scene only.')
        ctx = cls(scene, name or source.name, source=source)
        existing = ctx._ownership.record()
        if existing is not None:
            asset = source.get(KEY_ASSET)
            if name is not None and name != asset:
                raise ContextError('An owned asset cannot be renamed by adopting it again.')
            ctx.name = ctx._ownership.asset = asset
        ctx._ownership.ensure(new=True)
        source[KEY_ASSET] = ctx.name
        source[KEY_ROLE] = ROLE_SOURCE
        if existing is None:
            for key in (KEY_CACHE, KEY_REPORTS, KEY_EXPORT_READY):
                if key in source:
                    del source[key]
        return ctx

    @classmethod
    def find(cls, scene: bpy.types.Scene, name: str) -> AssetContext | None:
        """An existing asset by name, or None."""
        return cls(scene, name) if _find_source(scene, name) is not None else None

    @classmethod
    def for_object(cls, scene, obj):
        """Resolve an explicit selection by identity before using legacy labels."""
        if obj is None or not obj.get(KEY_ASSET):
            return None
        name = obj[KEY_ASSET]
        if obj.get(KEY_ROLE) == ROLE_SOURCE:
            return cls(scene, name, source=obj)
        from .ownership import ArtifactOwnership, KEY_MARKER
        owner = ArtifactOwnership(scene, None, name)
        root = owner._root()
        if root is not None:
            for entry in root['entries'].values():
                if any(item.get('object') == obj for item in entry['objects'].values()):
                    source = entry.get('source')
                    if source is None:
                        raise ContextError('The original source was deleted. This result can still be exported if it is complete.')
                    return cls(scene, source[KEY_ASSET], source=source)
        source = _find_source(scene, name)
        if source is not None and source.get(KEY_MARKER) is None:
            return cls(scene, name, source=source)
        raise ContextError('This object is an independent copy, not an owned processing result. Use Selected Model to start a new asset.')

    @classmethod
    def list_assets(cls, scene: bpy.types.Scene) -> list[str]:
        return sorted(
            {
                str(obj[KEY_ASSET])
                for obj in scene.objects
                if obj.get(KEY_ROLE) == ROLE_SOURCE and KEY_ASSET in obj
            }
        )

    def exists(self) -> bool:
        try:
            self.source_object
            return True
        except ContextError:
            return False

    @property
    def work_collection(self) -> bpy.types.Collection:
        """The one collection holding work in progress, created on demand."""
        self._ownership.ensure()
        collection = self._ownership.collection()
        if collection is not None:
            return collection
        collection = bpy.data.collections.new(f"{self.name}{WORK_SUFFIX}")
        collection[KEY_WORK] = self.name
        self.scene.collection.children.link(collection)
        self._ownership.set_collection(collection)
        return collection

    def collection(self, slot: str) -> bpy.types.Collection:
        """Where a stage's output belongs."""
        return self.work_collection

    @property
    def source_object(self) -> bpy.types.Object:
        # Tags discover an asset when a context is created. They cannot transfer
        # an existing operation to another object copied from the original.
        source = self._source
        try:
            if (source is not None and self.scene.objects.get(source.name) == source
                    and source.get(KEY_ASSET) == self.name and source.get(KEY_ROLE) == ROLE_SOURCE):
                return source
        except ReferenceError:
            pass  # Blender invalidates the handle when its object/scene is removed.
        raise ContextError(f"asset {self.name!r} original source is no longer available")

    def _tagged(self, role: str) -> list[bpy.types.Object]:
        return self._ownership.objects(role)

    def require_owned(self, *, readonly=False):
        self.source_object
        self._ownership.validate(readonly=readonly)

    def ownership_diagnostic(self):
        try:
            self.require_owned()
        except ContextError as error:
            return str(error)
        return None

    def artifact(self, role: str) -> bpy.types.Object | None:
        """The object a given stage produced, if it still exists."""
        found = self._tagged(role)
        return found[0] if found else None

    def artifacts(self, role: str) -> list[bpy.types.Object]:
        """Every object tagged with this role."""
        return self._tagged(role)

    def register_artifact(self, obj: bpy.types.Object, role: str, slot: str) -> None:
        """Tag an object as a stage's output and file it where it belongs."""
        if role == ROLE_SOURCE:
            raise ContextError("the source role is reserved for the user's own model")

        self._ownership.check_registration(obj, role)

        if role == ROLE_DELIVERABLE:
            from .map_images import KEY_MAP_IMAGES
            for key in (KEY_REPORTS, KEY_EXPORT_READY, KEY_MAP_IMAGES):
                if key in obj:
                    del obj[key]
            targets = list(self.source_object.users_collection) or [self.scene.collection]
        else:
            targets = [self.work_collection]

        obj[KEY_ASSET] = self.name
        obj[KEY_ROLE] = role
        obj[KEY_SLOT] = slot
        for collection in list(obj.users_collection):
            collection.objects.unlink(obj)
        for collection in targets:
            collection.objects.link(obj)
        self._ownership.add(obj, role)

    def finish_pending_retirement(self):
        """Complete existing deletion intent before another transaction snapshots it."""
        record = self._ownership.ensure()
        roles = {row['role'] for row in record['objects'].values()
                 if row.get('object') is None or 'retiring_mesh' in row}
        if not roles:
            return
        self._ownership.retire(roles, pending_only=True)
        if ROLE_DELIVERABLE in roles and self.artifact(ROLE_DELIVERABLE) is None:
            from ..bridge import uv_preview
            uv_preview.clear(self)

    def clear_artifacts(self, role: str) -> int:
        """Delete this stage's own output. Returns how many objects went."""
        removed = self._ownership.retire({role})
        if role == ROLE_DELIVERABLE:
            from ..bridge import uv_preview

            uv_preview.clear(self)
        return removed

    def reset_generated(self) -> None:
        """Return a cancelled Auto asset to its source, leaving other assets alone."""
        self.clear_artifacts(ROLE_DELIVERABLE)
        self.discard_preparation()

        self.resources.release()

    def discard_preparation(self) -> None:
        """Remove an abandoned first attempt, preserving source and user data."""
        self._ownership.ensure()
        if self.artifact(ROLE_DELIVERABLE) is not None:
            return
        for role in ("working", "cleaned", "proxy", "cage"):
            self.clear_artifacts(role)
        owned_collection = self._ownership.collection()
        for collection in [owned_collection] if owned_collection is not None else []:
            if (
                collection.get(KEY_WORK) == self.name
                and not collection.objects
                and not collection.children
                and not (
                    self.auto_archive is not None and self.auto_archive.keeps_collection(collection)
                )
            ):
                bpy.data.collections.remove(collection)
                self._ownership.set_collection(None)
        # Generated resources still belong to this attempt after source deletion.
        # Source metadata and selection do not belong to a same-tag replacement.
        if not self.exists():
            return
        self.invalidate_cache()
        stored = self._state.get(KEY_REPORTS)
        reports = json.loads(stored) if stored else {}
        for stage in (
            "ingest",
            "cleanup",
            "analyze",
            "retopo",
            "seams",
            "uv",
            "bake",
            "qa",
            "output",
            "export",
        ):
            reports.pop(stage, None)
        self._state[KEY_REPORTS] = json.dumps(reports, sort_keys=True)

        layer = bpy.context.view_layer
        layer.update()
        source = self.source_object
        if source.name in layer.objects:
            for obj in list(bpy.context.selected_objects):
                obj.select_set(False)
            source.hide_set(False)
            source.select_set(True)
            layer.objects.active = source

    def deliver(self, *, preserve_working: bool = False) -> list[str]:
        """Hand the asset over and take the scaffolding away."""
        self._ownership.ensure()
        collection = self._ownership.collection()
        if collection is None:
            return []

        removed: list[str] = []
        temporary = {obj: role for role in ('working', 'cleaned', 'proxy', 'cage')
                     for obj in self._ownership.objects(role)}
        doomed = set()
        for obj in list(collection.all_objects):
            if obj not in temporary:
                continue

            if preserve_working and temporary[obj] in {"working", "cleaned", "proxy"}:
                obj.hide_set(True)
                obj.hide_render = True
                continue
            removed.append(obj.name)
            if self._stage_operation is not None:
                self._stage_operation.retire([obj])
                continue
            doomed.add(obj)

        if self._stage_operation is None:
            roles = {'cage'} if preserve_working else {'working', 'cleaned', 'proxy', 'cage'}
            self._ownership.retire(roles, objects=doomed)

        if not collection.all_objects and not collection.children:
            bpy.data.collections.remove(collection)
            self._ownership.set_collection(None)
        return removed

    def can_finish_here(self) -> bool:
        """A partial Manual result still has temporary work to clean up."""
        cached = self.cached_keys
        if (
            "retopo" not in cached
            or {"bake", "output"} & cached.keys()
            or self.artifact(ROLE_DELIVERABLE) is None
        ):
            return False
        collection = self._ownership.collection()
        return collection is not None and (
            (not collection.all_objects and not collection.children)
            or any(self._ownership.objects(role) for role in ("working", "cleaned", "proxy", "cage"))
        )

    def finish_here(self) -> list[str]:
        """Keep the partial result and its cache for a later Manual continuation."""
        if not self.can_finish_here():
            raise ContextError("Finish Here needs an intermediate Retopo, Seams or Unwrap result.")
        removed = self.deliver()
        self.set_report(
            "finish_here",
            {
                "stage": next(
                    stage for stage in ("uv", "seams", "retopo") if stage in self.cached_keys
                ),
                "object": self.artifact(ROLE_DELIVERABLE).name,
                "removed": removed,
            },
        )
        return removed

    @property
    def _state(self) -> bpy.types.Object:
        return self.source_object

    @property
    def recipe(self) -> AssetRecipe:
        stored = self._state.get(KEY_RECIPE)
        return AssetRecipe.from_json(stored) if stored else AssetRecipe()

    @recipe.setter
    def recipe(self, value: AssetRecipe) -> None:
        self._state[KEY_RECIPE] = value.to_json()

    @property
    def cached_keys(self) -> dict[str, str]:
        """Fingerprints of the settings each stage last actually ran with."""
        stored = self._state.get(KEY_CACHE)
        return json.loads(stored) if stored else {}

    def set_cached_key(self, stage: str, key: str) -> None:
        keys = self.cached_keys
        keys[stage] = key
        self._state[KEY_CACHE] = json.dumps(keys, sort_keys=True)

    def invalidate_cache(self, stages: frozenset[str] | None = None) -> None:
        """Forget cached fingerprints, so those stages run again."""
        if stages is None:
            self._state[KEY_CACHE] = "{}"
            return
        keys = {k: v for k, v in self.cached_keys.items() if k not in stages}
        self._state[KEY_CACHE] = json.dumps(keys, sort_keys=True)

    def report(self, stage: str) -> dict | None:
        """What a stage found last time it ran, if anything."""
        stored = self._state.get(KEY_REPORTS)
        report = json.loads(stored).get(stage) if stored else None
        if stage == 'bake' and report and 'maps' in report:
            report['maps'] = self.resolve_maps(report['maps'])
        return report

    def resolve_maps(self, maps, *, required=False):
        from .map_images import references, resolve
        images = {key: row.get('id') for key, row in self.resources.snapshot().items()
                  if row['kind'] == 'image'}
        images.update(references(self.artifact(ROLE_DELIVERABLE)))
        return resolve(maps, images, required=required)

    def set_report(self, stage: str, payload: dict) -> None:
        """Record a stage's findings so the UI can show them without re-running."""
        stored = self._state.get(KEY_REPORTS)
        reports = json.loads(stored) if stored else {}
        reports[stage] = payload
        self._state[KEY_REPORTS] = json.dumps(reports, sort_keys=True)

    @property
    def job_state_path(self) -> Path:
        """Where to leave breadcrumbs for a job in progress."""
        blend = bpy.data.filepath
        base = Path(blend).parent / ".rebuilt" if blend else Path(bpy.app.tempdir)
        return base / f"{self.name}.job.json"


def _find_source(scene: bpy.types.Scene, name: str) -> bpy.types.Object | None:
    matches = [obj for obj in scene.objects if obj.get(KEY_ROLE) == ROLE_SOURCE and obj.get(KEY_ASSET) == name]
    if len(matches) > 1:
        raise ContextError('More than one source shares this asset: ' + ', '.join(obj.name for obj in matches)
                           + '. Select the independent copy and use Use Selected Model to give it its own asset.')
    return matches[0] if matches else None
