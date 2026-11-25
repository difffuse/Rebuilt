"""Handing the finished asset over."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import bpy

from ..bridge import image_io, output_delivery, output_materials
from ..bridge.image_refs import image_from_entry, map_key
from ..bridge.output_paths import require_exact_destination
from ..core import engines
from ..core.file_names import asset_stem
from ..core.jobs import ProgressReporter
from ..core.recipe import AssetRecipe
from ..pipeline.context import AssetContext, ContextError
from ..pipeline.export_context import preserve_export_state
from .base import Stage
from .retopo import ROLE_LOW


class OutputStage(Stage):
    name: ClassVar[str] = "output"
    slot: ClassVar[str] = "low"
    role: ClassVar[str] = "output"
    label: ClassVar[str] = "Finishing"
    retain_working: bool = False

    def cache_key(self, recipe: AssetRecipe) -> str:

        return (
            f"{recipe.profile}:{recipe.bake.fingerprint()}:material-{output_materials.MATERIAL_REVISION}:map-names-4:map-references-1"
        )

    def gather(self, ctx: AssetContext) -> dict[str, Any]:
        if ctx.artifact(ROLE_LOW) is None:
            raise ContextError("output needs retopo to have run first")
        return {"bake": ctx.report("bake"), "name": ctx.name}

    def compute(self, payload: dict[str, Any], report: ProgressReporter) -> dict[str, Any]:
        report.update(1.0, "Ready")
        return payload

    def apply(self, ctx: AssetContext, result: dict[str, Any]) -> None:
        from ..pipeline.stage_operation import stage_operation
        with stage_operation(ctx, self.name):
            self._apply(ctx, result)

    def _apply(self, ctx: AssetContext, result: dict[str, Any]) -> None:
        ctx.require_owned()
        low = ctx.artifact(ROLE_LOW)
        bake = result.get('bake') or {}
        maps = ctx.resolve_maps(bake.get('maps', []), required=True)
        output_delivery.required_map_images(maps)
        output_materials.local_output_mesh(low, ctx=ctx)
        low.name = f"{ctx.name}_LOD0"
        low.data.name = low.name

        _match_origin(low, ctx.source_object)

        material = output_materials.assemble_material(low, maps, ctx=ctx)
        folder = _textures_folder(ctx.name)
        saved = _map_file_state(maps, ctx.name, folder)
        if saved and saved == (bake.get("preview") or {}).get("file_state"):
            written = list(saved)
        else:
            written = output_delivery.write_maps(
                maps, ctx.name, engines.get("generic"), folder, relink=False,
                cancel_files=ctx.cancel_files
            )

        removed = ctx.deliver(preserve_working=self.retain_working)

        ctx.set_report(
            self.name,
            {
                "object": low.name,
                "material": material.name if material else "",
                "material_revision": output_materials.MATERIAL_REVISION,
                "maps_on_disk": written,
                "removed": removed,
                "working_data_retained": self.retain_working,
                "notes": (
                    []
                    if written or not maps
                    else ["Save the .blend to have the maps written beside it."]
                ),
            },
        )

        preserve_export_state(ctx)


def show_baked_maps(ctx: AssetContext, maps: list[dict[str, Any]], *, clear_preview=True) -> dict:
    """Show completed maps immediately, keeping working data for Manual edits."""
    from ..pipeline.stage_operation import stage_operation
    from ..bridge.image_refs import describe
    with stage_operation(ctx, 'preview') as operation:
        prepared = ctx.resolve_maps(maps, required=True)
        output_delivery.required_map_images(prepared)
        fresh = {row.get('id') for key, row in ctx.resources.snapshot().items()
                 if key not in operation.record['resources'] and row['kind'] == 'image'}
        for entry in prepared:
            image = image_from_entry(entry)
            if image not in fresh:
                image = ctx.resources.create('bake', image_io.copy_image_pixels, image)
                entry['image_ref'] = ctx.resources.key(image)
                describe(entry, image)
        preview = _show_baked_maps(ctx, prepared, clear_preview=clear_preview)
    maps[:] = prepared
    return preview


def _show_baked_maps(ctx, maps, *, clear_preview):
    """Update the caller's transaction-owned descriptors when files are relinked."""
    low = ctx.artifact(ROLE_LOW)
    if low is None:
        raise ContextError("bake preview needs a low mesh")
    output_delivery.required_map_images(maps)
    material = output_materials.assemble_material(low, maps, ctx=ctx, clear_preview=clear_preview)
    folder = _textures_folder(ctx.name)

    written = output_delivery.write_maps(
        maps, ctx.name, engines.get("generic"), folder, relink=True, cancel_files=ctx.cancel_files
    )
    for entry in maps:
        image = image_from_entry(entry) if entry.get("baked") else None
        if image is not None:
            image.pack()
    cage = ctx.artifact("cage")
    if cage is not None:
        if bpy.context.view_layer.objects.active == cage:
            cage.select_set(False)
            low.hide_set(False)
            low.select_set(True)
            bpy.context.view_layer.objects.active = low
        cage.hide_set(True)
    return {
        "material": material.name if material else "",
        "material_revision": output_materials.MATERIAL_REVISION,
        "maps_on_disk": written,
        "file_state": _map_file_state(maps, ctx.name, folder),
    }


def _map_file_state(maps: list[dict[str, Any]], asset: str, folder: Path | None) -> dict:
    """Recognize this session's unchanged preview files; never reuse dirty maps."""
    if folder is None:
        return {}
    state = {}
    for entry in maps:
        if not entry.get("baked"):
            continue
        image = image_from_entry(entry)
        if image is None or image.is_dirty or image.source != "FILE":
            return {}
        path = folder / output_delivery.map_filename(engines.get("generic"), asset, entry, image)
        if Path(bpy.path.abspath(image.filepath)).resolve() != path.resolve():
            return {}
        try:
            stat = path.stat()
        except OSError:
            return {}
        state[str(path)] = [
            image.as_pointer(),
            stat.st_mtime_ns,
            stat.st_size,
            image.colorspace_settings.name,
        ]
    return state


def _textures_folder(asset: str) -> Path | None:
    """Locate textures beside the saved blend file, or return None."""
    if not bpy.data.filepath:
        return None
    stem = asset_stem(asset)
    # Keep existing ordinary assets' folder locations. Escaped names must not
    # pass through clean_name, which would collapse distinct escape sequences.
    folder = (bpy.path.clean_name(asset).strip(" .") or "Asset") if stem == asset else stem
    destination = Path(bpy.data.filepath).parent / "textures" / folder
    require_exact_destination(destination)
    return destination


def export_files(
    ctx: AssetContext,
    target_engine: str,
    export_format: str,
    unity_pipeline: str = "urp",
    destination_folder: str | Path | None = None,
) -> dict[str, Any]:
    """Export an existing asset and engine-specific maps without rebuilding it."""
    if unity_pipeline not in {"urp", "builtin", "hdrp"}:
        raise ContextError(f"Unknown Unity render pipeline: {unity_pipeline}")
    low = ctx.artifact(ROLE_LOW)
    if low is None:
        raise ContextError("nothing to export: the asset has not been built yet")
    if destination_folder is None and not bpy.data.filepath:
        raise ContextError("Choose an export folder or save the .blend before exporting.")

    try:
        engine_target = engines.get(target_engine)
    except engines.EngineError:
        engine_target = engines.get("generic")

    if engine_target.identifier == "unity":
        export_format = "fbx"
    elif engine_target.identifier in {"unreal", "godot"}:
        export_format = "gltf"

    destination_folder = (
        Path(destination_folder).expanduser().resolve()
        if destination_folder is not None
        else Path(bpy.data.filepath).parent / ctx.recipe.output.export_folder
    )
    try:
        destination_folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ContextError(f"Cannot create the export folder: {exc}") from exc
    from ..pipeline.map_images import for_export
    maps = for_export(low, ctx.resolve_maps((ctx.report("bake") or {}).get("maps", [])))
    asset_folder = ""
    material = guide = ""
    if engine_target.identifier == "unreal":
        exported, notes, asset_folder, guide = _export_unreal_folder(
            low, ctx.name, maps, destination_folder
        )
        written = []
    elif engine_target.identifier == "godot":
        exported, notes, asset_folder, guide = _export_godot_folder(
            low, ctx.name, maps, destination_folder
        )
        written = []
    elif engine_target.identifier == "unity":
        exported, notes, asset_folder, material, written, guide = _export_unity_folder(
            low, ctx.name, maps, destination_folder, unity_pipeline
        )
    else:
        exported, notes, written = _export_generic_files(
            low, ctx.name, maps, engine_target, export_format, destination_folder
        )

    unreal_package = ""
    unreal_importer = ""
    import dataclasses

    ctx.recipe = dataclasses.replace(
        ctx.recipe,
        output=dataclasses.replace(
            ctx.recipe.output,
            target_engine=engine_target.identifier,
            export_format=export_format if exported else ctx.recipe.output.export_format,
            unity_pipeline=unity_pipeline,
        ),
    )
    report = {
        "destination_folder": str(destination_folder),
        "target_engine": engine_target.identifier,
        "format": export_format,
        "asset_folder": asset_folder,
        "exported": exported,
        "maps_on_disk": written,
        "import_guide": guide,
        "unreal_package": unreal_package,
        "unreal_importer": unreal_importer,
        "unity_material": material,
        "unity_package": "",
        "unity_pipeline": unity_pipeline if engine_target.identifier == "unity" else "",
        "notes": list(
            engine_target.notes if exported and engine_target.identifier not in {"unreal", "godot"} else ()
        )
        + notes,
    }
    ctx.set_report("export", report)
    preserve_export_state(ctx)
    return report


def _export_generic_files(low, asset, maps, engine_target, export_format, destination):
    """Generate the complete export, then publish its flat files in one batch."""
    import shutil
    from tempfile import TemporaryDirectory
    from ..pipeline.cancel_files import CancelFiles, FileCommitUncertain

    with TemporaryDirectory(prefix="rebuilt-export-") as temporary:
        folder = Path(temporary)
        exported, notes, written, _expected, _metadata = _export_with_maps(
            low, asset, maps, engine_target, export_format, folder)
        if export_format != "none" and not exported:
            raise ContextError("; ".join(notes) or "Model export failed.")
        prepared = [Path(path) for path in written]
        if exported:
            prepared.append(Path(exported))
        if any(not path.is_file() or path.parent != folder for path in prepared):
            raise ContextError("Export failed: a required output file was not generated.")
        for source in prepared:
            require_exact_destination(destination / source.name)

        if prepared:
            journal = CancelFiles(
                bpy.utils.user_resource('CONFIG', path='rebuilt/recovery', create=True),
                origin={'version': 1, 'kind': 'export', 'blend': bpy.data.filepath,
                        'asset': asset})
            journal.hold()
            try:
                try:
                    for source in prepared:
                        journal.write(destination / source.name,
                                      lambda staging, source=source: shutil.copy2(source, staging))
                    journal.commit()
                except FileCommitUncertain:
                    raise
                except BaseException:
                    if journal.phase != 'committed':
                        journal.rollback()
                    raise
                journal.cleanup()
            finally:
                journal.release()

    return (str(destination / Path(exported).name) if exported else "", notes,
            [str(destination / Path(path).name) for path in written])


def _export_with_maps(low, asset, maps, engine_target, export_format, folder):
    written = output_delivery.write_maps(maps, asset, engine_target, folder, relink=False)
    expected = {
        map_key(entry): folder
        / output_delivery.map_filename(engine_target, asset, entry, image_from_entry(entry))
        for entry in maps
        if entry.get("baked") and image_from_entry(entry) is not None
    }
    if any(str(path) not in written for path in expected.values()):
        raise ContextError("Export failed: one or more textures could not be written.")
    metadata = {}
    exported, notes = _export_asset(
        low,
        asset,
        export_format,
        engine_target,
        folder,
        image_paths=expected,
        metadata=metadata,
        embed_textures=engine_target.identifier not in {"unity", "unreal"},
    )
    return exported, notes, written, expected, metadata


def _export_unity_folder(low, asset, maps, destination, pipeline):
    """Publish FBX, native material and referenced textures with stable metadata."""
    from tempfile import TemporaryDirectory
    from ..bridge.export_folder import publish
    from ..bridge.unity_export import write_import_files
    from ..bridge.unity_material import restore_metadata, write_material, referenced_files

    stem = asset_stem(asset)
    target = destination / stem
    require_exact_destination(target)
    try:
        with TemporaryDirectory(prefix=".rebuilt-unity-", dir=destination) as temporary:
            folder = Path(temporary)
            if not (target / f"{stem}.fbx.meta").exists():
                restore_metadata(destination / f"{stem}.unitypackage", folder, asset)
            if target.is_dir():
                for path in target.glob("*.meta"):
                    (folder / path.name).write_bytes(path.read_bytes())
            exported, notes, written, expected, metadata = _export_with_maps(
                low, asset, maps, engines.get("unity"), "fbx", folder
            )
            if not exported:
                raise ContextError("; ".join(notes) or "Unity FBX export failed.")
            packed, guide = write_import_files(
                low, maps, expected, asset, folder, pipeline=pipeline
            )
            material, generated = write_material(
                low, maps, expected, asset, folder, exported,
                metadata.get("material_names", []),
                packed_path=packed[0] if packed else None, pipeline=pipeline,
            )
            files = referenced_files([exported, material, *written, *packed, *generated])
            publish(destination, stem, "unity", [
                *files, *(Path(str(p) + ".meta") for p in files), Path(guide),
            ])
            textures = [str(target / p.name) for p in files if p.suffix not in {".fbx", ".mat"}]
    except (RuntimeError, OSError, ValueError) as exc:
        raise ContextError(f"Unity folder export failed: {exc}") from exc
    notes.append("Copy the entire asset folder into Unity Assets, including its .meta files.")
    return (str(target / Path(exported).name), notes, str(target),
            str(target / Path(material).name), textures, str(target / Path(guide).name))


def _export_unreal_folder(low, asset, maps, destination):
    return _export_glb_folder(low, asset, maps, destination, "unreal")


def _export_godot_folder(low, asset, maps, destination):
    return _export_glb_folder(low, asset, maps, destination, "godot")


def _export_glb_folder(low, asset, maps, destination, target_engine):
    from tempfile import TemporaryDirectory
    from ..bridge.export_folder import publish
    from ..bridge.gltf_export import export_glb

    stem = asset_stem(asset)
    engine = engines.get(target_engine)
    require_exact_destination(destination / stem)
    try:
        with TemporaryDirectory(prefix=f".rebuilt-{target_engine}-", dir=destination) as temporary:
            exported, notes = export_glb(
                low, asset, maps, Path(temporary), target_engine=target_engine
            )
            guide = Path(temporary) / f"{stem}_{engine.label}_Import.md"
            template = Path(__file__).resolve().parents[1] / "resources" / f"{target_engine}-import.md"
            guide.write_text(template.read_text(encoding="utf-8").format(model=Path(exported).name), encoding="utf-8")
            target = publish(destination, stem, target_engine, [Path(exported), guide])
    except (RuntimeError, OSError, ValueError) as exc:
        raise ContextError(f"{engine.label} folder export failed: {exc}") from exc
    return str(target / Path(exported).name), notes, str(target), str(target / guide.name)


def _match_origin(low: bpy.types.Object, source: bpy.types.Object) -> None:
    """Put the asset's origin where the scan's was."""
    import bpy
    from mathutils import Matrix

    bpy.context.view_layer.update()
    origin = source.matrix_world.translation.copy()
    target = Matrix.Translation(origin)
    transform = target.inverted() @ low.matrix_world
    if transform != Matrix.Identity(4):
        low.data.transform(transform)
    low.matrix_world = target
    low.data.update()


def _export_asset(
    low: bpy.types.Object,
    asset: str,
    requested_format: str,
    engine_target: engines.EngineTarget,
    destination_folder: Path,
    image_paths=None,
    metadata=None,
    embed_textures=True,
) -> tuple[str, list[str]]:
    """Export the asset and return its path and any warnings."""
    if requested_format == "none":
        return "", []
    if engine_target.identifier in {"unity", "unreal"}:
        requested_format = "fbx"
    destination_folder.mkdir(parents=True, exist_ok=True)
    destination_path = (
        destination_folder / f"{engines.mesh_name(engine_target, asset)}.{requested_format}"
    )

    previous_selection = [o for o in bpy.context.selected_objects]
    previous_active = bpy.context.view_layer.objects.active
    for export_object in previous_selection:
        export_object.select_set(False)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low

    export_warnings: list[str] = []
    try:
        if requested_format == "fbx":
            from ..bridge.fbx_export import export_copy

            with export_copy(
                low,
                image_paths,
                apply_transform=engine_target.identifier in {"unity", "unreal"},
                ensure_material=engine_target.identifier in {"unity", "unreal"},
                triangulate=engine_target.identifier == "unity",
            ) as copied:
                if engine_target.identifier == "unreal":
                    copied.matrix_world.translation = (0.0, 0.0, 0.0)
                low.select_set(False)
                copied.select_set(True)
                bpy.context.view_layer.objects.active = copied
                bpy.context.view_layer.update()
                if metadata is not None:
                    metadata["material_names"] = [
                        slot.material.name
                        for slot in copied.material_slots
                        if slot.material is not None
                    ]
                bpy.ops.export_scene.fbx(
                    filepath=str(destination_path),
                    use_selection=True,
                    object_types={"MESH"},
                    axis_forward=engine_target.axis_forward,
                    axis_up=engine_target.axis_up,
                    apply_scale_options="FBX_SCALE_ALL",
                    bake_space_transform=engine_target.identifier in {"unity", "unreal"},
                    bake_anim=False,
                    mesh_smooth_type="FACE",
                    use_tspace=True,
                    path_mode="COPY" if embed_textures else "RELATIVE",
                    embed_textures=embed_textures,
                )
        elif requested_format == "gltf":
            bpy.ops.export_scene.gltf(
                filepath=str(destination_path.with_suffix(".glb")),
                export_format="GLB",
                use_selection=True,
            )
            destination_path = destination_path.with_suffix(".glb")
        else:
            return "", [f"Unknown export format {requested_format!r}; nothing was written."]
    except (RuntimeError, OSError) as exc:
        return "", [f"Export failed: {exc}"]
    finally:
        low.select_set(False)
        for export_object in previous_selection:
            try:
                export_object.select_set(True)
            except ReferenceError:
                pass
        if previous_active is not None:
            try:
                bpy.context.view_layer.objects.active = previous_active
            except ReferenceError:
                pass

    return str(destination_path), export_warnings
