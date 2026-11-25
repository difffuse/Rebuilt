"""Driving Blender's baker, and putting the scene back afterwards."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence
import os
import sys

import bpy
import numpy as np

from . import image_io, materials


SENTINEL = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)


NON_COLOUR_PASSES = frozenset({"NORMAL", "POSITION", "ROUGHNESS", "AO", "UV"})


@dataclass(frozen=True)
class BakeRequest:
    """One invocation of the baker."""

    bake_type: str
    image: bpy.types.Image
    pass_filter: frozenset[str] = frozenset()

    self_bake: bool = False
    use_clear: bool = False

    samples: int | None = None

    ao_distance: float | None = None


@contextmanager
def ambient_occlusion_distance(scene, distance: float | None):
    """Borrow a private world for a scale-aware AO bake, restoring on failure."""
    if distance is None:
        yield
        return
    original = scene.world
    temporary = original.copy() if original else bpy.data.worlds.new("rebuilt_ao_world")
    try:
        temporary.light_settings.distance = max(float(distance), 1e-6)
        temporary.light_settings.ao_factor = 1.0
        scene.world = temporary
        yield
    finally:
        scene.world = original
        bpy.data.worlds.remove(temporary)


@contextmanager
def metal_archive_guard():
    """Avoid Blender 5.2 Metal archive crashes during the bake session."""
    if sys.platform != "darwin" or bpy.app.version[:2] != (5, 2):
        yield
        return
    key = "CYCLES_METAL_DISABLE_BINARY_ARCHIVES"
    previous = os.environ.get(key)
    os.environ[key] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


@contextmanager
def scene_state(scene: bpy.types.Scene) -> Iterator[None]:
    with metal_archive_guard(), _scene_state(scene):
        yield


@contextmanager
def _scene_state(scene: bpy.types.Scene) -> Iterator[None]:
    """Save everything a bake touches and restore it, whatever happens."""
    render = scene.render
    bake = render.bake
    view_layer = bpy.context.view_layer

    saved: dict[str, Any] = {
        "engine": render.engine,
        "threads_mode": render.threads_mode,
        "threads": render.threads,
        "active": view_layer.objects.active,
        "selection": [obj for obj in view_layer.objects if obj.select_get()],
        "hide_render": {obj.name: obj.hide_render for obj in scene.objects},
        "bake": {
            name: getattr(bake, name)
            for name in (
                "use_selected_to_active",
                "cage_extrusion",
                "max_ray_distance",
                "use_cage",
                "cage_object",
                "margin",
                "margin_type",
                "normal_space",
                "use_clear",
                "target",
            )
        },
    }
    cycles = getattr(scene, "cycles", None)
    if cycles is not None:
        saved["samples"] = cycles.samples
        saved["use_denoising"] = cycles.use_denoising
        saved["device"] = cycles.device
    saved["compute"] = _borrow_gpu()
    render.threads_mode = "AUTO"

    try:
        yield
    finally:
        _return_gpu(saved["compute"])
        for name, value in saved["bake"].items():
            try:
                setattr(bake, name, value)
            except (AttributeError, TypeError):
                pass
        render.engine = saved["engine"]
        render.threads = saved["threads"]
        render.threads_mode = saved["threads_mode"]
        if cycles is not None:
            cycles.samples = saved["samples"]
            cycles.use_denoising = saved["use_denoising"]
            cycles.device = saved["device"]

        for name, hidden in saved["hide_render"].items():
            obj = scene.objects.get(name)
            if obj is not None:
                obj.hide_render = hidden

        for obj in view_layer.objects:
            obj.select_set(False)
        for obj in saved["selection"]:
            try:
                obj.select_set(True)
            except ReferenceError:
                pass
        try:
            view_layer.objects.active = saved["active"]
        except ReferenceError:
            view_layer.objects.active = None


def _borrow_gpu():
    """Temporarily select supported GPU acceleration and snapshot device flags."""
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except (KeyError, AttributeError):
        return None
    previous = prefs.compute_device_type
    flags = {(d.id, d.type): d.use for d in prefs.devices}
    backends = ("METAL",) if sys.platform == "darwin" else ("OPTIX", "CUDA", "HIP", "ONEAPI")
    for backend in backends:
        try:
            prefs.compute_device_type = backend
            devices = prefs.get_devices_for_type(backend)
        except Exception:
            continue
        for device in prefs.devices:
            flags.setdefault((device.id, device.type), device.use)
        gpus = [device for device in devices if device.type == backend]
        if gpus:
            for device in devices:
                device.use = device.type == backend
            return prefs, previous, flags
    prefs.compute_device_type = "NONE"
    return prefs, previous, flags


def _return_gpu(borrowed) -> None:
    if borrowed is None:
        return
    prefs, previous, flags = borrowed
    try:
        prefs.compute_device_type = previous
        for device in prefs.devices:
            key = (device.id, device.type)
            if key in flags:
                device.use = flags[key]
    except (TypeError, ReferenceError):
        pass


def _gpu_enabled() -> bool:
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        backend = prefs.compute_device_type
        return backend != "NONE" and any(d.type == backend and d.use for d in prefs.devices)
    except (KeyError, AttributeError):
        return False


def new_image(name: str, size: int, non_colour: bool) -> bpy.types.Image:
    """A fresh float image; a requested name cannot authorize replacing data."""
    image = bpy.data.images.new(name, size, size, alpha=True, float_buffer=True)
    # Float buffers must contain linear values. An sRGB bake target stores
    # encoded values that float-image shaders and later pixel writes treat as
    # linear, brightening the material and its exported texture.
    image.colorspace_settings.name = "Non-Color" if non_colour else "Linear Rec.709"
    return image


def fill(image: bpy.types.Image, colour: np.ndarray) -> None:
    """Paint the whole image one colour."""
    width, height = image.size
    image.pixels.foreach_set(np.tile(colour.astype(np.float32), width * height))


def ensure_bake_material(
    obj: bpy.types.Object, image: bpy.types.Image, name: str, *, resources=None
) -> bpy.types.Material:
    """Give the low a material whose active node is the bake target."""
    material = (resources.create('bake', bpy.data.materials.new, name)
                if resources is not None else bpy.data.materials.new(name))
    material.use_nodes = True

    if obj.get("rebuilt_asset"):
        material["rebuilt_asset"] = obj["rebuilt_asset"]

    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    tree.nodes.active = texture
    texture.select = True

    operation = resources.owner.operation if resources is not None else None
    if operation is not None and operation.record.get('low') == obj:
        operation.require_private_mesh(obj)
    elif obj.data.users > 1 or obj.data.library is not None:
        obj.data = obj.data.copy()
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return material


def set_bake_target(obj: bpy.types.Object, image: bpy.types.Image) -> None:
    """Point the low's material at a different image, keeping everything else."""
    for slot in obj.data.materials:
        if slot is None or not slot.use_nodes:
            continue
        for node in slot.node_tree.nodes:
            if node.type == "TEX_IMAGE":
                node.image = image
                slot.node_tree.nodes.active = node
                node.select = True


def emission_material(name: str, colour: Sequence[float]) -> bpy.types.Material:
    """A material that emits a flat colour and nothing else."""
    material = bpy.data.materials.new(name)
    material.use_nodes = True

    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*colour, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def constant_scalar_channel(obj: bpy.types.Object, socket: str) -> float | None:
    """Recognise uniform scalar inputs on directly connected Principled surfaces."""
    if socket not in {"Metallic", "Specular IOR Level"}:
        return None
    if any(modifier.show_render for modifier in obj.modifiers):
        return None
    indices = np.empty(len(obj.data.polygons), dtype=np.int32)
    obj.data.polygons.foreach_get("material_index", indices)
    values = []
    for index in np.unique(indices):
        if index >= len(obj.material_slots):
            return None
        material = obj.material_slots[index].material
        if material is None or not material.use_nodes:
            return None
        tree = material.node_tree
        if tree.animation_data is not None:
            return None
        outputs = [n for n in tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output]
        if len(outputs) != 1:
            return None
        output = outputs[0]
        if output.inputs["Volume"].is_linked or output.inputs["Displacement"].is_linked:
            return None
        surface = output.inputs["Surface"]
        if not surface.is_linked:
            return None
        shader = surface.links[0].from_node
        if shader.type != "BSDF_PRINCIPLED":
            return None
        source = shader.inputs.get(socket)
        if source is None or source.is_linked or source.type != "VALUE":
            return None
        value = float(source.default_value)
        if not np.isfinite(value):
            return None
        values.append(value)
    return values[0] if values and all(value == values[0] for value in values) else None


def channel_material(source: bpy.types.Material, socket: str) -> bpy.types.Material:
    """A copy of `source` that emits one of its shader inputs as colour."""
    copy = source.copy()
    copy.name = f"rebuilt_channel_{socket.lower().replace(' ', '_')}"
    if not copy.use_nodes:
        copy.use_nodes = True

    tree = copy.node_tree
    color = materials.find_colour_source(copy)
    principled = color.node if color is not None and color.node.type == "BSDF_PRINCIPLED" else None
    output = next(
        (n for n in tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None
    )
    if output is None:
        output = tree.nodes.new("ShaderNodeOutputMaterial")

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0

    source_socket = (
        color if socket == "Base Color" else principled.inputs.get(socket) if principled else None
    )
    if socket in {"Emission Color", "Emission Strength"}:
        emitter = color.node if color is not None and color.node.type == "EMISSION" else None
        if copy.get("rebuilt_original_shader") == "EMISSION":
            emitter = next((n for n in tree.nodes if n.type == "EMISSION" and n != emission), None)
        if emitter is not None:
            source_socket = emitter.inputs["Color" if socket == "Emission Color" else "Strength"]

    if source_socket is None:
        from ..core.bake.channels import EXTRA_CHANNELS

        defaults = {entry[1]: entry[3] for entry in EXTRA_CHANNELS.values()}
        value = {"Metallic": 0.0, "Alpha": 1.0, **defaults}.get(socket, 0.5)
        emission.inputs["Color"].default_value = (
            value if isinstance(value, tuple) else (value, value, value, 1.0)
        )
    elif source_socket.is_linked:
        tree.links.new(source_socket.links[0].from_socket, emission.inputs["Color"])
    else:
        value = source_socket.default_value
        if hasattr(value, "__len__"):
            emission.inputs["Color"].default_value = (*tuple(value)[:3], 1.0)
        else:
            emission.inputs["Color"].default_value = (value, value, value, 1.0)

    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return copy


@contextmanager
def channel_substitution(obj: bpy.types.Object, socket: str) -> Iterator[None]:
    """Swap every material on `obj` for one emitting the named channel."""
    originals = [slot.material for slot in obj.material_slots]
    replacements: list[bpy.types.Material] = []

    try:
        if not originals:
            replacement = emission_material("rebuilt_channel_missing", (0.5, 0.5, 0.5))
            replacements.append(replacement)
            obj.data.materials.append(replacement)
        else:
            for slot in obj.material_slots:
                if slot.material is None:
                    continue
                replacement = channel_material(slot.material, socket)
                replacements.append(replacement)
                slot.material = replacement
        yield
    finally:
        if not originals:
            obj.data.materials.clear()
        else:
            for slot, previous in zip(obj.material_slots, originals):
                slot.material = previous
        for material in replacements:
            if material.users == 0:
                bpy.data.materials.remove(material)


@contextmanager
def substituted_materials(obj: bpy.types.Object, material: bpy.types.Material) -> Iterator[None]:
    """Temporarily give an object one material, then give its own back."""
    original = [slot.material for slot in obj.material_slots]
    try:
        if not original:
            obj.data.materials.append(material)
        else:
            for slot in obj.material_slots:
                slot.material = material
        yield
    finally:
        if not original:
            obj.data.materials.clear()
        else:
            for slot, previous in zip(obj.material_slots, original):
                slot.material = previous


def prepare(
    scene: bpy.types.Scene,
    low: bpy.types.Object,
    high: bpy.types.Object,
    cage: bpy.types.Object | None,
    samples: int,
    margin_px: int,
    normal_space: str,
    max_ray_distance: float = 0.0,
) -> None:
    """Put the scene into a state the baker will accept."""
    scene.render.engine = "CYCLES"
    if hasattr(scene, "cycles"):
        scene.cycles.samples = max(samples, 1)

        scene.cycles.device = "GPU" if _gpu_enabled() else "CPU"

        scene.cycles.use_denoising = False

    bake = scene.render.bake
    bake.use_selected_to_active = True

    bake.margin = 0
    bake.margin_type = "ADJACENT_FACES"
    bake.normal_space = normal_space
    bake.target = "IMAGE_TEXTURES"
    bake.use_cage = cage is not None
    bake.cage_object = cage

    bake.cage_extrusion = 0.0

    bake.max_ray_distance = max(max_ray_distance, 0.0)

    isolate(scene, keep=(high, low))
    _select(scene, [high, low], active=low)


def isolate(scene: bpy.types.Scene, keep: Sequence[bpy.types.Object]) -> None:
    """Hide every other mesh from the baker's rays."""
    kept = {obj.name for obj in keep if obj is not None}
    for obj in scene.objects:
        if obj.type == "MESH":
            obj.hide_render = obj.name not in kept


def run(
    scene: bpy.types.Scene,
    low: bpy.types.Object,
    high: bpy.types.Object,
    request: BakeRequest,
) -> np.ndarray:
    """Bake one pass and return its pixels."""
    set_bake_target(low, request.image)
    if not request.use_clear:
        fill(request.image, SENTINEL)

    bake = scene.render.bake
    previous_projection = bake.use_selected_to_active
    bake.use_selected_to_active = not request.self_bake

    cycles = getattr(scene, "cycles", None)
    previous_samples = cycles.samples if cycles is not None else None
    if request.samples is not None and cycles is not None:
        cycles.samples = max(request.samples, 1)

    if request.self_bake:
        _select(scene, [low], active=low)
    else:
        _select(scene, [high, low], active=low)

    try:
        kwargs: dict[str, Any] = {
            "type": request.bake_type,
            "use_clear": request.use_clear,
        }
        if request.pass_filter:
            kwargs["pass_filter"] = set(request.pass_filter)
        with ambient_occlusion_distance(
            scene, request.ao_distance if request.bake_type == "AO" else None
        ):
            bpy.ops.object.bake(**kwargs)
    finally:
        bake.use_selected_to_active = previous_projection
        if previous_samples is not None:
            cycles.samples = previous_samples

    return image_io.read_pixels(request.image)


def _select(
    scene: bpy.types.Scene, objects: Sequence[bpy.types.Object], active: bpy.types.Object
) -> None:
    """Set both the selection and the active object; the baker reads both."""
    view_layer = bpy.context.view_layer
    for obj in view_layer.objects:
        obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    view_layer.objects.active = active
