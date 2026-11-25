"""Scene settings for Auto, Manual and optional export."""

from __future__ import annotations

import math

import bpy

from ..core import profiles
from ..core.bake.channels import EXTRA_CHANNELS


def _profile_items(self, context) -> list[tuple[str, str, str]]:
    """Profiles, read from disk rather than hard-coded."""
    global _PROFILE_ITEMS
    _PROFILE_ITEMS = [
        (identifier, profile.label, profile.description)
        for identifier, profile in sorted(profiles.available().items())
        if profile.listed
    ]
    return _PROFILE_ITEMS


_PROFILE_ITEMS: list[tuple[str, str, str]] = []

_TEXTURE_ITEMS = (
    ("1024", "1K", ""),
    ("2048", "2K", ""),
    ("4096", "4K", ""),
    ("8192", "8K", ""),
)


_PROPOSING = False
_PROPOSING_MAPS = False

_MAP_PROPERTIES = {
    "NORMAL": "bake_normal",
    "BASE_COLOR": "bake_base_color",
    "AO": "bake_ao",
    "ROUGHNESS": "bake_roughness",
    "METALLIC": "bake_metallic",
    "OPACITY": "bake_opacity",
}
_MAP_PROPERTIES.update({name: "bake_" + name.lower() for name in EXTRA_CHANNELS})


_EXPORT_FORMATS = [
    ("fbx", "FBX", "FBX mesh with assigned materials and textures in a Unity folder", 0),
    ("gltf", "glTF (.glb)", "Open format, textures embedded in one file", 1),
]
_UNITY_EXPORT_FORMATS = [
    ("fbx", "Unity Folder", "A model folder with FBX, materials, textures and Unity metadata", 0),
]
_GLB_EXPORT_FORMATS = _EXPORT_FORMATS[1:]


def _export_format_items(self, context):
    if self.target_engine == "unity":
        return _UNITY_EXPORT_FORMATS
    if self.target_engine in {"unreal", "godot"}:
        return _GLB_EXPORT_FORMATS
    return _EXPORT_FORMATS


def _on_target_engine_changed(self, context):
    if self.target_engine == "unity":
        self.export_format = "fbx"
    elif self.target_engine in {"unreal", "godot"}:
        self.export_format = "gltf"


class REBUILT_Settings(bpy.types.PropertyGroup):
    """What the next run will be built from."""

    profile: bpy.props.EnumProperty(
        options=set(),
        name="Asset Type",
        description=(
            "The model category. Sets the budget, the texel density and the "
            "tolerances. The analysis proposes one; whatever you choose wins"
        ),
        items=_profile_items,
        update=lambda self, context: self._on_profile_changed(),
    )

    profile_chosen: bpy.props.BoolProperty(default=False, options={"HIDDEN"})

    detail: bpy.props.EnumProperty(
        name="Detail",
        description="How much geometry and texture the asset gets",
        items=[
            ("draft", "Draft", "Quarter budget"),
            ("balanced", "Balanced", "The profile's intended budget"),
            ("high", "High", "Double budget"),
            ("max", "Max", "Four times budget"),
        ],
        default="balanced",
        options={"HIDDEN"},
        update=lambda self, context: self._sync_budget_from_preset(),
    )

    def _on_profile_changed(self) -> None:
        """Record an explicit profile choice unless detection is proposing it."""
        if not _PROPOSING:
            self.profile_chosen = True
        self._sync_budget_from_preset()

    def _sync_budget_from_preset(self) -> None:
        """Choosing a preset visibly moves the quad count and the texture."""
        from ..core.profiles import DETAIL_SCALES, get as get_profile

        try:
            chosen = get_profile(self.profile)
        except Exception:
            return
        scale = DETAIL_SCALES.get(self.detail, 1.0)
        self.target_quads = max(int(chosen.target_faces * scale), 8)
        texture_size_px = str(int(chosen.texture_resolution))
        if texture_size_px in {item[0] for item in _TEXTURE_ITEMS}:
            self.texture_resolution = texture_size_px

        self.uv_unwrap_method = chosen.unwrap_method
        self.uv_padding_px = chosen.padding_px
        self.uv_rotate_method = chosen.rotate_method
        if not self.bake_maps_chosen:
            maps = (
                set(self.bake_detected_maps.split(","))
                if self.bake_detected_maps
                else {m.upper() for m in chosen.maps}
            )
            self._set_bake_maps(maps)

    texture_resolution: bpy.props.EnumProperty(
        options=set(),
        name="Texture",
        description="Resolution of the baked maps",
        items=_TEXTURE_ITEMS,
        default="4096",
    )
    topology: bpy.props.EnumProperty(
        options=set(),
        name="Topology",
        description="What the finished mesh is made of",
        items=[
            (
                "quads",
                "Quads",
                "Quads are polygons with four edges and four vertices. "
                "Rebuild the model with mostly quad faces and edge flow that follows its shape",
            ),
            (
                "triangles",
                "Triangles",
                "For a static prop or a model better suited to triangle reduction",
            ),
        ],
        default="quads",
    )
    remesher: bpy.props.EnumProperty(
        options=set(),
        name="Remesher",
        description="Which engine lays the quads down",
        items=[
            (
                "auto",
                "AutoRemesher",
                "The bundled AutoRemesher engine (MIT). Sizes its quads by curvature "
                "inside the solver, so Adaptive Size costs no extra refinement pass. "
                "The default; retains usable native results and reports mesh warnings",
                0,
            ),
            (
                "quadriflow",
                "QuadriFlow (Blender)",
                "Blender's built-in quad remesher. Also used as the AutoRemesher fallback",
                2,
            ),
        ],
        default="auto",
    )
    adaptive_size: bpy.props.FloatProperty(
        options=set(),
        name="Adaptive Size",
        description=(
            "How much the quad size follows the shape: at 0% every quad is the "
            "same size, at 100% curved detail gets small quads and flat areas "
            "get large ones, without changing the total count. AutoRemesher "
            "sizes by curvature inside its own solver; QuadriFlow takes a single global size"
        ),
        default=100.0,
        min=0.0,
        max=100.0,
        subtype="PERCENTAGE",
    )
    adaptive_shape: bpy.props.FloatProperty(
        options=set(),
        name="Adaptive Shape",
        description=(
            "How far a quad may stretch along the flatter direction: at 0% "
            "quads stay square, at 100% they elongate along ridges and tubes "
            "the way the form runs. Only AutoRemesher honours this — it is a "
            "decision its solver takes while it sizes, not something that can "
            "be refined on afterwards"
        ),
        default=100.0,
        min=0.0,
        max=100.0,
        subtype="PERCENTAGE",
    )
    target_quads: bpy.props.IntProperty(
        options=set(),
        name="Quads",
        description=("Target quad count in Strict mode; starting budget in Dynamic mode"),
        default=25_000,
        min=8,
        soft_max=200_000,
        step=500,
    )
    quad_budget_mode: bpy.props.EnumProperty(
        options=set(),
        name="Quad Count",
        items=(
            (
                "strict",
                "Strict",
                "Aim within 1% using bounded count corrections; keep the closest usable mesh and report any mismatch",
            ),
            (
                "dynamic",
                "Dynamic",
                "Aim for the starting budget or higher, up to four times it; limit extra refinement and report any mismatch",
            ),
        ),
        default="strict",
    )
    cloth_folds: bpy.props.BoolProperty(
        options=set(),
        name="Cloth Folds",
        description=(
            "Guide quad rows and spacing along curved folds on open cloth surfaces. "
            "Disable for open mechanical shells or decorative moldings"
        ),
        default=True,
    )
    keep_open_edges: bpy.props.BoolProperty(
        options=set(),
        name="Keep Open Edges",
        description=(
            "Leave the model's openings open instead of sealing them. A model cut "
            "off at ground level has a hole underneath; sealing it weaves a lid "
            "that can push vertices up through the surface. Turn this on when "
            "the opening is part of the object and you want it left alone"
        ),
        default=False,
    )

    uv_keep_marked_seams: bpy.props.BoolProperty(
        options=set(),
        name="Keep Marked Seams",
        description="Keep every currently marked seam on the retopology mesh and generate the remaining cuts. Includes previously generated seams; clear unwanted marks before recalculating",
        default=False,
    )
    uv_island_preference: bpy.props.EnumProperty(
        options=set(),
        name="UV Islands",
        description="How strongly generated seams favor fewer UV islands",
        items=[
            ("BALANCED", "Balanced", "Use the standard seam layout and small-fragment cleanup"),
            ("FEWER", "Fewer", "Try joining larger adjacent islands without adding overlap or stretch. May take longer; some geometry still needs separate islands"),
        ],
        default="BALANCED",
    )
    uv_unwrap_method: bpy.props.EnumProperty(
        options=set(),
        name="Method",
        description="How the surface is flattened",
        items=[
            (
                "ANGLE_BASED",
                "Angle Based",
                "Angle-preserving unwrapping, suited to organic models",
            ),
            (
                "CONFORMAL",
                "Conformal",
                "Also angle-preserving, and faster. Struggles where the surface "
                "folds back on itself",
            ),
            (
                "MINIMUM_STRETCH",
                "Minimum Stretch",
                "Spreads the stretch instead of preserving angles: more even texel "
                "coverage, at the cost of islands that read less like the shape",
            ),
        ],
        default="ANGLE_BASED",
    )
    uv_seam_angle: bpy.props.FloatProperty(
        options=set(),
        name="Seam Angle",
        description=(
            "Cut a seam where the surface bends more than this. Lower cuts more "
            "and flattens easier; higher keeps islands whole and readable"
        ),
        default=math.radians(40.0),
        min=math.radians(5.0),
        max=math.radians(180.0),
        subtype="ANGLE",
    )
    uv_padding_px: bpy.props.IntProperty(
        options=set(),
        name="Padding",
        description=(
            "Pixels of empty space around each island, so filtering and mipmaps "
            "do not bleed one island into its neighbour"
        ),
        default=8,
        min=0,
        soft_max=64,
        subtype="PIXEL",
    )
    uv_packer: bpy.props.EnumProperty(
        options=set(),
        name="Packer",
        description="Which engine lays the islands out on the texture",
        items=[
            (
                "uvpack",
                "UVPack",
                "The bundled packer: heuristic search over island outlines, "
                "validated overlap-free, padding honoured exactly. Fills the tile "
                "better than Blender's on every layout measured",
            ),
            (
                "blender",
                "Blender",
                "Blender's own packer. The baseline, and what takes over if the "
                "bundled one is missing or declines a layout",
            ),
        ],
        default="uvpack",
    )
    uv_pack_seconds: bpy.props.FloatProperty(
        options=set(),
        name="Pack Search",
        description=(
            "Seconds UVPack may spend improving the final layout: up to 2 "
            "refining the scale, anything past that compacting with free "
            "rotation. 0 is the plain search alone"
        ),
        default=2.0,
        min=0.0,
        soft_max=30.0,
        step=100,
        subtype="TIME_ABSOLUTE",
        unit="TIME_ABSOLUTE",
    )
    uv_shape_method: bpy.props.EnumProperty(
        options=set(),
        name="Packing",
        description="How closely the packer is allowed to follow each island's outline",
        items=[
            (
                "CONCAVE",
                "Exact Shape",
                "Follows the real outline. The tightest fit and the slowest",
            ),
            ("CONVEX", "Convex Hull", "Follows a taut wrap around each island. A good middle"),
            ("AABB", "Bounding Box", "Boxes only. Fastest, and wastes the most space"),
        ],
        default="CONCAVE",
    )
    uv_rotate_method: bpy.props.EnumProperty(
        options=set(),
        name="Rotation",
        description="How islands may be turned while packing",
        items=[
            ("ANY", "Any Angle", "Best use of the tile"),
            (
                "CARDINAL",
                "90° Steps",
                "Islands turn only in quarter turns: nearly all the packing gain, "
                "and a layout that still reads",
            ),
            (
                "AXIS_ALIGNED",
                "Axis Aligned",
                "Islands stay square to the tile — easier to read and to touch up by hand",
            ),
        ],
        default="ANY",
    )

    show_asset_settings: bpy.props.BoolProperty(options=set(), name="Model Settings", default=True)

    show_material_maps: bpy.props.BoolProperty(options=set(), name="Material Channels", default=False)
    show_texture_output: bpy.props.BoolProperty(options=set(), name="Texture Output", default=False)
    bake_image_format: bpy.props.EnumProperty(
        options=set(),
        name="Format",
        items=[
            ("PNG", "PNG", "Lossless textures"),
            ("JPEG", "JPEG (.jpg)", "Smaller color textures; data maps remain lossless PNG"),
            ("TIFF", "TIFF", "Lossless editing interchange"),
            ("OPEN_EXR", "OpenEXR", "Linear HDR textures"),
        ],
        default="PNG",
    )
    bake_image_depth: bpy.props.EnumProperty(
        options=set(),
        name="Bit Depth",
        items=[
            ("8", "8-bit", "Smaller textures"),
            ("16", "16-bit", "Smoother gradients and normals"),
        ],
        default="16",
    )
    bake_png_compression: bpy.props.IntProperty(
        options=set(),
        name="Compression",
        default=15,
        min=0,
        max=100,
        description="Lossless PNG compression; higher values can save space but take longer",
    )
    bake_tiff_codec: bpy.props.EnumProperty(
        options=set(),
        name="Compression",
        items=[
            ("NONE", "None", ""),
            ("DEFLATE", "Deflate", "Lossless"),
            ("LZW", "LZW", "Lossless"),
        ],
        default="DEFLATE",
    )
    bake_jpeg_quality: bpy.props.IntProperty(
        options=set(),
        name="Quality",
        default=85,
        min=1,
        max=100,
        subtype="PERCENTAGE",
        description="JPEG color quality; lower values make smaller files with more visible artifacts",
    )
    bake_exr_depth: bpy.props.EnumProperty(
        options=set(),
        name="Bit Depth",
        items=[("16", "16-bit Float", "Half precision"), ("32", "32-bit Float", "Full precision")],
        default="16",
    )
    bake_exr_codec: bpy.props.EnumProperty(
        options=set(),
        name="Compression",
        items=[("ZIP", "ZIP", "Lossless"), ("PIZ", "PIZ", "Lossless"), ("NONE", "None", "")],
        default="ZIP",
    )
    bake_maps_chosen: bpy.props.BoolProperty(default=False, options={"HIDDEN"})
    bake_map_asset: bpy.props.StringProperty(default="", options={"HIDDEN"})
    bake_detected_maps: bpy.props.StringProperty(default="", options={"HIDDEN"})

    def _set_bake_maps(self, maps) -> None:
        global _PROPOSING_MAPS
        previous = _PROPOSING_MAPS
        _PROPOSING_MAPS = True
        try:
            for channel, prop in _MAP_PROPERTIES.items():
                setattr(self, prop, channel in maps)
        finally:
            _PROPOSING_MAPS = previous

    def _on_bake_maps_changed(self, context) -> None:
        if not _PROPOSING_MAPS:
            self.bake_maps_chosen = True
            obj = getattr(context, "active_object", None)
            if obj is not None and obj.type == "MESH":
                self.bake_map_asset = obj.get("rebuilt_asset", obj.name)

    def suggest_bake_maps(self, ctx, *, reset=False) -> None:
        """Propose source channels until the artist edits the map checkboxes."""
        from ..bridge.bake_maps import suggested_maps

        if reset or self.bake_map_asset not in ("", ctx.name):
            self.bake_maps_chosen = False
        self.bake_map_asset = ctx.name
        if self.bake_maps_chosen:
            return
        maps = suggested_maps(ctx.source_object)
        self.bake_detected_maps = ",".join(sorted(maps))
        self._set_bake_maps(maps)

    bake_normal: bpy.props.BoolProperty(
        options=set(),
        name="Normal",
        default=True,
        update=lambda self, context: self._on_bake_maps_changed(context),
        description="Source surface detail as a tangent-space normal map",
    )
    bake_base_color: bpy.props.BoolProperty(
        options=set(),
        name="Base Color",
        default=True,
        update=lambda self, context: self._on_bake_maps_changed(context),
        description="Source base color without scene lighting",
    )
    bake_ao: bpy.props.BoolProperty(
        options=set(),
        name="Ambient Occlusion",
        default=True,
        update=lambda self, context: self._on_bake_maps_changed(context),
        description="Contact shadow. The only map whose quality depends on samples",
    )
    bake_roughness: bpy.props.BoolProperty(
        options=set(),
        name="Roughness",
        default=False,
        update=lambda self, context: self._on_bake_maps_changed(context),
        description="Useful when the source material varies across the surface",
    )
    bake_metallic: bpy.props.BoolProperty(
        options=set(),
        name="Metallic",
        default=False,
        update=lambda self, context: self._on_bake_maps_changed(context),
        description="Useful when the source material varies across the surface",
    )
    bake_opacity: bpy.props.BoolProperty(
        options=set(),
        name="Opacity",
        default=False,
        update=lambda self, context: self._on_bake_maps_changed(context),
        description="Alpha, for models with cut-out geometry such as foliage",
    )
    bake_normal_space: bpy.props.EnumProperty(
        options=set(),
        name="Normal Space",
        items=[
            ("TANGENT", "Tangent", "What game engines expect. Deforms with the mesh"),
            ("OBJECT", "Object", "For props that never deform"),
        ],
        default="TANGENT",
    )
    bake_ao_samples: bpy.props.IntProperty(
        options=set(),
        name="AO Samples",
        description=(
            "Rays per texel for ambient occlusion. Only this map integrates over "
            "a hemisphere, so it is the only one the number changes: at 8 the "
            "result is noise, and the rest look identical at 1 or at 64"
        ),
        default=256,
        min=8,
        soft_max=1024,
    )

    target_engine: bpy.props.EnumProperty(
        options=set(),
        name="Engine",
        description="Which engine the asset is going to, and what that engine expects",
        items=[
            (
                "unity",
                "Unity",
                "One Unity folder with the model, assigned materials and textures",
                1,
            ),
            (
                "unreal",
                "Unreal",
                "One GLB with materials and textures for native Unreal import",
                2,
            ),
            (
                "godot",
                "Godot",
                "One GLB with materials and textures for Godot import",
                3,
            ),
        ],
        default="unity",
        update=_on_target_engine_changed,
    )
    unity_pipeline: bpy.props.EnumProperty(
        options=set(),
        name="Render Pipeline",
        description="Match the render pipeline used by the destination Unity project",
        items=[
            ("urp", "URP", "Universal Render Pipeline"),
            ("builtin", "Built-in", "Built-in Render Pipeline with Standard materials"),
            ("hdrp", "HDRP", "High Definition Render Pipeline"),
        ],
        default="urp",
    )
    export_format: bpy.props.EnumProperty(
        options=set(),
        name="Format",
        description="What file to write",
        items=_export_format_items,
        default=0,
    )

    show_retopo_options: bpy.props.BoolProperty(options=set(), name="Options", default=False)
    show_seams_options: bpy.props.BoolProperty(options=set(), name="Seam Options", default=False)
    show_uv_options: bpy.props.BoolProperty(options=set(), name="Options", default=False)
    show_bake_options: bpy.props.BoolProperty(options=set(), name="Options", default=False)

    mode: bpy.props.EnumProperty(
        options=set(),
        name="Mode",
        description="Let the add-on build the asset, or take it one stage at a time",
        items=[
            (
                "auto",
                "Auto",
                "Choose the type, the quad count and the texture size, press the "
                "button, and get the finished asset",
            ),
            (
                "manual",
                "Manual",
                "Run one stage at a time, with every setting in reach — check the "
                "retopology before unwrapping, place your own seams, redo just the bake",
            ),
        ],
        default="auto",
    )

    def planned_summary(self) -> tuple[int, int]:
        """Read the planned quad count and texture size without changing settings."""
        return int(self.target_quads), int(self.texture_resolution)

    def propose_profile(self, asset_class: str) -> None:
        """Suggest the detected profile until the artist explicitly chooses one."""
        global _PROPOSING

        if not asset_class or self.profile_chosen:
            return
        try:
            suggested_profile = profiles.for_asset_class(asset_class)
        except Exception:  # noqa: BLE001 — detection must not interrupt processing
            return
        if not suggested_profile.listed or suggested_profile.identifier == self.profile:
            return

        requested_budget, texture_size_px = self.target_quads, self.texture_resolution
        _PROPOSING = True
        try:
            self.profile = suggested_profile.identifier
        finally:
            _PROPOSING = False
        self.target_quads = requested_budget
        self.texture_resolution = texture_size_px

    def build_recipe(self, asset_class: str = "", *, flow_source=None):
        """The recipe these settings mean."""
        if asset_class:
            self.propose_profile(asset_class)
        profile = profiles.get(self.profile)
        import dataclasses
        from .flow_guides import recipe_fields

        recipe = profile.recipe("balanced")

        recipe = dataclasses.replace(
            recipe,
            retopo=dataclasses.replace(
                recipe.retopo,
                topology=self.topology,
                remesher=self.remesher,
                quad_budget_mode=self.quad_budget_mode,
                adaptive_density=self.adaptive_size > 0.0,
                curvature_adaptivity=max(0.0, min(1.0, self.adaptive_size / 100.0)),
                curvature_anisotropy=max(0.0, min(1.0, self.adaptive_shape / 100.0)),
                cloth_folds=self.cloth_folds if self.mode == "manual" else True,
                preserve_boundary=self.keep_open_edges,
                density_contrast=recipe.retopo.density_contrast * (self.adaptive_size / 100.0),
                **recipe_fields(self, flow_source),
            ),
        )

        map_controls = tuple(name for name, prop in _MAP_PROPERTIES.items() if getattr(self, prop))
        recipe = dataclasses.replace(
            recipe,
            uv=dataclasses.replace(
                recipe.uv,
                unwrap_method=self.uv_unwrap_method,
                island_preference=self.uv_island_preference if self.mode == "manual" else "BALANCED",
                keep_marked_seams=self.uv_keep_marked_seams if self.mode == "manual" else False,
                seam_angle_deg=math.degrees(self.uv_seam_angle),
                padding_px=self.uv_padding_px,
                shape_method=self.uv_shape_method,
                rotate_method=self.uv_rotate_method,
                packer=self.uv_packer,
                pack_seconds=self.uv_pack_seconds,
            ),
            bake=dataclasses.replace(
                recipe.bake,
                maps=map_controls or ("NORMAL",),
                normal_space=self.bake_normal_space,
                ao_samples=self.bake_ao_samples,
                image_format=self.bake_image_format,
                image_depth=self.bake_image_depth,
                png_compression=self.bake_png_compression,
                jpeg_quality=self.bake_jpeg_quality,
                tiff_codec=self.bake_tiff_codec,
                exr_depth=self.bake_exr_depth,
                exr_codec=self.bake_exr_codec,
            ),
        )

        size = int(self.texture_resolution)
        return dataclasses.replace(
            recipe,
            texture_resolution=size,
            retopo=dataclasses.replace(recipe.retopo, target_faces=self.target_quads),
            bake=dataclasses.replace(recipe.bake, resolution=size),
        )


for _channel, (_label, _socket, _color, _default) in EXTRA_CHANNELS.items():
    REBUILT_Settings.__annotations__[_MAP_PROPERTIES[_channel]] = bpy.props.BoolProperty(
        options=set(),
        name=_label,
        default=False,
        description=f"Transfer the source material's {_socket} input",
        update=lambda self, context: self._on_bake_maps_changed(context),
    )

_CLASSES = (REBUILT_Settings,)


@bpy.app.handlers.persistent
def _migrate_removed_engines(_unused=None) -> None:
    """Normalize engine-dependent export settings."""
    for scene in getattr(bpy.data, "scenes", ()):
        stored = scene.rebuilt
        if stored.get("remesher") in (1, 3, "instant", "field"):
            scene.rebuilt.remesher = "auto"
        if stored.get("target_engine") in (0, "generic"):
            stored.target_engine = "unity"
        if stored.target_engine == "unity":
            stored.export_format = "fbx"
        elif stored.target_engine in {"unreal", "godot"}:
            stored.export_format = "gltf"


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.rebuilt = bpy.props.PointerProperty(type=REBUILT_Settings)

    bpy.app.timers.register(_migrate_removed_engines, first_interval=0.0)
    bpy.app.handlers.load_post.append(_migrate_removed_engines)


def unregister() -> None:
    if bpy.app.timers.is_registered(_migrate_removed_engines):
        bpy.app.timers.unregister(_migrate_removed_engines)
    if _migrate_removed_engines in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_migrate_removed_engines)
    del bpy.types.Scene.rebuilt
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
