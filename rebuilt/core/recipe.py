"""Recipes: the serialisable settings that drive each pipeline stage."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable

SCHEMA_VERSION = 1


_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def migration(from_version: int) -> Callable[[Callable], Callable]:
    """Register an upgrade step from `from_version` to `from_version + 1`."""

    def decorator(func: Callable[[dict[str, Any]], dict[str, Any]]) -> Callable:
        if from_version in _MIGRATIONS:
            raise ValueError(f"a migration from version {from_version} is already registered")
        _MIGRATIONS[from_version] = func
        return func

    return decorator


class RecipeMigrationError(RuntimeError):
    """A payload cannot be brought up to the current schema version."""


def migrate(payload: dict[str, Any]) -> dict[str, Any]:
    """Bring a stored payload up to SCHEMA_VERSION, one registered step at a time."""
    if not isinstance(payload, dict):
        raise RecipeMigrationError('A stored recipe must be a JSON object.')
    version = payload.get("schema_version", 1)
    if type(version) is not int:
        raise RecipeMigrationError(f"schema_version must be an int, got {version!r}")
    if version > SCHEMA_VERSION:
        raise RecipeMigrationError(
            f"payload is schema version {version} but this build only understands "
            f"{SCHEMA_VERSION}. It was written by a newer Rebuilt."
        )

    payload = dict(payload)
    while version < SCHEMA_VERSION:
        step = _MIGRATIONS.get(version)
        if step is None:
            raise RecipeMigrationError(
                f"no migration registered from schema version {version} to {version + 1}"
            )
        payload = step(payload)
        payload["schema_version"] = version + 1
        version += 1
    return payload


def _decode(annotation: Any, value: Any) -> Any:
    """Rebuild a typed value from plain JSON data."""
    origin = typing.get_origin(annotation)

    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(annotation)
        if value is None and type(None) in args:
            return None
        for item_type in args:
            if item_type is type(None):
                continue
            try:
                return _decode(item_type, value)
            except ValueError:
                pass
        raise ValueError('Value does not match the stored field type.')

    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ValueError('Expected an array.')
        args = typing.get_args(annotation)
        item_type = args[0] if args else Any
        decoded = [_decode(item_type, item) for item in value]
        return tuple(decoded) if origin is tuple else decoded

    if dataclasses.is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ValueError('Expected a settings object.')
        return _from_dict(annotation, value)

    if annotation in (bool, int, str) and type(value) is not annotation:
        raise ValueError(f'Expected {annotation.__name__}.')
    if annotation is float:
        if type(value) not in (int, float):
            raise ValueError('Expected a finite number.')
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError('Expected a finite number.')

    return value


def _from_dict(cls: type, payload: dict[str, Any]) -> Any:
    if not isinstance(payload, dict):
        raise ValueError('Expected a settings object.')
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}

    unknown = set(payload) - known - {"schema_version"}
    if unknown:
        payload = {k: v for k, v in payload.items() if k not in unknown}

    kwargs = {}
    for name, value in payload.items():
        if name in known:
            try:
                kwargs[name] = _decode(hints[name], value)
            except ValueError as error:
                raise ValueError(f'{cls.__name__}.{name}: {error}') from error
    return cls(**kwargs)


@dataclass(frozen=True)
class Recipe:
    """Base for every recipe. Subclasses just add fields."""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Any:
        return _from_dict(cls, payload)

    def fingerprint(self) -> str:
        """Short stable hash of this recipe's values."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class IngestRecipe(Recipe):
    """How the scan is taken in and prepared."""

    apply_transforms: bool = True

    proxy_triangle_threshold: int = 2_000_000
    proxy_target_triangles: int = 500_000


@dataclass(frozen=True)
class CleanupRecipe(Recipe):
    """What counts as a defect, and which repairs are allowed."""

    floater_relative_size: float = 0.05
    merge_distance: float = 0.0001
    degenerate_area_ratio: float = 1e-8

    remove_floaters: bool = False
    merge_doubles: bool = False
    remove_degenerate: bool = False


@dataclass(frozen=True)
class AnalyzeRecipe(Recipe):
    """How the scan is measured and classified."""

    hard_edge_angle_deg: float = 35.0
    coplanar_angle_deg: float = 10.0
    curvature_percentile: float = 99.0
    curvature_weight: float = 0.6
    feature_weight: float = 0.4
    detect_symmetry: bool = True
    symmetry_extra_directions: int = 0

    asset_class_override: str = ""


@dataclass(frozen=True)
class RetopoRecipe(Recipe):
    strategy: str = "auto"

    topology: str = "quads"

    remesher: str = "auto"
    target_faces: int = 25_000

    quad_budget_mode: str = "strict"
    adaptive_density: bool = True
    density_contrast: float = 3.0

    curvature_adaptivity: float = 1.0
    curvature_anisotropy: float = 1.0
    cloth_folds: bool = True
    flow_guides_key: str = ""
    flow_guides_strength: float = 12.0
    preserve_sharp: bool = True

    preserve_boundary: bool = False

    planar_angle_ladder: tuple[float, ...] = (5.0, 8.0, 10.0, 12.0, 15.0, 20.0)
    project_to_surface: bool = True
    max_deviation: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:

        if self.remesher in {"instant", "field"}:
            object.__setattr__(self, "remesher", "auto")


@dataclass(frozen=True)
class UVRecipe(Recipe):
    unwrap_method: str = "ANGLE_BASED"
    island_preference: str = "BALANCED"
    keep_marked_seams: bool = False

    seam_angle_deg: float = 40.0

    max_stretch: float = 0.15
    texel_density: float = 1024.0

    preserve_texel_density: bool = False
    padding_px: int = 8
    rotate_method: str = "ANY"
    shape_method: str = "CONCAVE"
    use_udim: bool = False

    packer: str = "uvpack"

    pack_seconds: float = 2.0


@dataclass(frozen=True)
class BakeRecipe(Recipe):
    maps: tuple[str, ...] = ("NORMAL", "BASE_COLOR", "AO")
    resolution: int = 4096

    samples: int = 4

    ao_samples: int = 256
    cage_safety_factor: float = 1.25
    max_retries: int = 2
    normal_space: str = "TANGENT"
    image_format: str = "PNG"
    image_depth: str = "16"
    png_compression: int = 15
    jpeg_quality: int = 85
    tiff_codec: str = "DEFLATE"
    exr_depth: str = "16"
    exr_codec: str = "ZIP"


@dataclass(frozen=True)
class OutputRecipe(Recipe):
    """Record the most recent optional export settings."""

    target_engine: str = "generic"

    export_format: str = "none"

    export_folder: str = "export"
    unity_pipeline: str = "urp"


@dataclass(frozen=True)
class QARecipe(Recipe):
    max_deviation_p95: float = 0.002
    max_uv_stretch_p95: float = 0.15
    min_valid_bake_pixels: float = 0.995
    allow_uv_overlap: bool = False
    warnings_as_errors: bool = False


@dataclass(frozen=True)
class AssetRecipe(Recipe):
    """Everything needed to reproduce one asset from one scan."""

    schema_version: int = SCHEMA_VERSION
    profile: str = "generic_organic"
    detail: str = "balanced"
    texture_resolution: int = 4096
    ingest: IngestRecipe = field(default_factory=IngestRecipe)
    cleanup: CleanupRecipe = field(default_factory=CleanupRecipe)
    analyze: AnalyzeRecipe = field(default_factory=AnalyzeRecipe)
    retopo: RetopoRecipe = field(default_factory=RetopoRecipe)
    uv: UVRecipe = field(default_factory=UVRecipe)
    bake: BakeRecipe = field(default_factory=BakeRecipe)
    qa: QARecipe = field(default_factory=QARecipe)
    output: OutputRecipe = field(default_factory=OutputRecipe)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AssetRecipe:
        return _from_dict(cls, migrate(payload))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, text: str) -> AssetRecipe:
        return cls.from_dict(json.loads(text))
