"""Output profiles: what an asset is *for*, expressed as data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..recipe import AssetRecipe, BakeRecipe, QARecipe, RetopoRecipe, UVRecipe

PROFILE_DIR = Path(__file__).parent


DETAIL_SCALES: dict[str, float] = {
    "draft": 0.25,
    "balanced": 1.0,
    "high": 2.0,
    "max": 4.0,
}


class ProfileError(ValueError):
    """A profile file is missing, malformed, or names something unknown."""


@dataclass(frozen=True)
class Profile:
    """One way of turning a scan into an asset."""

    identifier: str
    label: str
    description: str = ""

    listed: bool = True
    target_faces: int = 25_000
    texture_resolution: int = 4096
    texel_density: float = 1024.0
    padding_px: int = 8
    maps: tuple[str, ...] = ("NORMAL", "BASE_COLOR", "AO")
    strategy: str = "auto"
    unwrap_method: str = "ANGLE_BASED"
    rotate_method: str = "ANY"
    max_deviation: float = 0.01
    max_stretch: float = 0.15
    limits: dict[str, float] = field(default_factory=dict)

    def recipe(self, detail: str = "balanced") -> AssetRecipe:
        """The recipe this profile implies at a given level of detail."""
        try:
            scale = DETAIL_SCALES[detail]
        except KeyError:
            raise ProfileError(
                f"unknown detail level {detail!r}; expected one of {sorted(DETAIL_SCALES)}"
            ) from None

        resolution = _clamp_power_of_two(int(self.texture_resolution * min(scale, 2.0)))

        return AssetRecipe(
            profile=self.identifier,
            detail=detail,
            texture_resolution=resolution,
            retopo=RetopoRecipe(
                strategy=self.strategy,
                target_faces=max(int(self.target_faces * scale), 8),
                max_deviation=self.max_deviation,
            ),
            uv=UVRecipe(
                unwrap_method=self.unwrap_method,
                max_stretch=self.max_stretch,
                texel_density=self.texel_density,
                padding_px=self.padding_px,
                rotate_method=self.rotate_method,
            ),
            bake=BakeRecipe(maps=tuple(self.maps), resolution=resolution),
            qa=QARecipe(
                max_deviation_p95=self.limits.get("max_deviation_p95", 0.002),
                max_uv_stretch_p95=self.max_stretch,
                min_valid_bake_pixels=self.limits.get("min_valid_bake_pixels", 0.995),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "label": self.label,
            "description": self.description,
            "target_faces": self.target_faces,
            "texture_resolution": self.texture_resolution,
            "texel_density": self.texel_density,
            "padding_px": self.padding_px,
            "maps": list(self.maps),
            "strategy": self.strategy,
            "unwrap_method": self.unwrap_method,
            "rotate_method": self.rotate_method,
            "max_deviation": self.max_deviation,
            "max_stretch": self.max_stretch,
            "limits": dict(self.limits),
        }


_KNOWN_KEYS = frozenset(Profile.__dataclass_fields__) - {"identifier"}


def from_dict(identifier: str, payload: dict[str, Any]) -> Profile:
    """Build a profile, refusing anything it does not understand."""
    unknown = set(payload) - _KNOWN_KEYS
    if unknown:
        raise ProfileError(f"profile {identifier!r} has unknown key(s): {sorted(unknown)}")
    if "maps" in payload:
        payload = {**payload, "maps": tuple(payload["maps"])}
    return Profile(identifier=identifier, **payload)


@lru_cache(maxsize=1)
def available() -> dict[str, Profile]:
    """Every profile shipped with the add-on, by identifier."""
    profiles: dict[str, Profile] = {}
    for path in sorted(PROFILE_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProfileError(f"could not read profile {path.name}: {exc}") from exc
        profiles[path.stem] = from_dict(path.stem, payload)
    return profiles


def get(identifier: str) -> Profile:
    try:
        return available()[identifier]
    except KeyError:
        raise ProfileError(f"unknown profile {identifier!r}; have {sorted(available())}") from None


def for_asset_class(asset_class: str) -> Profile:
    """The profile that suits a detected class, falling back to the generic one."""
    profiles = available()

    direct = {
        "hard_surface": "hard_surface",
        "architecture": "hard_surface",
        "generic_organic": "generic_organic",
    }.get(asset_class, "")
    return profiles.get(direct) or profiles["generic_organic"]


def _clamp_power_of_two(value: int, low: int = 256, high: int = 8192) -> int:
    """Round a texture size up to a power of two inside sane bounds."""
    clamped = max(low, min(int(value), high))
    candidate = 1 << (clamped - 1).bit_length()
    return max(low, min(candidate, high))
