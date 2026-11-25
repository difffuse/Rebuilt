"""Judging a baked map from the numbers, not by eye."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..analysis.mesh_graph import connected_components


@dataclass(frozen=True)
class BakeQuality:
    """How a single baked map came out."""

    covered_texels: int = 0
    written_texels: int = 0
    missed_texels: int = 0
    stray_texels: int = 0

    ray_distance_median: float = 0.0
    ray_distance_p95: float = 0.0
    ray_distance_max: float = 0.0

    @property
    def valid_fraction(self) -> float:
        """Texels that were written from the right place, over texels that should be."""
        if self.covered_texels == 0:
            return 1.0
        good = self.covered_texels - self.missed_texels - self.stray_texels
        return float(max(good, 0) / self.covered_texels)

    def passes(self, minimum_valid: float) -> bool:
        return self.valid_fraction >= minimum_valid

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["valid_fraction"] = self.valid_fraction
        return payload


@dataclass(frozen=True)
class FaultRegions:
    """Where the faults are, so a retry can target them."""

    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    count: int = 0
    sizes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @property
    def largest(self) -> int:
        return int(self.sizes.max()) if self.sizes.size else 0


def written_mask(pixels: np.ndarray, sentinel: np.ndarray, tolerance: float = 1e-5) -> np.ndarray:
    """Which texels the bake actually touched."""
    if pixels.ndim != 2 or pixels.shape[1] != 4:
        raise ValueError(f"expected (N, 4) pixels, got {pixels.shape}")
    return ~np.all(np.abs(pixels - sentinel.reshape(1, 4)) <= tolerance, axis=1)


def assess(
    covered: np.ndarray,
    written: np.ndarray,
    baked_positions: np.ndarray,
    surface_positions: np.ndarray,
    envelope: np.ndarray,
) -> tuple[BakeQuality, np.ndarray, np.ndarray]:
    """Compare where each ray landed against where it should have."""
    missed = covered & ~written
    landed = covered & written

    distance = np.full(covered.shape[0], 0.0)
    if landed.any():
        distance[landed] = np.linalg.norm(
            baked_positions[landed] - surface_positions[landed], axis=1
        )

    stray = landed & ((distance > envelope) | ~np.isfinite(distance))

    good = landed & ~stray
    measured = distance[good]

    return (
        BakeQuality(
            covered_texels=int(covered.sum()),
            written_texels=int(landed.sum()),
            missed_texels=int(missed.sum()),
            stray_texels=int(stray.sum()),
            ray_distance_median=float(np.median(measured)) if measured.size else 0.0,
            ray_distance_p95=float(np.percentile(measured, 95)) if measured.size else 0.0,
            ray_distance_max=float(measured.max()) if measured.size else 0.0,
        ),
        missed,
        stray,
    )


def fault_regions(mask: np.ndarray, size: int) -> FaultRegions:
    """Group faulty texels into connected blobs, four-connected."""
    if mask.shape[0] != size * size:
        raise ValueError(f"mask has {mask.shape[0]} texels, expected {size * size}")

    empty = np.full(mask.shape[0], -1, dtype=np.int64)
    if not mask.any():
        return FaultRegions(empty)

    grid = mask.reshape(size, size)
    faulty = np.flatnonzero(mask)

    index = np.full(size * size, -1, dtype=np.int64)
    index[faulty] = np.arange(len(faulty))
    index = index.reshape(size, size)

    pairs = []
    vertical = grid[:-1, :] & grid[1:, :]
    if vertical.any():
        pairs.append(np.stack([index[:-1, :][vertical], index[1:, :][vertical]], axis=1))
    horizontal = grid[:, :-1] & grid[:, 1:]
    if horizontal.any():
        pairs.append(np.stack([index[:, :-1][horizontal], index[:, 1:][horizontal]], axis=1))

    edges = (np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=np.int64)).astype(np.int64)

    labels = connected_components(len(faulty), edges)

    unique, inverse = np.unique(labels, return_inverse=True)

    compact = empty.copy()
    compact[faulty] = inverse

    return FaultRegions(
        labels=compact,
        count=int(unique.shape[0]),
        sizes=np.bincount(inverse, minlength=unique.shape[0]).astype(np.int64),
    )


def fault_interior(mask: np.ndarray, size: int, margin: int = 8) -> np.ndarray:
    """Faults too far from a valid neighbour for normal texture padding."""
    if mask.shape != (size * size,):
        raise ValueError("fault mask must match the square texture")
    core = mask.reshape(size, size).copy()
    for _ in range(max(margin, 0)):
        if not core.any():
            break
        reduced = core.copy()
        reduced[1:] &= core[:-1]
        reduced[:-1] &= core[1:]
        reduced[:, 1:] &= core[:, :-1]
        reduced[:, :-1] &= core[:, 1:]
        core = reduced
    return core.reshape(-1)


def grow_offsets_at(
    offsets: np.ndarray,
    vertex_faults: np.ndarray,
    factor: float,
    ceiling: np.ndarray | None = None,
) -> np.ndarray:
    """Push the cage further out where the bake went wrong, and only there."""
    grown = offsets.copy()
    grown[vertex_faults] = offsets[vertex_faults] * factor
    if ceiling is not None:
        grown = np.minimum(grown, ceiling)
    return grown
