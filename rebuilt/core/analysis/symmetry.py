"""Finding a mirror plane."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SymmetryPlane:
    """A candidate mirror plane and how well the mesh matches across it."""

    normal: tuple[float, float, float]
    origin: tuple[float, float, float]

    error: float
    confidence: float

    @property
    def is_symmetric(self) -> bool:
        return self.confidence >= 0.8


SYMMETRY_TOLERANCE = 0.02


COARSEN_CELL = 0.03


def coarsen(points: np.ndarray, diagonal: float, cell: float = COARSEN_CELL) -> np.ndarray:
    """Reduce a point cloud to one representative per grid cell."""
    size = diagonal * cell
    if size <= 0 or points.shape[0] == 0:
        return points

    cells = np.round(points / size).astype(np.int64)
    _, first = np.unique(cells, axis=0, return_index=True)
    return points[first]


def _nearest_distances(queries: np.ndarray, targets: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Distance from each query to its nearest target, in chunks."""
    result = np.empty(queries.shape[0], dtype=np.float64)
    for start in range(0, queries.shape[0], chunk):
        block = queries[start : start + chunk]
        distances = np.linalg.norm(block[:, None, :] - targets[None, :, :], axis=2)
        result[start : start + block.shape[0]] = distances.min(axis=1)
    return result


def _subsample(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    if points.shape[0] <= count:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(points.shape[0], size=count, replace=False)]


def score_plane(
    positions: np.ndarray,
    normal: np.ndarray,
    origin: np.ndarray,
    *,
    diagonal: float,
    samples: int = 800,
    targets: int = 4000,
    seed: int = 0,
) -> SymmetryPlane:
    """How well the point set mirrors about one plane."""
    normal = normal / np.linalg.norm(normal)

    coarse = coarsen(positions, diagonal)
    query_points = _subsample(coarse, samples, seed)
    target_points = _subsample(coarse, targets, seed + 1)

    offsets = (query_points - origin) @ normal
    reflected = query_points - 2.0 * np.outer(offsets, normal)

    distances = _nearest_distances(reflected, target_points)
    error = float(distances.mean() / diagonal) if diagonal > 0 else 1.0

    confidence = float(np.clip(1.0 - error / (2.0 * SYMMETRY_TOLERANCE), 0.0, 1.0))

    return SymmetryPlane(
        normal=tuple(normal.tolist()),
        origin=tuple(origin.tolist()),
        error=error,
        confidence=confidence,
    )


def find_symmetry(
    positions: np.ndarray, seed: int = 0, extra_directions: int = 0
) -> SymmetryPlane | None:
    """The best mirror plane through the centroid, or None if there isn't one."""
    if positions.shape[0] < 8:
        return None

    positions = positions.astype(np.float64)
    diagonal = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    if diagonal <= 0:
        return None

    centroid = positions.mean(axis=0)
    centred = positions - centroid
    _, _, principal = np.linalg.svd(centred, full_matrices=False)

    candidates = [principal[i] for i in range(3)]
    for index in range(extra_directions):
        angle = np.pi * index / max(extra_directions, 1)
        candidates.append(np.cos(angle) * principal[0] + np.sin(angle) * principal[1])

    scored = [
        score_plane(positions, normal, centroid, diagonal=diagonal, seed=seed)
        for normal in candidates
    ]
    best = min(scored, key=lambda plane: plane.error)

    return best if best.error < SYMMETRY_TOLERANCE else None
