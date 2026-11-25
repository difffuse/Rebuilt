"""Fitting simple shapes to parts of a scan."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CylinderFit:
    """A cylinder fitted to a set of oriented surface samples."""

    axis: tuple[float, float, float]
    origin: tuple[float, float, float]
    radius: float

    radial_error: float

    axis_confidence: float

    @property
    def is_cylindrical(self) -> bool:
        return self.radial_error < 0.05 and self.axis_confidence > 0.9


def fit_cylinder(points: np.ndarray, normals: np.ndarray) -> CylinderFit:
    """Fit a cylinder to oriented samples."""
    if points.shape[0] < 3:
        return CylinderFit((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.0, 1.0, 0.0)

    points = points.astype(np.float64)
    normals = normals.astype(np.float64)

    scatter = normals.T @ normals
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    axis = eigenvectors[:, 0]
    axis = axis / np.linalg.norm(axis)

    spread = float(eigenvalues[1:].mean())
    confidence = 1.0 - float(eigenvalues[0]) / spread if spread > 0 else 0.0

    centroid = points.mean(axis=0)
    relative = points - centroid

    along = relative @ axis
    perpendicular = relative - np.outer(along, axis)
    radii = np.linalg.norm(perpendicular, axis=1)

    radius = float(radii.mean())
    error = float(radii.std() / radius) if radius > 0 else 1.0

    return CylinderFit(
        axis=tuple(axis.tolist()),
        origin=tuple(centroid.tolist()),
        radius=radius,
        radial_error=error,
        axis_confidence=float(np.clip(confidence, 0.0, 1.0)),
    )


def angle_between_axes(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    """Angle in degrees between two axes, ignoring their sign."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0))))
