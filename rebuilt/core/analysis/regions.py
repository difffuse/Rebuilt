"""Segmenting a scan into flat regions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import MeshArrays
from .mesh_graph import connected_components, face_areas, face_normals, manifold_edges


MIN_PLANAR_FACES = 4
MIN_PLANAR_AREA_FRACTION = 0.05


@dataclass(frozen=True)
class PlanarRegion:
    """A patch of surface grown from faces that meet at shallow angles."""

    index: int
    face_count: int
    area: float
    normal: tuple[float, float, float]
    origin: tuple[float, float, float]

    flatness_error: float

    @property
    def is_flat(self) -> bool:
        """Whether the region lies on a plane."""
        return self.flatness_error < 0.01


@dataclass(frozen=True)
class Segmentation:
    """Every face labelled with the region it belongs to."""

    face_labels: np.ndarray
    regions: list[PlanarRegion]
    region_count: int = 0
    total_area: float = 0.0

    def planar_area_ratio(self, total_area: float | None = None) -> float:
        """Fraction of the surface sitting in a substantial flat region."""
        total = self.total_area if total_area is None else total_area
        if total <= 0:
            return 0.0
        return float(sum(r.area for r in self.regions if r.is_flat) / total)


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares plane through a point set."""
    if points.shape[0] < 3:
        origin = points.mean(axis=0) if points.shape[0] else np.zeros(3)
        return origin, np.array([0.0, 0.0, 1.0]), 0.0

    origin = points.mean(axis=0)
    centred = points - origin

    _, _, right = np.linalg.svd(centred, full_matrices=False)
    normal = right[-1]

    distances = np.abs(centred @ normal)
    return origin, normal, float(distances.max())


def segment_planar_regions(
    mesh: MeshArrays,
    coplanar_angle_deg: float = 10.0,
    normals: np.ndarray | None = None,
) -> Segmentation:
    """Group faces into regions that are flat to within `coplanar_angle_deg`."""
    if mesh.num_faces == 0:
        return Segmentation(np.empty(0, dtype=np.int64), [])

    if normals is None:
        normals = face_normals(mesh)

    edges = manifold_edges(mesh, normals)
    threshold = np.radians(coplanar_angle_deg)
    joins = (
        edges.faces[edges.dihedral <= threshold]
        if edges.count
        else np.empty((0, 2), dtype=np.int64)
    )

    face_labels = connected_components(mesh.num_faces, joins)
    areas = face_areas(mesh)
    total_area = float(areas.sum())
    diagonal = float(np.linalg.norm(mesh.positions.max(axis=0) - mesh.positions.min(axis=0)))

    region_count = int(face_labels.max()) + 1
    region_area = np.bincount(face_labels, weights=areas, minlength=region_count)
    region_faces = np.bincount(face_labels, minlength=region_count)

    big_enough = region_faces >= MIN_PLANAR_FACES
    if total_area > 0:
        big_enough |= region_area >= MIN_PLANAR_AREA_FRACTION * total_area
    significant = np.flatnonzero(big_enough)

    order = np.argsort(face_labels, kind="stable")
    sorted_labels = face_labels[order]

    regions: list[PlanarRegion] = []
    for index in significant:
        low, high = np.searchsorted(sorted_labels, [index, index + 1])
        vertices = _region_vertices(mesh, order[low:high])
        origin, normal, deviation = fit_plane(mesh.positions[vertices].astype(np.float64))

        regions.append(
            PlanarRegion(
                index=int(index),
                face_count=int(region_faces[index]),
                area=float(region_area[index]),
                normal=tuple(normal.tolist()),
                origin=tuple(origin.tolist()),
                flatness_error=deviation / diagonal if diagonal > 0 else 0.0,
            )
        )

    regions.sort(key=lambda r: r.area, reverse=True)
    return Segmentation(
        face_labels=face_labels,
        regions=regions,
        region_count=region_count,
        total_area=total_area,
    )


def _region_vertices(mesh: MeshArrays, faces: np.ndarray) -> np.ndarray:
    """Unique vertex indices used by the given faces."""
    if faces.size == 0:
        return np.empty(0, dtype=np.int64)

    sizes = mesh.face_sizes[faces].astype(np.int64)
    starts = mesh.face_starts[faces].astype(np.int64)
    total = int(sizes.sum())
    within = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    corners = np.repeat(starts, sizes) + within
    return np.unique(mesh.corner_verts[corners].astype(np.int64))
