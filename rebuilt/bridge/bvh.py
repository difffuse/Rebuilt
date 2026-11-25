"""Nearest-surface queries, for measuring how far the low mesh strayed."""

from __future__ import annotations

import numpy as np
from mathutils.bvhtree import BVHTree

from ..core.analysis.mesh_graph import fan_triangles
from ..core.types import MeshArrays


class SurfaceQuery:
    """A built BVH over a triangulated surface."""

    def __init__(self, mesh: MeshArrays) -> None:
        triangles, _ = fan_triangles(mesh)
        self._empty = triangles.shape[0] == 0
        if self._empty:
            self._tree = None
            return

        self._tree = BVHTree.FromPolygons(
            [tuple(map(float, point)) for point in mesh.positions],
            [tuple(map(int, tri)) for tri in triangles],
            all_triangles=True,
        )

    @classmethod
    def from_mesh(cls, mesh: MeshArrays) -> SurfaceQuery:
        return cls(mesh)

    def nearest_distances(self, points: np.ndarray) -> np.ndarray:
        """Distance from each point to the nearest point on the surface."""
        if self._empty or points.shape[0] == 0:
            return np.full(points.shape[0], np.inf)

        distances = np.empty(points.shape[0], dtype=np.float64)
        for index, point in enumerate(points):
            _, _, _, distance = self._tree.find_nearest(tuple(map(float, point)))
            distances[index] = np.inf if distance is None else distance
        return distances

    def nearest_points(self, points: np.ndarray, sides: np.ndarray | None = None) -> np.ndarray:
        """The nearest point on the surface to each query point."""
        if self._empty or points.shape[0] == 0:
            return points.astype(np.float64, copy=True)

        projected = np.empty((points.shape[0], 3), dtype=np.float64)
        for index, point in enumerate(points):
            location, hit_normal, _, _ = self._tree.find_nearest(tuple(map(float, point)))
            if location is None:
                projected[index] = point
            elif (
                sides is not None
                and hit_normal is not None
                and float(np.dot(sides[index], hit_normal)) < 0.0
            ):
                projected[index] = point
            else:
                projected[index] = location
        return projected

    def inward_spans(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        caps: np.ndarray,
        epsilon: float = 1e-5,
    ) -> np.ndarray:
        """How thick the object is under each surface point, along its normal."""
        spans = np.full(points.shape[0], np.inf)
        if self._empty or points.shape[0] == 0:
            return spans

        for index in range(points.shape[0]):
            origin = points[index] - normals[index] * epsilon
            direction = -normals[index]
            location, _, _, distance = self._tree.ray_cast(
                tuple(map(float, origin)), tuple(map(float, direction)), float(caps[index])
            )
            if location is not None and distance is not None:
                spans[index] = distance + epsilon
        return spans

    def first_hits(
        self, points: np.ndarray, directions: np.ndarray, caps: np.ndarray, epsilon: float = 1e-5
    ) -> np.ndarray:
        """Distance to the first surface along each direction, `inf` past the cap."""
        hits = np.full(points.shape[0], np.inf)
        if self._empty or points.shape[0] == 0:
            return hits
        for index in range(points.shape[0]):
            origin = points[index] + directions[index] * epsilon
            location, _, _, distance = self._tree.ray_cast(
                tuple(map(float, origin)), tuple(map(float, directions[index])), float(caps[index])
            )
            if location is not None and distance is not None:
                hits[index] = distance + epsilon
        return hits

    def outward_reach(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        caps: np.ndarray,
        epsilon: float = 1e-5,
        max_crossings: int = 8,
    ) -> np.ndarray:
        """How far the surface rises above each point along its own normal."""
        result = np.zeros(points.shape[0], dtype=np.float64)
        if self._empty:
            return result

        for index in range(points.shape[0]):
            direction = tuple(map(float, normals[index]))
            origin = points[index].astype(np.float64) + normals[index] * epsilon
            travelled = 0.0

            for _ in range(max_crossings):
                remaining = float(caps[index]) - travelled
                if remaining <= epsilon:
                    break
                location, _, _, distance = self._tree.ray_cast(
                    tuple(map(float, origin)), direction, remaining
                )
                if location is None:
                    break
                travelled += distance + epsilon
                origin = np.array(location, dtype=np.float64) + normals[index] * epsilon
                result[index] = travelled

        return result


class PointQuery:
    """Nearest distances to a point set, through a KD-tree."""

    def __init__(self, points: np.ndarray) -> None:
        from mathutils.kdtree import KDTree

        self._empty = points.shape[0] == 0
        self._tree = KDTree(points.shape[0])
        for index, point in enumerate(points):
            self._tree.insert(tuple(map(float, point)), index)
        self._tree.balance()

    def nearest_distances(self, points: np.ndarray) -> np.ndarray:
        if self._empty or points.shape[0] == 0:
            return np.full(points.shape[0], np.inf)
        out = np.empty(points.shape[0], dtype=np.float64)
        for index, point in enumerate(points):
            _, _, distance = self._tree.find(tuple(map(float, point)))
            out[index] = np.inf if distance is None else distance
        return out


def sample_surface(mesh: MeshArrays) -> np.ndarray:
    """Points to measure a mesh's accuracy at: its vertices and face centres."""
    if mesh.num_faces == 0:
        return mesh.positions.astype(np.float64)

    corner_verts = mesh.corner_verts.astype(np.int64)
    face_of_corner = np.repeat(
        np.arange(mesh.num_faces, dtype=np.int64), mesh.face_sizes.astype(np.int64)
    )
    positions = mesh.positions.astype(np.float64)

    summed = np.stack(
        [
            np.bincount(
                face_of_corner, weights=positions[corner_verts, axis], minlength=mesh.num_faces
            )
            for axis in range(3)
        ],
        axis=1,
    )
    centres = summed / mesh.face_sizes.astype(np.float64)[:, None]

    return np.concatenate([positions, centres])
