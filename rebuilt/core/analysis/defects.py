"""Finding what is wrong with a scan, without fixing any of it."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..types import MeshArrays
from .mesh_graph import (
    Component,
    boundary_loop_count,
    components_of,
    duplicate_vertex_groups,
    edge_topology,
    face_areas,
)


PLAUSIBLE_SIZE_RANGE = (0.001, 10_000.0)


@dataclass(frozen=True)
class Floater:
    """A disconnected fragment small enough to be scanning debris."""

    component: int
    face_count: int
    area: float
    relative_size: float


@dataclass
class DefectReport:
    """What a scan looks like before anything has been done to it."""

    vertex_count: int = 0
    face_count: int = 0
    triangle_count: int = 0

    bbox_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_max: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_diagonal: float = 0.0

    component_count: int = 0
    floaters: list[Floater] = field(default_factory=list)

    boundary_edge_count: int = 0
    hole_count: int = 0
    non_manifold_edge_count: int = 0
    degenerate_face_count: int = 0
    duplicate_vertex_groups: int = 0

    scale_warning: str | None = None

    @property
    def floater_count(self) -> int:
        return len(self.floaters)

    @property
    def is_watertight(self) -> bool:
        return self.boundary_edge_count == 0 and self.non_manifold_edge_count == 0

    @property
    def is_clean(self) -> bool:
        """Nothing here needs the user's attention."""
        return (
            not self.floaters
            and self.non_manifold_edge_count == 0
            and self.degenerate_face_count == 0
            and self.duplicate_vertex_groups == 0
            and self.scale_warning is None
        )

    def findings(self) -> list[str]:
        """Human-readable list of what was found, worst first."""
        lines: list[str] = []
        if self.scale_warning:
            lines.append(self.scale_warning)
        if self.floaters:
            area = sum(f.area for f in self.floaters)
            lines.append(f"{len(self.floaters)} loose fragment(s), {area:.4g} total area")
        if self.non_manifold_edge_count:
            lines.append(f"{self.non_manifold_edge_count} non-manifold edge(s)")
        if self.degenerate_face_count:
            lines.append(f"{self.degenerate_face_count} zero-area face(s)")
        if self.duplicate_vertex_groups:
            lines.append(f"{self.duplicate_vertex_groups} group(s) of coincident vertices")
        if self.hole_count:
            lines.append(f"{self.hole_count} open boundary loop(s)")
        return lines

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DefectReport:
        payload = dict(payload)
        payload["floaters"] = [Floater(**f) for f in payload.get("floaters", [])]
        for key in ("bbox_min", "bbox_max"):
            if key in payload:
                payload[key] = tuple(payload[key])
        return cls(**payload)


def assess_scale(diagonal: float) -> str | None:
    """Flag a scan that is probably in the wrong units."""
    low, high = PLAUSIBLE_SIZE_RANGE
    if diagonal <= 0:
        return "Scan has no size — is the mesh empty?"
    if diagonal < low:
        return (
            f"Scan is {diagonal:.4g} units across, which is suspiciously small. "
            "Check the import scale before continuing."
        )
    if diagonal > high:
        return (
            f"Scan is {diagonal:.4g} units across, which is suspiciously large. "
            "Check the import scale before continuing."
        )
    return None


def find_floaters(components: list[Component], relative_size: float) -> list[Floater]:
    """Fragments whose bounding box is a small fraction of the main body's."""
    if len(components) < 2:
        return []

    largest = max(c.bbox_diagonal for c in components)
    if largest <= 0:
        return []

    main = max(components, key=lambda c: c.area)
    return [
        Floater(
            component=c.index,
            face_count=c.face_count,
            area=c.area,
            relative_size=c.bbox_diagonal / largest,
        )
        for c in components
        if c.index != main.index and c.bbox_diagonal / largest < relative_size
    ]


def scan_defects(
    mesh: MeshArrays,
    *,
    floater_relative_size: float = 0.05,
    merge_distance: float = 0.0001,
    degenerate_area_ratio: float = 1e-8,
    report=None,
) -> DefectReport:
    """Inspect a mesh and describe everything questionable about it."""

    def step(fraction: float, message: str) -> None:
        if report is not None:
            report.update(fraction, message)

    result = DefectReport(
        vertex_count=mesh.num_verts,
        face_count=mesh.num_faces,
        triangle_count=int(np.maximum(mesh.face_sizes - 2, 0).sum()),
    )

    if mesh.num_verts == 0:
        result.scale_warning = assess_scale(0.0)
        return result

    step(0.05, "Measuring")
    bbox_min = mesh.positions.min(axis=0)
    bbox_max = mesh.positions.max(axis=0)
    result.bbox_min = tuple(bbox_min.tolist())
    result.bbox_max = tuple(bbox_max.tolist())
    result.bbox_diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    result.scale_warning = assess_scale(result.bbox_diagonal)

    step(0.2, "Checking edges")
    topology = edge_topology(mesh)
    result.boundary_edge_count = int(topology.boundary.sum())
    result.non_manifold_edge_count = int(topology.non_manifold.sum())

    step(0.4, "Tracing boundaries")
    result.hole_count = boundary_loop_count(topology)

    step(0.55, "Finding loose fragments")
    _, components = components_of(mesh)
    result.component_count = len(components)
    result.floaters = find_floaters(components, floater_relative_size)

    step(0.8, "Checking faces")
    areas = face_areas(mesh)
    if areas.size:
        mean_area = float(areas.mean())
        threshold = mean_area * degenerate_area_ratio
        result.degenerate_face_count = int((areas <= threshold).sum())

    step(0.95, "Checking for doubles")
    result.duplicate_vertex_groups = duplicate_vertex_groups(mesh.positions, merge_distance)

    step(1.0, "Done")
    return result
