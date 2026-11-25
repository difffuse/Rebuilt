"""The AnalysisGraph: everything understood about a scan before remeshing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..types import MeshArrays
from .classify import (
    DEFAULT_CLASSIFIER,
    Classification,
    ClassificationEvidence,
    Classifier,
    axis_alignment_ratio,
)
from .curvature import mean_curvature, normalise_curvature
from .features import Features, feature_proximity, find_features
from .fields import ImportanceWeights, importance_field
from .mesh_graph import components_of, face_areas, face_normals, manifold_edges
from .regions import Segmentation, segment_planar_regions
from .symmetry import SymmetryPlane, find_symmetry


@dataclass
class AnalysisGraph:
    """What the scan is, and where its detail lives."""

    classification: Classification = field(default_factory=Classification)
    symmetry: SymmetryPlane | None = None

    curvature: np.ndarray = field(default_factory=lambda: np.zeros(0))
    importance: np.ndarray = field(default_factory=lambda: np.zeros(0))

    face_region: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    region_count: int = 0
    planar_area_ratio: float = 0.0
    sharp_edge_count: int = 0
    boundary_edge_count: int = 0
    total_area: float = 0.0

    def summary(self) -> dict[str, Any]:
        """The parts worth storing in the scene and showing in the UI."""
        return {
            "asset_class": self.classification.asset_class,
            "confidence": self.classification.confidence,
            "needs_review": self.classification.needs_review,
            "evidence": self.classification.evidence.as_dict(),
            "scores": self.classification.scores,
            "region_count": self.region_count,
            "planar_area_ratio": self.planar_area_ratio,
            "sharp_edge_count": self.sharp_edge_count,
            "boundary_edge_count": self.boundary_edge_count,
            "total_area": self.total_area,
            "symmetry": (
                {
                    "normal": list(self.symmetry.normal),
                    "error": self.symmetry.error,
                    "confidence": self.symmetry.confidence,
                }
                if self.symmetry is not None
                else None
            ),
        }


def analyse(
    mesh: MeshArrays,
    *,
    hard_edge_angle_deg: float = 35.0,
    coplanar_angle_deg: float = 10.0,
    curvature_percentile: float = 99.0,
    weights: ImportanceWeights | None = None,
    detect_symmetry: bool = True,
    symmetry_extra_directions: int = 0,
    classifier: Classifier | None = None,
    report=None,
) -> AnalysisGraph:
    """Run the full analysis over a mesh."""

    def step(fraction: float, message: str) -> None:
        if report is not None:
            report.update(fraction, message)

    graph = AnalysisGraph()
    if mesh.num_verts == 0:
        return graph

    step(0.05, "Computing normals")
    normals = face_normals(mesh)
    areas = face_areas(mesh)
    graph.total_area = float(areas.sum())

    step(0.2, "Measuring curvature")
    edges = manifold_edges(mesh, normals)
    raw_curvature = mean_curvature(mesh, normals)
    graph.curvature = normalise_curvature(raw_curvature, curvature_percentile)

    step(0.4, "Finding features")
    features: Features = find_features(mesh, hard_edge_angle_deg, normals, edges)
    graph.sharp_edge_count = features.sharp_count
    graph.boundary_edge_count = features.boundary_count

    step(0.55, "Building importance field")
    proximity = feature_proximity(mesh, features)
    graph.importance = importance_field(graph.curvature, proximity, weights)

    step(0.7, "Segmenting flat regions")
    segmentation: Segmentation = segment_planar_regions(mesh, coplanar_angle_deg, normals)
    graph.face_region = segmentation.face_labels
    graph.region_count = segmentation.region_count
    graph.planar_area_ratio = segmentation.planar_area_ratio(graph.total_area)

    if detect_symmetry:
        step(0.85, "Looking for symmetry")
        graph.symmetry = find_symmetry(mesh.positions, extra_directions=symmetry_extra_directions)

    step(0.95, "Classifying")
    _, components = components_of(mesh)
    diagonal = float(np.linalg.norm(mesh.positions.max(axis=0) - mesh.positions.min(axis=0)))
    evidence = ClassificationEvidence(
        planar_area_ratio=graph.planar_area_ratio,
        sharp_edge_ratio=(features.sharp_count / edges.count) if edges.count else 0.0,
        axis_alignment=axis_alignment_ratio(normals, areas),
        bbox_diagonal=diagonal,
        component_count=len(components),
    )
    graph.classification = (classifier or DEFAULT_CLASSIFIER).classify(evidence)

    step(1.0, "Done")
    return graph
