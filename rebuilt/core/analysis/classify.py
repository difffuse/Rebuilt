"""Deciding what kind of thing the scan is."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

HARD_SURFACE = "hard_surface"
GENERIC_ORGANIC = "generic_organic"
ARCHITECTURE = "architecture"
UNKNOWN = "unknown"


MINIMUM_SCORE = 0.35


ARCHITECTURE_MIN_SIZE = 5.0


@dataclass(frozen=True)
class ClassificationEvidence:
    """The measurements a decision was made from."""

    planar_area_ratio: float = 0.0
    sharp_edge_ratio: float = 0.0
    axis_alignment: float = 0.0
    bbox_diagonal: float = 0.0
    component_count: int = 0

    def as_dict(self) -> dict[str, float]:
        return {
            "planar_area_ratio": self.planar_area_ratio,
            "sharp_edge_ratio": self.sharp_edge_ratio,
            "axis_alignment": self.axis_alignment,
            "bbox_diagonal": self.bbox_diagonal,
            "component_count": float(self.component_count),
        }


@dataclass(frozen=True)
class Classification:
    """What the scan probably is, and how sure that is."""

    asset_class: str = UNKNOWN
    confidence: float = 0.0
    evidence: ClassificationEvidence = field(default_factory=ClassificationEvidence)
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def needs_review(self) -> bool:
        return self.asset_class == UNKNOWN or self.confidence < 0.6


class Classifier(Protocol):
    """Anything that can name an asset class from measurements."""

    def classify(self, evidence: ClassificationEvidence) -> Classification: ...


def axis_alignment_ratio(
    normals: np.ndarray, areas: np.ndarray, tolerance_deg: float = 10.0
) -> float:
    """Area fraction whose normal points along a world axis."""
    total = float(areas.sum())
    if total <= 0 or normals.shape[0] == 0:
        return 0.0

    threshold = np.cos(np.radians(tolerance_deg))
    aligned = (np.abs(normals) >= threshold).any(axis=1)
    return float(areas[aligned].sum() / total)


class HeuristicClassifier:
    """The default rule-based classifier."""

    def classify(self, evidence: ClassificationEvidence) -> Classification:
        planar = float(np.clip(evidence.planar_area_ratio, 0.0, 1.0))

        creased = float(np.clip(evidence.sharp_edge_ratio * 10.0, 0.0, 1.0))
        aligned = float(np.clip(evidence.axis_alignment, 0.0, 1.0))

        big_enough = evidence.bbox_diagonal >= ARCHITECTURE_MIN_SIZE

        scores = {
            HARD_SURFACE: 0.7 * planar + 0.3 * creased,
            ARCHITECTURE: (0.5 * planar + 0.5 * aligned) * (1.0 if big_enough else 0.35),
            GENERIC_ORGANIC: (1.0 - planar) * (1.0 - 0.5 * creased),
        }

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        (best_class, best_score), (_, runner_up) = ranked[0], ranked[1]

        if best_score < MINIMUM_SCORE:
            return Classification(UNKNOWN, 0.0, evidence, scores)

        margin = (best_score - runner_up) / best_score if best_score > 0 else 0.0
        confidence = float(np.clip(best_score * (0.55 + 0.45 * min(margin * 2.0, 1.0)), 0.0, 1.0))

        return Classification(best_class, confidence, evidence, scores)


DEFAULT_CLASSIFIER: Classifier = HeuristicClassifier()
