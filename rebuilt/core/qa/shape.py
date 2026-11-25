"""How faithfully the low mesh reproduces the high one."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DeviationReport:
    """Distance from the low mesh to the high surface, in relative terms."""

    median: float = 0.0
    p95: float = 0.0
    maximum: float = 0.0
    mean: float = 0.0
    sample_count: int = 0
    unresolved: int = 0

    def within(self, tolerance: float) -> bool:
        return self.p95 <= tolerance

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deviation_report(distances: np.ndarray, diagonal: float) -> DeviationReport:
    """Summarise nearest-surface distances relative to the object's size."""
    if distances.size == 0 or diagonal <= 0:
        return DeviationReport(sample_count=int(distances.size))

    finite = np.isfinite(distances)
    unresolved = int((~finite).sum())
    usable = distances[finite] / diagonal

    if usable.size == 0:
        return DeviationReport(sample_count=int(distances.size), unresolved=unresolved)

    return DeviationReport(
        median=float(np.median(usable)),
        p95=float(np.percentile(usable, 95)),
        maximum=float(usable.max()),
        mean=float(usable.mean()),
        sample_count=int(distances.size),
        unresolved=unresolved,
    )


def budget_error(achieved: int, target: int) -> float:
    """Signed fractional miss against a polygon budget."""
    if target <= 0:
        return 0.0
    return (achieved - target) / target
