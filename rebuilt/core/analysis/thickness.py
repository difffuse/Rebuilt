"""What a quad budget can and cannot hold: the thin parts of a scan."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CELLS_PER_WALL = 1.5


WARN_SHARE = 0.05
SEVERE_SHARE = 0.20


@dataclass(frozen=True)
class ThinReport:
    """How much of the surface a budget cannot hold."""

    share: float
    cell: float
    quads_to_keep: int

    @property
    def warns(self) -> bool:
        return self.share > WARN_SHARE

    @property
    def severe(self) -> bool:
        return self.share > SEVERE_SHARE


def cell_size(area: float, quads: int) -> float:
    """Edge of a quad when `quads` of them tile `area`."""
    return float(np.sqrt(max(area, 0.0) / max(quads, 1)))


def thin_share(
    thickness: np.ndarray, area: float, quads: int, cells: float = CELLS_PER_WALL
) -> float:
    """Fraction of the sampled surface thinner than `cells` quad edges."""
    thickness = np.asarray(thickness, dtype=np.float64)
    if thickness.size == 0:
        return 0.0
    return float((thickness < cells * cell_size(area, quads)).mean())


def quads_to_keep(
    thickness: np.ndarray, area: float, max_share: float = WARN_SHARE, cells: float = CELLS_PER_WALL
) -> int:
    """The budget at which no more than `max_share` of the surface is thinner than the cell: the cell must fit the thickness at that quantile."""
    thickness = np.asarray(thickness, dtype=np.float64)
    finite = thickness[np.isfinite(thickness)]
    if finite.size == 0 or area <= 0:
        return 0
    wall = float(np.quantile(finite, max_share))
    if wall <= 0:
        return 0
    cell = wall / cells
    return int(np.ceil(area / (cell * cell)))


def judge(thickness: np.ndarray, area: float, quads: int) -> ThinReport:
    return ThinReport(
        share=thin_share(thickness, area, quads),
        cell=cell_size(area, quads),
        quads_to_keep=quads_to_keep(thickness, area),
    )
