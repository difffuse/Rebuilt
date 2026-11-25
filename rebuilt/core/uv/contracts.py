"""Owned native seam results; no Python engine or topology reconstruction."""
from dataclasses import dataclass, field

import numpy as np


def _indices():
    return np.empty(0, dtype=np.int64)


@dataclass
class NativeSeamResult:
    """UVs/corner labels are in host-loop order; faces retain input order.

    Edge endpoints and core-to-host edge indices retain native ordering. Seams
    use host edge order. Every array owns its storage and survives Session.close.
    Chart numbers are contiguous and local to this result. Reports are separate
    trees: native_report is untouched, report adds application runtime metadata.
    This record is not permission to apply to subsequently edited geometry.
    """

    accepted: bool = False
    reason: str = ""
    uv: np.ndarray | None = None
    seams: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    chart_of_face: np.ndarray = field(default_factory=_indices)
    chart_of_corner: np.ndarray = field(default_factory=_indices)
    exempt: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    invalid_charts: np.ndarray = field(default_factory=_indices)
    changed_faces: np.ndarray = field(default_factory=_indices)
    edge_vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64))
    host_loop_indices: np.ndarray = field(default_factory=_indices)
    core_edge_to_host: np.ndarray = field(default_factory=_indices)
    geometry_fingerprint: str = ""
    backend: str = ""
    uv_valid: bool = False
    topology_valid: bool = False
    budget_satisfied: bool = False
    partial: bool = False
    cancelled: bool = False
    timed_out: bool = False
    native_report: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    build_info: dict = field(default_factory=dict)

    _application_session: object = field(default=None, repr=False, compare=False)

    def release_application(self):
        """Release the already closed native result snapshot after apply/refusal."""
        self._application_session = None

    def authorize(self, current_fingerprint):
        if self._application_session is None:
            raise ValueError('No validated native UV proposal is available; run Seams again.')
        result = self._application_session.result_for_apply(current_fingerprint)
        if not (result['accepted'] and result['uv_valid']):
            raise ValueError('The final native UV proposal was not accepted.')
        return result

    @property
    def cut_edges(self):
        """Return native core endpoint pairs selected by the host-order seam mask."""
        mapped = self.core_edge_to_host >= 0
        cuts = np.zeros(len(self.core_edge_to_host), dtype=bool)
        cuts[mapped] = self.seams[self.core_edge_to_host[mapped]]
        return self.edge_vertices[cuts].copy()

    @property
    def n_charts(self) -> int:
        return int(self.chart_of_face.max(initial=-1)) + 1
