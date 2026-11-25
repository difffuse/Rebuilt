"""The local ray distance field, and the cage that encodes it."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..analysis.mesh_graph import edge_topology
from ..types import MeshArrays


@dataclass(frozen=True)
class CageField:
    """The offsets a cage is built from, and what they cost to obtain."""

    offsets: np.ndarray
    floor: float
    minimum: float = 0.0
    maximum: float = 0.0
    mean: float = 0.0

    capped: int = 0

    smoothed: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("offsets")
        return payload


def cage_offsets(
    mesh: MeshArrays,
    reach: np.ndarray,
    scale: np.ndarray,
    diagonal: float,
    safety: float = 1.25,
    floor_fraction: float = 0.004,
    cap_multiple: float = 1.5,
    smoothing_passes: int = 4,
) -> CageField:
    """Turn measured distances into a cage displacement per vertex."""
    if reach.shape[0] != mesh.num_verts or scale.shape[0] != mesh.num_verts:
        raise ValueError("reach and scale must have one value per vertex")

    floor = max(diagonal, 0.0) * floor_fraction
    if mesh.num_verts == 0:
        return CageField(np.zeros(0), floor)

    wanted = np.maximum(reach.astype(np.float64), 0.0) * safety

    cap = np.maximum(scale.astype(np.float64) * cap_multiple, floor)
    capped = wanted > cap
    offsets = np.maximum(np.minimum(wanted, cap), floor)

    smoothed = _limit_variation(mesh, offsets, smoothing_passes)
    changed = int((np.abs(smoothed - offsets) > 1e-12).sum())

    return CageField(
        offsets=smoothed,
        floor=float(floor),
        minimum=float(smoothed.min()),
        maximum=float(smoothed.max()),
        mean=float(smoothed.mean()),
        capped=int(capped.sum()),
        smoothed=changed,
    )


def _limit_variation(mesh: MeshArrays, offsets: np.ndarray, passes: int) -> np.ndarray:
    """Stop the offset changing faster along an edge than the edge is long."""
    topology = edge_topology(mesh)
    edges = topology.edges
    if edges.shape[0] == 0 or passes <= 0:
        return offsets.copy()

    positions = mesh.positions.astype(np.float64)
    lengths = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)

    result = offsets.copy()
    for _ in range(passes):
        left, right = result[edges[:, 0]], result[edges[:, 1]]
        allowed_left = right - lengths
        allowed_right = left - lengths

        raised = np.zeros_like(result)
        np.maximum.at(raised, edges[:, 0], allowed_left)
        np.maximum.at(raised, edges[:, 1], allowed_right)

        updated = np.maximum(result, raised)
        if np.array_equal(updated, result):
            break
        result = updated

    return result


def cage_positions(mesh: MeshArrays, normals: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Where the cage's vertices go. Shape (V, 3)."""
    if normals.shape != (mesh.num_verts, 3):
        raise ValueError(f"expected {(mesh.num_verts, 3)} normals, got {normals.shape}")
    if offsets.shape[0] != mesh.num_verts:
        raise ValueError("offsets must have one value per vertex")

    return mesh.positions.astype(np.float64) + normals * offsets[:, None]


def envelope(offsets: np.ndarray, reach: np.ndarray, slack: float = 1.5) -> np.ndarray:
    """The furthest a ray may legitimately travel from each vertex."""
    return (np.maximum(offsets, 0.0) + np.maximum(reach, 0.0)) * slack
