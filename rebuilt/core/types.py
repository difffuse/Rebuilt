"""Core data types. Plain dataclasses over numpy arrays — no bpy anywhere."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class MeshArrays:
    """A mesh as flat numpy arrays."""

    positions: np.ndarray  # (V, 3)
    corner_verts: np.ndarray  # (C,), indices into positions
    face_starts: np.ndarray  # (F,), offsets into corner_verts
    face_sizes: np.ndarray  # (F,)

    _analysis_cache: tuple | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def num_verts(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.face_sizes.shape[0])

    @property
    def num_corners(self) -> int:
        return int(self.corner_verts.shape[0])

    def face_corners(self, face: int) -> np.ndarray:
        """Vertex indices of one face, in winding order."""
        start = int(self.face_starts[face])
        return self.corner_verts[start : start + int(self.face_sizes[face])]

    def validate(self) -> None:
        """Fail loudly on malformed input."""
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(f"positions must be (V, 3), got {self.positions.shape}")
        for name in ("corner_verts", "face_starts", "face_sizes"):
            array = getattr(self, name)
            if array.ndim != 1:
                raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
        if self.face_starts.shape != self.face_sizes.shape:
            raise ValueError(
                f"face_starts {self.face_starts.shape} and "
                f"face_sizes {self.face_sizes.shape} must match"
            )
        if self.num_faces and int(self.face_sizes.min()) < 3:
            raise ValueError("every face needs at least 3 corners")
        expected_corners = int(self.face_sizes.sum())
        if expected_corners != self.num_corners:
            raise ValueError(
                f"face_sizes sum to {expected_corners} corners "
                f"but corner_verts has {self.num_corners}"
            )
        if self.num_corners:
            if int(self.corner_verts.max()) >= self.num_verts:
                raise ValueError("corner_verts references a vertex past the end of positions")
            if int(self.corner_verts.min()) < 0:
                raise ValueError("corner_verts contains a negative index")

    def copy(self) -> MeshArrays:
        return MeshArrays(
            positions=self.positions.copy(),
            corner_verts=self.corner_verts.copy(),
            face_starts=self.face_starts.copy(),
            face_sizes=self.face_sizes.copy(),
        )
