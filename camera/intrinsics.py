"""Pinhole camera intrinsics: storage and (de)serialization.

Calibration itself already exists for this rig's cameras -- see
camera/rig_camera.py (reads factory/SDK intrinsics at connect time, same as
bosch-crate/yam-bimanual-calib's camera_backend.py) and
camera/rig_calibration.py (loads the solved hand-eye extrinsics). This module
is just the data container the two of them hand around.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: list[float]  # OpenCV order: k1, k2, p1, p2, k3

    def to_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64
        )

    def to_dist_array(self) -> np.ndarray:
        return np.array(self.dist_coeffs, dtype=np.float64)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Intrinsics":
        return cls(**json.loads(Path(path).read_text()))
