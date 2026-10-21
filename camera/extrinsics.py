"""A rigid 4x4 transform between two frames, with (de)serialization."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


class Extrinsics:
    """Homogeneous transform T such that p_dst = T @ p_src (p as [x,y,z,1])."""

    def __init__(self, matrix: np.ndarray):
        assert matrix.shape == (4, 4)
        self.matrix = matrix.astype(np.float64)

    @classmethod
    def identity(cls) -> "Extrinsics":
        return cls(np.eye(4))

    @classmethod
    def from_R_t(cls, R: np.ndarray, t: np.ndarray) -> "Extrinsics":
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(t).flatten()
        return cls(T)

    @property
    def R(self) -> np.ndarray:
        return self.matrix[:3, :3]

    @property
    def t(self) -> np.ndarray:
        return self.matrix[:3, 3]

    def inverse(self) -> "Extrinsics":
        R = self.R
        t = self.t
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        return Extrinsics(T_inv)

    def compose(self, other: "Extrinsics") -> "Extrinsics":
        """self.compose(other) == apply `other` first, then `self`."""
        return Extrinsics(self.matrix @ other.matrix)

    def apply(self, point_xyz: np.ndarray) -> np.ndarray:
        p = np.asarray(point_xyz, dtype=np.float64)
        homog = np.append(p, 1.0)
        return (self.matrix @ homog)[:3]

    @classmethod
    def load_yaml(cls, path: Path) -> "Extrinsics":
        """Loads the `T_*.yaml` format solve_handeye.py/solve_overhead_handeye.py
        write in yam-bimanual-calib's extrinsics/ dir: a `matrix` key plus
        assorted metadata we don't need here."""
        data = yaml.safe_load(Path(path).read_text())
        return cls(np.array(data["matrix"]))

    def __repr__(self) -> str:
        return f"Extrinsics(\n{self.matrix}\n)"
