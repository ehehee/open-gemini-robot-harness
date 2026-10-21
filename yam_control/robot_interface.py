"""Common interface both the mock and real YAM backends implement.

Cartesian deltas are expressed in the arm base frame: +x forward, +y left, +z up.
`get_ee_pose` returns the full 6-DoF pose (not just position) since hand-eye
calibration needs orientation too.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from camera.extrinsics import Extrinsics


@dataclass(frozen=True)
class CartesianPlan:
    """A solved-but-not-yet-executed move: everything a caller needs to show
    the operator where the arm *would* go before any motor is commanded.

    `target_pose` is the FK of `target_joints`, i.e. the pose the arm will
    actually reach, not the one that was requested -- IK converges to a
    tolerance, so compare the two if you care about the residual.
    """

    # Joint angles here are ENCODER space -- what get_joint_positions() reports
    # and what command_joint_pos() takes. Anything feeding MuJoCo (rendering,
    # FK) wants model space instead; use the *_joints_model properties.
    start_joints: np.ndarray  # (6,) rad, encoder
    target_joints: np.ndarray  # (6,) rad, encoder
    start_pose: Extrinsics  # T_base_gripper now
    target_pose: Extrinsics  # T_base_gripper after the move
    requested_xyz: np.ndarray  # (3,) the position IK was asked for
    joint_offsets: np.ndarray  # (6,) rad, q_model = q_encoder + offset

    @property
    def joint_delta(self) -> np.ndarray:
        """Unaffected by the offsets -- they cancel in a difference."""
        return self.target_joints - self.start_joints

    @property
    def start_joints_model(self) -> np.ndarray:
        return self.start_joints + self.joint_offsets

    @property
    def target_joints_model(self) -> np.ndarray:
        return self.target_joints + self.joint_offsets

    @property
    def position_error_m(self) -> float:
        """How far the IK solution's actual position lands from what we asked."""
        return float(np.linalg.norm(self.target_pose.t - self.requested_xyz))


class YamRobotInterface(ABC):
    @abstractmethod
    def connect(self) -> None:
        ...

    @abstractmethod
    def disconnect(self) -> None:
        ...

    @abstractmethod
    def get_ee_pose(self) -> Extrinsics:
        """Return the current end-effector pose as T_base_gripper."""

    @abstractmethod
    def get_joint_positions(self) -> list[float]:
        ...

    @abstractmethod
    def move_to_joint_positions(self, positions: list[float], duration_s: float | None = None) -> None:
        """Move to an absolute joint configuration (radians). Used directly by
        calibration pose sweeps, since it doesn't depend on IK."""

    @abstractmethod
    def move_cartesian_delta(self, dx: float, dy: float, dz: float) -> None:
        """Move the end-effector by the given delta, in meters, in the base frame."""

    @abstractmethod
    def set_gripper(self, open_fraction: float) -> None:
        """0.0 = fully closed, 1.0 = fully open."""

    @abstractmethod
    def home(self) -> None:
        ...

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()
