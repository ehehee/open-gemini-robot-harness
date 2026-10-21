"""The contract every motion verifier implements: one verdict, nothing more.

This replaces the human looking at a viser ghost before pressing EXECUTE. The
answer is deliberately one of exactly three words. There are no reason
strings, distances or contact pairs: whatever a backend learns about *why* a
motion failed goes to a logging.debug line for whoever is developing the
backend, and never into the result, so it can never reach the VLM.

Precedence is fixed: IK is checked first, and a motion with no valid joint
solution is IK_FAILURE without ever being collision-checked.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from camera.extrinsics import Extrinsics


class Verdict(str, Enum):
    OK = "OK"
    IK_FAILURE = "IK_FAILURE"
    COLLISION = "COLLISION"


@dataclass(frozen=True)
class VerifiedMotion:
    verdict: Verdict
    # (6,) ENCODER-space joints -- what move_to_joint_positions() takes. Set
    # only when verdict is OK, and it is exactly the endpoint of the segment
    # that was checked, so the caller can only execute what was verified.
    target_joints: np.ndarray | None = None


class MotionVerifier(Protocol):
    """Joint angles in and out are ENCODER space (what get_joint_positions()
    reports); converting to model space is the verifier's job.

    Both methods verify the straight joint-space segment from start_q to the
    target, i.e. the path RealYamRobot.move_to_joint_positions() interpolates
    along -- not a Cartesian straight line, which is not what the arm drives.
    """

    def verify_pose(self, start_q: np.ndarray, target: Extrinsics) -> VerifiedMotion:
        """target is T_base_grasp_site, same frame as RealYamRobot.get_ee_pose()."""

    def verify_joints(self, start_q: np.ndarray, target_q: np.ndarray) -> VerifiedMotion:
        """For moves that are already in joint space (home)."""

    def verify_gripper(self, start_q: np.ndarray, open_from: float | None, open_to: float) -> VerifiedMotion:
        """Opening/closing the gripper (0 closed .. 1 open, get_gripper()'s
        convention) with the arm still at start_q. OK or COLLISION only."""
