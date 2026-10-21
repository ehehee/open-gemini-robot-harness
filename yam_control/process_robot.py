"""RealYamRobot with i2rt running in its OWN process.

Why: i2rt's motor control loop is a Python thread that must send every motor
a command every few ms; the DM motors disable themselves ("loss
communication", error 0xD) if commands stop for longer than their CAN
watchdog. Anything in the same process that holds the GIL for a while --
importing torch, creating the TRI-Stereo ONNX session, opening the ZED,
serializing a big point cloud for viser -- starves that thread. On the real
rig that tripped the right arm at connect 6 times in 7, and once mid-start
even with the cameras opened after the connect; with nothing else in the
process it connected 3/3 clean.

So the child process owns the CAN bus and i2rt and does nothing else; the
parent keeps its own copy of the kinematic model for FK/IK (pure math, no
hardware) and forwards only the hardware calls. Other properties:

  * The child ignores SIGINT, so Ctrl+C in the parent still goes through the
    parent's normal shutdown (which releases the arm via disconnect()).
  * If the parent dies without disconnecting, the child notices within
    0.5 s and releases the arm itself -- it never stays powered unattended.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
import threading
import time
from multiprocessing.managers import BaseManager

import numpy as np

from yam_control.yam_backend import RealYamRobot

_REMOTE_METHODS = [
    "connect", "disconnect", "model_xml_path", "is_healthy", "get_joint_positions", "get_gripper", "get_state",
    "commanded_joint_positions", "commanded_gripper", "stream_joint_path", "move_to_joint_positions",
    "set_gripper", "close_gripper_until_contact",
]


class _ChildRobot(RealYamRobot):
    """Runs inside the child. Releases the arm if the parent goes away."""

    def connect(self) -> None:
        super().connect()
        parent = os.getppid()

        def orphan_guard() -> None:
            while self._robot is not None:
                time.sleep(0.5)
                if os.getppid() != parent:
                    print("[robot process] parent died -- releasing the arm", flush=True)
                    self.disconnect()
                    os._exit(0)

        threading.Thread(target=orphan_guard, daemon=True).start()

    def model_xml_path(self) -> str:
        return self._robot.xml_path


class _RobotManager(BaseManager):
    pass


_RobotManager.register("Robot", _ChildRobot, exposed=_REMOTE_METHODS)


def _child_init() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class ProcessYamRobot(RealYamRobot):
    """Drop-in RealYamRobot: same constructor and API. FK/IK/limits run here
    on a local model; reading/commanding the arm goes to the child."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ctor = (args, kwargs)
        self._mgr = None
        self._remote = None

    def connect(self) -> None:
        self._mgr = _RobotManager(ctx=mp.get_context("spawn"))
        self._mgr.start(initializer=_child_init)
        args, kwargs = self._ctor
        self._remote = self._mgr.Robot(*args, **kwargs)
        self._remote.connect()
        self._robot = self._remote  # marks "connected" for _require_connected
        self._load_fk_model(self._remote.model_xml_path())

    def disconnect(self) -> None:
        remote, mgr = self._remote, self._mgr
        self._remote = self._mgr = self._robot = None
        try:
            if remote is not None:
                remote.disconnect()
        finally:
            if mgr is not None:
                mgr.shutdown()

    # -- hardware calls, forwarded to the child -----------------------------------------
    def is_healthy(self) -> tuple[bool, str]:
        if self._remote is None:
            return False, "not connected"
        try:
            return tuple(self._remote.is_healthy())
        except Exception as e:  # noqa: BLE001 -- the child itself is gone
            return False, f"robot process unreachable: {e}"

    def get_joint_positions(self) -> list[float]:
        self._require_connected()
        return list(self._remote.get_joint_positions())

    def get_gripper(self) -> float:
        self._require_connected()
        return float(self._remote.get_gripper())

    def get_state(self) -> dict[str, list[float]]:
        self._require_connected()
        return dict(self._remote.get_state())

    def commanded_joint_positions(self) -> np.ndarray:
        self._require_connected()
        return np.asarray(self._remote.commanded_joint_positions())

    def commanded_gripper(self) -> float:
        self._require_connected()
        return float(self._remote.commanded_gripper())

    def stream_joint_path(self, q_path, durations, rate_hz: float = 200.0) -> None:
        self._require_connected()
        self._remote.stream_joint_path(np.asarray(q_path), np.asarray(durations), rate_hz)

    def move_to_joint_positions(self, positions, duration_s: float | None = None) -> None:
        self._require_connected()
        self._remote.move_to_joint_positions(list(positions), duration_s)

    def set_gripper(self, open_fraction: float) -> None:
        self._require_connected()
        self._remote.set_gripper(float(open_fraction))

    def close_gripper_until_contact(self, effort_threshold: float = 1.5, max_duration_s: float = 3.0,
                                    poll_hz: float = 20.0) -> bool:
        self._require_connected()
        return bool(self._remote.close_gripper_until_contact(effort_threshold, max_duration_s, poll_hz))
