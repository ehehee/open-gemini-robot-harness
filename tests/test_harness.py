"""Both versions of the harness against i2rt's MuJoCo sim: no hardware, no
API key. The /plan programs go straight to BimanualSession.run_plan, which is
what the robot server calls.

    pytest tests/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.bimanual import ARMS, BimanualSession, bimanual_prompt  # noqa: E402
from agent.gap_checker import AUTO_CHECK_TEXT, GapChecker  # noqa: E402
from agent.observer import Observer, Workspace  # noqa: E402
from agent.primitives import Primitives  # noqa: E402
from camera.rig_calibration import load_extrinsic, load_overhead_intrinsics  # noqa: E402
from camera.synthetic_overhead import synthetic_overhead  # noqa: E402

BLOCK = (0.38, 0.05)  # synthetic block centre, right-arm base frame; 5 cm cube


# -- prompt ------------------------------------------------------------------------
def test_prompts_differ_only_in_who_checks():
    human, auto = bimanual_prompt("human"), bimanual_prompt("auto")
    for old, new in AUTO_CHECK_TEXT.items():
        assert old in human and old not in auto and new in auto
    for old, new in AUTO_CHECK_TEXT.items():
        auto = auto.replace(new, old)
    assert auto == human
    with pytest.raises(ValueError):
        bimanual_prompt("nobody")


# -- synthetic overhead ------------------------------------------------------------------
def test_synthetic_overhead_deprojects_onto_the_table():
    T, K = load_extrinsic("T_rightbase_overhead"), load_overhead_intrinsics()
    obs = Observer(T, Workspace(x=(0.12, 0.60), y=(-0.35, 0.75), z=(-0.20, 0.45)))
    obs.set_frame(synthetic_overhead(T, K))
    assert abs(obs.table_z) < 0.005
    obs.render([])  # u, v are fractions of the last image Gemini was sent
    # The block's top centre, projected into the image and back.
    p = T.inverse().apply(np.array([BLOCK[0], BLOCK[1], 0.05]))
    u, v = (p[0] / p[2] * K.fx + K.cx) / K.width, (p[1] / p[2] * K.fy + K.cy) / K.height
    x, y = obs.deproject(u=u, v=v, z_plane=0.05)
    assert np.hypot(x - BLOCK[0], y - BLOCK[1]) < 0.01


# -- autonomous checker ----------------------------------------------------------------------
class _Field:
    def __init__(self, value=None):
        self.value, self.content, self.image, self.visible = value, "", None, False


class HeadlessViewer:
    """BimanualViewer's interface without viser; wait_decision is the checker."""

    reviewer = "motion checker"

    def __init__(self, checker):
        self.checker, self.stop_requested = checker, False
        self.auto, self.pause = _Field(False), _Field(False)
        self.image, self.wrist, self.status_md = _Field(), _Field(), _Field()

    def wait_decision(self, step):
        return self.checker.decide(step)

    def __getattr__(self, name):  # show_steps, set_now_arm, set_cloud, ...
        return lambda *a, **k: None


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    from yam_control.yam_backend import RealYamRobot

    run_dir = tmp_path_factory.mktemp("run")
    robots = {a: RealYamRobot(channel="can0", gripper_type="crank_4310", sim=True) for a in ARMS}
    for r in robots.values():
        r.connect()
    T_world_left = load_extrinsic("T_leftbase_rightbase").inverse().matrix
    T, K = load_extrinsic("T_rightbase_overhead"), load_overhead_intrinsics()
    observer = Observer(T, Workspace(x=(0.12, 0.60), y=(-0.35, 0.75), z=(-0.20, 0.45)))
    frame = synthetic_overhead(T, K)
    arms = {a: Primitives(robots[a], np.array(robots[a].get_joint_positions()),
                          Workspace(x=(0.12, 0.60), y=(-0.35, 0.35), z=(-0.20, 0.45)),
                          table_z_fn=lambda: observer.table_z, max_joint_speed=1.5,
                          T_world_base=None if a == "right" else T_world_left) for a in ARMS}
    checker = GapChecker(arms, T_world_left, run_dir)
    s = BimanualSession(arms, observer, HeadlessViewer(checker), lambda: frame, run_dir)
    checker.session = s
    s.observe()
    yield s
    for r in robots.values():
        r.disconnect()


def _run(session, code: str) -> str:
    out = session.run_plan(code)
    return out["executed"][-1]["result"] if out["executed"] else out["status"]


def test_open_table_is_ok(session):
    assert "rejected" not in _run(session, "right.goto(x=0.30, y=-0.15, z=0.03, pitch_deg=90)")
    _run(session, "right.home()")


def test_fingers_into_the_block_collide(session):
    assert _run(session, f"right.goto(x={BLOCK[0]}, y={BLOCK[1]}, z=0.03, pitch_deg=90)") \
        == "rejected by motion checker: COLLISION"


def test_unreachable_is_ik_failure(session):
    assert _run(session, "right.goto(x=0.95, y=0.0, z=0.10)") == "rejected by motion checker: IK_FAILURE"


def test_arm_into_the_other_arm_collides(session):
    assert "rejected" not in _run(session, "left.goto(x=0.32, y=0.20, z=0.15)")
    assert _run(session, "right.goto(x=0.32, y=0.20, z=0.15)") == "rejected by motion checker: COLLISION"
    _run(session, "left.home()")
