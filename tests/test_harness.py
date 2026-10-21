"""Both versions of the harness, the automatic checker and a scripted grasp,
in the RoboSimStudio scene (sim/scenes/scene_7010_v2): physics, rendered
cameras, no hardware, no API key. The /plan programs go straight to
BimanualSession.run_plan, which is what the robot server calls.

    pytest tests/          # needs RoboSimStudio and a CUDA GPU; builds the scene once (~2 min)

Tests run in file order on one scene; the grasp is last because it moves the clamp.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.bimanual import bimanual_prompt  # noqa: E402
from agent.gap_checker import AUTO_CHECK_TEXT  # noqa: E402

needs_sim = pytest.mark.skipif(importlib.util.find_spec("robosimstudio") is None,
                               reason="RoboSimStudio is not installed")


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


# -- in the sim ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def h(tmp_path_factory):
    from sim.procedures import build

    harness = build(run_dir=tmp_path_factory.mktemp("run"), check=True)
    yield harness
    harness.close()


def _last(out: dict) -> str:
    return out["executed"][-1]["result"] if out["executed"] else out["status"]


@needs_sim
def test_overhead_sees_the_table_where_the_scene_puts_it(h):
    # The mat is world z = 0; the right arm's base sits 6 mm below it.
    assert abs(h.session.obs_.table_z - (-h.hub.T_world_base["right"][2, 3])) < 0.003


@needs_sim
def test_unreachable_is_ik_failure(h):
    assert _last(h.plan("right.goto(x=0.95, y=0.0, z=0.10)")) == "rejected by motion checker: IK_FAILURE"


@needs_sim
def test_arm_into_the_other_arm_collides(h):
    assert "rejected" not in _last(h.plan("left.goto(x=0.32, y=0.20, z=0.15)"))
    assert _last(h.plan("right.goto(x=0.32, y=0.20, z=0.15)")) == "rejected by motion checker: COLLISION"
    assert "rejected" not in _last(h.plan("left.home()"))


@needs_sim
def test_fingers_into_the_clamp_body_collide(h):
    # A plain goto (not a grasp, so the fingertips may not touch the cloud) down onto the clamp's frame.
    from sim.procedures import DEFAULT_SCENE, part_axis

    c, _ = part_axis(DEFAULT_SCENE.parent, "clamp", "0000.obj", h.hub.object_pose("clamp"))
    x, y, _ = h.to_base(c)
    assert _last(h.plan(f"right.goto(x={x:.4f}, y={y:.4f}, z=0.012, pitch_deg=90)")) \
        == "rejected by motion checker: COLLISION"


@needs_sim
def test_scripted_grasp_lifts_the_clamp(h):
    from sim.procedures import DEFAULT_SCENE, scripted_grasp

    r = scripted_grasp(h, DEFAULT_SCENE.parent)
    assert r["grasp_status"] == "completed", r
    assert r["holding"] and r["gripper_open"] > 0.1, r  # closed on the sleeve, not on air
    assert r["object_rise_m"] > 0.8 * r["lifted_m"], r  # it came up with the gripper and stayed
