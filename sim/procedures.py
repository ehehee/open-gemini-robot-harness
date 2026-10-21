"""The harness in the RoboSimStudio sim without Gemini or viser, and a
scripted grasp computed from the object's geometry. Used by
tools/scripted_grasp.py and tests/test_harness.py.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO / "sim" / "scenes" / "scene_7010_v2" / "scene.py"
# i2rt's grasp_site (the harness's pinch point) is 7 mm further along the
# approach axis than the pad centre: put the pads on the part's centre line.
PAD_CENTRE_ABOVE_SITE_M = 0.007
GRASP_LIFT_M = 0.08  # grasp()'s default approach_height: it ends by lifting back up that far


class _Field:
    def __init__(self, value=None):
        self.value, self.content, self.image, self.visible = value, "", None, False


class HeadlessViewer:
    """BimanualViewer's interface without viser: gap_guard decides, or every
    step is approved when there is no checker."""

    def __init__(self, checker=None):
        self.checker, self.stop_requested = checker, False
        self.reviewer = "motion checker" if checker else "operator"
        self.auto, self.pause = _Field(False), _Field(False)
        self.image, self.wrist, self.status_md = _Field(), _Field(), _Field()

    def wait_decision(self, step):
        return self.checker.decide(step) if self.checker else ("approve", "")

    def __getattr__(self, name):  # show_steps, set_now_arm, set_cloud, ...
        return lambda *a, **k: None


@dataclass
class SimHarness:
    hub: object
    stop: object
    session: object
    arms: dict
    run_dir: Path
    T_base_world: np.ndarray = field(default=None)  # world -> right arm base (the harness's frame)

    def plan(self, program: str, verbose: bool = True) -> dict:
        """POST /plan, as the robot server runs it: the plan, then a fresh observation."""
        out = self.session.run_plan(program)
        self.session.observe()
        if verbose:
            print(f"[plan] {program.strip()}  -> {out['status']}", flush=True)
            for e in out["executed"]:
                print(f"   {e['step']:<48} {e['result']}", flush=True)
        return out

    def to_base(self, p_world) -> np.ndarray:
        T = self.T_base_world
        return T[:3, :3] @ np.asarray(p_world, float) + T[:3, 3]

    def close(self) -> None:
        self.stop.set()


def build(scene: Path = DEFAULT_SCENE, run_dir: Path | None = None, check: bool = True) -> SimHarness:
    """Build the scene and the harness on it, observe once, and return."""
    from scipy.spatial.transform import Rotation  # noqa: F401  (import before the sim thread swaps the clocks)

    from camera.rig_calibration import config_dir
    from sim import rss_backend

    run_dir = run_dir or REPO / "runs" / ("sim_" + time.strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[sim] building {scene} ...", flush=True)
    hub, stop = rss_backend.start(scene, config_dir(), run_dir / "sim_config")
    os.environ["RIG_CONFIG_DIR"] = str(run_dir / "sim_config")
    print(f"[sim] ready in {time.time() - t0:.0f} s", flush=True)

    from agent.bimanual import ARMS, BimanualSession
    from agent.observer import Observer, Workspace
    from agent.primitives import Primitives
    from camera.rig_calibration import load_extrinsic

    if not hasattr(Primitives.execute, "__wrapped__"):
        Primitives.execute = rss_backend.sim_clocked(Primitives.execute)
    SimYamRobot = rss_backend.sim_robot_class()
    xml = str(rss_backend.i2rt_arm_xml(run_dir / "i2rt_arm.xml"))
    robots = {a: SimYamRobot(hub, a, xml) for a in ARMS}
    for r in robots.values():
        r.connect()
    T_world_left = load_extrinsic("T_leftbase_rightbase").inverse().matrix
    observer = Observer(load_extrinsic("T_rightbase_overhead"),
                        Workspace(x=(0.12, 0.60), y=(-0.35, 0.75), z=(-0.20, 0.45)))
    arms = {a: Primitives(robots[a], np.array(robots[a].get_joint_positions()),
                          Workspace(x=(0.12, 0.60), y=(-0.35, 0.35), z=(-0.20, 0.45)),
                          table_z_fn=lambda: observer.table_z, max_joint_speed=0.6,
                          T_world_base=None if a == "right" else T_world_left) for a in ARMS}
    checker = None
    if check:
        from agent.gap_checker import GapChecker

        checker = GapChecker(arms, T_world_left, run_dir, gripper_parts=hub.gripper_parts(run_dir / "sim_parts"))
    session = BimanualSession(arms, observer, HeadlessViewer(checker), rss_backend.SimOverhead(hub).grab, run_dir)
    if checker:
        checker.session = session
    session.observe()
    for p in arms.values():
        p.ws.z = (observer.table_z - 0.005, 0.45)
    print(f"[obs] table_z = {observer.table_z:+.4f} m in the right-arm base frame "
          f"(scene mat: {-hub.T_world_base['right'][2, 3]:+.4f})", flush=True)
    return SimHarness(hub, stop, session, arms, run_dir, np.linalg.inv(hub.T_world_base["right"]))


def part_axis(scene_dir: Path, obj: str, part: str, pose7: np.ndarray, inward_m: float = 0.0):
    """(centre, horizontal long axis) of one mesh part of a scene object, in
    the world: bounds centre and principal axis (as the scene's own policy
    computes them), moved inward_m toward the object's origin along the axis."""
    import trimesh
    from scipy.spatial.transform import Rotation

    mesh = trimesh.load(scene_dir / "assets" / obj / part, process=False)
    centre = mesh.bounds.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov(mesh.vertices.T))
    t = vectors[:, -1].copy()
    t[2] = 0
    t /= np.linalg.norm(t)
    centre = centre - t * np.sign(np.dot(centre, t)) * inward_m
    R = Rotation.from_quat(pose7[3:7])
    return pose7[:3] + R.apply(centre), R.apply(t)


def grasp_programs(h: SimHarness, scene_dir: Path, obj: str, part: str, inward_m: float) -> list[str]:
    """Top-down right.grasp programs on the part's centre, jaws closing across
    it: the same jaw line with the wrist either way round."""
    centre_w, axis_w = part_axis(scene_dir, obj, part, h.hub.object_pose(obj), inward_m)
    c, a = h.to_base(centre_w), h.T_base_world[:3, :3] @ axis_w
    jaw = math.degrees(math.atan2(a[1], a[0])) + 90.0
    z = c[2] - PAD_CENTRE_ABOVE_SITE_M
    print(f"[grasp] {obj}/{part}: centre ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}) in the right-arm base frame, "
          f"long axis {math.degrees(math.atan2(a[1], a[0])):.1f} deg -> pinch z {z:.4f}, jaws at {jaw:.1f} deg",
          flush=True)
    yaws = [((jaw + 180.0 * k + 180.0) % 360.0) - 180.0 for k in (0, 1)]
    return [f"right.grasp(x={c[0]:.4f}, y={c[1]:.4f}, z={z:.4f}, yaw_deg={y:.1f})" for y in yaws]


def scripted_grasp(h: SimHarness, scene_dir: Path, obj: str = "clamp", part: str = "0008.obj",
                   inward_m: float = 0.017, extra_lift_m: float = 0.04, hold_s: float = 2.0) -> dict:
    """Grasp, lift and hold through the harness; did the object come up with
    the gripper and stay in the jaws?"""
    p0 = h.hub.object_pose(obj)
    out, program = None, None
    for program in grasp_programs(h, scene_dir, obj, part, inward_m):
        out = h.plan(program)
        if out["status"] not in ("rejected_by_operator", "plan_error"):
            break
        print("   -> rejected; same jaw line with the wrist turned 180 deg", flush=True)
    st = h.session.states["right"]
    result = {"object": obj, "part": part, "program": program, "grasp_status": out["status"],
              "grasp_steps": out["executed"], "rise_after_grasp_m": float(h.hub.object_pose(obj)[2] - p0[2])}
    lifted = GRASP_LIFT_M
    if out["status"] == "completed" and st.holding and extra_lift_m > 0:
        lift = h.plan(f"right.move_by(dz={extra_lift_m:.3f}, speed=0.04)")
        result["lift_status"] = lift["status"]
        if lift["status"] == "completed":
            lifted += extra_lift_m
    h.hub.sleep(hold_s)  # sim seconds: does it stay in the jaws?
    h.session.observe()
    st = h.session.states["right"]
    rise = float(h.hub.object_pose(obj)[2] - p0[2])
    result.update(holding=st.holding, gripper_open=st.gripper, lifted_m=lifted, object_rise_m=rise, hold_s=hold_s,
                  success=bool(out["status"] == "completed" and st.holding and rise > 0.8 * lifted))
    return result
