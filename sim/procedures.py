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


# -- handover -----------------------------------------------------------------------------
def _wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def placement(h: SimHarness, scene_dir: Path, obj: str = "clamp", container: str = "tray") -> dict:
    """The scene policy's placement test: every vertex of the object inside
    the container floor's outline, resting on it, and still."""
    import json

    import trimesh
    from scipy.spatial.transform import Rotation
    from shapely.geometry import MultiPoint, Point

    def evaluate():
        sim = h.hub.sim
        floor = trimesh.load(scene_dir / "assets" / container / "0001.obj", process=False)
        interior = MultiPoint(floor.vertices[:, :2]).convex_hull.buffer(-0.001)
        pose, tp = np.asarray(sim[obj].pose), np.asarray(sim[container].pose)
        pts = []
        for part in json.loads((scene_dir / "assets" / obj / "parts.json").read_text()):
            m = trimesh.load(scene_dir / "assets" / obj / part["path"], process=False)
            pts.append(Rotation.from_quat(tp[3:]).inv().apply(Rotation.from_quat(pose[3:]).apply(m.vertices)
                                                               + pose[:3] - tp[:3]))
        pts = np.concatenate(pts)
        inside = float(np.mean([interior.covers(Point(p[:2])) for p in pts]))
        sim.contacts.update(sim.step_count)
        force = float(np.linalg.norm(sim.contacts.pairwise_force(
            sim.scene.handles.entity(obj).collision_shapes, sim.scene.handles.entity(container).collision_shapes)))
        speed = float(np.linalg.norm(sim[obj].velocity[:3]))
        return {"inside_fraction": inside, "contact_force_N": force, "speed_m_s": speed,
                "placed": inside >= 0.999 and force > 0.1 and speed < 0.02}

    return h.hub.call(evaluate)


def scripted_handover(h: SimHarness, scene_dir: Path, say=print) -> dict:
    """Right picks the clamp up by its rubber sleeve, presents it, left
    receives it by the other handle, right lets go (alone, then the receiver
    is rechecked), left carries it over the tray and lets go. Every motion is
    a /plan program through the harness, checked by gap_guard. Grasp poses
    come from the scene's meshes; the handover protocol is the prompt's."""
    import json

    import trimesh
    from scipy.spatial.transform import Rotation

    log: list[dict] = []

    def plan(program: str, why: str) -> dict:
        say(f"{why}\n/plan  {program}")
        out = h.plan(program)
        log.append({"why": why, "program": program, "status": out["status"],
                    "results": [e["result"] for e in out["executed"]]})
        return out

    def held(side):
        return h.session.states[side].holding

    def fail(stage):
        return {"success": False, "stage": stage, "log": log}

    # 1. pick up
    for program in grasp_programs(h, scene_dir, "clamp", "0008.obj", 0.017):
        out = plan(program, "1. Right arm grasps the clamp's rubber sleeve (pose from the mesh)")
        if out["status"] not in ("rejected_by_operator", "plan_error"):
            break
    if out["status"] != "completed" or not held("right"):
        return fail("pick up")
    meet = h.to_base([0.04, -0.065, 0.14])  # where the scene's own policy presents it
    if plan(f"right.goto(x={meet[0]:.4f}, y={meet[1]:.4f}, z={meet[2]:.4f}, speed=0.05)",
            "2. Right arm presents the clamp between the arms")["status"] != "completed":
        return fail("present")

    # 2. receive by the other handle: horizontal approach along it, jaws closing vertically
    cw, tw = part_axis(scene_dir, "clamp", "0018.obj", h.hub.object_pose("clamp"))
    c, a = h.to_base(cw), h.T_base_world[:3, :3] @ tw
    if a[1] > 0:
        a = -a  # approach toward -y: the left arm comes from its own side
    a[2] = 0
    a /= np.linalg.norm(a)
    az = math.degrees(math.atan2(a[1], a[0]))
    site = c + PAD_CENTRE_ABOVE_SITE_M * a
    pre = site - 0.06 * a
    o = f"pitch_deg=0, azimuth_deg={az:.1f}, roll_deg=90"
    if plan(f"left.goto(x={pre[0]:.4f}, y={pre[1]:.4f}, z={pre[2]:.4f}, {o}, speed=0.06)",
            "3. Left arm lines up with the free handle")["status"] != "completed":
        return fail("receive approach")
    if plan(f"left.goto(x={site[0]:.4f}, y={site[1]:.4f}, z={site[2]:.4f}, {o}, speed=0.03)",
            "4. Left arm slides onto the handle")["status"] != "completed":
        return fail("receive")
    if plan("left.close_gripper()", "5. Left gripper closes on the handle")["status"] != "completed" or not held("left"):
        return fail("receive close")

    # 3. transfer: the giver opens alone, then the receiver must still hold
    z0 = float(h.hub.object_pose("clamp")[2])
    plan("right.open_gripper()", "6. Right gripper lets go -- alone, so the result can be checked")
    h.hub.sleep(1.0)
    h.session.observe()
    drop = z0 - float(h.hub.object_pose("clamp")[2])
    say(f"7. Recheck: left holding={held('left')}, right holding={held('right')}, clamp moved {drop * 1000:.1f} mm")
    if not held("left") or held("right"):
        return fail("transfer")
    transfer = {"left_holding": True, "clamp_drop_on_release_m": drop}
    back = h.to_base([0.04, -0.165, 0.18])
    plan(f"right.goto(x={back[0]:.4f}, y={back[1]:.4f}, z={back[2]:.4f}, speed=0.06)", "8. Right arm backs away")

    # 4. place: clamp long axis along the tray's, its bounds centred over the tray
    root = scene_dir / "assets" / "clamp"
    V = np.vstack([trimesh.load(root / p["path"], process=False).vertices
                   for p in json.loads((root / "parts.json").read_text())])
    mid_local = (V.min(0) + V.max(0)) / 2
    cp, tp = h.hub.object_pose("clamp"), h.hub.object_pose("tray")
    yaw_c = math.degrees(Rotation.from_quat(cp[3:]).as_euler("xyz")[2])
    yaw_t = math.degrees(Rotation.from_quat(tp[3:]).as_euler("xyz")[2])  # tray floor's long side is its local x
    tcp = h.arms["left"].fk(h.session.states["left"].q_measured)[:3, 3]
    mid = h.to_base(cp[:3] + Rotation.from_quat(cp[3:]).apply(mid_local))
    tray = h.to_base(tp[:3])
    placed_out = None
    # Turn the wrist (about the vertical through the pinch) to line the clamp
    # up with the tray, but never by more than 90 deg: a 118 deg turn swung the
    # cantilevered clamp out of the jaws. Then the smaller turns that reach.
    aligned = _wrap180(yaw_t - yaw_c)
    aligned = aligned - 180.0 if aligned > 90 else aligned + 180.0 if aligned <= -90 else aligned
    for delta in [aligned, 40.0, 20.0, 0.0, -20.0]:
        d = math.radians(delta)
        Rz = np.array([[math.cos(d), -math.sin(d), 0], [math.sin(d), math.cos(d), 0], [0, 0, 1]])
        for height in (0.06, 0.08, 0.10):
            t = tray + np.array([0, 0, height]) - Rz @ (mid - tcp)
            placed_out = plan(f"left.goto(x={t[0]:.4f}, y={t[1]:.4f}, z={t[2]:.4f}, pitch_deg=0, "
                              f"azimuth_deg={_wrap180(az + delta):.1f}, roll_deg=90, speed=0.04)",
                              f"9. Left arm carries it over the tray ({height * 100:.0f} cm up, turned to fit)")
            if placed_out["status"] == "completed":
                break
        if placed_out["status"] == "completed":
            break
    if placed_out["status"] != "completed" or not held("left"):
        return fail("carry") | {"transfer": transfer}
    # Ground truth: is the clamp still in the hand? (holding reads the jaws, and
    # jaws resting at their close setpoint look the same with or without it.)
    tcp = h.arms["left"].fk(h.session.states["left"].q_measured)[:3, 3]
    gap = float(np.linalg.norm(h.to_base(h.hub.object_pose("clamp")[:3]) - tcp))
    say(f"9. Carried: clamp origin {gap * 100:.1f} cm from the left pinch point")
    if gap > 0.12:
        return fail("carry: dropped") | {"transfer": transfer}
    # The jaws close vertically in this grip, so opening swings the lower finger
    # down: if gap_guard sees it hitting the tray or table, rise a little first.
    for k in range(4):
        out = plan("left.open_gripper()", "10. Left gripper releases the clamp into the tray")
        if out["status"] == "completed":
            break
        plan("left.move_by(dz=0.02, speed=0.03)", "10. Release would hit the tray: rise 2 cm first")
    if held("left"):
        return fail("release") | {"transfer": transfer}
    h.hub.sleep(1.5)
    if plan("left.move_by(dz=0.06, speed=0.05)", "11. Left arm lifts clear")["status"] != "completed":
        plan("left.home()", "11. Left arm returns home")
    h.hub.sleep(2.0)
    h.session.observe()
    place = placement(h, scene_dir)
    say(f"12. Placement: {place['inside_fraction'] * 100:.1f}% of the clamp inside the tray, "
        f"{place['contact_force_N']:.2f} N on it, {'PLACED' if place['placed'] else 'NOT fully placed'}")
    return {"success": bool(place["placed"]), "stage": "complete", "transfer": transfer, "placement": place,
            "log": log}
