"""The post's Code-as-Policies primitives, on top of RealYamRobot.

Each primitive call expands into small OPS (one straight-line move, one
gripper action, one guarded contact move...). An op is resolved into a
concrete SEGMENT -- IK solved, swept path computed, safety checks run --
only from a known arm state, so the same op is resolved twice:
  * once against a simulated state, chained, to preview the whole /plan
    program before anything moves;
  * again from the arm's ACTUAL state right before it runs, which is the
    segment the operator approves (a contact move that stops early, or a
    grasp that slips, makes everything after it differ from the preview).

Frames/units: arm base frame (+x forward, +y left, +z up), meters, degrees.
The controlled point is grasp_site = the pinch center at the fingertips
(see yam_control/grasp_planner.py for how its axes were read off the model):
site +z = approach direction, site +y = the line the jaws close along.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable

import mujoco
import numpy as np

from viz.viser_robot import set_arm_qpos
from yam_control.yam_backend import RealYamRobot

NAN = float("nan")
LINEAR_STEP_M = 0.01  # straight-line moves: an IK waypoint every 1 cm ...
ANGULAR_STEP_RAD = math.radians(5)  # ... or every 5 deg of rotation
MAX_UNGATED_JOINT_DELTA_RAD = 0.5  # human review: larger joint changes need the override
FINGER_BODIES = ("gripper", "tip_left", "tip_right")


def orientation(jaw_yaw_deg: float, pitch_deg: float, azimuth_deg: float, roll_deg: float) -> np.ndarray:
    """3x3 R_base_site.

    pitch_deg: approach angle below horizontal (90 = straight down).
    azimuth_deg: heading of the approach in the XY plane (0 = +x, 90 = +y);
        irrelevant when pitch is 90.
    jaw_yaw_deg: for top-down (pitch >= 80), the XY direction the jaws close
        along (0 = +x, 90 = +y). To grip a handle lying along direction t,
        use t + 90. Ignored for tilted approaches, where the jaws close
        horizontally across the approach direction.
    roll_deg: extra rotation about the approach axis.
    """
    p, az = math.radians(pitch_deg), math.radians(azimuth_deg)
    a = np.array([math.cos(p) * math.cos(az), math.cos(p) * math.sin(az), -math.sin(p)])
    if pitch_deg >= 80:
        yaw = math.radians(jaw_yaw_deg)
        j = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    else:
        j = np.array([-math.sin(az), math.cos(az), 0.0])
    j = _rotate(j, a, math.radians(roll_deg))
    j = j - (j @ a) * a
    j /= np.linalg.norm(j)
    return np.stack([np.cross(j, a), j, a], axis=1)


def _rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return (v * math.cos(angle) + np.cross(axis, v) * math.sin(angle)
            + axis * (axis @ v) * (1 - math.cos(angle)))


def _rot_about(axis: np.ndarray, angle: float) -> np.ndarray:
    return np.stack([_rotate(e, axis, angle) for e in np.eye(3)], axis=1)


def describe_orientation(R: np.ndarray) -> dict:
    a, j = R[:, 2], R[:, 1]
    return {
        "pitch_deg": round(math.degrees(math.asin(float(np.clip(-a[2], -1, 1)))), 1),
        "azimuth_deg": round(math.degrees(math.atan2(a[1], a[0])), 1),
        "jaw_yaw_deg": round(math.degrees(math.atan2(j[1], j[0])), 1),
    }


# A move that ends further than this from its planned end point was stopped by
# something (another arm, the object, the table) -- not gravity sag, which is
# ~3-30 mm on this rig. 2026-10-15: misses of 100-150 mm were reported as
# "gravity sag" while the blocked left arm cooked its motor 4.
BLOCKED_M = 0.04
# _recheck_holding: jaws open wider than this, or squeezing harder than this, still hold something.
HOLD_MIN_OPEN = 0.02
HOLD_MIN_EFFORT = 0.15


@dataclass
class ArmState:
    q: np.ndarray  # SETPOINT, encoder space -- what plans start from
    gripper: float  # 0 closed .. 1 open (measured)
    q_measured: np.ndarray | None = None  # where the arm actually is (sags below q)
    holding: bool = False
    status: str = "ok"


@dataclass
class Segment:
    """A resolved op: exactly what would be commanded, plus what to check."""

    label: str
    kind: str  # "move" | "contact" | "gripper" | "close" | "observe" | "done"
    call_index: int
    q_path: np.ndarray | None = None  # (N, 6) encoder waypoints, first = start
    durations: np.ndarray | None = None  # (N-1,) seconds per waypoint hop
    gripper: float | None = None
    ee_path: np.ndarray | None = None  # (N, 3) pinch positions
    min_finger_z: float | None = None
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    op: "Op | None" = None

    @property
    def q_end(self) -> np.ndarray | None:
        return None if self.q_path is None else self.q_path[-1]

    @property
    def max_joint_delta(self) -> float:
        return 0.0 if self.q_path is None else float(np.abs(self.q_path[-1] - self.q_path[0]).max())

    @property
    def needs_override(self) -> bool:
        return self.max_joint_delta > MAX_UNGATED_JOINT_DELTA_RAD


@dataclass
class Op:
    kind: str
    label: str
    call_index: int
    # move: state -> 4x4 target pose;  contact: dz/speed/max_lag/relief;
    # gripper: value;  observe: look_at kwargs
    target: Callable[[ArmState], np.ndarray] | None = None
    joints: np.ndarray | None = None  # move to a joint configuration instead of a pose
    speed: float = 0.1
    params: dict = field(default_factory=dict)


# Parameter lists, in positional order, with defaults (None = required; NAN =
# "keep current"). These are also what the system prompt documents.
SIGNATURES: dict[str, dict[str, float | None]] = {
    "look_at": {"x": None, "y": None, "z": None, "grid_z": NAN, "zoom": 2.0},
    "deproject": {"u": None, "v": None, "z_plane": None},
    "deproject_depth": {"u": None, "v": None},
    "goto": {"x": None, "y": None, "z": None, "arc_z": NAN, "yaw_deg": NAN, "pitch_deg": NAN,
             "azimuth_deg": NAN, "roll_deg": 0.0, "speed": 0.10},
    "move_by": {"forward": 0.0, "left": 0.0, "dz": 0.0, "dyaw_deg": 0.0, "dpitch_deg": 0.0,
                "droll_deg": 0.0, "speed": 0.05},
    "move_until_contact": {"dz": None, "speed": 0.03, "max_lag": 0.012, "relief": 0.002},
    "grasp": {"x": None, "y": None, "z": None, "yaw_deg": NAN, "pitch_deg": 90.0, "azimuth_deg": 0.0,
              "approach_height": 0.08, "speed": 0.10},
    "place": {"x": None, "y": None, "z": None, "yaw_deg": NAN, "approach_height": 0.08, "speed": 0.10},
    "place_until_contact": {"x": None, "y": None, "z": None, "yaw_deg": NAN, "max_lag": 0.012,
                            "relief": 0.002, "approach_height": 0.08, "speed": 0.10},
    "open_gripper": {},
    "close_gripper": {},
    "home": {},
    "done": {},
}
DESCEND_SPEED = 0.04  # final approach of grasp/place, m/s


class Primitives:
    def __init__(self, robot: RealYamRobot, home_joints: np.ndarray, workspace, table_z_fn: Callable[[], float],
                 max_joint_speed: float = 0.6, min_segment_s: float = 0.8, T_world_base: np.ndarray | None = None):
        """T_world_base: where this arm's base sits in the shared WORLD frame
        that every target, path and table height is expressed in (bimanual:
        the right arm's base frame). Identity = single-arm, world == base.
        `workspace` stays in the arm's OWN base frame."""
        self.robot = robot
        self.T_wb = np.eye(4) if T_world_base is None else np.asarray(T_world_base, float)
        self.T_bw = np.linalg.inv(self.T_wb)
        self.model = robot.mj_model
        self.home_joints = np.asarray(home_joints, float)
        self.ws = workspace
        self.table_z = table_z_fn
        self.max_joint_speed = max_joint_speed
        self.min_segment_s = min_segment_s
        self._data = mujoco.MjData(self.model)
        self._site = self.model.site("grasp_site").id
        self._finger_geoms = []
        for g in range(self.model.ngeom):
            if (self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                    and self.model.body(self.model.geom_bodyid[g]).name in FINGER_BODIES):
                mid = self.model.geom_dataid[g]
                adr, n = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
                self._finger_geoms.append((g, self.model.mesh_vert[adr:adr + n].copy()))

    # -- state ------------------------------------------------------------------
    def actual_state(self, prev: ArmState | None = None) -> ArmState:
        # Plan from the setpoint: the measured pose sits below it under
        # gravity, and planning from that would lower the arm every segment.
        s = ArmState(q=self.robot.commanded_joint_positions(), gripper=self.robot.get_gripper(),
                     q_measured=np.array(self.robot.get_joint_positions()))
        if prev is not None:
            s.holding, s.status = prev.holding, prev.status
        return s

    def fk_base(self, q) -> np.ndarray:
        """T_base_site -- the frame IK works in."""
        return self.robot._fk(np.asarray(q, float)).matrix

    def fk(self, q) -> np.ndarray:
        """T_world_site -- the frame everything else (targets, paths, table) is in."""
        return self.T_wb @ self.fk_base(q)

    # -- expansion: primitive call -> ops ---------------------------------------------
    def expand(self, name: str, kw: dict, i: int) -> list[Op]:
        def keep(v, cur):
            return cur if math.isnan(v) else v

        def pose_fn(x, y, z, yaw=NAN, pitch=NAN, az=NAN, roll=0.0):
            """-> list of equivalent candidate poses. A parallel-jaw grasp is
            symmetric under a 180 deg turn about the approach axis, and often
            only one of the two is reachable (on the real arm, jaw_yaw -40 had
            no IK solution while +140 did), so both are offered and resolve()
            uses whichever the arm can reach with the least joint motion."""
            def f(s: ArmState) -> list[np.ndarray]:
                T = self.fk(s.q)
                cur = describe_orientation(T[:3, :3])
                if all(math.isnan(v) for v in (yaw, pitch, az)) and roll == 0.0:
                    Rs = [T[:3, :3]]
                else:
                    args = (keep(pitch, cur["pitch_deg"]), keep(az, cur["azimuth_deg"]))
                    y0 = keep(yaw, cur["jaw_yaw_deg"])
                    Rs = [orientation(y0, *args, roll), orientation(y0, *args, roll + 180.0)]
                out = []
                for R in Rs:
                    P = np.eye(4)
                    P[:3, :3], P[:3, 3] = R, (x, y, z)
                    out.append(P)
                return out
            return f

        def above(z_of: Callable[[ArmState], float]):
            """Same xy + orientation as now, at a new height."""
            def f(s: ArmState) -> np.ndarray:
                T = self.fk(s.q).copy()
                T[2, 3] = z_of(s)
                return T
            return f

        k = kw
        if name == "look_at":
            return [Op("observe", "look_at", i, params=dict(k))]
        if name == "open_gripper":
            return [Op("gripper", "open gripper", i, params={"value": 1.0})]
        if name == "close_gripper":
            return [Op("close", "close gripper until contact", i)]
        if name == "home":
            return [Op("move", "home", i, joints=self.home_joints, speed=0.15)]
        if name == "done":
            return [Op("done", "done", i)]
        if name == "goto":
            ops = []
            if not math.isnan(k["arc_z"]):
                ops.append(Op("move", f"goto: rise to arc_z={k['arc_z']:.3f}", i,
                              target=above(lambda s, a=k["arc_z"]: max(a, self.fk(s.q)[2, 3])), speed=k["speed"]))
                ops.append(Op("move", "goto: traverse at arc_z", i, speed=k["speed"],
                              target=pose_fn(k["x"], k["y"], k["arc_z"], k["yaw_deg"], k["pitch_deg"],
                                             k["azimuth_deg"], k["roll_deg"])))
            ops.append(Op("move", f"goto ({k['x']:.3f}, {k['y']:.3f}, {k['z']:.3f})", i, speed=k["speed"],
                          target=pose_fn(k["x"], k["y"], k["z"], k["yaw_deg"], k["pitch_deg"],
                                         k["azimuth_deg"], k["roll_deg"])))
            return ops
        if name == "move_by":
            def f(s: ArmState, k=k) -> np.ndarray:
                T = self.fk(s.q).copy()
                R = T[:3, :3]
                a = R[:, 2]
                R = _rot_about(np.array([0, 0, 1.0]), math.radians(k["dyaw_deg"])) @ R
                h = np.cross(a, [0, 0, 1.0])
                if np.linalg.norm(h) > 1e-6:
                    R = _rot_about(h, -math.radians(k["dpitch_deg"])) @ R
                R = _rot_about(R[:, 2], math.radians(k["droll_deg"])) @ R
                T[:3, :3] = R
                T[:3, 3] += (k["forward"], k["left"], k["dz"])
                return T
            return [Op("move", f"move_by fwd={k['forward']:+.3f} left={k['left']:+.3f} dz={k['dz']:+.3f}",
                       i, target=f, speed=k["speed"])]
        if name == "move_until_contact":
            return [Op("contact", f"move_until_contact dz={k['dz']:+.3f}", i, speed=k["speed"], params=dict(k))]
        if name == "grasp":
            h = k["approach_height"]
            return [
                Op("gripper", "grasp: open gripper", i, params={"value": 1.0}),
                Op("move", f"grasp: above target (+{h:.2f} m)", i, speed=k["speed"],
                   target=pose_fn(k["x"], k["y"], k["z"] + h, k["yaw_deg"], k["pitch_deg"], k["azimuth_deg"]),
                   params={"fallback_floor_z": k["z"]}),
                Op("move", f"grasp: descend to z={k['z']:.3f}", i, speed=DESCEND_SPEED,
                   target=pose_fn(k["x"], k["y"], k["z"], k["yaw_deg"], k["pitch_deg"], k["azimuth_deg"])),
                Op("close", "grasp: close until contact", i),
                Op("move", "grasp: lift", i, speed=DESCEND_SPEED, target=above(lambda s: k["z"] + h),
                   params={"fallback_heights": True}),
            ]
        if name in ("place", "place_until_contact"):
            h = k["approach_height"]
            ops = [Op("move", f"{name}: above target (+{h:.2f} m)", i, speed=k["speed"],
                      target=pose_fn(k["x"], k["y"], k["z"] + h, k["yaw_deg"]), params={"fallback_floor_z": k["z"]})]
            if name == "place":
                ops.append(Op("move", f"place: descend to z={k['z']:.3f}", i, speed=DESCEND_SPEED,
                              target=pose_fn(k["x"], k["y"], k["z"], k["yaw_deg"])))
            else:
                ops.append(Op("contact", f"place_until_contact: down to at most z={k['z'] - 0.02:.3f}", i,
                              speed=DESCEND_SPEED,
                              params={"dz": -(h + 0.02), "max_lag": k["max_lag"], "relief": k["relief"]}))
            ops += [Op("gripper", f"{name}: release", i, params={"value": 1.0}),
                    Op("move", f"{name}: retreat up", i, speed=DESCEND_SPEED,
                       target=above(lambda s: self.fk(s.q)[2, 3] + h))]
            return ops
        raise ValueError(f"no expansion for {name}")

    # -- resolution: op + state -> segment ------------------------------------------------
    def resolve(self, op: Op, s: ArmState) -> Segment:
        seg = Segment(op.label, op.kind, op.call_index, op=op)
        if op.kind in ("observe", "done"):
            return seg
        if op.kind == "gripper":
            seg.gripper = op.params["value"]
            return seg
        if op.kind == "close":
            seg.gripper = 0.0
            return seg
        if op.kind == "contact":
            # Preview the FULL extent; execution stops early on contact.
            T = self.fk(s.q).copy()
            T[2, 3] += op.params["dz"]
            self._linear(seg, s.q, [T], op.params["speed"])
            self._check(seg, contact=True)
            if seg.ee_path is not None:
                seg.warnings.append("stops on contact -- path shown to full extent")
            return seg
        if op.joints is not None:
            q_path = np.linspace(s.q, op.joints, 30)
            seg.q_path = q_path
            dur = max(self.min_segment_s, float(np.abs(op.joints - s.q).max()) / self.max_joint_speed)
            seg.durations = np.full(len(q_path) - 1, dur / (len(q_path) - 1))
            seg.warnings.append("joint-space move (path is not a straight line)")
        else:
            try:
                cands = op.target(s)
            except Exception as e:  # noqa: BLE001 -- surface to operator/model
                seg.blockers.append(f"could not compute target: {e}")
                return seg
            cands = cands if isinstance(cands, list) else [cands]
            if not self.ws.contains((self.T_bw @ cands[0])[:3, 3]):
                seg.blockers.append(f"target {np.round(cands[0][:3, 3], 3).tolist()} is outside this arm's "
                                    f"workspace")
            self._linear(seg, s.q, cands, op.speed)
            if seg.blockers and "fallback_floor_z" in op.params:
                # Same joint4 limit, approach side: hover lower over the target.
                floor = op.params["fallback_floor_z"]
                for frac in (0.75, 0.5, 0.3):
                    lower = [T.copy() for T in cands]
                    for T in lower:
                        T[2, 3] = floor + frac * (T[2, 3] - floor)
                    trial = Segment(seg.label, seg.kind, seg.call_index, op=op)
                    self._linear(trial, s.q, lower, op.speed)
                    if not trial.blockers:
                        trial.warnings.append(f"full approach height unreachable (joint limit); hovering "
                                              f"{(lower[0][2, 3] - floor) * 100:.1f} cm above the target instead")
                        seg = trial
                        break
            if seg.blockers and op.params.get("fallback_heights"):
                # A lift that can't reach the full height (top-down near the
                # edge of reach puts joint4 at its limit -- hit on the real
                # arm, holding the brush) should lift as high as it CAN rather
                # than leave the object pinned on the table.
                start_z = self.fk(s.q)[2, 3]
                for frac in (0.75, 0.5, 0.3):
                    lower = [T.copy() for T in cands]
                    for T in lower:
                        T[2, 3] = start_z + frac * (T[2, 3] - start_z)
                    trial = Segment(seg.label, seg.kind, seg.call_index, op=op)
                    self._linear(trial, s.q, lower, op.speed)
                    if not trial.blockers:
                        trial.warnings.append(f"full lift unreachable (joint limit); lifting "
                                              f"{(lower[0][2, 3] - start_z) * 100:.1f} cm instead")
                        seg = trial
                        break
        self._check(seg)
        return seg

    def _linear(self, seg: Segment, q0: np.ndarray, T_goals: list[np.ndarray], speed: float) -> None:
        """Straight line if the arm can follow one; otherwise, for a LARGE
        move only, a joint-space move to an IK solution of the goal.

        The fallback exists for the first approach from a folded start pose:
        swinging the gripper from pointing forward to pointing down while
        translating outward in a straight line runs the wrist out of reach
        partway (seen on the real arm: "IK failed 82% of the way"), though the
        goal itself is easily reachable. A joint-space path isn't straight, so
        it's flagged, and its real swept path is still what's drawn and
        checked against the table. Short moves (descend, lift, nudges) never
        fall back -- those must be straight or not at all.

        T_goals are WORLD poses; IK runs in the arm's base frame."""
        T_goals = [self.T_bw @ T for T in T_goals]
        best, first_fail = None, None
        for T_goal in T_goals:  # straight line to any candidate, least joint motion wins
            probe = Segment(seg.label, seg.kind, seg.call_index)
            self._straight(probe, q0, T_goal, speed)
            if probe.blockers:
                first_fail = first_fail or probe
            elif best is None or probe.max_joint_delta < best.max_joint_delta:
                best = probe
        if best is not None:
            seg.q_path, seg.durations, seg.warnings = best.q_path, best.durations, seg.warnings + best.warnings
            return
        probe = first_fail
        T0, T_goal = self.fk_base(q0), T_goals[0]
        dist = float(np.linalg.norm(T_goal[:3, 3] - T0[:3, 3]))
        angle = math.acos(float(np.clip((np.trace(T0[:3, :3].T @ T_goal[:3, :3]) - 1) / 2, -1, 1)))
        if dist < 0.05 and angle < math.radians(20):
            seg.blockers += probe.blockers
            return
        q_goal = None
        for T_goal, seed in ((T, sd) for T in T_goals
                             for sd in (q0, self.home_joints, np.array([0.0, 1.8, 1.6, -1.3, 0.0, 0.0]))):
            try:
                plan = self.robot.plan_to_pose(T_goal, seed, attempts=5)
            except RuntimeError:
                continue
            if plan.position_error_m <= 0.003 and not self.robot.joints_outside_limits(plan.target_joints):
                if q_goal is None or np.abs(plan.target_joints - q0).max() < np.abs(q_goal - q0).max():
                    q_goal = plan.target_joints
        if q_goal is None:
            seg.blockers += probe.blockers + ["no IK solution for the goal pose either"]
            return
        seg.q_path = np.linspace(q0, q_goal, 30)
        total = max(self.min_segment_s, float(np.abs(q_goal - q0).max()) / self.max_joint_speed)
        seg.durations = np.full(29, total / 29)
        seg.warnings.append(f"joint-space move: no straight line ({probe.blockers[0]}), "
                            "check the drawn path")

    def _straight(self, seg: Segment, q0: np.ndarray, T_goal: np.ndarray, speed: float) -> None:
        """Straight line in position, slerp in rotation, IK every cm/5 deg,
        each solve seeded with the previous one so the path stays continuous.
        T_goal and everything here are in the BASE frame."""
        T0 = self.fk_base(q0)
        dist = float(np.linalg.norm(T_goal[:3, 3] - T0[:3, 3]))
        dR = T0[:3, :3].T @ T_goal[:3, :3]
        angle = math.acos(float(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        n = max(1, int(math.ceil(max(dist / LINEAR_STEP_M, angle / ANGULAR_STEP_RAD))))
        rotvec = _log_so3(dR)
        qs, q = [q0.copy()], q0.copy()
        for t in np.linspace(0, 1, n + 1)[1:]:
            T = np.eye(4)
            T[:3, :3] = T0[:3, :3] @ _exp_so3(rotvec * t)
            T[:3, 3] = (1 - t) * T0[:3, 3] + t * T_goal[:3, 3]
            try:
                plan = self.robot.plan_to_pose(T, q)
            except RuntimeError:
                seg.blockers.append(f"IK failed {t * 100:.0f}% of the way along the straight line")
                return
            if plan.position_error_m > 0.003:
                seg.blockers.append(f"IK residual {plan.position_error_m * 1000:.1f} mm along the path")
                return
            q = plan.target_joints
            qs.append(q.copy())
        seg.q_path = np.array(qs)
        hop = dist / n / max(speed, 1e-3)
        jhop = np.abs(np.diff(seg.q_path, axis=0)).max(axis=1) / self.max_joint_speed
        seg.durations = np.maximum(hop, jhop)
        total = float(seg.durations.sum())
        if total < self.min_segment_s:
            seg.durations *= self.min_segment_s / max(total, 1e-6)
        jumps = np.abs(np.diff(seg.q_path, axis=0)).max(axis=1)
        if jumps.max() > 0.35:
            seg.blockers.append(f"IK branch jump of {jumps.max():.2f} rad between waypoints (the straight line "
                                "passes near a singularity/joint limit) -- try another orientation or an arc_z")

    def _check(self, seg: Segment, contact: bool = False) -> None:
        if seg.q_path is None:
            return
        for q in (seg.q_path[0], seg.q_path[-1]):
            seg.blockers += [f"joint limit: {b}" for b in self.robot.joints_outside_limits(q)]
        pts, lowest = [], np.inf
        for q in seg.q_path:
            set_arm_qpos(self.model, self._data, self.robot.to_model_q(q))
            pts.append(self.T_wb[:3, :3] @ self._data.site_xpos[self._site] + self.T_wb[:3, 3])
            zrow = self.T_wb[2, :3]  # world z of a base-frame point = zrow . p + T_wb[2, 3]
            for g, verts in self._finger_geoms:
                R = self._data.geom_xmat[g].reshape(3, 3)
                wz = (verts @ (R.T @ zrow)) + zrow @ self._data.geom_xpos[g] + self.T_wb[2, 3]
                lowest = min(lowest, float(wz.min()))
        seg.ee_path, seg.min_finger_z = np.array(pts), lowest
        clearance = lowest - self.table_z()
        if clearance < (-0.025 if contact else -0.003):
            seg.blockers.append(f"fingers would go {-clearance * 1000:.0f} mm below the table")
        elif clearance < 0.003:
            seg.warnings.append(f"fingers come within {clearance * 1000:.0f} mm of the table")
        if seg.needs_override:
            seg.warnings.append(f"large joint change ({seg.max_joint_delta:.2f} rad)")

    def preview(self, ops: list[Op], s: ArmState) -> list[Segment]:
        """Resolve every op against a simulated, chained state. No motion."""
        s = replace(s, q=s.q.copy())
        segs = []
        for op in ops:
            seg = self.resolve(op, s)
            segs.append(seg)
            if seg.q_end is not None and not seg.blockers:
                s.q = seg.q_end.copy()
            if seg.gripper is not None:
                s.gripper = seg.gripper
        return segs

    # -- execution ------------------------------------------------------------------------
    def execute(self, seg: Segment, s: ArmState, close_duration_s: float = 2.0) -> tuple[ArmState, str]:
        """THIS MOVES THE ARM. Returns (new state, one-line result)."""
        r = self.robot
        if seg.kind == "gripper":
            r.set_gripper(seg.gripper)
            time.sleep(1.0)
            if seg.gripper > 0.5:
                s.holding = False
            return self.actual_state(s), f"gripper -> {r.get_gripper():.2f}"
        if seg.kind == "close":
            contact = r.close_gripper_until_contact(max_duration_s=close_duration_s)
            time.sleep(0.3)
            g = r.get_gripper()
            # Jaws commanded shut but stopped well short of closed = something
            # is between them, even without an effort spike (on the real rig the
            # brush handle stopped them at 0.26 with contact=False).
            s.holding = bool((contact and g > 0.02) or g > 0.08)
            s.status = "ok" if s.holding else "grasp_failed"
            return self.actual_state(s), f"contact={contact}, gripper={g:.2f}, holding={s.holding}"
        if seg.kind == "move":
            self._stream(seg.q_path, seg.durations)
            time.sleep(0.3)  # let it settle before measuring
            T = self.fk(r.get_joint_positions())
            err = float(np.linalg.norm(T[:3, 3] - seg.ee_path[-1]))
            if err > BLOCKED_M:
                new = self.actual_state(s)
                new.status = "blocked"
                return self._recheck_holding(new, (
                    f"BLOCKED: stopped {err * 1000:.0f} mm short of the planned end point -- something is in "
                    f"the way (the other arm, the object, the table). Look again before retrying"))
            return self._recheck_holding(self.actual_state(s),
                                         f"reached, measured {err * 1000:.1f} mm from plan (gravity sag)")
        if seg.kind == "contact":
            new, msg = self._contact(seg, s)
            return self._recheck_holding(new, msg)
        raise ValueError(seg.kind)

    def _recheck_holding(self, s: ArmState, msg: str) -> tuple[ArmState, str]:
        """`holding` is decided once, at the close; an object can slip out
        afterwards (2026-10-15: the fan slid out of the jaws during the lift and
        obs kept saying holding=true with the jaws reading 0.002). After every
        move, re-check: still holding if the jaws stopped short of closed OR
        they are squeezing something -- a thin wall pinched at the fingertips
        reads ~0.006 but with grip effort ~0.3, where empty jaws sit near 0."""
        if not s.holding:
            return s, msg
        g = s.gripper
        try:
            eff = abs(float(self.robot.get_state().get("gripper_eff", [np.nan])[0]))
        except Exception:  # noqa: BLE001 -- a backend without get_state: position only
            eff = float("nan")
        if g > HOLD_MIN_OPEN or (eff == eff and eff > HOLD_MIN_EFFORT):
            return s, msg
        s.holding, s.status = False, "dropped"
        return s, (msg + f"; DROPPED: gripper reads {g:.3f} with no grip force"
                   + ("" if eff != eff else f" (effort {eff:.2f})") + " -- holding is now false")

    def _stream(self, q_path: np.ndarray, durations: np.ndarray) -> None:
        self.robot.stream_joint_path(q_path, durations)

    def _contact(self, seg: Segment, s: ArmState) -> tuple[ArmState, str]:
        """Walk the planned waypoints; after each, compare where the arm IS
        with where it was commanded. A contact shows up as tracking lag (the
        position loop can't push through the object). On lag > max_lag, back
        off by `relief` and stop."""
        p = seg.op.params
        # The arm always trails its setpoint a bit under gravity, so contact
        # is lag GROWING beyond that resting offset, not lag itself.
        base = float(np.linalg.norm(self.fk(self.robot.get_joint_positions())[:3, 3] - seg.ee_path[0]))
        for k, (q, d) in enumerate(zip(seg.q_path[1:], seg.durations), start=1):
            self.robot.move_to_joint_positions(q.tolist(), duration_s=float(d))
            time.sleep(0.05)
            lag = float(np.linalg.norm(self.fk(self.robot.get_joint_positions())[:3, 3] - seg.ee_path[k])) - base
            if lag > p["max_lag"]:
                here = self.fk(self.robot.commanded_joint_positions())
                back = here.copy()
                back[2, 3] -= math.copysign(p["relief"], p["dz"])  # back off against the travel
                try:
                    plan = self.robot.plan_to_pose(self.T_bw @ back, self.robot.commanded_joint_positions())
                    self.robot.execute_plan(plan, duration_s=0.5)
                except RuntimeError:
                    pass
                s.status = "contact"
                return self.actual_state(s), (f"contact after {abs(seg.ee_path[k][2] - seg.ee_path[0][2]) * 1000:.0f} mm "
                                              f"(lag {lag * 1000:.1f} mm), relieved {p['relief'] * 1000:.0f} mm")
        s.status = "no_contact"
        return self.actual_state(s), "full travel, no contact detected"


def _log_so3(R: np.ndarray) -> np.ndarray:
    angle = math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if angle < 1e-8:
        return np.zeros(3)
    if math.pi - angle < 1e-6:
        # 180 deg: axis from the largest diagonal entry
        i = int(np.argmax(np.diag(R)))
        v = R[:, i] + np.eye(3)[i]
        return v / np.linalg.norm(v) * angle
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(angle))
    return w * angle


def _exp_so3(v: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    return _rot_about(v / angle, angle)
