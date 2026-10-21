"""Automatic motion checking: gap_guard in the operator's seat.

The human-review harness holds every step in viser until a person approves
it. Here the same step goes to GapChecker instead, which answers exactly one
of OK / IK_FAILURE / COLLISION. OK runs the step; anything else rejects the
plan, and only that word goes back to Gemini as the reason.

The check, per step (one or both arms):
  - planner blockers (IK found no path, a target below the table, ...):
    "below the table" -> COLLISION, any other -> IK_FAILURE;
  - every waypoint inside the joint limits, else IK_FAILURE;
  - each moving arm along the trajectory it will actually be streamed
    (RealYamRobot.stream_joint_path: from its current setpoint, min-jerk
    timed), with the other arm moving along its own trajectory on the same
    clock, against: itself, the other arm, the table plane and a fresh
    overhead cloud as a heightfield (gap_guard.scene.CollisionScene), else
    COLLISION. A grasp's descent / lift and a place's release may touch the
    cloud with the fingertips (that is the object); move/place_until_contact
    are only checked arm-vs-arm and self (touching the table is their point);
  - gripper open/close: the finger sweep against both arms and the table,
    not the cloud (what the jaws close on is the point of closing them).

COLLISION means actual contact (0 mm clearance) of the convex hulls. The
fingertips collide as convex pieces at each gripper's measured opening, and
the right wrist D405 + bracket is included (camera.wrist_extrinsic). Points
within HELD_RADIUS_M of a gripper that is holding something are dropped from
the cloud, so a held object is neither an obstacle nor checked.

Validated in simulation (the clamp twin, tools/sim_argon_rerun.py in the
development repo) against physics contacts and against the recorded human
decisions. What it does NOT establish: that a receiving grasp will hold the
object once the giver lets go. The handover protocol in the prompt (observe
before release, recheck holding) is still what covers that.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from agent.bimanual import ARMS, BimanualViewer
from camera.extrinsics import Extrinsics
from gap_guard.verifier import Verdict

HELD_RADIUS_M = 0.10
FINGERS_MAY_TOUCH = ("grasp: descend", "grasp: lift", "place: descend", "place: release", "place: retreat",
                     "place_until_contact: release", "place_until_contact: retreat")

# The prompt sentences that describe the human operator, and what replaces
# them when gap_guard is the checker. Matched exactly: bimanual_prompt()
# raises if one is missing, so the two prompts cannot drift apart silently.
AUTO_CHECK_TEXT = {
    "A human operator previews every motion segment in 3D and must approve it\n"
    "before the arm moves. If they reject, their feedback is in the response --\n"
    "follow it.":
    "An automatic motion checker previews every motion segment and must pass it\n"
    "before the arm moves. If it rejects one, the response says why in one word:\n"
    "IK_FAILURE (no valid joint path) or COLLISION (something would be hit).",
    "Arm links (and the right\n  wrist camera) within 3 cm of the other arm need the operator's override.\n"
    "  The two GRIPPERS may come close -- a handover needs it -- that is only a\n"
    "  warning the operator checks.":
    "The arms, the right\n  wrist camera and the grippers are checked for collisions.\n"
    "  The two GRIPPERS may come close -- a handover needs it -- as long as they\n"
    "  do not collide.",
}


class _Why(logging.Handler):
    """Collects gap_guard.scene's debug lines (which pair, how deep) for the
    log. The verdict sent to Gemini never carries them."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def min_jerk_path(q_now: np.ndarray, seg) -> tuple:
    """(q(t), total s): RealYamRobot.stream_joint_path's trajectory -- from
    the current setpoint through the waypoints, min-jerk timed overall."""
    path = np.vstack([q_now, seg.q_path[1:]])
    knots = np.concatenate([[0.0], np.cumsum(seg.durations)])
    total = float(knots[-1])

    def q(t):
        if total <= 0:
            return path[-1]
        tau = min(max(t / total, 0.0), 1.0)
        v = total * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
        k = min(int(np.searchsorted(knots, v, side="right")) - 1, len(path) - 2)
        a = (v - knots[k]) / max(knots[k + 1] - knots[k], 1e-9)
        return (1 - a) * path[k] + a * path[k + 1]

    return q, total


class GapChecker:
    """decide(step) -> ("approve", "") or ("reject", "IK_FAILURE" | "COLLISION").
    Every decision is appended to <run_dir>/gap_checks.jsonl."""

    def __init__(self, arms: dict, T_world_left: np.ndarray, run_dir: Path, gripper_parts: dict | None = None):
        from camera.wrist_extrinsic import wrist_parts

        self.arms = arms
        self.T_base_world = {"right": np.eye(4), "left": np.linalg.inv(T_world_left)}  # world = right base
        self.arm_xml = {a: arms[a].robot.mj_xml for a in ARMS}
        self.gripper_parts = gripper_parts if gripper_parts is not None else {a: wrist_parts(a) for a in ARMS}
        self.out = run_dir / "gap_checks.jsonl"
        self.session = None  # set once the BimanualSession exists
        self.on_record = None  # called with each decision record
        self.n = 0
        self._why = _Why()
        log = logging.getLogger("gap_guard.scene")
        log.setLevel(logging.DEBUG)
        log.addHandler(self._why)
        log.propagate = False

    def decide(self, step) -> tuple[str, str]:
        self._why.lines.clear()
        why: list[str] = []
        t0 = time.perf_counter()
        try:
            verdict = self._verdict(step, why)
        except Exception as e:  # noqa: BLE001 -- a checker that cannot check must not approve
            why.append(f"checker error: {e!r}")
            verdict = Verdict.COLLISION
        self.n += 1
        rec = {"n": self.n, "t": round(time.time(), 3), "label": step.label, "verdict": verdict.value,
               "ms": round((time.perf_counter() - t0) * 1e3, 1), "why": why[:4], "warnings": step.warnings}
        with self.out.open("a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        if self.on_record is not None:
            self.on_record(rec)
        print(f"  [gap_guard] {step.label[:60]:<60} {verdict.value}" + (f"  ({why[0]})" if why else ""))
        return ("approve", "") if verdict is Verdict.OK else ("reject", verdict.value)

    # -- the check
    def _verdict(self, step, why: list[str]) -> Verdict:
        if step.blockers:
            why += step.blockers
            return Verdict.COLLISION if any("below the table" in b for b in step.blockers) else Verdict.IK_FAILURE
        st = self.session.states
        moving = {a: s for a, s in step.segs.items() if s.q_path is not None and s.durations is not None}
        grips = {a: s for a, s in step.segs.items() if s.kind in ("gripper", "close")}
        if not moving and not grips:
            return Verdict.OK
        cloud = self._cloud_world(st)
        trajs = {}  # arm -> (q_model(t), total s)
        for a in ARMS:
            robot = self.arms[a].robot
            if a in moving:
                q_enc, total = min_jerk_path(robot.commanded_joint_positions(), moving[a])
                trajs[a] = ((lambda f=q_enc, r=robot: lambda t: r.to_model_q(f(t)))(), total)
            else:
                q = robot.to_model_q(st[a].q_measured if st[a].q_measured is not None else st[a].q)
                trajs[a] = ((lambda q=q: lambda t: q)(), 0.0)
        T_end = max(total for _, total in trajs.values())
        travel = max(float(np.abs(np.diff(s.q_path, axis=0)).sum(0).max()) for s in moving.values()) if moving else 0
        times = np.linspace(0.0, T_end, max(50, int(np.ceil(travel / 0.005))) + 1)
        for a in list(moving) + list(grips):
            b = "left" if a == "right" else "right"
            qa, qb = trajs[a][0], trajs[b][0]
            scene = self._scene(a, b, qa(0.0), qb(0.0), st, cloud)
            if a in moving:
                seg = moving[a]
                lo, hi = scene.a_jnt_range[:, 0] - 1e-3, scene.a_jnt_range[:, 1] + 1e-3
                q_model = [self.arms[a].robot.to_model_q(q) for q in seg.q_path]
                bad = [i for i, q in enumerate(q_model) if np.any(q < lo) or np.any(q > hi)]
                if bad:
                    why.append(f"[{a}] waypoint {bad[0]} outside joint limits")
                    return Verdict.IK_FAILURE
                at = {"kinds": ("arm", "self")} if seg.kind == "contact" else {
                    "cloud": "fingers_may_touch" if seg.label.startswith(FINGERS_MAY_TOUCH) else "check"}
                baseline = scene.contact_distances(qa(0.0), other_q=qb(0.0), **at)
                for t in times:
                    if scene.in_collision(qa(t), baseline, other_q=qb(t), **at):
                        why.append(f"[{a}] {self._last_contact()} at t={t:.2f}/{T_end:.2f} s")
                        return Verdict.COLLISION
            else:
                seg = grips[a]
                q = qa(0.0)
                f0, f1 = scene.finger_q(st[a].gripper), scene.finger_q(0.0 if seg.kind == "close" else seg.gripper)
                baseline = scene.contact_distances(q, fingers=f0, cloud="ignore")
                for x in np.linspace(0.0, 1.0, 41):
                    if scene.in_collision(q, baseline, fingers=(1 - x) * f0 + x * f1, cloud="ignore"):
                        why.append(f"[{a}] gripper: {self._last_contact()}")
                        return Verdict.COLLISION
        return Verdict.OK

    def _scene(self, a, b, qa, qb, st, cloud_world):
        from gap_guard.scene import CollisionScene

        T_aw = self.T_base_world[a]
        T_ab = T_aw @ np.linalg.inv(self.T_base_world[b])
        # A point on the table in front of arm a, for the table height in a's base frame.
        table_world = np.linalg.inv(T_aw) @ np.array([0.35, 0.0, 0.0, 1.0])
        table_world[2] = self.session.obs_.table_z
        return CollisionScene(
            self.arm_xml[a], other_arm_xml=self.arm_xml[b], T_this_other=Extrinsics(T_ab), other_q_model=qb,
            start_q_model=qa, gripper_parts=self.gripper_parts[a], other_gripper_parts=self.gripper_parts[b],
            gripper_open=st[a].gripper, other_gripper_open=st[b].gripper,
            table_z=float((T_aw @ table_world)[2]),
            cloud_points=cloud_world @ T_aw[:3, :3].T + T_aw[:3, 3])

    def _cloud_world(self, st) -> np.ndarray:
        """The latest overhead cloud in the world (right base) frame, minus
        what any holding gripper holds. Empty when the overhead is down."""
        if self.session.overhead_error is not None or self.session.obs_.frame is None:
            return np.zeros((0, 3))
        pts = self.session.obs_.cloud_base()[0]
        for a in ARMS:
            if st[a].holding:
                p = self.arms[a].fk(st[a].q_measured if st[a].q_measured is not None else st[a].q)[:3, 3]
                pts = pts[np.linalg.norm(pts - p, axis=1) > HELD_RADIUS_M]
        return pts

    def _last_contact(self) -> str:
        lines = [w for w in self._why.lines if w.startswith("contact")]
        return lines[-1] if lines else "contact"


class AutoCheckViewer(BimanualViewer):
    """The same viser page (both arms live, the cloud, planned paths, the
    images Gemini gets), with gap_guard deciding instead of the buttons. The
    Stop button still stops the session; a "gap_guard" panel lists verdicts."""

    reviewer = "motion checker"

    def __init__(self, server, arms, checker: GapChecker, real: bool):
        super().__init__(server, arms, real=real)
        self.checker = checker
        for h in (self.approve_btn, self.reject_btn, self.auto, self.feedback, self.override):
            h.visible = False
        self.status_md.content = ("**REAL ARMS. gap_guard approves every step.** Keep the e-stop in reach."
                                  if real else "**Sim. gap_guard approves every step.**")
        with server.gui.add_folder("gap_guard"):
            self.gap_md = server.gui.add_markdown("_no checks yet_")
        self._gap_rows: list[str] = []
        checker.on_record = self._log

    def wait_decision(self, step):
        self.next_md.content = self.describe(step)
        if self.stop_requested:
            return "reject", "Operator stopped the session."
        choice, verdict = self.checker.decide(step)
        self.status_md.content = f"**gap_guard: {verdict or 'OK'}** -- {step.label}"
        return choice, verdict

    def _log(self, rec: dict) -> None:
        self._gap_rows.append(f"| {rec['n']} | {rec['label'][:48]} | {rec['verdict']} | {rec['ms']:.0f} |")
        self.gap_md.content = "| # | step | verdict | ms |\n|---|---|---|---|\n" + "\n".join(self._gap_rows[-12:])
