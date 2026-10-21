"""Two-arm (bimanual) version of the robot server, viewer and prompt.

Same shape as the single-arm agent (agent/robot_server.py): Gemini POSTs a
Code-as-Policies program to /plan, every motion is previewed in viser and
held for approval. What changes:

  * Programs address an arm by name: `left.grasp(...)`, `right.goto(...)`.
    Lines run one after another; `parallel(left.X(...), right.Y(...))` runs
    one call on each arm at the same time.
  * ONE shared world frame for every number Gemini sees or sends: the RIGHT
    arm's base frame. The left arm's base is placed in it by the solved
    T_leftbase_rightbase (about 40 cm to +y); each arm's Primitives converts
    to its own base before IK (agent/primitives.py, T_world_base).
  * A step is the set of segments approved together: one segment for a
    sequential call, one per arm (paired in order) for a parallel call.
  * Arm-vs-arm clearance is checked on every step: both arms' swept meshes
    (the moving arm(s) along their paths, the other one where it stands),
    sampled in time. Too close blocks the step.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from agent.gemini_agent import SYSTEM_PROMPT
from agent.observer import Observer
from agent.plan_program import Call, PlanError, parse_plan
from agent.primitives import SIGNATURES, ArmState, Primitives, Segment, describe_orientation
from agent.viewer import ApprovalViewer
from viz.viser_robot import RobotMeshes, set_arm_qpos

ARMS = ("left", "right")
CLEARANCE_BLOCK_M = 0.03
CLEARANCE_WARN_M = 0.06
# Gripper vs gripper never blocks: a handover needs the two hands within a
# few mm of each other (2026-10-15: a feasible handover nudge measured 18 mm
# and was blocked). It is shown as a warning; the operator judges it in the
# preview, and the joint stall guard (yam_backend) stops an arm that pushes.
HAND_BODIES = ("gripper", "tip_left", "tip_right")
HAND_WARN_M = 0.03


# -- arm-vs-arm clearance ----------------------------------------------------------------
class ArmCloud:
    """A few hundred surface points of one arm's meshes, placeable at any q
    in the world frame. Coarse on purpose: it's a proximity check, not a
    collision checker -- the 3 cm threshold absorbs the decimation.

    `is_hand` marks the points on the gripper and fingertips (HAND_BODIES):
    gripper-vs-gripper distance is reported separately and never blocks.
    `extra_points` (gripper-body frame) adds geometry the MJCF lacks -- the
    right arm's wrist camera bracket -- counted as ARM, so it does block."""

    def __init__(self, prims: Primitives, per_geom: int = 60, extra_points: np.ndarray | None = None):
        m = prims.model
        self.prims, self.model, self.data = prims, m, mujoco.MjData(m)
        rng = np.random.default_rng(0)
        self.geoms, hand = [], []
        for g in range(m.ngeom):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.body(m.geom_bodyid[g]).name in ("world", "base"):
                continue  # the bases don't move and are 40 cm apart
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            v = v[rng.choice(len(v), min(per_geom, len(v)), replace=False)]
            self.geoms.append((g, v))
            hand.append(np.full(len(v), m.body(m.geom_bodyid[g]).name in HAND_BODIES))
        self.extra = extra_points
        self.gripper_body = m.body("gripper").id
        if extra_points is not None:
            hand.append(np.zeros(len(extra_points), bool))
        self.is_hand = np.concatenate(hand)

    def points(self, q_encoder: np.ndarray) -> np.ndarray:
        set_arm_qpos(self.model, self.data, self.prims.robot.to_model_q(q_encoder))
        pts = [(self.data.geom_xmat[g].reshape(3, 3) @ v.T).T + self.data.geom_xpos[g] for g, v in self.geoms]
        if self.extra is not None:
            b = self.gripper_body
            pts.append((self.data.xmat[b].reshape(3, 3) @ self.extra.T).T + self.data.xpos[b])
        P = np.vstack(pts)
        T = self.prims.T_wb
        return P @ T[:3, :3].T + T[:3, 3]


def wrist_camera_box_points(T_gripper_cam, size=(0.07, 0.05, 0.045), n: int = 6) -> np.ndarray:
    """Surface points of a box around the right wrist D405 + its bracket, in
    the gripper BODY frame (the frame ArmCloud.points places extra points in).
    The camera pose comes from the eye-in-hand calibration; the box is a
    generous envelope (the D405 itself is 42 x 42 x 23 mm), centred a little
    behind the lens. Not in the MJCF, so without this nothing checks it --
    and it did get in the left arm's way on 2026-10-15."""
    from agent.wrist_view import R_OURS_THEIRS

    hx, hy, hz = (v / 2 for v in size)
    g = np.linspace(-1.0, 1.0, n)
    face = np.array([(a, b) for a in g for b in g])
    pts = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            q = np.insert(face, axis, sign, axis=1)
            pts.append(q * np.array([hx, hy, hz]))
    P = np.unique(np.vstack(pts), axis=0)
    P[:, 2] -= 0.015  # body of the camera sits behind the optical origin
    T = R_OURS_THEIRS @ np.asarray(T_gripper_cam.matrix)
    return P @ T[:3, :3].T + T[:3, 3]


def min_arm_distance(clouds: dict, paths: dict, samples: int = 12) -> tuple[float, float]:
    """paths: arm -> (N,6) joint path (a static arm: a 1-row path). Both are
    sampled at the same normalized times, which is how parallel steps run.
    -> (closest distance involving an arm link or the wrist camera,
        closest gripper-to-gripper distance)."""
    a, b = ARMS
    hand_pair = clouds[a].is_hand[:, None] & clouds[b].is_hand[None, :]
    best_arm = best_hand = np.inf
    for t in np.linspace(0, 1, samples):
        qs = {}
        for arm in ARMS:
            p = paths[arm]
            qs[arm] = p[min(int(round(t * (len(p) - 1))), len(p) - 1)]
        A, B = clouds[a].points(qs[a]), clouds[b].points(qs[b])
        D = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
        best_arm = min(best_arm, float(D[~hand_pair].min()))
        best_hand = min(best_hand, float(D[hand_pair].min()))
    return best_arm, best_hand


# -- viewer --------------------------------------------------------------------------------
@dataclass
class Step:
    """Segments approved and run together (one per arm at most)."""

    segs: dict[str, Segment]
    extra_blockers: list[str] = field(default_factory=list)
    extra_warnings: list[str] = field(default_factory=list)
    extra_override: list[str] = field(default_factory=list)  # approvable once the operator ticks override

    @property
    def blockers(self):
        return [f"[{a}] {b}" for a, s in self.segs.items() for b in s.blockers] + self.extra_blockers

    @property
    def warnings(self):
        return [f"[{a}] {w}" for a, s in self.segs.items() for w in s.warnings] + self.extra_warnings

    @property
    def needs_override(self):
        return any(s.needs_override for s in self.segs.values()) or bool(self.extra_override)

    @property
    def label(self):
        return " || ".join(f"[{a}] {s.label}" for a, s in self.segs.items())


class BimanualViewer(ApprovalViewer):
    reviewer = "operator"  # who approves steps, as named in rejection results

    def __init__(self, server, arms: dict[str, Primitives], real: bool):
        super().__init__(server, arms["right"].robot, real)  # GUI + right-arm meshes (world == right base)
        self.arms = arms
        self._ghosts = {"right": self.ghost}
        self._planned = {"right": self.planned}
        self._data = {"right": (self._gd, self._pd)}
        lp = arms["left"]
        self._ghosts["left"] = RobotMeshes(server, lp.model, name="left_now", color=(130, 130, 140), opacity=0.35)
        self._planned["left"] = RobotMeshes(server, lp.model, name="left_next", color=(110, 160, 230))
        self._planned["left"].set_visible(False)
        self._data["left"] = (mujoco.MjData(lp.model), mujoco.MjData(lp.model))
        self.wrist.visible = False
        # _arm_at is called from the session thread (after each step) and from
        # the state logger's thread (follow_arm); both share the MjData above.
        self._arm_lock = threading.Lock()
        self._followed: dict[str, tuple[np.ndarray, float]] = {}

    def _arm_at(self, arm, which, q, gripper):
        prims = self.arms[arm]
        meshes = (self._ghosts if which == "now" else self._planned)[arm]
        from camera.extrinsics import Extrinsics

        with self._arm_lock:
            data = self._data[arm][0 if which == "now" else 1]
            set_arm_qpos(prims.model, data, prims.robot.to_model_q(q))
            for name in ("joint7", "joint8"):
                j = prims.model.joint(name)
                data.qpos[j.qposadr[0]] = j.range[1] * (1.0 - gripper)
            mujoco.mj_kinematics(prims.model, data)
            meshes.update(data, base_transform=None if arm == "right" else Extrinsics(prims.T_wb))

    def set_now_arm(self, arm, q, gripper):
        self._arm_at(arm, "now", q, gripper)

    def follow_arm(self, arm, state: dict) -> None:
        """StateLogger.on_sample: draw the arm where it really is, live. Skips
        samples where nothing moved, so an idle arm doesn't fill scene.viser
        with identical pose messages."""
        q = np.asarray(state["joint_pos"], dtype=float)
        g = state.get("gripper_pos")
        prev = self._followed.get(arm)
        gripper = float(g[0]) if g else (prev[1] if prev else 1.0)
        if prev is not None and np.abs(q - prev[0]).max() < 1e-3 and abs(gripper - prev[1]) < 5e-3:
            return
        self._followed[arm] = (q, gripper)
        self.set_now_arm(arm, q, gripper)

    def show_steps(self, steps: list[Step], next_idx: int, results: dict[int, str], states: dict[str, ArmState]):
        for h in self._seg_handles:
            h.remove()
        self._seg_handles = []
        sc = self.server.scene
        rows = []
        colors = {"right": ((0, 200, 80), (120, 200, 140)), "left": ((60, 130, 255), (140, 170, 230))}
        for i, st in enumerate(steps):
            state = "done" if i < next_idx else ("**NEXT**" if i == next_idx else "")
            flag = " BLOCKED" if st.blockers and i >= next_idx else ""
            rows.append(f"| {i + 1} | {st.label} | {state}{flag} | {results.get(i, '')} |")
            if i < next_idx:
                continue
            for arm, s in st.segs.items():
                if s.ee_path is None:
                    continue
                active = i == next_idx
                self._seg_handles += [
                    sc.add_spline_catmull_rom(f"plan/{i}_{arm}/path", s.ee_path,
                                              color=colors[arm][0 if active else 1],
                                              thickness=0.004 if active else 0.0015),
                    sc.add_label(f"plan/{i}_{arm}/label", f"{i + 1}{arm[0].upper()}", position=s.ee_path[-1]),
                ]
        self.program_md.content = "| # | step | state | result |\n|---|---|---|---|\n" + "\n".join(rows)
        nxt = steps[next_idx] if next_idx < len(steps) else None
        for arm in ARMS:
            s = nxt.segs.get(arm) if nxt is not None else None
            if s is not None and s.q_end is not None and not s.blockers:
                self._arm_at(arm, "next", s.q_end, states[arm].gripper)
                self._planned[arm].set_visible(True)
            elif s is not None and s.gripper is not None:
                self._arm_at(arm, "next", states[arm].q, s.gripper)
                self._planned[arm].set_visible(True)
            else:
                self._planned[arm].set_visible(False)

    def describe(self, step) -> str:
        if not isinstance(step, Step):
            return super().describe(step)
        parts = [f"### Next: {step.label}"]
        for arm, s in step.segs.items():
            body = super().describe(s).split("\n\n", 1)
            parts.append(f"#### {arm} arm\n\n" + (body[1] if len(body) > 1 else ""))
        parts += [f"**BLOCKED:** {b}" for b in step.extra_blockers]
        parts += [f"**Needs your override:** {o}" for o in step.extra_override]
        parts += [f"**Warning:** {w}" for w in step.extra_warnings]
        return "\n\n".join(parts)


# -- session ---------------------------------------------------------------------------------
class BimanualSession:
    def __init__(self, arms: dict[str, Primitives], observer: Observer, viewer: BimanualViewer,
                 grab_overhead, run_dir: Path, close_duration_s: float = 2.0, events=None,
                 wrist_cam=None, wrist_view=None):
        """wrist_cam (camera/d405_wrist.py) is on the RIGHT arm. With
        wrist_view (agent/wrist_view.py, from T_right_wrist_gripper_cam) the
        wrist image Gemini gets carries the same world grid + reticle as the
        overhead one and its depth goes into viser as a second cloud; without
        it, the raw wrist image."""
        self.wrist_cam, self.wrist_view = wrist_cam, wrist_view
        self.arms, self.obs_, self.viewer = arms, observer, viewer
        self.grab_overhead, self.run_dir = grab_overhead, run_dir
        self.close_duration_s = close_duration_s
        self.events = events or (lambda *a, **k: None)  # agent/run_log.EventLog
        self.lock = threading.Lock()
        self.states = {a: arms[a].actual_state() for a in ARMS}
        self.clouds = {a: ArmCloud(arms[a]) for a in ARMS}
        try:  # the right wrist D405 + bracket is not in the MJCF; add it to the clearance check
            from camera.rig_calibration import load_extrinsic

            box = wrist_camera_box_points(load_extrinsic("T_right_wrist_gripper_cam"))
            self.clouds["right"] = ArmCloud(arms["right"], extra_points=box)
        except Exception as e:  # noqa: BLE001
            print(f"[clearance] right wrist camera NOT in the clearance check ({e})")
        self.n_obs = 0
        self.overhead_error: str | None = None  # set by observe() when the overhead grab fails
        self.last_images: dict[str, np.ndarray] = {}

    @property
    def state(self) -> ArmState:
        """For the agent loop's done-check (same attribute as RobotSession)."""
        return self.states["right"]

    def _arm_obs(self, arm) -> dict:
        s = self.states[arm]
        T = self.arms[arm].fk(s.q)
        Tm = self.arms[arm].fk(s.q_measured if s.q_measured is not None else s.q)
        return {"ee_pos": [round(float(v), 4) for v in Tm[:3, 3]],
                "ee_pos_commanded": [round(float(v), 4) for v in T[:3, 3]],
                **describe_orientation(Tm[:3, :3]),
                "gripper_open": round(s.gripper, 3), "holding": s.holding, "status": s.status}

    def observe(self) -> dict:
        # A failed overhead grab must not reach Gemini as the old image (2026-10-15:
        # seven turns planned from one frozen frame after the ZED dropped off USB).
        try:
            self.obs_.set_frame(self.grab_overhead())
            self.overhead_error = None
        except RuntimeError as e:
            self.overhead_error = str(e)
            self.events("overhead_unavailable", error=self.overhead_error)
            print(f"[obs] OVERHEAD UNAVAILABLE: {e}")
        for a in ARMS:
            self.states[a] = self.arms[a].actual_state(self.states[a])
        self.n_obs += 1
        self.last_images, images, path = {}, [], None
        if self.overhead_error is None:
            reticles = []
            for a in ARMS:
                s = self.states[a]
                reticles.append((self.arms[a].fk(s.q_measured if s.q_measured is not None else s.q), s.gripper,
                                 a.upper()))
            img = self.obs_.render(reticles)
            path = self.run_dir / f"obs_{self.n_obs:03d}_overhead.png"
            img.save(path)
            self.last_images["overhead"] = np.asarray(img)
            images.append(str(path))
        wrist_path = self._observe_wrist()
        if wrist_path is not None:
            images.append(str(wrist_path))
        if self.overhead_error is None:
            self.viewer.image.image = self.last_images["overhead"]  # show it now, not after a plan
            self.viewer.set_cloud(*self.obs_.cloud_base())
        for a in ARMS:
            s = self.states[a]
            self.viewer.set_now_arm(a, s.q_measured if s.q_measured is not None else s.q, s.gripper)
        arm_obs = {a: self._arm_obs(a) for a in ARMS}
        self.events("observe", n=self.n_obs, **arm_obs, images=images)
        out = {
            "frame": "right arm base: +x forward, +y left (toward the left arm), +z up; meters",
            "left": arm_obs["left"], "right": arm_obs["right"],
            "table_z": round(self.obs_.table_z, 4), "grid_z": round(self.obs_.view.grid_z, 4),
            "view": ({"center": [round(float(v), 3) for v in self.obs_.view.center], "zoom": self.obs_.view.zoom}
                     if self.obs_.view.center is not None else "full overhead frame"),
        }
        if path is not None:
            out["overhead_cam_path"] = str(path)
        else:
            out["overhead_status"] = (
                f"UNAVAILABLE ({self.overhead_error}). There is NO overhead image this turn, and the last one "
                "you saw is out of date. deproject/deproject_depth are disabled until it is back. Do not plan "
                "moves to new targets from memory; call GET /obs again (the camera re-opens itself when it can), "
                "or only make moves you can verify in the wrist image.")
        if wrist_path is not None:
            out["wrist_cam_path"] = str(wrist_path) + "  (camera on the RIGHT gripper)"
        return out

    def _observe_wrist(self) -> Path | None:
        if self.wrist_cam is None:
            return None
        from PIL import Image

        s = self.states["right"]
        q_now = s.q_measured if s.q_measured is not None else s.q
        if self.wrist_view is not None:
            rgb, depth = self.wrist_cam.grab_rgbd()
            wrist = np.asarray(self.wrist_view.render(rgb, q_now, self.arms["right"].fk(q_now), s.gripper,
                                                      self.obs_.view.grid_z, self.obs_.ws))
            if depth is not None:
                self.viewer.set_wrist_cloud(*self.wrist_view.cloud(rgb, depth, q_now))
        else:
            wrist = self.wrist_cam.grab()
        path = self.run_dir / f"obs_{self.n_obs:03d}_wrist.png"
        Image.fromarray(wrist).save(path)
        self.last_images["wrist"] = wrist
        self.viewer.wrist.visible = True
        self.viewer.wrist.image = wrist
        return path

    def _expand(self, call: Call, i: int):
        return [(call.arm, op) for op in self.arms[call.arm].expand(call.name, call.kwargs, i)]

    def _build_steps(self, items) -> list[list[tuple[str, object]]]:
        """-> list of steps, each a list of (arm, op) to run together."""
        steps = []
        for i, item in enumerate(items):
            if isinstance(item, list):  # parallel: pair the two arms' ops in order
                a, b = (self._expand(c, i) for c in item)
                for k in range(max(len(a), len(b))):
                    steps.append([x for x in (a[k] if k < len(a) else None, b[k] if k < len(b) else None) if x])
            else:
                steps += [[x] for x in self._expand(item, i)]
        return steps

    def _resolve_step(self, step_ops, states) -> Step:
        segs = {arm: self.arms[arm].resolve(op, states[arm]) for arm, op in step_ops}
        st = Step(segs)
        if all(s.q_path is not None or s.kind in ("gripper", "close", "observe", "done") for s in segs.values()):
            paths = {a: (segs[a].q_path if a in segs and segs[a].q_path is not None
                         else states[a].q[None, :]) for a in ARMS}
            if any(p.shape[0] > 1 for p in paths.values()):
                d_arm, d_hand = min_arm_distance(self.clouds, paths)
                if d_arm < CLEARANCE_BLOCK_M:
                    st.extra_override.append(
                        f"arm links / wrist camera come within {d_arm * 1000:.0f} mm of the other arm -- check the "
                        f"preview; tick override only if they really clear")
                elif d_arm < CLEARANCE_WARN_M:
                    st.extra_warnings.append(f"arm links / wrist camera come within {d_arm * 1000:.0f} mm of the other arm")
                if d_hand < HAND_WARN_M:
                    st.extra_warnings.append(f"grippers come within {d_hand * 1000:.0f} mm of each other "
                                             f"(expected in a handover -- check the fingers in the preview)")
        return st

    def _preview(self, steps_ops, states) -> list[Step]:
        sim = {a: ArmState(q=states[a].q.copy(), gripper=states[a].gripper) for a in ARMS}
        out = []
        for step_ops in steps_ops:
            st = self._resolve_step(step_ops, sim)
            out.append(st)
            for a, s in st.segs.items():
                if s.q_end is not None and not s.blockers:
                    sim[a].q = s.q_end.copy()
                if s.gripper is not None:
                    sim[a].gripper = s.gripper
        return out

    def _step_preview_fields(self, st: Step) -> dict:
        """What the approval window showed for this step, per arm."""
        segs = {}
        for a, s in st.segs.items():
            segs[a] = {"label": s.label, "kind": s.kind,
                       "q_start": None if s.q_path is None else s.q_path[0].tolist(),
                       "q_end": None if s.q_end is None else s.q_end.tolist(),
                       "ee_end": None if s.ee_path is None else s.ee_path[-1].tolist(),
                       "duration_s": None if s.durations is None else float(s.durations.sum())}
        return {"label": st.label, "blockers": st.blockers, "warnings": st.warnings,
                "needs_override": st.needs_override, "segs": segs}

    def run_plan(self, code: str) -> dict:
        self.events("plan_received", code=code)
        try:
            if self.overhead_error is None:
                pure = {"deproject": self.obs_.deproject, "deproject_depth": self.obs_.deproject_depth}
            else:  # it would project through the last, stale overhead frame
                def _no_overhead(*_a, **_k):
                    raise PlanError("overhead camera unavailable: deproject is disabled until GET /obs "
                                    "returns a live overhead image")
                pure = {"deproject": _no_overhead, "deproject_depth": _no_overhead}
            items = parse_plan(code, SIGNATURES, pure, receivers=ARMS)
        except (PlanError, ValueError) as e:
            self.events("plan_error", error=str(e))
            return {"status": "plan_error", "error": str(e), "executed": []}
        steps_ops = self._build_steps(items)
        for a in ARMS:
            self.states[a] = self.arms[a].actual_state(self.states[a])
            self.states[a].status = "ok"
        steps = self._preview(steps_ops, self.states)
        log, results, outcome = [], {}, "completed"
        for idx, step_ops in enumerate(steps_ops):
            # re-resolve from the ACTUAL state (cheap enough for one step)
            steps[idx:] = self._preview(steps_ops[idx:], self.states)
            st = steps[idx]
            self.viewer.show_steps(steps, idx, results, self.states)
            kinds = {s.kind for s in st.segs.values()}
            if kinds <= {"observe"}:
                for s in st.segs.values():
                    p, v = s.op.params, self.obs_.view
                    v.center, v.zoom = np.array([p["x"], p["y"], p["z"]]), max(1.0, float(p["zoom"]))
                    if not np.isnan(p["grid_z"]):
                        v.grid_z = float(p["grid_z"])
                results[idx] = "view updated"
                log.append({"step": st.label, "result": results[idx]})
                continue
            if "done" in kinds:
                for a in ARMS:
                    self.states[a].status = "done"
                results[idx] = "done"
                log.append({"step": "done", "result": "task marked done"})
                break
            self.events("step_pending", index=idx, **self._step_preview_fields(st))
            choice, feedback = self.viewer.wait_decision(st)
            self.events("operator_decision", index=idx, label=st.label, choice=choice, feedback=feedback,
                        stop_requested=self.viewer.stop_requested)
            if choice != "approve":
                why = feedback or ("; ".join(st.blockers) or "no reason given")
                results[idx] = "REJECTED"
                log.append({"step": st.label, "result": f"rejected by {self.viewer.reviewer}: {why}"})
                outcome = "session_stopped" if self.viewer.stop_requested else "rejected_by_operator"
                break
            self.viewer.status_md.content = f"**Executing:** {st.label}"
            self.events("exec_start", index=idx, label=st.label)
            res, errs = {}, {}

            def run(arm, seg):
                try:
                    self.states[arm], res[arm] = self.arms[arm].execute(seg, self.states[arm],
                                                                        self.close_duration_s)
                except Exception as e:  # noqa: BLE001
                    errs[arm] = str(e)

            threads = [threading.Thread(target=run, args=(a, s)) for a, s in st.segs.items()]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            results[idx] = "; ".join(f"[{a}] {res.get(a) or 'ERROR ' + errs.get(a, '')}" for a in st.segs)
            # An arm that wasn't in this step can still lose its object because of it
            # (2026-10-15: the clamp slid out of the LEFT jaws 0.4 s after the RIGHT
            # gripper opened, and obs kept saying left holding=true). Re-check every
            # other arm that is holding something.
            for a in ARMS:
                if a in st.segs or not self.states[a].holding:
                    continue
                s, msg = self.arms[a]._recheck_holding(self.arms[a].actual_state(self.states[a]), "")
                self.states[a] = s
                if s.status == "dropped":
                    results[idx] += f"; [{a}] {msg.lstrip('; ')} (during the other arm's step)"
            log.append({"step": st.label, "result": results[idx]})
            print(f"  [{st.label}] {results[idx]}")
            self.events("exec_end" if not errs else "exec_error", index=idx, label=st.label,
                        result=results[idx], errors=errs or None,
                        arms={a: {"q_measured": None if self.states[a].q_measured is None
                                  else self.states[a].q_measured.tolist(),
                                  "gripper": self.states[a].gripper, "holding": self.states[a].holding,
                                  "status": self.states[a].status} for a in st.segs})
            for a in ARMS:
                s = self.states[a]
                self.viewer.set_now_arm(a, s.q_measured if s.q_measured is not None else s.q, s.gripper)
            if errs:
                outcome = "execution_error"
                break
            bad = [self.states[a].status for a in ARMS
                   if (a in st.segs and self.states[a].status in ("grasp_failed", "blocked"))
                   or self.states[a].status == "dropped"]
            if bad:
                outcome = bad[0]
                break  # let Gemini look before doing anything else
        self.viewer.show_steps(steps, len(steps), results, self.states)
        if self.viewer.auto.value:
            self.viewer.auto.value = False
        self.events("plan_end", status=outcome, n_steps=len(steps_ops))
        return {"status": outcome, "executed": log}


# -- prompt ------------------------------------------------------------------------------------
def bimanual_prompt(checker: str = "human") -> str:
    """The single-arm prompt with every `robot.` turned into an arm name, plus
    what two arms change. Kept derived so the two never drift apart.

    checker: "human" (an operator approves every step in viser) or "auto"
    (gap_guard does; agent/gap_checker.py swaps the operator sentences)."""
    p = SYSTEM_PROMPT.replace("a real 6-DoF YAM robot arm with a\nparallel-jaw gripper",
                              "TWO real 6-DoF YAM robot arms (LEFT and RIGHT),\neach with a parallel-jaw gripper,")
    p = p.replace("x, y = robot.deproject(u=0.52, v=0.48, z_plane=0.015)\nrobot.grasp(x=x, y=y, z=0.015, yaw_deg=-35)",
                  "x, y = right.deproject(u=0.52, v=0.48, z_plane=0.015)\n"
                  "parallel(right.grasp(x=x, y=y, z=0.015, yaw_deg=-35), left.goto(x=0.30, y=0.45, z=0.15))\n"
                  "left.open_gripper()")
    p = p.replace("2. Wrist camera (if present):", "2. Wrist camera (if present; on the RIGHT gripper only):")
    p = p.replace("robot.<primitive>(...)", "left.<primitive>(...) / right.<primitive>(...)")
    p = p.replace("assignments from robot.deproject*", "parallel(...), assignments from <arm>.deproject*")
    p = p.replace("robot.", "<arm>.")
    p = p.replace("## Frames and units\nArm base frame:", "## Frames and units\nONE shared frame for both arms -- "
                  "the RIGHT arm's base frame:")
    p = p.replace("Stay inside obs.workspace.", "The LEFT arm's base is at about (0, +0.40, 0) in this frame. "
                  "Each arm reaches about 0.12-0.55 m from its own base; give each arm the objects on its side.")
    p += """
## Two arms
- <arm> is `left` or `right`. obs has a "left" and a "right" block (ee_pos is
  where the arm IS; ee_pos_commanded where it was told to be -- they differ by
  the arm's gravity sag), plus the shared table/grid/view fields.
- Lines run one after another. `parallel(left.X(...), right.Y(...))` runs one
  call on each arm AT THE SAME TIME, segment by segment (e.g. both grasp
  phases in step). Use it for independent motions; keep anything where the
  arms come close sequential.
- The overhead image shows one reticle per arm, labelled LEFT / RIGHT.
- Every step is checked for the arms coming close. Arm links (and the right
  wrist camera) within 3 cm of the other arm need the operator's override.
  The two GRIPPERS may come close -- a handover needs it -- that is only a
  warning the operator checks. Still: approach the object from the side away
  from the other gripper, grasp a part a few cm from the other gripper's
  fingers, and close the second gripper only once both are in place.
- If a step comes back "blocked", the arm stopped short because something is
  in the way; look at the images and change the approach -- never resend it.
- deproject* doesn't depend on the arm: left.deproject and right.deproject
  return the same world point.

## Grasping and handover protocol (follow it)
1. SAFE grasp pose. Choose every grasp and approach so the gripper, wrist
   and the rest of the arm stay clear of the other arm, the other gripper's
   fingers (and the right wrist camera on its +y side) and every obstacle
   on the table. Approach the object from the side facing away from the
   other arm.
2. STABLE grasp. Grip a thick, solid part close to the object's centre of
   mass, with the jaws closing across it. Never grip a tip, a thin wall or
   rib, a tapered end or a far end: those held only while something else
   carried the weight and slipped out the moment it had to carry the object
   alone (fan corners, clamp jaw tips and thin handles all slipped today).
3. CONFIRM before relying on it. Right after a close, gripper_open should be
   clearly above 0 for a solid part (>= ~0.1); near 0 means it caught an
   edge -- open and regrasp before lifting. After lifting, look again
   (gripper_open, holding, wrist image) before carrying.
4. Handover release. Close the receiving gripper first; then call GET /obs
   (no motion) and confirm the receiver has a STABLE grasp (rule 2: a thick
   part near the centre of mass, gripper_open >= ~0.1, holding true, visible
   in the images). Only then open the giving gripper -- alone, in its own
   /plan, so its result is checked -- and confirm the receiver still holds
   before moving either arm. If the receiver's grasp is not stable, regrasp
   with the receiver; do not release.
5. Keep the load short. Holding a load out at arm's length heats the
   shoulder within minutes; keep the holding arm compact (near its own base)
   and don't park a held object in the air while you deliberate.

## Learned on this rig
- `yaw_deg` is the jaw LINE, so it is modulo 180 deg (-20 and 160 are the
  same jaw line, but put the wrist camera on opposite sides). goto() only
  applies yaw_deg when pitch_deg=90; to turn the wrist otherwise, or to pick
  the direction of rotation, use move_by(dyaw_deg=...).
- A held object hangs below the jaws by up to its own length. Before carrying
  it, lift it until its lowest point clears the table and everything on it;
  then carry slowly (speed<=0.04) -- dragging, snagging or a fast base swing
  pulls it out of the jaws.
- Seen straight down, an object lying on the table directly under the jaws
  looks like it is held. Trust gripper_open/holding, and lift a little and
  look again when unsure.
- To place in a container, release over its MIDDLE, low enough to drop in
  cleanly but with the object clear of the rim while moving there.
- Keep grasp z at least ~6 mm above table_z: the arm sags several mm on the
  way down, and lower jams the fingertips into the table.
- status "dropped" (the object slipped out -- also reported for an arm that
  was not moving) or "blocked": look again before planning anything else.
"""
    if checker == "auto":
        from agent.gap_checker import AUTO_CHECK_TEXT

        for old, new in AUTO_CHECK_TEXT.items():
            if old not in p:
                raise RuntimeError(f"operator text not found in the prompt: {old[:60]!r}")
            p = p.replace(old, new)
    elif checker != "human":
        raise ValueError(f"checker must be 'human' or 'auto', not {checker!r}")
    return p
