"""viser UI for the Gemini agent: the 3D scene every proposed segment is
checked in, and the buttons that gate the arm.

Scene (arm base frame): overhead point cloud (refreshed every observation),
the robot model where it is NOW (grey ghost) and where the NEXT segment ends
(solid), the path every remaining segment of the current /plan sweeps (next
one bright, rest faint, numbered), the other arm for context.

Panel: Gemini's latest observation/thought/code and the exact image it saw,
the program as segments, details of the next one, and Approve / Reject.
"""
from __future__ import annotations

import threading

import mujoco
import numpy as np
import viser

from viz.viser_robot import RobotMeshes, set_arm_qpos


class ApprovalViewer:
    def __init__(self, server: viser.ViserServer, robot, real: bool):
        self.server, self.robot, self.model = server, robot, robot.mj_model
        sc, gui = server.scene, server.gui
        sc.set_up_direction("+z")
        sc.add_frame("base", axes_length=0.1, axes_radius=0.004)
        self.ghost = RobotMeshes(server, self.model, name="arm_now", color=(130, 130, 140), opacity=0.35)
        self.planned = RobotMeshes(server, self.model, name="arm_next")
        self.planned.set_visible(False)
        self._gd, self._pd = mujoco.MjData(self.model), mujoco.MjData(self.model)
        self._seg_handles: list = []
        self.cloud = None

        self._decision = threading.Event()
        self._choice: tuple[str, str] = ("", "")
        self.stop_requested = False

        with gui.add_folder("Gemini"):
            self.turn_md = gui.add_markdown("_waiting for the first turn_")
            self.image = gui.add_image(np.zeros((90, 160, 3), np.uint8), label="what Gemini saw (overhead)")
            self.wrist = gui.add_image(np.zeros((90, 160, 3), np.uint8), label="wrist D405")
        with gui.add_folder("Approve"):
            self.program_md = gui.add_markdown("_no plan yet_")
            self.next_md = gui.add_markdown("")
            self.override = gui.add_checkbox("Override (large joint change / arms close) -- enable anyway", False,
                                             visible=False)
            self.approve_btn = gui.add_button("APPROVE + EXECUTE SEGMENT", color="red", disabled=True)
            self.auto = gui.add_checkbox("Auto-approve rest of this plan (stops on any warning)", False)
            self.feedback = gui.add_text("Feedback to Gemini", "", multiline=True)
            self.reject_btn = gui.add_button("REJECT PLAN (send feedback)", disabled=True)
            self.status_md = gui.add_markdown("**Sim arm.**" if not real else "**REAL ARM.** Nothing has moved yet.")
        with gui.add_folder("Agent"):
            self.pause = gui.add_checkbox("Pause before next Gemini turn", False)
            self.stop_btn = gui.add_button("Stop agent")
        with gui.add_folder("View", expand_by_default=False):
            self.show_cloud = gui.add_checkbox("Show point cloud", True)
            self.point_size = gui.add_slider("Point size", 0.001, 0.01, 0.0005, 0.003)
            self.show_wrist_cloud = gui.add_checkbox("Show wrist (D405) point cloud", True)

        self.approve_btn.on_click(lambda _: self._decide("approve"))
        self.reject_btn.on_click(lambda _: self._decide("reject"))
        self.override.on_update(lambda _: self._sync_approve())
        self.stop_btn.on_click(lambda _: self._stop())
        self.show_cloud.on_update(lambda _: self.cloud is not None and setattr(self.cloud, "visible", self.show_cloud.value))
        self.point_size.on_update(lambda _: self.cloud is not None and setattr(self.cloud, "point_size", self.point_size.value))
        self._pending = None
        self.wrist_cloud = None
        self.show_wrist_cloud.on_update(lambda _: self.wrist_cloud is not None and setattr(
            self.wrist_cloud, "visible", self.show_wrist_cloud.value))

        @server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = (-0.25, -0.65, 0.55)
            client.camera.look_at = (0.35, 0.0, 0.0)

    # -- scene ------------------------------------------------------------------
    def _arm(self, meshes, data, q, gripper):
        set_arm_qpos(self.model, data, self.robot.to_model_q(q))
        for name in ("joint7", "joint8"):  # 0 = jaws apart; indicative only
            j = self.model.joint(name)
            data.qpos[j.qposadr[0]] = j.range[1] * (1.0 - gripper)
        mujoco.mj_kinematics(self.model, data)
        meshes.update(data)

    def set_cloud(self, xyz, rgb) -> None:
        self.cloud = self.server.scene.add_point_cloud(
            "overhead_cloud", xyz, rgb, point_size=self.point_size.value, point_shape="rounded",
            visible=self.show_cloud.value)

    def set_wrist_cloud(self, xyz, rgb) -> None:
        self.wrist_cloud = self.server.scene.add_point_cloud(
            "wrist_cloud", xyz, rgb, point_size=self.point_size.value * 0.6, point_shape="rounded",
            visible=self.show_wrist_cloud.value)

    def set_now(self, q, gripper) -> None:
        self._arm(self.ghost, self._gd, q, gripper)

    def set_turn(self, turn: int, text: str, overhead=None, wrist=None) -> None:
        self.turn_md.content = f"**Turn {turn}**\n\n{text}"
        if overhead is not None:
            self.image.image = overhead
        if wrist is not None:
            self.wrist.image = wrist

    def show_program(self, segs, next_idx: int, results: dict[int, str], state_q, state_gripper) -> None:
        for h in self._seg_handles:
            h.remove()
        self._seg_handles = []
        sc = self.server.scene
        rows = []
        for i, s in enumerate(segs):
            state = "done" if i < next_idx else ("**NEXT**" if i == next_idx else "")
            flag = " BLOCKED" if s.blockers and i >= next_idx else ""
            rows.append(f"| {i + 1} | {s.label} | {state}{flag} | {results.get(i, '')} |")
            if i >= next_idx and s.ee_path is not None:
                active = i == next_idx
                self._seg_handles += [
                    sc.add_spline_catmull_rom(f"plan/{i}/path", s.ee_path, color=(0, 200, 80) if active
                                              else (120, 200, 140), thickness=0.004 if active else 0.0015),
                    sc.add_label(f"plan/{i}/label", str(i + 1), position=s.ee_path[-1]),
                ]
        self.program_md.content = "| # | segment | state | result |\n|---|---|---|---|\n" + "\n".join(rows)
        nxt = segs[next_idx] if next_idx < len(segs) else None
        if nxt is not None and nxt.q_end is not None and not nxt.blockers:
            self._arm(self.planned, self._pd, nxt.q_end, state_gripper)
            self.planned.set_visible(True)
        elif nxt is not None and nxt.gripper is not None:
            self._arm(self.planned, self._pd, state_q, nxt.gripper)
            self.planned.set_visible(True)
        else:
            self.planned.set_visible(False)

    # -- approval -----------------------------------------------------------------
    def describe(self, seg) -> str:
        md = [f"### Next: {seg.label}"]
        if seg.ee_path is not None:
            a, b = seg.ee_path[0], seg.ee_path[-1]
            md.append("| pinch | x | y | z |\n|---|---|---|---|\n"
                      f"| now | {a[0]:+.3f} | {a[1]:+.3f} | {a[2]:+.3f} |\n"
                      f"| end | {b[0]:+.3f} | {b[1]:+.3f} | {b[2]:+.3f} |")
            md.append(f"{len(seg.q_path) - 1} IK waypoints over **{seg.durations.sum():.1f} s**, "
                      f"largest joint change **{seg.max_joint_delta:.3f} rad**")
            md.append("| joint | now | end | delta |\n|---|---|---|---|\n" + "\n".join(
                f"| j{i + 1} | {x:+.3f} | {y:+.3f} | {y - x:+.3f} |"
                for i, (x, y) in enumerate(zip(seg.q_path[0], seg.q_path[-1]))))
        elif seg.kind == "close":
            md.append("No arm motion: close the jaws slowly, stop on a gripper-effort spike.")
        elif seg.kind == "gripper":
            md.append(f"No arm motion: gripper -> {seg.gripper:.1f} ({'open' if seg.gripper > 0.5 else 'closed'}).")
        md += [f"**BLOCKED:** {b}" for b in seg.blockers]
        md += [f"**Warning:** {w}" for w in seg.warnings]
        return "\n\n".join(md)

    def wait_decision(self, seg) -> tuple[str, str]:
        """Block until the operator approves or rejects. Auto-approve only
        passes segments with no warnings, no blockers and no override need."""
        self.next_md.content = self.describe(seg)
        if (self.auto.value and not seg.blockers and not seg.warnings and not seg.needs_override
                and not self.stop_requested):
            return "approve", ""
        self._pending = seg
        self._decision.clear()
        self.override.value = False
        self._sync_approve()
        self.reject_btn.disabled = False
        self._decision.wait()
        self._pending = None
        self.wrist_cloud = None
        self.show_wrist_cloud.on_update(lambda _: self.wrist_cloud is not None and setattr(
            self.wrist_cloud, "visible", self.show_wrist_cloud.value))
        self.approve_btn.disabled = self.reject_btn.disabled = True
        self.override.visible = False
        return self._choice

    def _sync_approve(self) -> None:
        seg = self._pending
        if seg is None:
            return
        self.override.visible = seg.needs_override and not seg.blockers
        self.approve_btn.disabled = bool(seg.blockers) or (seg.needs_override and not self.override.value)

    def _decide(self, choice: str) -> None:
        if self._pending is None:
            return
        if choice == "approve" and self.approve_btn.disabled:
            return
        self._choice = (choice, self.feedback.value.strip())
        self.feedback.value = ""
        self._decision.set()

    def _stop(self) -> None:
        self.stop_requested = True
        self.status_md.content = "**Stop requested** -- the current plan is rejected; the agent stops."
        self._choice = ("reject", "Operator stopped the session.")
        self._decision.set()
