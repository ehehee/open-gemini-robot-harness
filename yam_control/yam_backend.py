"""Real YAM control via i2rt (github.com/i2rt-robotics/i2rt, not on PyPI --
`pip install "i2rt @ git+https://github.com/i2rt-robotics/i2rt.git"`).

Unlike an earlier version of this file, every call here has been verified
against the actual installed i2rt 1.1.2 on this machine (and cross-checked
against bosch-crate/yam-bimanual-calib's own working scripts, which drive
these same physical arms): `get_yam_robot()`'s signature, `MotorChainRobot`'s
get_observations/command_joint_pos/move_joints/close, and FK/IK against the
combined arm+gripper MJCF `get_yam_robot` builds (at `robot.xml_path`).

Two things were NOT obvious from the docs and were only caught by actually
instantiating a sim robot and testing:

1. `i2rt.robots.kinematics.Kinematics.fk/ik` take a q matching the MODEL's
   full `nq`, not just the 6 arm joints -- the combined arm+gripper model has
   nq=8 (6 arm joints + 2 gripper mechanism joints), so a bare 6-vector raises
   a broadcast error. This is exactly why bosch-crate/yam-bimanual-calib's own
   solve_handeye.py avoids this class for FK and does raw MuJoCo instead; IK
   here works around it by padding q to 8 with zeros for the (unused, since
   the gripper site's pose doesn't depend on them) gripper mechanism joints
   and discarding those two entries from the result.
2. FK here reads the `grasp_site` SITE (defined directly on the gripper body,
   so its pose is independent of gripper finger-joint values), not the
   `gripper` BODY bosch-crate's own FK helper uses for their hand-eye solve --
   a different, equally valid choice of end-effector frame. That difference
   doesn't affect any of their already-solved extrinsics (those are
   base-to-camera, with the gripper frame already composed out), but do not
   mix T_base_gripper from here with anything computed against their
   "gripper"-body convention without accounting for the fixed offset between
   the two frames.

Pass sim=True to run against i2rt's own MuJoCo-backed SimRobot instead of real
hardware -- same Robot protocol, no CAN bus required. That's how this file was
tested without a physical arm.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from camera.extrinsics import Extrinsics
from yam_control.robot_interface import CartesianPlan, YamRobotInterface

_EE_SITE = "grasp_site"

# Gripper guard (see _gripper_guard and _check_gripper_reading). On 2026-10-15
# the left gripper's motor zero shifted by ~pi, so with the recorded limits it
# read 2.11 on the 0..1 scale; an open command then drove it into the closed
# hard stop at saturated effort (8.8) until motor 7 tripped "over temperature"
# ~10 s later. Normal grasps on this rig peak at ~0.9 and holding sits ~0.1.
GRIPPER_READ_RANGE = (-0.1, 1.1)  # a reading outside this: limits/zero are wrong, refuse to command
GRIPPER_STALL_EFFORT = 4.0        # |gripper_eff| above this ...
GRIPPER_STALL_S = 1.0             # ... for this long, with the jaws not moving -> stop pushing
GRIPPER_STALL_MOVE = 0.01         # "not moving": less than this (0..1 units) over the window

# Joint stall guard (see _joint_guard). On 2026-10-15 the left arm was blocked
# (by the other arm / the clamp) and kept pushing: joints 4-5 sat ~0.7 rad off
# their setpoints and motor 4 went 50 -> 101 C in ~23 s, tripping the driver
# and dropping both arms. Gravity sag on this rig is ~0.02-0.05 rad per joint.
JOINT_STALL_ERR_RAD = 0.15        # a joint this far from its setpoint ...
JOINT_STALL_VEL = 0.05            # ... and slower than this (rad/s) ...
JOINT_STALL_S = 0.75              # ... for this long -> blocked: stop pushing


class RealYamRobot(YamRobotInterface):
    def __init__(
        self,
        channel: str,
        arm_type: str = "yam",
        gripper_type: str = "crank_4310",
        sim: bool = False,
        ee_mass: float | None = None,
        gripper_limits: tuple[float, float] | None = None,
        joint_offsets_rad: list[float] | None = None,
    ):
        self.channel = channel
        self.arm_type = arm_type
        self.gripper_type = gripper_type
        self.sim = sim
        self.ee_mass = ee_mass
        # crank_4310 has needs_calibration=true and no stored limits, so every
        # connect otherwise runs detect_gripper_limits(), which drives the jaws
        # shut then open at 0.5 Nm to rediscover the range. Pass the already
        # known [closed, open] to skip that -- faster, and it leaves whatever
        # is between the jaws alone. Re-detect after any gripper reassembly.
        self.gripper_limits = gripper_limits
        # q_model = q_encoder + offset. i2rt's get_robot.py hardcodes
        # motor_offsets = [0.0]*n, so the encoder-zero vs model-zero mismatch
        # is NOT corrected anywhere in the driver -- any FK/IK we do against
        # raw readings inherits it. On this rig that is right j5 = +1.46 deg.
        # See camera.rig_calibration.arm_joint_offsets_rad. Zeros = no-op, so
        # sim and un-fitted arms behave exactly as before.
        self.joint_offsets = np.zeros(6) if joint_offsets_rad is None else np.asarray(
            joint_offsets_rad, dtype=float)
        if self.joint_offsets.shape != (6,):
            raise ValueError(f"joint_offsets_rad must have 6 entries, got {self.joint_offsets.shape}")
        self.num_joints = 6  # arm joints; gripper is handled separately
        self._robot = None
        # Last setpoint actually sent (6 arm joints + gripper), or None before
        # the first command. Under gravity the arm sits BELOW its setpoint (on
        # the real right arm, 8-30 mm at the gripper), so anything that
        # re-derives a command from the MEASURED position walks the setpoint
        # down a little every time -- and restarting interpolation from the
        # measured position each waypoint makes the setpoint saw back and
        # forth, which is what made streamed moves visibly shake.
        self._cmd: np.ndarray | None = None
        self._mj_xml = None
        self._mj_model = None
        self._mj_data = None
        self._site_id = None
        self._qadr = None
        self._kinematics = None
        self.gripper_guard_trips = 0
        self.joint_guard_trips = 0
        self.last_block: str | None = None
        self._abort_stream = False

    def connect(self) -> None:
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType

        self._robot = get_yam_robot(
            channel=self.channel,
            arm_type=ArmType.from_string_name(self.arm_type),
            gripper_type=GripperType.from_string_name(self.gripper_type),
            sim=self.sim,
            ee_mass=self.ee_mass,
            gripper_limits_override=(
                np.asarray(self.gripper_limits) if self.gripper_limits is not None else None
            ),
        )
        self._load_fk_model(self._robot.xml_path)
        if "gripper_eff" in self._robot.get_observations():
            import threading

            threading.Thread(target=self._gripper_guard, args=(self._robot,), daemon=True,
                             name=f"gripper-guard-{self.channel}").start()
        import threading

        threading.Thread(target=self._joint_guard, args=(self._robot,), daemon=True,
                         name=f"joint-guard-{self.channel}").start()

    def _joint_guard(self, robot, hz: float = 50.0) -> None:
        """Stall guard for the six arm joints, next to i2rt's control loop
        (the child process under ProcessYamRobot). A joint far from its
        setpoint, not moving, for JOINT_STALL_S = the arm is pushing against
        something. Then: abort any streamed path and set the setpoint to
        where the arm IS, so the motors stop fighting the obstacle. The move
        then ends short of its target and Primitives reports it as blocked."""
        import time

        since = None
        while self._robot is robot:
            try:
                obs = robot.get_observations()
                cmd = self._cmd
            except Exception:  # noqa: BLE001 -- shutting down
                return
            if cmd is not None:
                q = np.asarray(obs["joint_pos"], dtype=float)
                dq = np.abs(np.asarray(obs["joint_vel"], dtype=float))
                err = np.abs(cmd[:6] - q)
                stuck = (err > JOINT_STALL_ERR_RAD) & (dq < JOINT_STALL_VEL)
                now = time.monotonic()
                if not stuck.any():
                    since = None
                elif since is None:
                    since = now
                elif now - since >= JOINT_STALL_S:
                    j = int(np.argmax(np.where(stuck, err, 0.0)))
                    self.joint_guard_trips += 1
                    self.last_block = (f"joint {j + 1} stuck {err[j]:.2f} rad from its setpoint for "
                                       f"{now - since:.1f} s")
                    print(f"[joint guard {self.channel}] {self.last_block} -- blocked; stopped pushing "
                          f"(setpoint -> measured)", flush=True)
                    self._abort_stream = True
                    try:
                        self._cmd = np.append(q, cmd[6])
                        robot.command_joint_pos(self._cmd)
                    except Exception:  # noqa: BLE001
                        return
                    since = None
            time.sleep(1.0 / hz)

    def _gripper_guard(self, robot, hz: float = 20.0) -> None:
        """Stall guard, in the same process as i2rt's control loop (the child
        process under ProcessYamRobot), so it works however busy the agent's
        process is. If the gripper pushes harder than GRIPPER_STALL_EFFORT for
        GRIPPER_STALL_S without moving, move its setpoint to where the jaws
        ARE: the position error, and with it the effort, drops to ~0. A normal
        grasp never gets near the threshold, so this only fires on a fight with
        a hard stop (wrong limits) or a jam."""
        import time

        since, start_pos = None, None
        while self._robot is robot:
            try:
                obs = robot.get_observations()
                eff, pos = abs(float(obs["gripper_eff"][0])), float(obs["gripper_pos"][0])
            except Exception:  # noqa: BLE001 -- shutting down underneath us
                return
            now = time.monotonic()
            if eff < GRIPPER_STALL_EFFORT:
                since = None
            elif since is None:
                since, start_pos = now, pos
            elif now - since >= GRIPPER_STALL_S and abs(pos - start_pos) < GRIPPER_STALL_MOVE:
                self.gripper_guard_trips += 1
                print(f"[gripper guard {self.channel}] effort {eff:.1f} for {now - since:.1f} s with the jaws "
                      f"not moving (reads {pos:.3f}) -- stopped pushing (setpoint -> measured)", flush=True)
                try:
                    q = self._cmd[:6] if self._cmd is not None else np.asarray(obs["joint_pos"], dtype=float)
                    self._cmd = np.append(q, pos)
                    robot.command_joint_pos(self._cmd)
                except Exception:  # noqa: BLE001
                    return
                since = None
            time.sleep(1.0 / hz)

    def _check_gripper_reading(self) -> float:
        """Refuse to command a gripper whose reading is outside 0..1 (plus a
        margin): the recorded limits no longer match the motor, so any target
        is somewhere other than where it looks -- possibly through a hard stop."""
        g = self.get_gripper()
        lo, hi = GRIPPER_READ_RANGE
        if not lo <= g <= hi:
            raise RuntimeError(
                f"gripper on {self.channel} reads {g:.3f}, outside 0..1: its recorded gripper_limits no longer "
                f"match the motor (zero shifted?). Not commanding it. Re-measure the limits "
                f"(run_human_check.py --real --measure-left-gripper for the left arm) and update config/rig.yaml.")
        return g

    def _load_fk_model(self, xml_path: str) -> None:
        import mujoco

        from i2rt.robots.kinematics import Kinematics

        self._mj_xml = Path(xml_path).read_text()  # i2rt writes it to a temp file
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_data = mujoco.MjData(self._mj_model)
        self._site_id = self._mj_model.site(_EE_SITE).id
        self._qadr = [self._mj_model.joint(f"joint{i}").qposadr[0] for i in range(1, 7)]
        self._kinematics = Kinematics(xml_path, site_name=_EE_SITE)

    def disconnect(self) -> None:
        """Release the arm (all torques -> zero) without the shutdown race.

        MotorChainRobot.close() stops its server thread, then immediately calls
        DMChainCanInterface.close(), which sets running=False and shuts the bus
        down with nothing joining the control loop in between -- so that loop is
        mid-iteration when the socket fd goes to -1, and dies printing "DM Error
        in control loop". Harmless (torques are already zero by then) but it
        looks exactly like a real motor fault in a log.

        Stopping the halves in an order that has a gap in it avoids the whole
        thing. Lifted from yam-bimanual-calib/scripts/arm_homing.py::_release,
        which documents the race in full; getattr-guarded so a driver that
        doesn't look like this one just gets closed.

        THE ARM GOES LIMP HERE. Park it somewhere it can hold unpowered first.
        """
        import time

        robot = self._robot
        self._robot = None
        if robot is None:
            return
        chain = getattr(robot, "motor_chain", None)
        thread = getattr(robot, "_server_thread", None)
        event = getattr(robot, "_stop_event", None)
        if chain is not None and thread is not None and event is not None:
            try:
                event.set()
                thread.join(timeout=5.0)
                # MultiDMChainCanInterface fans out over several buses; a plain
                # DMChainCanInterface is its own single element.
                for one in getattr(chain, "interfaces", None) or [chain]:
                    one.running = False
                time.sleep(0.2)  # let the loop drain; i2rt's period is ~7 ms
            except Exception as e:  # noqa: BLE001
                print(f"  (could not quiesce the control loop: {e})")
        robot.close()

    @property
    def mj_model(self):
        """For visualization (examples/06) -- the same MuJoCo model FK/IK use."""
        return self._mj_model

    @property
    def mj_xml(self) -> str:
        """The MJCF text mj_model was compiled from (absolute mesh paths), for
        building larger MuJoCo scenes around this arm (gap_guard.scene)."""
        return self._mj_xml

    @property
    def mj_data(self):
        """Reflects whatever q was last passed through _fk(); call get_ee_pose()
        first if you want this to match the robot's current joint positions."""
        return self._mj_data

    def to_model_q(self, q6_encoder) -> np.ndarray:
        """Encoder reading -> MuJoCo model angles (q_model = q_encoder + delta)."""
        return np.asarray(q6_encoder, dtype=float) + self.joint_offsets

    def to_encoder_q(self, q6_model) -> np.ndarray:
        """MuJoCo model angles -> what you actually command the motors."""
        return np.asarray(q6_model, dtype=float) - self.joint_offsets

    def _fk(self, q6: np.ndarray) -> Extrinsics:
        """q6 is an ENCODER reading; offsets are applied on the way into FK."""
        import mujoco

        self._mj_data.qpos[:] = 0.0
        for adr, val in zip(self._qadr, self.to_model_q(q6)):
            self._mj_data.qpos[adr] = val
        mujoco.mj_kinematics(self._mj_model, self._mj_data)
        T = np.eye(4)
        T[:3, :3] = self._mj_data.site_xmat[self._site_id].reshape(3, 3)
        T[:3, 3] = self._mj_data.site_xpos[self._site_id]
        return Extrinsics(T)

    def get_ee_pose(self) -> Extrinsics:
        """T_base_{grasp_site} -- see module docstring for how this frame
        relates (or doesn't) to bosch-crate's own "gripper"-body convention."""
        self._require_connected()
        q6 = np.array(self.get_joint_positions())
        return self._fk(q6)

    def get_joint_positions(self) -> list[float]:
        self._require_connected()
        return self._robot.get_observations()["joint_pos"].tolist()

    def get_gripper(self) -> float:
        """0.0 = fully closed, 1.0 = fully open (i2rt's own normalized convention)."""
        self._require_connected()
        return float(self._robot.get_observations()["gripper_pos"][0])

    def get_state(self) -> dict[str, list[float]]:
        """One snapshot for continuous logging (agent/state_logger.py):
        measured joint_pos/vel/eff, gripper_pos/vel/eff (when there is a
        gripper) and the setpoint (cmd_q, cmd_gripper). A single i2rt read, so
        over ProcessYamRobot it's one round trip instead of five. Plain lists,
        so it pickles across the process boundary."""
        self._require_connected()
        obs = self._robot.get_observations()
        out = {k: np.asarray(obs[k], dtype=float).ravel().tolist()
               for k in ("joint_pos", "joint_vel", "joint_eff", "gripper_pos", "gripper_vel", "gripper_eff")
               if k in obs}
        cmd = self._cmd
        out["cmd_q"] = (cmd[:6] if cmd is not None else np.asarray(obs["joint_pos"], dtype=float)).tolist()
        out["cmd_gripper"] = [float(cmd[6]) if cmd is not None else
                              float(np.asarray(obs.get("gripper_pos", [np.nan]))[0])]
        js = getattr(self._robot, "_joint_state", None)  # MotorChainRobot fills these every control tick
        for key in ("temp_mos", "temp_rotor"):
            t = getattr(js, key, None)
            if t is not None:
                out[key] = np.asarray(t, dtype=float).ravel().tolist()
        out["gripper_guard_trips"] = [float(self.gripper_guard_trips)]
        out["joint_guard_trips"] = [float(self.joint_guard_trips)]
        return out

    def is_healthy(self) -> tuple[bool, str]:
        """(ok, reason). False once i2rt's motor chain has stopped -- e.g. a
        motor dropped off the bus ("loss communication"), after which its
        control thread exits and the arm is NOT under control, though
        get_observations() keeps returning the last (stale) state. Seen on the
        real rig: motor 2 timed out at connect and everything above carried on
        as if connected."""
        if self._robot is None:
            return False, "not connected"
        chain = getattr(self._robot, "motor_chain", None)
        if chain is not None:
            for one in getattr(chain, "interfaces", None) or [chain]:
                if getattr(one, "running", True) is False:
                    return False, f"motor chain on {self.channel} stopped (a motor lost communication?)"
        thread = getattr(self._robot, "_server_thread", None)
        if thread is not None and not thread.is_alive():
            return False, f"i2rt robot server thread on {self.channel} exited"
        return True, "ok"

    def _send(self, q7: np.ndarray) -> None:
        ok, why = self.is_healthy()
        if not ok:
            raise RuntimeError(f"refusing to command the arm: {why}")
        self._cmd = np.asarray(q7, dtype=float).copy()
        self._robot.command_joint_pos(self._cmd)

    def commanded_joint_positions(self) -> np.ndarray:
        """The arm setpoint (encoder space). Equal to the measured position
        until the first command; plan from THIS, not the measured position,
        so sag under gravity doesn't accumulate into the setpoint."""
        self._require_connected()
        return self._cmd[:6].copy() if self._cmd is not None else np.array(self.get_joint_positions())

    def commanded_gripper(self) -> float:
        self._require_connected()
        return float(self._cmd[6]) if self._cmd is not None else self.get_gripper()

    def stream_joint_path(self, q_path: np.ndarray, durations: np.ndarray, rate_hz: float = 200.0) -> None:
        """Follow a waypoint path as ONE continuous trajectory: piecewise
        linear through the waypoints, with a minimum-jerk time scaling over
        the whole thing (smooth start and stop, no stop-and-go at each
        waypoint), sent at a fixed rate on wall-clock deadlines. Starts from
        the current SETPOINT; q_path[0] should be that setpoint (it is when
        the plan was made from commanded_joint_positions()). The gripper
        setpoint is held unchanged throughout."""
        import time

        self._require_connected()
        q_path = np.asarray(q_path, dtype=float)
        q_path = np.vstack([self.commanded_joint_positions(), q_path[1:]])
        knots = np.concatenate([[0.0], np.cumsum(durations)])
        total = float(knots[-1])
        if total <= 0:
            self._send(np.append(q_path[-1], self.commanded_gripper()))
            return
        n = max(1, int(total * rate_hz))
        t0 = time.monotonic()
        self._abort_stream = False
        for i in range(1, n + 1):
            if self._abort_stream:  # the joint guard found the arm blocked: stop where it is
                return
            tau = i / n
            v = total * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)  # min-jerk progress
            k = min(int(np.searchsorted(knots, v, side="right")) - 1, len(q_path) - 2)
            a = (v - knots[k]) / max(knots[k + 1] - knots[k], 1e-9)
            # Re-read the gripper setpoint every tick: the stall guard may move it mid-path.
            self._send(np.append((1 - a) * q_path[k] + a * q_path[k + 1], self.commanded_gripper()))
            delay = t0 + i / rate_hz * (total * rate_hz / n) - time.monotonic()
            if delay > 0:
                time.sleep(delay)

    def move_to_joint_positions(self, positions: list[float], duration_s: float | None = None) -> None:
        """Interpolates locally rather than using MotorChainRobot's own
        move_joints(), since that method isn't part of the shared Robot
        protocol -- i2rt's SimRobot (used for sim=True) doesn't have it.
        From the current SETPOINT, min-jerk timed; gripper setpoint held."""
        self._require_connected()
        assert len(positions) == self.num_joints
        target = np.asarray(positions, dtype=float)
        if not duration_s:
            self._send(np.append(target, self.commanded_gripper()))
            return
        self.stream_joint_path(np.stack([self.commanded_joint_positions(), target]), np.array([duration_s]))

    def move_cartesian_delta(self, dx: float, dy: float, dz: float) -> None:
        self.execute_plan(self.plan_cartesian_delta(dx, dy, dz))

    def plan_cartesian_delta(self, dx: float, dy: float, dz: float) -> CartesianPlan:  # noqa: D401
        """Solve the IK for a cartesian delta WITHOUT commanding the robot.

        Split out from move_cartesian_delta so a caller can render/approve the
        result first (examples/08 previews it in viser); reading joint
        positions and running IK touch no motors. Raises RuntimeError if IK
        does not converge -- i.e. a plan that fails here never reaches the arm.
        """
        self._require_connected()
        q6 = np.array(self.get_joint_positions())
        target = self._fk(q6).matrix.copy()
        target[:3, 3] += np.array([dx, dy, dz])
        return self._plan_to_pose_matrix(target, q6)

    def move_to_position(self, x: float, y: float, z: float) -> None:
        """Absolute position move, keeping the arm's CURRENT orientation fixed.

        Deliberately position-only: we haven't derived/verified a yaw/pitch ->
        grasp_site-orientation mapping for this gripper (grasp_site's axes
        aren't documented, and getting that wrong on real hardware is a bad
        place to guess). Every example built on this moves straight down from
        whatever orientation the arm is already holding when it starts.
        """
        self._require_connected()
        q6 = np.array(self.get_joint_positions())
        target = self._fk(q6).matrix.copy()
        target[:3, 3] = np.array([x, y, z])
        self.execute_plan(self._plan_to_pose_matrix(target, q6))

    def plan_to_pose(self, target: np.ndarray, q6_start=None, attempts: int = 3) -> CartesianPlan:
        """Solve IK for an absolute 4x4 T_base_{grasp_site} WITHOUT moving.

        Unlike move_to_position this sets orientation too -- the caller owns
        that choice (examples/11 builds top-down grasp orientations from the
        gripper geometry in the model). q6_start (encoder space) defaults to
        the current reading; pass a previous plan's target_joints to chain
        waypoints and preview a whole sequence before any of it runs.

        i2rt's IK takes ~200 small steps per call, which isn't always enough
        for a large reorientation, so a non-converged result is fed back in as
        the next seed up to `attempts` times before giving up.

        NOTE: i2rt's IK is called without joint limits -- check the result
        with joints_outside_limits() before executing it.
        """
        self._require_connected()
        q6 = self.commanded_joint_positions() if q6_start is None else np.asarray(q6_start, dtype=float)
        return self._plan_to_pose_matrix(np.asarray(target, dtype=float), q6, attempts=attempts)

    def joints_outside_limits(self, q6_encoder) -> list[str]:
        """Human-readable list of joints whose MODEL angle is outside the
        MJCF's own joint range; empty if all are within range."""
        bad = []
        for i, q in enumerate(self.to_model_q(q6_encoder), start=1):
            lo, hi = self._mj_model.joint(f"joint{i}").range
            if not lo - 1e-6 <= q <= hi + 1e-6:
                bad.append(f"joint{i}={q:+.3f} rad not in [{lo:+.3f}, {hi:+.3f}]")
        return bad

    def execute_plan(self, plan: CartesianPlan, duration_s: float | None = None) -> None:
        """Command a plan produced by plan_cartesian_delta(). THIS MOVES THE ARM.

        Pass duration_s to interpolate there over that long instead of sending
        a single setpoint -- worth doing on real hardware for anything but the
        smallest step.
        """
        self.move_to_joint_positions(plan.target_joints.tolist(), duration_s=duration_s)

    def _plan_to_pose_matrix(self, target: np.ndarray, q6_init: np.ndarray, attempts: int = 1) -> CartesianPlan:
        # Kinematics.ik needs a full-nq q (model has 2 extra gripper-mechanism
        # joints beyond the 6 arm ones); grasp_site's pose doesn't depend on
        # them, so padding with zeros and discarding them from the result is
        # exact, not an approximation -- see module docstring point 1.
        # IK runs entirely in MODEL space; q6_init arrives as an encoder
        # reading, and the solution comes back as model angles that have to be
        # converted before they can be commanded.
        init_q = np.append(self.to_model_q(q6_init), np.zeros(self._mj_model.nq - 6))
        for _ in range(attempts):
            ok, q_new = self._kinematics.ik(target, site_name=_EE_SITE, init_q=init_q)
            if ok:
                break
            init_q = np.asarray(q_new).copy()
        if not ok:
            raise RuntimeError(f"IK did not converge for target=\n{target}\nfrom q={q6_init}")
        q6_new = self.to_encoder_q(np.asarray(q_new[:6]))
        return CartesianPlan(
            start_joints=np.asarray(q6_init, dtype=float).copy(),
            target_joints=q6_new,
            start_pose=self._fk(q6_init),
            target_pose=self._fk(q6_new),
            requested_xyz=target[:3, 3].copy(),
            joint_offsets=self.joint_offsets.copy(),
        )

    def close_gripper_until_contact(
        self, effort_threshold: float = 1.5, max_duration_s: float = 3.0, poll_hz: float = 20.0
    ) -> bool:
        """Closes the gripper gradually, watching gripper_eff (motor effort)
        for a contact spike instead of closing blind to a fixed position --
        this is the real substitute for the wrist-cam visual jaw check we
        don't have a camera for (see the discussion that led here). Returns
        True if contact was detected, False if it closed fully (or timed out)
        without ever seeing one -- on i2rt's bare sim model there's no scene
        object to contact, so this will legitimately return False there.
        """
        import time

        self._require_connected()
        self._check_gripper_reading()
        steps = max(1, int(max_duration_s * poll_hz))
        # Ramp down from where the jaws ARE, not from fully open: re-closing an
        # already-closed gripper must never open it first (that drops what
        # it's holding -- nearly happened on the real rig with the brush).
        start_open = self.commanded_gripper()
        for i in range(1, steps + 1):
            target_open = max(0.0, start_open * (1.0 - i / steps))
            self.set_gripper(target_open)
            time.sleep(1.0 / poll_hz)
            effort = abs(self._robot.get_observations()["gripper_eff"][0])
            if effort > effort_threshold:
                return True
        return False

    def set_gripper(self, open_fraction: float) -> None:
        """Arm held at its SETPOINT (not its measured, sagged position)."""
        self._require_connected()
        self._check_gripper_reading()
        self._send(np.append(self.commanded_joint_positions(), max(0.0, min(1.0, open_fraction))))

    def home(self) -> None:
        self._require_connected()
        self.move_to_joint_positions([0.0] * self.num_joints, duration_s=2.0)

    def _require_connected(self) -> None:
        if self._robot is None:
            raise RuntimeError("Call connect() before sending commands")
