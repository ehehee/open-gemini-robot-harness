"""RoboSimStudio as the simulator: the harness runs unchanged above the
robot driver, against a physically simulated scene.

What this provides, for any RoboSimStudio live scene (a `scene.py` that
binds `scene`, an `rss.Scene` with a two-arm YAM embodiment whose arms are
named `right` and `left`, e.g. sim/scenes/scene_7010_v2):

  RssSim        builds the scene with two cameras added -- the overhead at
                the calibrated T_rightbase_overhead (mounted on the right arm's
                base body, so it lands right in any scene layout) and the
                right-wrist D405 at T_right_wrist_gripper_cam -- and owns it
                in one thread (run_forever), stepping physics continuously.
                Every other thread goes through it (call/render/snapshot).
  SimYamRobot   yam_control's RealYamRobot with a sim arm under it instead of
                i2rt's CAN driver: the same streaming, setpoint tracking,
                gripper close-until-contact and stall guards as on the rig.
  SimOverhead / SimWristCamera
                the camera interfaces agent/ uses, rendered from the sim.
  write_config  config/ for this scene: the real rig's camera calibration,
                plus T_leftbase_rightbase read off the scene's arm bases.

Time: the sim runs slower than real time, so the threads that drive arms
(streaming, guards, Primitives.execute) run on SIM time -- time.sleep and
time.monotonic are switched per thread (sim_clocked). Gemini, viser and the
HTTP server stay on the wall clock.

Joints are model space (the scene's arms are the i2rt model; the gripper
body agrees with i2rt FK to 0.04 mm), so SimYamRobot runs with zero joint
offsets. The scene's own TCP is the pad centre, 7 mm behind i2rt's
grasp_site, which is the pinch point every harness coordinate refers to.
"""
from __future__ import annotations

import importlib.util
import queue
import shutil
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

SIDES = ("right", "left")
OVERHEAD, WRIST = "overhead", "right_wrist"
# Right wrist D405 colour stream, 848x480: intrinsics estimated on the rig
# (multiview fit, 1.4 px RMS) -- its factory intrinsics were not archived.
WRIST_K = (449.4862868782468, 448.78525538256036, 423.34910312173764, 251.38234986874284)
# The gripper's measured opening is reported as RealYamRobot.get_gripper()
# does: 0 shut .. 1 open, from the finger joints over their travel.
# gripper_eff stands in for the motor's effort: proportional to the finger
# force the position servo is applying (how far the jaws lag the command),
# saturating where the scene's actuator does. Scaled so the harness's
# close-until-contact threshold (1.5) trips at ~24 N per finger -- the
# squeeze the scene's validated handover grasps with -- and the stall guard
# (4.0) never does.
FINGER_KE_N_PER_M = 4000.0
FINGER_EFFORT_LIMIT_N = 25.0
EFFORT_PER_NEWTON = 1.5 / 23.7

# -- sim time for the threads that drive the arms --------------------------------
_tl = threading.local()
_real_sleep, _real_monotonic = time.sleep, time.monotonic
_HUB: "RssSim | None" = None


def _sleep(dt):
    if getattr(_tl, "sim", False):
        _HUB.sleep(dt)
    else:
        _real_sleep(dt)


def _monotonic():
    return _HUB.now() if getattr(_tl, "sim", False) else _real_monotonic()


def sim_clocked(fn):
    """Run fn with this thread on sim time."""
    def wrapper(*a, **k):
        prev = getattr(_tl, "sim", False)
        _tl.sim = True
        try:
            return fn(*a, **k)
        finally:
            _tl.sim = prev
    wrapper.__wrapped__ = fn
    return wrapper


def _pose(T: np.ndarray) -> tuple:
    return (*T[:3, 3], *Rotation.from_matrix(T[:3, :3]).as_quat())


def _matrix(pose7) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(pose7[3:7]).as_matrix()
    T[:3, 3] = pose7[:3]
    return T


def load_scene(path: Path):
    """Run a live scene.py and return its module (scene, BUILD, ...)."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"rss_scene_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


class RssSim:
    def __init__(self, scene_path: Path, config_dir: Path, control_mode: str = "joint_position"):
        from robosimstudio.specs import CameraSpec

        from agent.wrist_view import R_OURS_THEIRS

        self.mod = load_scene(scene_path)
        scene = self.mod.scene
        overhead = yaml.safe_load((config_dir / "extrinsics" / "T_rightbase_overhead.yaml").read_text())
        k = overhead["metadata"]["intrinsics"]
        T_gc = np.array(yaml.safe_load((config_dir / "extrinsics" / "T_right_wrist_gripper_cam.yaml").read_text())
                        ["matrix"])
        scene.add(CameraSpec(name=OVERHEAD, width=k["width"], height=k["height"], fx=k["fx"], fy=k["fy"],
                             cx=k["cx"], cy=k["cy"], parent_frame="right_base",
                             xform=_pose(np.array(overhead["matrix"]))))
        fx, fy, cx, cy = WRIST_K
        scene.add(CameraSpec(name=WRIST, width=848, height=480, fx=fx, fy=fy, cx=cx, cy=cy,
                             parent_frame="right_gripper", xform=_pose(R_OURS_THEIRS @ T_gc)))
        self.sim = sim = scene.build(control_mode=control_mode, **getattr(self.mod, "BUILD", {}))
        sim.run(1)
        sim.sync()
        self.dt = float(sim.pipeline.control_dt)
        self.order = list(sim.scene.embodiment.iks)  # the action's arm order
        if sorted(self.order) != sorted(SIDES):
            raise RuntimeError(f"scene arms are {self.order}; the harness needs 'right' and 'left'")
        self.coords = {s: list(sim.scene.handles.arms[s].arm_coords) for s in SIDES}
        labels, start = list(sim.model.joint_label), sim.model.joint_q_start.numpy()
        upper = sim.model.joint_limit_upper.numpy()
        self.fingers = {s: [int(start[i]) for i, l in enumerate(labels)
                            if l.split("/")[-1] in (f"{s}_joint7", f"{s}_joint8")] for s in SIDES}
        self.finger_travel = {s: float(upper[self.fingers[s][0]]) for s in SIDES}
        body_labels, bq = list(sim.model.body_label), sim.state.body_q.numpy()
        self.T_world_base = {s: _matrix(bq[next(i for i, l in enumerate(body_labels)
                                                if l.split("/")[-1] == f"{s}_base")]) for s in SIDES}
        self.cams = {c.name: c for c in sim.cameras}
        self.cmd = self._hold_command()
        self.t = 0.0
        self.cond = threading.Condition()
        self._req: queue.Queue = queue.Queue()
        self._cmd_lock = threading.Lock()
        self._renderers: dict = {}
        self.snap: dict = {}
        self.video: SimVideo | None = None
        self.caption = ""  # drawn under the video frames
        self._steps = 0
        self._publish()

    def _hold_command(self) -> np.ndarray:
        """Joint targets = where the arms are; grippers held at their opening."""
        jq = self.sim.state.joint_q.numpy().astype(float)
        out = []
        for s in self.order:
            out.extend(jq[self.coords[s]])
            out.append(1.0 - self._open(jq, s))
        return np.array(out)

    def _open(self, jq, s) -> float:
        return float(np.clip(np.mean([jq[c] for c in self.fingers[s]]) / self.finger_travel[s], 0.0, 1.0))

    def write_config(self, src: Path, dst: Path) -> Path:
        """config/ for this scene: src's rig.yaml and camera calibration, and
        T_leftbase_rightbase from where the scene put the two arm bases."""
        (dst / "extrinsics").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "rig.yaml", dst / "rig.yaml")
        for f in ("T_rightbase_overhead", "T_right_wrist_gripper_cam"):
            shutil.copy2(src / "extrinsics" / f"{f}.yaml", dst / "extrinsics" / f"{f}.yaml")
        T_lr = np.linalg.inv(self.T_world_base["left"]) @ self.T_world_base["right"]
        (dst / "extrinsics" / "T_leftbase_rightbase.yaml").write_text(yaml.safe_dump(
            {"name": "T_leftbase_rightbase", "matrix": T_lr.tolist(),
             "metadata": {"source": "RoboSimStudio scene arm bases"}}))
        return dst

    def gripper_parts(self, out_dir: Path) -> dict:
        """What the scene bolts to each gripper beyond i2rt's arm model (the
        wrist camera assemblies, a wrist cable), as gap_guard CollisionScene
        gripper parts: [(mesh file, T_gripperbody_part)] per arm. Without them
        the overhead camera sees those parts as obstacles sitting on the
        gripper. Read from the scene's own robot MJCF, whose {side}_gripper
        body is i2rt's `gripper` body; primitives are written out as meshes."""
        import xml.etree.ElementTree as ET

        import trimesh

        from camera.extrinsics import Extrinsics
        from robosimstudio.robots.registry import EMBODIMENTS

        mjcf = Path(EMBODIMENTS[self.sim.request.embodiment].asset.path)
        root = ET.parse(mjcf).getroot()
        meshdir = mjcf.parent / (root.find("compiler").get("meshdir", "") if root.find("compiler") is not None else "")
        files = {m.get("name") or Path(m.get("file")).stem: (m.get("file"), m.get("scale"))
                 for m in root.iter("mesh")}

        def local(el) -> np.ndarray:
            T = np.eye(4)
            if el.get("quat"):
                w, x, y, z = (float(v) for v in el.get("quat").split())
                T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
            if el.get("pos"):
                T[:3, 3] = [float(v) for v in el.get("pos").split()]
            return T

        out_dir.mkdir(parents=True, exist_ok=True)
        parts = {}
        for side in SIDES:
            grip = root.find(f".//body[@name='{side}_gripper']")
            found = [(np.eye(4), g) for g in grip.findall("geom") if "usb_lead" in g.get("name", "")]
            for body in grip.iter("body"):
                if body.get("name", "").endswith("_camera_assembly"):
                    found += [(local(body), g) for g in body.findall("geom")]
            parts[side] = []
            for T_body, g in found:
                kind, name = g.get("type", "sphere"), g.get("name")
                size = [float(v) for v in (g.get("size") or "0").split()]
                T = T_body @ local(g)
                if kind == "mesh":
                    f, scale = files[g.get("mesh")]
                    path = Path(f) if Path(f).is_absolute() else meshdir / f
                    if scale:
                        m = trimesh.load(path, force="mesh")
                        m.apply_scale([float(v) for v in scale.split()])
                        path = out_dir / f"{name}.obj"
                        m.export(path)
                elif kind == "box":
                    path = out_dir / f"{name}.obj"
                    trimesh.creation.box(extents=[2 * v for v in size]).export(path)
                elif kind == "cylinder":
                    path = out_dir / f"{name}.obj"
                    trimesh.creation.cylinder(radius=size[0], height=2 * size[1]).export(path)
                elif kind == "capsule" and g.get("fromto"):
                    a, b = np.array([float(v) for v in g.get("fromto").split()]).reshape(2, 3)
                    d = b - a
                    m = trimesh.creation.capsule(radius=size[0], height=float(np.linalg.norm(d)))
                    z = np.array([0.0, 0.0, 1.0])
                    R, _ = Rotation.align_vectors([d / np.linalg.norm(d)], [z])
                    Tc = np.eye(4)
                    Tc[:3, :3], Tc[:3, 3] = R.as_matrix(), (a + b) / 2
                    path, T = out_dir / f"{name}.obj", T_body @ Tc
                    m.export(path)
                else:
                    continue
                parts[side].append((str(path), Extrinsics(T)))
        return parts

    # -- other threads
    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        with self.cond:
            target = self.t + max(dt, 0.0)
            while self.t < target - 1e-9:
                self.cond.wait(1.0)

    def call(self, fn):
        box, done = {}, threading.Event()
        self._req.put((fn, box, done))
        done.wait()
        if "err" in box:
            raise box["err"]
        return box["v"]

    def set_cmd(self, side: str, q6, closed: float) -> None:
        i = self.order.index(side) * 7
        with self._cmd_lock:
            self.cmd[i:i + 6] = q6
            self.cmd[i + 6] = closed

    def snapshot(self, side: str) -> dict:
        with self.cond:
            return self.snap[side]

    def render(self, name: str) -> dict:
        return self.call(lambda: self.frame(name))

    def start_video(self, path: Path, fps: float = 10.0, main: str = "phone") -> None:
        """Record from now on, in sim time: the `main` camera large, the
        overhead and right-wrist views beside it, `caption` underneath."""
        self.call(lambda: setattr(self, "video", SimVideo(self, path, fps, main)))

    def stop_video(self) -> Path | None:
        def stop():
            v, self.video = self.video, None
            if v is not None:
                v.close()
            return None if v is None else v.path
        return self.call(stop)

    def object_pose(self, name: str) -> np.ndarray:
        """World pose (x, y, z, qx, qy, qz, qw) of a named scene body."""
        return self.call(lambda: np.asarray(self.sim[name].pose, dtype=float))

    # -- the sim's own thread
    def frame(self, name: str) -> dict:
        """rgb + depth of one camera, one renderer per camera, built once."""
        r = self._renderers.get(name)
        if r is None:
            from robosimstudio.solvers.render import choose_renderer

            backend, _ = choose_renderer(device=self.sim.model.device)
            r = self._renderers[name] = self.sim.make_renderer(backend, cameras=[self.cams[name]],
                                                               channels=["rgb", "depth"])
            r.render(self.sim.state)  # the first frame only builds the stage
        self.sim.sync()
        f = r.render(self.sim.state)[name]
        return {c: np.asarray(f[c]) for c in ("rgb", "depth") if c in f}

    def _publish(self) -> None:
        jq = self.sim.state.joint_q.numpy().astype(float)
        jqd = self.sim.state.joint_qd.numpy().astype(float)
        snap = {}
        for s in SIDES:
            open_ = self._open(jq, s)
            cmd_open = 1.0 - float(self.cmd[self.order.index(s) * 7 + 6])
            force = min(max(0.0, open_ - cmd_open) * self.finger_travel[s] * FINGER_KE_N_PER_M, FINGER_EFFORT_LIMIT_N)
            snap[s] = {"q": jq[self.coords[s]].copy(), "qd": jqd[self.coords[s]].copy(), "open": open_,
                       "eff": EFFORT_PER_NEWTON * force}
        self.snap = snap

    def run_forever(self, stop: threading.Event) -> None:
        global _HUB
        _HUB = self
        while not stop.is_set():
            while True:
                try:
                    fn, box, done = self._req.get_nowait()
                except queue.Empty:
                    break
                try:
                    box["v"] = fn()
                except Exception as e:  # noqa: BLE001 -- handed back to the caller
                    box["err"] = e
                done.set()
            with self._cmd_lock:
                action = self.cmd.astype(np.float32)
            self.sim.step(action, mode="joint_position")
            self._steps += 1
            if self.video is not None:
                self.video.capture(self._steps)
            with self.cond:
                self.t += self.dt
                self._publish()
                self.cond.notify_all()


class SimVideo:
    """An mp4 of the sim in sim time (one frame every 1/fps sim seconds)."""

    def __init__(self, hub: RssSim, path: Path, fps: float, main: str):
        import imageio
        from PIL import ImageFont

        self.hub, self.path = hub, Path(path)
        self.every = max(1, int(round(1.0 / (fps * hub.dt))))
        self.writer = imageio.get_writer(str(self.path), fps=fps, codec="libx264", quality=7, macro_block_size=8)
        self.main = main if main in hub.cams else OVERHEAD
        self.side = [n for n in (OVERHEAD, WRIST) if n in hub.cams and n != self.main]
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 24)
            self.small = ImageFont.truetype("DejaVuSans.ttf", 18)
        except OSError:
            self.font = self.small = ImageFont.load_default()

    def _rgb(self, name, size):
        from PIL import Image

        return Image.fromarray(np.asarray(self.hub.frame(name)["rgb"])[..., :3].astype(np.uint8)).resize(size)

    def capture(self, step: int) -> None:
        if step % self.every:
            return
        from PIL import Image, ImageDraw

        canvas = Image.new("RGB", (1920, 1080), (18, 18, 20))
        canvas.paste(self._rgb(self.main, (1280, 720)), (0, 0))
        for k, name in enumerate(self.side[:2]):
            canvas.paste(self._rgb(name, (640, 360)), (1280, 360 * k))
        d = ImageDraw.Draw(canvas)
        for k, name in enumerate([self.main] + self.side[:2]):
            d.text((12, 10) if k == 0 else (1292, 10 + 360 * (k - 1)), name, fill=(255, 230, 0), font=self.small)
        d.text((24, 736), f"sim t = {self.hub.t:6.1f} s", fill=(170, 170, 170), font=self.small)
        for i, line in enumerate(self.hub.caption.splitlines()[:10]):
            d.text((24, 770 + 30 * i), line, fill=(235, 235, 235), font=self.font)
        self.writer.append_data(np.asarray(canvas))

    def close(self) -> None:
        self.writer.close()


def start(scene_path: Path, config_src: Path, config_dst: Path) -> tuple[RssSim, threading.Event]:
    """Build the scene in its own thread (RoboSimStudio state stays on the
    thread that made it), switch arm-driving threads to sim time, and return
    once it is stepping."""
    out, ready, stop = {}, threading.Event(), threading.Event()

    def owner():
        try:
            hub = out["hub"] = RssSim(scene_path, config_src)
            hub.write_config(config_src, config_dst)
        except BaseException as e:  # noqa: BLE001
            out["err"] = e
            ready.set()
            return
        global _HUB
        _HUB = hub
        time.sleep, time.monotonic = _sleep, _monotonic
        ready.set()
        hub.run_forever(stop)

    threading.Thread(target=owner, daemon=True, name="robosimstudio").start()
    ready.wait()
    if "err" in out:
        raise out["err"]
    sys.modules.setdefault("pyrealsense2", _pyrealsense_shim())
    return out["hub"], stop


# -- the driver and cameras the harness talks to ---------------------------------------
class RssArm:
    """The part of i2rt's Robot protocol RealYamRobot uses, on one sim arm."""

    def __init__(self, hub: RssSim, side: str):
        self.hub, self.side = hub, side

    def get_observations(self) -> dict:
        s = self.hub.snapshot(self.side)
        return {"joint_pos": s["q"].copy(), "joint_vel": s["qd"].copy(), "joint_eff": np.zeros(6),
                "gripper_pos": np.array([s["open"]]), "gripper_vel": np.zeros(1),
                "gripper_eff": np.array([s["eff"]])}

    def command_joint_pos(self, q7) -> None:
        q7 = np.asarray(q7, dtype=float)
        self.hub.set_cmd(self.side, q7[:6], 1.0 - float(np.clip(q7[6], 0.0, 1.0)))

    def close(self) -> None:
        pass


def sim_robot_class():
    from yam_control.yam_backend import RealYamRobot

    class SimYamRobot(RealYamRobot):
        """RealYamRobot, unchanged above the driver: connect() hands it a sim
        arm and the i2rt arm model, and its stall guards run on sim time."""

        def __init__(self, hub: RssSim, side: str, xml_path: str):
            super().__init__(channel=f"rss-{side}", gripper_type="crank_4310", sim=True)
            self._hub, self._side, self._xml_path = hub, side, xml_path

        def connect(self) -> None:
            self._robot = RssArm(self._hub, self._side)
            self._load_fk_model(self._xml_path)
            for guard in (self._gripper_guard, self._joint_guard):
                threading.Thread(target=sim_clocked(guard), args=(self._robot,), daemon=True,
                                 name=f"{guard.__name__}-{self._side}").start()

        def disconnect(self) -> None:
            self._robot = None

    return SimYamRobot


def i2rt_arm_xml(out: Path) -> Path:
    """The i2rt arm + crank_4310 MJCF the harness plans with, written to out."""
    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

    xml = combine_arm_and_gripper_xml(ArmType.from_string_name("yam"), GripperType.from_string_name("crank_4310"))
    xml = Path(xml).read_text() if Path(str(xml)).exists() else str(xml)
    out.write_text(xml)
    return out


class SimOverhead:
    def __init__(self, hub: RssSim):
        from camera.zed_overhead import OverheadFrame

        self.hub, self.Frame, self.cam = hub, OverheadFrame, hub.cams[OVERHEAD]

    def grab(self):
        f = self.hub.render(OVERHEAD)
        return self.Frame(rgb=np.asarray(f["rgb"])[..., :3].astype(np.uint8).copy(),
                          depth=np.asarray(f["depth"], dtype=np.float32).copy(),
                          fx=self.cam.fx, fy=self.cam.fy, cx=self.cam.cx, cy=self.cam.cy)

    def grab_rgb(self):
        return self.grab().rgb

    def close(self):
        pass


class SimWristCamera:
    """camera/d405_wrist.WristCamera's interface on the sim's wrist camera."""

    def __init__(self, hub: RssSim):
        self.hub = hub
        c = hub.cams[WRIST]
        self.intrinsics = types.SimpleNamespace(width=c.width, height=c.height, fx=c.fx, fy=c.fy,
                                                ppx=c.cx, ppy=c.cy, coeffs=[0.0] * 5, model=None)

    def grab_rgbd(self):
        f = self.hub.render(WRIST)
        return (np.asarray(f["rgb"])[..., :3].astype(np.uint8).copy(),
                np.asarray(f["depth"], dtype=np.float32).copy())

    def grab(self):
        return self.grab_rgbd()[0]

    grab_rgb = grab

    def close(self):
        pass


def _pyrealsense_shim():
    """WristView projects through pyrealsense2; the sim camera is a plain pinhole."""
    rs = types.ModuleType("pyrealsense2")

    def project(intr, p):
        return [intr.fx * p[0] / p[2] + intr.ppx, intr.fy * p[1] / p[2] + intr.ppy]

    rs.rs2_project_point_to_pixel = project
    return rs
