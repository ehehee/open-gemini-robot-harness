"""One overhead ZED-M capture kept as an ORGANIZED depth map (H x W) next to
its RGB frame, rather than a flattened point list.

Deprojection needs the organized form: Gemini points at
a pixel in the RGB frame, and the 3D grasp point is read back from the depth
at that same pixel. A flattened cloud has lost that pixel correspondence.

Conventions: left camera,
COORDINATE_SYSTEM.IMAGE (+X right, +Y down, +Z out of the lens) -- the frame
cv2.solvePnP used to produce T_{side}base_overhead.yaml.

`OverheadFrame.save/load` write a single .npz, so a capture can be replayed
(--replay) to iterate on Gemini/planning without the camera.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class OverheadFrame:
    rgb: np.ndarray  # (H, W, 3) uint8, left camera
    depth: np.ndarray  # (H, W) float32 meters along +Z; 0 / nan = invalid
    fx: float
    fy: float
    cx: float
    cy: float

    def depth_at(self, u: float, v: float, window: int = 7) -> float | None:
        """Median valid depth in a window around (u, v) -- one pixel of learned
        or classical stereo is too noisy to trust on its own, and the exact
        pixel Gemini picks may sit on an edge with no valid depth."""
        H, W = self.depth.shape
        r = window // 2
        ui, vi = int(round(u)), int(round(v))
        patch = self.depth[max(0, vi - r): min(H, vi + r + 1), max(0, ui - r): min(W, ui + r + 1)]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        return float(np.median(valid)) if valid.size else None

    def deproject(self, u: float, v: float, z: float) -> np.ndarray:
        return np.array([(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z])

    def points(self, stride: int = 2, max_depth: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
        """(xyz Nx3 in the camera frame, rgb Nx3 uint8) for rendering."""
        H, W = self.depth.shape
        us, vs = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
        d = self.depth[::stride, ::stride]
        valid = np.isfinite(d) & (d > 0) & (d <= max_depth)
        z = d[valid]
        xyz = np.stack([(us[valid] - self.cx) * z / self.fx, (vs[valid] - self.cy) * z / self.fy, z], axis=1)
        return xyz, self.rgb[::stride, ::stride][valid]

    def save(self, path: Path) -> None:
        np.savez_compressed(path, rgb=self.rgb, depth=self.depth,
                            K=np.array([self.fx, self.fy, self.cx, self.cy]))

    @classmethod
    def load(cls, path: Path) -> "OverheadFrame":
        data = np.load(path)
        fx, fy, cx, cy = data["K"].tolist()
        return cls(rgb=data["rgb"], depth=data["depth"].astype(np.float32), fx=fx, fy=fy, cx=cx, cy=cy)


def grab_overhead_frame(method: str = "tri_stereo", tri_stereo_variant: str = "c64") -> OverheadFrame:
    """method="tri_stereo" uses third_party/raiden's learned depth (denser on
    this rig's low-texture table, see examples/10's docstring for the
    numbers); anything else is an sl.DEPTH_MODE name, e.g. "ULTRA"."""
    import pyzed.sl as sl

    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.NONE if method == "tri_stereo" else getattr(sl.DEPTH_MODE, method.upper())
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")
    try:
        cal = zed.get_camera_information().camera_configuration.calibration_parameters
        fx, fy, cx, cy = cal.left_cam.fx, cal.left_cam.fy, cal.left_cam.cx, cal.left_cam.cy
        baseline = abs(cal.get_camera_baseline())

        runtime = sl.RuntimeParameters()
        for _ in range(20):  # let auto-exposure settle
            zed.grab(runtime)

        img_mat = sl.Mat()
        zed.retrieve_image(img_mat, sl.VIEW.LEFT)
        left_bgr = img_mat.get_data()[:, :, :3].copy()

        if method == "tri_stereo":
            from camera.tri_stereo_depth import TriStereoDepth

            right_mat = sl.Mat()
            zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
            right_bgr = right_mat.get_data()[:, :, :3].copy()
            depth = TriStereoDepth(tri_stereo_variant).predict(left_bgr, right_bgr, fx, baseline)
        else:
            depth_mat = sl.Mat()
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            depth = depth_mat.get_data().copy()
        depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
        return OverheadFrame(rgb=left_bgr[:, :, ::-1].copy(), depth=depth, fx=fx, fy=fy, cx=cx, cy=cy)
    finally:
        zed.close()


class OverheadCamera:
    """Keeps the ZED open (and the TRI Stereo model loaded) between grabs --
    the agent observes every turn, and re-opening the camera +
    re-loading the model each time costs several seconds per turn.

    Every grab is checked. A failed grab used to leave the previous image in
    place and return it as if new -- on 2026-10-15 the ZED dropped off USB
    mid-run and Gemini planned seven turns from one frozen frame. Now a
    failure raises, and after REOPEN_AFTER failures in a row the camera is
    closed and re-opened (at most every REOPEN_MIN_INTERVAL_S): the SDK never
    re-attaches a camera that dropped out by itself, so once it is back on USB
    this is what brings the run's overhead view back."""

    REOPEN_AFTER = 3
    REOPEN_MIN_INTERVAL_S = 5.0

    def __init__(self, method: str = "tri_stereo", tri_stereo_variant: str = "c64"):
        import pyzed.sl as sl

        self._sl = sl
        self.method = method
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 30
        init.depth_mode = sl.DEPTH_MODE.NONE if method == "tri_stereo" else getattr(sl.DEPTH_MODE, method.upper())
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        self._init = init
        self._fails, self._last_reopen, self.reopens = 0, 0.0, 0
        self.zed = sl.Camera()
        status = self.zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {status}")
        cal = self.zed.get_camera_information().camera_configuration.calibration_parameters
        self.K = (cal.left_cam.fx, cal.left_cam.fy, cal.left_cam.cx, cal.left_cam.cy)
        self.baseline = abs(cal.get_camera_baseline())
        self._tri = None
        if method == "tri_stereo":
            from camera.tri_stereo_depth import TriStereoDepth

            self._tri = TriStereoDepth(tri_stereo_variant)
        self._runtime = sl.RuntimeParameters()
        # grab() (agent /obs) and grab_rgb() (agent/cam_recorder.py) run on
        # different threads; the SDK's grab/retrieve pair must not interleave.
        self._lock = threading.Lock()
        for _ in range(20):  # let auto-exposure settle
            self.zed.grab(self._runtime)

    def _grab_locked(self) -> bool:
        """One zed.grab(), reopening the camera after repeated failures. Call
        with self._lock held. False = no new image (the caller must not
        retrieve one: it would be the old frame)."""
        sl = self._sl
        if self.zed.grab(self._runtime) == sl.ERROR_CODE.SUCCESS:
            self._fails = 0
            return True
        self._fails += 1
        if (self._fails >= self.REOPEN_AFTER
                and time.monotonic() - self._last_reopen >= self.REOPEN_MIN_INTERVAL_S
                and self._reopen_locked()):
            if self.zed.grab(self._runtime) == sl.ERROR_CODE.SUCCESS:
                self._fails = 0
                return True
        return False

    def _reopen_locked(self) -> bool:
        sl = self._sl
        self._last_reopen = time.monotonic()
        print(f"[zed] {self._fails} failed grabs in a row -- closing and re-opening the camera", flush=True)
        try:
            self.zed.close()
        except Exception:  # noqa: BLE001
            pass
        self.zed = sl.Camera()
        status = self.zed.open(self._init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"[zed] re-open failed: {status} (is it back on USB?); will retry", flush=True)
            return False
        self.reopens += 1
        for _ in range(10):  # auto-exposure
            self.zed.grab(self._runtime)
        print(f"[zed] re-opened (#{self.reopens})", flush=True)
        return True

    def grab(self) -> OverheadFrame:
        sl = self._sl
        with self._lock:
            for _ in range(3):  # drop buffered frames so the image is current
                if not self._grab_locked():
                    raise RuntimeError(f"overhead camera unavailable: ZED grab failed "
                                       f"({self._fails} in a row, {self.reopens} re-opens so far)")
            mat = sl.Mat()
            self.zed.retrieve_image(mat, sl.VIEW.LEFT)
            left_bgr = mat.get_data()[:, :, :3].copy()
            right_bgr = None
            if self._tri is not None:
                self.zed.retrieve_image(mat, sl.VIEW.RIGHT)
                right_bgr = mat.get_data()[:, :, :3].copy()
            else:
                self.zed.retrieve_measure(mat, sl.MEASURE.DEPTH)
                depth = mat.get_data().copy()
        if right_bgr is not None:  # the model runs outside the lock: recording keeps going meanwhile
            depth = self._tri.predict(left_bgr, right_bgr, self.K[0], self.baseline)
        depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
        fx, fy, cx, cy = self.K
        return OverheadFrame(rgb=left_bgr[:, :, ::-1].copy(), depth=depth, fx=fx, fy=fy, cx=cx, cy=cy)

    def grab_rgb(self) -> np.ndarray:
        """Left colour image only, no depth: cheap enough to call a few
        times a second for video (agent/cam_recorder.py)."""
        sl = self._sl
        with self._lock:
            # A failed grab leaves the last image in place; returning it would
            # put a frozen frame in the video under a fresh timestamp.
            if not self._grab_locked():
                raise RuntimeError(f"ZED grab failed ({self._fails} in a row)")
            mat = sl.Mat()
            self.zed.retrieve_image(mat, sl.VIEW.LEFT)
            return mat.get_data()[:, :, 2::-1].copy()

    def close(self) -> None:
        self.zed.close()
