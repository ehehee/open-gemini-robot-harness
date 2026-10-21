"""Any plain USB (UVC) camera -- e.g. a webcam on a stand looking at the
workspace from the side -- as a video source for agent/cam_recorder.py.

The overhead camera can't see height (is one gripper above the other, how
deep is the object in the jaws), and during a handover both grippers block
its view of the object. A side view shows exactly that.

OpenCV buffers frames, so read() returns stale ones when called at the
recorder's slower rate; a thread reads continuously and grab_rgb() returns
the newest frame.
"""
from __future__ import annotations

import threading
import time

import numpy as np


class UvcCamera:
    def __init__(self, device: str | int, width: int = 1280, height: int = 720, fps: int = 30):
        """device: an index ("2" / 2) or a path ("/dev/video6")."""
        import cv2

        if isinstance(device, str) and device.isdigit():
            device = int(device)
        self.device = device
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {device!r}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"uvc-{device}")
        self._thread.start()
        t0 = time.time()
        while self._frame is None and time.time() - t0 < 5.0:
            time.sleep(0.05)
        if self._frame is None:
            self.close()
            raise RuntimeError(f"camera {device!r} opened but delivered no frames")

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, bgr = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = bgr

    def grab_rgb(self) -> np.ndarray:
        with self._lock:
            if self._frame is None:
                raise RuntimeError(f"no frame from camera {self.device!r}")
            return self._frame[:, :, ::-1].copy()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.cap.release()
