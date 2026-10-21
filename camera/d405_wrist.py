"""Persistent stream from the wrist-mounted D405 (serial: config/rig.yaml).

Colour always; depth too (aligned to colour) when `depth=True`. Its pose on
the arm is the eye-in-hand calibration T_gripper_cam
(config/extrinsics/T_right_wrist_gripper_cam.yaml, solved 2026-10-14);
agent/wrist_view.py turns that plus the live arm pose into a world-anchored
overlay and point cloud.
"""
from __future__ import annotations

import threading

import numpy as np

D405_SERIAL = "218722271150"


class WristCamera:
    def __init__(self, serial: str = D405_SERIAL, depth: bool = True, width: int = 848, height: int = 480):
        import pyrealsense2 as rs

        self._rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        self.depth_enabled = depth
        if depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        profile = self.pipeline.start(config)
        # Intrinsics of THIS stream (the calibration ran at 1280x720; the
        # extrinsic is resolution-independent, the intrinsics are not).
        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale() if depth else None
        self._align = rs.align(rs.stream.color) if depth else None
        # The agent's /obs and agent/cam_recorder.py both pull frames, from
        # different threads; one consumer of the pipeline at a time.
        self._lock = threading.Lock()
        for _ in range(30):  # auto-exposure
            self.pipeline.wait_for_frames()

    def _frames(self):
        with self._lock:
            for _ in range(3):  # drop buffered frames
                frames = self.pipeline.wait_for_frames()
            return self._align.process(frames) if self._align is not None else frames

    def grab_rgb(self) -> np.ndarray:
        """One current colour frame, without depth alignment (for video)."""
        with self._lock:
            frames = self.pipeline.wait_for_frames()
            return np.asanyarray(frames.get_color_frame().get_data())[:, :, ::-1].copy()

    def grab(self) -> np.ndarray:
        return np.asanyarray(self._frames().get_color_frame().get_data())[:, :, ::-1].copy()

    def grab_rgbd(self) -> tuple[np.ndarray, np.ndarray | None]:
        """(rgb uint8 HxWx3, depth float32 HxW meters aligned to colour, 0 = invalid)."""
        f = self._frames()
        rgb = np.asanyarray(f.get_color_frame().get_data())[:, :, ::-1].copy()
        if not self.depth_enabled:
            return rgb, None
        d = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * self.depth_scale
        return rgb, d

    def close(self) -> None:
        self.pipeline.stop()
