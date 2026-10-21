"""The wrist D405, placed in the world by the eye-in-hand calibration.

    T_base_cam = FK_body("gripper")(q_encoder + joint_offsets) @ T_gripper_cam

That is exactly the convention yam-bimanual-calib's solve_handeye.py used to
produce T_gripper_cam: FK at the MuJoCo body `gripper` (the wrist flange, not
this project's grasp_site), recomputed with fk.joint_offsets applied. Same
combined model, same offsets as RealYamRobot.to_model_q, so the chain closes.

Gives Gemini the post's original setup -- a 2 cm world grid and a pinch
reticle drawn on the WRIST image -- and gives viser a wrist point cloud.
Projection goes through librealsense's own rs2_project_point_to_pixel, so the
D405's distortion model is applied exactly.
"""
from __future__ import annotations

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from camera.extrinsics import Extrinsics
from viz.viser_robot import set_arm_qpos

# yam-bimanual-calib evaluates FK on its own model file, whose `gripper` body
# sits at the SAME point as this project's but is rotated 180 deg about
# (x+y)/sqrt(2): x<->y swapped, z flipped. Measured from the calibration run's
# own stored T_base_gripper vs our FK at the same joint angles: identical over
# all 52 captures of run_052 to 0.0005 deg / 0.003 mm, i.e. a modelling
# convention, not noise. T_gripper_cam is expressed in THEIR frame, so:
#   T_base_cam = T_base_gripper(ours) @ R_OURS_THEIRS @ T_gripper_cam
# Without it the camera came out looking backwards and no grid was drawn.
R_OURS_THEIRS = np.array([[0.0, 1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, -1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])


class WristView:
    def __init__(self, robot, T_gripper_cam: Extrinsics, intrinsics, T_world_base: np.ndarray | None = None):
        import pyrealsense2 as rs

        self._rs = rs
        self.robot, self.T_gc = robot, T_gripper_cam.matrix
        self.intr = intrinsics
        self.T_wb = np.eye(4) if T_world_base is None else T_world_base
        self.model = robot.mj_model
        self.data = mujoco.MjData(self.model)
        self.body = self.model.body("gripper").id

    def T_world_cam(self, q_encoder) -> np.ndarray:
        set_arm_qpos(self.model, self.data, self.robot.to_model_q(q_encoder))
        T = np.eye(4)
        T[:3, :3] = self.data.xmat[self.body].reshape(3, 3)
        T[:3, 3] = self.data.xpos[self.body]
        return self.T_wb @ T @ R_OURS_THEIRS @ self.T_gc

    def _project(self, pts_cam: np.ndarray):
        """(N,3) camera-frame points -> (N,2) pixels; NaN where behind the lens
        or well outside the field of view.

        The D405's inverse Brown-Conrady polynomial is only fitted inside the
        image (normalized radius ~1.12 at the corner). Past ~1.7 it turns
        back: r ~2.05 lands inside the image again and r ~2.22 lands exactly
        on the principal point. A close-up wrist view puts most of the grid
        there, which drew a starburst at the image centre and stray lines.
        """
        out = np.full((len(pts_cam), 2), np.nan)
        for i, p in enumerate(pts_cam):
            if p[2] > 0.02 and np.hypot(p[0] / p[2], p[1] / p[2]) < 1.3:
                out[i] = self._rs.rs2_project_point_to_pixel(self.intr, [float(p[0]), float(p[1]), float(p[2])])
        return out

    def render(self, rgb: np.ndarray, q_encoder, ee_T: np.ndarray, gripper_open: float, grid_z: float,
               ws) -> Image.Image:
        """Same overlay as the overhead view: 2 cm grid at grid_z (10 cm
        intersections labelled), green pinch circle + jaw bar, orange toward the
        wrist, magenta drop to the grid."""
        img = Image.fromarray(rgb).convert("RGB")
        draw = ImageDraw.Draw(img, "RGBA")
        try:
            font = ImageFont.load_default(size=14)
        except TypeError:
            font = ImageFont.load_default()
        W, H = img.size
        T_cw = np.linalg.inv(self.T_world_cam(q_encoder))

        def to_cam(P):
            P = np.atleast_2d(np.asarray(P, float))
            return P @ T_cw[:3, :3].T + T_cw[:3, 3]

        def polyline(P, fill, width):
            px = self._project(to_cam(P))
            for a, b in zip(px[:-1], px[1:]):
                if np.isfinite(a).all() and np.isfinite(b).all() and \
                        max(abs(a[0]), abs(b[0])) < 4 * W and max(abs(a[1]), abs(b[1])) < 4 * H:
                    draw.line([tuple(a), tuple(b)], fill=fill, width=width)

        xs = np.arange(np.ceil(ws.x[0] / 0.02) * 0.02, ws.x[1] + 1e-9, 0.02)
        ys = np.arange(np.ceil(ws.y[0] / 0.02) * 0.02, ws.y[1] + 1e-9, 0.02)
        major = lambda v: abs(v / 0.1 - round(v / 0.1)) < 1e-6  # noqa: E731
        # Lines are drawn as 1 cm pieces so a line passing behind the camera
        # (normal for a close-up wrist view) doesn't get one wild long segment.
        ys_fine = np.arange(ys[0], ys[-1] + 1e-9, 0.01)
        xs_fine = np.arange(xs[0], xs[-1] + 1e-9, 0.01)
        for x in xs:
            seg = np.stack([np.full(len(ys_fine), x), ys_fine, np.full(len(ys_fine), grid_z)], 1)
            polyline(seg, (255, 255, 255, 150 if major(x) else 60), 2 if major(x) else 1)
        for y in ys:
            seg = np.stack([xs_fine, np.full(len(xs_fine), y), np.full(len(xs_fine), grid_z)], 1)
            polyline(seg, (255, 255, 255, 150 if major(y) else 60), 2 if major(y) else 1)
        for x in filter(major, xs):
            for y in filter(major, ys):
                p = self._project(to_cam([x, y, grid_z]))[0]
                if np.isfinite(p).all() and 0 <= p[0] < W - 40 and 0 <= p[1] < H - 30:
                    draw.text((p[0] + 3, p[1] + 1), f"x{x:.1f}", fill=(255, 255, 0, 255), font=font,
                              stroke_width=2, stroke_fill=(0, 0, 0, 255))
                    draw.text((p[0] + 3, p[1] + 14), f"y{y:+.1f}", fill=(0, 255, 255, 255), font=font,
                              stroke_width=2, stroke_fill=(0, 0, 0, 255))

        pinch, R = ee_T[:3, 3], ee_T[:3, :3]
        jaw_half = 0.02 + 0.02 * gripper_open
        polyline(np.linspace(pinch, [pinch[0], pinch[1], grid_z], 8), (255, 0, 255, 200), 2)
        polyline(np.stack([pinch - R[:, 1] * jaw_half, pinch + R[:, 1] * jaw_half]), (0, 255, 0, 255), 3)
        for pt, r, col in ((pinch, 7, (0, 255, 0, 255)), ([pinch[0], pinch[1], grid_z], 4, (255, 0, 255, 255))):
            p = self._project(to_cam(pt))[0]
            if np.isfinite(p).all():
                draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], outline=col, width=2)
        return img

    def cloud(self, rgb: np.ndarray, depth: np.ndarray, q_encoder, stride: int = 3, max_depth: float = 0.6):
        """(xyz world Nx3, rgb Nx3) from the aligned D405 depth."""
        H, W = depth.shape
        vs, us = np.mgrid[0:H:stride, 0:W:stride]
        d = depth[::stride, ::stride]
        ok = (d > 0.03) & (d < max_depth)
        u, v, z = us[ok].astype(float), vs[ok].astype(float), d[ok]
        # Undistorted pinhole deprojection is plenty for display (D405 colour
        # distortion is a few px at the edges); rs2_deproject per point would
        # be ~50k Python calls a frame.
        i = self.intr
        pts = np.stack([(u - i.ppx) / i.fx * z, (v - i.ppy) / i.fy * z, z], 1)
        T = self.T_world_cam(q_encoder)
        return pts @ T[:3, :3].T + T[:3, 3], rgb[::stride, ::stride][ok]
