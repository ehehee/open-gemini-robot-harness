"""What Gemini sees each turn, and the pixel -> world mapping behind
robot.deproject.

The post overlays a 2 cm world grid and a 3D pinch-center reticle on a
CALIBRATED WRIST camera. This harness draws it on the overhead ZED (solved
extrinsic T_rightbase_overhead), which sees both arms; the right wrist D405
gets the same overlay in agent/wrist_view.py when T_right_wrist_gripper_cam
is present. Deprojection goes through the overhead only.

look_at(x, y, z) doesn't move the arm here (the overhead camera is fixed); it
re-centres and zooms the overhead view on that point. u, v given to deproject
are normalized [0, 1] coordinates of the LAST image sent, crop included.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from camera.extrinsics import Extrinsics
from camera.zed_overhead import OverheadFrame


@dataclass
class View:
    center: np.ndarray | None = None  # world point the crop is centred on; None = full frame
    zoom: float = 1.0
    grid_z: float = 0.0


@dataclass
class Workspace:
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    def contains(self, p, tol: float = 1e-6) -> bool:
        return all(lo - tol <= v <= hi + tol for v, (lo, hi) in zip(p, (self.x, self.y, self.z)))

    def as_dict(self) -> dict:
        return {k: [round(float(v), 3) for v in r] for k, r in (("x", self.x), ("y", self.y), ("z", self.z))}


def estimate_table_z(cloud_base: np.ndarray, ws: Workspace) -> float:
    """Most common height of the cloud over the workspace footprint -- the
    table is by far the largest flat thing there."""
    m = ((cloud_base[:, 0] > ws.x[0]) & (cloud_base[:, 0] < ws.x[1])
         & (cloud_base[:, 1] > ws.y[0]) & (cloud_base[:, 1] < ws.y[1]))
    z = cloud_base[m, 2]
    if z.size < 500:
        raise RuntimeError("too few depth points over the workspace to find the table")
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.004, 0.004))
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2)


class Observer:
    def __init__(self, T_base_cam: Extrinsics, workspace: Workspace):
        self.T_base_cam = T_base_cam
        self.T_cam_base = T_base_cam.inverse()
        self.ws = workspace
        self.frame: OverheadFrame | None = None
        self.view = View()
        self._crop = (0.0, 0.0, 1.0, 1.0)  # x0, y0, w, h in full-frame pixels, of the last image sent
        self.table_z: float | None = None

    # -- geometry -------------------------------------------------------------
    def project(self, p_base) -> tuple[float, float] | None:
        f = self.frame
        pc = self.T_cam_base.apply(np.asarray(p_base, float))
        if pc[2] <= 1e-3:
            return None
        return pc[0] / pc[2] * f.fx + f.cx, pc[1] / pc[2] * f.fy + f.cy

    def _full_px(self, u: float, v: float) -> tuple[float, float]:
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            raise ValueError(f"u, v must be normalized to [0, 1] of the last image, got ({u}, {v})")
        x0, y0, w, h = self._crop
        return x0 + u * w, y0 + v * h

    def deproject(self, u: float, v: float, z_plane: float) -> tuple[float, float]:
        """Intersect the camera ray through (u, v) with the plane z = z_plane."""
        pu, pv = self._full_px(u, v)
        f = self.frame
        ray = self.T_base_cam.R @ np.array([(pu - f.cx) / f.fx, (pv - f.cy) / f.fy, 1.0])
        o = self.T_base_cam.t
        if abs(ray[2]) < 1e-6:
            raise ValueError("camera ray is parallel to that plane")
        p = o + (z_plane - o[2]) / ray[2] * ray
        return float(p[0]), float(p[1])

    def deproject_depth(self, u: float, v: float) -> tuple[float, float, float]:
        """Extension beyond the post: the overhead has stereo depth, so give
        the 3D surface point under (u, v) directly."""
        pu, pv = self._full_px(u, v)
        d = self.frame.depth_at(pu, pv)
        if d is None:
            raise ValueError(f"no valid depth near ({u:.3f}, {v:.3f}); use deproject with a z_plane")
        p = self.T_base_cam.apply(self.frame.deproject(pu, pv, d))
        return float(p[0]), float(p[1]), float(p[2])

    def cloud_base(self, stride: int = 2) -> tuple[np.ndarray, np.ndarray]:
        xyz, rgb = self.frame.points(stride=stride)
        return (self.T_base_cam.R @ xyz.T).T + self.T_base_cam.t, rgb

    # -- image ------------------------------------------------------------------
    def set_frame(self, frame: OverheadFrame) -> None:
        self.frame = frame
        if self.table_z is None:
            self.table_z = estimate_table_z(self.cloud_base(stride=4)[0], self.ws)
            self.view.grid_z = self.table_z

    def render(self, ee_T, gripper_open: float | None = None) -> Image.Image:
        """Overhead RGB + 2 cm grid at view.grid_z + pinch reticle(s), cropped
        to the view. ee_T is one 4x4 pose (with gripper_open), or a list of
        (pose, gripper_open, label) -- one reticle per arm, labelled."""
        reticles = [(ee_T, gripper_open, "")] if isinstance(ee_T, np.ndarray) else list(ee_T)
        img = Image.fromarray(self.frame.rgb).convert("RGB")
        draw = ImageDraw.Draw(img, "RGBA")
        try:
            font = ImageFont.load_default(size=14)
        except TypeError:  # Pillow < 10.1
            font = ImageFont.load_default()
        gz = self.view.grid_z

        def line(a, b, fill, width):
            pa, pb = self.project(a), self.project(b)
            if pa and pb:
                draw.line([pa, pb], fill=fill, width=width)

        xs = np.arange(np.ceil(self.ws.x[0] / 0.02) * 0.02, self.ws.x[1] + 1e-9, 0.02)
        ys = np.arange(np.ceil(self.ws.y[0] / 0.02) * 0.02, self.ws.y[1] + 1e-9, 0.02)

        def major(v):
            return abs(v / 0.1 - round(v / 0.1)) < 1e-6

        for x in xs:
            line((x, ys[0], gz), (x, ys[-1], gz), (255, 255, 255, 150 if major(x) else 60), 2 if major(x) else 1)
        for y in ys:
            line((xs[0], y, gz), (xs[-1], y, gz), (255, 255, 255, 150 if major(y) else 60), 2 if major(y) else 1)
        # Label every 10 cm intersection rather than the grid's edges -- the
        # overhead only sees part of the workspace, so the edges are mostly
        # off-frame (and always are once zoomed in).
        W, H = img.size
        for x in filter(major, xs):
            for y in filter(major, ys):
                p = self.project((x, y, gz))
                if p and 0 <= p[0] < W - 40 and 0 <= p[1] < H - 30:
                    draw.text((p[0] + 3, p[1] + 1), f"x{x:.1f}", fill=(255, 255, 0, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0, 255))
                    draw.text((p[0] + 3, p[1] + 14), f"y{y:+.1f}", fill=(0, 255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0, 255))

        # pinch-center reticle: the point itself, the jaw axis, the approach
        # direction, and a drop line to the grid plane so height reads in 2D.
        for T_ee, g_open, label in reticles:
            pinch, R = T_ee[:3, 3], T_ee[:3, :3]
            jaw_half = 0.02 + 0.02 * g_open
            line(pinch, (pinch[0], pinch[1], gz), (255, 0, 255, 200), 2)
            line(pinch - R[:, 1] * jaw_half, pinch + R[:, 1] * jaw_half, (0, 255, 0, 255), 3)
            line(pinch, pinch - R[:, 2] * 0.04, (255, 128, 0, 255), 2)
            for pt, r, col in ((pinch, 7, (0, 255, 0, 255)), ((pinch[0], pinch[1], gz), 4, (255, 0, 255, 255))):
                p = self.project(pt)
                if p:
                    draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], outline=col, width=2)
            p = self.project(pinch)
            if label and p:
                draw.text((p[0] + 10, p[1] - 22), label, fill=(0, 255, 0, 255), font=font,
                          stroke_width=2, stroke_fill=(0, 0, 0, 255))

        if self.view.center is not None and self.view.zoom > 1.0:
            c = self.project(self.view.center) or (W / 2, H / 2)
            w, h = W / self.view.zoom, H / self.view.zoom
            x0 = float(np.clip(c[0] - w / 2, 0, W - w))
            y0 = float(np.clip(c[1] - h / 2, 0, H - h))
            self._crop = (x0, y0, w, h)
            img = img.crop((int(x0), int(y0), int(x0 + w), int(y0 + h))).resize((W, H), Image.BICUBIC)
            # crop() truncates to ints; keep the mapping exact for deproject
            self._crop = (float(int(x0)), float(int(y0)), float(int(x0 + w) - int(x0)), float(int(y0 + h) - int(y0)))
        else:
            self._crop = (0.0, 0.0, float(W), float(H))
        return img
