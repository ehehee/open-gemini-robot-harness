"""A synthetic overhead frame for sim runs without a camera: the table plane
and one block, ray-cast through the calibrated overhead pose and intrinsics,
so the grid overlay, deprojection and the collision cloud all line up with the
arms exactly as on the rig.

Only for trying the harness end to end; it is not a model of any real scene.
"""
from __future__ import annotations

import numpy as np

from camera.extrinsics import Extrinsics
from camera.intrinsics import Intrinsics
from camera.zed_overhead import OverheadFrame

TABLE_RGB = (176, 168, 150)
BLOCK_RGB = (200, 40, 40)


def synthetic_overhead(T_base_cam: Extrinsics, K: Intrinsics, table_z: float = 0.0,
                       block_center=(0.38, 0.05), block_size=(0.05, 0.05, 0.05)) -> OverheadFrame:
    """T_base_cam: camera -> right arm base. The block stands on the table,
    axis-aligned, centred at block_center (x, y) in the right arm's base frame."""
    W, H = K.width, K.height
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays_cam = np.stack([(u - K.cx) / K.fx, (v - K.cy) / K.fy, np.ones_like(u)], -1).reshape(-1, 3)
    R, o = T_base_cam.R, T_base_cam.t
    d = rays_cam @ R.T  # ray directions in the base frame, scaled so the camera-z component is 1

    # Table plane: o_z + s * d_z = table_z
    with np.errstate(divide="ignore", invalid="ignore"):
        s_table = (table_z - o[2]) / d[:, 2]
    s_table[~np.isfinite(s_table) | (s_table <= 0)] = np.inf

    # Block: slab test against the axis-aligned box
    lo = np.array([block_center[0] - block_size[0] / 2, block_center[1] - block_size[1] / 2, table_z])
    hi = lo + np.array([block_size[0], block_size[1], block_size[2]])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - o) / d, (hi - o) / d
    t_near = np.nanmax(np.minimum(t1, t2), axis=1)
    t_far = np.nanmin(np.maximum(t1, t2), axis=1)
    hit_block = (t_near <= t_far) & (t_near > 0)
    s_block = np.where(hit_block, t_near, np.inf)

    s = np.minimum(s_table, s_block)  # camera-frame depth, since rays_cam has z = 1
    depth = np.where(np.isfinite(s), s, 0.0).astype(np.float32).reshape(H, W)
    rgb = np.empty((H * W, 3), np.uint8)
    rgb[:] = TABLE_RGB
    rgb[s_block < s_table] = BLOCK_RGB
    rgb[~np.isfinite(s)] = 0
    # A little texture so the image doesn't read as a flat fill.
    noise = np.random.default_rng(0).integers(-6, 7, size=(H * W, 1))
    rgb = np.clip(rgb.astype(int) + noise, 0, 255).astype(np.uint8).reshape(H, W, 3)
    return OverheadFrame(rgb=rgb, depth=depth, fx=K.fx, fy=K.fy, cx=K.cx, cy=K.cy)
