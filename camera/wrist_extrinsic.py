"""CAD-derived (not hand-eye solved) flange-to-wrist-camera extrinsic for a
YAM arm with a crank_4310 gripper and wrist-mounted D405, lifted directly from
i2rt's own station model for this exact mount:
i2rt/robot_models/station/yam_station_crank_4310_d405/{.xml,README.md}.

This is the mechanical design transform (CAD), not a measured hand-eye
calibration. That README states it is byte-identical between the left/right
mounts and "holds at every arm configuration" since the camera is rigidly
flange-mounted -- accurate to design tolerance, which the README puts at
"<=1.5 arcsec from exact" for the rotation. Good enough to place a wrist
point cloud correctly in the scene; not a substitute for an actual eye-in-hand
solve (collect_poses.py/solve_handeye.py in yam-bimanual-calib) if you need
sub-mm precision.

Frame note: `{side}_gripper` in that station model is the SAME body i2rt's own
combined arm+gripper model (what RealYamRobot/get_yam_robot builds) calls
plain "gripper" -- the joint6/flange output body, NOT the `grasp_site` site
RealYamRobot.get_ee_pose() reads. Use gripper_body_fk() below (not
get_ee_pose()) to get T_base_gripperbody before composing with T_GRIPPER_D405.
"""
from __future__ import annotations

import numpy as np

from camera.extrinsics import Extrinsics

# Full-precision pos/quat(w,x,y,z) copied from yam_station_crank_4310_d405.xml's
# left_camera/right_camera body (byte-identical for both sides per the README).
_POS = np.array([0.0016999000000607195, -0.07972864298920912, 0.06602077467211452])
_QUAT_WXYZ = np.array(
    [-0.906306513345646, 0.4226209931651356, 1.6645277061133612e-06, 7.761806802841775e-07]
)


def _quat_wxyz_to_mat3(q: np.ndarray) -> np.ndarray:
    import mujoco

    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, q)
    return mat.reshape(3, 3)


T_GRIPPER_D405 = Extrinsics.from_R_t(_quat_wxyz_to_mat3(_QUAT_WXYZ), _POS)


def gripper_body_fk(model, joints) -> Extrinsics:
    """T_base_gripperbody for the 6 arm joints -- the "gripper" BODY (flange),
    distinct from the `grasp_site` SITE used elsewhere in this project. See
    module docstring for why that distinction matters here."""
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    for i, val in enumerate(joints, start=1):
        data.qpos[model.joint(f"joint{i}").qposadr[0]] = val
    mujoco.mj_kinematics(model, data)
    body_id = model.body("gripper").id
    T = np.eye(4)
    T[:3, :3] = data.xmat[body_id].reshape(3, 3)
    T[:3, 3] = data.xpos[body_id]
    return Extrinsics(T)


def wrist_parts(side: str) -> list[tuple[str, Extrinsics]]:
    """What is bolted to each gripper that i2rt's arm model lacks, as
    gap_guard.scene.CollisionScene gripper parts. The right gripper carries
    the D405 (below). The left one carries a different camera whose mount
    CAD we do not have -- it is NOT modelled, so nothing checks it."""
    return [d405_wrist_mount()] if side == "right" else []


def d405_wrist_mount() -> tuple[str, Extrinsics]:
    """(mesh file, T_gripperbody_part) of i2rt's own CAD for this mount: the
    D405 and its bracket merged into one mesh, placed at T_GRIPPER_D405 (the
    station model gives the mesh no offset from that body). For collision
    checking -- an arm model without it is missing the part that sticks out
    furthest above the gripper."""
    from pathlib import Path

    import i2rt

    mesh = Path(i2rt.__file__).parent / "robot_models" / "station" / "assets" / "d405_wrist_crank_4310.stl"
    return str(mesh), T_GRIPPER_D405
