"""Shared viser rendering of the YAM arm's MuJoCo model.

Collects the model's mesh geoms once and refreshes their viser transforms from
an MjData -- the same approach i2rt uses in
`i2rt.utils.viser_control_interface.ViserControlInterface` (see that file for
the original; this is the same pattern driven by poses we supply rather than a
live teleop loop).

Instantiated more than once in one scene: per arm, one translucent ghost at
the arm's current pose and one solid arm at the planned pose.
"""
from __future__ import annotations

import mujoco
import numpy as np
import viser

from camera.extrinsics import Extrinsics


def mat3_to_wxyz(mat3: np.ndarray) -> np.ndarray:
    """MuJoCo 3x3 rotation -> the (w,x,y,z) quaternion viser wants."""
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat3).flatten())
    return q


def set_arm_qpos(model: mujoco.MjModel, data: mujoco.MjData, joints) -> None:
    """Write the 6 arm joints into `data` and run FK.

    Everything else in qpos (the 2 gripper-mechanism joints of the combined
    arm+gripper model) is zeroed -- the arm links' poses don't depend on them,
    so this renders the arm exactly, and the jaws at their zero position.
    """
    data.qpos[:] = 0.0
    for i, val in enumerate(joints, start=1):
        data.qpos[model.joint(f"joint{i}").qposadr[0]] = val
    mujoco.mj_kinematics(model, data)


class RobotMeshes:
    """One renderable copy of the arm. Create several with different `name`s to
    show several configurations at once."""

    def __init__(
        self,
        server: viser.ViserServer,
        model: mujoco.MjModel,
        name: str = "robot",
        color: tuple[int, int, int] | None = None,
        opacity: float | None = None,
        visible: bool = True,
    ):
        """color=None keeps each geom's own colour from the model; pass one to
        tint the whole arm a single colour (used for the ghost)."""
        self.model = model
        self.geom_ids = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        self.handles = {}
        for g in self.geom_ids:
            mesh_id = model.geom_dataid[g]
            v_adr, v_num = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            f_adr, f_num = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
            verts = model.mesh_vert[v_adr : v_adr + v_num]
            faces = model.mesh_face[f_adr : f_adr + f_num]
            geom_color = color if color is not None else tuple(int(c * 255) for c in model.geom_rgba[g][:3])
            self.handles[g] = server.scene.add_mesh_simple(
                f"{name}/geom_{g}", verts, faces,
                color=geom_color, opacity=opacity, visible=visible,
                # Translucent meshes need both faces drawn or the arm looks
                # hollow from the side where only backfaces are visible.
                side="double" if opacity is not None and opacity < 1.0 else "front",
            )

    def update(self, data: mujoco.MjData, base_transform: Extrinsics | None = None) -> None:
        """base_transform places this arm's OWN base in some other frame first
        -- e.g. rendering a second arm in the primary arm's base frame via the
        real, already-solved T_leftbase_rightbase."""
        for g in self.geom_ids:
            h = self.handles[g]
            pos, mat = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3)
            if base_transform is not None:
                pos = base_transform.apply(pos)
                mat = base_transform.R @ mat
            h.position = np.asarray(pos).copy()
            h.wxyz = mat3_to_wxyz(mat)

    def set_visible(self, visible: bool) -> None:
        for h in self.handles.values():
            h.visible = visible
