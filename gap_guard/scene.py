"""One MuJoCo collision world per observation, in the planning arm's base frame.

What is in it:
  - the planning arm ("a/"), the i2rt MJCF that FK/IK already use, at the
    origin -- so this world's frame IS the planning arm's base frame;
  - the other arm ("b/"), the same kind of MJCF, placed at T_this_other and
    frozen at its snapshot joints;
  - rigid parts bolted to a gripper that the i2rt arm model lacks, per arm
    (`gripper_parts` on this one, `other_gripper_parts` on the other; e.g.
    the right wrist D405: camera.wrist_extrinsic.d405_wrist_mount()) -- as
    arm geometry, so they are checked AND their points are stripped from the
    cloud with the rest of the arm;
  - the fingertips as convex pieces (data/crank_4310, from
    tools/decompose_gripper_tips.py), not as i2rt's tip meshes: each tip mesh
    is the finger and its rack in one part, and its hull -- what MuJoCo
    collides -- is ~4x the finger's volume, a solid wedge filling the jaws;
  - each arm's fingers where its gripper actually is (gripper_open /
    other_gripper_open, RealYamRobot.get_gripper()'s 0 closed .. 1 open).
    Neither end of the travel is a safe default: open tips stand wide, but
    closing ones sweep into the space between the jaws -- which is where the
    other gripper is during a handover. So an opening left as None makes
    each fingertip collide as its whole swept travel instead;
  - a table plane at table_z;
  - the overhead point cloud as ONE heightfield: per cell, the highest point
    the camera saw. An overhead camera cannot see under anything, so filling
    each column down to the table is the honest reading of its data, and one
    hfield compiles in milliseconds where tens of thousands of voxel boxes
    take minutes.

Who can hit whom, via contype/conaffinity bits:
  _ARM = 1: the planning arm's moving geoms (link1..tips, gripper housing);
  _ENV = 2: everything else (other arm, table, heightfield).
The planning arm's own base only collides with the planning arm (it is bolted
to the table and does not move), and environment geoms never collide with each
other, so every contact generated is one the planning arm's motion can cause.

Self-collision pairs that are not checked, the same heuristic MoveIt's SRDF
setup uses: parent/child links (MuJoCo's own filterparent skips bodies welded
to the world, which is exactly base/link1), plus any pair already touching at
the arm's rest pose (all joints zero, where the real arm physically sits) --
those are convex-hull artifacts, not reachable collisions.

Meshes collide as their convex hulls, so this errs toward COLLISION near
concave parts (between the jaws, inside the link elbows).
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np

from camera.extrinsics import Extrinsics

log = logging.getLogger(__name__)

_ARM, _ENV = 1, 2
_VISUAL_ONLY = 3  # geom group of meshes kept for display, not collision (the tips' own meshes)
# Rubber pads on the fingers' inner faces (what touches a part first): the
# face found on the tip mesh within PAD_LENGTH_M of the fingertip, a box
# PAD_PROUD_M proud of it. Same construction as sim_tasks/handover/records.py.
PAD_LENGTH_M, PAD_PROUD_M, PAD_THICKNESS_M = 0.025, 0.0015, 0.003
TIP_PIECES = Path(__file__).resolve().parent / "data" / "crank_4310"
_A, _B = "a/", "b/"


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).flatten())
    return q


def heightmap(points: np.ndarray, floor_z: float, cell: float, radius: float,
              min_points: int) -> np.ndarray:
    """(n, n) grid of heights above floor_z covering [-radius, radius]^2, row
    0 at y = -radius (MuJoCo's hfield row order). A cell needs min_points hits
    to count, so single stray stereo points don't become pillars. One empty
    cell is kept around the border so MuJoCo's [0, 1] normalization has a true
    zero to anchor on."""
    n = int(np.ceil(2 * radius / cell)) + 2
    H = np.zeros((n, n))
    if len(points) == 0:
        return H
    col = np.floor((points[:, 0] + radius) / cell).astype(int) + 1
    row = np.floor((points[:, 1] + radius) / cell).astype(int) + 1
    ok = (col >= 1) & (col < n - 1) & (row >= 1) & (row < n - 1)
    col, row, h = col[ok], row[ok], points[ok, 2] - floor_z
    flat = row * n + col
    counts = np.bincount(flat, minlength=n * n)
    top = np.zeros(n * n)
    np.maximum.at(top, flat, h)
    top[counts < min_points] = 0.0
    return top.reshape(n, n)


class CollisionScene:
    """Compiled collision world. `in_collision(q6_model)` is the only query."""

    def __init__(
        self,
        arm_xml: str,
        *,
        other_arm_xml: str | None = None,
        T_this_other: Extrinsics | None = None,
        other_q_model: np.ndarray | None = None,
        start_q_model: np.ndarray | None = None,
        gripper_parts: Sequence[tuple[str, Extrinsics]] = (),
        other_gripper_parts: Sequence[tuple[str, Extrinsics]] = (),
        gripper_open: float | None = None,
        other_gripper_open: float | None = None,
        table_z: float | None = 0.0,
        cloud_points: np.ndarray | None = None,
        cell_size: float = 0.01,
        min_points_per_cell: int = 3,
        margin: float = 0.0,
        self_margin: float | None = None,
        arm_margin: float | None = None,
        workspace_radius: float = 0.9,
        table_clearance: float = 0.02,
        max_height: float = 1.0,
        self_filter_margin: float = 0.03,
    ):
        # Clearance below which a pair counts as colliding, per kind of pair:
        # env (table, cloud), arm (the other arm), self (planning arm itself).
        # 0 = actual contact of the (already conservative) convex hulls: the
        # layer reports TRUE collisions. Replaying every recorded operator
        # decision (tools/replay_gap_guard.py --sweep) put all false alarms
        # inside a 5 mm buffer and every human-rejected arm-vs-arm motion at
        # 17+ mm of penetration, so a buffer only costs agreement.
        self.thresholds = {"env": margin, "arm": margin if arm_margin is None else arm_margin,
                           "self": margin if self_margin is None else self_margin}
        self.margin = max(self.thresholds.values())  # contacts are generated out to this
        spec = mujoco.MjSpec()
        spec.attach(mujoco.MjSpec.from_string(arm_xml), prefix=_A, frame=spec.worldbody.add_frame())
        has_b = other_arm_xml is not None
        if has_b:
            T = T_this_other.matrix if T_this_other is not None else np.eye(4)
            frame_b = spec.worldbody.add_frame(pos=T[:3, 3], quat=_mat_to_quat(T[:3, :3]))
            spec.attach(mujoco.MjSpec.from_string(other_arm_xml), prefix=_B, frame=frame_b)
        for prefix in (_A, _B) if has_b else (_A,):
            for i, (mesh_file, T) in enumerate(gripper_parts if prefix == _A else other_gripper_parts):
                name = f"{prefix}gripper_part{i}"
                spec.add_mesh(name=name, file=str(mesh_file))
                spec.body(prefix + "gripper").add_geom(
                    name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                    pos=T.matrix[:3, 3], quat=_mat_to_quat(T.matrix[:3, :3]))
            self._tip_pieces(spec, prefix)
        if table_z is not None:
            spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                    pos=[0, 0, table_z], size=[2, 2, 0.01])

        self._open = {_A: gripper_open, _B: other_gripper_open}
        # Arms first: their poses are needed to strip the arms' own points out
        # of the cloud before the heightfield is built from it.
        self._compile(spec, other_q_model)
        for prefix in (_A, _B) if has_b else (_A,):
            self._add_pads(spec, prefix)
        self._compile(spec, other_q_model)
        unknown = [prefix for prefix in ((_A, _B) if has_b else (_A,)) if self._open[prefix] is None]
        if unknown:
            for prefix in unknown:
                self._add_swept_tips(spec, prefix)
            self._compile(spec, other_q_model)
        if cloud_points is not None and len(cloud_points):
            floor_z = table_z if table_z is not None else 0.0
            p = np.asarray(cloud_points, dtype=float)
            keep = ((np.linalg.norm(p[:, :2], axis=1) < workspace_radius)
                    & (p[:, 2] > floor_z + table_clearance) & (p[:, 2] < floor_z + max_height))
            p = p[keep]
            p = p[~self._on_arms(p, start_q_model, self_filter_margin)]
            H = heightmap(p, floor_z, cell_size, workspace_radius, min_points_per_cell)
            log.debug("heightfield: %d points, %d occupied cells, top %.3f m",
                      len(p), int((H > 0).sum()), float(H.max()))
            if H.max() > 0:
                half = H.shape[0] * cell_size / 2
                spec.add_hfield(name="cloud", size=[half, half, float(H.max()), 0.01],
                                nrow=H.shape[0], ncol=H.shape[1],
                                userdata=(H / H.max()).flatten().tolist())
                spec.worldbody.add_geom(name="cloud", type=mujoco.mjtGeom.mjGEOM_HFIELD,
                                        hfieldname="cloud", pos=[0, 0, floor_z])
                self._compile(spec, other_q_model)

    def _compile(self, spec: mujoco.MjSpec, other_q_model) -> None:
        self.model = m = spec.compile()
        self.data = mujoco.MjData(m)
        self._a_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(_A)}
        self._b_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(_B)}
        a_base = m.body(_A + "base").id
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if m.geom_group[g] == _VISUAL_ONLY:
                m.geom_contype[g] = m.geom_conaffinity[g] = 0
            elif b in self._a_bodies:
                # i2rt ships the gripper housing with contype 0; it must hit the world here.
                m.geom_contype[g] = _ARM
                m.geom_conaffinity[g] = _ARM if b == a_base else _ARM | _ENV
            else:
                m.geom_contype[g], m.geom_conaffinity[g] = _ENV, 0
            m.geom_margin[g] = self.margin

        self._a_qadr = np.array([m.joint(f"{_A}joint{i}").qposadr[0] for i in range(1, 7)])
        self._b_qadr = (np.array([m.joint(f"{_B}joint{i}").qposadr[0] for i in range(1, 7)])
                        if self._b_bodies else None)
        self._a_fingers = np.array([m.joint(_A + j).qposadr[0] for j in ("joint7", "joint8")])
        self._cloud_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cloud")  # -1 if none
        self._a_tips = {m.body(_A + "tip_left").id, m.body(_A + "tip_right").id}
        self.a_jnt_range = np.array([m.jnt_range[m.joint(f"{_A}joint{i}").id] for i in range(1, 7)])
        self._qpos0 = np.zeros(m.nq)
        self._set_fingers(_A)
        if self._b_bodies:
            self._set_fingers(_B)
            if other_q_model is not None:
                for i, v in enumerate(other_q_model, start=1):
                    self._qpos0[m.joint(f"{_B}joint{i}").qposadr[0]] = v

        self._allowed = {frozenset((b, int(m.body_parentid[b]))) for b in range(1, m.nbody)}
        # The jaws meeting each other is a gripper closing on nothing.
        self._allowed.add(frozenset((m.body(_A + "tip_left").id, m.body(_A + "tip_right").id)))
        self._allowed |= set(self.contact_distances(np.zeros(6), only="self"))

    def _finger_ends(self, prefix: str) -> tuple[float, float]:
        """(open, closed) finger-joint values. Open is the end where the
        fingertips are farthest from the pinch point -- not where the tip
        BODIES are farthest apart: i2rt's crank_4310 tip meshes sit far from
        their body origins and include the rack, and those origins are
        farthest apart with the jaws SHUT (joint 0 is closed, as i2rt's own
        gripper mapping has it)."""
        m, d = self.model, self.data
        j7, j8 = m.joint(prefix + "joint7"), m.joint(prefix + "joint8")
        site = m.site(prefix + "grasp_site").id
        tips = {m.body(prefix + "tip_left").id, m.body(prefix + "tip_right").id}
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in tips and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        gaps = []
        for end in (0, 1):
            d.qpos[:] = 0
            d.qpos[j7.qposadr[0]] = m.jnt_range[j7.id][end]
            d.qpos[j8.qposadr[0]] = m.jnt_range[j8.id][end]
            mujoco.mj_kinematics(m, d)
            gap = np.inf
            for g in geoms:
                k = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
                w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                gap = min(gap, float(np.linalg.norm(w - d.site_xpos[site], axis=1).min()))
            gaps.append(gap)
        lo, hi = m.jnt_range[j7.id]
        return (lo, hi) if gaps[0] >= gaps[1] else (hi, lo)

    def finger_q(self, open_frac: float | None, prefix: str = _A) -> float:
        """Finger-joint value for an opening (0 closed .. 1 open); the open end
        when it is unknown (the tips are then swept hulls covering the travel)."""
        open_q, closed_q = self._finger_ends(prefix)
        if open_frac is None:
            return float(open_q)
        return float(closed_q + np.clip(open_frac, 0.0, 1.0) * (open_q - closed_q))

    def _set_fingers(self, prefix: str) -> None:
        m = self.model
        q = self.finger_q(self._open.get(prefix), prefix)
        for j in ("joint7", "joint8"):
            self._qpos0[m.joint(prefix + j).qposadr[0]] = q

    @staticmethod
    def _tip_pieces(spec: mujoco.MjSpec, prefix: str) -> None:
        """Collide each fingertip as its convex pieces; its mesh stays, for display only."""
        for tip in ("tip_left", "tip_right"):
            pieces = sorted(TIP_PIECES.glob(f"{tip}_*.obj"))
            if not pieces:
                continue
            body = spec.body(prefix + tip)
            for g in [g for g in body.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH]:
                g.group = _VISUAL_ONLY
                for i, f in enumerate(pieces):  # the pieces are in the mesh file's coordinates: same geom frame
                    name = f"{prefix}{tip}_piece{i}"
                    spec.add_mesh(name=name, file=str(f))
                    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, pos=g.pos, quat=g.quat)

    def _add_pads(self, spec: mujoco.MjSpec, prefix: str) -> None:
        """A pad box on each fingertip's inner face, found on the tip mesh with
        the fingers open (where the faces stand apart)."""
        m, d = self.model, self.data
        open_q, _ = self._finger_ends(prefix)
        d.qpos[:] = 0
        for j in ("joint7", "joint8"):
            d.qpos[m.joint(prefix + j).qposadr[0]] = open_q
        mujoco.mj_kinematics(m, d)
        site = m.site(prefix + "grasp_site").id
        ps, Rs = d.site_xpos[site].copy(), d.site_xmat[site].reshape(3, 3).copy()
        local = {}
        for tip in ("tip_left", "tip_right"):
            b = m.body(prefix + tip).id
            pts = [m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]] @ d.geom_xmat[g].reshape(3, 3).T
                   + d.geom_xpos[g]
                   for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_group[g] == _VISUAL_ONLY
                   for k in [m.geom_dataid[g]]]
            v = (np.vstack(pts) - ps) @ Rs  # grasp-site frame: y the closing axis, z out to the fingertips
            local[tip] = v[v[:, 2] > -PAD_LENGTH_M]
        for tip, other in (("tip_left", "tip_right"), ("tip_right", "tip_left")):
            v = local[tip]
            sign = np.sign(local[other][:, 1].mean() - v[:, 1].mean())
            face = (v[:, 1] * sign).max()
            f = v[v[:, 1] * sign > face - 0.001]
            c = np.array([(f[:, 0].min() + f[:, 0].max()) / 2, sign * (face + PAD_PROUD_M - PAD_THICKNESS_M / 2),
                          (f[:, 2].min() + f[:, 2].max()) / 2])
            half = [(f[:, 0].max() - f[:, 0].min()) / 2, PAD_THICKNESS_M / 2, (f[:, 2].max() - f[:, 2].min()) / 2]
            b = m.body(prefix + tip).id
            Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
            spec.body(prefix + tip).add_geom(name=f"{prefix}{tip}_pad", type=mujoco.mjtGeom.mjGEOM_BOX, size=half,
                                             pos=Rb.T @ (ps + Rs @ c - pb), quat=_mat_to_quat(Rb.T @ Rs))

    def _add_swept_tips(self, spec: mujoco.MjSpec, prefix: str) -> None:
        """Give each fingertip a mesh that is its own hull at both ends of its
        travel, i.e. everything it can occupy while the gripper is anywhere --
        piece by piece, so the hull of a piece's travel stays that piece's."""
        m = self.model
        open_q, closed_q = self._finger_ends(prefix)
        for tip, joint in (("tip_left", "joint7"), ("tip_right", "joint8")):
            body = m.body(prefix + tip)
            axis = m.jnt_axis[m.joint(prefix + joint).id]
            for n, g in enumerate(range(body.geomadr[0], body.geomadr[0] + body.geomnum[0])):
                if m.geom_group[g] == _VISUAL_ONLY or m.geom_type[g] not in (mujoco.mjtGeom.mjGEOM_MESH,
                                                                              mujoco.mjtGeom.mjGEOM_BOX):
                    continue
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                    v = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * m.geom_size[g]
                else:
                    k = m.geom_dataid[g]
                    v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, m.geom_quat[g])
                v = v @ R.reshape(3, 3).T + m.geom_pos[g]  # tip body frame, at joint = 0
                v = np.concatenate([v + open_q * axis, v + closed_q * axis]) - open_q * axis  # relative to the open end
                name = f"{prefix}{tip}_swept{n}"
                spec.add_mesh(name=name, uservert=v.flatten().tolist())
                spec.body(prefix + tip).add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name)

    def _on_arms(self, points: np.ndarray, start_q_model, filter_margin: float) -> np.ndarray:
        """Mask of points inside either arm's per-geom bounding box (inflated
        by filter_margin) at the current configuration. The overhead camera
        sees both arms; those points are the arms, not obstacles."""
        m, d = self.model, self.data
        self._set_q(np.zeros(6) if start_q_model is None else start_q_model)
        hit = np.zeros(len(points), dtype=bool)
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) not in self._a_bodies | self._b_bodies:
                continue
            center, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:] + filter_margin
            R = d.geom_xmat[g].reshape(3, 3)
            local = (points - d.geom_xpos[g]) @ R - center  # into the geom frame
            hit |= np.all(np.abs(local) <= half, axis=1)
        return hit

    def _set_q(self, q6_model: np.ndarray, other_q: np.ndarray | None = None,
               fingers: float | None = None) -> None:
        """The planning arm at q6_model; optionally the other arm at other_q
        instead of its snapshot, and this arm's fingers at joint value `fingers`."""
        d = self.data
        d.qpos[:] = self._qpos0
        d.qpos[self._a_qadr] = q6_model
        if other_q is not None:
            d.qpos[self._b_qadr] = other_q
        if fingers is not None:
            d.qpos[self._a_fingers] = fingers
        mujoco.mj_kinematics(self.model, d)

    def kind(self, pair) -> str:
        b1, b2 = tuple(pair)
        if b1 in self._a_bodies and b2 in self._a_bodies:
            return "self"
        return "arm" if (b1 in self._b_bodies or b2 in self._b_bodies) else "env"

    def _pair_dists(self, q6_model: np.ndarray, cloud: str = "check", **at) -> dict[frozenset, float]:
        """Closest distance per body pair, for every pair MuJoCo generated a
        contact for (out to the pair's combined margin). `at`: _set_q's
        other_q / fingers. cloud: "check"; "ignore" (what closing jaws close
        on is the point of closing them); "fingers_may_touch" (this arm's
        fingertips may touch it -- a grasp's descent and lift -- the rest of
        the arm may not)."""
        self._set_q(q6_model, **at)
        mujoco.mj_collision(self.model, self.data)
        m, d = self.model, self.data
        out: dict[frozenset, float] = {}
        for c in d.contact[: d.ncon]:
            if cloud != "check" and self._cloud_geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self._cloud_geom else c.geom1
                if cloud == "ignore" or int(m.geom_bodyid[other]) in self._a_tips:
                    continue
            pair = frozenset((int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])))
            out[pair] = min(out.get(pair, np.inf), float(c.dist))
        return out

    def contact_distances(self, q6_model: np.ndarray, only: str | None = None,
                          kinds: tuple[str, ...] = ("env", "arm", "self"), **at) -> dict[frozenset, float]:
        """Pairs closer than their kind's threshold (allowed pairs excluded,
        except while _allowed is being built), of the given kinds. `at`: see
        _pair_dists."""
        allowed = getattr(self, "_allowed", set())
        return {p: dist for p, dist in self._pair_dists(q6_model, **at).items()
                if p not in allowed and dist < self.thresholds[self.kind(p)]
                and (only is None or self.kind(p) == only) and self.kind(p) in kinds}

    def pair_distances(self, q6_model: np.ndarray, upto: float = 0.05) -> dict[frozenset, float]:
        """Every non-allowed pair within `upto`, with its distance -- whatever
        the thresholds. For diagnostics and choosing margins."""
        m = self.model
        saved = m.geom_margin.copy()
        m.geom_margin[:] = upto / 2  # pair margin is the sum of the two
        try:
            dists = self._pair_dists(q6_model)
        finally:
            m.geom_margin[:] = saved
        return {p: dist for p, dist in dists.items() if p not in self._allowed and dist < upto}

    def clearances(self, q6_model: np.ndarray, upto: float = 0.05) -> dict[str, float]:
        """Smallest distance per kind of pair (self / arm / env) within `upto`."""
        out = {"self": np.inf, "arm": np.inf, "env": np.inf}
        for p, dist in self.pair_distances(q6_model, upto).items():
            out[self.kind(p)] = min(out[self.kind(p)], dist)
        return out

    def site_pose(self, q6_model: np.ndarray, site: str = "grasp_site") -> np.ndarray:
        """4x4 FK of a planning-arm site, in this world's (= arm base) frame."""
        self._set_q(q6_model)
        sid = self.model.site(_A + site).id
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[sid].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[sid]
        return T

    def body_pose(self, q6_model: np.ndarray, body: str) -> np.ndarray:
        self._set_q(q6_model)
        bid = self.model.body(_A + body).id
        T = np.eye(4)
        T[:3, :3] = self.data.xmat[bid].reshape(3, 3)
        T[:3, 3] = self.data.xpos[bid]
        return T

    def in_collision(self, q6_model: np.ndarray, baseline: dict[frozenset, float] | None = None,
                     deepen_tol: float = 0.001, **at) -> bool:
        """True if any pair is closer than its threshold. With `baseline` (the
        contact_distances() of the motion's start), a pair that was already
        that close at the start only counts once it gets more than deepen_tol
        closer: an arm resting on the table can lift off, but not push in.
        `at`: see _pair_dists (the other arm moving too, the fingers, the cloud)."""
        for pair, dist in self.contact_distances(q6_model, **at).items():
            if baseline is not None and pair in baseline and dist >= baseline[pair] - deepen_tol:
                continue
            b1, b2 = tuple(pair)
            log.debug("contact %s <-> %s (%.1f mm, %s)", self.model.body(b1).name,
                      self.model.body(b2).name, dist * 1e3, self.kind(pair))
            return True
        return False
