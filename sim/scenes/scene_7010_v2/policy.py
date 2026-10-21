"""Feedback-gated grasp, transfer, and released placement; no object attachment."""
import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent

class Handover:
    def __init__(self, sim):
        self.sim = sim
        self.names = list(sim.scene.embodiment.iks)
        self.q = sim.state.joint_q.numpy().astype(np.float64)
        self.grips = dict.fromkeys(self.names, 0.)
        self.events = []
        self.trace = []
        self.stage = 'initial'
        self.held = {}
        self.grip_drift = {}

    def relative_clamp(self, side):
        p,q=self.sim.scene.embodiment.iks[side].ee_pose(self.sim.state)
        obj=np.asarray(self.sim['clamp'].pose)
        r=Rotation.from_quat(q).inv()
        return r.apply(obj[:3]-p),r*Rotation.from_quat(obj[3:])

    def check_grip_drift(self):
        for side,(p0,r0) in self.held.items():
            p,r=self.relative_clamp(side)
            distance=float(np.linalg.norm(p-p0))
            angle=float(np.degrees((r0.inv()*r).magnitude()))
            metrics=self.grip_drift[side]
            metrics['max_translation_mm']=max(metrics['max_translation_mm'],distance*1000)
            metrics['max_rotation_deg']=max(metrics['max_rotation_deg'],angle)
            if distance>.004 or angle>5:
                raise RuntimeError(f'{self.stage}: {side} grip slipped {distance*1000:.2f} mm / {angle:.2f} degrees')

    def command(self):
        values = []
        for name in self.names:
            handle = self.sim.scene.handles.arms[name]
            values.extend(self.q[list(handle.arm_coords)])
            values.append(self.grips[name])
        return np.array(values, dtype=np.float32)

    def hold(self, seconds):
        for _ in range(round(seconds / self.sim.pipeline.control_dt)):
            self.sim.step(self.command(), mode='joint_position')
            if not self.trace or self.sim.time-self.trace[-1][0] >= .099:
                self.trace.append((float(self.sim.time),self.stage,self.sim.state.body_q.numpy().copy()))
                self.check_grip_drift()
            if not np.isfinite(self.sim.state.body_q.numpy()).all():
                raise RuntimeError('Nonfinite simulation state')
            yield

    def move(self, side, target, quat, seconds=2., orientation_tolerance=.12):
        ik = self.sim.scene.embodiment.iks[side]
        indices = list(self.sim.scene.handles.arms[side].arm_coords)
        with ik.host_fk():
            goal = ik.solve_q_full(self.q, target, quat)
            pos, orient = ik.fk_ee(goal)
            # The offset crank TCP can leave staged roll IK in a local minimum.
            # Refine all six arm joints against the same native FK and limits.
            if np.linalg.norm(pos-target) > .003 or (Rotation.from_quat(quat).inv()*Rotation.from_quat(orient)).magnitude() > .12:
                import mujoco
                source = mujoco.MjModel.from_xml_path(str(ROOT/'yam_crank.xml'))
                limits = np.array([source.joint(f'{side}_joint{i}').range for i in range(1, 7)])
                desired = Rotation.from_quat(quat)
                def residual(values):
                    candidate = self.q.copy(); candidate[indices] = values
                    p, q = ik.fk_ee(candidate)
                    return np.r_[(p-target)*100., (desired.inv()*Rotation.from_quat(q)).as_rotvec()*5.]
                best = None
                for seed in (goal[indices], self.q[indices]):
                    fit = least_squares(residual, np.clip(seed, limits[:,0]+1e-5, limits[:,1]-1e-5),
                                        bounds=(limits[:,0]+1e-6, limits[:,1]-1e-6),
                                        diff_step=1e-3, max_nfev=250)
                    if best is None or np.linalg.norm(fit.fun) < np.linalg.norm(best.fun): best = fit
                    if np.linalg.norm(fit.fun) < .01: break
                goal[indices] = best.x
                pos, orient = ik.fk_ee(goal)
        error = float(np.linalg.norm(np.asarray(pos) - target))
        angle = float((Rotation.from_quat(quat).inv() * Rotation.from_quat(orient)).magnitude())
        if error > .003 or angle > orientation_tolerance:
            raise RuntimeError(f'{self.stage}: unreachable {side} pose ({error:.4f} m, {angle:.3f} rad)')
        begin = self.q.copy()
        count = round(seconds / self.sim.pipeline.control_dt)
        for i in range(count):
            t = (i + 1) / count
            blend = t*t*(3 - 2*t)
            self.q[indices] = begin[indices] + blend*(goal[indices] - begin[indices])
            yield from self.hold(self.sim.pipeline.control_dt)
        actual = ik.ee_pose(self.sim.state)[0]
        if np.linalg.norm(actual - target) > .012:
            actual_q = self.sim.state.joint_q.numpy()[indices]
            raise RuntimeError(f'{self.stage}: {side} did not reach waypoint: actual={actual}, target={target}, joint_error={actual_q-self.q[indices]}')

    def forces(self, side):
        # Newton's CPU MuJoCo bridge does not populate the GPU contact-force
        # buffer. Read the actual solver contacts on that backend.
        solver = self.sim.pipeline.solver
        if getattr(solver, 'use_mujoco_cpu', False):
            import mujoco
            values = [0., 0.]
            model, data = solver.mj_model, solver.mj_data
            for i, contact in enumerate(data.contact):
                names = [model.geom(int(g)).name for g in contact.geom]
                if not any('clamp_' in n for n in names): continue
                for j, finger in enumerate(('left', 'right')):
                    if any(f'{side}_tip_{finger}' in n for n in names):
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, i, force)
                        values[j] += max(0., float(force[0]))
            return values
        shapes = self.sim.scene.handles.entity('clamp').collision_shapes
        self.sim.contacts.update(self.sim.step_count)
        handle = self.sim.scene.handles.arms[side]
        return [float(np.linalg.norm(self.sim.contacts.pairwise_force(
            shapes, np.concatenate([indices for _, indices in group]))))
            for group in handle.finger_groups]

    def require_pinch(self, side):
        forces = self.forces(side)
        self.events.append({'stage': self.stage, 'time_s': self.sim.time,
                            'side': side, 'finger_forces_N': forces})
        if len(forces) != 2 or min(forces) < .15:
            raise RuntimeError(f'{self.stage}: {side} has no bilateral pinch: {forces}')
        if side not in self.held:
            self.held[side]=self.relative_clamp(side)
            self.grip_drift[side]={'max_translation_mm':0.,'max_rotation_deg':0.}

    def grasp_frame(self, part):
        mesh = trimesh.load(ROOT/'assets/clamp'/part, process=False)
        center = mesh.bounds.mean(axis=0)
        _, vectors = np.linalg.eigh(np.cov(mesh.vertices.T))
        tangent = vectors[:, -1]
        tangent[2] = 0
        tangent /= np.linalg.norm(tangent)
        if part == '0008.obj':
            # Grip the inner, straighter section of the rubber sleeve to
            # reduce the moment arm of the clamp's weight.
            center -= tangent*np.sign(np.dot(center,tangent))*.017
        pose = np.asarray(self.sim['clamp'].pose)
        rot = Rotation.from_quat(pose[3:])
        center = pose[:3] + rot.apply(center)
        tangent = rot.apply(tangent)
        z = np.array([0., 0., -1.])
        x = tangent.copy(); x[2] = 0; x /= np.linalg.norm(x)
        if part == '0018.obj':
            # Receive from the outside of the second handle, horizontally.
            # This separates the two wrist housings vertically at handover.
            z = tangent.copy()
            if z[1] > 0: z *= -1
            y = np.cross([0., 0., 1.], z); y /= np.linalg.norm(y)
            x = np.cross(y, z)
        else:
            # A parallel-jaw pinch is invariant under a half turn about approach.
            # Choose the equivalent sleeve direction reachable by the crank wrist.
            x = -x
            y = np.cross(z, x)
        return center, Rotation.from_matrix(np.column_stack([x, y, z])).as_quat()

    def run(self):
        self.stage = 'settle'
        yield from self.hold(.5)
        initial_z = self.sim['clamp'].position[2]
        pick, right_rot = self.grasp_frame('0008.obj')
        # Centre the pads on the sleeve. Its centre is already above the
        # 7.4 mm pad reach, so no extra height offset is needed for mat clearance.
        self.stage = 'pregrasp'
        yield from self.move('right', pick + [0, 0, .10], right_rot)
        self.stage = 'grasp'
        yield from self.move('right', pick, right_rot, 2.5)
        for grip in np.linspace(0, .95, 40):
            self.grips['right'] = float(grip)
            yield from self.hold(.02)
        yield from self.hold(.8)
        self.require_pinch('right')
        self.stage = 'lift'
        yield from self.move('right', pick + [0, 0, .17], right_rot, 3.)
        self.require_pinch('right')
        yield from self.hold(2.)
        self.require_pinch('right')
        if self.sim['clamp'].position[2] < initial_z + .10:
            raise RuntimeError('Clamp did not lift with the giving hand')
        self.stage = 'present'
        meet = np.array([.04, -.065, .14])
        yield from self.move('right', meet, right_rot, 3.)
        receive, left_rot = self.grasp_frame('0018.obj')
        self.stage = 'receive approach'
        approach = Rotation.from_quat(left_rot).apply([0, 0, 1])
        yield from self.move('left', receive - approach*.06, left_rot, 3.)
        yield from self.move('left', receive, left_rot, 2.5)
        for grip in np.linspace(0, .95, 40):
            self.grips['left'] = float(grip)
            yield from self.hold(.02)
        yield from self.hold(.8)
        self.require_pinch('left')
        self.stage = 'transfer'
        self.held.pop('right',None)
        self.grips['right'] = 0.
        yield from self.hold(.6)
        self.require_pinch('left')
        yield from self.move('right', meet + [0, -.10, .04], right_rot)
        self.require_pinch('left')
        self.events.append({'stage': 'handover_confirmed', 'time_s': self.sim.time})
        self.stage = 'carry to tray'
        tcp,tcp_quat = self.sim.scene.embodiment.iks['left'].ee_pose(self.sim.state)
        obj=np.asarray(self.sim['clamp'].pose)
        obj_rot=Rotation.from_quat(obj[3:])
        offset=obj_rot.inv().apply(np.asarray(tcp)-obj[:3])
        relative_rot=obj_rot.inv()*Rotation.from_quat(tcp_quat)
        tray_pose=np.asarray(self.sim['tray'].pose);tray=tray_pose[:3]
        # Align the tool with the long tray dimension, using the measured grip
        # transform and a reachable wrist orientation. Centre its whole bounds.
        place_rot=Rotation.from_quat(tray_pose[3:])*Rotation.from_euler('z',np.pi-.3)
        vertices=np.vstack([trimesh.load(ROOT/'assets/clamp'/p['path'],process=False).vertices
                           for p in json.loads((ROOT/'assets/clamp/parts.json').read_text())])
        midpoint=(vertices.min(axis=0)+vertices.max(axis=0))/2;midpoint[2]=0
        desired=tray+[0,0,.08]-place_rot.apply(midpoint)
        target=desired+place_rot.apply(offset)
        left_rot=(place_rot*relative_rot).as_quat()
        yield from self.move('left', target, left_rot, 4.)
        self.require_pinch('left')
        self.stage = 'centre over tray'
        # Reobserve the held object: its grip transform can shift during carry.
        # Release above the rim so the horizontal receiver does not hit it.
        # Leave a little room toward the near rim for the jaw-release motion.
        for _ in range(2):
            tcp = self.sim.scene.embodiment.iks['left'].ee_pose(self.sim.state)[0]
            correction = desired - np.asarray(self.sim['clamp'].position)
            if np.linalg.norm(correction) < .012: break
            target = np.asarray(tcp) + correction
            # Placement is unkeyed: allow wrist rotation while maintaining a
            # verified pinch, rather than demanding the handover wrist pose.
            yield from self.move('left', target, left_rot, 1.5, orientation_tolerance=.35)
            yield from self.hold(.3)
            self.require_pinch('left')
        self.stage = 'release'
        self.held.pop('left',None)
        # Open smoothly: a step command kicks the clamp sideways off the
        # crank fingers' curved shells as it drops into the tray.
        for grip in np.linspace(self.grips['left'], 0., 40):
            self.grips['left'] = float(grip)
            yield from self.hold(.02)
        yield from self.hold(.6)
        self.stage = 'withdraw'
        withdraw_rot = self.sim.scene.embodiment.iks['left'].ee_pose(self.sim.state)[1]
        yield from self.move('left', target + [0, 0, .10], withdraw_rot, orientation_tolerance=.2)
        yield from self.hold(2.)
        from shapely.geometry import MultiPoint, Point
        floor = trimesh.load(ROOT/'assets/tray/0001.obj', process=False)
        interior = MultiPoint(floor.vertices[:,:2]).convex_hull.buffer(-.001)
        def evaluate():
            pose=np.asarray(self.sim['clamp'].pose);tray_pose=np.asarray(self.sim['tray'].pose)
            points=[]
            for part in json.loads((ROOT/'assets/clamp/parts.json').read_text()):
                mesh=trimesh.load(ROOT/'assets/clamp'/part['path'],process=False)
                world=Rotation.from_quat(pose[3:]).apply(mesh.vertices)+pose[:3]
                points.append(Rotation.from_quat(tray_pose[3:]).inv().apply(world-tray_pose[:3]))
            points=np.concatenate(points)
            coverage=float(np.mean([interior.covers(Point(p[:2])) for p in points]))
            clamp_shapes=self.sim.scene.handles.entity('clamp').collision_shapes
            tray_shapes=self.sim.scene.handles.entity('tray').collision_shapes
            self.sim.contacts.update(self.sim.step_count)
            force=float(np.linalg.norm(self.sim.contacts.pairwise_force(clamp_shapes,tray_shapes)))
            speed=float(np.linalg.norm(self.sim['clamp'].velocity[:3]))
            clear=min(float(np.linalg.norm(self.sim.scene.embodiment.iks[side].ee_pose(self.sim.state)[0]-pose[:3])) for side in self.names)
            success=coverage>=.999 and force>.1 and speed<.02 and clear>.08
            return success,dict(whole_mesh_inside_fraction=coverage,tray_contact_force_N=force,speed_m_s=speed,hand_clearance_m=clear)
        success,metrics=evaluate()
        for _ in range(6):
            if success:break
            yield from self.hold(1.)
            success,metrics=evaluate()
        self.events.append({'stage':'released_placement','success':success,**metrics})
        if not success:raise RuntimeError('Released placement failed: '+str(metrics))
        yield from self.hold(1.)
        success,metrics=evaluate()
        if not success:raise RuntimeError('Placement did not remain stable for one second')
        self.stage = 'complete'

def handover(sim, session=None):
    panels = list(session.arms.values()) if session else []
    controller = Handover(sim)
    for panel in panels: panel.suspend()
    error = None
    try:
        yield from controller.run()
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        for panel in panels: panel.resume()
        (ROOT/'handover_result.json').write_text(json.dumps({
            'success': controller.stage == 'complete', 'last_stage': controller.stage,
            'error': error, 'events': controller.events,
            'grip_drift': controller.grip_drift,
            'object_attachment': False, 'object_pose_commands': False,
        }, indent=2, default=str))
        if controller.trace:
            np.savez_compressed(ROOT/'policy_trace.npz',time=np.array([r[0] for r in controller.trace]),stage=np.array([r[1] for r in controller.trace]),body_q=np.stack([r[2] for r in controller.trace]))
