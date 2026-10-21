"""Display and contact geometry for independently authored multi-material assets."""
from pathlib import Path
import json
import numpy as np
import trimesh
import newton
import warp as wp
from robosimstudio.domains.rigid.entity import EntityRigid
from robosimstudio.scene.registry import register_entity
from robosimstudio.scene.build.bodies import preserve_body_inertia
from robosimstudio.core.render.finish import Finish

@register_entity('video7010_v2_parts')
class VideoParts(EntityRigid):
 def build(self,builder,episode):
  start=builder.shape_count;super().build(builder,episode)
  for i in range(start,builder.shape_count):builder.shape_flags[i]=0
  folder=Path(self.spec.asset.path).parent
  records=json.loads((folder/'parts.json').read_text());self.surface={};groups={};contacts=[]
  for rec in records:
   mesh=trimesh.load(folder/rec['path'],process=False)
   key=(tuple(rec['color']),rec['metallic'],rec['roughness']);groups.setdefault(key,[]).append(mesh)
   if rec['collision']:contacts.append((mesh,rec['label']))
  for (color,metal,rough),parts in groups.items():
   mesh=trimesh.util.concatenate(parts);cfg=builder.default_shape_cfg.copy();cfg.has_shape_collision=False;cfg.has_particle_collision=False
   with preserve_body_inertia(builder,self.body):
    idx=builder.add_shape_mesh(body=self.body,mesh=newton.Mesh(np.asarray(mesh.vertices,dtype=np.float32),np.asarray(mesh.faces,dtype=np.int32).ravel(),compute_inertia=False),cfg=cfg,color=color,label=self.name+'_surface')
   self.surface[idx]=(color,Finish(metallic=metal,roughness=rough))
  cfg=builder.default_shape_cfg.copy();cfg.mu=.8
  for mesh,label in contacts:
   rubber=label in ['curved_dipped_handle_sleeve','rubber_jaw_boot']
   cfg.mu=1.2 if rubber else .8
   cfg.mu_torsional=.006 if rubber else .005
   cfg.mu_rolling=.001 if rubber else .0001
   # Thin tray walls must stay hollow; each short circumferential sector is convex.
   if label=='tapered_wall_and_rolled_rim':
    n=len(mesh.vertices)//9;pieces=[trimesh.convex.convex_hull(mesh.vertices[np.array([k*n+j for k in range(9) for j in [i,(i+1)%n]])]) for i in range(n)]
   elif label in ['irregular_cable_turn','loose_crossing_lead','seven_turn_torsion_spring','individual_cable_turn','loose_cable_end','spring_leg']:
    # These do not carry the grasp; use short local segments, never fill the coil centre.
    vertices=mesh.vertices;pieces=[]
    for k in range(0,len(vertices)-30,60):
     chunk=vertices[k:k+90]
     if len(chunk)>4:pieces.append(trimesh.convex.convex_hull(chunk))
   elif label in ['wound_masking_tape','cardboard_core_edge']:
    radii=np.linalg.norm(mesh.vertices[:,:2],axis=1);ri,ro=radii.min(),radii.max();z0,z1=mesh.bounds[:,2];pieces=[]
    for k in range(48):
     theta=np.array([k,k+1])*(2*np.pi/48);pts=np.vstack([ro*np.c_[np.cos(theta),np.sin(theta)],ri*np.c_[np.cos(theta[::-1]),np.sin(theta[::-1])]])
     pieces.append(trimesh.convex.convex_hull(np.vstack([np.c_[pts,np.full(4,z0)],np.c_[pts,np.full(4,z1)]])))
   elif label=='pierced_formed_sidewall':
    # Convex contact strips retain the open centre between jaw and handle.
    pieces=[]
    for tri in mesh.triangles[mesh.face_normals[:,2]>.9]:
     lo=tri.copy();lo[:,2]-=.001;pieces.append(trimesh.convex.convex_hull(np.vstack((tri,lo))))
   elif label=='molded_handle':
    # Preserve the handle's waist instead of filling it with one broad hull.
    pieces=[]
    edges=np.linspace(mesh.bounds[0,0],mesh.bounds[1,0],13)
    for lo,hi in zip(edges[:-1],edges[1:]):
     section=mesh.slice_plane([lo,0,0],[1,0,0]).slice_plane([hi,0,0],[-1,0,0])
     if len(section.vertices)>3:pieces.append(section.convex_hull)
   else:pieces=[mesh.convex_hull]
   for m in pieces:
    with preserve_body_inertia(builder,self.body):
     idx=builder.add_shape_mesh(body=self.body,mesh=newton.Mesh(np.asarray(m.vertices,dtype=np.float32),np.asarray(m.faces,dtype=np.int32).ravel(),compute_inertia=False),cfg=cfg,label=self.name+'_'+label)
    builder.shape_flags[idx]&=~int(newton.ShapeFlags.VISIBLE)
    if rubber:
     attr=builder.custom_attributes['mujoco:condim']
     if attr.values is None:attr.values={}
     attr.values[idx]=6
  if contacts and not self.spec.static:
   physical=trimesh.util.concatenate([m for m,_ in contacts])
   if physical.volume>1e-12:
    physical.density=float(builder.body_mass[self.body])/physical.volume
    inertia=np.asarray(physical.moment_inertia)+np.eye(3)*1e-9
    builder.body_com[self.body]=wp.vec3(*physical.center_mass)
    builder.body_inertia[self.body]=wp.mat33(*inertia.ravel())
    builder.body_inv_inertia[self.body]=wp.mat33(*np.linalg.inv(inertia).ravel())
  if self.spec.static:
   joint=builder.add_joint_fixed(-1,self.body,parent_xform=builder.body_q[self.body],label=self.name+'_fixed');builder.add_articulation([joint])
 def apply_display(self,builder):
  super().apply_display(builder)
  for idx,(color,finish) in self.surface.items():builder.shape_color[idx]=color;self.finishes[idx]=finish

# The live loader reimports this module; replace the earlier class explicitly.
# Same-source registration otherwise keeps the first class and stale contacts.
from robosimstudio.scene.registry import ENTITIES
ENTITIES.replace('video7010_v2_parts',VideoParts)
