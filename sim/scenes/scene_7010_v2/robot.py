"""Manufacturer YAM/crank CAD; fresh pose fit to frame-zero observations."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import numpy as np
import mujoco
import newton
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from robosimstudio.specs import ArmSpec,AssetSpec,EmbodimentSpec,GripperSpec,ActuationSpec
from robosimstudio.robots.registry import EMBODIMENTS,GRIPPERS
from robosimstudio.robots.embodiments.yam_look import split_yam_white_tubes
from layout import T,K,xy
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'robot'  # i2rt CAD at 120c3c8, both arms, calibrated mount (robot/PROVENANCE.txt)

def prepare():
 root=ET.parse(SOURCE/'yam_calibrated.xml').getroot()
 for mesh in root.findall('asset/mesh'):mesh.set('file',str(SOURCE/mesh.get('file')))
 # Exact-component CAD decomposition: no voxel remeshing/inflation of the jaws.
 for finger in ['left','right']:
  for part in sorted((ROOT/'finger_colliders').glob(f'tip_{finger}_*.obj')):
   ET.SubElement(root.find('asset'),'mesh',name=part.stem,file=str(part))
 for side in ['right','left']:
  for finger in ['left','right']:
   body=root.find(f".//body[@name='{side}_tip_{finger}']")
   original=body.find('geom')
   original.set('contype','0');original.set('conaffinity','0')
   parts=sorted((ROOT/'finger_colliders').glob(f'tip_{finger}_*.obj'))
   if not parts:raise RuntimeError('Build the CAD finger colliders before launching')
   for part in parts:
    attr={k:original.get(k) for k in ['pos','quat'] if original.get(k) is not None}
    ET.SubElement(body,'geom',name=f'{side}_contact_{part.stem}',type='mesh',mesh=part.stem,
                  mass='0',group='3',condim='6',friction='1.0 .006 .001',**attr)
 for side,y in [('right',-.2004),('left',.2004)]:
  b=root.find(f"worldbody/body[@name='{side}_base']");b.set('pos',f'-.21 {y} -.006');b.set('quat','1 0 0 0')
  # Physical CAD fingers, coupled slider actuation.
  for parent in root.iter():
   for element in list(parent):
    if element.get('name','').startswith('right_usb_lead_'):parent.remove(element)
  grip=root.find(f".//body[@name='{side}_gripper']")
  if side=='left':
   cam=grip.find("body[@name='left_camera_assembly']")
   if cam is not None:grip.remove(cam)
   cam=ET.SubElement(grip,'body',name='left_camera_assembly',pos='0 -.08 .065',quat='.9063 -.4226 0 0')
   ET.SubElement(cam,'geom',name='left_camera_housing',type='box',size='.037 .012 .013',rgba='.025 .025 .026 1',contype='0',conaffinity='0',mass='0')
   for x in [-.025,0,.025]:ET.SubElement(cam,'geom',name=f'left_lens_{x}',type='cylinder',size='.005 .001',pos=f'{x} 0 .014',rgba='.008 .012 .014 1',contype='0',conaffinity='0',mass='0')
 equality=ET.SubElement(root,'equality')
 for side in ['right','left']:
  ET.SubElement(equality,'joint',joint1=f'{side}_joint8',joint2=f'{side}_joint7',polycoef='0 1 0 0 0',solref='.003 1')
  actuator=root.find('actuator')
  for elem in list(actuator):
   # Both target channels receive the same command; equality enforces coupling.
   pass
 path=ROOT/'yam_crank.xml';path.write_text(ET.tostring(root,encoding='unicode'))
 model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model);V=np.linalg.inv(T)
 def project(p):
  v=V[:3,:3]@p+V[:3,3];v=K@v;return v[:2]/v[2]
 # Both wrist faces and TCP are read from raw frame 0. Base spacing uses calibration.
 goals={'right':[(745,280),(706,466)],'left':[(1341,256),(1327,468)]};homes={}
 for side in goals:
  idx=[model.joint(f'{side}_joint{i}').qposadr[0] for i in range(1,7)]
  for j in [7,8]:data.qpos[model.joint(f'{side}_joint{j}').qposadr[0]]=.039574
  def err(q):
   data.qpos[idx]=q;mujoco.mj_forward(model,data);body=model.body(f'{side}_gripper').id;site=model.site(f'{side}_grasp_site').id
   orient=data.xmat[body].reshape(3,3)
   return np.r_[(project(data.xpos[body])-goals[side][0])/120,(project(data.site_xpos[site])-goals[side][1])/120,(orient[:,2]-[1,0,0])*.6,(orient[:,0]-[0,-1,0])*.6,q*.15]
  lo=[model.joint(f'{side}_joint{i}').range[0]+.001 for i in range(1,7)];hi=[model.joint(f'{side}_joint{i}').range[1]-.001 for i in range(1,7)]
  sol=least_squares(err,[0,.02,.02,-.1,0,0],bounds=(lo,hi),max_nfev=250);homes[side]=sol.x.tolist()
 # Rebuild the visible blue camera lead from the source image, attached to the wrist.
 for side in homes:
  for i,q in enumerate(homes[side],1):data.qpos[model.joint(f'{side}_joint{i}').qposadr[0]]=q
 mujoco.mj_forward(model,data);bid=model.body('right_gripper').id
 pixel_path=[(603,207),(619,200),(640,190),(676,184),(710,178),(732,182),(731,188),(700,192),(655,192),(629,198)]
 world=np.array([np.r_[xy(p,.23),.23] for p in pixel_path]);local=(world-data.xpos[bid])@data.xmat[bid].reshape(3,3)
 grip=root.find(".//body[@name='right_gripper']")
 for i,(a,b) in enumerate(zip(local[:-1],local[1:])):
  ET.SubElement(grip,'geom',name=f'right_usb_lead_{i}',type='capsule',fromto=' '.join(map(str,np.r_[a,b])),size='.0023',rgba='.025 .27 .61 1',contype='0',conaffinity='0',mass='0')
 path.write_text(ET.tostring(root,encoding='unicode'))
 (ROOT/'robot_pose.json').write_text(json.dumps({'home':homes,'source':'i2rt 120c3c81400171174604e503943f8d1ebc891058 CAD','base_spacing_m':.4008,'estimated':['base midpoint relative to grid','joint pose fit','left camera housing']},indent=2))
 return path,homes

def appearance(builder,shapes):
 split_yam_white_tubes(builder,shapes)
 for i in range(*shapes):
  label=builder.shape_label[i]
  if 'right_camera_housing' in label:builder.shape_color[i]=(.72,.74,.71)
  if 'right_camera_bracket' in label:builder.shape_color[i]=(.83,.84,.80)
  if 'base_geom_0' in label:builder.shape_color[i]=(.04,.04,.043)
  if 'usb_lead' in label:builder.shape_color[i]=(.025,.27,.61)
  if '_contact_tip_' in label:builder.shape_flags[i]&=~int(newton.ShapeFlags.VISIBLE)

def register():
 path,home=prepare()
 grip=GripperSpec(name='video7010_crank',finger_names=('{arm}_joint7','{arm}_joint8'),follower_names=(),finger_open_q=(.039574,.039574),finger_closed_q=(0.,0.),open_width=.039574,span=.075,tcp_body='{arm}_gripper',tcp_offset=(-.000088,.04459,.13976),tcp_offset_rot=(0,0,-.707107,-.707107),approach_axis=(0,0,1),close_axis=(1,0,0),finger_bodies=(('{arm}_tip_left',),('{arm}_tip_right',)),tip_reach=.007)
 (GRIPPERS.replace if grip.name in GRIPPERS else GRIPPERS.register)(grip.name,grip)
 arms=tuple(ArmSpec(name=s,joint_names=tuple(f'{s}_joint{i}' for i in range(1,7)),gripper=grip.name,home_q=tuple(home[s]),home_gripper_q=((.0097 if s=='right' else .039574),)*2) for s in ['right','left'])
 spec=EmbodimentSpec(name='video7010_yam',asset=AssetSpec(path=str(path)),arms=arms,collapse_fixed_joints=False,reset_joint_q=True,enable_self_collisions=False,cross_arm_collisions=True,show_collision_shapes=True,color=(.04,.04,.043),appearance=appearance,gripper_actuation=ActuationSpec(target_ke=4000,target_kd=30,effort_limit=25))
 (EMBODIMENTS.replace if spec.name in EMBODIMENTS else EMBODIMENTS.register)(spec.name,spec)
 return spec.name
