"""Fresh visual reconstruction; asset review precedes policy adaptation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
import robosimstudio as rss
from robosimstudio.specs import CameraSpec, AssetSpec, CellSpec
from robosimstudio.core.render.finish import Finish
from robosimstudio.specs.physics import backend
import native_assets
from layout import LAYOUT,T,K,xy

from robosimstudio.scene.registry import CELLS
cell=CellSpec(name='video7010_table',ground_z=-.78,surface_z=0)
(CELLS.replace if cell.name in CELLS else CELLS.register)(cell.name,cell)
scene=rss.Scene(cell='video7010_table',name='7010 v2 — newly authored assets',physics=rss.MUJOCO_500HZ.with_backend('*',backend('mujoco',nconmax=8192,njmax=16384,iterations=50,ls_iterations=30,impratio=100,enable_multiccd=True)))
# Mat surface, fine blue 25mm ruling and separate foam tiles.
scene.add(rss.slab(name='bench_top',half=(.395,.80,.014),at=(.225,0,-.020),color=(.12,.12,.105)))
scene.add(rss.slab(name='foam_mat',half=(.25,.65,.003),at=(.235,0,-.003),color=(.95,.95,.91)))
for i in range(-26,27):scene.add(rss.slab(name=f'grid_y_{i+26}',half=(.25,.00038,.00003),at=(.235,i*.025,.00004),color=(.16,.37,.62)))
for i in range(20):scene.add(rss.slab(name=f'grid_x_{i}',half=(.00038,.65,.00003),at=(i*.025,0,.00004),color=(.16,.37,.62)))
# Rear jigsaw teeth from the footage, each 25mm pitch. Front/side lie outside camera.
for i in range(-25,26,2):scene.add(rss.slab(name=f'foam_rear_tab_{i+25}',half=(.008,.008,.003),at=(-.023,i*.025,-.003),color=(.95,.95,.91)))
for name,p0,p1 in [('left',(776,469),(694,788)),('right',(1317,452),(1380,800)),('cross',(695,788),(1380,800))]:
 a,b=xy(p0),xy(p1);d=b-a;mid=(a+b)/2;ang=np.arctan2(d[1],d[0]);scene.add(rss.slab(name='yellow_tape_'+name,half=(np.linalg.norm(d)/2,.011,.00008),at=(*mid,.0002,*Rotation.from_euler('z',ang).as_quat()),color=(.87,.77,.035)))
for name,record in LAYOUT.items():
 folder=ROOT/'assets'/name
 scene.add(rss.body(name=name,asset=AssetSpec(path=str(folder/'anchor.obj')),entity='video7010_v2_parts',at=tuple(np.array(record['position'])+[0,0,(-.001 if name=='clamp' else .001)]),quat=tuple(Rotation.from_euler('z',record['yaw']).as_quat()),mass=.15 if name=='clamp' else .25,static=name!='clamp',friction=.8))
from robot import register
scene.add(rss.robot(register()))
from environment import add
add(scene)
# Reference phone and independent close-up camera.
scene.add(CameraSpec(name='phone',width=1920,height=1080,fov_y=float(np.degrees(2*np.arctan(540/K[1,1]))),cx=960,cy=540,xform=(*T[:3,3],*Rotation.from_matrix(T[:3,:3]).as_quat()),channels=('rgb',)))
BUILD={'device':'cuda:0'}

from policy import handover
routines={"Grasp, hand over, place (new assets)":handover}
