"""Bench fixtures and background placement from the source camera."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import robosimstudio as rss
from robosimstudio.specs import AssetSpec
from layout import xy
ROOT=Path(__file__).resolve().parent

def add(scene):
 def slab(name,ext,pos,color,angle=0):
  scene.add(rss.slab(name=name,half=tuple(np.array(ext)/2),at=(*pos,*Rotation.from_euler('z',angle).as_quat()),color=color))
 def asset(name,pos,yaw=0):
  folder=ROOT/'assets'/name
  scene.add(rss.body(name=name,asset=AssetSpec(path=str(folder/'anchor.obj')),entity='video7010_v2_parts',at=pos,quat=tuple(Rotation.from_euler('z',yaw).as_quat()),mass=.5,static=True))
 asset('camera_mast',(-.145,0,-.006))
 asset('paper_cup',(*xy((242,469)),-.006))
 asset('floral_cup',(*xy((329,397)),-.006))
 asset('keyboard',(*xy((55,472)),-.007),np.pi/2)
 slab('rear_wood_bench',(.48,1.60,.025),(-.41,0,-.0185),(.68,.45,.20))
 # Four legs of the two work surfaces, with the cross rails.
 for x in [-.45,.48]:
  for y in [-.80,.80]:
   top=-.031 if x<0 else -.034
   slab(f'bench_leg_{x}_{y}',(.035,.035,top+.78),(x,y,(top-.78)/2),(.19,.20,.20))
 # Three small blue fixture blocks behind the mat, each a clamp-like slotted shape.
 for k,y in enumerate([-.082,-.114,.082]):
  for side in [-1,1]:slab(f'blue_fixture_{k}_jaw_{side}',(.052,.008,.038),(-.145,y+side*.010,.013),(.075,.075,.18))
  slab(f'blue_fixture_{k}_bridge',(.012,.028,.012),(-.164,y,0.),(.07,.07,.17))
 # Notebook: page block, cover, spine, ruled pages and handwritten-sized strokes.
 p=xy((112,395));slab('notebook_cover',(.10,.14,.003),(*p,.009),(.28,.22,.13),.1)
 slab('notebook_pages',(.096,.135,.003),(*p,.012),(.81,.79,.65),.1)
 rng=np.random.default_rng(8)
 for i in range(11):
  for j in range(4):
   x=p[0]-.036+i*.006;y=p[1]-.046+j*.022
   slab(f'notebook_ink_{i}_{j}',(.0006,rng.uniform(.009,.019),.0001),(x,y,.0136),(.16,.17,.16),rng.uniform(-.1,.1))
 # Background shipping boxes have lid seams, flaps and packing tape.
 for k,(x,y,z,w,d,h) in enumerate([(-.42,.61,.14,.35,.35,.29),(-.51,.80,.24,.42,.32,.48)]):
  slab(f'carton_{k}',(w,d,h),(x,y,z),(.48,.32,.16))
  slab(f'carton_tape_{k}',(w,.042,.0006),(x,y,z+h/2+.0004),(.57,.40,.22))
  slab(f'carton_lid_seam_{k}',(w,.0007,.0007),(x,y,z+h/2+.001),(.18,.12,.065))
 asset('office_chair',(-.60,-.54,-.22))
 # Left display, stand and bezel, angled away from the work area.
 slab('monitor_rear_shell',(.035,.29,.24),(.05,-.76,.15),(.21,.23,.23),-.2)
 slab('monitor_stand',(.10,.12,.008),(.05,-.76,-.002),(.36,.38,.37))
 slab('monitor_pedestal',(.025,.035,.10),(.05,-.76,.052),(.35,.37,.36))
