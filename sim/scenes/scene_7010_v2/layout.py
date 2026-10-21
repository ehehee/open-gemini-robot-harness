"""Camera and placements from measured raw-frame grid intersections, no prior meshes."""
from pathlib import Path
import json
import numpy as np
import cv2
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parent
PIX=np.array([[1010.7,503.3],[1188.7,498.3],[997.3,693.3],[1203.3,690.0]])
WORLD=np.array([[0,0],[0,.125],[.15,0],[.15,.125]])
H=cv2.getPerspectiveTransform(WORLD.astype('float32'),PIX.astype('float32'));HI=np.linalg.inv(H)
def xy(pixel,z=0):
 ray=T[:3,:3]@np.linalg.solve(K,np.r_[pixel,1]);return (T[:3,3]+ray*((z-T[2,3])/ray[2]))[:2]
def pose(pixel,forward,z=0):
 p=xy(pixel,z);d=xy(forward,z)-p;return [*p,0],float(np.arctan2(d[1],d[0]))
K=np.array([[993.,0,960],[0,993.,540],[0,0,1.]])
_,r,t=cv2.solvePnP(np.c_[WORLD,np.zeros(4)],PIX,K,None,flags=cv2.SOLVEPNP_ITERATIVE)
def residual(q):
 k=K.copy();k[0,0]=k[1,1]=q[6];return (cv2.projectPoints(np.c_[WORLD,np.zeros(4)],q[:3],q[3:6],k,None)[0][:,0]-PIX).ravel()
q=least_squares(residual,np.r_[r.ravel(),t.ravel(),993.],x_scale='jac').x
R=cv2.Rodrigues(q[:3])[0];T=np.eye(4);T[:3,:3]=R.T;T[:3,3]=-R.T@q[3:6];K[0,0]=K[1,1]=q[6]
LAYOUT={}
for name,p,f in [('brush_25',(715,792),(791,784)),('brush_50',(671,902),(761,935)),('hammer',(996,1021),(1002,1070)),('clamp',(734,700),(831,626)),('tray',(1370,807),(1600,850)),('tape_roll',(403,618),(450,618)),('cable_coil',(1797,589),(1910,590)),('folding_tool',(274,833),(310,750))]:
 height={'clamp':.013,'brush_25':.006,'brush_50':.006,'hammer':.012,'tray':.003,'tape_roll':.0125,'cable_coil':.012,'folding_tool':.01}[name]
 pos,yaw=pose(p,f,height);LAYOUT[name]=dict(position=pos,yaw=yaw)
if __name__=='__main__':
 (ROOT/'measurements.json').write_text(json.dumps(dict(source='IMG_7010 2.MOV frame 0',grid_pitch_m=.025,pixels=PIX.tolist(),world=WORLD.tolist(),camera_to_world=T.tolist(),K=K.tolist(),layout=LAYOUT,assumptions=['zero lens distortion','centered principal point','depths from visible side views; hidden surfaces estimated']),indent=2));print(json.dumps(LAYOUT,indent=2))

# Tape centre fitted from the observed top-rim contour, not its bounding box.
LAYOUT['tape_roll']['position']=[0.10198453433804093, -0.42346006751945053, 0]
