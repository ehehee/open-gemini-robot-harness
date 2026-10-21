"""A scripted grasp through the harness, in the RoboSimStudio sim: no Gemini.

Computes a good grasp from the object's geometry (the scene's own mesh: the
clamp's rubber sleeve, centred on its straight section, jaws closing across
it), writes it as the /plan program Gemini would send --

    right.grasp(x=..., y=..., z=..., yaw_deg=...)
    right.move_by(dz=0.04, speed=0.04)          # --extra-lift

-- runs it through BimanualSession.run_plan with gap_guard checking every
step, exactly as run_auto_check.py would, and then checks physically that the
object came up with the gripper (grasp() ends by lifting 8 cm) and is still in
the jaws after holding it there. Exit status 0 = grasped and lifted.

    python tools/scripted_grasp.py                 # scene_7010_v2, the clamp
    python tools/scripted_grasp.py --no-check      # every step approved, no gap_guard

The grasp pose uses the sim's ground truth (the object's mesh and pose), so
this tests the harness below the model -- skills, IK, checking, execution,
gripper, physics -- not Gemini's perception.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sim.procedures import DEFAULT_SCENE, build, scripted_grasp  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--object", default="clamp")
    ap.add_argument("--part", default="0008.obj", help="mesh of the part to grasp (the clamp's rubber sleeve)")
    ap.add_argument("--inward", type=float, default=0.017, help="m toward the object's middle along the part")
    ap.add_argument("--extra-lift", type=float, default=0.04, help="m to lift after grasp()'s own 8 cm (0 = none)")
    ap.add_argument("--hold", type=float, default=2.0, help="sim seconds to hold before checking")
    ap.add_argument("--no-check", action="store_true", help="approve every step instead of gap_guard")
    ap.add_argument("--out", type=Path, default=REPO / "runs")
    args = ap.parse_args()

    h = build(args.scene, args.out / ("scripted_grasp_" + time.strftime("%Y%m%d_%H%M%S")), check=not args.no_check)
    try:
        r = scripted_grasp(h, args.scene.parent, args.object, args.part, args.inward, args.extra_lift, args.hold)
    finally:
        h.close()
    (h.run_dir / "scripted_grasp.json").write_text(json.dumps(r, indent=2, default=float))
    print(f"\n[result] holding={r['holding']}  gripper_open={r['gripper_open']:.3f}  {args.object} rose "
          f"{r['object_rise_m'] * 1000:.1f} mm with the gripper lifted {r['lifted_m'] * 1000:.0f} mm, after "
          f"{r['hold_s']:g} s held  -> {'GRASPED AND LIFTED' if r['success'] else 'FAILED'}")
    print(f"[result] {h.run_dir / 'scripted_grasp.json'}  (images: obs_*.png)")
    return 0 if r["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
