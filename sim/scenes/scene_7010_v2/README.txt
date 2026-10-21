Vendored from the RoboSimStudio reconstruction of the clamp-handover bench
(IMG_7010), for open-gemini-robot-harness's simulation. Changes from the
original: robot.py reads the arm meshes from robot/ (i2rt CAD at 120c3c8,
robot/PROVENANCE.txt, robot/LICENSE.i2rt) instead of an absolute path; the
recordings, review images and live/ run folder are not included. The
harness builds this scene itself (sim/rss_backend.py) and adds the
overhead and right-wrist cameras; ./launch.sh below is the original
standalone way to open it in a RoboSimStudio live page.

IMG_7010 reconstruction and physical handover

Launch: ./launch.sh
Native RoboSimStudio page: http://localhost:8100
Run ./run_policy.sh, or Reset then select "Grasp, hand over, place (new assets)".
GPU visibility was confirmed; the live scene uses physical GPU 2.

Reference: /home/shadeform/gasket/IMG_7010 2.MOV, supplemented by the
IMG_7047.MOV rotating clamp close-up. Props were independently authored;
rejected prop meshes were not reused. Replacements include the curved split
hammer claw and forged head, molded brush handles and bristles, traced cable
bundle/connectors, measured tape roll, layered folding hex-key set, clamp,
tray, fixtures and surrounding furniture. The tape is approximately 105 mm
OD and 77 mm ID, assuming a 25 mm height. Unseen geometry remains estimated.

The robot retains manufacturer YAM/crank CAD from i2rt commit
120c3c81400171174604e503943f8d1ebc891058. Finger collision pieces were rebuilt
from separate watertight CAD components without voxel preprocessing to align
physical contact with visible surfaces. Both fingers are driven and coupled.
The grip is centered on the straight rubber sleeve instead of its tapered end.

Validation completed on 2026-10-19:
- All native scene checks pass (live/check.json).
- Full physical grasp, lift/hold, handover, carry, release and withdrawal pass.
- Entire clamp mesh lies inside the tray; tray support force is 1.471 N.
- Final speed is 0.0000061 m/s; hand clearance is 204 mm.
- Maximum relative grip motion: donor 1.74 mm / 2.09 degrees;
  receiver 3.47 mm / 2.57 degrees. This is not zero slip.
- Cross-arm collisions are enabled. The policy uses joint commands and
  contact feedback, with no object attachment or object-pose commands.
- Policy aborts if held-object drift exceeds 4 mm or 5 degrees.

Rubber friction is a simulation assumption: sliding coefficient 1.2,
torsional parameter 0.006 and rolling parameter 0.001, with 6D contacts.
These material values were not measured from video. The successful rollout
validates this initial configuration, not robustness to all perturbations.

Artifacts:
  handover_validated.mp4       successful 34.9-second physical rollout
  handover_result.json        measured task and grip outcomes
  grasp_contact.png           visible donor contact close-up
  handover_contact.png        receiver contact close-up
  contact_gap_audit.json      visible-surface audit after donor release
  placement_result.png        released clamp in tray
  current_scene_vs_ref.png    reconstruction / source comparison
  asset_audit.json            geometry dimensions and provenance
