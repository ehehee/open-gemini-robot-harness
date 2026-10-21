"""A scripted bimanual handover through the harness, in the RoboSimStudio sim,
recorded as video: no Gemini.

Right arm picks the clamp up by its rubber sleeve and presents it; left arm
receives it by the other handle; right lets go alone and the receiver is
rechecked (the prompt's handover protocol); left carries it over the tray and
releases it. Every motion is a /plan program through the harness, checked by
gap_guard, with grasp poses computed from the scene's meshes.

    python tools/scripted_handover.py              # -> runs/scripted_handover_<stamp>/handover.mp4

The video is in sim time: the phone camera large, the overhead and right
wrist views beside it, and under it the program being run and gap_guard's
verdicts. Exit status 0 = transferred and placed in the tray.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sim.procedures import DEFAULT_SCENE, build, scripted_handover  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--fps", type=float, default=10.0, help="video frames per sim second")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "runs")
    args = ap.parse_args()

    h = build(args.scene, args.out / ("scripted_handover_" + time.strftime("%Y%m%d_%H%M%S")), check=True)
    verdicts: list[str] = []
    step = {"text": "Starting"}

    def caption():
        h.hub.caption = step["text"] + ("\n" + "\n".join(verdicts[-4:]) if verdicts else "")

    def say(text: str) -> None:
        print(text, flush=True)
        step["text"] = text
        verdicts.clear()
        caption()

    checker = h.session.viewer.checker
    checker.on_record = lambda rec: (verdicts.append(f"gap_guard: {rec['label'][:58]}  ->  {rec['verdict']}"), caption())
    if not args.no_video:
        h.hub.start_video(h.run_dir / "handover.mp4", fps=args.fps)
    try:
        r = scripted_handover(h, args.scene.parent, say=say)
        h.hub.sleep(1.0)
    finally:
        video = None if args.no_video else h.hub.stop_video()
        h.close()
    (h.run_dir / "scripted_handover.json").write_text(json.dumps(r, indent=2, default=float))
    t = r.get("transfer", {})
    print(f"\n[result] stage={r['stage']}  transferred={bool(t)}  "
          + (f"clamp moved {t['clamp_drop_on_release_m'] * 1000:.1f} mm on release  " if t else "")
          + (f"placed={r['placement']['placed']} ({r['placement']['inside_fraction'] * 100:.1f}% inside)"
             if "placement" in r else ""))
    if video:
        print(f"[result] video: {video}")
    return 0 if r["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
