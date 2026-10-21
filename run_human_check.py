"""Open Gemini robot harness, human-check version: every motion step is held
in viser (both arms now and planned, the point cloud, the path) until you
approve it. Reject sends your feedback to Gemini, which proposes again.

    python run_human_check.py --task "Pick up the clamp with the arm closest to it, hand it over to the other arm, then place the clamp inside the blue bin."
    python run_human_check.py --real --reasoning-effort minimal --time-budget-s 10 --thought-words 40 --task "..."
    python run_human_check.py --serve-only            # no Gemini: POST /plan yourself

Sim (i2rt's MuJoCo) by default; --real drives both arms. See README.md.
"""
from agent.launch import main

if __name__ == "__main__":
    main(checker="human")
