"""Open Gemini robot harness, autonomous-check version: gap_guard checks every
motion step before it runs and answers OK / IK_FAILURE / COLLISION. OK
executes; a rejection sends only that word back to Gemini, which proposes
again. No human approves motions; viser still shows everything and its Stop
button stops the session.

    python run_auto_check.py --task "Pick up the clamp with the arm closest to it, hand it over to the other arm, then place the clamp inside the blue bin."
    python run_auto_check.py --real --reasoning-effort minimal --time-budget-s 10 --thought-words 40 --task "..."

Sim (i2rt's MuJoCo) by default; --real drives both arms. See README.md.
"""
from agent.launch import main

if __name__ == "__main__":
    main(checker="auto")
