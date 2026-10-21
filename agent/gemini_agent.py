"""The agent loop from the Gemini 4 Argon robotics post: each turn the model
writes one observation sentence (+ `# Thought:`, and `# Plan:` on turn 1)
followed by ONE curl command against the local robot server; the response
and the new camera images come back as the next turn.

The curl is not run in a shell -- that would let model output run arbitrary
commands. It's parsed (URL + body) and only GET /obs or POST /plan on the
local server are honoured.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import shlex
import time
import urllib.request
from pathlib import Path

import numpy as np
from openai import OpenAI
from PIL import Image

from agent.run_log import GeminiLog
from vlm.gemini_client import DEFAULT_MODEL, OPENROUTER_BASE_URL

SYSTEM_PROMPT = """You are Gemini 4 Argon controlling a real 6-DoF YAM robot arm with a
parallel-jaw gripper, through a local HTTP robot server. Complete the user's
task in a closed loop: observe, act with motion primitives, verify visually
after every turn in the updated images.

## Turn format
Simply output 1 sentence observation followed immediately by your `curl`
command. On Turn 1 also add a `# Plan:` line with your overall plan, and
select VLM call granularity upfront on Turn 1 (how many primitives you will
send per /plan call -- fewer means more visual checks). Every turn includes a
`# Thought:` line. Exactly ONE curl command per turn. Nothing else.

## Endpoints
Observe (no motion):
curl -s http://127.0.0.1:8081/obs

Act (body is a Code-as-Policies program, one primitive call per line):
curl -s -X POST http://127.0.0.1:8081/plan --data-binary @- <<'EOF'
x, y = robot.deproject(u=0.52, v=0.48, z_plane=0.015)
robot.grasp(x=x, y=y, z=0.015, yaw_deg=-35)
EOF

/plan returns {"status", "executed": [per-segment results], "obs": {...}} and
you then receive the new images. status is one of completed, plan_error,
grasp_failed, dropped (holding was lost during a move: the object slipped out), blocked (a move stopped well
short of its target: something is in the way -- the other arm, the object, the table; never just retry it), rejected_by_operator, execution_error, session_stopped, overhead_unavailable.

A human operator previews every motion segment in 3D and must approve it
before the arm moves. If they reject, their feedback is in the response --
follow it. Programs may only contain robot.<primitive>(...) calls,
assignments from robot.deproject*, and arithmetic on numbers/variables.

## Frames and units
Arm base frame: +x forward (away from the arm base), +y left, +z up. Meters
and degrees. Positions are of the PINCH CENTER (between the fingertips;
nothing of the gripper is lower than it in a top-down grasp). The table is at
obs.table_z; never command z below it. Stay inside obs.workspace.

## Images
1. Overhead camera (calibrated). A 2 cm world grid is drawn at height
   obs.grid_z (white; thicker lines every 10 cm, each 10 cm intersection
   labelled with its x (yellow) and y (cyan)). The reticle: green circle =
   pinch center, green bar = the line the jaws close along (its length shows
   how open they are), orange = from the pinch center toward the wrist (the
   gripper body), magenta = vertical drop from the pinch center to the
   grid plane (its length shows height above grid_z). If obs.view is a zoom,
   the image is a crop centred on that point.
2. Wrist camera (if present): a close-up from a camera on the gripper,
   calibrated, with the SAME 2 cm world grid (at obs.grid_z, 10 cm
   intersections labelled) and the same green pinch reticle / jaw bar and
   magenta drop line drawn on it. Use it for fine alignment: the jaw bar
   should cross the handle, the magenta dot should sit on it. You still
   deproject through the overhead image only.

## Primitives
robot.look_at(x, y, z, grid_z=<keep>, zoom=2.0)
    No motion. Re-centres/zooms the overhead image on (x, y, z) and moves the
    grid to grid_z, starting from the NEXT image you receive. zoom=1 = full frame.
x, y = robot.deproject(u, v, z_plane)
    Normalized image coords (0..1, origin top-left) of the LAST overhead image
    you received -> the world point on the plane z = z_plane.
x, y, z = robot.deproject_depth(u, v)
    Same pixel, but returns the 3D surface point from stereo depth (noisy, ~1-2 cm).
robot.grasp(x, y, z, yaw_deg=<keep>, pitch_deg=90, azimuth_deg=0, approach_height=0.08, speed=0.10)
    Open, move to approach_height above (x, y, z), descend straight to z,
    close until contact, lift back up. pitch_deg=90 is top-down. For top-down,
    yaw_deg is the XY direction the JAWS CLOSE ALONG (0 = along +x, 90 = along
    +y): to grip a handle lying along direction t, use yaw_deg = t + 90.
    Sets obs.holding; status grasp_failed if nothing was caught.
robot.place(x, y, z, yaw_deg=<keep>, approach_height=0.08, speed=0.10)
    Above, descend to z, open, retreat up.
robot.place_until_contact(x, y, z, yaw_deg=<keep>, max_lag=0.012, relief=0.002, approach_height=0.08, speed=0.10)
    Above, descend until contact (searches to 2 cm below z), back off relief, open, retreat.
robot.goto(x, y, z, arc_z=<none>, yaw_deg=<keep>, pitch_deg=<keep>, azimuth_deg=<keep>, roll_deg=0, speed=0.10)
    Straight-line move of the pinch center. With arc_z: rise to arc_z, traverse, descend.
    pitch_deg = approach angle below horizontal (90 = down); azimuth_deg = heading of a tilted approach.
robot.move_by(forward=0, left=0, dz=0, dyaw_deg=0, dpitch_deg=0, droll_deg=0, speed=0.05)
    Relative straight-line move; forward/left are +x/+y of the base frame.
robot.move_until_contact(dz, speed=0.03, max_lag=0.012, relief=0.002)
    Vertical guarded move; stops when the arm lags the command by max_lag (contact), backs off relief.
robot.open_gripper()   robot.close_gripper()   robot.home()   robot.done()
    close_gripper closes until contact. home returns to the start pose.
    Call robot.done() (alone) when the task is verified complete.

## Tips
- Look before you act: zoom in with look_at on the target first, then
  deproject on the zoomed image -- deprojection is more precise there.
- A brush handle is a few cm tall: grasp z is about table_z + half the handle height.
- Reach: pointing straight down (pitch_deg=90) runs out of wrist range
  beyond about x=0.43 (and lower near the edge). If a segment comes back
  blocked by IK, retry with pitch_deg=80 -- yaw_deg still sets the jaw
  direction there -- or grasp a point on the object closer to the base.
- After a grasp, check holding and the images. On grasp_failed, re-observe
  (look_at) and retry with a corrected point/yaw; don't repeat the same command.
"""


def extract_curl(reply: str) -> tuple[str, str | None]:
    """-> ("obs", None) or ("plan", code). Raises ValueError with a message
    the model can act on."""
    text = re.sub(r"```[a-zA-Z]*\n?", "", reply)
    m = re.search(r"\bcurl\b", text)
    if not m:
        raise ValueError("No curl command found. Output 1 observation sentence, # Thought:, then ONE curl.")
    cmd = text[m.start():]
    url = re.search(r"https?://(?:127\.0\.0\.1|localhost):8081(/\w+)", cmd)
    if not url or url.group(1) not in ("/obs", "/plan"):
        raise ValueError("The curl must target http://127.0.0.1:8081/obs or /plan.")
    if url.group(1) == "/obs":
        return "obs", None
    heredoc = re.search(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n(.*?)\n\s*\1\b", cmd, re.S)
    if heredoc:
        return "plan", heredoc.group(2)
    first = cmd.replace("\\\n", " ")
    try:
        argv = shlex.split(first.split("\n\n")[0])
    except ValueError as e:
        raise ValueError(f"Could not parse the curl command ({e}). Use the heredoc form from the instructions.")
    for i, a in enumerate(argv):
        if a in ("-d", "--data", "--data-raw", "--data-binary") and i + 1 < len(argv):
            body = argv[i + 1]
            try:
                return "plan", json.loads(body)["code"]
            except (ValueError, KeyError, TypeError):
                return "plan", body
    raise ValueError("POST /plan needs a body. Use the heredoc form from the instructions.")


def truncated_reply_message(max_tokens: int) -> str:
    """What the model is told when its reply hit max_tokens. Without this it
    got a misleading format error (no curl yet) or, with the heredoc cut
    before EOF, a plan_error from a garbled program -- 6 of 30 turns in
    runs/bimanual_20261014_191740, 310 s of Gemini time, nothing moved."""
    return (f"Your reply was cut off at the {max_tokens}-token limit before it ended, so NOTHING was run "
            f"(a cut-off command is never executed). Keep the reasoning short: reply with the 1-sentence "
            f"observation, a brief # Thought:, and ONE complete curl command.")


def _data_url(img: np.ndarray, max_side: int = 1280) -> str:
    im = Image.fromarray(img)
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class GeminiAgent:
    def __init__(self, session, viewer, run_dir: Path, port: int = 8081, max_turns: int = 30,
                 keep_images_turns: int = 2, system_prompt: str = SYSTEM_PROMPT, max_tokens: int = 12000,
                 reasoning_effort: str | None = None, reasoning_tokens: int | None = None,
                 thought_words: int | None = None, time_budget_s: float | None = None):
        """reasoning_effort / reasoning_tokens: OpenRouter's `reasoning` control
        over the model's HIDDEN thinking (effort "none".."high", or a token
        budget); unset = the provider default. thought_words: tell the model
        up front how long its visible `# Thought:` may be and that a reply
        over max_tokens is cut off and not run -- by default it only learns
        about the limit after hitting it (truncated_reply_message)."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY (see .env.example).")
        self.model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        self.session, self.viewer, self.run_dir = session, viewer, run_dir
        self.base = f"http://127.0.0.1:{port}"
        self.max_turns, self.keep_images_turns = max_turns, keep_images_turns
        # 4000 cut off 6 of 30 replies in a bimanual run; 12000 leaves room for
        # long geometry reasoning. A reply that still hits it is never executed.
        self.max_tokens = max_tokens
        self.reasoning = None
        if reasoning_tokens is not None:
            self.reasoning = {"max_tokens": int(reasoning_tokens)}
        elif reasoning_effort is not None:
            self.reasoning = {"effort": reasoning_effort}
        if time_budget_s is not None:
            # ~50 tokens/s measured for this model on 2026-10-15 (102 replies; median 13.6 s,
            # 20 of them over 30 s, the longest 156 s of deliberation while an arm held a load).
            budget_tokens = int(time_budget_s * 45)
            system_prompt += (f"\n\n## Time limit\nYou must answer within {time_budget_s:g} seconds of receiving "
                              f"each message -- about {budget_tokens} tokens in total. An arm may be holding an object "
                              f"while you think. Decide quickly: one observation sentence, a `# Thought:` of one or "
                              f"two short sentences, then the curl. Do not re-derive geometry you already worked "
                              f"out; if unsure, take a small safe step and look again.\n")
        if thought_words is not None:
            system_prompt += (f"\n\n## Length\nKeep `# Thought:` to at most {thought_words} words: decide, "
                              f"don't deliberate in writing. Your whole reply has a hard limit of {max_tokens} "
                              f"tokens; a reply cut off at that limit is NOT executed.\n")
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self.glog = GeminiLog(run_dir)  # every request + raw response, recoverable

    API_RETRIES = 3
    API_RETRY_WAIT_S = (3.0, 10.0, 20.0)

    def _complete(self, max_tokens: int):
        """The ONLY way the loops call the model: logs the exact request
        (images as their exact bytes) before sending and the raw response
        after, including failures.

        Retries API failures: on 2026-10-15 OpenRouter returned a malformed
        (non-JSON) body after 53 s, the exception ended the program, and the
        shutdown released both arms -- the right one dropped the clamp it was
        holding. Raises only after API_RETRIES failed attempts."""
        params = {"max_tokens": max_tokens}
        if self.reasoning is not None:
            params["extra_body"] = {"reasoning": self.reasoning}
        for attempt in range(1, self.API_RETRIES + 1):
            k = self.glog.request(self.model, self.messages, params)
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=self.messages, **params)
                if not getattr(resp, "choices", None):
                    raise RuntimeError(f"API response has no choices: {resp!r}"[:300])
            except Exception as e:  # noqa: BLE001 -- network / provider / malformed body
                self.glog.response(k, None, t0, error=repr(e))
                if attempt == self.API_RETRIES:
                    raise
                wait = self.API_RETRY_WAIT_S[attempt - 1]
                print(f"[agent] Gemini API call failed ({e!r}); retry {attempt}/{self.API_RETRIES - 1} in {wait:.0f} s")
                time.sleep(wait)
                continue
            self.glog.response(k, resp, t0)
            return resp

    def _http(self, kind: str, code: str | None) -> dict:
        if kind == "obs":
            req = urllib.request.Request(self.base + "/obs")
        else:
            req = urllib.request.Request(self.base + "/plan", data=code.encode(), method="POST",
                                         headers={"Content-Type": "text/plain"})
        with urllib.request.urlopen(req, timeout=None) as r:  # /plan blocks on the operator
            return json.loads(r.read())

    def _user_turn(self, text: str, with_images: bool) -> dict:
        content = [{"type": "text", "text": text}]
        if with_images:
            for name, img in self.session.last_images.items():
                content.append({"type": "text", "text": f"[{name} camera]"})
                content.append({"type": "image_url", "image_url": {"url": _data_url(img)}})
        return {"role": "user", "content": content}

    def _trim_images(self) -> None:
        """Keep images only in the last few user turns -- every turn carries
        1-2 full frames, and old ones cost tokens without helping."""
        user_idx = [i for i, m in enumerate(self.messages) if m["role"] == "user"]
        for i in user_idx[:-self.keep_images_turns]:
            c = self.messages[i]["content"]
            if isinstance(c, list):
                self.messages[i]["content"] = [p if p["type"] == "text" else
                                               {"type": "text", "text": "(older image omitted)"} for p in c]

    def run(self, task: str) -> None:
        log = (self.run_dir / "transcript.jsonl").open("a")
        self.messages.append(self._user_turn(f"Task: {task}\n\nTurn 1.", with_images=False))
        for turn in range(1, self.max_turns + 1):
            while self.viewer.pause.value and not self.viewer.stop_requested:
                time.sleep(0.3)
            if self.viewer.stop_requested:
                print("[agent] stopped by operator")
                break
            t0 = time.time()
            try:
                resp = self._complete(max_tokens=self.max_tokens)
            except Exception as e:  # noqa: BLE001 -- give up on the model, NOT on the robot
                print(f"[agent] Gemini API failed {self.API_RETRIES} times ({e!r}); stopping the agent loop. "
                      f"The arms stay powered where they are.")
                log.write(json.dumps({"turn": turn, "error": f"API failed: {e!r}", "t": time.time()}) + "\n")
                break
            reply = (resp.choices[0].message.content or "").strip()
            truncated = resp.choices[0].finish_reason == "length"
            print(f"\n=== Turn {turn} ({time.time() - t0:.1f}s) ===\n{reply}")
            self.messages.append({"role": "assistant", "content": reply})
            self.viewer.set_turn(turn, "```\n" + reply + "\n```")
            entry = {"turn": turn, "reply": reply, "t": time.time()}
            if truncated:  # never run a command from a cut-off reply
                entry.update(error="reply cut off at max_tokens", truncated=True)
                text, images = truncated_reply_message(self.max_tokens), False
                print(f"[agent] reply hit max_tokens={self.max_tokens}; not executed")
            else:
                try:
                    kind, code = extract_curl(reply)
                    result = self._http(kind, code)
                    entry.update(kind=kind, code=code, result=result)
                    text = (f"Response from {'GET /obs' if kind == 'obs' else 'POST /plan'}:\n"
                            + json.dumps(result, indent=1))
                    images = True
                except ValueError as e:
                    entry.update(error=str(e))
                    text, images = f"Format error: {e}", False
            log.write(json.dumps(entry, default=str) + "\n")
            log.flush()
            imgs = self.session.last_images if images else {}
            self.viewer.set_turn(turn, "```\n" + reply + "\n```", imgs.get("overhead"), imgs.get("wrist"))
            status = entry.get("result", {}).get("status") or entry.get("result", {}).get("obs", {}).get("status")
            if status in ("done", "session_stopped") or self.session.state.status == "done":
                print(f"[agent] finished: {status or 'done'}")
                break
            self.messages.append(self._user_turn(text + f"\n\nTurn {turn + 1}.", with_images=images))
            self._trim_images()
        log.close()
