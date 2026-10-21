"""Complete, recoverable record of a run: every Gemini call and every robot
event, with wall-clock timestamps (ms) for lining up an external video.

runs/<stamp>/
  run_meta.json           argv, task, mode, model, system prompt, git commit,
                          start time; plus code.patch (uncommitted diff) and
                          calib/ (the calibration files the run used)
  gemini/
    call_001_request.json the EXACT request: model, params, full message
                          list in order. Images are not inlined; each image
                          part points at the file holding the exact bytes sent
                          ("image_file") with its sha256.
    call_001_img_01.jpg   ...those bytes, byte-for-byte
    call_001_response.json the raw API response (resp.model_dump()): text,
                          finish_reason, token usage, provider model id, plus
                          t_sent / t_received / latency
  events.jsonl            one line per robot-side event: observe, plan
                          received/parsed, segment previewed, operator
                          decision (+ feedback), execution start / end (+
                          result), watchdog -- each with "t" (ISO, ms) and
                          "t_unix"
  video/<camera>.mp4      continuous camera video (agent/cam_recorder.py,
                          --record-fps), wall clock burned in, with
                          <camera>_frames.csv giving each frame's t_unix
  scene.viser             the viser 3D scene over the whole run (robot
                          previews, live arms, point cloud), in real time;
                          written on exit. Replay offline with viser's
                          player. The GUI panel isn't in it -- the approval
                          window's content is in events.jsonl.

load_request(path) rebuilds the message list with the images re-inlined, so
any call can be re-sent to the API exactly as it was.
"""
from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


def now() -> dict:
    t = time.time()
    return {"t": datetime.fromtimestamp(t).isoformat(timespec="milliseconds"), "t_unix": round(t, 3)}


class EventLog:
    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / "events.jsonl"
        self._lock = threading.Lock()

    def __call__(self, event: str, /, **fields) -> None:
        # positional-only, so fields may themselves be called "kind", "event"...
        line = json.dumps({**now(), "event": event, **fields}, default=str)
        with self._lock, self.path.open("a") as f:
            f.write(line + "\n")


class SceneRecorder:
    """Records every scene message viser sends (meshes, robot poses, point
    clouds) from construction on, then writes run_dir/scene.viser on save().

    viser's serializer only advances its clock on insert_sleep(), so a
    background thread feeds it wall-clock time; without that the whole run
    would replay as a single instant.

    scene.viser.json records the wall-clock time of the recording's t=0, so
    the 3D replay can be lined up with the camera videos and state.csv
    (tools/run_report.py)."""

    def __init__(self, server, run_dir: Path, tick_s: float = 0.1):
        self.path = Path(run_dir) / "scene.viser"
        self._ser = server.get_scene_serializer()
        self.t0_unix = time.time()
        self._stop = threading.Event()
        self._saved = False
        self._thread = threading.Thread(target=self._clock, args=(tick_s,), daemon=True)
        self._thread.start()

    def _clock(self, tick_s: float) -> None:
        last = time.monotonic()
        while not self._stop.wait(tick_s):
            t = time.monotonic()
            self._ser.insert_sleep(t - last)
            last = t

    def save(self) -> Path | None:
        """Idempotent; safe to call from a finally block."""
        if self._saved:
            return None
        self._saved = True
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.path.write_bytes(self._ser.serialize())
        self.path.with_suffix(".viser.json").write_text(
            json.dumps({"t0_unix": self.t0_unix, "t_saved_unix": time.time()}))
        return self.path


class GeminiLog:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir) / "gemini"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def request(self, model: str, messages: list[dict], params: dict) -> int:
        self.n += 1
        k = self.n
        out, img_i = [], 0
        for m in messages:
            content = m["content"]
            if isinstance(content, list):
                parts = []
                for p in content:
                    if p.get("type") == "image_url":
                        img_i += 1
                        url = p["image_url"]["url"]
                        header, b64 = url.split(",", 1)
                        raw = base64.b64decode(b64)
                        ext = "jpg" if "jpeg" in header else header.split("/")[1].split(";")[0]
                        name = f"call_{k:03d}_img_{img_i:02d}.{ext}"
                        (self.dir / name).write_bytes(raw)
                        parts.append({"type": "image_url", "image_file": name, "mime_header": header,
                                      "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
                    else:
                        parts.append(p)
                out.append({**m, "content": parts})
            else:
                out.append(m)
        rec = {"call": k, **now(), "model": model, "params": params, "messages": out}
        (self.dir / f"call_{k:03d}_request.json").write_text(json.dumps(rec, indent=1))
        return k

    def response(self, k: int, resp, t_sent: float, error: str | None = None) -> None:
        t = time.time()
        rec = {"call": k, "t_sent": datetime.fromtimestamp(t_sent).isoformat(timespec="milliseconds"),
               "t_received": datetime.fromtimestamp(t).isoformat(timespec="milliseconds"),
               "latency_s": round(t - t_sent, 3)}
        if error is not None:
            rec["error"] = error
        else:
            try:
                rec["raw"] = resp.model_dump()
            except Exception:  # noqa: BLE001
                rec["raw"] = str(resp)
        (self.dir / f"call_{k:03d}_response.json").write_text(json.dumps(rec, indent=1, default=str))


def load_request(path: str | Path) -> dict:
    """Read a call_NNN_request.json back with every image re-inlined as the
    data URL that was sent -- ready to pass to chat.completions.create."""
    path = Path(path)
    rec = json.loads(path.read_text())
    for m in rec["messages"]:
        if isinstance(m["content"], list):
            for p in m["content"]:
                if p.get("type") == "image_url" and "image_file" in p:
                    raw = (path.parent / p["image_file"]).read_bytes()
                    assert hashlib.sha256(raw).hexdigest() == p["sha256"], f"{p['image_file']} changed"
                    p["image_url"] = {"url": p["mime_header"] + "," + base64.b64encode(raw).decode()}
                    for key in ("image_file", "mime_header", "sha256", "bytes"):
                        p.pop(key)
    return rec


def write_run_meta(run_dir: Path, args, system_prompt: str, model: str, calib_files: list[Path]) -> None:
    run_dir = Path(run_dir)
    repo = Path(__file__).resolve().parents[1]

    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, timeout=10).stdout
        except Exception:  # noqa: BLE001
            return ""

    (run_dir / "code.patch").write_text(git("diff", "HEAD") + "\n# untracked files:\n" +
                                        git("ls-files", "--others", "--exclude-standard"))
    cdir = run_dir / "calib"
    cdir.mkdir(exist_ok=True)
    for f in calib_files:
        if Path(f).exists():
            shutil.copy2(f, cdir / Path(f).name)
    meta = {**now(), "argv": sys.argv, "args": vars(args), "model": model,
            "system_prompt": system_prompt, "git_commit": git("rev-parse", "HEAD").strip(),
            "git_dirty": bool(git("status", "--porcelain").strip()),
            "calib_files": [str(f) for f in calib_files]}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str))
