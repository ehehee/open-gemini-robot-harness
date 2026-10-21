"""Continuous camera video for a run, so it can be lined up with the Gemini
log and any external (phone) video afterwards.

The agent itself only saves one overhead frame per /obs; between turns --
operator review, the arm actually moving -- nothing was recorded. This grabs
each camera at a fixed rate in its own thread and writes

  runs/<stamp>/video/<name>.mp4          H.264, wall clock burned into the corner
  runs/<stamp>/video/<name>_frames.csv   frame, t_unix, t (ISO, ms) per frame

Times are the same clock as gemini/*.json, events.jsonl and transcript.jsonl.
The mp4 plays at the nominal fps; the CSV holds when each frame was really
taken (a camera busy with an /obs grab can make the rate jitter).

Only plain colour is grabbed here (OverheadCamera.grab_rgb / WristCamera.
grab_rgb), never stereo depth, and the cameras' own locks keep it from
interleaving with the agent's /obs grabs.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "DejaVuSansMono-Bold.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


class CameraRecorder:
    def __init__(self, run_dir: Path, sources: dict[str, Callable[[], np.ndarray]], fps: float = 5.0,
                 events=None, max_side: int = 1280):
        """sources: name -> function returning an RGB uint8 HxWx3 frame."""
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH; install it or pass --record-fps 0")
        self.dir = Path(run_dir) / "video"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sources, self.fps, self.max_side = sources, fps, max_side
        self.events = events or (lambda *a, **k: None)
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._run, args=(n, g), daemon=True, name=f"rec-{n}")
                         for n, g in sources.items()]

    def start(self) -> "CameraRecorder":
        for t in self._threads:
            t.start()
        self.events("recording_start", cameras=list(self.sources), fps=self.fps, dir=str(self.dir))
        return self

    def stop(self) -> None:
        """Idempotent; safe from a finally block. Finishes the mp4 files."""
        if self._stop.is_set():
            return
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5.0)
        self.events("recording_end")

    def _run(self, name: str, grab: Callable[[], np.ndarray]) -> None:
        enc, csv, size, n, failed = None, None, None, 0, 0
        font = None
        period = 1.0 / self.fps
        next_t = time.time()
        try:
            while not self._stop.is_set():
                try:
                    rgb = grab()
                    t = time.time()
                except Exception as e:  # noqa: BLE001 -- a camera hiccup shouldn't end the run
                    if not failed:
                        print(f"[rec] {name}: grab failed ({e!r}); retrying")
                        self.events("recording_grab_failed", camera=name, error=repr(e))
                    failed += 1
                    # Back off 1, 2, 4 .. 15 s: a failing ZED grab blocks for ~3 s
                    # holding the GIL, which stalls every other thread (other
                    # cameras, state.csv) -- retrying every second froze them all.
                    self._stop.wait(min(15.0, 2.0 ** (failed - 1)))
                    continue
                if failed:
                    print(f"[rec] {name}: recovered after {failed} failed grabs")
                    self.events("recording_grab_recovered", camera=name, failures=failed)
                failed = 0
                im = Image.fromarray(rgb)
                if max(im.size) > self.max_side:
                    im.thumbnail((self.max_side, self.max_side))
                if enc is None:
                    # libx264 + yuv420p need even dimensions
                    size = (im.width - im.width % 2, im.height - im.height % 2)
                    font = _font(max(14, size[1] // 28))
                    enc = subprocess.Popen(
                        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                         "-s", f"{size[0]}x{size[1]}", "-r", str(self.fps), "-i", "-",
                         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                         str(self.dir / f"{name}.mp4")], stdin=subprocess.PIPE)
                    csv = (self.dir / f"{name}_frames.csv").open("w")
                    csv.write("frame,t_unix,t\n")
                if im.size != size:
                    im = im.crop((0, 0, *size))
                stamp = datetime.fromtimestamp(t).isoformat(sep=" ", timespec="milliseconds")
                d = ImageDraw.Draw(im)
                x0, y0, x1, y1 = d.textbbox((8, 6), stamp, font=font)
                d.rectangle((x0 - 4, y0 - 3, x1 + 4, y1 + 3), fill=(0, 0, 0))
                d.text((8, 6), stamp, font=font, fill=(255, 255, 255))
                enc.stdin.write(im.tobytes())
                csv.write(f"{n},{t:.3f},{datetime.fromtimestamp(t).isoformat(timespec='milliseconds')}\n")
                n += 1
                if n % 50 == 0:
                    csv.flush()
                next_t += period
                delay = next_t - time.time()
                if delay < -period:  # fell behind (camera busy); don't try to catch up in a burst
                    next_t = time.time()
                elif delay > 0:
                    self._stop.wait(delay)
        finally:
            if enc is not None:
                enc.stdin.close()
                enc.wait(timeout=30)
            if csv is not None:
                csv.close()
            print(f"[rec] {name}: {n} frames -> {self.dir / (name + '.mp4')}")
