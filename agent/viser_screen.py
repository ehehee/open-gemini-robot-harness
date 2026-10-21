"""Screenshots of the whole viser page -- 3D view plus the GUI panels (approval
window, buttons, plan text) -- as a CameraRecorder source.

scene.viser (run_log.SceneRecorder) replays the 3D scene but viser doesn't
serialize GUI panels, so what the operator was actually looking at when they
approved or rejected a step was lost. This opens the viser URL in a headless
Chromium (Playwright) as one more client and screenshots it on demand:

  ViserScreen(url).grab  -> RGB uint8 HxWx3, like OverheadCamera.grab_rgb

GUI state is shared between viser clients, so the headless page shows the
same panels as the operator's browser. Its 3D camera is its own: whatever the
viewer's on_client_connect sets (agent/viewer.py), not the operator's
mouse-driven view.

Playwright's sync API must stay on the thread that started it, so a worker
thread owns the browser. It runs Chrome's screencast (Page.startScreencast),
which pushes a JPEG every time the page repaints -- ~8 fps with software WebGL
on a moving scene, vs ~1.5 fps from page.screenshot(), which forces a fresh
render per call. grab() returns the newest pushed frame, so a frame can be up
to one repaint (~0.1 s) older than the time the recorder stamps on it.

Needs:  pip install playwright && python -m playwright install chromium
"""
from __future__ import annotations

import base64
import io
import threading

import numpy as np
from PIL import Image


# viser shows this toast whenever WebGL falls back to software, which it always
# does in headless Chromium; it would sit over the burned-in clock.
_CLOSE_SOFTWARE_TOAST = """() => {
  for (const n of document.querySelectorAll('.mantine-Notification-root'))
    if (/software/i.test(n.textContent)) n.querySelector('button')?.click();
}"""


class ViserScreen:
    def __init__(self, url: str, size: tuple[int, int] = (1600, 900), quality: int = 85):
        self.url, self.size, self.quality = url, size, quality
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="viser-screen")
        self._thread.start()
        if not self._ready.wait(60.0):
            raise RuntimeError(f"headless Chromium didn't load {url} within 60 s")
        if self._error is not None:
            raise RuntimeError(f"viser screen capture failed to start: {self._error!r}") from self._error

    def _run(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            self._error = ImportError("playwright not installed: pip install playwright && "
                                      "python -m playwright install chromium")
            self._ready.set()
            return
        try:
            with sync_playwright() as p:
                # The rig may have no display; swiftshader gives headless WebGL either way
                # (and was faster than ANGLE-on-Vulkan on an NVIDIA box).
                browser = p.chromium.launch(args=["--enable-unsafe-swiftshader"])
                try:
                    page = browser.new_page(viewport={"width": self.size[0], "height": self.size[1]})
                    page.goto(self.url, wait_until="load", timeout=30_000)
                    page.wait_for_selector("canvas", timeout=30_000)
                    cdp = page.context.new_cdp_session(page)

                    def on_frame(f: dict) -> None:
                        with self._lock:
                            self._jpeg = base64.b64decode(f["data"])
                        cdp.send("Page.screencastFrameAck", {"sessionId": f["sessionId"]})

                    cdp.on("Page.screencastFrame", on_frame)
                    cdp.send("Page.startScreencast", {"format": "jpeg", "quality": self.quality,
                                                      "maxWidth": self.size[0], "maxHeight": self.size[1]})
                    page.wait_for_timeout(1500)  # first frames; the toast appears on load
                    self._ready.set()
                    while not self._stop.is_set():
                        page.evaluate(_CLOSE_SOFTWARE_TOAST)
                        page.wait_for_timeout(1000)  # pumps the screencast events meanwhile
                    cdp.send("Page.stopScreencast")
                finally:
                    browser.close()
        except Exception as e:  # noqa: BLE001 -- surfaced to __init__ / grab()
            self._error = e
            self._ready.set()

    def grab(self) -> np.ndarray:
        if not self._thread.is_alive():
            raise RuntimeError(f"viser screen capture stopped: {self._error!r}")
        with self._lock:
            jpeg = self._jpeg
        if jpeg is None:
            raise RuntimeError("no frame from the viser page yet")
        return np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))

    def close(self) -> None:
        """Idempotent; call after the recorder has stopped."""
        self._stop.set()
        self._thread.join(timeout=10.0)
