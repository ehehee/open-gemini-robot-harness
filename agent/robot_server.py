"""Local robot server in the shape the Gemini 4 Argon robotics post uses:

    GET  http://127.0.0.1:8081/obs   -> observation JSON (+ image paths)
    POST http://127.0.0.1:8081/plan  -> Code-as-Policies program; blocks until
                                        it has run (or been rejected), then
                                        returns the log + a fresh observation

The difference from the post: every motion segment is held until a human
approves it in the viser UI (agent/viewer.py). Rejections go back to the
model as the /plan response, with the operator's feedback.

It's plain HTTP on localhost, so you can also drive it by hand with curl.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from agent.observer import Observer
from agent.plan_program import PlanError, parse_plan
from agent.primitives import SIGNATURES, ArmState, Primitives, describe_orientation
from agent.viewer import ApprovalViewer


class RobotSession:
    def __init__(self, prims: Primitives, observer: Observer, viewer: ApprovalViewer, grab_overhead,
                 grab_wrist, run_dir: Path, close_duration_s: float = 2.0, wrist_cam=None, wrist_view=None,
                 events=None):
        """wrist_cam + wrist_view (agent/wrist_view.py): when given, the wrist
        image gets the same world grid + reticle as the overhead one, and its
        depth goes into viser as a second point cloud."""
        self.wrist_cam, self.wrist_view = wrist_cam, wrist_view
        self.events = events or (lambda *a, **k: None)  # agent/run_log.EventLog
        self.prims, self.obs_, self.viewer = prims, observer, viewer
        self.grab_overhead, self.grab_wrist = grab_overhead, grab_wrist
        self.run_dir = run_dir
        self.close_duration_s = close_duration_s
        self.lock = threading.Lock()
        self.state = prims.actual_state()
        self.n_obs = 0
        self.last_images: dict[str, np.ndarray] = {}
        self.last_plan_report: dict | None = None

    # -- /obs ---------------------------------------------------------------------
    def observe(self) -> dict:
        from PIL import Image

        self.obs_.set_frame(self.grab_overhead())
        self.state = self.prims.actual_state(self.state)
        T = self.prims.fk(self.state.q)
        overhead = self.obs_.render(T, self.state.gripper)
        self.n_obs += 1
        o_path = self.run_dir / f"obs_{self.n_obs:03d}_overhead.png"
        overhead.save(o_path)
        out = {
            "ee_pos": [round(float(v), 4) for v in T[:3, 3]],
            **describe_orientation(T[:3, :3]),
            "gripper_open": round(self.state.gripper, 3),
            "holding": self.state.holding,
            "status": self.state.status,
            "table_z": round(self.obs_.table_z, 4),
            "grid_z": round(self.obs_.view.grid_z, 4),
            "view": ({"center": [round(float(v), 3) for v in self.obs_.view.center], "zoom": self.obs_.view.zoom}
                     if self.obs_.view.center is not None else "full overhead frame"),
            "workspace": self.prims.ws.as_dict(),
            "overhead_cam_path": str(o_path),
        }
        self.last_images = {"overhead": np.asarray(overhead)}
        if self.wrist_view is not None:
            rgb, depth = self.wrist_cam.grab_rgbd()
            q_now = self.state.q_measured if self.state.q_measured is not None else self.state.q
            T_now = self.prims.fk(q_now)
            wrist = np.asarray(self.wrist_view.render(rgb, q_now, T_now, self.state.gripper,
                                                      self.obs_.view.grid_z, self.obs_.ws))
            w_path = self.run_dir / f"obs_{self.n_obs:03d}_wrist.png"
            Image.fromarray(wrist).save(w_path)
            out["wrist_cam_path"] = str(w_path)
            self.last_images["wrist"] = wrist
            if depth is not None:
                self.viewer.set_wrist_cloud(*self.wrist_view.cloud(rgb, depth, q_now))
        elif self.grab_wrist is not None:
            wrist = self.grab_wrist()
            w_path = self.run_dir / f"obs_{self.n_obs:03d}_wrist.png"
            Image.fromarray(wrist).save(w_path)
            out["wrist_cam_path"] = str(w_path)
            self.last_images["wrist"] = wrist
        # Show the images the moment they're captured -- not only after a plan
        # has run, which left the panel blank while the first step waited.
        self.viewer.image.image = self.last_images["overhead"]
        if "wrist" in self.last_images:
            self.viewer.wrist.image = self.last_images["wrist"]
        self.viewer.set_cloud(*self.obs_.cloud_base())
        self.viewer.set_now(self.state.q_measured if self.state.q_measured is not None else self.state.q, self.state.gripper)
        self.events("observe", n=self.n_obs, ee_pos=out["ee_pos"], gripper_open=out["gripper_open"],
                    holding=out["holding"], status=out["status"], images=[v for k, v in out.items() if k.endswith("_path")])
        return out

    # -- /plan --------------------------------------------------------------------
    def run_plan(self, code: str) -> dict:
        self.events("plan_received", code=code)
        try:
            calls = parse_plan(code, SIGNATURES, pure={"deproject": self.obs_.deproject,
                                                       "deproject_depth": self.obs_.deproject_depth})
        except (PlanError, ValueError) as e:
            self.events("plan_error", error=str(e))
            return {"status": "plan_error", "error": str(e), "executed": []}

        ops = [op for i, c in enumerate(calls) for op in self.prims.expand(c.name, c.kwargs, i)]
        log, results = [], {}
        self.state = self.prims.actual_state(self.state)
        self.state.status = "ok"
        failed_call = None
        segs = self.prims.preview(ops, self.state)
        outcome = "completed"
        for idx, op in enumerate(ops):
            if failed_call is not None and op.call_index != failed_call:
                results[idx] = "skipped"
                log.append({"segment": op.label, "result": "skipped (earlier grasp failed)"})
                continue
            # The preview chained each segment from the previous one's
            # predicted end. If the arm isn't there (contact stop, tracking
            # error, a failed IK upstream), re-resolve everything from here
            # from where it ACTUALLY is.
            seg = segs[idx]
            if seg.q_path is not None and np.abs(seg.q_path[0] - self.state.q).max() > 0.01 or (
                    seg.q_path is None and seg.op.kind in ("move", "contact")):
                segs[idx:] = self.prims.preview(ops[idx:], self.state)
                seg = segs[idx]
            self.viewer.show_program(segs, idx, results, self.state.q, self.state.gripper)
            if seg.kind == "observe":
                p = seg.op.params
                v = self.obs_.view
                v.center, v.zoom = np.array([p["x"], p["y"], p["z"]]), max(1.0, float(p["zoom"]))
                if not np.isnan(p["grid_z"]):
                    v.grid_z = float(p["grid_z"])
                results[idx] = "view updated (see next image)"
                log.append({"segment": seg.label, "result": results[idx]})
                continue
            if seg.kind == "done":
                self.state.status = "done"
                results[idx] = "done"
                log.append({"segment": "done", "result": "task marked done"})
                break
            self.events("segment_pending", index=idx, label=seg.label, kind=seg.kind,
                        blockers=seg.blockers, warnings=seg.warnings,
                        q_start=None if seg.q_path is None else seg.q_path[0].tolist(),
                        q_end=None if seg.q_end is None else seg.q_end.tolist(),
                        ee_end=None if seg.ee_path is None else seg.ee_path[-1].tolist(),
                        duration_s=None if seg.durations is None else float(seg.durations.sum()))
            choice, feedback = self.viewer.wait_decision(seg)
            self.events("operator_decision", index=idx, label=seg.label, choice=choice, feedback=feedback)
            if choice != "approve":
                why = feedback or ("; ".join(seg.blockers) if seg.blockers else "no reason given")
                results[idx] = "REJECTED"
                log.append({"segment": seg.label, "result": f"rejected by operator: {why}"})
                outcome = "rejected_by_operator"
                self.state.status = "rejected_by_operator"
                break
            self.viewer.status_md.content = f"**Executing:** {seg.label}"
            try:
                self.events("exec_start", index=idx, label=seg.label)
                self.state, result = self.prims.execute(seg, self.state, self.close_duration_s)
                self.events("exec_end", index=idx, label=seg.label, result=result,
                            q_measured=None if self.state.q_measured is None else self.state.q_measured.tolist(),
                            gripper=self.state.gripper)
            except Exception as e:  # noqa: BLE001 -- report, keep serving
                result = f"execution error: {e}"
                self.events("exec_error", index=idx, label=seg.label, error=str(e))
                outcome = "execution_error"
                results[idx] = result
                log.append({"segment": seg.label, "result": result})
                break
            results[idx] = result
            log.append({"segment": seg.label, "result": result})
            print(f"  [{seg.label}] {result}")
            self.viewer.set_now(self.state.q_measured if self.state.q_measured is not None else self.state.q, self.state.gripper)
            self.viewer.status_md.content = f"**Done:** {seg.label} -- {result}"
            if self.state.status in ("grasp_failed", "dropped", "blocked"):
                failed_call = op.call_index
                outcome = self.state.status
        self.viewer.show_program(segs, len(segs), results, self.state.q, self.state.gripper)
        if self.viewer.auto.value:
            self.viewer.auto.value = False  # auto-approve is per plan
        report = {"status": outcome, "executed": log}
        if outcome == "rejected_by_operator" and self.viewer.stop_requested:
            report["status"] = "session_stopped"
        return report


def serve(session: RobotSession, port: int = 8081) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, indent=1).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") != "/obs":
                return self._send(404, {"error": "GET /obs or POST /plan"})
            with session.lock:
                self._send(200, session.observe())

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/plan":
                return self._send(404, {"error": "GET /obs or POST /plan"})
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
            try:
                code = json.loads(raw)["code"]
            except (ValueError, KeyError, TypeError):
                code = raw
            with session.lock:
                t0 = time.time()
                report = session.run_plan(code)
                report["obs"] = session.observe()
                report["seconds"] = round(time.time() - t0, 1)
            self._send(200, report)

        def log_message(self, *a):  # keep the console for the agent transcript
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd
