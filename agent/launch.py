"""Launcher for both versions of the harness (run_human_check.py,
run_auto_check.py). Gemini drives BOTH arms: each turn it writes one scene
sentence, a thought and one `curl` that POSTs a program of skill calls to the
local robot server (agent/robot_server.py, agent/bimanual.py). Programs name
the arm (`left.grasp(...)`, `right.goto(...)`); `parallel(left.X(...),
right.Y(...))` moves both at once; every coordinate is in the right arm's base
frame. Each skill expands into segments that are IK-solved, then checked:

  checker="human"  every step is held in viser (point cloud + both arms, now
                   and planned) until an operator approves or rejects it; the
                   rejection text goes back to Gemini.
  checker="auto"   gap_guard decides (agent/gap_checker.py): OK runs the step,
                   IK_FAILURE / COLLISION rejects it and only that word goes
                   back to Gemini. viser still shows everything, and its Stop
                   button still stops the session.

Defaults to i2rt's MuJoCo sim for BOTH arms. --real needs the left arm's
gripper_limits in config/rig.yaml first: without them every connect runs
i2rt's jaw sweep on the left gripper (drives it shut, then fully open).
Measure them once with --real --measure-left-gripper.

Gemini sees the overhead image plus the right arm's wrist D405 (calibrated
overlay; --no-wrist to leave it out). Every run records runs/<stamp>/:
run_meta.json (args, prompt, model, config), events.jsonl, gemini/ (every
request and reply), obs_*.png, state.csv (both arms' joints, grippers and
motor temperatures at --state-hz), video/ (overhead, right wrist, any
--side-cam), scene.viser, and with checker="auto" gap_checks.jsonl.
On exit BOTH arms are released (go limp): home them first.
"""
import argparse
import contextlib
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import viser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.bimanual import ARMS, BimanualSession, BimanualViewer, bimanual_prompt
from agent.gemini_agent import GeminiAgent
from agent.observer import Observer, Workspace
from agent.primitives import Primitives
from agent.robot_server import serve
from agent.run_log import EventLog, SceneRecorder, write_run_meta
from camera.rig_calibration import (
    arm_channel,
    arm_gripper_limits,
    arm_gripper_type,
    arm_joint_offsets_rad,
    camera_serial,
    config_files,
    load_extrinsic,
    load_overhead_intrinsics,
)
from camera.zed_overhead import OverheadFrame
from vlm.gemini_client import DEFAULT_MODEL
from yam_control.process_robot import ProcessYamRobot
from yam_control.yam_backend import RealYamRobot

logging.basicConfig(level=logging.WARNING, format="%(message)s")
RUNS = Path(__file__).resolve().parents[1] / "runs"


def _arm_kwargs(side: str, sim: bool) -> dict:
    # ee_mass deliberately not passed: see config/rig.yaml.
    if sim:
        return {"channel": "can0", "gripper_type": "crank_4310", "sim": True}
    return {"channel": arm_channel(side), "gripper_type": arm_gripper_type(side), "sim": False,
            "gripper_limits": arm_gripper_limits(side), "joint_offsets_rad": arm_joint_offsets_rad(side)}


def main(checker: str) -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # OPENROUTER_API_KEY, ...
    ap = argparse.ArgumentParser(description={
        "human": "Open Gemini robot harness -- every step approved by a human in viser.",
        "auto": "Open Gemini robot harness -- every step checked by gap_guard (OK / IK_FAILURE / COLLISION).",
    }[checker])
    ap.add_argument("--task", default="Pick up the clamp with the arm closest to it, hand it over to the other arm, "
                                      "then place the clamp inside the blue bin.")
    ap.add_argument("--real", action="store_true", help="drive BOTH real arms instead of i2rt's sim")
    ap.add_argument("--measure-left-gripper", action="store_true",
                    help="connect the left arm once so i2rt measures its jaw travel (the jaws WILL sweep), "
                         "print the pair to put in config/rig.yaml, and exit")
    ap.add_argument("--replay", help="captures/*.npz overhead frame instead of the live ZED")
    ap.add_argument("--zed-method", default="tri_stereo", choices=["tri_stereo", "zed"],
                    help="overhead depth: TRI Stereo learned depth (third_party/raiden) or the ZED SDK's")
    ap.add_argument("--zed-in-sim", action="store_true",
                    help="sim arms with the live overhead ZED (default in sim: a synthetic frame)")
    ap.add_argument("--serve-only", action="store_true")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=12000,
                    help="hard cap on each Gemini reply; a reply cut off there is not executed")
    ap.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high"],
                    help="OpenRouter reasoning effort for Gemini's hidden thinking (default: provider's)")
    ap.add_argument("--reasoning-tokens", type=int, help="token budget for hidden thinking (overrides effort)")
    ap.add_argument("--time-budget-s", type=float,
                    help="tell Gemini it must answer within N seconds (and roughly how many tokens that is)")
    ap.add_argument("--thought-words", type=int,
                    help="tell Gemini its # Thought: may be at most N words and that replies over "
                         "--max-tokens are not run (default: not told)")
    ap.add_argument("--max-joint-speed", type=float, default=0.6)
    ap.add_argument("--port", type=int, default=8090, help="viser")
    ap.add_argument("--robot-port", type=int, default=8081)
    ap.add_argument("--record-fps", type=float, default=10.0,
                    help="continuous overhead/wrist/side video in runs/<stamp>/video/ (0 = off)")
    ap.add_argument("--no-wrist", action="store_true",
                    help="don't open the right arm's D405 wrist camera (not shown to Gemini, not recorded)")
    ap.add_argument("--side-cam", action="append", default=[], metavar="NAME=DEVICE",
                    help="also record a USB camera, e.g. side=/dev/video6 or side=6 (repeatable)")
    ap.add_argument("--state-hz", type=float, default=50.0,
                    help="joint + gripper telemetry for both arms -> runs/<stamp>/state.csv (0 = off)")
    ap.add_argument("--record-viser", action="store_true",
                    help="also record the whole viser page (3D + approval panels) via headless Chromium; "
                         "needs playwright")
    if checker == "auto":
        ap.add_argument("--yes", action="store_true",
                        help="with --real: skip the confirmation that no human approves motions")
    args = ap.parse_args()
    sim = not args.real

    if args.measure_left_gripper:
        kw = _arm_kwargs("left", sim=False)
        kw["gripper_limits"] = None
        print("[measure] connecting the LEFT arm -- the gripper will close fully, then open fully. "
              "Keep the jaws empty.")
        with RealYamRobot(**kw) as r:
            limits = getattr(r._robot, "_gripper_limits", None)
        print(f"\nLeft gripper limits [closed, open]: {None if limits is None else [float(x) for x in limits]}")
        print("Paste into config/rig.yaml under arms.left as:  gripper_limits: [<closed>, <open>]")
        return

    if args.real and arm_gripper_limits("left") is None:
        sys.exit("Left arm has no gripper_limits in config/rig.yaml -- connecting would sweep its jaws. "
                 "Run with --real --measure-left-gripper first.")

    if args.real and checker == "auto" and not args.yes:
        print("[REAL + AUTO] gap_guard approves every step: NO human approves any motion. It does not model\n"
              "  the held object, the left wrist camera, or anything the overhead camera cannot see.")
        if input("Type 'go' with the e-stop in your hand: ").strip() != "go":
            sys.exit("Not started.")
    run_dir = RUNS / (f"bimanual_{checker}_" + time.strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True)
    print("[sim] both arms in i2rt's MuJoCo sim -- no CAN traffic." if sim else
          "[REAL] both arms. E-stops in reach? Workspace clear between the arms?")

    cams = []
    recorder = None
    cam_rec = None
    viser_screen = None
    state_log = None
    # Full, recoverable record (agent/run_log.py): run_meta.json
    # (+ code.patch, calib/), gemini/, events.jsonl, scene.viser.
    events = EventLog(run_dir)
    prompt = bimanual_prompt(checker)
    write_run_meta(run_dir, args, prompt, os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
                   config_files(["T_rightbase_overhead", "T_leftbase_rightbase", "T_right_wrist_gripper_cam"]))
    events("session_start", real=args.real, task=args.task, serve_only=args.serve_only, checker=checker)

    T_world_left = load_extrinsic("T_leftbase_rightbase").inverse().matrix  # world = right base
    try:
        # Real arms: i2rt in their own processes, so nothing here (camera/ML
        # start-up, viser, Gemini) can starve their control loops -- see process_robot.py.
        robot_cls = RealYamRobot if sim else ProcessYamRobot
        # The ExitStack is exited first, i.e. before either arm is released --
        # the state logger must stop reading the arms before they disconnect.
        with robot_cls(**_arm_kwargs("right", sim)) as right, robot_cls(**_arm_kwargs("left", sim)) as left, \
                contextlib.ExitStack() as before_release:
            robots = {"left": left, "right": right}
            time.sleep(1.0)
            for name, r in robots.items():
                ok, why = r.is_healthy()
                if not ok:
                    raise RuntimeError(f"{name} arm not under control after connect: {why}")
            # Cameras only after both arms are up: with the ZED + TRI Stereo model +
            # D405 already running, i2rt's connect handshake intermittently missed a
            # motor's reply and the motors tripped their CAN watchdog (5 of 6
            # connects failed on the rig, 3 of 3 clean without cameras).
            if args.replay:
                frame = OverheadFrame.load(Path(args.replay))
                grab_overhead = lambda: frame  # noqa: E731
            elif sim and not args.zed_in_sim:
                from camera.synthetic_overhead import synthetic_overhead

                frame = synthetic_overhead(load_extrinsic("T_rightbase_overhead"), load_overhead_intrinsics())
                grab_overhead = lambda: frame  # noqa: E731
                print("[obs] sim: synthetic overhead frame (table + red block); --zed-in-sim for the real camera")
            else:
                from camera.zed_overhead import OverheadCamera

                cam = OverheadCamera(args.zed_method)
                grab_overhead = cam.grab
                cams.append(cam)

            observer = Observer(load_extrinsic("T_rightbase_overhead"),
                                Workspace(x=(0.12, 0.60), y=(-0.35, 0.75), z=(-0.20, 0.45)))
            arms = {}
            for name in ARMS:
                arms[name] = Primitives(robots[name], np.array(robots[name].get_joint_positions()),
                                        Workspace(x=(0.12, 0.60), y=(-0.35, 0.35), z=(-0.20, 0.45)),
                                        table_z_fn=lambda: observer.table_z, max_joint_speed=args.max_joint_speed,
                                        T_world_base=None if name == "right" else T_world_left)
            server = viser.ViserServer(port=args.port)
            if server.get_port() == args.robot_port:
                raise RuntimeError(f"viser fell through to the robot server's port {args.robot_port}; pass --port")
            if checker == "auto":
                from agent.gap_checker import AutoCheckViewer, GapChecker

                gap = GapChecker(arms, T_world_left, run_dir)
                viewer = AutoCheckViewer(server, arms, gap, real=args.real)
            else:
                viewer = BimanualViewer(server, arms, real=args.real)
            recorder = SceneRecorder(server, run_dir)
            # Right-arm wrist D405: shown to Gemini every turn (calibrated overlay
            # when T_right_wrist_gripper_cam exists) and recorded.
            wrist_cam = wrist_view = None
            if not args.no_wrist:
                try:
                    from camera.d405_wrist import WristCamera

                    wrist_cam = WristCamera(serial=camera_serial("right_wrist"), depth=True)
                    cams.append(wrist_cam)
                except Exception as e:  # noqa: BLE001 -- run on without it rather than not at all
                    print(f"[wrist] NOT using the right wrist camera: {e}")
                    events("wrist_camera_failed", error=repr(e))
            if wrist_cam is not None:
                try:
                    from agent.wrist_view import WristView

                    wrist_view = WristView(arms["right"].robot, load_extrinsic("T_right_wrist_gripper_cam"),
                                           wrist_cam.intrinsics)
                    print("[wrist] calibrated: T_right_wrist_gripper_cam -> overlay + point cloud for Gemini")
                except Exception as e:  # noqa: BLE001
                    print(f"[wrist] no usable T_right_wrist_gripper_cam ({e}) -- raw wrist image to Gemini")
            session = BimanualSession(arms, observer, viewer, grab_overhead, run_dir, events=events,
                                      wrist_cam=wrist_cam, wrist_view=wrist_view)
            if checker == "auto":
                gap.session = session

            def watchdog() -> None:
                while True:
                    time.sleep(0.5)
                    for name, r in robots.items():
                        ok, why = r.is_healthy()
                        if not ok and why == "not connected":
                            return  # we disconnected on purpose (shutting down)
                        if not ok:
                            print(f"\n[WATCHDOG] {name} arm: {why} -- stopping everything.", flush=True)
                            events("watchdog_fault", arm=name, reason=why)
                            viewer.stop_requested = True
                            viewer.status_md.content = f"**ARM FAULT ({name}):** {why}. Session stopped."
                            os.kill(os.getpid(), signal.SIGINT)
                            return

            threading.Thread(target=watchdog, daemon=True).start()
            # Joint + gripper telemetry (agent/state_logger.py), which also moves
            # the arms in viser live, so the 3D view/recording shows the motion.
            if args.state_hz > 0:
                from agent.state_logger import StateLogger

                state_log = StateLogger(run_dir, robots, hz=args.state_hz, on_sample=viewer.follow_arm,
                                        events=events).start()
                before_release.callback(state_log.stop)
                print(f"[state] logging both arms at {args.state_hz:g} Hz -> {state_log.path}")
            # Continuous camera video (agent/cam_recorder.py), same clock as the logs.
            rec_sources = {}
            if any(c.__class__.__name__ == "OverheadCamera" for c in cams):
                rec_sources["overhead"] = cams[0].grab_rgb
            if wrist_cam is not None:
                rec_sources["right_wrist"] = wrist_cam.grab_rgb
            for spec in args.side_cam if args.record_fps > 0 else []:
                name, _, dev = spec.partition("=")
                name, dev = (name, dev) if dev else ("side", name)
                try:
                    from camera.uvc_camera import UvcCamera

                    uvc = UvcCamera(dev)
                    cams.append(uvc)
                    rec_sources[name] = uvc.grab_rgb
                except Exception as e:  # noqa: BLE001
                    print(f"[rec] NOT recording {name} ({dev}): {e}")
                    events("recording_grab_failed", camera=name, error=repr(e))
            if args.record_viser and args.record_fps > 0:
                from agent.viser_screen import ViserScreen

                try:
                    viser_screen = ViserScreen(f"http://localhost:{server.get_port()}")
                    rec_sources["viser"] = viser_screen.grab
                except Exception as e:  # noqa: BLE001 -- don't lose the run over a recording
                    print(f"[rec] NOT recording the viser page: {e}")
                    events("recording_grab_failed", camera="viser", error=repr(e))
            if args.record_fps > 0 and rec_sources:
                from agent.cam_recorder import CameraRecorder

                cam_rec = CameraRecorder(run_dir, rec_sources, fps=args.record_fps, events=events).start()
                print(f"[rec] recording {', '.join(rec_sources)} at {args.record_fps:g} fps -> {cam_rec.dir}")
            session.observe()
            for p in arms.values():
                p.ws.z = (observer.table_z - 0.005, 0.45)
            print(f"[obs] table_z = {observer.table_z:+.3f} m (right-arm base frame)")
            serve(session, args.robot_port)
            print(f"\n[viser] http://localhost:{server.get_port()}  -- " +
                  ("approve/reject every step there" if checker == "human" else "gap_guard decides; Stop is there"))
            print(f"[robot] http://127.0.0.1:{args.robot_port}/obs, /plan   logs: {run_dir}")
            if args.serve_only:
                print("Serve-only. Ctrl+C to quit (releases both arms).")
                while True:
                    time.sleep(1)
            try:
                GeminiAgent(session, viewer, run_dir, port=args.robot_port, max_turns=args.max_turns,
                            max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort,
                            reasoning_tokens=args.reasoning_tokens, thought_words=args.thought_words,
                            time_budget_s=args.time_budget_s,
                            system_prompt=prompt).run(args.task)
            except Exception as e:  # noqa: BLE001
                # An agent-side failure must not release the arms (2026-10-15: an
                # unhandled API error exited the program and the right arm dropped
                # the clamp it was holding). Keep holding; the operator decides.
                import traceback

                traceback.print_exc()
                events("agent_crashed", error=repr(e))
                print(f"\n[agent] CRASHED ({e!r}) -- the arms are STILL POWERED and holding. The robot server "
                      f"and viser are still up (you can POST /plan, e.g. home). Ctrl+C releases both arms.")
            print("Agent finished. Ctrl+C to quit (releases both arms).")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nQuit -- releasing both arms.")
    finally:
        events("session_end")
        if state_log is not None:
            state_log.stop()  # normally already stopped, before the arms were released
        if cam_rec is not None:
            cam_rec.stop()  # before the cameras close
        if viser_screen is not None:
            viser_screen.close()
        if recorder is not None:
            print(f"[log] 3D scene -> {recorder.save()}")
        for c in cams:
            c.close()


if __name__ == "__main__":
    main()
