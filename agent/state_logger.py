"""Continuous joint + gripper telemetry for a run, so a handover (or any
grasp) can be read off the arms themselves instead of guessed from 5 fps
video: when each gripper closed or opened, how hard it was squeezing
(gripper_eff), whether both jaws held the object at once, when it slipped.

events.jsonl only has each arm's state at the END of a step. This polls
every arm at a fixed rate in one thread and writes

  runs/<stamp>/state.csv   one row per arm per sample:
      t_unix, arm, q0..q5, dq0..dq5, tau0..tau5,
      grip, grip_vel, grip_eff, cmd_q0..cmd_q5, cmd_grip,
      temp_mos0..6, temp_rotor0..6   (degC, motors 1-7; 7 = gripper)

Same clock as gemini/*.json, events.jsonl and video/*_frames.csv. Joints are
encoder space (rad), grip is i2rt's normalized 0 = closed .. 1 = open,
tau/grip_eff are the motors' reported effort; cmd_* is the setpoint the arm
was being driven to (measured q sags below it under gravity). The motor
temperatures are what the DM drivers report every control tick; a motor at
or above TEMP_WARN_C gets one warning (console + events.jsonl), so a stall is
visible before the driver trips -- the 2026-10-15 left-gripper overheat gave
no warning at all. A trip of the gripper stall guard
(yam_backend._gripper_guard) is logged as a `gripper_guard_trip` event.

on_sample(arm, state) is called with the same snapshot at a lower rate
(on_sample_hz) -- the launcher uses it to move the arms in viser while they
move for real, so the viser recording shows motion, not step-to-step jumps.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

_N = 6
_M = 7  # motors incl. the gripper
TEMP_WARN_C = 75.0  # information only: the DM drivers trip at 100 C (OT_Value, read from all 14 motors)
COLUMNS = (["t_unix", "arm"] + [f"q{i}" for i in range(_N)] + [f"dq{i}" for i in range(_N)]
           + [f"tau{i}" for i in range(_N)] + ["grip", "grip_vel", "grip_eff"]
           + [f"cmd_q{i}" for i in range(_N)] + ["cmd_grip"]
           + [f"temp_mos{i}" for i in range(_M)] + [f"temp_rotor{i}" for i in range(_M)])


def _row(t: float, arm: str, s: dict) -> str:
    def vals(key, n):
        v = list(s.get(key) or [])[:n]
        return [f"{x:.5f}" for x in v] + [""] * (n - len(v))

    cells = ([f"{t:.4f}", arm] + vals("joint_pos", _N) + vals("joint_vel", _N) + vals("joint_eff", _N)
             + vals("gripper_pos", 1) + vals("gripper_vel", 1) + vals("gripper_eff", 1)
             + vals("cmd_q", _N) + vals("cmd_gripper", 1)
             + vals("temp_mos", _M) + vals("temp_rotor", _M))
    return ",".join(cells) + "\n"


class StateLogger:
    def __init__(self, run_dir: Path, robots: dict, hz: float = 50.0,
                 on_sample: Callable[[str, dict], None] | None = None, on_sample_hz: float = 10.0,
                 events=None):
        """robots: arm name -> RealYamRobot / ProcessYamRobot (needs get_state)."""
        self.path = Path(run_dir) / "state.csv"
        self.robots, self.hz = robots, hz
        self.on_sample, self.on_sample_period = on_sample, 1.0 / on_sample_hz if on_sample_hz > 0 else None
        self.events = events or (lambda *a, **k: None)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="state-logger")
        self.n = 0

    def start(self) -> "StateLogger":
        self._thread.start()
        self.events("state_log_start", arms=list(self.robots), hz=self.hz, path=str(self.path))
        return self

    def stop(self) -> None:
        """Idempotent; safe from a finally block."""
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        self.events("state_log_end", rows=self.n)

    def _run(self) -> None:
        period = 1.0 / self.hz
        live = dict.fromkeys(self.robots, True)
        failed: set[str] = set()
        last_cb = dict.fromkeys(self.robots, 0.0)
        trips = dict.fromkeys(self.robots, 0.0)
        jtrips = dict.fromkeys(self.robots, 0.0)
        hot: set[tuple[str, int]] = set()
        next_t = time.time()
        with self.path.open("w") as f:
            f.write(",".join(COLUMNS) + "\n")
            while not self._stop.is_set() and any(live.values()):
                for arm, robot in self.robots.items():
                    if not live[arm]:
                        continue
                    try:
                        s = robot.get_state()
                        t = time.time()
                    except Exception as e:  # noqa: BLE001 -- a read hiccup shouldn't end the log
                        ok, why = robot.is_healthy()
                        if why == "not connected":  # released on purpose (shutting down)
                            live[arm] = False
                        elif arm not in failed:
                            print(f"[state] {arm}: read failed ({e!r}); retrying")
                            self.events("state_log_read_failed", arm=arm, error=repr(e))
                            failed.add(arm)
                        continue
                    failed.discard(arm)
                    f.write(_row(t, arm, s))
                    self.n += 1
                    self._watch(arm, s, trips, hot)
                    n = (s.get("joint_guard_trips") or [0.0])[0]
                    if n > jtrips[arm]:
                        jtrips[arm] = n
                        print(f"[state] {arm}: JOINT STALL GUARD tripped (#{int(n)}) -- arm was blocked")
                        self.events("joint_guard_trip", arm=arm, count=int(n))
                    if self.on_sample is not None and t - last_cb[arm] >= self.on_sample_period:
                        last_cb[arm] = t
                        try:
                            self.on_sample(arm, s)
                        except Exception as e:  # noqa: BLE001 -- the display must never stop the log
                            print(f"[state] on_sample({arm}) failed: {e!r}")
                if self.n % 200 < len(self.robots):
                    f.flush()
                next_t += period
                delay = next_t - time.time()
                if delay < -period:  # fell behind; don't burst to catch up
                    next_t = time.time()
                elif delay > 0:
                    self._stop.wait(delay)
        print(f"[state] {self.n} rows -> {self.path}")

    def _watch(self, arm: str, s: dict, trips: dict, hot: set) -> None:
        n = (s.get("gripper_guard_trips") or [0.0])[0]
        if n > trips[arm]:
            trips[arm] = n
            print(f"[state] {arm}: gripper stall guard tripped (#{int(n)})")
            self.events("gripper_guard_trip", arm=arm, count=int(n), grip=(s.get("gripper_pos") or [None])[0])
        for kind in ("temp_rotor", "temp_mos"):
            for i, c in enumerate(s.get(kind) or []):
                if c >= TEMP_WARN_C and (arm, i) not in hot:
                    hot.add((arm, i))
                    print(f"[state] WARNING {arm} motor {i + 1} {kind} {c:.0f} C")
                    self.events("motor_hot", arm=arm, motor=i + 1, sensor=kind, temp_c=c)
