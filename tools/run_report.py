"""One-page HTML preview of a harness run (run_human_check.py / run_auto_check.py): camera
videos with clickable event markers, joint/gripper/temperature plots from
state.csv, and every Gemini turn with the images it saw, its reply and what
each step did. Plus a link to the 3D replay of scene.viser.

    python tools/run_report.py runs/bimanual_20261015_172035     # -> runs/bimanual_20261015_172035_report.html
    python -m http.server 8765 --directory runs    # then open
    http://localhost:8765/bimanual_20261015_172035_report.html

The run folder is only READ -- the page is written next to it (or to --out)
and links to the recording's files where they are; nothing in the recording
is modified or copied. Skipping operator decision time is done by the
viewer at playback time, not by editing the videos or scene. The page loads
Chart.js from cdnjs. For the 3D replay, build viser's web client once
(`viser-build-client --out-dir <dir>`) and pass --viser-client <url> -- the
page then links <url>?playbackPath=<scene.viser>.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path

ARMS = ("left", "right")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _state_series(run: Path, every: int = 10) -> dict:
    """state.csv -> per-arm downsampled series (t relative to the first row)."""
    out = {a: {"t": [], "grip": [], "eff": [], "tmax": []} for a in ARMS}
    path = run / "state.csv"
    if not path.exists():
        return {"t0": None, **out}
    t0, counts = None, {a: 0 for a in ARMS}
    with path.open() as f:
        for row in csv.DictReader(f):
            arm = row["arm"]
            if arm not in out:
                continue
            t = float(row["t_unix"])
            t0 = t if t0 is None else t0
            counts[arm] += 1
            if counts[arm] % every:
                continue
            s = out[arm]
            s["t"].append(round(t - t0, 2))
            s["grip"].append(float(row["grip"] or "nan"))
            s["eff"].append(abs(float(row["grip_eff"] or "nan")))
            temps = [float(row[k]) for k in row if k.startswith("temp_rotor") and row[k]]
            s["tmax"].append(max(temps) if temps else None)
    return {"t0": t0, **out}


def _video_time(run: Path, name: str, t_unix: float, fps: float) -> float | None:
    """Seconds into video/<name>.mp4 for a wall-clock time (frames.csv maps frame -> t)."""
    path = run / "video" / f"{name}_frames.csv"
    if not path.exists():
        return None
    best = None
    with path.open() as f:
        for row in csv.DictReader(f):
            if float(row["t_unix"]) >= t_unix:
                best = int(row["frame"])
                break
    return None if best is None else round(best / fps, 2)


def _frame_table(run: Path, name: str, fps: float, every: int = 10) -> list[list[float]]:
    """[[video_seconds, t_unix], ...] every `every` frames: the video plays at the
    nominal fps, the real capture times are in <name>_frames.csv."""
    path = run / "video" / f"{name}_frames.csv"
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows):
        if i % every == 0 or i == len(rows) - 1:
            out.append([round(int(row["frame"]) / fps, 2), float(row["t_unix"])])
    return out


def _scene_t0(run: Path, events: list[dict]) -> float | None:
    """Wall-clock time of scene.viser's t=0: from the sidecar SceneRecorder writes,
    else estimated as (end of recording) - (recording duration)."""
    side = run / "scene.viser.json"
    if side.exists():
        return json.loads(side.read_text())["t0_unix"]
    scene = run / "scene.viser"
    if not scene.exists():
        return None
    try:
        import msgspec
        import zstandard

        b = scene.read_bytes()
        inner = zstandard.ZstdDecompressor().decompress(b[8:], max_output_size=int.from_bytes(b[:8], "little"))
        n = int.from_bytes(inner[:8], "little")
        duration = msgspec.msgpack.decode(inner[8:8 + n])["durationSeconds"]
    except Exception:  # noqa: BLE001
        return None
    end = next((e["t_unix"] for e in reversed(events) if e["event"] in ("recording_end", "session_end")), None)
    return None if end is None else end - duration


def _obs_images(result) -> list[str]:
    """Image basenames referenced by an /obs or /plan result."""
    if not isinstance(result, dict):
        return []
    obs = result.get("obs", result)
    names = []
    for key in ("overhead_cam_path", "wrist_cam_path"):
        v = obs.get(key) if isinstance(obs, dict) else None
        if v:
            names.append(Path(v.split("  (")[0]).name)
    return names


def build(run: Path, viser_client: str | None, out: Path | None = None) -> Path:
    run = Path(os.path.abspath(run))   # keep symlinks as given: links are relative to where the page is served
    out = Path(os.path.abspath(out or run.parent / f"{run.name}_report.html"))
    if run.resolve() in out.resolve().parents:
        raise SystemExit("refusing to write inside the run folder -- the recording stays untouched; pick another --out")
    base = os.path.relpath(run, out.parent).replace(os.sep, "/") + "/"   # links from the page to the recording
    meta = json.loads((run / "run_meta.json").read_text()) if (run / "run_meta.json").exists() else {}
    events = _load_jsonl(run / "events.jsonl")
    turns = _load_jsonl(run / "transcript.jsonl")
    fps = float((meta.get("args") or {}).get("record_fps") or 10.0)
    series = _state_series(run)
    t0 = series["t0"] or (events[0]["t_unix"] if events else 0.0)

    # Gemini latency/tokens per call
    calls = {}
    for f in sorted((run / "gemini").glob("call_*_response.json")):
        d = json.loads(f.read_text())
        u = (d.get("raw") or {}).get("usage") or {}
        calls[d.get("call")] = (d.get("latency_s"), u.get("completion_tokens"), d.get("error"))

    # Key moments for the plots and the video markers
    marks = []
    for e in events:
        ev = e["event"]
        if ev in ("exec_end", "exec_error"):
            label = e.get("label", "")
            res = e.get("result", "")
            kind = ("drop" if "DROPPED" in res else "block" if "BLOCKED" in res
                    else "grip" if ("gripper" in label or "close" in label) else "move")
            marks.append({"t": round(e["t_unix"] - t0, 2), "t_unix": e["t_unix"], "kind": kind,
                          "label": label, "result": res})
        elif ev in ("watchdog_fault", "joint_guard_trip", "gripper_guard_trip", "agent_crashed",
                    "overhead_unavailable"):
            marks.append({"t": round(e["t_unix"] - t0, 2), "t_unix": e["t_unix"], "kind": "fault",
                          "label": ev, "result": json.dumps({k: v for k, v in e.items() if k not in ("t", "t_unix", "event")})})
    for m in marks:
        m["video_t"] = _video_time(run, "overhead", m["t_unix"], fps)
    frames = _frame_table(run, "overhead", fps)
    # Operator decision time: each step waits in viser from step_pending until the
    # operator clicks (operator_decision). The viewer can skip these intervals.
    gaps, pending = [], None
    for e in events:
        if e["event"] == "step_pending":
            pending = e["t_unix"]
        elif e["event"] == "operator_decision" and pending is not None:
            if e["t_unix"] > pending:
                gaps.append([pending, e["t_unix"]])
            pending = None
    decision_s = sum(b - a for a, b in gaps)
    scene_t0 = _scene_t0(run, events) if viser_client else None
    replay_3d = bool(viser_client and (run / "scene.viser").exists() and scene_t0 is not None)
    wall0 = frames[0][1] if frames else t0

    # Per-turn cards: reply, images, and the steps that ran before the next turn
    cards = []
    for i, tr in enumerate(turns):
        t_start = tr["t"]
        t_end = turns[i + 1]["t"] if i + 1 < len(turns) else float("inf")
        steps = []
        for e in events:
            if not (t_start <= e["t_unix"] < t_end):
                continue
            if e["event"] == "operator_decision":
                fb = f" -- \"{e['feedback']}\"" if e.get("feedback") else ""
                steps.append(("decision", f"{e.get('label', '')}: {e.get('choice')}{fb}"))
            elif e["event"] in ("exec_end", "exec_error"):
                steps.append(("result", f"{e.get('label', '')}: {e.get('result', '')}"))
            elif e["event"] == "plan_end":
                steps.append(("status", f"plan status: {e.get('status')}"))
        lat, toks, err = calls.get(tr["turn"], (None, None, None))
        cards.append({"turn": tr["turn"], "t": round(t_start - t0, 1), "reply": tr.get("reply", ""),
                      "error": tr.get("error") or err, "images": _obs_images(tr.get("result")),
                      "steps": steps, "latency": lat, "tokens": toks,
                      "video_t": _video_time(run, "overhead", t_start, fps)})

    args = meta.get("args") or {}
    summary = [
        ("Task", args.get("task", "")),
        ("Model", f"{meta.get('model', '')}  (reasoning: {args.get('reasoning_effort') or 'default'}, "
                  f"time budget: {args.get('time_budget_s') or '-'} s)"),
        ("Started", meta.get("t", "")),
        ("Turns", str(len(turns))),
        ("Commit", f"{(meta.get('git_commit') or '')[:10]}{' (dirty)' if meta.get('git_dirty') else ''}"),
    ]
    n_drop = sum(m["kind"] == "drop" for m in marks)
    n_block = sum(m["kind"] == "block" for m in marks)
    n_fault = sum(m["kind"] == "fault" for m in marks)
    summary.append(("Drops / blocked / faults", f"{n_drop} / {n_block} / {n_fault}"))
    summary.append(("Operator decision time", f"{decision_s:.0f} s over {len(gaps)} decisions "
                                              f"(skipped in the replay by default)"))

    viser_link = ""
    if viser_client and (run / "scene.viser").exists():
        viser_link = (f'<a class="btn" target="_blank" href="{html.escape(viser_client)}?playbackPath='
                      f'{html.escape("/" + run.name + "/scene.viser")}">Open 3D replay (scene.viser)</a>')

    def esc(s):
        return html.escape(str(s))

    card_html = []
    for c in cards:
        imgs = "".join(f'<a href="{esc(base + n)}" target="_blank"><img loading="lazy" src="{esc(base + n)}"></a>' for n in c["images"])
        steps = "".join(f'<li class="{k}">{esc(v)}</li>' for k, v in c["steps"])
        meta_line = f"t+{c['t']} s"
        if c["latency"] is not None:
            meta_line += f" · Gemini {c['latency']:.1f} s"
        if c["tokens"]:
            meta_line += f", {c['tokens']} tokens"
        seek = (f' · <a href="#" onclick="seek({c["video_t"]});return false;">jump video</a>'
                if c["video_t"] is not None else "")
        err = f'<div class="err">{esc(c["error"])}</div>' if c["error"] else ""
        card_html.append(f"""
<section class="card" id="turn{c['turn']}">
  <h3>Turn {c['turn']} <span class="meta">{esc(meta_line)}{seek}</span></h3>
  <div class="cols"><pre class="reply">{esc(c['reply'])}</pre><div class="imgs">{imgs}</div></div>
  {err}<ul class="steps">{steps}</ul>
</section>""")

    mark_html = "".join(
        f'<li class="{m["kind"]}"><a href="#" onclick="seek({m["video_t"]});return false;">t+{m["t"]:.1f} s</a> '
        f'{esc(m["label"])} <span class="res">{esc(m["result"])[:160]}</span></li>'
        for m in marks if m["video_t"] is not None)

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run Report</title>
<style>
:root {{ --bg:#fafaf9; --fg:#1c1917; --muted:#78716c; --card:#fff; --line:#e7e5e4; --accent:#2563eb;
        --drop:#dc2626; --block:#d97706; --grip:#059669; --fault:#7c3aed; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0c0a09; --fg:#e7e5e4; --muted:#a8a29e;
        --card:#1c1917; --line:#292524; --accent:#60a5fa; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; margin:0; padding:16px; }}
main {{ max-width:1200px; margin:0 auto; }}
h1 {{ font-size:22px; margin:4px 0 12px; }} h2 {{ font-size:17px; margin:24px 0 8px; }}
table.sum td {{ padding:2px 12px 2px 0; vertical-align:top; }} table.sum td:first-child {{ color:var(--muted); white-space:nowrap; }}
.btn {{ display:inline-block; margin:10px 0; padding:6px 12px; border:1px solid var(--accent); color:var(--accent);
        border-radius:6px; text-decoration:none; }}
.replay {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
.replay.with3d {{ grid-template-columns:1.35fr 1fr; grid-template-rows:auto auto; }}
.replay.with3d .p3d {{ grid-row:1 / span 2; display:flex; flex-direction:column; }}
.replay.with3d .p3d iframe {{ flex:1; min-height:420px; }}
@media (max-width:860px) {{ .replay, .replay.with3d {{ grid-template-columns:1fr; }} .replay.with3d .p3d {{ grid-row:auto; }} }}
iframe {{ width:100%; border:1px solid var(--line); border-radius:6px; background:#fff; }}
video {{ width:100%; background:#000; border-radius:6px; display:block; }}
.bar {{ display:flex; align-items:center; gap:10px; margin-top:8px; background:var(--card); border:1px solid var(--line);
        border-radius:8px; padding:6px 10px; }}
.bar button {{ width:34px; height:30px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--fg); cursor:pointer; }}
.bar select {{ background:var(--bg); color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:3px; }}
.mono {{ font-family:ui-monospace, monospace; font-size:12.5px; white-space:nowrap; }}
.track {{ position:relative; flex:1; height:30px; }}
.track input {{ position:absolute; inset:8px 0 auto 0; width:100%; margin:0; accent-color:var(--accent); }}
#b_ticks {{ position:absolute; left:0; right:0; top:0; height:8px; }}
#b_ticks span {{ position:absolute; top:0; width:3px; height:8px; margin-left:-1px; border-radius:1px; cursor:pointer; }}
.charts {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(340px, 1fr)); gap:14px; }}
.chart {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px; height:240px; }}
ul.marks {{ columns:2 340px; padding-left:18px; font-size:13px; }} ul.marks .res {{ color:var(--muted); }}
li.drop {{ color:var(--drop); }} li.block {{ color:var(--block); }} li.fault {{ color:var(--fault); }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin:10px 0; }}
.card h3 {{ margin:0 0 6px; font-size:15px; }} .meta {{ color:var(--muted); font-weight:normal; font-size:12px; }}
.cols {{ display:grid; grid-template-columns:minmax(0, 1fr) minmax(0, 1fr); gap:10px; }}
@media (max-width:760px) {{ .cols {{ grid-template-columns:1fr; }} }}
pre.reply {{ white-space:pre-wrap; word-break:break-word; margin:0; font-size:12.5px; max-height:340px; overflow:auto; }}
.imgs img {{ width:49%; margin:0 1% 4px 0; border-radius:4px; }}
ul.steps {{ margin:6px 0 0; padding-left:18px; font-size:12.5px; }} ul.steps li.status {{ color:var(--muted); }}
.err {{ color:var(--drop); font-size:12.5px; margin-top:4px; }}
</style></head>
<body><main>
<h1>Run report: {esc(run.name)}</h1>
<table class="sum">{''.join(f'<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>' for k, v in summary)}</table>
{viser_link}

<h2>Replay</h2>
<div class="replay{' with3d' if replay_3d else ''}">
  {f'<div class="pane p3d"><iframe id="v3d" src="{html.escape(viser_client)}?playbackPath={html.escape("/" + run.name + "/scene.viser")}"></iframe><div class="meta">3D scene (scene.viser) &middot; drag to orbit</div></div>' if replay_3d else ''}
  <div class="pane"><video id="v_over" preload="auto" muted playsinline src="{esc(base)}video/overhead.mp4"></video><div class="meta">overhead (ZED-M)</div></div>
  <div class="pane"><video id="v_wrist" preload="auto" muted playsinline src="{esc(base)}video/right_wrist.mp4"></video><div class="meta">right wrist (D405)</div></div>
</div>
<div class="bar">
  <button id="b_play" title="Play / pause (space)">&#9654;</button>
  <span id="b_time" class="mono">0:00 / 0:00</span>
  <div class="track"><input id="b_seek" type="range" min="0" max="1" step="0.05" value="0"><div id="b_ticks"></div></div>
  <label class="mono" title="Cut out the time steps waited for the operator's approval"><input id="b_skip" type="checkbox" checked> skip my decision time</label>
  <select id="b_speed" title="Speed"><option>0.5x</option><option selected>1x</option><option>2x</option><option>4x</option><option>8x</option></select>
</div>
<div class="meta">One timeline for all views; with "skip my decision time" on it shows only agent + motion time, and playback jumps over every wait for your approval{' (the 3D scene follows it; loading the 34 MB recording takes a moment)' if replay_3d else ''}. Coloured ticks = key moments (green grasp/release, amber blocked, red dropped, violet fault); click one to jump.</div>

<h2>Key moments</h2>
<ul class="marks">{mark_html}</ul>

<h2>Grippers and motors</h2>
<div class="charts">
  <div class="chart"><canvas id="c_grip"></canvas></div>
  <div class="chart"><canvas id="c_eff"></canvas></div>
  <div class="chart"><canvas id="c_temp"></canvas></div>
</div>

<h2>Turns</h2>
{''.join(card_html)}
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
const S = {json.dumps({a: series[a] for a in ARMS})};
const MARKS = {json.dumps([{k: m[k] for k in ("t", "kind", "label")} for m in marks])};
// ---- shared timeline: overhead video is the clock; wrist video and 3D scene follow it ----
const FR = {json.dumps(frames)};            // [video_s, t_unix]
const WALL0 = {json.dumps(wall0)}, SCENE_T0 = {json.dumps(scene_t0)};
const vo = document.getElementById('v_over'), vw = document.getElementById('v_wrist');
const f3d = document.getElementById('v3d');
const bPlay = document.getElementById('b_play'), bSeek = document.getElementById('b_seek'),
      bTime = document.getElementById('b_time'), bSpeed = document.getElementById('b_speed'), bTicks = document.getElementById('b_ticks');
function interp(x, xs, ys) {{
  if (!xs.length) return x; if (x <= xs[0]) return ys[0] + (x - xs[0]); if (x >= xs[xs.length-1]) return ys[ys.length-1] + (x - xs[xs.length-1]);
  let lo = 0, hi = xs.length - 1; while (hi - lo > 1) {{ const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }}
  return ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / (xs[hi] - xs[lo]);
}}
const VT = FR.map(r => r[0]), TU = FR.map(r => r[1]);
const videoToWall = v => interp(v, VT, TU), wallToVideo = w => interp(w, TU, VT);
const WALL_SPAN = TU.length ? TU[TU.length-1] - TU[0] : 0;
// Operator decision intervals, as [start, end] seconds relative to the video start.
const GAPS = {json.dumps([[round(a - wall0, 2), round(b - wall0, 2)] for a, b in gaps])}.filter(g => g[1] > 0 && g[0] < WALL_SPAN);
const bSkip = document.getElementById('b_skip');
// wall time (s since video start) <-> shown time: with skipping, every gap before
// the moment is subtracted, and a moment inside a gap maps to the gap's start.
function wallToShown(w) {{
  if (!bSkip.checked) return w; let cut = 0;
  for (const [a, b] of GAPS) {{ if (w >= b) cut += b - a; else if (w > a) {{ cut += w - a; break; }} else break; }}
  return w - cut;
}}
function shownToWall(c) {{
  if (!bSkip.checked) return c; let w = c;
  for (const [a, b] of GAPS) {{ if (w >= a) w += b - a; else break; }}
  return w;
}}
const gapAt = w => GAPS.find(([a, b]) => w >= a && w < b);
let SPAN = WALL_SPAN;
function relayout() {{
  SPAN = wallToShown(WALL_SPAN); bSeek.max = SPAN.toFixed(2);
  for (const s of bTicks.children) s.style.left = (100 * wallToShown(+s.dataset.rel) / SPAN) + '%';
  refresh();
}}
const fmt = s => {{ s = Math.max(0, s); return Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0'); }};

// 3D: viser's playback bar is driven through its time box (same origin). It is kept
// paused and stepped to the shared time, so it follows any speed and every seek.
function viserTimeInput() {{
  try {{ const d = f3d && f3d.contentDocument; if (!d) return null;
        return [...d.querySelectorAll('input')].find(i => !i.readOnly && /^-?\\d+(\\.\\d+)?$/.test(i.value)) || null; }} catch (e) {{ return null; }}
}}
function viserPause() {{
  try {{ const d = f3d.contentDocument; const b = d.querySelector('.tabler-icon-player-pause-filled'); if (b) b.closest('button').click(); }} catch (e) {{}}
}}
let last3d = null;
function viserSeek(sceneT) {{
  const inp = viserTimeInput(); if (!inp) return;
  if (last3d !== null && Math.abs(sceneT - last3d) < 0.08) return;
  const w = f3d.contentWindow, set = Object.getOwnPropertyDescriptor(w.HTMLInputElement.prototype, 'value').set;
  set.call(inp, Math.max(0, sceneT).toFixed(2)); inp.dispatchEvent(new w.Event('input', {{ bubbles: true }}));
  last3d = sceneT; viserPause();
}}
function sync3d() {{ if (f3d && SCENE_T0 !== null) viserSeek(videoToWall(vo.currentTime) - SCENE_T0); }}

function seekWall(rel) {{
  if (bSkip.checked) {{ const g = gapAt(rel); if (g) rel = g[1]; }}
  const v = wallToVideo(TU[0] + rel); vo.currentTime = v; vw.currentTime = v; last3d = null; sync3d(); refresh();
}}
function seek(videoT) {{ if (videoT == null) return; seekWall(videoToWall(videoT) - TU[0]); document.querySelector('.replay').scrollIntoView({{behavior:'smooth', block:'start'}}); }}
function refresh() {{ const c = wallToShown(videoToWall(vo.currentTime) - TU[0]); if (!dragging) bSeek.value = c; bTime.textContent = fmt(c) + ' / ' + fmt(SPAN); }}
function setPlaying(on) {{
  if (on) {{ vw.currentTime = vo.currentTime; vo.play(); vw.play(); bPlay.innerHTML = '&#10074;&#10074;'; }}
  else {{ vo.pause(); vw.pause(); bPlay.innerHTML = '&#9654;'; }}
}}
bPlay.onclick = () => setPlaying(vo.paused);
document.addEventListener('keydown', e => {{ if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'SELECT') {{ e.preventDefault(); setPlaying(vo.paused); }} }});
let dragging = false;
bSeek.addEventListener('input', () => {{ dragging = true; seekWall(shownToWall(parseFloat(bSeek.value))); }});
bSkip.onchange = relayout;
bSeek.addEventListener('change', () => {{ dragging = false; }});
bSpeed.onchange = () => {{ const r = parseFloat(bSpeed.value); vo.playbackRate = r; vw.playbackRate = r; }};
vo.addEventListener('ended', () => setPlaying(false));
function skipGap() {{
  // while playing, jump over an operator-decision interval the moment we enter it
  if (!bSkip.checked || vo.paused) return;
  const g = gapAt(videoToWall(vo.currentTime) - TU[0]); if (g) seekWall(g[1]);
}}
vo.addEventListener('timeupdate', skipGap);
setInterval(() => {{
  if (!vo.paused) {{ skipGap(); if (Math.abs(vw.currentTime - vo.currentTime) > 0.3) vw.currentTime = vo.currentTime; sync3d(); }}
  refresh();
}}, 100);
if (f3d) f3d.addEventListener('load', () => {{ const t = setInterval(() => {{ if (viserTimeInput()) {{ clearInterval(t); last3d = null; sync3d(); }} }}, 500); }});
// key-moment ticks on the timeline
const KIND = {{ grip: 'var(--grip)', block: 'var(--block)', drop: 'var(--drop)', fault: 'var(--fault)' }};
for (const m of {json.dumps([{"t_unix": m["t_unix"], "kind": m["kind"], "label": m["label"]} for m in marks if m["kind"] != "move"])}) {{
  const rel = m.t_unix - TU[0]; if (!WALL_SPAN || rel < 0 || rel > WALL_SPAN) continue;
  const s = document.createElement('span'); s.dataset.rel = rel; s.style.background = KIND[m.kind];
  s.title = m.label; s.onclick = () => seekWall(rel); bTicks.appendChild(s);
}}
relayout();
const css = getComputedStyle(document.documentElement);
const C = {{ left:'#2563eb', right:'#ea580c', drop: css.getPropertyValue('--drop'), block: css.getPropertyValue('--block'),
             grip: css.getPropertyValue('--grip'), fault: css.getPropertyValue('--fault'), move:'#a8a29e' }};
function chart(id, key, title, yMax) {{
  const ds = ['left', 'right'].map(a => ({{ label: a + ' arm', borderColor: C[a], backgroundColor: C[a], pointRadius: 0, borderWidth: 1.5,
      data: S[a].t.map((t, i) => ({{x: t, y: S[a][key][i]}})) }}));
  ds.push({{ label: 'events', type: 'scatter', pointRadius: 3, data: MARKS.filter(m => m.kind !== 'move').map(m => ({{x: m.t, y: yMax * 0.97, m}})),
      backgroundColor: MARKS.filter(m => m.kind !== 'move').map(m => C[m.kind]) }});
  new Chart(document.getElementById(id), {{ type: 'line', data: {{ datasets: ds }},
    options: {{ animation: false, maintainAspectRatio: false, parsing: false,
      plugins: {{ title: {{ display: true, text: title }},
        tooltip: {{ callbacks: {{ label: c => c.raw.m ? c.raw.m.label : (c.dataset.label + ': ' + (c.raw.y ?? '').toString().slice(0, 5)) }} }} }},
      scales: {{ x: {{ type: 'linear', title: {{ display: true, text: 's since start' }} }}, y: {{ min: 0, max: yMax }} }},
      onClick: (e, els) => {{ if (els.length) seek(null); }} }} }});
}}
chart('c_grip', 'grip', 'Gripper opening (0 closed, 1 open)', 1.05);
chart('c_eff', 'eff', 'Gripper effort |eff| (grip force)', 2.0);
chart('c_temp', 'tmax', 'Hottest motor coil per arm (C)', 105);
</script>
</body></html>
"""
    out.write_text(page)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", help="where to write the page (default: <run_dir>_report.html next to the run folder)")
    ap.add_argument("--viser-client", help="URL of a built viser client (viser-build-client) for the 3D replay link")
    a = ap.parse_args()
    print(build(Path(a.run_dir), a.viser_client, Path(a.out) if a.out else None))
