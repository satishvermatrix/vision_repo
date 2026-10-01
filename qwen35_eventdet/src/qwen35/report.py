"""Console rendering and per-video report assembly."""

from __future__ import annotations

import json
from pathlib import Path

from .client import estimate_cost


def _flat(s: str | None, n: int) -> str:
    if not s:
        return "-"
    t = " ".join(s.split())
    return t[:n] + ("..." if len(t) > n else "")


def print_window(idx: int, start: float, end: float, rec: dict | None, err: str | None):
    span = f"w{idx:02d} {start:6.1f}-{end:6.1f}s"
    if err:
        print(f"  {span} | ERROR: {_flat(err, 120)}")
        return
    present = [e for e, v in (rec.get("event_present") or {}).items() if v]
    verdict = rec.get("detected_event_type") or "none"
    conf = rec.get("confidence")
    if present:
        head = f"{'/'.join(present).upper()} conf {conf}"
    else:
        head = f"none (certainty {conf})"
    print(f"  {span} | {head}")
    print(f"    time_range: {rec.get('time_range_in_clip')} | summary: {_flat(rec.get('summary'), 110)}")


def print_video_report(video: Path, result: dict):
    print(f"\n=== {video.name} ({result['duration_s']:.1f}s, {result['n_windows']} windows) ===")
    for w in result["windows"]:
        print_window(w["idx"], w["start"], w["end"], w.get("verdict"), w.get("error"))
    confirmed = result.get("confirmed_events") or []
    if confirmed:
        print(f"  --> flagged windows: {len(confirmed)}")
        for c in confirmed:
            print(
                f"      {c['event']} @ {c['window'][0]:.1f}-{c['window'][1]:.1f}s "
                f"conf {c['confidence']} | {_flat(c['summary'], 90)}"
            )
    else:
        print("  --> no events detected")
    total_cost, est = _total_cost(result)
    print(
        f"  model: {result['model']} | mode: {result['mode']} | thinking: "
        f"{result['thinking']} | elapsed {result['elapsed_s']}s | cost "
        f"{'~' if est else ''}${total_cost:.4f}"
    )


def _total_cost(result: dict) -> tuple[float, bool]:
    total, any_est = 0.0, False
    for w in result["windows"]:
        if w.get("usage"):
            c, e = estimate_cost(w["usage"])
            total += c
            any_est = any_est or e
    return total, any_est


def save_report(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{Path(result['video']).stem}_report.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    return path
