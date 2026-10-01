#!/usr/bin/env python3
"""Video event detection with Qwen3.5-35B-A3B via OpenRouter (Phase 2).

Input videos are split into 5s windows; each window is sampled client-side at
8 fps into ~40 timestamped JPEG frames ('<T.TTT seconds>' labels, Qwen3.5's
native text-timestamp alignment format) and sent to qwen/qwen3.5-35b-a3b
with a strict verdict prompt for fall / fight / self_harm.

Usage:
  uv run env OPENROUTER_API_KEY=sk-or-... python detect.py VIDEO [VIDEO...] [options]
  uv run python detect.py ../qwen/data/videos/ --dry-run
  uv run env OPENROUTER_API_KEY=... python detect.py clip.mp4 --no-thinking
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from qwen35 import config  # noqa: E402
from qwen35.client import QwenClient  # noqa: E402
from qwen35.frames import (  # noqa: E402
    build_frame_parts,
    encode_window_clip,
    estimate_frame_tokens,
    extract_window_frames,
    ffprobe,
    plan_windows,
)
from qwen35.prompts import build_system_prompt  # noqa: E402
from qwen35.report import print_video_report, save_report  # noqa: E402


def gather_videos(paths: list[str]) -> list[Path]:
    videos: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            videos.extend(
                sorted(
                    q
                    for q in path.iterdir()
                    if q.suffix.lower() in {".mp4", ".webm", ".ogv", ".mkv", ".mov", ".avi"}
                )
            )
        elif path.is_file():
            videos.append(path)
        else:
            sys.exit(f"error: no such file or directory: {path}")
    if not videos:
        sys.exit("error: no input videos")
    return videos


def detect_video(
    video: Path,
    client: QwenClient | None,
    args,
) -> dict:
    t0 = time.perf_counter()
    meta = ffprobe(video)
    windows = plan_windows(meta["duration_s"], args.window, args.stride, args.min_window)
    if not windows:
        raise ValueError(
            f"{video.name} is {meta['duration_s']:.1f}s, shorter than min window {args.min_window}s"
        )
    frames_root = Path(args.frames_dir) / video.stem

    window_records: list[dict] = []
    for idx, (start, end) in enumerate(windows):
        rec = {"idx": idx, "start": start, "end": end, "n_frames": None, "usage": None}
        try:
            if args.mode == "video":
                clip = encode_window_clip(
                    video, start, end, args.frame_size,
                    frames_root / f"w{idx:02d}.mp4",
                )
                rec["n_frames"] = None  # provider-side sampling
                out = client.analyze_video(
                    clip, events=args.events, fps=args.fps, dur=end - start
                )
            else:
                items = extract_window_frames(
                    video, start, end, args.fps, args.frame_size,
                    args.max_frames, frames_root / f"w{idx:02d}", "f",
                )
                if not items:
                    raise ValueError("ffmpeg produced no frames for this window")
                rec["n_frames"] = len(items)
                out = client.analyze_frames(
                    build_frame_parts(items),
                    events=args.events,
                    fps=args.fps,
                    dur=end - start,
                )
            rec["usage"] = out["usage"]
            rec["verdict"] = out
        except Exception as e:  # keep sweeping windows
            rec["error"] = f"{type(e).__name__}: {e}"
        window_records.append(rec)

    confirmed = []
    for w in window_records:
        v = w.get("verdict") or {}
        present = [e for e, on in (v.get("event_present") or {}).items() if on]
        if present and (v.get("confidence") or 0) >= 50:
            confirmed.append(
                {
                    "event": max(present, key=lambda e: v["event_present"][e]) if len(present) > 1 else present[0],
                    "type": v.get("detected_event_type"),
                    "confidence": v.get("confidence"),
                    "time_range_in_clip": v.get("time_range_in_clip"),
                    "summary": v.get("summary"),
                    "visual_evidence": v.get("visual_evidence"),
                    "window": [w["start"], w["end"]],
                    "idx": w["idx"],
                }
            )

    return {
        "video": str(video),
        "duration_s": meta["duration_s"],
        "source_fps": meta["fps"],
        "resolution": f"{meta['width']}x{meta['height']}",
        "n_windows": len(windows),
        "windows": window_records,
        "confirmed_events": confirmed,
        "mode": args.mode,
        "fps": args.fps,
        "window_s": args.window,
        "stride_s": args.stride,
        "frame_size": args.frame_size,
        "events": args.events,
        "thinking": args.thinking,
        "structured": args.structured,
        "model": args.model,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


def dry_run(videos: list[Path], args):
    print(f"dry run: model={args.model} mode={args.mode} thinking={args.thinking}")
    print(
        f"windows: {args.window}s / stride {args.stride}s | sampling: {args.fps:g} fps "
        f"| frame size: {args.frame_size}px | events: {', '.join(args.events)}\n"
    )
    n_windows_total = 0
    for video in videos:
        try:
            meta = ffprobe(video)
        except Exception as e:
            print(f"- {video.name}: ffprobe failed ({e})")
            continue
        windows = plan_windows(meta["duration_s"], args.window, args.stride, args.min_window)
        n_windows_total += len(windows)
        per_frame = estimate_frame_tokens(1, args.frame_size)
        est_in = estimate_frame_tokens(
            min(int(args.fps * args.window) + 1, args.max_frames), args.frame_size
        ) + 900  # system + user prompt
        print(
            f"- {video.name}: {meta['duration_s']:.1f}s @ {meta['fps']:.1f}fps "
            f"{meta['width']}x{meta['height']} -> {len(windows)} window(s) "
            f"~{est_in} prompt tok/window ({per_frame} tok/frame)"
        )
        for i, (s, e) in enumerate(windows):
            print(f"    w{i:02d} {s:6.1f}-{e:6.1f}s")
    per_call = 22.0 if args.thinking else 4.0
    print(
        f"\n{n_windows_total} API call(s) | est. {n_windows_total * per_call:.0f}s wall "
        f"(~{per_call:.0f}s/call, thinking {'on' if args.thinking else 'off'}) "
        f"| ~${n_windows_total * (est_in * config.PRICE_PROMPT + 1500 * config.PRICE_COMPLETION):.3f}"
    )
    print("\nsystem prompt preview:")
    print(build_system_prompt(args.fps)[:1500] + "\n[...]")
    print(f"\nre-run without --dry-run (and with {config.API_KEY_ENV}) to execute.")


def main():
    ap = argparse.ArgumentParser(
        description="Video event detection (fall/fight/self_harm) with "
        "Qwen3.5-35B-A3B via OpenRouter"
    )
    ap.add_argument("videos", nargs="+", help="video file(s) or directory")
    ap.add_argument("--events", default=",".join(config.EVENTS),
                    help=f"comma list from {config.EVENTS} (default all)")
    ap.add_argument("--window", type=float, default=config.WINDOW_S)
    ap.add_argument("--stride", type=float, default=config.STRIDE_S,
                    help="window step; < window gives overlap")
    ap.add_argument("--min-window", type=float, default=config.MIN_WINDOW_S)
    ap.add_argument("--fps", type=float, default=config.FPS)
    ap.add_argument("--max-frames", type=int, default=config.MAX_FRAMES)
    ap.add_argument("--frame-size", type=int, default=config.FRAME_LONG_SIDE)
    ap.add_argument("--mode", choices=("frames", "video"), default="frames",
                    help="frames: client-sampled timestamped JPEGs (default); "
                    "video: base64 mp4 (provider-side sampling)")
    ap.add_argument("--thinking", dest="thinking", action="store_true", default=True)
    ap.add_argument("--no-thinking", dest="thinking", action="store_false")
    ap.add_argument("--structured", choices=("auto", "always", "never"), default="auto",
                    help="json_schema response_format: auto=only when thinking off")
    ap.add_argument("--model", default=config.MODEL_DEFAULT)
    ap.add_argument("--max-tokens", type=int, default=config.MAX_TOKENS)
    ap.add_argument("--timeout", type=float, default=config.TIMEOUT_S)
    ap.add_argument("--out-dir", default=config.RESULTS_DIR)
    ap.add_argument("--frames-dir", default=config.FRAMES_DIR)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    events = [e.strip() for e in args.events.split(",") if e.strip()]
    bad = [e for e in events if e not in config.EVENTS]
    if bad:
        sys.exit(f"error: unknown events {bad}; valid: {list(config.EVENTS)}")
    args.events = events

    videos = gather_videos(args.videos)

    if args.dry_run:
        dry_run(videos, args)
        return

    client = QwenClient(
        model=args.model,
        thinking=args.thinking,
        structured=args.structured,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    out_dir = Path(args.out_dir)
    for video in videos:
        print(f"[detect] {video.name} ...")
        try:
            result = detect_video(video, client, args)
        except ValueError as e:
            print(f"  skipped: {e}")
            continue
        print_video_report(video, result)
        path = save_report(result, out_dir)
        print(f"  report saved: {path}")


if __name__ == "__main__":
    main()
