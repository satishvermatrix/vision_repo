#!/usr/bin/env python3
"""Download a small Kinetics-400 subset and extract frames."""

from __future__ import annotations

import csv
import subprocess
import urllib.request
from pathlib import Path

ANNOTATION_URL = "https://s3.amazonaws.com/kinetics/400/annotations/train.csv"
CLASSES = [
    "slapping",
    "faceplanting",
    "punching person (boxing)",
    "skydiving",
]
VIDEOS_PER_CLASS = 10
MAX_ATTEMPTS_PER_CLASS = 60
FPS = 1

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
VIDEO_DIR = DATA / "videos"
FRAME_DIR = DATA / "frames"
ANNOT_PATH = DATA / "annotations" / "k400_subset.csv"


def class_dir_name(label: str) -> str:
    return label.replace(" ", "_").replace("(", "").replace(")", "")


def load_or_build_subset() -> list[dict[str, str]]:
    ANNOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ANNOT_PATH.exists():
        with ANNOT_PATH.open(newline="") as f:
            return list(csv.DictReader(f))

    print(f"Downloading annotations from {ANNOTATION_URL}")
    with urllib.request.urlopen(ANNOTATION_URL, timeout=60) as resp:
        text = resp.read().decode("utf-8")

    reader = csv.DictReader(text.splitlines())
    rows = [row for row in reader if row["label"] in CLASSES]
    fieldnames = ["label", "youtube_id", "time_start", "time_end", "split", "is_cc"]
    with ANNOT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} matching rows to {ANNOT_PATH}")
    return rows


def download_clip(youtube_id: str, start: int, end: int, out_path: Path) -> bool:
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "-f",
        "bv*[height<=360]+ba/b[height<=360]/w",
        "--download-sections",
        f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "--merge-output-format",
        "mp4",
        "-o",
        str(out_path),
        "--",
        url,
    ]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"  skip {youtube_id}: {exc}")
        return False

    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 10_000:
        err = (result.stderr or result.stdout or "").strip().splitlines()
        tail = err[-1] if err else f"exit {result.returncode}"
        print(f"  skip {youtube_id}: {tail}")
        if out_path.exists():
            out_path.unlink()
        return False
    return True


def extract_frames(video_path: Path, frame_dir: Path) -> bool:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("frame_*.jpg"):
        old.unlink()
    pattern = str(frame_dir / "frame_%04d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={FPS}",
        "-q:v",
        "3",
        pattern,
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    frames = list(frame_dir.glob("frame_*.jpg"))
    if result.returncode != 0 or len(frames) < 2:
        for f in frames:
            f.unlink()
        if frame_dir.exists():
            frame_dir.rmdir()
        return False
    return True


def main() -> None:
    rows = load_or_build_subset()
    by_class: dict[str, list[dict[str, str]]] = {c: [] for c in CLASSES}
    for row in rows:
        by_class[row["label"]].append(row)

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, int] = {}
    for label in CLASSES:
        folder = class_dir_name(label)
        got = 0
        print(f"\n=== {label} -> {folder} ===")
        for row in by_class[label][:MAX_ATTEMPTS_PER_CLASS]:
            if got >= VIDEOS_PER_CLASS:
                break
            youtube_id = row["youtube_id"]
            start = int(float(row["time_start"]))
            end = int(float(row["time_end"]))
            video_path = VIDEO_DIR / folder / f"{youtube_id}.mp4"
            frame_path = FRAME_DIR / folder / youtube_id

            if frame_path.exists() and list(frame_path.glob("frame_*.jpg")):
                print(f"  have {youtube_id}")
                got += 1
                continue

            video_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  download {youtube_id} ({start}-{end}s) [{got + 1}/{VIDEOS_PER_CLASS}]")
            if not download_clip(youtube_id, start, end, video_path):
                continue
            if not extract_frames(video_path, frame_path):
                print(f"  skip {youtube_id}: ffmpeg extracted too few frames")
                continue
            n_frames = len(list(frame_path.glob("frame_*.jpg")))
            print(f"  ok {youtube_id} ({n_frames} frames)")
            got += 1
        summary[folder] = got
        if got < VIDEOS_PER_CLASS:
            print(f"  only got {got}/{VIDEOS_PER_CLASS} clips")

    print("\nDone:")
    for folder, n in summary.items():
        print(f"  {folder}: {n} clips -> {FRAME_DIR / folder}")


def reextract_existing() -> None:
    """Re-cut frames from already-downloaded mp4s at FPS (no YouTube)."""
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"No videos at {VIDEO_DIR}")
    n_ok = 0
    n_fail = 0
    for video_path in sorted(VIDEO_DIR.glob("*/*.mp4")):
        rel = video_path.relative_to(VIDEO_DIR)
        frame_path = FRAME_DIR / rel.parent / video_path.stem
        print(f"  {rel} @ {FPS} fps")
        if extract_frames(video_path, frame_path):
            n_frames = len(list(frame_path.glob("frame_*.jpg")))
            print(f"    ok ({n_frames} frames)")
            n_ok += 1
        else:
            print("    fail")
            n_fail += 1
    print(f"\nRe-extracted {n_ok} clips at {FPS} fps ({n_fail} failed)")


if __name__ == "__main__":
    import sys

    if "--reextract-only" in sys.argv:
        reextract_existing()
    else:
        main()
