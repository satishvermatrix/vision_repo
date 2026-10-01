"""Window planning, frame extraction, and timestamped content assembly.

Two transports (see docs/ARCHITECTURE.md):

- frames (default): ffmpeg samples each 5s window at 8 fps (~40 frames),
  frames are downscaled and sent as base64 JPEG image_url parts, each
  preceded by a '<T.TTT seconds>' text part. This replicates Qwen3.5's
  native text-timestamp alignment format and guarantees the target fps,
  because OpenRouter does not forward mm_processor_kwargs to providers
  (server-side sampling is the Qwen default fps=2).
- video: the window is re-encoded as a small mp4 and sent as a base64
  video_url part; the provider then does its own frame sampling.
"""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path


def ffprobe(path: Path) -> dict:
    """Return duration_s, fps, width, height for a video file."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_streams", "-show_format", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    v = next(s for s in info["streams"] if s.get("codec_type") == "video")
    rate = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1"
    num, _, den = rate.partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    return {
        "duration_s": float(info["format"]["duration"]),
        "fps": fps,
        "width": int(v.get("width", 0)),
        "height": int(v.get("height", 0)),
    }


def plan_windows(
    duration: float, window: float, stride: float, min_len: float
) -> list[tuple[float, float]]:
    """Split [0, duration) into windows of `window` seconds stepping `stride`.

    Short videos get a single shorter window; tail windows shorter than
    min_len are dropped.
    """
    if duration < min_len:
        return []
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration - 1e-6:
        end = min(start + window, duration)
        if end - start >= min_len:
            windows.append((round(start, 3), round(end, 3)))
        start += stride
    return windows


def extract_window_frames(
    video: Path,
    start: float,
    end: float,
    fps: float,
    long_side: int,
    max_frames: int,
    out_dir: Path,
    prefix: str,
) -> list[tuple[Path, float]]:
    """Sample one window at `fps` with ffmpeg; return (frame_path, t_in_window)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{start:.3f}", "-i", str(video),
            "-t", f"{end - start:.3f}",
            "-vf",
            f"fps={fps},scale={long_side}:{long_side}:"
            "force_original_aspect_ratio=decrease:force_divisible_by=2",
            "-frames:v", str(max_frames),
            "-q:v", "2",
            str(out_dir / f"{prefix}_%03d.jpg"),
        ],
        check=True, capture_output=True, text=True,
    )
    frames = sorted(out_dir.glob(f"{prefix}_*.jpg"))
    return [(f, i / fps) for i, f in enumerate(frames)]


def encode_window_clip(
    video: Path, start: float, end: float, long_side: int, out_path: Path
) -> Path:
    """Re-encode one window as a small mp4 (for --mode video)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{start:.3f}", "-i", str(video),
            "-t", f"{end - start:.3f}",
            "-vf",
            f"scale={long_side}:{long_side}:"
            "force_original_aspect_ratio=decrease:force_divisible_by=2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-an",
            "-movflags", "+faststart",
            str(out_path),
        ],
        check=True, capture_output=True, text=True,
    )
    return out_path


def b64_data_url(path: Path, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def build_frame_parts(frame_items: list[tuple[Path, float]]) -> list[dict]:
    """Interleave Qwen-native timestamp labels with base64 frame images.

    Format mirrors the Qwen3.5 processor: each frame group is preceded by
    its '<T.TTT seconds>' timestamp text; timestamps use 3 decimals because
    at 8 fps the native 1-decimal form would collide (0.125 vs 0.250).
    """
    parts: list[dict] = []
    for path, t in frame_items:
        parts.append({"type": "text", "text": f"<{t:.3f} seconds>"})
        parts.append(
            {"type": "image_url", "image_url": {"url": b64_data_url(path, "image/jpeg")}}
        )
    return parts


def estimate_frame_tokens(n_frames: int, long_side: int) -> int:
    """Approximate visual tokens: (long_side/32)^2 per frame (patch 16 x merge 2)."""
    per_frame = (long_side // 32) ** 2
    return n_frames * per_frame
