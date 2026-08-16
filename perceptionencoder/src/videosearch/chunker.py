"""Split a video into fixed-length time windows and sample frames from each."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image


@dataclass
class Chunk:
    video: str
    start: float
    end: float
    frames: list[Image.Image]


def _read_frame_at(cap: "cv2.VideoCapture", frame_idx: int) -> Image.Image | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def iter_chunks(
    video_path: str | Path,
    chunk_sec: float = 2.0,
    overlap_sec: float = 0.0,
    frames_per_chunk: int = 8,
) -> Iterator[Chunk]:
    """Yield time-windowed chunks with uniformly sampled frames.

    A chunk with no readable frames is skipped.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or total_frames <= 0:
            return
        duration = total_frames / fps

        step = max(chunk_sec - overlap_sec, 1e-3)
        min_chunk = min(0.5, chunk_sec * 0.5)  # skip tiny trailing slivers
        start = 0.0
        while start < duration - 1e-6:
            end = min(start + chunk_sec, duration)
            if end - start < min_chunk:
                break
            # Uniformly spaced sample timestamps within [start, end).
            times = np.linspace(start, end, num=frames_per_chunk, endpoint=False)
            frames: list[Image.Image] = []
            for t in times:
                idx = min(int(round(t * fps)), total_frames - 1)
                img = _read_frame_at(cap, idx)
                if img is not None:
                    frames.append(img)
            if frames:
                yield Chunk(video=video_path, start=float(start), end=float(end), frames=frames)
            start += step
    finally:
        cap.release()
