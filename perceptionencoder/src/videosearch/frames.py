"""Sample frames from a video time window for VLM input."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def sample_window_frames(
    video_path: str | Path,
    start: float,
    end: float,
    num_frames: int = 6,
) -> list[Image.Image]:
    """Uniformly sample RGB PIL frames from [start, end) seconds."""
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or total <= 0:
            return []
        duration = total / fps
        start = max(0.0, float(start))
        end = min(float(end), duration)
        if end <= start:
            end = min(start + 0.5, duration)

        times = np.linspace(start, end, num=num_frames, endpoint=False)
        frames: list[Image.Image] = []
        for t in times:
            idx = min(int(round(t * fps)), total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        return frames
    finally:
        cap.release()
