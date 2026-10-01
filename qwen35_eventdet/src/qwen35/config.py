"""Central configuration: model, transport, sampling, presets, prices."""

from __future__ import annotations

import os

MODEL_DEFAULT = "qwen/qwen3.5-35b-a3b"
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"

EVENTS: tuple[str, ...] = ("fall", "fight", "self_harm")

# --- windowing & sampling (client-side, guarantees the target fps) ---
WINDOW_S = 5.0  # analysis window length
STRIDE_S = 5.0  # step between window starts (use < WINDOW_S for overlap)
FPS = 8.0  # frames per second sampled from each window
MIN_WINDOW_S = 1.0  # drop tail windows shorter than this
FRAME_LONG_SIDE = 512  # long-side cap for extracted frames (multiple of 32)
MAX_FRAMES = 48  # hard cap on frames per window (6s at 8fps)

# --- generation ---
MAX_TOKENS = 8192
TIMEOUT_S = 240.0
MAX_RETRIES = 2

# Qwen3.5 recommended sampling presets (model card "Best Practices"):
# thinking mode: temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5
# non-thinking: temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5
THINKING_PRESET = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}
NON_THINKING_PRESET = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}

# Canonical OpenRouter pricing (USD/token); used only for cost estimates when
# OpenRouter does not report usage.cost. Darkbloom (fp4) is cheaper; routed
# provider may vary.
PRICE_PROMPT = 0.0000001625
PRICE_COMPLETION = 0.0000013

# --- inference output ---
RESULTS_DIR = os.environ.get("QWEN35_RESULTS_DIR", "results")
FRAMES_DIR = os.environ.get("QWEN35_FRAMES_DIR", "results/_frames")


def api_key() -> str | None:
    return os.environ.get(API_KEY_ENV)
