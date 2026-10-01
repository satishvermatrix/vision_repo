"""qwen35: video event detection with Qwen3.5-35B-A3B via OpenRouter.

Phase 1 (text_probe.py) validated API access, parameter pass-through, and
JSON behavior. Phase 2 (this package + detect.py) is the video detector:
5s windows sampled client-side at 8 fps, frames interleaved with Qwen-native
'<T.TTT seconds>' timestamp labels, strict verdict prompts for fall / fight /
self_harm, answers normalized from strict JSON.
"""
