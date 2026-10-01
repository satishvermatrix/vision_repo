# Qwen3.5-35B-A3B: architecture and video processing

What the model is, how it turns video into tokens, and why this project
samples frames client-side. Sources: HF model card (`Qwen/Qwen3.5-35B-A3B`,
2026-02-24), its `config.json` / `video_preprocessor_config.json`,
transformers `Qwen3VLProcessor` source, Qwen3-VL technical report
(arXiv 2511.21631), and OpenRouter docs/endpoints — verified during the
Phase 1 build.

## 1. Model architecture

Qwen3.5 is natively multimodal — every model in the family (0.8B–397B) is
trained from the start on interleaved text/image/video tokens (early
fusion). There is no separate "-VL" variant.

**35B-A3B** = 35B total parameters, **3B activated per token**.

```
                    ┌────────────────────────────────────────────┐
 video/image ──►    │ Vision encoder (reuses Qwen3-VL ViT)       │
                    │ 27 layers, hidden 1152, patch 16,          │
                    │ temporal patch 2, 2x2 spatial merge        │
                    │ → ~0.4B params, projects to 2048           │
                    └───────────────┬────────────────────────────┘
                                    │ visual tokens + ViT features
 text ─────────────────────────────►│
                                    ▼
                    ┌────────────────────────────────────────────┐
                    │ 40-layer hybrid LM, hidden 2048            │
                    │ 10 × [ 3× Gated DeltaNet→MoE              │
                    │        1× Gated Attention→MoE ]            │
                    │ MoE: 256 experts, 8 routed + 1 shared      │
                    │      active/token (~3B params)            │
                    │ context: 262,144 native (YaRN → 1.01M)     │
                    └────────────────────────────────────────────┘
```

- **Hybrid attention, 3:1.** 30 of 40 layers are Gated DeltaNet *linear*
  attention (O(n) in sequence length — 40 frames ≈ 10k+ visual tokens is
  cheap), 10 layers are standard Gated Attention (16 Q heads / 2 KV heads,
  GQA, head dim 256). Layout repeats 10×.
- **Sparse MoE.** 256 experts, 8 routed + 1 shared active per token;
  expert FFN dim 512. This is why per-token cost looks like a ~3B model
  while knowledge capacity looks like a 35B model.
- **Positional encoding: Interleaved MRoPE.** Only 64 of 256 head dims are
  rotary (partial rotary 0.25); the rotary dims are split into three
  components (temporal/height/width, `mrope_section [11,11,10]`) and
  *interleaved* across frequency bands (a Qwen3-VL innovation — Qwen2.5-VL's
  chunked split left the time axis mostly high-frequency, hurting long
  video). Visual and text positions share this scheme, so interleaving
  `<T.TTT seconds>` text with frames is positionally natural.
- **Text-timestamp alignment.** Qwen3-VL replaced absolute-time T-RoPE with
  *explicit textual timestamps*: the processor prefixes every 2-frame
  temporal patch with `<3.0 seconds>` text. Qwen3.5 keeps this. The model
  therefore reads timestamp labels literally and can emit timestamps in
  answers — the mechanism behind `time_range_in_clip` in our verdict JSON.
- **DeepStack:** Qwen3-VL injects multi-level ViT features into matching
  LLM layers (`deepstack_visual_indexes [8,16,24]`). The released Qwen3.5
  checkpoints ship `deepstack_visual_indexes: []` — not enabled here.
- **Other:** MTP (multi-token prediction) trained for speculative decoding;
  248k vocab (padded); tokenizer "Qwen3"; video benchmarks: VideoMME 86.6
  (w/ subs) / 82.5, MLVU 85.6, MVBench 74.8, LVBench 71.4, MMVU 72.3.
- **Thinking by default.** Qwen3.5 emits `<think>…</think>` before answers
  and has **no** `/think` / `/no_think` soft switch. On OpenRouter, thinking
  is controlled by the `reasoning` parameter (`{"enabled": false}` to
  disable); reasoning returns separately in `message.reasoning`, so it never
  pollutes the JSON verdict. Thinking/non-thinking have official sampling
  presets (see `src/qwen35/config.py`).

## 2. How Qwen3.5 processes video (natively)

Given a `video_url` input:

1. **Frame sampling** (`Qwen3VLVideoProcessor`): default **fps=2** and
   `do_sample_frames=True` → `num_frames = total_frames / source_fps * fps`,
   clamped to [4, **768**] frames; uniform `linspace` sampling. fps and
   `nframes` are mutually exclusive.
2. **Temporal patching:** frames are grouped 2-at-a-time (Conv3D kernel
   (2,16,16); a single image degrades to Conv2D). Odd counts pad by
   repeating the last frame.
3. **Token math:** after 16-px patches + 2×2 merge, each 2-frame group
   yields (H/32)×(W/32) tokens. E.g. 448×448 → 196 tokens per temporal
   patch (98/frame). Total budget: `longest_edge 25,165,824` pixel-frames in
   `video_preprocessor_config.json` = **12,288 video tokens** default
   (~768 tokens/frame ceiling with capping enabled); extendable to 224k
   video tokens for hour-scale video.
4. **Timestamp injection:** `<T.T seconds>` text is prepended to every
   temporal patch group (timestamps = frame_index / source_fps, averaged
   within the 2-frame group).

## 3. Why this project samples at 8 fps client-side (the OpenRouter reality)

- OpenRouter **drops `mm_processor_kwargs`** (not in its parameter schema,
  not forwarded) and does no transcoding itself — the provider's server
  samples video with its own defaults, i.e. **fps=2** → ~10 frames from a
  5-second clip. A strike or a fall lasts 0.3–0.5s; at 2 fps the event can
  fall entirely between two sampled frames.
- Only a self-hosted vLLM (`--media-io-kwargs '{"video": {"num_frames":
  -1}}'` + per-request `mm_processor_kwargs {"fps": 8}`) gives native
  control over fps; that is out of scope for an OpenRouter client.
- So the default transport here is: **ffmpeg samples each 5s window at
  8 fps → ~40 JPEG frames (512px long side) → each frame is sent as a
  base64 `image_url` part preceded by a `<T.TTT seconds>` text part** —
  byte-for-byte the same format the model's own processor would construct
  (3-decimal timestamps because 8 fps needs 125 ms resolution).
  Token cost: (512/32)² = 256 tokens/frame → ~10.2k visual tokens + ~0.9k
  prompt tokens per window (~$0.002 input at canonical pricing).
- `--mode video` sends the window as a small base64 mp4 `video_url` for
  transport comparison — provider-side sampling then applies (fps=2).
- 40 separate images are *not* identical to native video: each image gets
  its own vision group (no 2-frame temporal patches), so temporal
  compression is lost, but frame *count* and *timing* are exactly what we
  choose — the right trade for fast surveillance events.

## 4. What Phase 1 validated (results/text_probe/)

- Model reachable; routed to Darkbloom/SiliconFlow (fp4/fp8 quants;
  Alibaba serves native precision — pin with `provider.order` if needed).
- `reasoning {"enabled": true/false}` works through OpenRouter; reasoning
  returns cleanly in `message.reasoning` (no inline `<think>` in content).
- Thinking ON: ~20s/call, correct verdicts; thinking OFF: ~3s, also
  correct on clear cases; **seed=42 gave byte-identical answers + token
  counts across repeats**.
- `response_format: json_schema` (structured outputs) works with thinking
  off → used by `--structured auto` in `detect.py`.
- Strict JSON prompt discipline is good even without the schema
  (thinking-on runs returned clean JSON).
