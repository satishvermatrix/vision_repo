# qwen35: video event detection with Qwen3.5-35B-A3B (OpenRouter)

Single-stage video event detector: 5-second surveillance windows are sampled
client-side at 8 fps and sent as ~40 timestamped JPEG frames to
`qwen/qwen3.5-35b-a3b` via OpenRouter, which returns a strict JSON verdict
for **fall / fight / self_harm** with confidence, time range, evidence, and
benign-lookalike refutation.

```
video ──ffmpeg──> 5s windows ──8fps──> ~40 frames @ 512px
                                          │ each frame preceded by '<T.TTT seconds>'
                                          │   (Qwen3.5 native text-timestamp alignment)
                                          ▼
              qwen/qwen3.5-35b-a3b (OpenRouter, thinking ON by default)
                                          │
                                          ▼
              { event_present: {fall, fight, self_harm}, detected_event_type,
                confidence 0-100, time_range_in_clip, summary,
                visual_evidence[], possible_alternative_explanations }
```

Phase 1 (`text_probe.py`) validated API access, parameter pass-through
(reasoning on/off, seed, structured outputs), and JSON behavior before any
video calls — see `results/text_probe/` and `SUMMARY.md`.

## Why client-side 8 fps sampling (not `video_url`)

OpenRouter does **not** forward `mm_processor_kwargs` to providers, so with
native video input the server samples at the Qwen3.5 default **fps=2**
(~10 frames from a 5s clip) — too coarse for fast events like strikes or
falls. Sampling frames client-side at 8 fps and interleaving them with
`<T.TTT seconds>` timestamp text replicates the model's native
text-timestamp alignment format and guarantees the sampling rate on every
provider. See `docs/ARCHITECTURE.md`.

## Setup

Requires: uv, ffmpeg/ffprobe (system), an OpenRouter API key.

```bash
uv sync
export OPENROUTER_API_KEY=sk-or-...
```

## Usage

```bash
# plan only (no key, no calls): windows, frame counts, token/cost estimates
uv run python detect.py VIDEO.mp4 --dry-run

# detect on a 5s clip (or a longer video / a directory)
uv run env OPENROUTER_API_KEY=sk-or-... python detect.py clip.mp4
uv run env OPENROUTER_API_KEY=sk-or-... python detect.py data/videos/

# fast mode: thinking off + OpenRouter structured outputs (~4s/window)
uv run env OPENROUTER_API_KEY=sk-or-... python detect.py clip.mp4 --no-thinking

# compare transports: native base64 video_url (server-side fps=2 sampling)
uv run env OPENROUTER_API_KEY=sk-or-... python detect.py clip.mp4 --mode video

# text-only parameter probe (Phase 1)
uv run env OPENROUTER_API_KEY=sk-or-... python text_probe.py
```

Options that matter: `--events fall,fight,self_harm` · `--window 5 --stride 5`
(use stride < window for overlap) · `--fps 8` · `--frame-size 512` (visual
tokens ≈ (size/32)^2 per frame, 256 at 512px) · `--thinking/--no-thinking`
· `--structured auto|always|never` (json_schema response_format; `auto` uses
it only when thinking is off — the combos validated in Phase 1).

## Output

Per video: console report + `results/<stem>_report.json` containing per-window
verdicts (event booleans, type, confidence, time range, summary, evidence,
alternatives, provider, latency, token usage) and aggregated
`confirmed_events` (any event marked present with confidence >= 50).
Extracted frames are kept under `results/_frames/<stem>/wNN/` for auditing.

## Layout

```
text_probe.py             Phase 1: text-only parameter probe
detect.py                 Phase 2 CLI: windowing, sampling, verdicts, reports
src/qwen35/
  config.py               model, sampling/window defaults, presets, prices
  prompts.py              the prompt bank (COUNTS / TELLTALE SIGNS / DO NOT COUNT)
  frames.py               ffprobe, window planning, 8fps extraction, timestamp parts
  client.py               OpenRouter client, thinking presets, JSON parsing/normalizing
  report.py               console + JSON report rendering
docs/ARCHITECTURE.md      model architecture + how it processes video
docs/PROMPTS.md           prompt design rationale
SUMMARY.md                build/run record
```

Prompt bank is data: to fix a false positive, extend that event's
DO NOT COUNT list in `src/qwen35/prompts.py` (see docs/PROMPTS.md).
