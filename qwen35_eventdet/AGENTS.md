# AGENTS.md

Video event detection (fall / fight / self_harm) with Qwen3.5-35B-A3B via
OpenRouter: 5s windows sampled client-side at 8 fps as timestamped JPEG
frames. Read `README.md` for usage, `docs/ARCHITECTURE.md` for the model /
video-processing background, `docs/PROMPTS.md` before touching prompts, and
`SUMMARY.md` for the build/run record.

## Environment

- Device: NVIDIA DGX Spark (GB10 aarch64, 131 GB unified memory), but this
  project needs **no GPU** — all inference is via OpenRouter; only
  ffmpeg/ffprobe run locally.
- Python 3.12 in a uv-managed venv. Use `uv run ...`; never pip-install.
  Single dependency: `openai` (plus ffmpeg/ffprobe on PATH).
- API access needs `OPENROUTER_API_KEY` in the environment. Env vars do NOT
  persist between shell calls here — pass inline:
  `uv run env OPENROUTER_API_KEY=sk-or-... python detect.py ...`.

## Commands

```bash
uv sync
uv run python detect.py ../qwen/data/videos/ --dry-run     # plan only
uv run env OPENROUTER_API_KEY=sk-or-... python detect.py clip.mp4
uv run env OPENROUTER_API_KEY=sk-or-... python detect.py clip.mp4 --no-thinking
uv run env OPENROUTER_API_KEY=sk-or-... python text_probe.py            # Phase 1 probe
```

- Defaults: thinking ON (~20–45s/window, reasoning captured in reports);
  `--no-thinking` ≈ 3–4s/window with structured outputs (`--structured auto`).
- `--mode video` sends a base64 mp4 instead of frames (server-side sampling,
  Qwen default fps=2 — comparison only, see docs/ARCHITECTURE.md).

## Code map

`text_probe.py` (Phase 1 text probe) · `detect.py` (CLI) · `src/qwen35/`:
`config.py` (model, sampling presets, windowing defaults, prices) ·
`prompts.py` (prompt bank — the quality lever) · `frames.py` (ffprobe, window
planning, 8fps extraction, timestamped content parts) · `client.py`
(OpenRouter client, thinking/structured modes, JSON extraction/normalizing)
· `report.py` (console + JSON reports).

## Conventions

- Prompt bank is data: false positives → extend the event's DO NOT COUNT
  list; misses → extend TELLTALE SIGNS. Do not add per-video logic.
- Reports/CSVs/frames under `results/` are outputs; do not edit by hand.
- Long API runs (thinking mode, many windows) must run in the background
  with a log file (`nohup env OPENROUTER_API_KEY=... uv run python -u
  detect.py ... > /tmp/opencode/<name>.log 2>&1 &`), then poll with `tail`.
- Keep answers strict-JSON; the answer schema in `prompts.py` and
  `VERDICT_JSON_SCHEMA` must stay in sync (three events, same fields).
- Sample videos live in `../qwen/data/videos/` with `labels.json` — reuse
  them for trials; do not copy 100+ MB into this repo.
