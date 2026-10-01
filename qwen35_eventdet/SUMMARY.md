# qwen35 — build & run summary

Event detection (fall / fight / self_harm) with `qwen/qwen3.5-35b-a3b` via
OpenRouter. No GPU used — ffmpeg samples locally, all inference is API.

## Phase 1 — text probe (`text_probe.py`, results/text_probe/)

8/8 calls OK, $0.0057 total, 101s wall. Key findings:

| finding | evidence |
|---|---|
| thinking ON ≈ 20–25s/call, correct verdicts | runs A, D, F |
| thinking OFF ≈ 1–4s/call, also correct (clear cases) | runs B, C, E, G |
| `reasoning {"enabled": false}` works; reasoning returns in `message.reasoning`, never inline in content | runs B/C/E/G |
| seed=42 → byte-identical answers + identical token counts | runs D.1/D.2 |
| `response_format json_schema` works (thinking off) | run E |
| routing varied: Darkbloom (fp4) / SiliconFlow (fp8) | all runs |

## Phase 2 — video detector (`detect.py` + src/qwen35/)

Design: 5s windows → client-side ffmpeg sampling at 8 fps → ~40 JPEG frames
(512px) → each preceded by `<T.TTT seconds>` text (Qwen3.5's native
text-timestamp alignment format) → strict verdict prompt
(COUNTS / TELLTALE SIGNS ACROSS FRAMES / DO NOT COUNT per event) →
JSON verdict with confidence, time range, evidence, alternatives.
Rationale: OpenRouter drops `mm_processor_kwargs`, so native `video_url`
would be server-sampled at Qwen's default fps=2 (~10 frames / 5s) — too
coarse for fast events. See docs/ARCHITECTURE.md, docs/PROMPTS.md.

Smoke: fall_03 single window → FALL conf 95, time 2.5–3.5s, 46s, $0.0035
(routed SiliconFlow).

### Trial: ../qwen sample set (thinking ON, frames mode, 8 fps)

33 windows planned over 8 videos; 30 completed, 3 lost to an OpenRouter
**402 credits-exhausted** error on fight_staged w00/w03/w04 (account ran
out mid-run; ~$0.06 total spend until then).

| video | windows | result |
|---|---|---|
| bg_animation | 2 | clean, none/100 ×2 |
| distractor_combatives (sparring) | 12 | **clean, none/85–100 ×12** — the classic fight FP, refuted in every window |
| distractor_dancing | 3 | clean, none/95 ×3 |
| distractor_yoga | 5 | clean, none/95–100 ×5 (girl lying on rug doing exercises — refuted) |
| fall_01 | 2 | **fall conf 95** @ 0–5s (3.0–4.4s) |
| fall_02 | 2 | **fall conf 95** @ 0–5s |
| fall_03 | 1 | **miss this pass**: none/90, "slowly and deliberately lowers themselves onto the floor" |
| fight_staged | 6 | **fight conf 75** @ 5–10s; 3 windows lost to 402 |

Notes:
- All 22 distractor/background windows → **zero false positives**, single
  call each (../qwen needed a local embedding stage + API stage for the
  same outcome; its combatives FP required Stage 2 to kill).
- fall_03 is a slow, staged chair-collapse; the smoke run caught it
  (conf 95, "sliding off chair, no hand support", SiliconFlow) but the
  trial pass (different routing) read it as deliberate lowering. Root
  cause: the fall definition lacked a bracing/no-bracing contrast.
  → fixed in `src/qwen35/prompts.py` (TELLTALE: "slides or slumps off a
  chair ... with NO bracing - arms hanging, no hands pushing off the seat,
  no feet planted first, no glance at the floor, landing in an awkward
  heap"; DO NOT COUNT: deliberate lowering WITH control). **Pending
  re-validation** (blocked on credits).
- fight_staged detected as fight conf 75 — ../qwen's 235B-Thinking had
  refuted this sample as play ("moving in sync, no strikes"); Qwen3.5-35B
  reads it as a real altercation. The sample IS a staged fight, so this is
  plausibly the correct label (../qwen called it an open labeling question).
- Latency ~25–46s/window (thinking ON); cost ~$0.003–0.004/window.
  `--no-thinking` is ~3–4s/window (Phase 1), for faster passes.

## Open items

- **Credits**: the key's account is free-tier with **$0 purchased credits**
  (`GET /api/v1/credits` → `total_credits: 0`, `total_usage: $0.157`,
  `is_free_tier: true`). The trial+revalidation ran on an initial allowance
  that is now exhausted. Deposit needed at openrouter.ai/credits (account
  owning key `sk-or-v1-682...97e`), or switch to a `:free` model — only
  `qwen/qwen3.8-27b:free` (text+image+video, 50 req/day) exists.
- Re-validation after the fall prompt fix: **fall_03 now caught** — FALL
  conf 90, 2.250s-3.500s ("leans forward and collapses"), single window.
  fight_staged partial (2/6 windows before 402): w00 none/100, w01 none/90
  ("brief physical contact that resolves") — w01 read as FIGHT conf 75 in
  the first trial; borderline staged sample, routing varies. w02-w05 lost.
- Validation of `--no-thinking` + `--structured auto` on video (fast mode).
- `--mode video` transport comparison (server-side fps=2 sampling).
- Optional: provider pinning (`provider.order`) for routing consistency.
