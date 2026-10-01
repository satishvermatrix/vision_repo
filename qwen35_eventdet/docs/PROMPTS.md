# Prompt design for the Qwen3.5-35B-A3B event detector

The prompt bank lives in `src/qwen35/prompts.py`. It is data: to fix a
false positive, extend that event's DO NOT COUNT list; to fix a miss,
extend TELLTALE SIGNS — then re-run. This note explains each design choice.

## Shape of the prompt

Every window gets one system prompt + one user message:

```
system: verdict engine instructions
        (frame/timestamp format, event definitions, judgment rules,
         calibrated confidence, exact JSON schema)
user:   <task text>
        <0.000 seconds> [frame 0]
        <0.125 seconds> [frame 1]
        ...                                  (~40 frames @ 8 fps)
        "Answer with the JSON object only."
```

## Choice 1 — COUNTS / TELLTALE SIGNS / DO NOT COUNT per event

Each event is defined in three blocks, and the two "evidence" blocks are
written for what a *frame-sampled clip* actually shows:

- **COUNTS** — the qualifying definition, with the boundary cases made
  explicit (stairs, collapse, knocked down; one-sided assault counts as
  fight; punching a wall does NOT count as self_harm, punching your own
  head does).
- **TELLTALE SIGNS ACROSS FRAMES** — observable, *temporal* evidence:
  downward acceleration, failed balance recovery, impact + aftermath
  (lying still, failed get-up attempts, people rushing to help), flinch /
  block / retreat of a strike target, onlooker scattering. At 8 fps the
  single most informative channel is *change between consecutive frames*,
  so the prompt says so explicitly (rule 2) and every event's cues are
  phrased as cross-frame sequences.
- **DO NOT COUNT** — the benign lookalikes that must be refuted. This is
  the main false-positive lever, carried over from ../qwen's lesson: the
  distractor bank (yoga, sparring, dancing, play-wrestling, first aid,
  cooking) is what kills false positives, not terser event definitions.
  The list per event matches the known hard negatives for each class.

## Choice 2 — Qwen-native timestamp format

Frames are interleaved with `<T.TTT seconds>` text labels — the *same
format the model's own processor constructs* for video (`replace_video_token`
emits `f"<{t} seconds>" + <|vision_start|>…` per temporal patch). Qwen3.5's
text-timestamp alignment was trained with these, so temporal grounding
(`time_range_in_clip`) comes for free. 3 decimals (vs the native 1) because
8 fps needs 125 ms resolution.

## Choice 3 — aftermath rule (rule 3)

A fall or strike can happen *between* two frames or start before the first
frame. The prompt makes aftermath evidence (person sprawled, being helped
up, visibly injured) sufficient to count, and demands the summary say so.
Without this rule the model tends to answer "none" when the exact impact
moment is missing.

## Choice 4 — explicit refutation step (rule 5)

Before finalizing, the model must actively match the DO NOT COUNT list and,
if a lookalike fits better, mark false and explain. This converts the
lookalike list from decoration into a mandatory comparison, mirroring
../qwen's margin condition (event vs lookalike) inside a single call.

## Choice 5 — calibrated confidence (rule 6)

A fixed verbal scale (90–100 unmistakable … 0–29 unlikely) so numbers are
comparable across calls and aggregatable per video; the same scale ../qwen's
Stage 2 used successfully. `detect.py` counts a window as confirmed at
confidence >= 50, which corresponds to "likely".

## Choice 6 — strict JSON + OpenRouter structured outputs

- The exact answer schema is shown in the system prompt (output-format
  standardization, per the Qwen3.5 model card's best practices).
- With `--no-thinking`, `detect.py` also passes `response_format:
  json_schema` (`--structured auto`) so the API *enforces* the schema
  (validated in Phase 1 run E).
- With thinking on, the prompt is the only authority (Phase 1 run A
  returned clean JSON anyway); `extract_json()` is the safety net (direct
  parse → fenced block → brace scan).
- Trailing "Answer with the JSON object only." after the frames — a second
  format anchor at the far end of a ~10k-token visual sequence.

## Choice 7 — thinking mode policy

Thinking stays ON by default: ambiguous surveillance footage benefits from
reasoning (../qwen's thinking Stage 2 produced decisive lookalike
refutations), and Qwen3.5 reasoning returns separately
(`message.reasoning`), captured per window in the reports. `--no-thinking`
switches to the fast preset (~3s vs ~20–45s per window) with structured
outputs on. Both use the model card's official sampling presets
(thinking: temp 1.0 / top_p 0.95 / top_k 20 / presence_penalty 1.5;
non-thinking: temp 0.7 / top_p 0.8 / top_k 20 / presence_penalty 1.5).

## Choice 8 — what the prompt deliberately does NOT do

- No "you are being tested" framing, no examples with answers (in-context
  examples tempt copy-paste of the example's confidence/type).
- No camera-position speculation; the fixed-camera, static-background
  prior is stated once in the system prompt and used only for motion
  reasoning.
- No partial-credit language — booleans per category keep aggregation
  simple; nuance goes in summary/evidence/alternatives.
- Event set is exactly fall / fight / self_harm (per project scope); if a
  class is added, add an EVENT_DEFINITIONS entry + schema keys — the rest
  is mechanical.

## Maintenance loop

1. Run `detect.py` on labeled clips (or a false-positive repro case).
2. For each wrong window, read `reasoning` + `possible_alternative_
   explanations` in `results/<stem>_report.json` to see *why*.
3. Extend TELLTALE SIGNS (misses) or DO NOT COUNT (false positives) for
   that event in `src/qwen35/prompts.py`.
4. Re-run the same clips (frames are cached under `results/_frames/`).
