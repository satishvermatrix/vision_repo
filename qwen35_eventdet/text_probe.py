#!/usr/bin/env python3
"""Text-only probe for Qwen3.5-35B-A3B (qwen/qwen3.5-35b-a3b) via OpenRouter.

Phase 1 of the qwen35 project: before any video inference, send a small fixed
set of event-detection-flavored TEXT prompts under different sampling /
thinking parameters and report how answers, latency, and token usage change.

Validates:
  - OpenRouter access + model slug
  - parameter pass-through (reasoning on/off, seed, structured outputs)
  - JSON verdict behavior planned for the Phase 2 video detector

Usage:
  uv run env OPENROUTER_API_KEY=sk-... python text_probe.py
  uv run python text_probe.py --dry-run        # plan only, no key, no calls
  uv run python text_probe.py --runs A,B,E    # subset of runs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

MODEL_DEFAULT = "qwen/qwen3.5-35b-a3b"
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"
MAX_TOKENS_DEFAULT = 8192
TIMEOUT_S_DEFAULT = 240.0
MAX_RETRIES = 2

# Canonical OpenRouter pricing for qwen/qwen3.5-35b-a3b (USD per token); used
# only to *estimate* cost when OpenRouter does not report usage.cost itself.
PRICE_PROMPT = 0.0000001625
PRICE_COMPLETION = 0.0000013

# ---------------------------------------------------------------------------
# Prompts (event-detection flavored, no video)
# ---------------------------------------------------------------------------

VERDICT_SYSTEM = """You are a precise safety analysis engine for surveillance incidents.

You will be given a text description of a scene with a timeline. Analyze it ONLY for these event categories:
- fall: a person falling down (accidental fall / collapse)
- fight: two or more people fighting (physical altercation / violent attack)
- aggression: one-sided hostile or threatening conduct short of mutual combat (shoving, chasing, brandishing objects, intimidation, attempted assault)
- self_harm: a person attempting self-harm (deliberate self-injury)

Rules:
1. Judge strictly by what is described. Do not speculate beyond the given facts.
2. A category is true ONLY if the event is clearly happening or just happened in the description.
3. Consider benign lookalikes (stumbling recovered, play-wrestling, helping someone up, first aid). These are NOT events.
4. "confidence" is calibrated: 90-100 unmistakable; 60-89 likely; 30-59 uncertain; 0-29 unlikely.
5. If nothing significant is happening, set every category to false and detected_event_type to "none".

Respond with ONLY a JSON object (no markdown, no extra text) matching exactly:
{
  "event_present": {"fall": false, "fight": false, "aggression": false, "self_harm": false},
  "detected_event_type": "fall | fight | aggression | self_harm | none",
  "confidence": 0,
  "time_range_in_clip": "e.g. 1.5s-3.0s, or n/a for text input",
  "summary": "2-3 sentence description of what happens",
  "visual_evidence": "bullet list of concrete observed cues",
  "possible_alternative_explanations": "what benign activity could look like this"
}"""

VERDICT_USER = """Scene description (from a stairwell camera, timeline 0s-6s):
- 0-1s: a man carrying a cardboard box walks toward the top of the stairs.
- 1-2s: he misjudges the top step, the box tips forward, he grabs the handrail but loses his balance.
- 2-4s: he tumbles down four steps and lands seated at the landing, the box spilled beside him.
- 4-6s: he stays seated for a moment rubbing his ankle; a second person arrives, helps him stand up; both smile briefly.

Analyze this incident for the event categories. Answer with the JSON object only."""

CUES_PROMPT = """You review surveillance footage. In practice, how do you tell a real fight apart from play-wrestling or martial-arts sparring? List the most reliable visual cues. Answer with at most 6 short bullets, one line each, no extra text."""

INSTRUCTION_PROMPT = """Choose the single best-matching event name for: "a person deliberately and repeatedly striking their own head with their fists". Pick exactly one of: fall, fight, aggression, self_harm. Output only that word in lowercase, no punctuation, no explanation."""

PROMPTS: dict[str, list[dict]] = {
    "verdict": [
        {"role": "system", "content": VERDICT_SYSTEM},
        {"role": "user", "content": VERDICT_USER},
    ],
    "cues": [{"role": "user", "content": CUES_PROMPT}],
    "instruction": [{"role": "user", "content": INSTRUCTION_PROMPT}],
}

VERDICT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "event_present": {
            "type": "object",
            "properties": {
                "fall": {"type": "boolean"},
                "fight": {"type": "boolean"},
                "aggression": {"type": "boolean"},
                "self_harm": {"type": "boolean"},
            },
            "required": ["fall", "fight", "aggression", "self_harm"],
            "additionalProperties": False,
        },
        "detected_event_type": {
            "type": "string",
            "enum": ["fall", "fight", "aggression", "self_harm", "none"],
        },
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "time_range_in_clip": {"type": "string"},
        "summary": {"type": "string"},
        "visual_evidence": {"type": "string"},
        "possible_alternative_explanations": {"type": "string"},
    },
    "required": [
        "event_present",
        "detected_event_type",
        "confidence",
        "time_range_in_clip",
        "summary",
        "visual_evidence",
        "possible_alternative_explanations",
    ],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Run matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    prompt_id: str
    desc: str
    thinking: bool | None  # None = provider default (don't send the param)
    temperature: float
    top_p: float
    top_k: int = 20
    presence_penalty: float = 1.5
    seed: int | None = None
    structured: bool = False
    repeats: int = 1


RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        "A", "verdict",
        "thinking ON, Qwen thinking preset (temp=1.0)",
        thinking=True, temperature=1.0, top_p=0.95,
    ),
    RunSpec(
        "B", "verdict",
        "thinking OFF, Qwen non-thinking preset (temp=0.7)",
        thinking=False, temperature=0.7, top_p=0.8,
    ),
    RunSpec(
        "C", "verdict",
        "thinking OFF, low temp=0.2 (determinism check)",
        thinking=False, temperature=0.2, top_p=0.8,
    ),
    RunSpec(
        "D", "verdict",
        "thinking ON + seed=42 (reproducibility; run twice)",
        thinking=True, temperature=1.0, top_p=0.95, seed=42, repeats=2,
    ),
    RunSpec(
        "E", "verdict",
        "thinking OFF + response_format json_schema (structured outputs)",
        thinking=False, temperature=0.7, top_p=0.8, structured=True,
    ),
    RunSpec(
        "F", "cues",
        "thinking ON, domain-knowledge answer",
        thinking=True, temperature=1.0, top_p=0.95,
    ),
    RunSpec(
        "G", "instruction",
        "thinking OFF, strict-format compliance",
        thinking=False, temperature=0.7, top_p=0.8,
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extra(obj, key):
    """Read `key` from a pydantic model via attribute or model_extra."""
    if obj is None:
        return None
    v = getattr(obj, key, None)
    if v is None:
        extra = getattr(obj, "model_extra", None)
        if extra:
            v = extra.get(key)
    return v


def jsonable(v):
    """Convert pydantic/other objects into JSON-safe structures."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    md = getattr(v, "model_dump", None)
    if callable(md):
        try:
            return jsonable(md())
        except Exception:
            pass
    return repr(v)


def preview(s: str | None, n: int = 400) -> str:
    if not s:
        return "-"
    flat = " ".join(s.split())
    return flat[:n] + ("..." if len(flat) > n else "")


def estimate_cost(usage: dict) -> tuple[float, bool]:
    """(cost_usd, is_estimate) — prefers OpenRouter's usage.cost."""
    if usage.get("cost_usd") is not None:
        try:
            return float(usage["cost_usd"]), False
        except (TypeError, ValueError):
            pass
    pt = usage.get("prompt_tokens") or 0
    ct = usage.get("completion_tokens") or 0
    return pt * PRICE_PROMPT + ct * PRICE_COMPLETION, True


def select_runs(spec: str) -> list[RunSpec]:
    if not spec.strip():
        return list(RUNS)
    ids = {x.strip().upper() for x in spec.split(",") if x.strip()}
    known = {r.run_id for r in RUNS}
    bad = sorted(ids - known)
    if bad:
        sys.exit(f"error: unknown run ids {bad}; valid: {sorted(known)}")
    return [r for r in RUNS if r.run_id in ids]


def split_inline_think(content: str | None) -> tuple[str | None, str | None]:
    """Some providers inline thinking as <think>...</think> inside content."""
    if not content or "<think>" not in content:
        return content, None
    pre, _, post = content.partition("</think>")
    inline = pre.replace("<think>", "").strip() or None
    return (post.strip() or None), inline


def call_model(client, model: str, run: RunSpec, max_tokens: int, timeout: float):
    kwargs = dict(
        model=model,
        messages=PROMPTS[run.prompt_id],
        max_tokens=max_tokens,
        temperature=run.temperature,
        top_p=run.top_p,
        presence_penalty=run.presence_penalty,
        timeout=timeout,
    )
    extra: dict = {"top_k": run.top_k}
    if run.thinking is not None:
        extra["reasoning"] = {"enabled": run.thinking}
    if run.structured:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "event_verdict",
                "strict": True,
                "schema": VERDICT_JSON_SCHEMA,
            },
        }
    if run.seed is not None:
        kwargs["seed"] = run.seed
    kwargs["extra_body"] = extra
    t0 = time.perf_counter()
    resp = client.chat.completions.create(**kwargs)
    return resp, time.perf_counter() - t0


def capture(resp, latency: float) -> dict:
    choice = resp.choices[0]
    msg = choice.message
    content, inline_think = split_inline_think(msg.content)
    reasoning = _extra(msg, "reasoning") or _extra(msg, "reasoning_content")
    usage = resp.usage
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    return {
        "content": content,
        "inline_think": inline_think,
        "reasoning": reasoning,
        "reasoning_details": jsonable(_extra(msg, "reasoning_details")),
        "finish_reason": choice.finish_reason,
        "provider": _extra(resp, "provider"),
        "served_model": resp.model,
        "response_id": resp.id,
        "latency_s": round(latency, 2),
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "reasoning_tokens": getattr(details, "reasoning_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "cost_usd": _extra(usage, "cost"),
        },
    }


def print_run_block(label: str, run: RunSpec, rec: dict | None, err: str | None):
    print(f"--- Run {label}: {run.prompt_id} | {run.desc} ---")
    if err:
        print(f"  ERROR: {err}")
        print()
        return
    u = rec["usage"]
    cost, est = estimate_cost(u)
    print(f"  provider: {rec.get('provider') or '-'} | served: {rec.get('served_model') or '-'} | finish: {rec.get('finish_reason')}")
    print(
        f"  latency: {rec['latency_s']}s | prompt {u.get('prompt_tokens')} tok"
        f" | reasoning {u.get('reasoning_tokens')} tok | completion {u.get('completion_tokens')} tok"
        f" | cost {'~' if est else ''}${cost:.5f}"
    )
    if rec.get("reasoning"):
        print(f"  reasoning: {preview(rec['reasoning'], 240)}")
    if rec.get("inline_think"):
        print(f"  inline <think>: {preview(rec['inline_think'], 240)}")
    print(f"  answer: {preview(rec.get('content'), 400)}")
    print()


def print_table(calls: list[dict]):
    head = (
        f"{'run':<5}{'prompt':<12}{'think':<6}{'temp':<5}{'seed':<5}{'fmt':<5}"
        f"{'latency':>8}{'p_tok':>7}{'r_tok':>7}{'c_tok':>7}{'cost':>10}  answer"
    )
    print(head)
    print("-" * len(head))
    for c in calls:
        run: RunSpec = c["run"]
        if c.get("error"):
            row = (
                f"{c['label']:<5}{run.prompt_id:<12}"
                f"{'on' if run.thinking else 'off' if run.thinking is False else '-':<6}"
                f"{run.temperature:<5}{'-' if run.seed is None else run.seed:<5}"
                f"{'-' if not run.structured else 'json':<5}"
                f"{'-':>8}{'-':>7}{'-':>7}{'-':>7}{'-':>10}  ERROR: {preview(c['error'], 60)}"
            )
        else:
            u = c["capture"]["usage"]
            cost, est = estimate_cost(u)
            row = (
                f"{c['label']:<5}{run.prompt_id:<12}"
                f"{'on' if run.thinking else 'off' if run.thinking is False else '-':<6}"
                f"{run.temperature:<5}{'-' if run.seed is None else run.seed:<5}"
                f"{'-' if not run.structured else 'json':<5}"
                f"{c['capture']['latency_s']:>7}s"
                f"{u.get('prompt_tokens') or 0:>7}"
                f"{u.get('reasoning_tokens') or 0:>7}"
                f"{u.get('completion_tokens') or 0:>7}"
                f"{'~' if est else ''}${cost:.5f}".rjust(10)
                + "  " + preview(c["capture"].get("content"), 60)
            )
        print(row)
    print()


def dry_run(runs: list[RunSpec], model: str, max_tokens: int, timeout: float):
    n_calls = sum(r.repeats for r in runs)
    print(f"dry run: model={model} max_tokens={max_tokens} timeout={timeout}s")
    print(f"would make {n_calls} API call(s) across {len(runs)} run(s)\n")
    for run in runs:
        p = asdict(run)
        print(f"--- Run {run.run_id}: {run.desc} ---")
        print(f"  params: {json.dumps(p, default=str)}")
        for m in PROMPTS[run.prompt_id]:
            print(f"  {m['role']} prompt: {preview(m['content'], 200)}")
        print()
    print("prompts are fixed; runs A-E exercise the verdict prompt, F/G the two")
    print("auxiliary prompts. Re-run without --dry-run (and with"
          f" {API_KEY_ENV}) to execute.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Text-only parameter probe for qwen/qwen3.5-35b-a3b via OpenRouter"
    )
    ap.add_argument("--model", default=MODEL_DEFAULT, help=f"default {MODEL_DEFAULT}")
    ap.add_argument("--runs", default="", help="comma-separated subset of A-G (default all)")
    ap.add_argument("--out-dir", default="results/text_probe")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS_DEFAULT)
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S_DEFAULT)
    ap.add_argument("--dry-run", action="store_true", help="print plan only, no calls")
    args = ap.parse_args()

    runs = select_runs(args.runs)

    if args.dry_run:
        dry_run(runs, args.model, args.max_tokens, args.timeout)
        return

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        sys.exit(
            f"error: {API_KEY_ENV} is not set. Use:\n"
            f"  uv run env {API_KEY_ENV}=sk-or-... python text_probe.py\n"
            f"(or --dry-run to inspect the plan without calls)"
        )

    from openai import OpenAI

    client = OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        timeout=args.timeout,
        max_retries=MAX_RETRIES,
    )

    calls: list[dict] = []
    t_start = time.perf_counter()
    for run in runs:
        for rep in range(1, run.repeats + 1):
            label = run.run_id if run.repeats == 1 else f"{run.run_id}.{rep}"
            rec, err = None, None
            try:
                resp, latency = call_model(client, args.model, run, args.max_tokens, args.timeout)
                rec = capture(resp, latency)
            except Exception as e:  # keep sweeping on failures
                err = f"{type(e).__name__}: {e}"
            calls.append({"label": label, "run": run, "capture": rec, "error": err})
            print_run_block(label, run, rec, err)

    print_table(calls)

    # seed reproducibility: compare repeated runs of the same seeded spec
    for run in runs:
        if run.seed is None or run.repeats < 2:
            continue
        reps = [c for c in calls if c["run"].run_id == run.run_id and c.get("capture")]
        if len(reps) >= 2:
            contents = [c["capture"].get("content") for c in reps]
            same = all(c == contents[0] for c in contents[1:])
            print(f"seed reproducibility (run {run.run_id}, seed={run.seed}): "
                  f"answers {'IDENTICAL' if same else 'DIFFER'} across {len(reps)} calls")
    print()

    ok = [c for c in calls if not c.get("error")]
    total_cost = sum(estimate_cost(c["capture"]["usage"])[0] for c in ok)
    print(f"total: {len(ok)}/{len(calls)} calls ok | wall {time.perf_counter() - t_start:.1f}s | "
          f"cost {'~' if any(estimate_cost(c['capture']['usage'])[1] for c in ok) else ''}${total_cost:.5f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    try:
        import openai
        openai_version = openai.__version__
    except Exception:
        openai_version = None
    payload = {
        "meta": {
            "model": args.model,
            "base_url": BASE_URL,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "max_tokens": args.max_tokens,
            "timeout_s": args.timeout,
            "openai_client": openai_version,
            "argv": sys.argv[1:],
        },
        "calls": [
            {
                "label": c["label"],
                "run": asdict(c["run"]),
                "capture": jsonable(c["capture"]),
                "error": c["error"],
            }
            for c in calls
        ],
    }
    out_path.write_text(json.dumps(jsonable(payload), indent=2))
    print(f"full results saved to {out_path}")


if __name__ == "__main__":
    main()
