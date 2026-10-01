"""OpenRouter (OpenAI-compatible) client for Qwen3.5-35B-A3B verdicts.

Modes validated in Phase 1 (results/text_probe):
- thinking ON (model default): reasoning returns separately in
  message.reasoning, JSON comes from the prompt alone (Phase 1 run A:
  clean JSON, conf 95, ~20s);
- thinking OFF (reasoning {enabled: false}): ~3s, and structured outputs
  (response_format json_schema) are enforced by the API (Phase 1 run E).

Sampling presets follow the Qwen3.5 model card "Best Practices" section.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import config
from .frames import b64_data_url
from .prompts import (
    TRAILING_REMINDER,
    VERDICT_JSON_SCHEMA,
    build_system_prompt,
    build_user_prompt,
)


def extract_json(text: str) -> dict:
    """Best-effort JSON extraction from a model reply (ported from ../qwen)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        start = -1
    raise ValueError(f"No JSON object found in reply: {text[:300]!r}")


def _extra(obj, key):
    if obj is None:
        return None
    v = getattr(obj, key, None)
    if v is None:
        extra = getattr(obj, "model_extra", None)
        if extra:
            v = extra.get(key)
    return v


def _split_inline_think(content: str | None) -> tuple[str | None, str | None]:
    if not content or "<think>" not in content:
        return content, None
    pre, _, post = content.partition("</think>")
    inline = pre.replace("<think>", "").strip() or None
    return (post.strip() or None), inline


def jsonable(v):
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


def estimate_cost(usage: dict) -> tuple[float, bool]:
    if usage.get("cost_usd") is not None:
        try:
            return float(usage["cost_usd"]), False
        except (TypeError, ValueError):
            pass
    pt = usage.get("prompt_tokens") or 0
    ct = usage.get("completion_tokens") or 0
    return pt * config.PRICE_PROMPT + ct * config.PRICE_COMPLETION, True


def normalize(answer: dict, events: list[str]) -> dict:
    """Coerce a parsed answer into fixed fields for fall/fight/self_harm."""
    present = answer.get("event_present")
    if not isinstance(present, dict):
        present = {e: bool(present) for e in events}
    event_present = {e: bool(present.get(e, False)) for e in events}
    conf = answer.get("confidence")
    try:
        conf = max(0, min(100, round(float(conf))))
    except (TypeError, ValueError):
        conf = None
    evidence = answer.get("visual_evidence")
    if not isinstance(evidence, list):
        evidence = [str(evidence)] if evidence else []
    return {
        "event_present": event_present,
        "detected_event_type": str(answer.get("detected_event_type", "none")),
        "confidence": conf,
        "time_range_in_clip": str(answer.get("time_range_in_clip", "n/a")),
        "summary": str(answer.get("summary", "")),
        "visual_evidence": evidence,
        "possible_alternative_explanations": str(
            answer.get("possible_alternative_explanations", "")
        ),
    }


class QwenClient:
    """Verdict client: thinking/non-thinking presets + structured outputs."""

    def __init__(
        self,
        model: str = config.MODEL_DEFAULT,
        thinking: bool = True,
        structured: str = "auto",  # auto | always | never
        max_tokens: int = config.MAX_TOKENS,
        timeout: float = config.TIMEOUT_S,
        max_retries: int = config.MAX_RETRIES,
        api_key: str | None = None,
    ):
        key = api_key or config.api_key()
        if not key:
            raise RuntimeError(
                f"{config.API_KEY_ENV} is not set; use "
                f"'uv run env {config.API_KEY_ENV}=sk-or-... python detect.py ...'"
            )
        from openai import OpenAI

        self.model = model
        self.thinking = thinking
        self.structured = structured
        self.max_tokens = max_tokens
        preset = config.THINKING_PRESET if thinking else config.NON_THINKING_PRESET
        self.sampling = preset
        self.client = OpenAI(
            base_url=config.BASE_URL,
            api_key=key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _use_structured(self) -> bool:
        if self.structured == "always":
            return True
        if self.structured == "never":
            return False
        return not self.thinking  # auto: validated combo from Phase 1

    def _chat(self, system: str, user_content: list[dict]) -> dict:
        kwargs: dict = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            max_tokens=self.max_tokens,
            temperature=self.sampling["temperature"],
            top_p=self.sampling["top_p"],
            presence_penalty=self.sampling["presence_penalty"],
        )
        extra: dict = {"top_k": self.sampling["top_k"]}
        extra["reasoning"] = {"enabled": self.thinking}
        if self._use_structured():
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "event_verdict",
                    "strict": True,
                    "schema": VERDICT_JSON_SCHEMA,
                },
            }
        kwargs["extra_body"] = extra
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - t0

        choice = resp.choices[0]
        msg = choice.message
        content, inline_think = _split_inline_think(msg.content)
        usage = resp.usage
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        if not content:
            raise ValueError("model returned empty content")
        answer = extract_json(content)
        return {
            "answer": answer,
            "reasoning": _extra(msg, "reasoning") or _extra(msg, "reasoning_content"),
            "inline_think": inline_think,
            "finish_reason": choice.finish_reason,
            "provider": _extra(resp, "provider"),
            "served_model": resp.model,
            "latency_s": round(latency, 2),
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "reasoning_tokens": getattr(details, "reasoning_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "cost_usd": _extra(usage, "cost"),
            },
        }

    def analyze_frames(
        self,
        frame_parts: list[dict],
        events: list[str],
        fps: float,
        dur: float,
    ) -> dict:
        """Verdict on one window sent as timestamped frames."""
        system = build_system_prompt(fps)
        n_frames = sum(1 for p in frame_parts if p["type"] == "image_url")
        user_head = build_user_prompt(events, n_frames, fps, dur)
        user_content = [{"type": "text", "text": user_head}, *frame_parts,
                        {"type": "text", "text": TRAILING_REMINDER}]
        res = self._chat(system, user_content)
        res.update(normalize(res.pop("answer"), events))
        return res

    def analyze_video(
        self,
        clip_path: Path,
        events: list[str],
        fps: float,
        dur: float,
    ) -> dict:
        """Verdict on one window sent as a base64 video (server-side sampling)."""
        system = build_system_prompt(fps)
        user_head = build_user_prompt(events, int(round(dur * fps)), fps, dur)
        user_head = user_head.replace(
            "The frames below are sequential, sampled at "
            f"{fps:g} fps from a {dur:.1f}-second window.",
            f"The video below is a {dur:.1f}-second window; the provider "
            "samples its frames internally.",
        )
        user_content = [
            {"type": "video_url", "video_url": {"url": b64_data_url(clip_path, "video/mp4")}},
            {"type": "text", "text": user_head},
        ]
        res = self._chat(system, user_content)
        res.update(normalize(res.pop("answer"), events))
        return res
