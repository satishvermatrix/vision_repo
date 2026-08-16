"""Qwen3-VL confirmer: decide if a clip window shows a target event."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

DEFAULT_VLM = "Qwen/Qwen3-VL-2B-Instruct"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / ".hf_home"))
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)


@dataclass
class VLMVerdict:
    present: bool
    confidence: float
    rationale: str
    raw_text: str


def _parse_json_object(text: str) -> dict:
    """Extract a JSON object from model output (tolerates markdown fences)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("expected JSON object")
    return data


class QwenVLConfirmer:
    """Load Qwen3-VL once; confirm events from a list of PIL frames."""

    def __init__(
        self,
        model_id: str = DEFAULT_VLM,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch = torch

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

    def confirm(
        self,
        frames: list[Image.Image],
        event_name: str,
        definition: str,
        pe_score: float | None = None,
    ) -> VLMVerdict:
        if not frames:
            return VLMVerdict(
                present=False,
                confidence=0.0,
                rationale="No frames could be decoded from the clip window.",
                raw_text="",
            )

        pe_note = (
            f"A retrieval model scored this window {pe_score:.3f} for '{event_name}'. "
            if pe_score is not None
            else ""
        )
        prompt = (
            f"You are verifying a video-event detector for safety/demo use.\n"
            f"Event label: {event_name}\n"
            f"Definition: {definition}\n"
            f"{pe_note}"
            "The images are ordered frames from a short video window.\n"
            "Decide whether THIS event is clearly present.\n"
            "Reply with ONLY a JSON object (no markdown) with keys:\n"
            '  "present": true or false,\n'
            '  "confidence": number from 0.0 to 1.0,\n'
            '  "rationale": short explanation (1-2 sentences).\n'
            "If unsure or only a weak proxy, set present=false and lower confidence."
        )

        content: list[dict] = [{"type": "image", "image": im} for im in frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=192, do_sample=False)

        in_len = inputs["input_ids"].shape[1]
        out_ids = generated[0, in_len:]
        raw = self.processor.batch_decode([out_ids], skip_special_tokens=True)[0].strip()

        try:
            data = _parse_json_object(raw)
            present = bool(data.get("present", False))
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            rationale = str(data.get("rationale", "")).strip() or raw
            return VLMVerdict(present=present, confidence=confidence, rationale=rationale, raw_text=raw)
        except Exception:
            # Fallback: keyword heuristic if JSON parse fails
            low = raw.lower()
            present = '"present": true' in low or '"present":true' in low
            return VLMVerdict(
                present=present,
                confidence=0.4 if present else 0.3,
                rationale=raw[:400] or "VLM returned unparseable output.",
                raw_text=raw,
            )

    def close(self) -> None:
        del self.model
        del self.processor
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
