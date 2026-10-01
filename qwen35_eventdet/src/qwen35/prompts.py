"""Prompt bank for the Qwen3.5-35B-A3B video event detector.

Design rationale (full write-up in docs/PROMPTS.md):

- One strict system prompt per call (cached prompt tokens are cheap), defining
  each event with three blocks:
    COUNTS            - what qualifies, including boundary cases
    TELLTALE SIGNS    - observable cross-frame evidence (motion, posture,
                        aftermath), the stuff a frame-sampled clip actually
                        shows at 8 fps
    DO NOT COUNT      - benign lookalikes that must be refuted explicitly;
                        this list is the main false-positive lever
  (mirrors the distractor-bank lesson from ../qwen: lookalike refutation is
  what kills false positives, not terser event definitions).
- Qwen3.5 is natively multimodal with text-timestamp alignment: frames are
  interleaved with '<T.TTT seconds>' text labels, matching the model's own
  video pre-training format, so temporal grounding and time_range_in_clip
  answers work without any positional hacks.
- Cross-frame reasoning is instructed explicitly: at 8 fps a strike or a fall
  can happen between two frames; the model must compare consecutive frames
  and use aftermath evidence (person sprawled, being helped up, injuries).
- Confidence is calibrated on a fixed verbal scale so numbers are comparable
  across calls, and the answer is a single strict JSON object; with thinking
  off, OpenRouter structured outputs (json_schema) enforce it at the API
  level; with thinking on, the prompt alone is authoritative (Phase 1
  verified clean JSON in both modes).
- Thinking mode: Qwen3.5 has no /think soft switch; thinking is the model
  default and reasoning returns separately in message.reasoning, so it never
  pollutes the JSON answer.
"""

from __future__ import annotations

import json

from .config import EVENTS

EVENT_LABELS = {
    "fall": "a person falling down (accidental fall or collapse)",
    "fight": "people fighting (physical altercation / violent attack)",
    "self_harm": "a person attempting self-harm (deliberate self-injury)",
}

EVENT_DEFINITIONS = {
    "fall": """### fall - a person falling down (accidental fall or collapse)
COUNTS: a person loses balance or control and ends up on the floor or ground involuntarily. This includes slipping, tripping, stumbling to the ground; tumbling down stairs; collapsing (medical); falling from furniture or a ladder; and being knocked to the ground by another person.
TELLTALE SIGNS ACROSS FRAMES: abrupt downward acceleration of the body from upright to floor level; failed balance recovery (arms flailing, stumbling recovery steps that fail); impact with the floor (body horizontal or slumped at floor level); aftermath such as lying still, slow or failed attempts to get up, clutching a body part, or someone rushing to help. A slow collapse still counts: the body slides or slumps off a chair or sags to the floor with NO bracing - arms hanging or limp, no hands pushing off the seat, no feet planted first, no glance at the floor, landing in an unarranged, awkward heap.
DO NOT COUNT: sitting down on a chair, bed, or sofa deliberately (controlled descent, arms used to lower, lands seated); deliberately lowering oneself from a chair to the floor WITH control - hands bracing the descent, feet placed first, eyes on the landing spot, body lowered in stages; lying down on purpose to rest or sleep; yoga, stretching, sit-ups, or deliberate floor exercise; squats and crouching; deliberate gymnastic or sports rolls; children rolling or tumbling playfully while laughing; a person already on the floor when the clip starts (unless clearly struggling or injured); objects falling with no person involved.""",
    "fight": """### fight - people fighting (physical altercation or violent attack)
COUNTS: two or more people exchanging physical violence - punches, slaps, kicks, or headbutts directed at another person; violent grabbing, dragging, hair-pulling, or choking; one person beating or assaulting a victim who is not fighting back; violently tackling someone to the ground; attacking another person with an object.
TELLTALE SIGNS ACROSS FRAMES: a limb swinging fast toward the other person's head or torso; the target flinching, blocking, dodging, or retreating; aggressive pursuit of someone trying to escape; clothing grabs with violent jerking; a person going to the ground from a blow; onlookers scattering or intervening to separate people; two bodies in rapid simultaneous struggle.
DO NOT COUNT: martial-arts or boxing training (uniforms, gloves, mats, coaches, controlled rehearsed exchanges, bowing or fist-bumping after); contact sports with a ball, referee, or team kit; play-wrestling and roughhousing (relaxed faces, smiling, no real force, participants keep returning willingly); dancing; hugging or affectionate contact; incidental bumping or pushing past in a crowd; helping an injured or agitated person up; staged or choreographed performance fights (film-set cues, repeated identical takes).""",
    "self_harm": """### self_harm - a person attempting self-harm (deliberate self-injury)
COUNTS: a person deliberately injuring their own body - repeatedly striking their own head, face, or body with fists or objects; hitting their head against a wall or the floor; cutting or stabbing their own skin with a blade or sharp object; burning their own skin.
TELLTALE SIGNS ACROSS FRAMES: blows or sharp-object motions directed at the person's OWN body, repeatedly and with real force; visible distress (crying, agitation) paired with self-directed violence; marks or injuries appearing on their own body across frames; escalating or repeated self-directed contact between frames.
DO NOT COUNT: applying a bandage or first aid to a wound (careful, protective gestures); medical treatment by a professional; shaving or grooming; cooking or normal kitchen knife use on food; tattoo or piercing work in a studio; exercising with equipment; hitting or kicking an object in frustration (punching a wall is NOT self-harm unless the person's own body is the target); scratching an itch, biting nails, or minor skin picking.""",
}

ANSWER_SCHEMA_EXAMPLE = {
    "event_present": {"fall": False, "fight": False, "self_harm": False},
    "detected_event_type": "fall | fight | self_harm | none",
    "confidence": 0,
    "time_range_in_clip": "e.g. 1.250s-3.500s, or n/a",
    "summary": "2-3 sentences describing what happens in the clip",
    "visual_evidence": [
        "concrete observed cue, with timestamps where possible"
    ],
    "possible_alternative_explanations": "which benign activity could look like this, or none",
}

# Strict JSON schema used with OpenRouter structured outputs (thinking off).
VERDICT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "event_present": {
            "type": "object",
            "properties": {
                "fall": {"type": "boolean"},
                "fight": {"type": "boolean"},
                "self_harm": {"type": "boolean"},
            },
            "required": ["fall", "fight", "self_harm"],
            "additionalProperties": False,
        },
        "detected_event_type": {
            "type": "string",
            "enum": ["fall", "fight", "self_harm", "none"],
        },
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "time_range_in_clip": {"type": "string"},
        "summary": {"type": "string"},
        "visual_evidence": {"type": "array", "items": {"type": "string"}},
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

SYSTEM_PROMPT_TEMPLATE = """You are a precise video safety analysis engine that reviews short surveillance clips frame by frame.

You will be shown a sequence of frames sampled at {fps:g} frames per second from one surveillance video window (fixed camera, mostly static background). Each frame is preceded by its timestamp label in the format <T.TTT seconds>, marking the frame's time within the window. Adjacent frames are {frame_gap_ms:.0f} ms apart. Motion appears as change between consecutive frames; a fast action (a strike, a fall) can happen between two frames.

## Event categories

Analyze the clip ONLY for these categories:

{event_definitions}

## Rules of judgment

1. Judge ONLY by what is visible in the frames. Do not speculate about anything outside the frames, and do not assume events that are not shown.
2. Track people and motion ACROSS frames: compare consecutive frames to detect strikes, falls, and other fast events. Use the timestamp labels to locate events in time.
3. An event may start before the first frame or continue after the last. If the moment itself is not visible but its immediate aftermath clearly is (a person sprawled on the floor, being helped up, visibly injured), still count it and say so in the summary.
4. For every category you mark true, list concrete visual evidence you actually observed, with timestamps where possible.
5. Before finalizing, actively consider the DO NOT COUNT lookalikes of that category. If the frames match a benign lookalike better than the event, mark the category false and explain in possible_alternative_explanations.
6. confidence is calibrated:
   90-100: unmistakable, clear visual evidence of the event
   60-89: likely - strong cues but some ambiguity, occlusion, or small frame size
   30-59: uncertain - weak, brief, or heavily obscured cues
   0-29: unlikely - little or no supporting evidence
7. If nothing significant is happening, set every category to false, detected_event_type to "none", and confidence to your certainty that nothing is happening.
8. time_range_in_clip uses the window-relative timestamp labels (e.g. "1.250s-3.500s"); use "n/a" if there is no event.

## Output format

Respond with ONLY one JSON object (no markdown fences, no commentary) with exactly these keys:
{schema}"""

USER_PROMPT_TEMPLATE = """Analyze this surveillance clip for the events: {event_names}.

The {n_frames} frames below are sequential, sampled at {fps:g} fps from a {dur:.1f}-second window. Each frame is preceded by its timestamp in seconds within this window. Judge strictly by what the frames show. Answer with the JSON object only."""

TRAILING_REMINDER = "Answer with the JSON object only."


def build_system_prompt(fps: float) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        fps=fps,
        frame_gap_ms=1000.0 / fps,
        event_definitions="\n\n".join(EVENT_DEFINITIONS[e] for e in EVENTS),
        schema=json.dumps(ANSWER_SCHEMA_EXAMPLE, indent=2),
    )


def build_user_prompt(events: list[str], n_frames: int, fps: float, dur: float) -> str:
    return USER_PROMPT_TEMPLATE.format(
        event_names=", ".join(EVENT_LABELS[e] for e in events),
        n_frames=n_frames,
        fps=fps,
        dur=dur,
    )
