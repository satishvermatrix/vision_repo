"""Event catalog for PE retrieval + VLM confirmation cascade."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventSpec:
    name: str
    # PE text queries (max / mean score across these)
    pe_prompts: tuple[str, ...]
    # Short definition shown to the VLM
    definition: str


EVENTS: dict[str, EventSpec] = {
    "fight": EventSpec(
        name="fight",
        pe_prompts=(
            "people fighting",
            "a violent fight between people",
            "two people punching each other",
            "a boxing match fight",
            "sword fighting combat",
        ),
        definition=(
            "Physical combat between people (boxing, wrestling, sword fight, brawl). "
            "NOT solo exercise on a punching bag, dance, or sports without interpersonal combat."
        ),
    ),
    "aggression": EventSpec(
        name="aggression",
        pe_prompts=(
            "aggressive confrontational behavior",
            "a person slapping someone",
            "an aggressive kick toward a person",
            "hostile physical aggression",
        ),
        definition=(
            "Hostile or aggressive physical action directed at a person "
            "(slap, kick toward someone, confrontation). "
            "NOT calm training, stretching, or non-hostile sports drills alone."
        ),
    ),
    "fall": EventSpec(
        name="fall",
        pe_prompts=(
            "a person falling down",
            "someone tumbling or falling",
            "a person losing balance and falling",
            "a body falling through the air",
        ),
        definition=(
            "A person falling, tumbling, or dropping through the air / to the ground. "
            "Includes somersaults, vaults, jumps that show falling motion. "
            "NOT someone standing still or walking normally."
        ),
    ),
    "crowd": EventSpec(
        name="crowd",
        pe_prompts=(
            "a dense crowd of people",
            "many people packed together in a crowd",
            "a large group of people in a public space",
            "crowd surfing over many people",
        ),
        definition=(
            "Many people gathered densely together (crowd, queue of many people, "
            "crowd surfing). NOT one or two isolated people."
        ),
    ),
    "self-harm": EventSpec(
        name="self-harm",
        pe_prompts=(
            "a person intentionally injuring themselves",
            "self-harm behavior",
        ),
        definition=(
            "Clear intentional self-harm. Kinetics rarely contains this; "
            "answer present=false unless unmistakable. Do not speculate."
        ),
    ),
}


def list_events() -> list[str]:
    return sorted(EVENTS.keys())


def get_event(name: str) -> EventSpec:
    key = name.strip().lower().replace("_", "-")
    if key not in EVENTS:
        raise KeyError(f"Unknown event {name!r}. Choose from: {', '.join(list_events())}")
    return EVENTS[key]
