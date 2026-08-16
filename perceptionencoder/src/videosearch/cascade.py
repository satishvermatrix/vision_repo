"""PE retrieval → VLM confirmation cascade for event demos."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from videosearch.events import EventSpec, get_event
from videosearch.frames import sample_window_frames
from videosearch.pe_model import DEFAULT_MODEL, PEModel
from videosearch.search import Index, SearchResult, load_index
from videosearch.vlm_confirm import DEFAULT_VLM, QwenVLConfirmer, VLMVerdict


@dataclass
class CascadeHit:
    rank: int
    video: str
    start: float
    end: float
    pe_score: float
    vlm_present: bool
    vlm_confidence: float
    vlm_rationale: str
    confirmed: bool


def pe_event_candidates(
    event: EventSpec,
    index: Index,
    model: PEModel,
    k: int = 8,
    min_pe_score: float = 0.20,
) -> list[SearchResult]:
    """Score chunks vs all PE prompts; keep unique top windows above threshold."""
    prompt_embs = model.embed_texts(list(event.pe_prompts))  # [P, D]
    # Max similarity across prompts for each chunk
    sims = index.embeddings @ prompt_embs.T  # [N, P]
    scores = sims.max(axis=1)

    order = np.argsort(-scores)
    results: list[SearchResult] = []
    seen: set[tuple[str, float, float]] = set()
    for i in order:
        score = float(scores[int(i)])
        if score < min_pe_score:
            break
        meta = index.chunks[int(i)]
        key = (meta["video"], float(meta["start"]), float(meta["end"]))
        if key in seen:
            continue
        seen.add(key)
        results.append(
            SearchResult(
                video=meta["video"],
                start=float(meta["start"]),
                end=float(meta["end"]),
                score=score,
            )
        )
        if len(results) >= k:
            break
    return results


def run_cascade(
    event_name: str,
    index_dir: str | Path,
    k: int = 5,
    min_pe_score: float = 0.20,
    vlm_frames: int = 6,
    vlm_model: str = DEFAULT_VLM,
    confirm_threshold: float = 0.55,
    report_path: str | Path | None = None,
) -> list[CascadeHit]:
    """
    1) PE retrieves top-k event candidates
    2) Free PE GPU memory
    3) Qwen3-VL confirms each candidate
    """
    event = get_event(event_name)
    index = load_index(index_dir)

    pe = PEModel(model_name=index.config.get("model_name", DEFAULT_MODEL))
    candidates = pe_event_candidates(
        event, index=index, model=pe, k=k, min_pe_score=min_pe_score
    )
    del pe
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if not candidates:
        print(f"No PE candidates for '{event.name}' above min_pe_score={min_pe_score}")
        return []

    print(f"\nEvent: {event.name}")
    print(f"PE candidates: {len(candidates)} (min_pe_score={min_pe_score})")
    print(f"Loading VLM: {vlm_model} ...")
    vlm = QwenVLConfirmer(model_id=vlm_model)

    hits: list[CascadeHit] = []
    try:
        for rank, cand in enumerate(candidates, 1):
            frames = sample_window_frames(
                cand.video, cand.start, cand.end, num_frames=vlm_frames
            )
            verdict: VLMVerdict = vlm.confirm(
                frames,
                event_name=event.name,
                definition=event.definition,
                pe_score=cand.score,
            )
            confirmed = bool(verdict.present and verdict.confidence >= confirm_threshold)
            hit = CascadeHit(
                rank=rank,
                video=cand.video,
                start=cand.start,
                end=cand.end,
                pe_score=cand.score,
                vlm_present=verdict.present,
                vlm_confidence=verdict.confidence,
                vlm_rationale=verdict.rationale,
                confirmed=confirmed,
            )
            hits.append(hit)

            flag = "CONFIRMED" if confirmed else "REJECTED"
            name = Path(cand.video).name
            print(
                f"{rank:2d}. [{flag}] pe={cand.score:.4f}  vlm_present={verdict.present} "
                f"vlm_conf={verdict.confidence:.2f}  [{cand.start:.2f}-{cand.end:.2f}s]  {name}"
            )
            print(f"    {verdict.rationale}")
    finally:
        vlm.close()

    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "event": event.name,
            "index_dir": str(index_dir),
            "vlm_model": vlm_model,
            "min_pe_score": min_pe_score,
            "confirm_threshold": confirm_threshold,
            "hits": [asdict(h) for h in hits],
            "num_confirmed": sum(1 for h in hits if h.confirmed),
        }
        path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote report: {path}")

    return hits
