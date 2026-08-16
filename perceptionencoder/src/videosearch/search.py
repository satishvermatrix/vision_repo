"""Load an index and run natural-language queries against it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from videosearch.pe_model import PEModel


@dataclass
class SearchResult:
    video: str
    start: float
    end: float
    score: float


@dataclass
class Index:
    embeddings: np.ndarray
    chunks: list[dict]
    config: dict


def load_index(index_dir: str | Path) -> Index:
    index_dir = Path(index_dir)
    emb = np.load(index_dir / "embeddings.npy")
    with open(index_dir / "chunks.json") as f:
        chunks = json.load(f)
    with open(index_dir / "config.json") as f:
        config = json.load(f)
    return Index(embeddings=emb, chunks=chunks, config=config)


def search(query: str, index: Index, model: PEModel, k: int = 5) -> list[SearchResult]:
    q = model.embed_texts([query])[0]  # already L2-normalized
    scores = index.embeddings @ q  # cosine similarity (both normalized)
    k = min(k, scores.shape[0])
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    results: list[SearchResult] = []
    for i in top:
        meta = index.chunks[int(i)]
        results.append(
            SearchResult(
                video=meta["video"],
                start=float(meta["start"]),
                end=float(meta["end"]),
                score=float(scores[int(i)]),
            )
        )
    return results
