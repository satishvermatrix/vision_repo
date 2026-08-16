"""Unit tests — no GPU / PE weights required."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from videosearch.indexer import find_videos
from videosearch.search import Index, SearchResult, load_index, search


KINETICS = Path("/home/satishv/study/vision/dataset/kinetics-dataset/data")


class FakePE:
    """Minimal stand-in for PEModel used by search()."""

    def __init__(self, query_vec: np.ndarray):
        self._q = query_vec.astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._q for _ in texts])


def test_find_videos_respects_limit(tmp_path: Path):
    for name in ("a.mp4", "b.mkv", "c.txt", "d.webm"):
        (tmp_path / name).write_bytes(b"x")
    found = find_videos(tmp_path)
    assert [p.name for p in found] == ["a.mp4", "b.mkv", "d.webm"]
    assert find_videos(tmp_path, limit=2) == found[:2]


def test_load_index_roundtrip(tmp_path: Path):
    emb = np.random.randn(4, 8).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    chunks = [
        {"video": f"/v/{i}.mp4", "start": float(i), "end": float(i + 2)}
        for i in range(4)
    ]
    cfg = {"model_name": "PE-Core-L14-336", "embed_dim": 8}
    np.save(tmp_path / "embeddings.npy", emb)
    (tmp_path / "chunks.json").write_text(json.dumps(chunks))
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    idx = load_index(tmp_path)
    assert idx.embeddings.shape == (4, 8)
    assert len(idx.chunks) == 4
    assert idx.config["model_name"] == "PE-Core-L14-336"


def test_search_ranks_by_cosine():
    # Three L2-normalized chunk vectors; query aligned with row 1.
    emb = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    chunks = [
        {"video": "a.mp4", "start": 0.0, "end": 2.0},
        {"video": "b.mp4", "start": 2.0, "end": 4.0},
        {"video": "c.mp4", "start": 4.0, "end": 6.0},
    ]
    index = Index(embeddings=emb, chunks=chunks, config={})
    model = FakePE(np.array([0.0, 1.0, 0.0], dtype=np.float32))

    results = search("anything", index=index, model=model, k=2)  # type: ignore[arg-type]
    assert len(results) == 2
    assert results[0].video == "b.mp4"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert results[1].score < results[0].score
    assert isinstance(results[0], SearchResult)


def test_search_k_clamped_to_index_size():
    emb = np.eye(2, dtype=np.float32)
    chunks = [
        {"video": "a.mp4", "start": 0.0, "end": 1.0},
        {"video": "b.mp4", "start": 1.0, "end": 2.0},
    ]
    index = Index(embeddings=emb, chunks=chunks, config={})
    model = FakePE(np.array([1.0, 0.0], dtype=np.float32))
    results = search("q", index=index, model=model, k=99)  # type: ignore[arg-type]
    assert len(results) == 2


@pytest.mark.skipif(not KINETICS.exists(), reason="Kinetics data not present")
def test_iter_chunks_on_real_video():
    from videosearch.chunker import iter_chunks

    videos = sorted(KINETICS.glob("*.mp4"))
    assert videos, "expected Kinetics mp4s"
    chunks = list(iter_chunks(videos[0], chunk_sec=2.0, frames_per_chunk=4))
    assert len(chunks) >= 1
    for ch in chunks:
        assert ch.end > ch.start
        assert len(ch.frames) > 0
        assert isinstance(ch.frames[0], Image.Image)
        # No tiny trailing slivers under 0.5s for 2s chunks.
        assert (ch.end - ch.start) >= 0.5 - 1e-6


def test_cli_parser_has_index_and_search():
    from videosearch.cli import build_parser

    p = build_parser()
    args = p.parse_args(["index", "--limit", "3"])
    assert args.command == "index"
    assert args.limit == 3
    args = p.parse_args(["search", "hello", "-k", "3"])
    assert args.command == "search"
    assert args.query == "hello"
    assert args.k == 3
