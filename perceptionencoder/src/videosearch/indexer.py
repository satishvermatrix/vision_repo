"""Build and persist an embedding index over video chunks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from videosearch.chunker import iter_chunks
from videosearch.pe_model import PEModel

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"}


@dataclass
class ChunkMeta:
    video: str
    start: float
    end: float


@dataclass
class IndexConfig:
    model_name: str
    chunk_sec: float
    overlap_sec: float
    frames_per_chunk: int
    embed_dim: int


def find_videos(video_dir: str | Path, limit: int | None = None) -> list[Path]:
    video_dir = Path(video_dir)
    files = sorted(p for p in video_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if limit is not None:
        files = files[:limit]
    return files


def build_index(
    video_dir: str | Path,
    out_dir: str | Path,
    model: PEModel,
    chunk_sec: float = 2.0,
    overlap_sec: float = 0.0,
    frames_per_chunk: int = 8,
    limit: int | None = None,
) -> IndexConfig:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(video_dir, limit=limit)
    if not videos:
        raise SystemExit(f"No video files found under {video_dir}")

    embeddings: list[np.ndarray] = []
    metas: list[ChunkMeta] = []

    for video in tqdm(videos, desc="indexing videos", unit="vid"):
        for chunk in iter_chunks(
            video,
            chunk_sec=chunk_sec,
            overlap_sec=overlap_sec,
            frames_per_chunk=frames_per_chunk,
        ):
            vec = model.embed_chunk(chunk.frames)
            embeddings.append(vec)
            metas.append(ChunkMeta(video=chunk.video, start=chunk.start, end=chunk.end))

    if not embeddings:
        raise SystemExit("No chunks were produced from the input videos.")

    emb = np.stack(embeddings).astype(np.float32)
    np.save(out_dir / "embeddings.npy", emb)

    with open(out_dir / "chunks.json", "w") as f:
        json.dump([asdict(m) for m in metas], f)

    cfg = IndexConfig(
        model_name=model.model_name,
        chunk_sec=chunk_sec,
        overlap_sec=overlap_sec,
        frames_per_chunk=frames_per_chunk,
        embed_dim=model.embed_dim,
    )
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"Indexed {len(videos)} videos -> {emb.shape[0]} chunks, dim={emb.shape[1]}")
    print(f"Index written to {out_dir}")
    return cfg
