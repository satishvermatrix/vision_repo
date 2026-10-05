"""Chunked video embedding with Qwen3-VL-Embedding.

Cuts a video into fixed-length chunks (default 5 s), embeds each chunk with a
configurable instruction (system prompt) and pixel / per-frame budgets, and saves
embeddings + metadata to .npz.

Run:
  ~/study/qwen3-embed-lab/.venv/bin/python tools/embed_video.py data/videos/bbb_1080.mp4 \
      --instruction "Represent the video chunk for retrieval." \
      --total-pixels 3932160 --fps 2

  # see what the model would see, without loading the model:
  ... --dry-run

Budget mechanics (notebook 03): per-frame pixel cap =
min(786,432 px, total_pixels * 2 / nframes), floor 131,072 px (128 tokens);
nframes ~= fps * chunk_seconds (capped by max_frames). 1 token = 1024 px.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from lab_helpers import (  # noqa: E402
    FRAME_MIN_TOKENS, FRAME_MAX_TOKENS, TOKEN_PIXELS,
    embed, inspect_entry, load_embedder, sim_matrix,
)

# The wrapper's own defaults (mirrored so --dry-run can run model-free).
WRAPPER_DEFAULTS = dict(
    default_instruction="Represent the user's input.", fps=1.0, max_frames=64,
    min_pixels=4 * 32 * 32, max_pixels=1800 * 32 * 32,
)
DEFAULT_TOTAL_PIXELS = int(8192 * TOKEN_PIXELS * 0.9)   # MODEL_SEQ_LEN default


def ffprobe(path) -> dict:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
        "-of", "json", str(path),
    ]).decode()
    info = json.loads(out)
    stream, fmt = info["streams"][0], info.get("format", {})
    num, den = (stream.get("avg_frame_rate") or "0/1").split("/")
    return {
        "width": stream["width"], "height": stream["height"],
        "source_fps": float(num) / float(den) if float(den) else None,
        "duration": float(fmt.get("duration") or stream.get("duration") or 0.0),
    }


def cut_chunks(src: Path, out_dir: Path, chunk_s: float, total_dur: float,
               min_chunk_s: float) -> list:
    n = int(np.ceil(total_dur / chunk_s - 1e-6))
    chunks = []
    for i in range(n):
        start = i * chunk_s
        length = min(chunk_s, total_dur - start)
        if length < min_chunk_s:
            print(f"[chunk {i:02d}] {start:7.2f}s +{length:.2f}s -> skipped "
                  f"(shorter than --min-chunk-seconds {min_chunk_s}s)")
            continue
        out = out_dir / f"chunk_{i:04d}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{length:.3f}",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", str(out),
        ], check=True)
        chunks.append({
            "index": i, "start": round(start, 3), "end": round(start + length, 3),
            "requested_s": round(length, 3),
            "actual_s": round(ffprobe(out)["duration"], 3),
            "file": str(out),
        })
    return chunks


def make_entry(chunk: dict, args) -> dict:
    entry = {"video": chunk["file"]}
    for key in ("instruction", "fps", "max_frames", "total_pixels", "min_pixels"):
        val = getattr(args, key)
        if val is not None:
            entry[key] = val
    return entry


def inspect_chunk(embedder, chunk: dict, args) -> dict:
    v = inspect_entry(embedder, make_entry(chunk, args))["videos"][0]
    return {k: v[k] for k in ("nframes", "width", "height", "frame_pixels",
                               "tokens_per_frame", "total_visual_tokens")}


def run(src: Path, args, chunk_dir: Path) -> Path | None:
    probe = ffprobe(src)
    print(f"[video] {src.name}: {probe['width']}x{probe['height']} @ "
          f"{probe['source_fps']:.2f} fps, {probe['duration']:.2f}s")
    print(f"[config] chunk={args.chunk_seconds}s model={args.model} "
          f"instruction={args.instruction or WRAPPER_DEFAULTS['default_instruction']!r}")
    print(f"[config] total_pixels={args.total_pixels or DEFAULT_TOTAL_PIXELS:,} "
          f"min_pixels={args.min_pixels or 'default'} "
          f"fps={args.fps or WRAPPER_DEFAULTS['fps']} "
          f"max_frames={args.max_frames or WRAPPER_DEFAULTS['max_frames']}")

    chunks = cut_chunks(src, chunk_dir, args.chunk_seconds, probe["duration"],
                        args.min_chunk_seconds)
    if not chunks:
        sys.exit("error: no chunks produced")

    # --dry-run: budgets only, model-free (build_conversation just duck-types these).
    embedder = (SimpleNamespace(**WRAPPER_DEFAULTS) if args.dry_run
                else load_embedder(args.model))

    print(f"\n[inspect] what the model sees per chunk"
          f"{' (dry run, no model)' if args.dry_run else ''}")
    for c in chunks:
        c["seen"] = inspect_chunk(embedder, c, args)
        s = c["seen"]
        print(f"  chunk {c['index']:02d} [{c['start']:7.2f}s .. {c['end']:7.2f}s] "
              f"actual={c['actual_s']:.2f}s -> {s['nframes']:3d} frames @ "
              f"{s['width']}x{s['height']} ({s['tokens_per_frame']:4d} tok/frame, "
              f"{s['total_visual_tokens']:6,d} tokens)")
    if args.dry_run:
        print("[dry-run] done (no embeddings written)")
        return None

    entries = [make_entry(c, args) for c in chunks]
    embs, t0 = [], time.perf_counter()
    for i in range(0, len(entries), args.batch_size):
        batch = entries[i:i + args.batch_size]
        e, dt = embed(embedder, batch, timed=True)
        embs.append(e.float().cpu())
        print(f"[embed] chunks {i:02d}..{i + len(batch) - 1:02d} "
              f"dim={tuple(e.shape)} {dt:.1f}s")
    embs = F.normalize(torch.cat(embs, dim=0), p=2, dim=-1)   # re-norm in fp32 (bf16 pass is coarse)
    embed_s = time.perf_counter() - t0
    sim = sim_matrix(embs, embs)

    out_path = args.out or LAB_ROOT / "data" / "embeddings" / f"{src.stem}_chunks.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "video": str(src), "probe": probe,
        "config": {k: v for k, v in vars(args).items() if k != "video"},
        "effective": {
            "instruction": args.instruction or WRAPPER_DEFAULTS["default_instruction"],
            "fps": args.fps or WRAPPER_DEFAULTS["fps"],
            "max_frames": args.max_frames or WRAPPER_DEFAULTS["max_frames"],
            "total_pixels": args.total_pixels or DEFAULT_TOTAL_PIXELS,
            "min_pixels": args.min_pixels or FRAME_MIN_TOKENS * TOKEN_PIXELS * 1.05,
            "per_frame_token_range": [FRAME_MIN_TOKENS, FRAME_MAX_TOKENS],
        },
        "chunks": chunks,
        "embedding_dim": int(embs.shape[1]),
        "embed_seconds": round(embed_s, 2),
    }
    np.savez_compressed(
        out_path,
        embeddings=embs.numpy().astype(np.float32),
        chunk_starts=np.array([c["start"] for c in chunks], dtype=np.float32),
        chunk_ends=np.array([c["end"] for c in chunks], dtype=np.float32),
        chunk_files=np.array([c["file"] for c in chunks]),
        sim=sim.astype(np.float32),
        meta=json.dumps(meta),
    )
    print(f"\n[saved] {out_path}  ({len(chunks)} chunks x {embs.shape[1]} dims, "
          f"{embed_s:.1f}s embed time)")

    if len(chunks) <= 30:
        print("[sim] chunk x chunk cosine similarity:")
        print("        " + "".join(f"{i:6d}" for i in range(len(chunks))))
        for i, row in enumerate(sim):
            print(f"  {i:3d}   " + "".join(f"{v:6.2f}" for v in row))
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Chunk a video (default 5 s) and embed each chunk with "
                    "Qwen3-VL-Embedding, with instruction / pixel / per-frame budget knobs.")
    ap.add_argument("video", type=Path, help="path to the video file")
    ap.add_argument("--chunk-seconds", type=float, default=5.0,
                    help="chunk length in seconds (default: 5)")
    ap.add_argument("--instruction", default=None,
                    help="system prompt for the chunks (default: embedder default "
                         f"'{WRAPPER_DEFAULTS['default_instruction']}')")
    ap.add_argument("--total-pixels", type=int, default=None,
                    help=f"total pixel budget per chunk (default: {DEFAULT_TOTAL_PIXELS:,} "
                         f"= MODEL_SEQ_LEN 8192 * 1024 * 0.9)")
    ap.add_argument("--min-pixels", type=int, default=None,
                    help="per-frame pixel floor (default: 128 tokens * 1024 px * 1.05)")
    ap.add_argument("--fps", type=float, default=None,
                    help="frames sampled per second within each chunk (default: 1.0)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="max frames per chunk (default: 64)")
    ap.add_argument("--model", choices=["2B", "8B"], default="2B")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="chunks embedded per forward pass (default: 4)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output .npz (default: data/embeddings/<stem>_chunks.npz)")
    ap.add_argument("--keep-chunks", action="store_true",
                    help="keep the cut chunks in data/chunks/<video-stem>/ instead of a temp dir")
    ap.add_argument("--min-chunk-seconds", type=float, default=0.25,
                    help="skip tail chunks shorter than this (default: 0.25)")
    ap.add_argument("--dry-run", action="store_true",
                    help="cut + inspect budgets only, no model load, no embeddings")
    args = ap.parse_args()

    src = args.video.expanduser().resolve()
    if not src.is_file():
        sys.exit(f"error: no such file: {src}")

    if args.keep_chunks:
        chunk_dir = LAB_ROOT / "data" / "chunks" / src.stem
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)
        chunk_dir.mkdir(parents=True)
        try:
            run(src, args, chunk_dir)
        finally:
            print(f"[chunks] kept in {chunk_dir}")
    else:
        with tempfile.TemporaryDirectory(prefix="embed_chunks_") as tmp:
            run(src, args, Path(tmp))


if __name__ == "__main__":
    main()
