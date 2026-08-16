"""Command-line interface: index videos and run NLQ search."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from videosearch.cascade import run_cascade
from videosearch.events import list_events
from videosearch.indexer import build_index
from videosearch.pe_model import DEFAULT_MODEL, PEModel
from videosearch.search import load_index, search
from videosearch.vlm_confirm import DEFAULT_VLM

DEFAULT_DATA = "/home/satishv/study/vision/dataset/kinetics-dataset/data"
DEFAULT_INDEX = "data/index"
DEFAULT_EVENTS_INDEX = "data/index_events"


def _cmd_index(args: argparse.Namespace) -> None:
    model = PEModel(model_name=args.model)
    build_index(
        video_dir=args.video_dir,
        out_dir=args.index_dir,
        model=model,
        chunk_sec=args.chunk_sec,
        overlap_sec=args.overlap_sec,
        frames_per_chunk=args.frames,
        limit=args.limit,
    )


def _export_clip(video: str, start: float, end: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-i", video, "-c", "copy", str(out_path),
    ]
    subprocess.run(cmd, check=False)


def _cmd_search(args: argparse.Namespace) -> None:
    index = load_index(args.index_dir)
    model = PEModel(model_name=index.config.get("model_name", DEFAULT_MODEL))
    results = search(args.query, index=index, model=model, k=args.k)

    print(f'\nQuery: "{args.query}"\n')
    for rank, r in enumerate(results, 1):
        name = Path(r.video).name
        print(f"{rank:2d}. score={r.score:.4f}  [{r.start:6.2f}-{r.end:6.2f}s]  {name}")

    if args.export_clips:
        out_dir = Path(args.export_clips)
        for rank, r in enumerate(results, 1):
            out = out_dir / f"{rank:02d}_{Path(r.video).stem}_{r.start:.0f}-{r.end:.0f}.mp4"
            _export_clip(r.video, r.start, r.end, out)
        print(f"\nExported {len(results)} clips to {out_dir}")


def _cmd_confirm(args: argparse.Namespace) -> None:
    """PE candidates → Qwen3-VL confirmation for an event label."""
    report = args.report
    if report is None:
        report = f"data/reports/{args.event}_confirm.json"

    hits = run_cascade(
        event_name=args.event,
        index_dir=args.index_dir,
        k=args.k,
        min_pe_score=args.min_pe_score,
        vlm_frames=args.vlm_frames,
        vlm_model=args.vlm_model,
        confirm_threshold=args.confirm_threshold,
        report_path=report,
    )

    if args.export_clips and hits:
        out_dir = Path(args.export_clips)
        for h in hits:
            tag = "ok" if h.confirmed else "fp"
            out = (
                out_dir
                / f"{h.rank:02d}_{tag}_pe{h.pe_score:.3f}_vlm{h.vlm_confidence:.2f}"
                f"_{Path(h.video).stem}_{h.start:.0f}-{h.end:.0f}.mp4"
            )
            _export_clip(h.video, h.start, h.end, out)
        print(f"Exported {len(hits)} clips to {out_dir}")

    n_ok = sum(1 for h in hits if h.confirmed)
    print(f"\nSummary: {n_ok}/{len(hits)} confirmed as '{args.event}'")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="videosearch",
        description="Natural-language video search with Meta's PE-Core encoder.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("index", help="Embed video chunks and build the index.")
    pi.add_argument("video_dir", nargs="?", default=DEFAULT_DATA, help="Directory of videos.")
    pi.add_argument("--index-dir", default=DEFAULT_INDEX)
    pi.add_argument("--model", default=DEFAULT_MODEL)
    pi.add_argument("--chunk-sec", type=float, default=2.0)
    pi.add_argument("--overlap-sec", type=float, default=0.0)
    pi.add_argument("--frames", type=int, default=8, help="Frames sampled per chunk.")
    pi.add_argument("--limit", type=int, default=None, help="Only index the first N videos.")
    pi.set_defaults(func=_cmd_index)

    ps = sub.add_parser("search", help="Run a natural-language query.")
    ps.add_argument("query", help="Natural-language query.")
    ps.add_argument("-k", type=int, default=5, help="Number of results.")
    ps.add_argument("--index-dir", default=DEFAULT_INDEX)
    ps.add_argument("--export-clips", default=None, help="Dir to write matching clips via ffmpeg.")
    ps.set_defaults(func=_cmd_search)

    pc = sub.add_parser(
        "confirm",
        help="PE→VLM cascade: retrieve event candidates, then confirm with Qwen3-VL.",
    )
    pc.add_argument(
        "event",
        choices=list_events(),
        help="Event to detect (fight, aggression, fall, crowd, self-harm).",
    )
    pc.add_argument("--index-dir", default=DEFAULT_EVENTS_INDEX)
    pc.add_argument("-k", type=int, default=5, help="PE candidates to send to VLM.")
    pc.add_argument("--min-pe-score", type=float, default=0.20)
    pc.add_argument("--confirm-threshold", type=float, default=0.55,
                    help="Min VLM confidence to mark CONFIRMED.")
    pc.add_argument("--vlm-model", default=DEFAULT_VLM)
    pc.add_argument("--vlm-frames", type=int, default=6)
    pc.add_argument("--report", default=None, help="JSON report path.")
    pc.add_argument("--export-clips", default=None,
                    help="Export candidate windows (ok_/fp_ prefix by verdict).")
    pc.set_defaults(func=_cmd_confirm)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
