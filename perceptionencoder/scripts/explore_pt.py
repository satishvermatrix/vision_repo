"""Explore the contents of a PE (or any PyTorch) .pt checkpoint.

Usage:
  uv run python scripts/explore_pt.py
  uv run python scripts/explore_pt.py /path/to/PE-Core-L14-336.pt
  uv run python scripts/explore_pt.py --prefix visual. --limit 30
  uv run python scripts/explore_pt.py --all
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch

_PROJECT = Path(__file__).resolve().parents[1]
_DEFAULT_HF = _PROJECT / ".hf_home" / "hub" / "models--facebook--PE-Core-L14-336"


def find_default_pt() -> Path | None:
    """Prefer the HF cache snapshot symlink; fall back to any *.pt under .hf_home."""
    if _DEFAULT_HF.exists():
        snaps = list((_DEFAULT_HF / "snapshots").glob("*/PE-Core-L14-336.pt"))
        if snaps:
            return snaps[0]
    hits = sorted((_PROJECT / ".hf_home").rglob("*.pt")) if (_PROJECT / ".hf_home").exists() else []
    return hits[0] if hits else None


def unwrap_state_dict(obj) -> tuple[dict, str]:
    """Return (state_dict, note) for common checkpoint wrappers."""
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    for key in ("state_dict", "weights", "model", "model_state_dict"):
        inner = obj.get(key)
        if isinstance(inner, dict) and inner:
            sample = next(iter(inner.values()))
            if torch.is_tensor(sample):
                return inner, f"unwrapped from top-level key '{key}'"

    # Heuristic: values look like tensors → already a state_dict
    sample_vals = list(obj.values())[:5]
    if obj and all(torch.is_tensor(v) for v in sample_vals):
        return obj, "top-level dict of tensors (state_dict)"
    return obj, "top-level dict (mixed / unknown layout)"


def prefix_of(name: str, depth: int = 1) -> str:
    parts = name.split(".")
    return ".".join(parts[:depth]) if parts else name


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def explore(path: Path, prefix: str | None, limit: int | None, show_all: bool) -> None:
    path = path.resolve()
    size = path.stat().st_size
    print(f"file: {path}")
    print(f"size on disk: {human_bytes(size)} ({size:,} bytes)")
    print()

    raw = torch.load(path, map_location="cpu", weights_only=True)
    print(f"torch.load type: {type(raw).__name__}")

    if isinstance(raw, dict):
        top_keys = list(raw.keys())
        print(f"top-level keys ({len(top_keys)}): {top_keys[:20]}"
              + (" ..." if len(top_keys) > 20 else ""))
        # Non-tensor metadata
        meta = {k: type(v).__name__ for k, v in raw.items() if not torch.is_tensor(v)}
        if meta and not all(k in ("state_dict", "weights", "model", "model_state_dict") for k in meta):
            interesting = {k: t for k, t in meta.items()
                           if k not in ("state_dict", "weights", "model", "model_state_dict")}
            if interesting:
                print(f"non-tensor top-level entries: {interesting}")

    sd, note = unwrap_state_dict(raw)
    print(f"state_dict: {note}")
    print(f"parameter tensors: {len(sd)}")
    print()

    # Filter
    items = [(k, v) for k, v in sd.items() if torch.is_tensor(v)]
    if prefix:
        items = [(k, v) for k, v in items if k.startswith(prefix)]
        print(f"filter: prefix={prefix!r} → {len(items)} tensors")
        print()

    # Totals
    total_params = sum(v.numel() for _, v in items)
    total_bytes = sum(v.numel() * v.element_size() for _, v in items)
    dtypes = Counter(str(v.dtype) for _, v in items)
    print(f"total parameters: {total_params:,} ({total_params / 1e9:.3f} B)")
    print(f"total tensor bytes (in-memory): {human_bytes(total_bytes)}")
    print(f"dtypes: {dict(dtypes)}")
    print()

    # Group by first / second path segment
    by1: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # count, params
    by2: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for k, v in items:
        p1, p2 = prefix_of(k, 1), prefix_of(k, 2)
        by1[p1][0] += 1
        by1[p1][1] += v.numel()
        by2[p2][0] += 1
        by2[p2][1] += v.numel()

    print("by top-level prefix:")
    for p, (n, params) in sorted(by1.items(), key=lambda x: -x[1][1]):
        print(f"  {p:30s}  tensors={n:4d}  params={params:>14,}  ({100 * params / max(total_params, 1):5.1f}%)")
    print()

    print("by two-level prefix (top 25 by params):")
    for p, (n, params) in sorted(by2.items(), key=lambda x: -x[1][1])[:25]:
        print(f"  {p:40s}  tensors={n:4d}  params={params:>14,}")
    print()

    # Per-tensor listing
    rows = items if show_all else items[: (limit or 40)]
    print(f"tensors ({'all' if show_all else f'first {len(rows)}'}):")
    print(f"  {'name':<70s} {'shape':<28s} {'dtype':<14s} {'params':>12s}")
    print("  " + "-" * 128)
    for k, v in rows:
        shape = str(tuple(v.shape))
        print(f"  {k:<70s} {shape:<28s} {str(v.dtype):<14s} {v.numel():>12,}")
    if not show_all and len(items) > len(rows):
        print(f"  ... ({len(items) - len(rows)} more; pass --all or --limit N)")


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect a PE / PyTorch .pt checkpoint.")
    p.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to .pt (default: PE-Core-L14-336.pt from .hf_home cache)",
    )
    p.add_argument("--prefix", default=None, help="Only show keys starting with this prefix")
    p.add_argument("--limit", type=int, default=40, help="Max tensors to list (ignored with --all)")
    p.add_argument("--all", action="store_true", help="List every tensor")
    args = p.parse_args()

    path = Path(args.path) if args.path else find_default_pt()
    if path is None or not path.exists():
        raise SystemExit(
            "No .pt found. Pass a path, or download PE-Core first "
            "(run videosearch once so .hf_home is populated)."
        )
    explore(path, prefix=args.prefix, limit=args.limit, show_all=args.all)


if __name__ == "__main__":
    # Avoid accidental HF network use if something imports pe_model later.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    main()
