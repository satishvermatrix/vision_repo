#!/usr/bin/env python3
"""Fetch ~1000 Kinetics-400 clips from event-relevant classes.

Reads train/val annotations, downloads CVDF tar parts as needed, and extracts
only matching mp4s into data/events/<class_slug>/.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
K400 = Path("/home/satishv/study/vision/dataset/kinetics-dataset")
ANN = K400 / "k400" / "annotations"
TARGZ_TRAIN = K400 / "k400_targz" / "train"
TARGZ_VAL = K400 / "k400_targz" / "val"
OUT = ROOT / "data" / "events"

RELEVANT = [
    "punching person (boxing)",
    "wrestling",
    "sword fighting",
    "slapping",
    "side kick",
    "arm wrestling",
    "punching bag",
    "surfing crowd",
    "waiting in line",
    "somersaulting",
    "vault",
    "bungee jumping",
    "ski jumping",
    "skydiving",
    "triple jump",
]

# Event taxonomy for the demo (proxy mapping)
EVENT_OF = {
    "punching person (boxing)": "fight",
    "wrestling": "fight",
    "sword fighting": "fight",
    "slapping": "aggression",
    "side kick": "aggression",
    "arm wrestling": "aggression",
    "punching bag": "aggression",
    "surfing crowd": "crowd",
    "waiting in line": "crowd",
    "somersaulting": "fall",
    "vault": "fall",
    "bungee jumping": "fall",
    "ski jumping": "fall",
    "skydiving": "fall",
    "triple jump": "fall",
}

TARGET_TOTAL = 1000
PER_CLASS_CAP = 80  # soft balance across classes


def slug(label: str) -> str:
    return (
        label.lower()
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "_")
        .replace("-", "_")
    )


def load_stem_to_label() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split in ("train.csv", "val.csv"):
        path = ANN / split
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                label = row["label"].strip().strip('"')
                if label not in RELEVANT:
                    continue
                yid = row["youtube_id"].strip().strip('"')
                start = int(float(row["time_start"]))
                end = int(float(row["time_end"]))
                stem = f"{yid}_{start:06d}_{end:06d}"
                mapping[stem] = label
    return mapping


def existing_extracted() -> dict[str, str]:
    """stem -> label for already extracted files."""
    have: dict[str, str] = {}
    if not OUT.exists():
        return have
    for p in OUT.rglob("*.mp4"):
        have[p.stem] = p.parent.name
    return have


def extract_from_tar(tar_path: Path, stem_to_label: dict[str, str], have: dict[str, str],
                     class_counts: Counter) -> int:
    if not tar_path.exists() or tar_path.stat().st_size < 1_000_000:
        return 0
    added = 0
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            members = [m for m in tf.getmembers() if m.name.endswith(".mp4")]
            for m in members:
                stem = Path(m.name).stem
                if stem not in stem_to_label or stem in have:
                    continue
                label = stem_to_label[stem]
                if class_counts[label] >= PER_CLASS_CAP:
                    continue
                dest_dir = OUT / slug(label)
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / f"{stem}.mp4"
                if dest.exists():
                    have[stem] = label
                    continue
                src = tf.extractfile(m)
                if src is None:
                    continue
                dest.write_bytes(src.read())
                have[stem] = label
                class_counts[label] += 1
                added += 1
                if sum(class_counts.values()) >= TARGET_TOTAL:
                    break
    except Exception as e:
        print(f"  WARN: failed {tar_path.name}: {e}")
        return added
    return added


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} -> {dest.name}")
    try:
        # wget -c for resume
        r = subprocess.run(
            ["wget", "-c", "-q", "--show-progress", "-O", str(dest), url],
            check=False,
        )
        return r.returncode == 0 and dest.exists() and dest.stat().st_size > 1_000_000
    except Exception as e:
        print(f"  download error: {e}")
        return False


def load_urls(path_file: Path) -> list[str]:
    if not path_file.exists():
        return []
    return [ln.strip() for ln in path_file.read_text().splitlines() if ln.strip().startswith("http")]


def main() -> None:
    assert (ANN / "train.csv").exists(), "annotations missing"
    stem_to_label = load_stem_to_label()
    print(f"annotated relevant stems: {len(stem_to_label)}")

    OUT.mkdir(parents=True, exist_ok=True)
    have = existing_extracted()
    class_counts: Counter = Counter()
    # recount from disk with real labels via stem map
    for stem in list(have):
        if stem in stem_to_label:
            class_counts[stem_to_label[stem]] += 1
        else:
            # keep file but don't count toward caps wrongly
            pass

    # Also symlink/copy matches already sitting in the flat data/ folder
    flat = K400 / "data"
    if flat.exists():
        for p in flat.glob("*.mp4"):
            if p.stem in stem_to_label and p.stem not in have:
                label = stem_to_label[p.stem]
                if class_counts[label] >= PER_CLASS_CAP:
                    continue
                dest_dir = OUT / slug(label)
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / p.name
                if not dest.exists():
                    dest.symlink_to(p.resolve())
                have[p.stem] = label
                class_counts[label] += 1

    print(f"starting with {sum(class_counts.values())} clips already")

    # 1) existing local train tars
    for tar_path in sorted(TARGZ_TRAIN.glob("part_*.tar.gz")):
        if sum(class_counts.values()) >= TARGET_TOTAL:
            break
        n = extract_from_tar(tar_path, stem_to_label, have, class_counts)
        if n:
            print(f"  {tar_path.name}: +{n}  total={sum(class_counts.values())}")

    # 2) download + extract val parts (20 total, smaller corpus)
    val_list = TARGZ_VAL / "k400_val_path.txt"
    if not val_list.exists():
        download(
            "https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt",
            val_list,
        )
    # re-fetch list with urllib if wget wrote HTML somehow
    if not val_list.exists() or val_list.stat().st_size < 100:
        urlretrieve(
            "https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt",
            val_list,
        )

    for url in load_urls(val_list):
        if sum(class_counts.values()) >= TARGET_TOTAL:
            break
        name = url.rstrip("/").split("/")[-1]
        dest = TARGZ_VAL / name
        if not dest.exists() or dest.stat().st_size < 1_000_000:
            ok = download(url, dest)
            if not ok:
                continue
        n = extract_from_tar(dest, stem_to_label, have, class_counts)
        print(f"  val/{name}: +{n}  total={sum(class_counts.values())}")

    # 3) more train parts until target
    train_list = TARGZ_TRAIN / "k400_train_path.txt"
    urls = load_urls(train_list)
    for url in urls:
        if sum(class_counts.values()) >= TARGET_TOTAL:
            break
        name = url.rstrip("/").split("/")[-1]
        dest = TARGZ_TRAIN / name
        if dest.exists() and dest.stat().st_size > 10_000_000:
            # already processed earlier
            continue
        ok = download(url, dest)
        if not ok:
            continue
        n = extract_from_tar(dest, stem_to_label, have, class_counts)
        print(f"  train/{name}: +{n}  total={sum(class_counts.values())}")
        # free disk: keep a few tars, delete after extract if we want
        # (leave downloaded tars for now — disk is plentiful)

    # write manifest
    manifest = OUT / "manifest.tsv"
    rows = []
    for p in sorted(OUT.rglob("*.mp4")):
        stem = p.stem
        label = stem_to_label.get(stem, p.parent.name)
        event = EVENT_OF.get(label, "other")
        rows.append((stem, label, event, str(p)))
    with manifest.open("w") as f:
        f.write("stem\tkinetics_label\tevent\tpath\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

    print("\n=== DONE ===")
    print(f"total clips: {len(rows)}")
    print("by kinetics label:")
    for lab, n in Counter(r[1] for r in rows).most_common():
        print(f"  {n:4d}  {lab}")
    print("by demo event:")
    for ev, n in Counter(r[2] for r in rows).most_common():
        print(f"  {n:4d}  {ev}")
    print(f"manifest: {manifest}")
    print(f"root: {OUT}")


if __name__ == "__main__":
    main()
