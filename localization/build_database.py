"""Embed a frame set and build the FAISS location database.

    python localization/build_database.py dataset/frames
    python localization/build_database.py dataset/frames --zones 5
"""

import argparse
import csv
from pathlib import Path

import dino
import database


def read_labels(frames_dir):
    """labels.csv sits next to the frames dir, written by extract_frames.py."""
    path = frames_dir.parent / "labels.csv"
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["filename"]: r for r in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("frames", type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="where to write the index (default: alongside the frames)")
    p.add_argument("--stride", type=int, default=1,
                   help="index every Nth frame (default: 1, all of them)")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--short-side", type=int, default=518)
    p.add_argument("--crop-bottom", type=float, default=0.12)
    p.add_argument("--zones", type=int, default=0,
                   help="auto-label the walk as N equal zones, for a readable demo")
    args = p.parse_args()

    paths = sorted(args.frames.glob("frame_*.jpg"))[args.offset::args.stride]
    if not paths:
        raise SystemExit(f"no frames in {args.frames}")

    labels = read_labels(args.frames)
    emb = dino.embed(paths, short_side=args.short_side,
                     crop_bottom=args.crop_bottom)

    rows = []
    for row, path in enumerate(paths):
        meta = labels.get(path.name, {})
        label = meta.get("label") or ""
        if args.zones and not label:
            zone = row * args.zones // len(paths)
            label = f"Zone {zone + 1}"
        rows.append({
            "row": row,
            "frame": meta.get("frame", path.stem.split("_")[-1]),
            "filename": path.name,
            "time_sec": meta.get("time_sec", ""),
            "label": label,
        })

    out_dir = args.out or args.frames.parent
    database.save(emb, rows, out_dir)

    print(f"indexed {len(emb)} frames ({emb.shape[1]}-dim) -> {out_dir}")
    if args.zones:
        print(f"auto-labelled as {args.zones} zones")


if __name__ == "__main__":
    main()
