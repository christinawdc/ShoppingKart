"""Look up where a ceiling image was taken.

    python localization/query_location.py dataset/frames/frame_0005.jpg --db dataset
    python localization/query_location.py dataset/frames/*.jpg --db dataset -k 3
"""

import argparse
from pathlib import Path

import dino
import database


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images", type=Path, nargs="+")
    p.add_argument("--db", type=Path, default=Path("dataset"))
    p.add_argument("-k", type=int, default=5, help="how many matches to show")
    p.add_argument("--short-side", type=int, default=518)
    p.add_argument("--crop-bottom", type=float, default=0.12)
    args = p.parse_args()

    db_emb, rows = database.load(args.db)
    emb = dino.embed(args.images, short_side=args.short_side,
                     crop_bottom=args.crop_bottom)
    sims, idx = database.search(db_emb, emb, args.k)

    for image, s, ix in zip(args.images, sims, idx):
        best = rows[ix[0]]
        where = best["label"] or f"frame {best['frame']}"
        print(f"\n{image.name}  ->  {where}   (t={best['time_sec']}s, "
              f"similarity {s[0]:.3f})")
        for rank, (sim, r) in enumerate(zip(s, ix), 1):
            m = rows[r]
            print(f"   {rank}. {m['filename']:20s} {m['label']:>10s} "
                  f"t={m['time_sec']:>7s}s  {sim:.3f}")


if __name__ == "__main__":
    main()
