"""Embed every frame with DINOv2 and render a frame-to-frame similarity matrix.

This is the diagnostic that decides whether ceiling place-recognition is viable in
a given part of the store. Read the output like this:

  bright diagonal      consecutive frames match -- expected, means the pipeline works
  thick diagonal band  many nearby frames look alike -- caps localization precision
  bright off-diagonal  two different places look identical -- aliasing, the real risk

    python localization/similarity_matrix.py
    python localization/similarity_matrix.py --frames dataset/IMG_0364_2/frames
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import dino


def report(sim, gap_window):
    n = len(sim)
    gaps = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))

    adjacent = sim[gaps == 1].mean()
    far = sim[gaps > gap_window]

    print(f"\nadjacent frames (gap 1)      mean similarity  {adjacent:.3f}")
    print(f"distant frames  (gap >{gap_window})      mean similarity  {far.mean():.3f}")
    print(f"                              max similarity  {far.max():.3f}")
    print(f"separation (adjacent - distant mean)         {adjacent - far.mean():.3f}")

    iu = np.triu(gaps > gap_window)
    pairs = sorted(((sim[i, j], i, j) for i, j in zip(*np.where(iu))), reverse=True)

    print(f"\ntop aliasing candidates (different places, gap >{gap_window}):")
    for s, i, j in pairs[:8]:
        print(f"  frame {i:3d} <-> {j:3d}   gap {abs(i - j):3d}   similarity {s:.3f}")

    return np.array([sim[gaps == g].mean() for g in range(n)])


def plot(sim, curve, out_path, gap_window):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    im = ax1.imshow(sim, cmap="viridis")
    ax1.set_title("frame-to-frame cosine similarity")
    ax1.set_xlabel("frame")
    ax1.set_ylabel("frame")
    fig.colorbar(im, ax=ax1, fraction=0.046)

    ax2.plot(curve, lw=2)
    ax2.axvline(gap_window, ls="--", c="grey", lw=1)
    ax2.set_title("mean similarity vs frame gap")
    ax2.set_xlabel("frame gap (0.5s each)")
    ax2.set_ylabel("mean cosine similarity")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nplot -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=Path, default=Path("dataset/frames"))
    p.add_argument("--out", type=Path, default=Path("dataset"))
    p.add_argument("--model", default="dinov2_vits14",
                   help="dinov2_vits14 (fast) / dinov2_vitb14 / dinov2_vitl14")
    p.add_argument("--short-side", type=int, default=518,
                   help="rounded to a multiple of 14 (default: 518)")
    p.add_argument("--crop-bottom", type=float, default=0.12,
                   help="crop this fraction off the bottom (default: 0.12)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gap-window", type=int, default=6,
                   help="frames closer than this are treated as the same place")
    args = p.parse_args()

    paths = sorted(args.frames.glob("frame_*.jpg"))
    if not paths:
        raise SystemExit(f"no frames in {args.frames} -- run extract_frames.py first")

    emb = dino.embed(paths, model_name=args.model, short_side=args.short_side,
                     crop_bottom=args.crop_bottom, batch_size=args.batch_size)
    sim = emb @ emb.T

    emb_dir = args.out / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_dir / "embeddings.npy", emb)
    print(f"embeddings {emb.shape} -> {emb_dir / 'embeddings.npy'}")

    curve = report(sim, args.gap_window)
    plot(sim, curve, args.out / "similarity.png", args.gap_window)


if __name__ == "__main__":
    main()
