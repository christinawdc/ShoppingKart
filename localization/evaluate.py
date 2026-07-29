"""Held-out test: index half the frames, localize the other half, measure the error.

Splits a single pass into alternating database and query frames. Every query frame
is therefore absent from the database, and the nearest database frame is one step
away in either direction -- so a perfect result is an error of 1 frame.

    python localization/evaluate.py dataset/frames
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import dino
import database


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("frames", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--short-side", type=int, default=518)
    p.add_argument("--crop-bottom", type=float, default=0.12)
    p.add_argument("--pairs", type=int, default=4,
                   help="how many query/match pairs to show in the montage")
    args = p.parse_args()

    paths = sorted(args.frames.glob("frame_*.jpg"))
    if len(paths) < 6:
        raise SystemExit(f"need at least 6 frames, found {len(paths)}")

    emb = dino.embed(paths, short_side=args.short_side,
                     crop_bottom=args.crop_bottom)

    db_i = np.arange(0, len(paths), 2)
    q_i = np.arange(1, len(paths), 2)

    sims, idx = database.search(emb[db_i], emb[q_i], k=1)
    pred = db_i[idx[:, 0]]
    err = np.abs(pred - q_i)

    print(f"\ndatabase {len(db_i)} frames   queries {len(q_i)} frames")
    print(f"exact neighbour (1 frame off): {(err <= 1).mean():.1%}")
    print(f"within 2 frames:               {(err <= 2).mean():.1%}")
    print(f"median error {np.median(err):.1f} frames   max error {err.max()} frames")

    worst = q_i[err.argmax()]
    print(f"worst query: frame {worst} matched frame {pred[err.argmax()]} "
          f"(similarity {sims[err.argmax(), 0]:.3f})")

    out = args.out or args.frames.parent / "evaluation.png"
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, args.pairs, height_ratios=[1.25, 1])

    ax = fig.add_subplot(gs[0, :args.pairs // 2])
    ax.plot([0, len(paths)], [0, len(paths)], "--", c="grey", lw=1, label="perfect")
    ax.scatter(q_i, pred, s=28, c="tab:blue", zorder=3, label="predicted")
    ax.set_xlabel("true frame (query)")
    ax.set_ylabel("matched frame (database)")
    ax.set_title("predicted vs true position")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, args.pairs // 2:])
    ax.step(q_i, err, where="mid", lw=1.6)
    ax.axhline(1, ls="--", c="tab:green", lw=1, label="ideal (1 frame)")
    ax.set_xlabel("true frame (query)")
    ax.set_ylabel("error (frames)")
    ax.set_title(f"localization error -- median {np.median(err):.0f}, max {err.max()}")
    ax.set_ylim(0, max(3, err.max() + 1))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # a few query/match pairs, so the numbers are backed by something visible
    show = np.linspace(0, len(q_i) - 1, args.pairs).astype(int)
    for col, qi in enumerate(show):
        sub = fig.add_subplot(gs[1, col])
        q_img = Image.open(paths[q_i[qi]])
        m_img = Image.open(paths[pred[qi]]).resize(q_img.size)
        sub.imshow(np.hstack([np.asarray(q_img), np.asarray(m_img)]))
        sub.set_title(f"query {q_i[qi]}  |  match {pred[qi]}\n"
                      f"sim {sims[qi, 0]:.3f}, err {err[qi]}", fontsize=8)
        sub.axis("off")

    fig.tight_layout()
    fig.savefig(out, dpi=125)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
