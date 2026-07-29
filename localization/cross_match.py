"""Match one pass against another -- the real localization test.

Builds a database from pass A and queries it with every frame of pass B, then
checks whether the matches land in a sensible order. Tries all four rotations of
B so we can see directly how much the cart's heading matters.

  correlation near +1   B is matched in the same order as A (same direction)
  correlation near -1   B is matched in reverse (B walked back down the aisle)
  correlation near  0   matching failed -- no consistent ordering

    python localization/cross_match.py dataset/IMG_0364_2/frames dataset/IMG_0365_2/frames
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import dino


def score(sim):
    """How consistently does B's frame order map onto A's?"""
    nn = sim.argmax(axis=1)
    q = np.arange(len(nn))
    # Spearman: rank correlation, robust to the nonlinear pace of a real walk
    rank = lambda v: np.argsort(np.argsort(v))
    r = np.corrcoef(rank(q), rank(nn))[0, 1]
    return nn, r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dir_a", type=Path, help="database pass")
    p.add_argument("dir_b", type=Path, help="query pass")
    p.add_argument("--short-side", type=int, default=518)
    p.add_argument("--crop-bottom", type=float, default=0.12)
    p.add_argument("--rotations", type=int, nargs="+", default=[0, 90, 180, 270])
    p.add_argument("--out", type=Path, default=Path("dataset/cross_match.png"))
    args = p.parse_args()

    paths_a = sorted(args.dir_a.glob("frame_*.jpg"))
    paths_b = sorted(args.dir_b.glob("frame_*.jpg"))
    if not paths_a or not paths_b:
        raise SystemExit("need frames in both directories")

    print(f"A (database) {args.dir_a}: {len(paths_a)} frames")
    print(f"B (query)    {args.dir_b}: {len(paths_b)} frames\n")

    emb_a = dino.embed(paths_a, short_side=args.short_side,
                       crop_bottom=args.crop_bottom)

    results = {}
    for rot in args.rotations:
        emb_b = dino.embed(paths_b, short_side=args.short_side,
                           crop_bottom=args.crop_bottom, rotate=rot)
        sim = emb_b @ emb_a.T          # rows = query frames of B
        nn, r = score(sim)
        results[rot] = (sim, nn, r)
        print(f"  rotate B {rot:3d}deg   mean sim {sim.mean():.3f}   "
              f"best {sim.max(axis=1).mean():.3f}   order correlation {r:+.3f}")

    best = max(results, key=lambda k: abs(results[k][2]))
    sim, nn, r = results[best]
    print(f"\nstrongest ordering at {best}deg rotation (correlation {r:+.3f}) -- "
          f"{'reversed, B walked the opposite way' if r < 0 else 'same direction'}")

    fig, axes = plt.subplots(1, len(results), figsize=(4.6 * len(results), 4.4))
    axes = np.atleast_1d(axes)
    for ax, (rot, (s, n, rr)) in zip(axes, sorted(results.items())):
        ax.imshow(s, cmap="viridis", aspect="auto")
        ax.plot(n, np.arange(len(n)), "r.", ms=4, label="best match")
        ax.set_title(f"B rotated {rot}deg\ncorrelation {rr:+.3f}")
        ax.set_xlabel("database frame (A)")
        ax.set_ylabel("query frame (B)")
        ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"plot -> {args.out}")


if __name__ == "__main__":
    main()
