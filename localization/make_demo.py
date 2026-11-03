"""Build a self-contained presentation page: replay one walk against a map built from another.

Matching is done here, once, with the real DINOv2 pipeline; the page just plays it back.

    python localization/make_demo.py dataset/IMG_0364_2/frames dataset/IMG_0365_2/frames
    python localization/make_demo.py dataset/IMG_0364_2/frames --self-test   # odd frames vs even frames
"""

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

import dino
import database

ZONES = ["Entrance", "Produce", "Dairy", "Checkout"]
# cart route on the minimap (viewBox 600x340), one polyline through the store
ROUTE = [(40, 300), (40, 60), (200, 60), (200, 300), (360, 300), (360, 60), (520, 60), (520, 300)]


def thumb(path, w=360, crop_bottom=0.12):
    img = Image.open(path).convert("RGB")
    W, H = img.size
    img = img.crop((0, 0, W, int(H * (1 - crop_bottom))))
    img = img.resize((w, round(w * img.size[1] / img.size[0])))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("map_frames", type=Path)
    p.add_argument("walk_frames", type=Path, nargs="?")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--out", type=Path, default=Path("demo.html"))
    args = p.parse_args()

    map_paths = sorted(args.map_frames.glob("frame_*.jpg"))
    if args.self_test:
        walk_paths, map_paths = map_paths[1::2], map_paths[0::2]
    else:
        walk_paths = sorted(args.walk_frames.glob("frame_*.jpg"))

    map_emb = dino.embed(map_paths, crop_bottom=0.12)
    walk_emb = dino.embed(walk_paths, crop_bottom=0.12)
    sims, idx = database.search(map_emb, walk_emb, k=1)

    matches = idx[:, 0]
    corr = np.corrcoef(np.arange(len(matches)), matches)[0, 1]
    print(f"map {len(map_paths)} frames, walk {len(walk_paths)} frames")
    print(f"match order correlation {corr:+.2f}  (near +1 = walk follows the map)")
    print("matches:", matches.tolist())
    print(f"mean similarity {sims.mean():.3f}")

    n = len(map_paths)
    data = {
        "zones": ZONES,
        "route": ROUTE,
        "map": [{"img": thumb(q), "zone": ZONES[i * len(ZONES) // n]}
                for i, q in enumerate(map_paths)],
        "walk": [{"img": thumb(q), "match": int(m), "sim": float(s)}
                 for q, m, s in zip(walk_paths, matches, sims[:, 0])],
    }
    html = (Path(__file__).parent / "demo_template.html").read_text()
    args.out.write_text(html.replace("__DATA__", json.dumps(data)))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
