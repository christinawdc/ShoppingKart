"""Build a self-contained presentation page: a shopping trip replayed with real ceiling frames.

The shopping list is planned into the shortest route (route.py). Then a recorded walk is
replayed frame by frame: each frame is matched against the ceiling map with DINOv2, the
match says which zone the cart is in, and the cart advances along the planned route
through that zone, ticking items off as it passes them. Matching is done here, once,
with the real pipeline; the page just plays it back, offline, with no server.

    python localization/make_demo.py dataset/IMG_0364_2/frames --self-test
    python localization/make_demo.py dataset/IMG_0364_2/frames --self-test --items milk apple chips bread juice
    python localization/make_demo.py dataset/IMG_0364_2/frames dataset/IMG_0365_2/frames   # walk vs map

--self-test replays the odd frames against a map of the even frames, so every replayed
frame is absent from the map. Map frames are split evenly into the store's zones in walk
order, the same split as the labels in dataset/database.csv.
"""

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

import database
import route

CROP_BOTTOM = 0.12


def thumb(path, w=360):
    img = Image.open(path).convert("RGB")
    W, H = img.size
    img = img.crop((0, 0, W, int(H * (1 - CROP_BOTTOM))))
    img = img.resize((w, round(w * img.size[1] / img.size[0])))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def embeddings(frames_dir, paths):
    """Use the embeddings similarity_matrix.py saved next to the frames (same settings), else embed."""
    cached = frames_dir.parent / "embeddings" / "embeddings.npy"
    if cached.exists():
        emb = np.load(cached)
        if len(emb) == len(paths):
            print(f"using cached {cached}")
            return emb
    import dino  # loads torch, only when there is no cache
    return dino.embed(paths, crop_bottom=CROP_BOTTOM)


def smooth(zones):
    """Median of 3, then never step back: the planned route only moves forward through the zones."""
    med = [sorted(zones[max(0, k - 1):k + 2])[len(zones[max(0, k - 1):k + 2]) // 2] for k in range(len(zones))]
    return [int(z) for z in np.maximum.accumulate(med)]


def cart_positions(zones, route_zone):
    """Where the cart is along the route (cell index) for each frame.

    The frames the cart spends in a zone are spread evenly over the route's stretch in that zone.
    """
    span = {z: (route_zone.index(z), len(route_zone) - 1 - route_zone[::-1].index(z)) for z in set(route_zone)}
    count = {z: zones.count(z) for z in set(zones)}
    seen, pos, out = {}, 0.0, []
    for z in zones:
        seen[z] = seen.get(z, 0) + 1
        s, e = span[z]
        pos = max(pos, s + seen[z] / count[z] * (e - s))
        out.append(round(pos, 2))
    return out


def simulate(map_frames, walk_frames=None, self_test=False, items=("milk", "apple", "chips", "bread", "juice")):
    """Match the walk's frames against the ceiling map, plan the route, and place the cart on it for each frame."""
    store, products = route.load_map()
    n_zones = len(store["zones"])

    all_paths = sorted(map_frames.glob("frame_*.jpg"))
    all_emb = embeddings(map_frames, all_paths)
    all_zone = [i * n_zones // len(all_paths) for i in range(len(all_paths))]
    if self_test:
        map_idx, walk_idx = list(range(0, len(all_paths), 2)), list(range(1, len(all_paths), 2))
        map_paths, map_emb = [all_paths[i] for i in map_idx], all_emb[map_idx]
        map_zone = [all_zone[i] for i in map_idx]
        walk_paths, walk_emb = [all_paths[i] for i in walk_idx], all_emb[walk_idx]
    else:
        map_paths, map_emb, map_zone = all_paths, all_emb, all_zone
        walk_paths = sorted(walk_frames.glob("frame_*.jpg"))
        walk_emb = embeddings(walk_frames, walk_paths)

    sims, idx = database.search(map_emb, walk_emb, k=1)
    matches = idx[:, 0]
    raw = [map_zone[m] for m in matches]
    zones = smooth(raw)

    r = route.plan(store, list(dict.fromkeys(items)))
    names = [z["name"] for z in store["zones"]]
    route_zone = [names.index(route.zone_at(store, x)["name"]) for x, _ in r["best"]["cells"]]
    return {"store": store, "products": products, "route": r,
            "map_paths": map_paths, "map_zone": map_zone, "walk_paths": walk_paths,
            "matches": matches, "sims": sims, "raw": raw, "zones": zones,
            "pos": cart_positions(zones, route_zone)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("map_frames", type=Path)
    p.add_argument("walk_frames", type=Path, nargs="?")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--items", nargs="+", default=["milk", "apple", "chips", "bread", "juice"])
    p.add_argument("--out", type=Path, default=Path("demo.html"))
    args = p.parse_args()

    sim = simulate(args.map_frames, args.walk_frames, args.self_test, args.items)
    store, products, r = sim["store"], sim["products"], sim["route"]
    map_paths, map_zone, walk_paths = sim["map_paths"], sim["map_zone"], sim["walk_paths"]
    matches, sims, raw, zones, pos = sim["matches"], sim["sims"], sim["raw"], sim["zones"], sim["pos"]
    best = r["best"]
    print(f"map {len(map_paths)} frames, walk {len(walk_paths)} frames, mean similarity {sims.mean():.3f}")
    print("matched zone :", " ".join(str(z + 1) for z in raw))
    print("smoothed zone:", " ".join(str(z + 1) for z in zones))
    print(f"route {best['dist']} m (list order {r['list']['dist']} m): " + " -> ".join(best["items"]))

    data = {
        "store": {k: store[k] for k in ("grid", "entrance", "checkout", "zones")},
        "items": [{"id": pid, "name": products[pid]["name"], "emoji": products[pid]["emoji"],
                   "spot": store["spots"][pid], "at": at}
                  for pid, at in zip(best["items"], best["item_at"])],
        "route": best["cells"],
        "listRoute": r["list"]["cells"],
        "dist": {"list": r["list"]["dist"], "best": best["dist"]},
        "map": [{"img": thumb(q), "zone": z} for q, z in zip(map_paths, map_zone)],
        "walk": [{"img": thumb(q), "match": int(m), "sim": float(s), "raw": rz, "zone": int(z), "pos": ps}
                 for q, m, s, rz, z, ps in zip(walk_paths, matches, sims[:, 0], raw, zones, pos)],
    }
    html = (Path(__file__).parent / "demo_template.html").read_text()
    args.out.write_text(html.replace("__DATA__", json.dumps(data)))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
