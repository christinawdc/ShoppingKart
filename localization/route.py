"""Plan the shortest walk through the store for a shopping list.

Walking distances come from a breadth-first search over the floor plan, which goes
around the shelves rather than through them. The visiting order is then solved
exactly with Held-Karp dynamic programming: start at the entrance, pick up every
item, finish at the checkout. A list of n items costs about 2^n * n^2 steps, which
takes well under a second for the whole 13-item catalogue.

    python localization/route.py milk apple chips bread juice
    python localization/route.py milk apple chips bread juice --out dataset/route.png
"""

import argparse
import json
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).parent
MAP_FILE = HERE / "store_map.json"
PRODUCTS_FILE = HERE.parent / "checkout" / "products.json"


def load_map():
    store = json.loads(MAP_FILE.read_text())
    products = {p["id"]: p for p in json.loads(PRODUCTS_FILE.read_text())}
    for pid, (x, y) in store["spots"].items():
        assert store["grid"][y][x] == ".", f"{pid} spot {x, y} is not on the floor"
        zone = zone_at(store, x)["name"]
        assert zone == products[pid]["zone"], f"{pid} is in {zone} on the map, {products[pid]['zone']} in the catalogue"
    return store, products


def zone_at(store, x):
    return next(z for z in store["zones"] if z["x"][0] <= x <= z["x"][1])


def bfs(grid, start):
    """Distance and parent for every floor cell reachable from start."""
    dist, parent = {start: 0}, {start: None}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if grid[ny][nx] == "." and (nx, ny) not in dist:
                dist[(nx, ny)] = dist[(x, y)] + 1
                parent[(nx, ny)] = (x, y)
                queue.append((nx, ny))
    return dist, parent


def walk_between(parent, goal):
    """Cells from the BFS source to goal, using that source's parent map."""
    path = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path[::-1]


def best_order(d):
    """Held-Karp. d is the distance matrix over [start, item 1..n, end].

    Returns the item indices (1..n) in visiting order and the total distance.
    """
    n = len(d) - 2
    if n == 0:
        return [], d[0][1]
    INF = float("inf")
    # cost[mask][j]: shortest walk from start that visits exactly the items in mask, ending at item j
    cost = [[INF] * n for _ in range(1 << n)]
    prev = [[-1] * n for _ in range(1 << n)]
    for j in range(n):
        cost[1 << j][j] = d[0][j + 1]
    for mask in range(1, 1 << n):
        for j in range(n):
            c = cost[mask][j]
            if c == INF:
                continue
            for k in range(n):
                if mask & (1 << k):
                    continue
                m2 = mask | (1 << k)
                c2 = c + d[j + 1][k + 1]
                if c2 < cost[m2][k]:
                    cost[m2][k], prev[m2][k] = c2, j
    full = (1 << n) - 1
    last = min(range(n), key=lambda j: cost[full][j] + d[j + 1][n + 1])
    total = cost[full][last] + d[last + 1][n + 1]
    order, mask = [], full
    while last != -1:
        order.append(last + 1)
        mask, last = mask & ~(1 << last), prev[mask][last]
    return order[::-1], total


def plan(store, items):
    """Distance in list order and in the optimised order, plus both cell-by-cell walks."""
    grid = store["grid"]
    stops = [tuple(store["entrance"])] + [tuple(store["spots"][i]) for i in items] + [tuple(store["checkout"])]
    searches = [bfs(grid, s) for s in stops]
    d = [[searches[a][0][stops[b]] for b in range(len(stops))] for a in range(len(stops))]

    def cells(seq):
        """The walk cell by cell, and the index in it where each stop is reached."""
        path, reached = [stops[seq[0]]], [0]
        for a, b in zip(seq, seq[1:]):
            path += walk_between(searches[a][1], stops[b])[1:]
            reached.append(len(path) - 1)
        return path, reached

    def summary(seq, dist):
        path, reached = cells(seq)
        return {"items": [items[i - 1] for i in seq[1:-1]], "dist": dist,
                "cells": path, "item_at": reached[1:-1]}

    last = len(stops) - 1
    list_seq = list(range(len(stops)))
    order, total = best_order(d)
    return {
        "list": summary(list_seq, sum(d[a][b] for a, b in zip(list_seq, list_seq[1:]))),
        "best": summary([0] + order + [last], total),
    }


ZONE_COLORS = ["#4C78A8", "#59A14F", "#F28E2B", "#B07AA1", "#E15759"]


def draw(ax, store, products, route, title, color):
    grid = store["grid"]
    H, W = len(grid), len(grid[0])
    for i, z in enumerate(store["zones"]):
        x0, x1 = z["x"]
        ax.add_patch(Rectangle((x0 - .5, .5), x1 - x0 + 1, H - 2, color=ZONE_COLORS[i], alpha=.12, lw=0))
        ax.text((x0 + x1) / 2, .2, z["name"], ha="center", va="bottom", fontsize=8,
                color=ZONE_COLORS[i], weight="bold")
    for y, row in enumerate(grid):
        for x, c in enumerate(row):
            if c != ".":
                ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, color="#222" if c == "#" else "#9a9a9a", lw=0))

    xs, ys = zip(*route["cells"])
    ax.plot(xs, ys, color=color, lw=2.2, alpha=.85, solid_capstyle="round", zorder=3)

    for n, pid in enumerate(route["items"], 1):
        x, y = store["spots"][pid]
        ax.scatter(x, y, s=180, color=color, edgecolor="white", lw=1.2, zorder=4)
        ax.text(x, y, str(n), ha="center", va="center", color="white", fontsize=7, weight="bold", zorder=5)
        ax.text(x + .45, y - .45, products[pid]["name"].split(" (")[0], fontsize=6.5, zorder=5)
    for (x, y), label in ((store["entrance"], "Entrance"), (store["checkout"], "Checkout")):
        ax.scatter(x, y, s=110, marker="s", color="black", zorder=4)
        ax.text(x, y + .75, label, ha="center", va="top", fontsize=7, weight="bold", color="white")

    ax.set_title(f"{title}: {route['dist']} m", fontsize=11)
    ax.set_xlim(-.5, W - .5)
    ax.set_ylim(H + .3, -.7)
    ax.set_aspect("equal")
    ax.axis("off")


def figure(store, products, r):
    """Side by side: the list-order walk and the optimised walk."""
    a, b = r["list"]["dist"], r["best"]["dist"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    draw(axes[0], store, products, r["list"], "In list order", "#E15759")
    draw(axes[1], store, products, r["best"], "Optimised", "#2E7D32")
    fig.suptitle(f"Shopping list route: {100 * (a - b) / a:.0f}% shorter", fontsize=13, weight="bold")
    fig.tight_layout()
    return fig


def main():
    store, products = load_map()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("items", nargs="+", metavar="item", help="product ids: " + " ".join(store["spots"]))
    p.add_argument("--out", type=Path, default=Path("dataset/route.png"))
    args = p.parse_args()

    unknown = [i for i in args.items if i not in store["spots"]]
    if unknown:
        raise SystemExit(f"unknown item(s): {' '.join(unknown)}\nknown: {' '.join(store['spots'])}")
    items = list(dict.fromkeys(args.items))  # drop repeats, keep the first

    r = plan(store, items)
    a, b = r["list"]["dist"], r["best"]["dist"]
    saved = 100 * (a - b) / a
    print(f"list order : {a:>3} m  " + " -> ".join(r["list"]["items"]))
    print(f"optimised  : {b:>3} m  " + " -> ".join(r["best"]["items"]))
    print(f"saves {a - b} m ({saved:.0f}%)")

    fig = figure(store, products, r)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
