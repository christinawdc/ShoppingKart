"""Store analytics: what shoppers buy and where in the store they spend their time.

There is no foot-traffic data from real customers yet, so the floor data comes from
simulated shoppers: each one gets a basket, walks the shortest route for it (route.py),
pauses at the items it picks and now and then to browse. Real checkouts from this
till (orders.jsonl) are added to the sales figures.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ORDERS_FILE = Path(__file__).parent / "orders.jsonl"
SHOPPERS = 400
WALK_S = 1.2  # seconds per metre at a shopping pace

# how often each product lands in a basket, relative to 1 (anything not listed, e.g. newly added items, is 1)
POPULAR = {"milk": 5, "bread": 5, "eggs": 4, "water": 4, "banana": 3, "apple": 3, "tomato": 3, "chips": 3,
           "choc": 2.5, "juice": 2, "rice": 2, "cheese": 1.5, "soap": 1.5, "toothpaste": 1.5, "tissues": 1.2,
           "jelly": 0.8}

# reference palette (dataviz skill): sequential blue ramp, diverging blue <-> red, categorical slot 2 for callouts
BLUES = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE, RED, ORANGE = "#2a78d6", "#e34948", "#eb6834"
SHELF, WALL = "#9a9a9a", "#222222"


def log_order(lines, total, when):
    """Called at checkout: keep a record of every real order for the analytics."""
    with ORDERS_FILE.open("a") as f:
        f.write(json.dumps({"when": when.isoformat(), "lines": {p["id"]: q for p, q in lines}, "total": total}) + "\n")


def beside_shelf(grid, x, y):
    return any(grid[y + dy][x + dx] == "S" for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))


@st.cache_data
def simulate(store, products, _route, n=SHOPPERS, seed=7):
    """Seconds spent on each floor cell (summed over shoppers) and each shopper's basket."""
    rng = random.Random(seed)
    grid, ids = store["grid"], [pid for pid in store["spots"] if pid in products]
    weights = [POPULAR.get(pid, 1) for pid in ids]
    seconds, baskets = defaultdict(float), []
    for _ in range(n):
        basket = set()
        for _ in range(rng.randint(2, 7)):
            basket.add(rng.choices(ids, weights)[0])
        best = _route.plan(store, sorted(basket))["best"]
        for x, y in best["cells"]:
            seconds[(x, y)] += WALK_S
            if beside_shelf(grid, x, y) and rng.random() < 0.04:
                seconds[(x, y)] += rng.uniform(5, 30)  # stopped to look at the shelf
        for pid in best["items"]:
            seconds[tuple(store["spots"][pid])] += rng.uniform(10, 40)  # choosing the item
        baskets.append({pid: rng.choice((1, 1, 1, 2, 3)) for pid in basket})
    return dict(seconds), baskets


def real_orders():
    if not ORDERS_FILE.exists():
        return []
    return [json.loads(line)["lines"] for line in ORDERS_FILE.read_text().splitlines() if line.strip()]


def hot_free_spots(store, seconds, k=3):
    """Busiest shelf-side floor cells with nothing stocked within 1 m: where new items get seen most."""
    grid, taken = store["grid"], [tuple(s) for s in store["spots"].values()]
    taken += [tuple(store["entrance"]), tuple(store["checkout"])]
    cells = [c for c in seconds if beside_shelf(grid, *c)
             and all(abs(c[0] - t[0]) + abs(c[1] - t[1]) > 1 for t in taken)]
    return sorted(cells, key=seconds.get, reverse=True)[:k]


def zone_of(store, x):
    return next(z["name"] for z in store["zones"] if z["x"][0] <= x <= z["x"][1])


def analytics_tab(products, route):
    store = json.loads(route.MAP_FILE.read_text())
    seconds, baskets = simulate(store, products, route)
    real = [b for b in real_orders() if all(pid in products for pid in b)]
    orders = baskets + real

    st.caption(f"{SHOPPERS} simulated shoppers walking the store, plus {len(real)} real "
               f"checkout{'s' if len(real) != 1 else ''} from this till.")

    units, revenue = defaultdict(int), defaultdict(float)
    for b in orders:
        for pid, q in b.items():
            units[pid] += q
            revenue[pid] += q * products[pid]["price"]
    total = sum(revenue.values())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders", f"{len(orders):,}")
    c2.metric("Revenue", f"₹{total:,.0f}")
    c3.metric("Average basket", f"₹{total / len(orders):,.0f}")
    c4.metric("Time in store", f"{sum(seconds.values()) / SHOPPERS / 60:.1f} min",
              help="Average per simulated shopper")

    # what people buy
    st.subheader("What people buy")
    by = st.segmented_control("Rank by", ["Units", "Revenue"], default="Units", label_visibility="collapsed") or "Units"
    sales = pd.DataFrame([{"Product": products[p]["name"], "Units": units[p], "Revenue": revenue[p],
                           "Zone": products[p]["zone"]} for p in units])
    top = sales.sort_values(by, ascending=False).head(10)
    st.altair_chart(alt.Chart(top).mark_bar(color=BLUE, cornerRadiusEnd=4, height={"band": 0.7}).encode(
        x=alt.X(f"{by}:Q", title="Units sold" if by == "Units" else "Revenue (₹)"),
        y=alt.Y("Product:N", sort="-x", title=None),
        tooltip=["Product", "Zone", "Units", alt.Tooltip("Revenue:Q", format=",.0f", title="Revenue (₹)")],
    ).properties(height=34 * len(top)), use_container_width=True)

    # where time goes vs where money comes from, per zone
    st.subheader("Time spent vs sales, by zone")
    zone_time = defaultdict(float)
    for (x, _), s in seconds.items():
        zone_time[zone_of(store, x)] += s
    zone_sales = sales.groupby("Zone")["Revenue"].sum()
    zones = pd.DataFrame([{
        "Zone": z["name"],
        "Share of time (%)": round(100 * zone_time[z["name"]] / sum(zone_time.values()), 1),
        "Share of sales (%)": round(100 * zone_sales.get(z["name"], 0) / total, 1),
    } for z in store["zones"]])
    zones["Gap (points)"] = (zones["Share of time (%)"] - zones["Share of sales (%)"]).round(1)
    zones["Reading"] = ["More time than sales: room to sell more" if g > 0 else "Sells more than its share of time"
                        for g in zones["Gap (points)"]]
    st.altair_chart(alt.Chart(zones).mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
        x=alt.X("Gap (points):Q", title="Share of time minus share of sales (percentage points)"),
        y=alt.Y("Zone:N", sort=[z["name"] for z in store["zones"]], title=None),
        color=alt.Color("Reading:N", scale=alt.Scale(
            domain=["More time than sales: room to sell more", "Sells more than its share of time"],
            range=[BLUE, RED]), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Zone", "Share of time (%)", "Share of sales (%)", "Gap (points)"],
    ).properties(height=40 * len(zones)), use_container_width=True)
    with st.expander("Table"):
        st.dataframe(zones.drop(columns="Reading"), hide_index=True, use_container_width=True)

    # heatmap of the floor
    st.subheader("Where shoppers spend their time")
    spots = hot_free_spots(store, seconds)
    cells = []
    for y, row in enumerate(store["grid"]):
        for x, c in enumerate(row):
            kind = {"#": "Wall", "S": "Shelf"}.get(c, "Floor")
            cells.append({"x": x, "y": y, "Kind": kind, "Zone": zone_of(store, x) if 0 < x < len(row) - 1 else "",
                          "Seconds per shopper": round(seconds.get((x, y), 0) / SHOPPERS, 2)})
    cells = pd.DataFrame(cells)
    pid_at = {tuple(v): k for k, v in store["spots"].items() if k in products}
    stocked = pd.DataFrame([{"x": x, "y": y, "Product": products[pid]["name"]} for (x, y), pid in pid_at.items()])
    picks = pd.DataFrame([{"x": x, "y": y, "n": str(i), "Zone": zone_of(store, x),
                           "Seconds per shopper": round(seconds[(x, y)] / SHOPPERS, 2)}
                          for i, (x, y) in enumerate(spots, 1)])

    px = 18
    W, H = len(store["grid"][0]), len(store["grid"])
    X = alt.X("x:O", axis=None, scale=alt.Scale(paddingInner=0))
    Y = alt.Y("y:O", axis=None, scale=alt.Scale(paddingInner=0))
    floor = alt.Chart(cells[cells.Kind == "Floor"]).mark_rect().encode(
        X, Y, color=alt.Color("Seconds per shopper:Q", scale=alt.Scale(range=BLUES),
                              legend=alt.Legend(orient="bottom", title="Seconds per shopper", gradientLength=240)),
        tooltip=["Zone", "Seconds per shopper"])
    solid = alt.Chart(cells[cells.Kind != "Floor"]).mark_rect().encode(
        X, Y, color=alt.Color("Kind:N", scale=alt.Scale(domain=["Shelf", "Wall"], range=[SHELF, WALL]), legend=None),
        tooltip=["Kind"])
    items = alt.Chart(stocked).mark_point(shape="square", size=40, filled=True, color="white",
                                          stroke=WALL, strokeWidth=1).encode(X, Y, tooltip=["Product"])
    ring = alt.Chart(picks).mark_circle(size=260, color=ORANGE, stroke="white", strokeWidth=2).encode(
        X, Y, tooltip=["n", "Zone", "Seconds per shopper"])
    label = alt.Chart(picks).mark_text(color="white", fontWeight="bold", fontSize=11).encode(X, Y, text="n")
    st.altair_chart((solid + floor + items + ring + label).resolve_scale(color="independent").properties(
        width=W * px, height=H * px), use_container_width=False)
    st.caption("Darker floor = more time spent there. White squares are products on the shelf. "
               "Orange numbers are busy shelf spots with nothing stocked nearby.")

    st.subheader("Where to add items")
    for i, (x, y) in enumerate(spots, 1):
        st.markdown(f"**{i}. {zone_of(store, x)}**, shelf beside ({x}, {y}): shoppers spend "
                    f"{seconds[(x, y)] / SHOPPERS:.1f} s here on average and there is nothing stocked within 1 m.")
    gap = zones.sort_values("Gap (points)", ascending=False).iloc[0]
    if gap["Gap (points)"] > 0:
        st.markdown(f"**{gap['Zone']}** gets {gap['Share of time (%)']}% of shopper time but only "
                    f"{gap['Share of sales (%)']}% of sales: the best zone for new or promoted items "
                    f"(add them in the **Add product** tab).")
