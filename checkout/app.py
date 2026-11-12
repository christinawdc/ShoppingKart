"""ShoppingKart checkout: browse items, build a cart, check out.

    streamlit run checkout/app.py
"""

import base64
import glob
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import serial
import streamlit as st
from PIL import Image, ImageDraw

import analytics
from bill import make_bill

PRODUCTS_FILE = Path(__file__).parent / "products.json"
PRODUCTS = {p["id"]: p for p in json.loads(PRODUCTS_FILE.read_text())}
LOCALIZATION = Path(__file__).resolve().parent.parent / "localization"
WALK = LOCALIZATION.parent / "dataset" / "IMG_0364_2" / "frames"  # the recorded walk demo.html replays


@st.cache_resource
def open_screen(port):
    return serial.Serial(port, 115200, timeout=0)


def board():
    """The ESP32 cart screen (esp32/screen.py) on USB, or None if it isn't plugged in."""
    ports = glob.glob("/dev/cu.usbmodem*")
    return open_screen(ports[0]) if ports else None


def screen(*fields):
    """Send one line to the cart screen."""
    try:
        if s := board():
            s.write(("|".join(map(str, fields)) + "\n").encode())
    except (serial.SerialException, OSError):
        open_screen.clear()  # unplugged or re-plugged: reopen on the next message


@st.fragment(run_every=0.5)
def boot_button():
    """BOOT on the ESP32 works as the Checkout button. Reading also drains the board's "ok" replies."""
    try:
        pressed = (s := board()) is not None and b"CHECKOUT" in s.read_all()
    except (serial.SerialException, OSError):
        open_screen.clear()
        return
    if pressed and st.session_state.cart:
        checkout()
        st.rerun()


def short(pid):
    return PRODUCTS[pid]["name"].split(" (")[0]


def money(x):
    return f"₹{x:,.0f}"


def change(pid, delta):
    qty = st.session_state.cart.get(pid, 0) + delta
    if qty > 0:
        st.session_state.cart[pid] = qty
    else:
        st.session_state.cart.pop(pid, None)
    cart = st.session_state.cart
    screen("ADD", f"{'+' if delta > 0 else '-'} {short(pid)}",
           f"{sum(PRODUCTS[i]['price'] * q for i, q in cart.items()):.0f}", sum(cart.values()))


# what a scanned code can mean: our own QR ("shoppingkart:<id>") or a code printed on the real pack
BY_CODE = {c: pid for pid, p in PRODUCTS.items() for c in p.get("codes", [])}


def decode_codes(jpeg_bytes):
    """Every QR code and barcode text found in the photo."""
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    found = []
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
    found.append(text.strip())
    _, barcodes, _ = cv2.barcode.BarcodeDetector().detectAndDecodeWithType(img)[:3]
    found += [t.strip() for t in barcodes or ()]
    return [c for c in found if c]


def product_for(code):
    code = code.removeprefix("shoppingkart:")
    return code if code in PRODUCTS else BY_CODE.get(code)


def scanner():
    st.caption("Point the camera at a product's QR code or barcode and take the photo.")
    shot = st.camera_input("Scan a product", key=f"scan_{st.session_state.scan_n}")
    if shot is None:
        return
    codes = decode_codes(shot.getvalue())
    pid = next((product_for(c) for c in codes if product_for(c)), None)
    if pid:
        change(pid, 1)
        st.session_state.scan_msg = ("success", f"Added {PRODUCTS[pid]['emoji']} {PRODUCTS[pid]['name']}")
    else:
        screen("BAD")
        st.session_state.scan_msg = ("warning", f"Read {', '.join(codes)}, but that isn't in the catalogue." if codes else
                                     "No code found, try again closer or in better light.")
    st.session_state.scan_n += 1  # fresh camera widget for the next scan
    st.rerun()


def catalog():
    cols = st.columns(3)
    for i, p in enumerate(PRODUCTS.values()):
        with cols[i % 3].container(border=True):
            st.markdown(f"### {p['emoji']}")
            st.write(f"**{p['name']}**")
            st.caption(f"{money(p['price'])} · {p['zone']}")
            st.button("Add", key=f"add_{p['id']}", on_click=change, args=(p["id"], 1),
                      use_container_width=True)


def cart_panel():
    cart = st.session_state.cart
    st.subheader("Your cart")
    if not cart:
        st.info("Cart is empty. Add something.")
        return

    for pid, qty in cart.items():
        p = PRODUCTS[pid]
        name, minus, count, plus = st.columns([4, 1, 1, 1])
        name.write(f"{p['emoji']} {p['name']}")
        minus.button("−", key=f"dec_{pid}", on_click=change, args=(pid, -1))
        count.write(f"**{qty}**")
        plus.button("+", key=f"inc_{pid}", on_click=change, args=(pid, 1))

    total = sum(PRODUCTS[pid]["price"] * q for pid, q in cart.items())
    st.divider()
    st.metric("Total", money(total))
    if st.button("Checkout", type="primary", use_container_width=True):
        checkout()
        st.rerun()


def checkout():
    cart = st.session_state.cart
    total = sum(PRODUCTS[pid]["price"] * q for pid, q in cart.items())
    st.session_state.receipt = {
        "order": f"SK-{random.randint(1000, 9999)}",
        "lines": [(PRODUCTS[pid], q) for pid, q in cart.items()],
        "total": total,
        "when": datetime.now(),
    }
    screen("PAID", st.session_state.receipt["order"], f"{total:.0f}", sum(cart.values()))
    r = st.session_state.receipt
    analytics.log_order(r["lines"], total, r["when"])
    st.session_state.cart = {}


def oled_map(grid, route_left, items_left, cart):
    """The store as a 128x64 1-bit picture: shelves, the route still to walk, items still to pick, the cart.

    Row 0-6 stays free for the caption the board writes on top.
    """
    w, h = len(grid[0]) - 2, len(grid) - 2  # inside the outer walls
    k = min(126 // w, 55 // h)
    ox, oy = (128 - w * k) // 2, 8
    corner = lambda x, y: (ox + (x - 1) * k, oy + (y - 1) * k)
    centre = lambda x, y: (ox + (x - 1) * k + k // 2, oy + (y - 1) * k + k // 2)

    img = Image.new("1", (128, 64))
    g = ImageDraw.Draw(img)
    g.rectangle([ox - 1, oy - 1, ox + w * k, oy + h * k], outline=1)
    for y, row in enumerate(grid):
        for x, c in enumerate(row):
            if c == "S":
                cx, cy = corner(x, y)
                g.rectangle([cx, cy, cx + k - 1, cy + k - 1], fill=1)
    if len(route_left) > 1:
        g.line([centre(*c) for c in route_left], fill=1)
    for spot in items_left:
        cx, cy = centre(*spot)
        g.rectangle([cx - 2, cy - 2, cx + 2, cy + 2], outline=1)
    cx, cy = centre(*cart)
    g.rectangle([cx - 3, cy - 3, cx + 3, cy + 3], fill=1)
    return img


def route_tab():
    sys.path.insert(0, str(LOCALIZATION))
    import make_demo
    import route

    store, products = route.load_map()
    # shortcut: route planning is exact (2^n * n^2), so lists stop at 12 items; use a heuristic for longer ones
    picked = st.multiselect(
        "Shopping list", list(store["spots"]), default=["milk", "apple", "chips", "bread", "juice"], max_selections=12,
        format_func=lambda pid: f"{products[pid]['emoji']} {products[pid]['name']}")
    if not picked:
        st.info("Add some items to plan a route.")
        return
    r = route.plan(store, picked)
    a, b = r["list"]["dist"], r["best"]["dist"]
    c1, c2, c3 = st.columns(3)
    c1.metric("In list order", f"{a} m")
    c2.metric("Optimised", f"{b} m", f"-{a - b} m", delta_color="inverse")
    c3.metric("Shorter by", f"{100 * (a - b) / a:.0f}%")
    st.markdown("**Pick-up order:** Entrance → " + " → ".join(short(p) for p in r["best"]["items"]) + " → Checkout")

    if not WALK.exists():
        st.info(f"The recorded walk ({WALK}) isn't on this machine, so the trip can't be replayed here.")
        return
    if not st.button("Simulate trip", type="primary"):
        fig = route.figure(store, products, r)
        st.pyplot(fig)
        route.plt.close(fig)
        return

    # the recorded ceiling walk, matched against the ceiling map, moves the cart along the planned route
    sim = make_demo.simulate(WALK, self_test=True, items=picked)
    best = sim["route"]["best"]
    frame_col, oled_col = st.columns(2)
    frame_box, oled_box = frame_col.empty(), oled_col.empty()
    for n, (path, zone, pos) in enumerate(zip(sim["walk_paths"], sim["zones"], sim["pos"]), 1):
        left = [(pid, at) for pid, at in zip(best["items"], best["item_at"]) if at > pos]
        zone_name = store["zones"][zone]["name"]
        caption = f"{zone_name} {len(picked) - len(left)}/{len(picked)}"
        img = oled_map(store["grid"], best["cells"][int(pos):],
                       [store["spots"][pid] for pid, _ in left], best["cells"][int(pos)])
        screen("MAP", base64.b64encode(img.tobytes()).decode(), caption)
        frame_box.image(str(path), caption=f"Ceiling frame {n}/{len(sim['walk_paths'])}: matched {zone_name}")
        oled_box.image(img.convert("L").resize((512, 256), Image.NEAREST),
                       caption=f"Cart screen: {caption}" + (f", next {short(left[0][0])}" if left else ", checkout"))
        time.sleep(1)
    st.success("Arrived at the checkout. Scan your items, then press Checkout or BOOT.")


def free_spot(store, zone, route):
    """A floor cell beside a shelf in that zone, as far as possible from the products already placed."""
    grid = store["grid"]
    x0, x1 = next(z["x"] for z in store["zones"] if z["name"] == zone)
    used = [tuple(s) for s in store["spots"].values()] + [tuple(store["entrance"]), tuple(store["checkout"])]
    free = [(x, y) for x, y in route.bfs(grid, tuple(store["entrance"]))[0]
            if x0 <= x <= x1 and (x, y) not in used
            and any(grid[y + dy][x + dx] == "S" for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
    return max(free, key=lambda c: min(abs(c[0] - u[0]) + abs(c[1] - u[1]) for u in used), default=None)


def add_tab():
    sys.path.insert(0, str(LOCALIZATION))
    import route

    st.caption("Scan the pack's barcode or QR code, then name it and pick where it sits in the store.")
    shot = st.camera_input("Scan a new product", key=f"new_{st.session_state.new_n}")
    codes = decode_codes(shot.getvalue()) if shot else []
    if shot and not codes:
        st.warning("No code found, try again closer or in better light.")
    store = json.loads(route.MAP_FILE.read_text())

    with st.form("new_product"):
        code = st.text_input("Code", value=codes[0] if codes else "", help="Filled in by the scan, or type it.")
        name = st.text_input("Item name")
        price = st.number_input("Price (₹)", min_value=1, value=50, step=1)
        zone = st.selectbox("Location", [z["name"] for z in store["zones"]])
        if not st.form_submit_button("Add product", type="primary"):
            return
    code, name = code.strip(), name.strip()
    if not code or not name:
        st.error("Scan or type a code, and give the item a name.")
        return
    if product_for(code):
        st.error(f"{code} is already {PRODUCTS[product_for(code)]['name']}.")
        return
    spot = free_spot(store, zone, route)
    if spot is None:
        st.error(f"No free shelf space left in {zone}.")
        return

    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "item"
    pid = next(f"{base}{n or ''}" for n in range(100) if f"{base}{n or ''}" not in PRODUCTS)
    products = list(PRODUCTS.values()) + [
        {"id": pid, "name": name, "price": int(price), "emoji": "📦", "zone": zone, "codes": [code]}]
    PRODUCTS_FILE.write_text(json.dumps(products, indent=2, ensure_ascii=False) + "\n")
    store["spots"][pid] = list(spot)
    text = json.dumps(store, indent=2, ensure_ascii=False)
    route.MAP_FILE.write_text(re.sub(r"\[\n\s+(\d+),\n\s+(\d+)\n\s+\]", r"[\1, \2]", text) + "\n")  # [x, y] on one line

    st.session_state.new_n += 1  # fresh camera for the next product
    st.session_state.scan_msg = ("success", f"Added 📦 {name} ({money(price)}) to {zone}. Scan it to put it in the cart.")
    st.rerun()


def receipt_view():
    r = st.session_state.receipt
    st.success(f"Order {r['order']} placed. Thank you!")
    for p, q in r["lines"]:
        st.write(f"{p['emoji']} {p['name']} × {q} — {money(p['price'] * q)}")
    st.divider()
    st.metric("Paid", money(r["total"]))
    st.download_button("Download bill (PDF)", make_bill(r["order"], r["when"], r["lines"], r["total"]),
                       file_name=f"bill-{r['order']}.pdf", mime="application/pdf",
                       type="primary", use_container_width=True)
    if st.button("Start a new order"):
        st.session_state.receipt = None
        screen("HI")
        st.rerun()


st.set_page_config(page_title="ShoppingKart Checkout", page_icon="🛒", layout="wide")
st.session_state.setdefault("cart", {})
st.session_state.setdefault("receipt", None)
st.session_state.setdefault("scan_n", 0)
st.session_state.setdefault("new_n", 0)
st.session_state.setdefault("scan_msg", None)

st.title("🛒 ShoppingKart")
if st.session_state.receipt:
    receipt_view()
else:
    left, right = st.columns([2, 1])
    with left:
        if st.session_state.scan_msg:
            getattr(st, st.session_state.scan_msg[0])(st.session_state.scan_msg[1])
            st.session_state.scan_msg = None
        tab_browse, tab_scan, tab_route, tab_add, tab_stats = st.tabs(
            ["Browse", "Scan QR", "Route", "Add product", "Analytics"])
        with tab_browse:
            catalog()
        with tab_scan:
            scanner()
        with tab_route:
            route_tab()
        with tab_add:
            add_tab()
        with tab_stats:
            sys.path.insert(0, str(LOCALIZATION))
            import route
            analytics.analytics_tab(PRODUCTS, route)
    with right:
        cart_panel()
        boot_button()
