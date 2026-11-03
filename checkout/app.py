"""ShoppingKart checkout: browse items, build a cart, check out.

    streamlit run checkout/app.py
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

PRODUCTS = {p["id"]: p for p in json.loads((Path(__file__).parent / "products.json").read_text())}


def money(x):
    return f"₹{x:,.0f}"


def change(pid, delta):
    qty = st.session_state.cart.get(pid, 0) + delta
    if qty > 0:
        st.session_state.cart[pid] = qty
    else:
        st.session_state.cart.pop(pid, None)


def decode_qr(jpeg_bytes):
    """Return the text in the first QR code found in the photo, or ''."""
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
    return text.strip()


def scanner():
    st.caption("Point the camera at a product's QR code and take the photo.")
    shot = st.camera_input("Scan a product", key=f"scan_{st.session_state.scan_n}")
    if shot is None:
        return
    code = decode_qr(shot.getvalue()).removeprefix("shoppingkart:")
    if code in PRODUCTS:
        change(code, 1)
        st.session_state.scan_msg = ("success", f"Added {PRODUCTS[code]['emoji']} {PRODUCTS[code]['name']}")
    else:
        st.session_state.scan_msg = ("warning", "No product QR found, try again closer or in better light.")
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
        st.session_state.receipt = {
            "order": f"SK-{random.randint(1000, 9999)}",
            "lines": [(PRODUCTS[pid], q) for pid, q in cart.items()],
            "total": total,
        }
        st.session_state.cart = {}
        st.rerun()


def receipt_view():
    r = st.session_state.receipt
    st.success(f"Order {r['order']} placed. Thank you!")
    for p, q in r["lines"]:
        st.write(f"{p['emoji']} {p['name']} × {q} — {money(p['price'] * q)}")
    st.divider()
    st.metric("Paid", money(r["total"]))
    if st.button("Start a new order"):
        st.session_state.receipt = None
        st.rerun()


st.set_page_config(page_title="ShoppingKart Checkout", page_icon="🛒", layout="wide")
st.session_state.setdefault("cart", {})
st.session_state.setdefault("receipt", None)
st.session_state.setdefault("scan_n", 0)
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
        tab_browse, tab_scan = st.tabs(["Browse", "Scan QR"])
        with tab_browse:
            catalog()
        with tab_scan:
            scanner()
    with right:
        cart_panel()
