"""ShoppingKart checkout: browse items, build a cart, check out.

    streamlit run checkout/app.py
"""

import json
import random
from pathlib import Path

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

st.title("🛒 ShoppingKart")
if st.session_state.receipt:
    receipt_view()
else:
    left, right = st.columns([2, 1])
    with left:
        catalog()
    with right:
        cart_panel()
