"""ShoppingKart app: map the ceiling, then localize against the map.

    streamlit run localization/app.py

Map tab:    snap ceiling photos as you walk, tag them with a location.
Locate tab: (next) snap a photo and find where it was taken.
"""

import time
from pathlib import Path

import numpy as np
import streamlit as st

import dino
import database

STORE = Path("store")
IMAGES = STORE / "images"
CROP_BOTTOM = 0.12  # same as build_database.py -- the operator's head


def load_store():
    if (STORE / database.EMB_FILE).exists():
        return database.load(STORE)
    return np.zeros((0, 384), dtype=np.float32), []


def add_photo(jpeg_bytes, label):
    """Save the photo, embed it, append it to the database."""
    emb_db, rows = load_store()
    IMAGES.mkdir(parents=True, exist_ok=True)

    n = len(rows)
    path = IMAGES / f"photo_{n:05d}.jpg"
    path.write_bytes(jpeg_bytes)

    emb = dino.embed([path], crop_bottom=CROP_BOTTOM, quiet=True)
    rows.append({"row": n, "frame": n, "filename": path.name,
                 "time_sec": f"{time.time():.0f}", "label": label})
    database.save(np.vstack([emb_db, emb]), rows, STORE)


def map_tab():
    emb_db, rows = load_store()

    label = st.text_input("Location tag", key="tag",
                          placeholder="e.g. Aisle 3 - Dairy",
                          help="Stays until you change it, so keep snapping as you walk.")
    shot = st.camera_input("Point the camera at the ceiling", key=f"cam_{len(rows)}")

    if shot is not None:
        if not label.strip():
            st.warning("Add a location tag first.")
        else:
            with st.spinner("Embedding..."):
                add_photo(shot.getvalue(), label.strip())
            st.rerun()  # fresh camera widget (key changes with the row count)

    st.metric("Photos mapped", len(rows))
    if rows:
        counts = {}
        for r in rows:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        st.bar_chart(counts)


def locate_tab():
    st.info("Coming next.")


st.set_page_config(page_title="ShoppingKart", page_icon="🛒")
st.title("ShoppingKart")
tab_map, tab_locate = st.tabs(["Map", "Locate"])
with tab_map:
    map_tab()
with tab_locate:
    locate_tab()
