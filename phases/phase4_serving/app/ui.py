from __future__ import annotations
import os, json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------- Page + basic theme ----------
st.set_page_config(
    page_title="Recommender System",
    page_icon="🛍️",
    layout="wide",
)

# ---------- Light styling (clean, “storefront” feel) ----------
st.markdown("""
<style>
:root{
  --topbar-h: 56px; /* height of the sticky title bar */
}

/* Push content down so tabs never sit under the sticky bars */
.main .block-container { padding-top: calc(var(--topbar-h) + 48px) !important; }

/* --- Sticky Topbar (app title) --- */
.topbar{
  position: sticky;
  top: 0;
  z-index: 1100;
  display: flex;
  align-items: center;
  gap: .6rem;
  padding: .75rem 1rem;
  background: var(--background-color, #0e1117);
  border-bottom: 1px solid rgba(255,255,255,0.08);
}
.topbar .title{
  font-weight: 600;
  font-size: 3rem;
  letter-spacing: .2px;
}

/* --- Tabs just below the topbar --- */
.stTabs [role="tablist"]{
  position: sticky;
  top: var(--topbar-h);
  z-index: 1000;
  background: var(--background-color, #0e1117);
  margin-top: 0;
  padding: .25rem .5rem;
  border-bottom: 1px solid rgba(255,255,255,0.08);
}

/* Tab buttons */
.stTabs [role="tab"]{
  padding: .6rem 1rem;
  margin-right: .25rem;
  border-radius: 10px 10px 0 0;
  border: 1px solid transparent;
}

/* Selected tab accent */
.stTabs [aria-selected="true"]{
  background: rgba(255,255,255,0.05);
  border-color: rgba(255,255,255,0.15);
}

/* --- Site styling --- */
.hero {
  padding: 1.25rem 1.5rem;
  border-radius: 16px;
  background: linear-gradient(135deg, #1f2937 0%, #0f172a 100%);
  color: #e5e7eb;
  border: 1px solid rgba(255,255,255,0.06);
  margin: .75rem 0 1rem 0;
}
.card {
  background: #0b1220;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px;
  padding: 1rem 1.25rem;
}
.small-label { font-size: 0.85rem; opacity: 0.9; }
.stButton>button {
  border-radius: 10px;
  padding: .5rem 1rem;
  border: 1px solid rgba(255,255,255,0.12);
}
</style>
""", unsafe_allow_html=True)

# ---------- Sticky Title (ABOVE tabs) ----------
st.markdown(
    '<div class="topbar"><div class="title">🛍️ Recommender System</div></div>',
    unsafe_allow_html=True
)

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]   # .../phase4_serving
PHASES = ROOT.parent
P1 = PHASES / "phase1_baselines"
P2 = PHASES / "phase2_candidates"
P3 = PHASES / "phase3_ranking"
API_BASE = "http://127.0.0.1:8080"

# ---------- Helpers ----------
def phase_status() -> dict:
    return {
        "Phase 1": {
            "popularity_recs": (P1 / "data/processed/recs_popularity.npy").exists(),
            "svd_recs": (P1 / "data/processed/recs_svd.npy").exists(),
            "metadata": (P1 / "data/processed/meta.json").exists(),
        },
        "Phase 2": {
            "item_item": (P2 / "outputs/item_item_candidates.json").exists(),
            "user_item": (P2 / "outputs/user_to_item_candidates.json").exists(),
        },
        "Phase 3": {
            "features": (P3 / "outputs/features.csv").exists(),
            "ranker": (P3 / "outputs/ranker.joblib").exists(),
        },
    }

def load_local_recs(kind: str) -> np.ndarray | None:
    if kind == "Popularity (Phase 1)":
        path = P1 / "data/processed/recs_popularity.npy"
    else:
        path = P1 / "data/processed/recs_svd.npy"
    if not path.exists():
        st.error(f"Missing file: {path}")
        return None
    return np.load(path)

def call_ranker_api(user_id: int, k: int) -> list[int]:
    try:
        r = requests.get(f"{API_BASE}/recommend", params={"user_id": user_id, "k": k}, timeout=5)
        if r.status_code == 200:
            return r.json().get("items", [])
        else:
            st.error(f"API {r.status_code}: {r.text}")
            return []
    except Exception as e:
        st.error(f"API call failed: {e}")
        return []

# ---------- Top Navigation (no sidebar dropdown) ----------
tabs = st.tabs(["🏠 Home", "🧠 Get Recommendations", "📡 API Status", "📦 Phase Status"])

# =========================
# Tab 1: Home (no phase badges here)
# =========================
with tabs[0]:
    # Title moved to sticky topbar; keep only the subtitle/hero content here.
    st.markdown("""
<div class="hero">
  <div class="small-label">A clean multi-phase pipeline: baselines ➜ candidates ➜ ranking ➜ serving</div>
</div>
""", unsafe_allow_html=True)

    colA, colB = st.columns((2, 1), gap="large")
    with colA:
        st.subheader("About This System")
        st.write(
            "- **Phase 1**: Baselines (Popularity, Matrix Factorization via SVD)\n"
            "- **Phase 2**: Candidate generation (Item-Item cosine, User-Item neighborhood)\n"
            "- **Phase 3**: Ranking (Logistic Regression / GBDT on simple features)\n"
            "- **Phase 4**: Serving (FastAPI API) + this Streamlit UI\n"
        )
        st.subheader("Quick Start")
        st.code(
            """# Phase 1
cd phases/phase1_baselines
python scripts/run_popularity.py
python scripts/run_mf_svd.py --factors 64 --n_iter 7

# Phase 2
cd ../phase2_candidates
python scripts/build_item_item.py --topk 100
python scripts/build_user_to_item.py --topk 100

# Phase 3
cd ../phase3_ranking
python scripts/build_features.py --interactions ../phase1_baselines/data/raw/interactions.csv --item-item ../phase2_candidates/outputs/item_item_candidates.json --user2item ../phase2_candidates/outputs/user_to_item_candidates.json
python scripts/train_ranker.py --model logreg

# Phase 4 (API)
cd ../phase4_serving
uvicorn app.main:app --reload --port 8080
""",
            language="bash",
        )

    with colB:
        st.subheader("Why this design?")
        st.markdown(
            """
- **Modular**: swap baselines, candidate generators, or rankers easily  
- **Fast to run**: SVD has no native compiler requirements  
- **Demo-ready**: simple API + UI to showcase results
"""
        )
        st.markdown("")
        st.markdown('<div class="card"><span class="small-label">Tip</span><br/>Use the “Get Recommendations” tab to try users and methods.</div>', unsafe_allow_html=True)

# =========================
# Tab 2: Get Recommendations
# =========================
with tabs[1]:
    st.subheader("Get Recommendations")
    col1, col2, col3 = st.columns([1, 1, 2], gap="large")
    with col1:
        user_id = st.number_input("User ID", min_value=0, value=0, step=1)
    with col2:
        k = st.slider("Number of recommendations", min_value=1, max_value=50, value=10)
    with col3:
        method = st.selectbox("Method", ["Popularity (Phase 1)", "SVD (Phase 1)", "Ranker API (Phase 3)"])
    st.markdown("")
    run = st.button("Get Recommendations")
    if run:
        if method in ("Popularity (Phase 1)", "SVD (Phase 1)"):
            arr = load_local_recs(method)
            if arr is not None:
                if user_id >= arr.shape[0]:
                    st.error(f"User {user_id} not found (max: {arr.shape[0]-1}).")
                else:
                    items = arr[int(user_id), :int(k)].tolist()
                    st.success(f"Top {k} for user {user_id} via {method}:")
                    st.write(items)
        else:
            items = call_ranker_api(int(user_id), int(k))
            if items:
                st.success(f"Top {k} for user {user_id} via Ranker (API):")
                st.write(items)

# =========================
# Tab 3: API Status
# =========================
with tabs[2]:
    st.subheader("API Status")
    cols = st.columns([1, 2])
    with cols[0]:
        if st.button("Check /health"):
            try:
                r = requests.get(f"{API_BASE}/health", timeout=5)
                if r.status_code == 200:
                    st.success("API healthy")
                    st.json(r.json())
                else:
                    st.error(f"API {r.status_code}: {r.text}")
            except Exception as e:
                st.error(f"API call failed: {e}")
    with cols[1]:
        st.markdown("**Base URL:** " + API_BASE)
        st.markdown("**Endpoints:**")
        st.markdown("- `GET /health` — health check")
        st.markdown("- `GET /recommend?user_id=<id>&k=<k>` — recommendations")

# =========================
# Tab 4: Phase Status
# =========================
with tabs[3]:
    st.subheader("Phase Completion Status")
    status = phase_status()
    for phase, files in status.items():
        ok = all(files.values())
        st.markdown(f"### {phase}")
        if ok:
            st.success(f"{phase} complete")
        else:
            st.error(f"{phase} incomplete")
        with st.expander(f"View {phase} files"):
            for name, exists in files.items():
                st.write(f"- {name}: {'✅' if exists else '❌'}")
