import streamlit as st
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Recommender System",
    page_icon="🛍️",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    padding: 8px 16px;
    background-color: #262730;
    border-radius: 8px;
}
.stTabs [aria-selected="true"] {
    background-color: #FF4B4B !important;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🛍️ Multi-Phase Recommender System")
st.markdown("A clean, modular recommender system with baseline → candidates → ranking → serving pipeline")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🧠 Get Recommendations", "📡 API Status", "📦 Phase Status"])

with tab1:
    st.header("Welcome to the Recommender System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌟 Features")
        st.markdown("""
        - **Phase 1**: Baseline models (Popularity, SVD Matrix Factorization)
        - **Phase 2**: Candidate generation (Item-Item similarity, User-Item neighborhood)  
        - **Phase 3**: Ranking with ML models (Logistic Regression, GBDT)
        - **Phase 4**: Production serving (FastAPI + Streamlit UI)
        """)
        
    with col2:
        st.subheader("📊 Model Performance")
        
        # Sample metrics
        metrics_data = {
            'Model': ['SVD', 'Logistic Regression', 'Popularity'],
            'Precision@10': [0.145, 0.234, 0.089],
            'Recall@10': [0.167, 0.198, 0.045],
            'Coverage': [67.2, 45.8, 23.1]
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True)

    st.subheader("🚀 Quick Start")
    with st.expander("View Pipeline Commands"):
        st.code("""
# Phase 1: Baseline models
python scripts/run_popularity.py
python scripts/run_mf_svd.py --factors 64 --n_iter 7

# Phase 2: Candidate generation  
python scripts/build_item_item.py --topk 100
python scripts/build_user_to_item.py --topk 100

# Phase 3: Train ranking model
python scripts/train_ranker.py --model logreg

# Phase 4: Start serving
streamlit run app/ui.py
        """, language="bash")

with tab2:
    st.header("🧠 Get Recommendations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Settings")
        user_id = st.number_input("User ID", min_value=0, max_value=99, value=42)
        num_recs = st.slider("Number of Recommendations", 5, 20, 10)
        method = st.selectbox(
            "Recommendation Method",
            ["Popularity (Phase 1)", "SVD (Phase 1)", "Item-Item (Phase 2)", "Ranker API (Phase 3)"]
        )
        
        if st.button("Get Recommendations", type="primary"):
            st.session_state.get_recs = True
    
    with col2:
        st.subheader("Recommendations")
        
        if st.session_state.get("get_recs", False):
            with st.spinner("Generating recommendations..."):
                # Generate sample recommendations for demo
                np.random.seed(user_id)
                sample_items = np.random.choice(50, num_recs, replace=False)
                sample_scores = np.random.uniform(0.6, 0.95, num_recs)
                
                rec_df = pd.DataFrame({
                    'Rank': range(1, num_recs + 1),
                    'Item ID': sample_items,
                    'Score': sample_scores,
                    'Method': method
                })
                
                st.success(f"Found {num_recs} recommendations for User {user_id}")
                st.dataframe(rec_df, use_container_width=True)
                
                # Show some stats
                avg_score = sample_scores.mean()
                st.metric("Average Score", f"{avg_score:.3f}")

with tab3:
    st.header("📡 API Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Service Health")
        
        # Mock API status for demo
        services = [
            ("Streamlit UI", "🟢 Online", "Serving requests"),
            ("FastAPI Backend", "🟡 Demo Mode", "Limited functionality"),
            ("Model Cache", "🟢 Ready", "Models loaded"),
            ("Database", "🟡 Sample Data", "Using demo dataset")
        ]
        
        for service, status, desc in services:
            with st.container():
                st.markdown(f"**{service}**: {status}")
                st.caption(desc)
                st.divider()
    
    with col2:
        st.subheader("API Endpoints")
        
        endpoints = [
            ("GET /health", "Health check"),
            ("GET /recommend", "Get recommendations"),
            ("POST /feedback", "Submit user feedback"),
            ("GET /stats", "System statistics")
        ]
        
        for endpoint, desc in endpoints:
            st.code(f"curl http://localhost:8080{endpoint}")
            st.caption(desc)
            st.divider()

with tab4:
    st.header("📦 Phase Status")
    
    # Mock phase status for demo
    phases = {
        "Phase 1: Baselines": {
            "popularity_model": True,
            "svd_model": True, 
            "evaluation_metrics": True
        },
        "Phase 2: Candidates": {
            "item_item_similarity": True,
            "user_item_candidates": True,
            "candidate_cache": False
        },
        "Phase 3: Ranking": {
            "feature_engineering": True,
            "model_training": True,
            "model_evaluation": False
        },
        "Phase 4: Serving": {
            "api_server": False,
            "ui_interface": True,
            "monitoring": False
        }
    }
    
    for phase_name, components in phases.items():
        st.subheader(phase_name)
        
        cols = st.columns(len(components))
        for i, (component, status) in enumerate(components.items()):
            with cols[i]:
                icon = "✅" if status else "❌"
                color = "green" if status else "red"
                st.markdown(f":{color}[{icon} {component.replace('_', ' ').title()}]")
        
        st.divider()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit, FastAPI, and scikit-learn</p>
    <p>📁 <a href='https://github.com/Aniwadkar/recommender-system' target='_blank'>View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)