"""
Main entry point for Streamlit Cloud deployment
"""
import streamlit as st
import sys
from pathlib import Path

# Set page config first (must be the first Streamlit command)
st.set_page_config(
    page_title="Recommender System",
    page_icon="🛍️", 
    layout="wide",
)

# Add the correct path to find our UI module
current_dir = Path(__file__).parent
ui_path = current_dir / "phases" / "phase4_serving" / "app"
sys.path.insert(0, str(ui_path))

# Import the simplified UI
try:
    exec(open(ui_path / "ui_simple.py").read())
except:
    # Fallback to original UI
    exec(open(ui_path / "ui.py").read())