"""
Main Streamlit app entry point for Streamlit Cloud
This file should be in the root directory
"""
import sys
from pathlib import Path

# Add the app directory to path
app_dir = Path(__file__).parent / "phases" / "phase4_serving" / "app"
sys.path.insert(0, str(app_dir))

# Import and run the main UI
from ui import *