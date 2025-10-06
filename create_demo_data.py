import pandas as pd
import numpy as np
import json
from pathlib import Path

def create_demo_data():
    """Create sample data for Streamlit Cloud demo"""
    np.random.seed(42)
    
    # Create sample interactions
    n_users, n_items = 100, 50
    n_interactions = 1000
    
    users = np.random.randint(0, n_users, n_interactions)
    items = np.random.randint(0, n_items, n_interactions)
    ratings = np.random.choice([1, 2, 3, 4, 5], n_interactions, p=[0.1, 0.1, 0.2, 0.3, 0.3])
    
    df = pd.DataFrame({
        'user_id': users,
        'item_id': items, 
        'rating': ratings,
        'timestamp': np.random.randint(1609459200, 1640995200, n_interactions)
    }).drop_duplicates(['user_id', 'item_id'])
    
    # Create directories
    dirs = [
        "phases/phase1_baselines/data/raw",
        "phases/phase1_baselines/data/processed", 
        "phases/phase2_candidates/outputs",
        "phases/phase3_ranking/outputs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Save interactions
    df.to_csv("phases/phase1_baselines/data/raw/interactions.csv", index=False)
    
    # Create sample recommendations
    recs_popularity = np.random.randint(0, n_items, (n_users, 10))
    recs_svd = np.random.randint(0, n_items, (n_users, 10))
    
    np.save("phases/phase1_baselines/data/processed/recs_popularity.npy", recs_popularity)
    np.save("phases/phase1_baselines/data/processed/recs_svd.npy", recs_svd)
    
    # Create metadata
    metadata = {"n_users": n_users, "n_items": n_items, "K": 10}
    with open("phases/phase1_baselines/data/processed/meta.json", "w") as f:
        json.dump(metadata, f)
    
    # Create sample candidates
    item_candidates = {str(i): list(np.random.choice(n_items, 20, replace=False)) for i in range(n_items)}
    user_candidates = {str(i): list(np.random.choice(n_items, 20, replace=False)) for i in range(n_users)}
    
    with open("phases/phase2_candidates/outputs/item_item_candidates.json", "w") as f:
        json.dump(item_candidates, f)
    
    with open("phases/phase2_candidates/outputs/user_to_item_candidates.json", "w") as f:
        json.dump(user_candidates, f)
    
    print("✓ Created demo data files")

if __name__ == "__main__":
    create_demo_data()