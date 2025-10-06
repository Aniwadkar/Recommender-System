from __future__ import annotations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import numpy as np
import pandas as pd
from scipy import sparse

from src.data.splits import leave_last_one_out, build_mappings, to_csr
from src.models.baselines.mf_svd import train_svd, recommend_svd

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--factors', type=int, default=64)
    ap.add_argument('--n_iter', type=int, default=7)
    ap.add_argument('--k', type=int, default=10)
    args = ap.parse_args()
    
    # Load data
    df = pd.read_csv('data/raw/interactions.csv')
    df_indexed, user_to_idx, item_to_idx = build_mappings(df)
    train_data, test_data = leave_last_one_out(df_indexed)
    
    # Convert to sparse matrix
    csr_matrix = to_csr(train_data, len(user_to_idx), len(item_to_idx))
    
    # Train SVD model
    U, V = train_svd(csr_matrix, factors=args.factors, n_iter=args.n_iter)
    
    # Generate recommendations
    recommendations = recommend_svd(U, V, csr_matrix, k=args.k)
    
    # Save results
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/recs_svd.npy', recommendations)
    
    # Process held-out data
    held_out = [[] for _ in range(len(user_to_idx))]
    for user_id, group in test_data.groupby('user_id'):
        held_out[user_id] = group['item_id'].tolist()
    
    # Save metadata
    with open('data/processed/heldout.json', 'w') as f:
        json.dump(held_out, f)
    
    metadata = {
        'n_users': len(user_to_idx),
        'n_items': len(item_to_idx),
        'K': args.k
    }
    with open('data/processed/meta.json', 'w') as f:
        json.dump(metadata, f)
    
    print('Saved SVD recommendations')
