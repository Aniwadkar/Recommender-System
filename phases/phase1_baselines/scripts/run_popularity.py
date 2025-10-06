from __future__ import annotations
import pandas as pd, numpy as np, os, json, sys

# Add the phase1_baselines directory to Python path so we can import from src
current_dir = os.path.dirname(__file__)
phase1_dir = os.path.dirname(current_dir)
sys.path.insert(0, phase1_dir)

from src.data.splits import leave_last_one_out, build_mappings

K = 10

if __name__ == '__main__':
    # Read data from the correct location
    data_file = os.path.join(phase1_dir, 'data', 'raw', 'interactions.csv')
    df = pd.read_csv(data_file)
    df_i, u2i, it2i = build_mappings(df)
    
    train, test = leave_last_one_out(df_i)
    n_users, len_items = len(u2i), len(it2i)
    
    # Calculate item popularity
    counts = train.groupby('item_id').size().reindex(range(len(it2i)), fill_value=0).to_numpy()
    ranked = np.argsort(-counts)
    
    # Build seen items per user
    seen = {u: set(g['item_id'].tolist()) for u, g in train.groupby('user_id')}
    
    # Generate recommendations
    recs = np.zeros((len(u2i), K), dtype=int)
    for u in range(len(u2i)):
        s = seen.get(u, set())
        picks = []
        for it in ranked:
            if it in s:
                continue
            picks.append(it)
            if len(picks) == K:
                break
        if len(picks) < K:
            picks += [ranked[0]] * (K - len(picks))
        recs[u] = np.array(picks, dtype=int)
    
    # Save results to the phase1_baselines data/processed directory
    processed_dir = os.path.join(phase1_dir, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    np.save(os.path.join(processed_dir, 'recs_popularity.npy'), recs)
    
    # Save held-out test data
    held = [[] for _ in range(len(u2i))]
    for u, g in test.groupby('user_id'):
        held[u] = g['item_id'].tolist()
    
    with open(os.path.join(processed_dir, 'heldout.json'), 'w') as f:
        json.dump(held, f)
    
    # Save metadata
    meta = {'n_users': len(u2i), 'n_items': len(it2i), 'K': K}
    with open(os.path.join(processed_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    
    print('Saved popularity recommendations')
    print(f'Users: {len(u2i)}, Items: {len(it2i)}, Recommendations per user: {K}')
