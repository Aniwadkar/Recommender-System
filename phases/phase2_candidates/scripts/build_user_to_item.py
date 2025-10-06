from __future__ import annotations
import argparse, os, json, numpy as np, pandas as pd
OUTDIR='outputs'
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--interactions', default='../phase1_baselines/data/raw/interactions.csv')
    ap.add_argument('--topk', type=int, default=100); a=ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True); df=pd.read_csv(a.interactions)
    users=sorted(df['user_id'].unique()); items=sorted(df['item_id'].unique())
    u2i={u:i for i,u in enumerate(users)}; it2i={m:i for i,m in enumerate(items)}
    user_items=[set() for _ in users]
    for _,r in df.iterrows(): user_items[u2i[r['user_id']]].add(it2i[r['item_id']])
    def jaccard(a,b): inter=len(a & b); union=len(a | b); return inter/union if union else 0.0
    user_recs={}
    for u_idx,a_set in enumerate(user_items):
        sims=[(v_idx, jaccard(a_set, b)) for v_idx,b in enumerate(user_items) if v_idx!=u_idx]
        sims.sort(key=lambda x: x[1], reverse=True); neighbors=[idx for idx,s in sims[:20] if s>0]
        bag={}
        for n in neighbors:
            for it in user_items[n]:
                if it not in a_set: bag[it]=bag.get(it,0)+1
        top=sorted(bag.items(), key=lambda x:x[1], reverse=True)[:a.topk]
        user_recs[users[u_idx]]=[items[it] for it,_ in top]
    # Convert numpy types to Python types for JSON serialization
    user_recs_serializable = {
        int(k): [int(item) for item in v] if isinstance(v, list) else int(v) 
        for k, v in user_recs.items()
    }
    json.dump(user_recs_serializable, open(os.path.join(OUTDIR,'user_to_item_candidates.json'),'w'))
    print('Wrote outputs/user_to_item_candidates.json')
