from __future__ import annotations
import argparse, os, json, numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
OUTDIR='outputs'
def build_item_item(df: pd.DataFrame, topk: int=100):
    users=sorted(df['user_id'].unique()); items=sorted(df['item_id'].unique())
    u2i={u:i for i,u in enumerate(users)}; it2i={m:i for i,m in enumerate(items)}
    rows=df['user_id'].map(u2i).to_numpy(); cols=df['item_id'].map(it2i).to_numpy(); data=np.ones(len(df), dtype=np.float32)
    mat=csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
    sims=cosine_similarity(mat.T, dense_output=False)
    cands={}
    for i in range(sims.shape[0]):
        row=sims[i].toarray().ravel(); row[i]=-1
        top_idx=np.argpartition(-row, range(min(topk, len(row)-1)))[:topk]
        top_sorted=top_idx[np.argsort(-row[top_idx])]
        cands[items[i]]=[items[j] for j in top_sorted if row[j]>0][:topk]
    return cands
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--interactions', default='../phase1_baselines/data/raw/interactions.csv')
    ap.add_argument('--topk', type=int, default=100); a=ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True); df=pd.read_csv(a.interactions); c=build_item_item(df, topk=a.topk)
    # Convert numpy types to Python types for JSON serialization
    c_serializable = {int(k): [int(item) for item in v] if isinstance(v, list) else int(v) for k, v in c.items()}
    json.dump(c_serializable, open(os.path.join(OUTDIR,'item_item_candidates.json'),'w'))
    print('Wrote outputs/item_item_candidates.json')
