
from __future__ import annotations
import argparse, json, os, numpy as np, pandas as pd
def recall_at_k(recommended, heldout, k=10):
    K=min(k,recommended.shape[1]); hits=0; total=0
    for recs, gt in zip(recommended[:,:K], heldout):
        if not gt: continue
        total+=1; hits += len(set(recs.tolist()) & set(gt))
    return hits/total if total else 0.0
def ndcg_at_k(recommended, heldout, k=10):
    import math; K=min(k,recommended.shape[1])
    def dcg(recs, gtset):
        s=0.0
        for i,item in enumerate(recs[:K], start=1):
            if item in gtset: s+=1.0/math.log2(i+1)
        return s
    vals=[]
    for recs, gt in zip(recommended, heldout):
        if not gt: continue
        gtset=set(gt); ideal=sum(1.0/np.log2(i+1) for i in range(1, min(len(gtset),K)+1))
        vals.append(dcg(recs, gtset)/(ideal if ideal>0 else 1.0))
    return float(np.mean(vals)) if vals else 0.0
def coverage(recommended, n_items): return np.unique(recommended).size/float(n_items)
def load_heldout(p='../phase1_baselines/data/processed'):
    return json.load(open(os.path.join(p,'heldout.json'))), json.load(open(os.path.join(p,'meta.json')))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--k', type=int, default=10); a=ap.parse_args()
    import joblib
    feat=pd.read_csv('outputs/features.csv'); mdl=joblib.load('outputs/ranker.joblib')
    feat['score']=mdl.predict_proba(feat[['item_pop','is_recent']])[:,1]
    k=a.k; users=feat['user_id'].unique(); recs=np.zeros((len(users),k), dtype=int); idx={u:i for i,u in enumerate(users)}
    for u,g in feat.groupby('user_id'):
        top=g.sort_values('score', ascending=False).head(k)['item_id'].tolist()
        if len(top)<k: top += [top[0] if top else 0]*(k-len(top))
        recs[idx[u]]=np.array(top[:k], dtype=int)
    held, meta = load_heldout()
    print('== Ranker =='); print('Recall@K:', round(recall_at_k(recs, held, k=k),4))
    print('NDCG@K:', round(ndcg_at_k(recs, held, k=k),4)); print('Coverage:', round(coverage(recs, meta['n_items']),4))
