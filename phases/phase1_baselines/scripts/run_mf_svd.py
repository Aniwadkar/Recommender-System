from __future__ import annotations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse, os, json, numpy as np, pandas as pd
from src.data.splits import leave_last_one_out, build_mappings, to_csr
from src.models.baselines.mf_svd import train_svd, recommend_svd
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--factors',type=int,default=64)
    ap.add_argument('--n_iter',type=int,default=7); ap.add_argument('--k',type=int,default=10); a=ap.parse_args()
    df=pd.read_csv('data/raw/interactions.csv'); df_i,u2i,it2i=build_mappings(df); train,test=leave_last_one_out(df_i)
    from scipy import sparse
    csr=to_csr(train, len(u2i), len(it2i))
    U,V=train_svd(csr, factors=a.factors, n_iter=a.n_iter)
    recs=recommend_svd(U,V,csr,k=a.k)
    os.makedirs('data/processed', exist_ok=True); np.save('data/processed/recs_svd.npy', recs)
    held=[[] for _ in range(len(u2i))]; 
    for u,g in test.groupby('user_id'): held[u]=g['item_id'].tolist()
    json.dump(held, open('data/processed/heldout.json','w'))
    json.dump({'n_users': len(u2i), 'n_items': len(it2i), 'K': a.k}, open('data/processed/meta.json','w'))
    print('Saved SVD recommendations')
