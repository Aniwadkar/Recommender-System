from __future__ import annotations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse, json, numpy as np
from src.utils.metrics import recall_at_k, ndcg_at_k, coverage
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--k',type=int,default=10)
    ap.add_argument('--which',choices=['popularity','svd','both'], default='both'); a=ap.parse_args()
    held=json.load(open('data/processed/heldout.json')); meta=json.load(open('data/processed/meta.json'))
    if a.which in ('popularity','both'):
        rec=np.load('data/processed/recs_popularity.npy'); 
        print('== Popularity =='); print('Recall@K:', round(recall_at_k(rec, held, k=a.k),4))
        print('NDCG@K:', round(ndcg_at_k(rec, held, k=a.k),4)); print('Coverage:', round(coverage(rec, meta['n_items']),4))
    if a.which in ('svd','both'):
        rec=np.load('data/processed/recs_svd.npy'); 
        print('== SVD =='); print('Recall@K:', round(recall_at_k(rec, held, k=a.k),4))
        print('NDCG@K:', round(ndcg_at_k(rec, held, k=a.k),4)); print('Coverage:', round(coverage(rec, meta['n_items']),4))
