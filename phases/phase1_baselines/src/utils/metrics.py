
from __future__ import annotations
import numpy as np
def recall_at_k(recommended: np.ndarray, heldout: list[list[int]], k: int=10) -> float:
    K = min(k, recommended.shape[1]); hits=0; total=0
    for recs, gt in zip(recommended[:, :K], heldout):
        if not gt: continue
        total += 1; hits += len(set(recs.tolist()) & set(gt))
    return hits/total if total else 0.0
def ndcg_at_k(recommended: np.ndarray, heldout: list[list[int]], k: int=10) -> float:
    K=min(k,recommended.shape[1])
    def dcg(recs, gtset):
        s=0.0
        for i,item in enumerate(recs[:K], start=1):
            if item in gtset: s += 1.0/np.log2(i+1)
        return s
    vals=[]
    for recs, gt in zip(recommended, heldout):
        if not gt: continue
        gtset=set(gt)
        ideal=sum(1.0/np.log2(i+1) for i in range(1, min(len(gtset), K)+1))
        vals.append(dcg(recs, gtset)/(ideal if ideal>0 else 1.0))
    return float(np.mean(vals)) if vals else 0.0
def coverage(recommended: np.ndarray, n_items: int) -> float:
    return np.unique(recommended).size / float(n_items)
