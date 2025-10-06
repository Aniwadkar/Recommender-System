
from __future__ import annotations
import argparse, os, numpy as np, pandas as pd
def generate(n_users=1000, n_items=1500, density=0.002, seed=42):
    rng=np.random.default_rng(seed); n=int(n_users*n_items*density)
    item_pop=np.clip(rng.power(1.5, size=n_items), 1e-4, None); item_pop=item_pop/item_pop.sum()
    users=rng.integers(0,n_users,size=n); items=rng.choice(n_items,size=n,p=item_pop)
    ts=rng.integers(1_700_000_000,1_725_000_000,size=n)
    df=pd.DataFrame({'user_id':users,'item_id':items,'timestamp':ts,'weight':1}).drop_duplicates(['user_id','item_id']).reset_index(drop=True)
    return df
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--n_users',type=int,default=1000)
    ap.add_argument('--n_items',type=int,default=1500); ap.add_argument('--density',type=float,default=0.002)
    ap.add_argument('--seed',type=int,default=42); a=ap.parse_args()
    os.makedirs('data/raw', exist_ok=True); generate(a.n_users,a.n_items,a.density,a.seed).to_csv('data/raw/interactions.csv', index=False)
    print('Wrote data/raw/interactions.csv')
