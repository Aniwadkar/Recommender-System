from __future__ import annotations
import argparse, json, os, numpy as np, pandas as pd
OUTDIR='outputs'
def build_label(df: pd.DataFrame):
    df=df.sort_values(['user_id','timestamp']); last=df.groupby('user_id').tail(1)
    return {(r.user_id, r.item_id) for r in last.itertuples(index=False)}
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--interactions', required=True); ap.add_argument('--item-item', required=True); ap.add_argument('--user2item', required=True)
    a=ap.parse_args(); os.makedirs(OUTDIR, exist_ok=True)
    df=pd.read_csv(a.interactions); item_item=json.load(open(a.item_item)); user2item=json.load(open(a.user2item))
    item_pop=df.groupby('item_id').size().to_dict(); labels=build_label(df)
    rows=[]; users=sorted(df['user_id'].unique())
    recent=df.sort_values('timestamp').groupby('user_id')['item_id'].tail(5).groupby(df['user_id']).apply(list).to_dict()
    recent={u:set(v if isinstance(v,list) else [v]) for u,v in recent.items()}
    for u in users:
        cand=set(); cand.update(user2item.get(str(u), [])); cand.update(user2item.get(u, []))
        for it in recent.get(u, []):
            for c in item_item.get(str(it), [])[:20]: cand.add(c)
            for c in item_item.get(it, [])[:20]: cand.add(c)
        if not cand:
            top_pop=sorted(item_pop.items(), key=lambda x:x[1], reverse=True)[:50]; cand={it for it,_ in top_pop}
        for it in cand:
            pop=item_pop.get(int(it), item_pop.get(str(it), 0)); is_recent=int(it in recent.get(u,set()))
            label=int((u,int(it)) in labels or (u,str(it)) in labels); rows.append([u,int(it),pop,is_recent,label])
    feat=pd.DataFrame(rows, columns=['user_id','item_id','item_pop','is_recent','label'])
    feat.to_csv(os.path.join(OUTDIR,'features.csv'), index=False); print('Wrote outputs/features.csv')
