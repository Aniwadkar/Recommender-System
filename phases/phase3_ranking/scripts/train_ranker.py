
from __future__ import annotations
import argparse, os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
OUTDIR='outputs'
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--model', choices=['logreg','gbdt'], default='logreg'); a=ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True); df=pd.read_csv(os.path.join(OUTDIR,'features.csv'))
    X=df[['item_pop','is_recent']]; y=df['label']
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    mdl=LogisticRegression(max_iter=1000) if a.model=='logreg' else GradientBoostingClassifier()
    mdl.fit(Xtr,ytr); prob=mdl.predict_proba(Xte)[:,1]; print('AUC:', round(roc_auc_score(yte, prob),4))
    joblib.dump(mdl, os.path.join(OUTDIR,'ranker.joblib')); print('Saved outputs/ranker.joblib')
