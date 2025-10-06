# Phase 1 — Baselines (Popularity + SVD)
```bash
cd phases/phase1_baselines
python scripts/make_synthetic.py --n_users 800 --n_items 1200 --density 0.003
python scripts/run_popularity.py
python scripts/run_mf_svd.py --factors 64 --n_iter 7
python scripts/evaluate.py --k 10
```
