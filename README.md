# 🛍️ Multi-Phase Recommender System

A clean, modular recommender system built with a multi-phase pipeline: baselines → candidates → ranking → serving.

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## 🌟 Features

- **Phase 1**: Baseline models (Popularity, SVD Matrix Factorization)
- **Phase 2**: Candidate generation (Item-Item similarity, User-Item neighborhood)
- **Phase 3**: Ranking with ML models (Logistic Regression, GBDT)
- **Phase 4**: Production serving (FastAPI + Streamlit UI)

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.8
pip install -r requirements.txt
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/recommender-system.git
cd recommender-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the complete pipeline**

```bash
# Phase 1: Build baseline models
cd phases/phase1_baselines
python scripts/run_popularity.py
python scripts/run_mf_svd.py --factors 64 --n_iter 7

# Phase 2: Generate candidates
cd ../phase2_candidates
python scripts/build_item_item.py --topk 100
python scripts/build_user_to_item.py --topk 100

# Phase 3: Train ranking model
cd ../phase3_ranking
python scripts/build_features.py \
  --interactions ../phase1_baselines/data/raw/interactions.csv \
  --item-item ../phase2_candidates/outputs/item_item_candidates.json \
  --user2item ../phase2_candidates/outputs/user_to_item_candidates.json
python scripts/train_ranker.py --model logreg

# Phase 4: Start the API server
cd ../phase4_serving
uvicorn app.main:app --reload --port 8080
```

4. **Launch the Streamlit UI**
```bash
# In a new terminal
cd phases/phase4_serving
streamlit run app/ui.py
```

## 📁 Project Structure

```
recommender-system/
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ .env.example                 # put API_BASE and any secrets (never commit .env)
│
├─ data/                        # (optional) small public demo data
│  ├─ raw/
│  └─ external/
│
├─ phases/
│  ├─ phase1_baselines/
│  │  ├─ data/
│  │  │  ├─ raw/                # interactions.csv, items.csv, etc.
│  │  │  └─ processed/          # recs_popularity.npy, recs_svd.npy, meta.json
│  │  ├─ scripts/               # CLI entry points to run phase 1 pipelines
│  │  │  ├─ run_popularity.py
│  │  │  └─ run_mf_svd.py
│  │  └─ src/
│  │     ├─ data/               # loaders, splitters (leave-last-out), to_csr()
│  │     ├─ models/             # SVD/ALS/baseline implementations
│  │     └─ utils/              # common helpers
│  │
│  ├─ phase2_candidates/
│  │  ├─ outputs/               # item_item_candidates.json, user_to_item_candidates.json
│  │  └─ scripts/
│  │     ├─ build_item_item.py
│  │     └─ build_user_to_item.py
│  │
│  └─ phase3_ranking/
│     ├─ outputs/               # features.csv, ranker.joblib
│     └─ scripts/
│        ├─ build_features.py
│        ├─ train_ranker.py
│        └─ eval_ranker.py
│
├─ api/                         # FastAPI service (Phase 4: serving)
│  ├─ main.py                   # FastAPI app (includes /health and /recommend)
│  ├─ routers/                  # route modules if you split endpoints
│  ├─ services/                 # business logic that calls models/files
│  ├─ models/                   # pydantic schemas
│  └─ utils/                    # config/env, logging, paths
│
├─ app/                         # Streamlit UI
│  └─ ui.py
│
├─ config/
│  ├─ logging.yaml              # central logging config (optional)
│  └─ paths.yaml                # relative paths if you want one source of truth
│
├─ tests/                       # unit tests (pytest)
│  ├─ test_phase1.py
│  ├─ test_phase2.py
│  ├─ test_phase3.py
│  └─ test_api.py
│
└─ scripts/                     # cross-phase utilities or convenience CLIs
   └─ bootstrap_demo.py         # optional: generate tiny demo artifacts on fresh clone
```

## 🛠️ Usage

### Web Interface

1. **Home**: Overview and quick start guide
2. **Get Recommendations**: Test different recommendation methods
3. **API Status**: Check API health and endpoints
4. **Phase Status**: Monitor pipeline completion

### API Endpoints

- `GET /health` - Health check
- `GET /recommend?user_id=<id>&k=<k>` - Get recommendations

Example:
```bash
curl "http://localhost:8080/recommend?user_id=123&k=10"
```

### Programmatic Usage

```python
import requests

# Get recommendations via API
response = requests.get("http://localhost:8080/recommend", 
                       params={"user_id": 123, "k": 10})
recommendations = response.json()["items"]
```

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=127.0.0.1
API_PORT=8080

# Model Parameters
SVD_FACTORS=64
SVD_ITERATIONS=7
CANDIDATE_TOPK=100
```

### Model Selection

Available models in Phase 3:
- `logreg`: Logistic Regression
- `gbdt`: Gradient Boosting Decision Trees

```bash
python scripts/train_ranker.py --model gbdt
```

## 📊 Performance

| Phase | Component | Training Time | Memory Usage |
|-------|-----------|---------------|--------------|
| 1 | Popularity | ~1s | Low |
| 1 | SVD | ~30s | Medium |
| 2 | Item-Item | ~5min | Medium |
| 2 | User-Item | ~2min | Low |
| 3 | Ranking | ~1min | Low |

## 🚀 Deployment

### Local Development

```bash
# Auto-restart on file changes
streamlit run app/ui.py --server.fileWatcherType polling
```

### Production Options

1. **Streamlit Cloud** (Recommended)
   - Push to GitHub
   - Deploy at [share.streamlit.io](https://share.streamlit.io)

2. **Docker**
```bash
docker build -t recommender-system .
docker run -p 8501:8501 recommender-system
```

3. **Cloud Platforms**
   - Heroku, Railway, Render
   - Google Cloud Run, AWS EC2

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔍 Troubleshooting

### Common Issues

**API not responding**
```bash
# Check if API is running
curl http://localhost:8080/health

# Restart API server
cd phases/phase4_serving
uvicorn app.main:app --reload --port 8080
```

**Missing model files**
```bash
# Check phase status in the UI or run:
python -c "from pathlib import Path; print('Phase 1:', Path('phases/phase1_baselines/data/processed/recs_popularity.npy').exists())"
```

**Dependencies issues**
```bash
pip install --upgrade -r requirements.txt
```

## 📧 Support

For questions and support:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the phase status in the web UI

---

**Built with ❤️ using Streamlit, FastAPI, and scikit-learn**
