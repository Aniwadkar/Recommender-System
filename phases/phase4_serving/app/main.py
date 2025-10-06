from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import os, json, numpy as np, joblib, pandas as pd

# go up two levels: app -> phase4_serving -> phases
P3_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'phase3_ranking', 'outputs'))

app = FastAPI(title="Recommender System API", version="1.0.0")
P3_DIR = os.path.join(os.path.dirname(__file__), '..', 'phase3_ranking', 'outputs')

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Recommender System API is running!",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc", 
            "ui": "/ui",
            "health": "/health"
        }
    }
@app.get('/health')
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "recommender-api"}
@app.get('/recommend')
def recommend(user_id: int, k: int=10):
    feat_path=os.path.join(P3_DIR,'features.csv'); ranker=os.path.join(P3_DIR,'ranker.joblib')
    if not (os.path.isfile(feat_path) and os.path.isfile(ranker)):
        raise HTTPException(status_code=400, detail='Run Phase 3 first.')
    mdl=joblib.load(ranker); feat=pd.read_csv(feat_path); g=feat[feat['user_id']==user_id]
    if g.empty: return {'user_id': user_id, 'items': []}
    g=g.assign(score=mdl.predict_proba(g[['item_pop','is_recent']])[:,1])
    top=g.sort_values('score', ascending=False).head(k)['item_id'].astype(int).tolist()
    return {'user_id': user_id, 'items': top}
@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    """Simple UI for the recommender system"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Recommender System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { color: #333; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Recommender System API</h1>
            <p>Welcome to the Recommender System API!</p>
            
            <h2>Available Endpoints:</h2>
            <div class="endpoint">
                <strong>GET /</strong> - API information
            </div>
            <div class="endpoint">
                <strong>GET /health</strong> - Health check
            </div>
            <div class="endpoint">
                <strong>GET /docs</strong> - Interactive API documentation
            </div>
            <div class="endpoint">
                <strong>GET /redoc</strong> - Alternative API documentation
            </div>
            
            <h2>Quick Links:</h2>
            <ul>
                <li><a href="/docs">API Documentation (Swagger)</a></li>
                <li><a href="/redoc">API Documentation (ReDoc)</a></li>
                <li><a href="/health">Health Check</a></li>
            </ul>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
