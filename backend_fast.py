#!/usr/bin/env python3
"""
Fast Enhanced RAG Pipeline Backend (No Agent Loading)
====================================================

Lightweight version that skips agent initialization for fast startup.
Perfect for frontend development and quick access to the dashboard.
"""

import sys
import os
import json
import logging
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import duckdb
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def safe_df_to_dict(df):
    """Convert DataFrame to dict with proper NaN handling for JSON serialization"""
    if df.empty:
        return []
    
    # Replace NaN values with None for JSON compatibility
    df_clean = df.replace({np.nan: None, pd.NA: None})
    
    # Convert numpy types to Python types
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            continue
        elif pd.api.types.is_integer_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].astype('Int64')
        elif pd.api.types.is_float_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].astype('float64')
    
    df_clean = df_clean.replace({np.nan: None, pd.NA: None})
    return df_clean.to_dict("records")

# Pydantic Models
class PredictionRequest(BaseModel):
    setup_id: str
    include_similar: bool = True
    similarity_limit: int = 5

class ScanRequest(BaseModel):
    start_date: str
    end_date: str
    min_probability: float = 0.5
    limit: int = 100

# Global variables
DB_PATH = "data/sentiment_system.duckdb"

# Initialize FastAPI app
app = FastAPI(
    title="Fast Enhanced RAG Pipeline API",
    description="Lightweight API for frontend development - no agent loading",
    version="2.0.0-fast"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root route to serve the frontend
@app.get("/")
async def serve_frontend():
    """Serve the main frontend application"""
    return FileResponse("index.html")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Fast health check without agent initialization"""
    try:
        # Test database connection
        conn = duckdb.connect(DB_PATH)
        test_result = conn.execute("SELECT COUNT(*) FROM labels").fetchone()
        conn.close()
        
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "total_setups": test_result[0] if test_result else 0,
            "mode": "fast_startup",
            "components": {
                "database": True,
                "frontend": True,
                "agents": "not_loaded_fast_mode"
            }
        }
        
        return status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )

# Setup management
@app.get("/api/setups")
async def get_setups(
    limit: int = Query(100, ge=1, le=1000),
    has_labels: bool = Query(True),
    random_sample: bool = Query(False)
):
    """Get available setups"""
    try:
        conn = duckdb.connect(DB_PATH)
        
        base_query = """
        SELECT DISTINCT l.setup_id, l.outperformance_10d,
               EXTRACT(YEAR FROM l.setup_id) as year,
               SUBSTR(l.setup_id, 1, POSITION('_' IN l.setup_id) - 1) as ticker
        FROM labels l
        """
        
        if has_labels:
            base_query += " WHERE l.outperformance_10d IS NOT NULL"
        
        if random_sample:
            base_query += " ORDER BY RANDOM()"
        else:
            base_query += " ORDER BY l.setup_id DESC"
        
        base_query += f" LIMIT {limit}"
        
        result = conn.execute(base_query).fetchall()
        conn.close()
        
        setups = [
            {
                "setup_id": row[0],
                "outperformance_10d": row[1],
                "year": row[2],
                "ticker": row[3]
            }
            for row in result
        ]
        
        return {"setups": setups, "count": len(setups)}
    
    except Exception as e:
        logger.error(f"Error fetching setups: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mock prediction endpoint for fast mode
@app.post("/api/predict")
async def make_prediction(request: PredictionRequest):
    """Mock prediction for fast mode"""
    setup_id = request.setup_id
    
    # Create realistic mock prediction
    mock_prediction = {
        "predicted_outperformance_10d": random.uniform(-8.0, 12.0),
        "confidence_score": random.uniform(0.4, 0.9),
        "prediction_class": random.choice(["POSITIVE", "NEGATIVE", "NEUTRAL"]),
        "reasoning": f"Mock prediction for {setup_id} - agents not loaded in fast mode",
        "agent_name": "fast_mode_mock"
    }
    
    # Get actual label if available
    try:
        conn = duckdb.connect(DB_PATH)
        result = conn.execute("SELECT outperformance_10d FROM labels WHERE setup_id = ?", [setup_id]).fetchone()
        conn.close()
        actual_label = result[0] if result else None
    except:
        actual_label = None
    
    # Mock similar setups if requested
    similar_setups = []
    if request.include_similar:
        try:
            conn = duckdb.connect(DB_PATH)
            similar_query = """
            SELECT setup_id, outperformance_10d
            FROM labels 
            WHERE setup_id != ? AND outperformance_10d IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
            """
            results = conn.execute(similar_query, [setup_id, request.similarity_limit]).fetchall()
            conn.close()
            
            for row in results:
                similar_setups.append({
                    "setup_id": row[0],
                    "similarity_score": random.uniform(0.7, 0.95),
                    "outperformance_10d": row[1],
                    "prediction_class": "POSITIVE" if row[1] > 0 else "NEGATIVE" if row[1] < 0 else "NEUTRAL"
                })
        except:
            pass
    
    return {
        "setup_id": setup_id,
        "prediction": mock_prediction,
        "similar_setups": similar_setups,
        "actual_label": actual_label,
        "timestamp": datetime.now().isoformat(),
        "mode": "fast_mock"
    }

# Data endpoints with real data
@app.get("/api/setup/{setup_id}/fundamentals")
async def get_setup_fundamentals(setup_id: str):
    """Get fundamentals data for a setup"""
    try:
        conn = duckdb.connect(DB_PATH)
        
        queries = {
            "structured_metrics": "SELECT * FROM financial_features WHERE setup_id = ?",
            "llm_features": "SELECT * FROM fundamentals_features WHERE setup_id = ?"
        }
        
        result = {"setup_id": setup_id}
        
        for data_type, query in queries.items():
            try:
                data = conn.execute(query, [setup_id]).fetchall()
                if data:
                    columns = [desc[0] for desc in conn.description]
                    result[data_type] = dict(zip(columns, data[0]))
                else:
                    result[data_type] = {}
            except Exception as e:
                logger.warning(f"Could not fetch {data_type}: {e}")
                result[data_type] = {}
        
        conn.close()
        return result
        
    except Exception as e:
        logger.error(f"Error fetching fundamentals for {setup_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/setup/{setup_id}/news")
async def get_setup_news(setup_id: str):
    """Get news data for a setup"""
    try:
        conn = duckdb.connect(DB_PATH)
        query = "SELECT * FROM news_features WHERE setup_id = ?"
        result = conn.execute(query, [setup_id]).fetchall()
        
        if result:
            columns = [desc[0] for desc in conn.description]
            news_data = dict(zip(columns, result[0]))
        else:
            news_data = {}
        
        conn.close()
        return {"setup_id": setup_id, "news_analysis": news_data}
        
    except Exception as e:
        logger.error(f"Error fetching news for {setup_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/setup/{setup_id}/userposts")
async def get_setup_userposts(setup_id: str):
    """Get userposts data for a setup"""
    try:
        conn = duckdb.connect(DB_PATH)
        query = "SELECT * FROM userposts_features WHERE setup_id = ?"
        result = conn.execute(query, [setup_id]).fetchall()
        
        if result:
            columns = [desc[0] for desc in conn.description]
            userposts_data = dict(zip(columns, result[0]))
        else:
            userposts_data = {}
        
        conn.close()
        return {"setup_id": setup_id, "userposts_analysis": userposts_data}
        
    except Exception as e:
        logger.error(f"Error fetching userposts for {setup_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Similar setups endpoint
@app.get("/api/setup/{setup_id}/similar")
async def find_similar_setups(
    setup_id: str,
    limit: int = Query(10, ge=1, le=50),
    min_similarity: float = Query(0.7, ge=0.0, le=1.0)
):
    """Find similar setups (mock for fast mode)"""
    try:
        conn = duckdb.connect(DB_PATH)
        
        similar_query = """
        SELECT setup_id, outperformance_10d
        FROM labels 
        WHERE setup_id != ? AND outperformance_10d IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
        """
        
        results = conn.execute(similar_query, [setup_id, limit]).fetchall()
        conn.close()
        
        similar_setups = []
        for row in results:
            similar_setups.append({
                "setup_id": row[0],
                "similarity_score": random.uniform(min_similarity, 1.0),
                "outperformance_10d": row[1],
                "prediction_class": "POSITIVE" if row[1] > 0 else "NEGATIVE" if row[1] < 0 else "NEUTRAL"
            })
        
        similar_setups.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "setup_id": setup_id,
            "similar_setups": similar_setups,
            "count": len(similar_setups),
            "mode": "fast_mock"
        }
        
    except Exception as e:
        logger.error(f"Error finding similar setups for {setup_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model performance endpoint
@app.get("/api/model-performance")
async def get_model_performance():
    """Mock model performance for fast mode"""
    performance_data = {
        "models": [
            {"name": "Random Forest", "precision": 0.426, "recall": 0.383, "f1_score": 0.302, "auc": 0.678},
            {"name": "Logistic Regression", "precision": 0.326, "recall": 0.289, "f1_score": 0.246, "auc": 0.567},
            {"name": "XGBoost", "precision": 0.411, "recall": 0.367, "f1_score": 0.285, "auc": 0.645},
            {"name": "LightGBM", "precision": 0.403, "recall": 0.359, "f1_score": 0.278, "auc": 0.634},
            {"name": "Ensemble", "precision": 0.369, "recall": 0.383, "f1_score": 0.302, "auc": 0.702}
        ],
        "best_model": "Ensemble",
        "evaluation_date": datetime.now().isoformat(),
        "mode": "fast_mock"
    }
    
    return performance_data

# Visualizations endpoint
@app.get("/api/visualizations")
async def get_visualizations():
    """Get visualization files"""
    try:
        viz_dirs = ["visualizations", "ml/analysis"]
        categories = {}
        
        for base_dir in viz_dirs:
            viz_path = Path(base_dir)
            if not viz_path.exists():
                continue
            
            for category_dir in viz_path.iterdir():
                if category_dir.is_dir():
                    files = []
                    for file_path in category_dir.iterdir():
                        if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.txt']:
                            files.append({
                                "filename": file_path.name,
                                "size": file_path.stat().st_size,
                                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                                "type": "image" if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg'] else "text"
                            })
                    
                    if files:
                        categories[category_dir.name] = files
        
        return {"categories": categories}
        
    except Exception as e:
        logger.error(f"Error getting visualizations: {e}")
        return {"categories": {}}

# WebSocket for live prediction (mock mode)
@app.websocket("/ws/live-prediction")
async def live_prediction_websocket(websocket: WebSocket):
    """Mock live prediction theater for fast mode"""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            if request_data.get("action") == "start_prediction":
                count = min(int(request_data.get("count", 5)), 20)
                await run_mock_theater(websocket, count)
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

async def run_mock_theater(websocket: WebSocket, count: int):
    """Run mock prediction theater"""
    import asyncio
    
    # Get real setup IDs
    try:
        conn = duckdb.connect(DB_PATH)
        result = conn.execute("SELECT setup_id FROM labels WHERE outperformance_10d IS NOT NULL ORDER BY RANDOM() LIMIT ?", [count]).fetchall()
        setup_ids = [row[0] for row in result]
        conn.close()
    except:
        setup_ids = [f"MOCK_{i}_2024-01-01" for i in range(count)]
    
    # Theater start
    await websocket.send_text(json.dumps({
        "type": "theater_start",
        "message": f"🎭 Starting Mock Agent Prediction Theater for {count} setups...",
        "total_setups": count,
        "timestamp": datetime.now().isoformat()
    }))
    
    await asyncio.sleep(0.5)
    
    await websocket.send_text(json.dumps({
        "type": "setups_selected",
        "message": f"📋 Selected {count} setups for mock prediction",
        "setups": setup_ids,
        "timestamp": datetime.now().isoformat()
    }))
    
    # Process each setup
    for i, setup_id in enumerate(setup_ids, 1):
        await websocket.send_text(json.dumps({
            "type": "setup_start",
            "setup_index": i,
            "total_setups": count,
            "setup_id": setup_id,
            "message": f"🎯 Processing Setup {i}/{count}: {setup_id}",
            "timestamp": datetime.now().isoformat()
        }))
        
        # Mock agent predictions
        agents = ["fundamentals", "news", "analyst_recommendations", "userposts"]
        agent_results = {}
        
        for agent in agents:
            await websocket.send_text(json.dumps({
                "type": "step_progress",
                "step": "embedding_generation",
                "agent": agent,
                "setup_id": setup_id,
                "message": f"🧠 {agent.title()} Agent: Generating embeddings...",
                "timestamp": datetime.now().isoformat()
            }))
            
            await asyncio.sleep(0.3)
            
            await websocket.send_text(json.dumps({
                "type": "step_progress",
                "step": "agent_prediction",
                "agent": agent,
                "setup_id": setup_id,
                "message": f"🔮 {agent.title()} Agent: Making prediction...",
                "timestamp": datetime.now().isoformat()
            }))
            
            await asyncio.sleep(0.3)
            
            # Mock result
            mock_result = {
                "predicted_outperformance_10d": random.uniform(-5.0, 8.0),
                "confidence_score": random.uniform(0.4, 0.9),
                "prediction_class": random.choice(["POSITIVE", "NEGATIVE", "NEUTRAL"]),
                "reasoning": f"Mock {agent} prediction",
                "agent_name": agent
            }
            
            agent_results[agent] = mock_result
            
            await websocket.send_text(json.dumps({
                "type": "agent_prediction_complete",
                "agent": agent,
                "setup_id": setup_id,
                "result": mock_result,
                "message": f"✅ {agent.title()} Agent: Prediction complete",
                "timestamp": datetime.now().isoformat()
            }))
        
        # Mock ensemble
        await websocket.send_text(json.dumps({
            "type": "step_progress",
            "step": "ensemble_prediction",
            "setup_id": setup_id,
            "message": "🎯 Creating ensemble prediction from all agents...",
            "timestamp": datetime.now().isoformat()
        }))
        
        await asyncio.sleep(0.5)
        
        ensemble_result = {
            "predicted_outperformance_10d": np.mean([r["predicted_outperformance_10d"] for r in agent_results.values()]),
            "confidence_score": np.mean([r["confidence_score"] for r in agent_results.values()]),
            "prediction_class": random.choice(["POSITIVE", "NEGATIVE", "NEUTRAL"]),
            "reasoning": "Mock ensemble prediction",
            "agent_count": len(agent_results)
        }
        
        # Get actual label
        try:
            conn = duckdb.connect(DB_PATH)
            result = conn.execute("SELECT outperformance_10d FROM labels WHERE setup_id = ?", [setup_id]).fetchone()
            conn.close()
            actual_label = result[0] if result else None
        except:
            actual_label = None
        
        await websocket.send_text(json.dumps({
            "type": "setup_complete",
            "setup_index": i,
            "setup_id": setup_id,
            "agent_predictions": agent_results,
            "ensemble_prediction": ensemble_result,
            "actual_label": actual_label,
            "message": f"🎉 Setup {i}/{count} complete: {setup_id}",
            "timestamp": datetime.now().isoformat()
        }))
    
    # Theater complete
    await websocket.send_text(json.dumps({
        "type": "theater_complete",
        "message": f"🎭 Mock Agent Prediction Theater complete! Processed {count} setups.",
        "total_setups": count,
        "timestamp": datetime.now().isoformat()
    }))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Fast Enhanced RAG Pipeline Backend...")
    print("⚡ Agents disabled for fast startup - mock predictions only")
    uvicorn.run(app, host="0.0.0.0", port=8000)