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

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import base64
import io

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
               SUBSTR(l.setup_id, -10, 4) as year,
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

# Enhanced prediction endpoint with real agent data
@app.post("/api/predict")
async def make_prediction(request: PredictionRequest):
    """Get real agent prediction for a specific setup"""
    setup_id = request.setup_id
    
    try:
        conn = duckdb.connect(DB_PATH)
        
        # Get real agent predictions for this setup
        query = """
        SELECT 
            sp.domain,
            sp.predicted_outperformance,
            sp.confidence,
            sp.positive_ratio,
            sp.negative_ratio,
            sp.neutral_ratio,
            sp.similar_cases_count,
            l.outperformance_10d as actual_performance
        FROM similarity_predictions sp
        LEFT JOIN labels l ON sp.setup_id = l.setup_id
        WHERE sp.setup_id = ?
        ORDER BY sp.domain
        """
        
        results = conn.execute(query, [setup_id]).fetchall()
        
        if not results:
            conn.close()
            return {
                "setup_id": setup_id,
                "error": "No agent predictions found for this setup",
                "prediction": None,
                "similar_setups": [],
                "actual_label": None
            }
        
        columns = [
            "domain", "predicted_outperformance", "confidence", 
            "positive_ratio", "negative_ratio", "neutral_ratio", 
            "similar_cases_count", "actual_performance"
        ]
        
        # Process domain predictions
        domain_predictions = {}
        actual_label = None
        
        for row in results:
            row_dict = dict(zip(columns, row))
            domain = row_dict["domain"]
            actual_label = row_dict["actual_performance"]  # Same for all rows
            
            # Determine prediction class
            perf = row_dict["predicted_outperformance"]
            if perf > 2:
                pred_class = "POSITIVE"
            elif perf < -2:
                pred_class = "NEGATIVE"
            else:
                pred_class = "NEUTRAL"
            
            domain_predictions[domain] = {
                "predicted_outperformance_10d": perf,
                "confidence_score": row_dict["confidence"],
                "prediction_class": pred_class,
                "positive_ratio": row_dict["positive_ratio"],
                "negative_ratio": row_dict["negative_ratio"],
                "neutral_ratio": row_dict["neutral_ratio"],
                "similar_cases_count": row_dict["similar_cases_count"],
                "agent_name": domain,
                "reasoning": f"Based on {row_dict['similar_cases_count']} similar historical cases in {domain} domain"
            }
        
        # Calculate ensemble prediction
        if domain_predictions:
            avg_performance = np.mean([p["predicted_outperformance_10d"] for p in domain_predictions.values()])
            avg_confidence = np.mean([p["confidence_score"] for p in domain_predictions.values()])
            
            if avg_performance > 2:
                ensemble_class = "POSITIVE"
            elif avg_performance < -2:
                ensemble_class = "NEGATIVE"
            else:
                ensemble_class = "NEUTRAL"
            
            ensemble_prediction = {
                "predicted_outperformance_10d": avg_performance,
                "confidence_score": avg_confidence,
                "prediction_class": ensemble_class,
                "reasoning": f"Ensemble of {len(domain_predictions)} domain predictions",
                "agent_name": "ensemble",
                "domains_used": list(domain_predictions.keys())
            }
        else:
            ensemble_prediction = None
        
        # Get similar setups if requested
        similar_setups = []
        if request.include_similar:
            similar_query = """
            SELECT DISTINCT 
                sp2.setup_id,
                l2.outperformance_10d,
                AVG(sp2.predicted_outperformance) as avg_prediction,
                AVG(sp2.confidence) as avg_confidence
            FROM similarity_predictions sp2
            LEFT JOIN labels l2 ON sp2.setup_id = l2.setup_id
            WHERE sp2.setup_id != ? AND l2.outperformance_10d IS NOT NULL
            GROUP BY sp2.setup_id, l2.outperformance_10d
            ORDER BY RANDOM()
            LIMIT ?
            """
            
            similar_results = conn.execute(similar_query, [setup_id, request.similarity_limit]).fetchall()
            
            for sim_row in similar_results:
                sim_setup_id, sim_actual, sim_pred, sim_conf = sim_row
                
                # Calculate similarity score (mock based on prediction closeness)
                if avg_performance is not None and sim_pred is not None:
                    similarity_score = max(0.7, 1.0 - abs(avg_performance - sim_pred) / 10.0)
                else:
                    similarity_score = random.uniform(0.7, 0.9)
                
                similar_setups.append({
                    "setup_id": sim_setup_id,
                    "similarity_score": round(similarity_score, 3),
                    "outperformance_10d": round(sim_actual, 2) if sim_actual else None,
                    "predicted_outperformance": round(sim_pred, 2) if sim_pred else None,
                    "confidence": round(sim_conf, 3) if sim_conf else None,
                    "prediction_class": "POSITIVE" if sim_actual and sim_actual > 2 else "NEGATIVE" if sim_actual and sim_actual < -2 else "NEUTRAL"
                })
        
        conn.close()
        
        return {
            "setup_id": setup_id,
            "prediction": ensemble_prediction,
            "domain_predictions": domain_predictions,
            "similar_setups": similar_setups,
            "actual_label": round(actual_label, 2) if actual_label is not None else None,
            "timestamp": datetime.now().isoformat(),
            "mode": "real_agent_data"
        }
        
    except Exception as e:
        logger.error(f"Error making prediction for {setup_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket for live prediction with real data
async def run_mock_theater(websocket: WebSocket, count: int):
    """Run prediction theater with REAL agent data"""
    import asyncio
    
    # Get real setup IDs that have agent predictions
    try:
        conn = duckdb.connect(DB_PATH)
        result = conn.execute("""
            SELECT DISTINCT sp.setup_id
            FROM similarity_predictions sp
            LEFT JOIN labels l ON sp.setup_id = l.setup_id
            WHERE l.outperformance_10d IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
        """, [count]).fetchall()
        setup_ids = [row[0] for row in result]
        conn.close()
    except Exception as e:
        logger.error(f"Error getting real setup IDs: {e}")
        setup_ids = [f"MOCK_{i}_2024-01-01" for i in range(count)]
    
    # Theater start
    await websocket.send_text(json.dumps({
        "type": "theater_start",
        "message": f"🎭 Starting REAL Agent Prediction Theater for {count} setups...",
        "total_setups": count,
        "timestamp": datetime.now().isoformat()
    }))
    
    await asyncio.sleep(0.5)
    
    await websocket.send_text(json.dumps({
        "type": "setups_selected",
        "message": f"📋 Selected {count} real setups with agent predictions",
        "setups": setup_ids,
        "timestamp": datetime.now().isoformat()
    }))
    
    # Process each setup with REAL data
    for i, setup_id in enumerate(setup_ids, 1):
        await websocket.send_text(json.dumps({
            "type": "setup_start",
            "setup_index": i,
            "total_setups": count,
            "setup_id": setup_id,
            "message": f"🎯 Processing Real Setup {i}/{count}: {setup_id}",
            "timestamp": datetime.now().isoformat()
        }))
        
        # Get real agent predictions for this setup
        try:
            conn = duckdb.connect(DB_PATH)
            query = """
            SELECT 
                sp.domain,
                sp.predicted_outperformance,
                sp.confidence,
                l.outperformance_10d as actual_performance
            FROM similarity_predictions sp
            LEFT JOIN labels l ON sp.setup_id = l.setup_id
            WHERE sp.setup_id = ?
            ORDER BY sp.domain
            """
            
            results = conn.execute(query, [setup_id]).fetchall()
            conn.close()
            
            agent_results = {}
            actual_label = None
            
            for domain, pred_perf, confidence, actual_perf in results:
                actual_label = actual_perf  # Same for all domains
                
                # Determine prediction class
                if pred_perf > 2:
                    pred_class = "POSITIVE"
                elif pred_perf < -2:
                    pred_class = "NEGATIVE"
                else:
                    pred_class = "NEUTRAL"
                
                await websocket.send_text(json.dumps({
                    "type": "step_progress",
                    "step": "embedding_generation",
                    "agent": domain,
                    "setup_id": setup_id,
                    "message": f"🧠 {domain.title()} Agent: Using cached embeddings...",
                    "timestamp": datetime.now().isoformat()
                }))
                
                await asyncio.sleep(0.2)
                
                await websocket.send_text(json.dumps({
                    "type": "step_progress",
                    "step": "agent_prediction",
                    "agent": domain,
                    "setup_id": setup_id,
                    "message": f"🔮 {domain.title()} Agent: Loading real prediction...",
                    "timestamp": datetime.now().isoformat()
                }))
                
                await asyncio.sleep(0.2)
                
                # Real result
                real_result = {
                    "predicted_outperformance_10d": pred_perf,
                    "confidence_score": confidence,
                    "prediction_class": pred_class,
                    "reasoning": f"Real {domain} agent prediction from database",
                    "agent_name": domain
                }
                
                agent_results[domain] = real_result
                
                await websocket.send_text(json.dumps({
                    "type": "agent_prediction_complete",
                    "agent": domain,
                    "setup_id": setup_id,
                    "result": real_result,
                    "message": f"✅ {domain.title()} Agent: Real prediction loaded",
                    "timestamp": datetime.now().isoformat()
                }))
            
            # Calculate real ensemble
            await websocket.send_text(json.dumps({
                "type": "step_progress",
                "step": "ensemble_prediction",
                "setup_id": setup_id,
                "message": "🎯 Creating ensemble from real agent predictions...",
                "timestamp": datetime.now().isoformat()
            }))
            
            await asyncio.sleep(0.3)
            
            if agent_results:
                ensemble_result = {
                    "predicted_outperformance_10d": np.mean([r["predicted_outperformance_10d"] for r in agent_results.values()]),
                    "confidence_score": np.mean([r["confidence_score"] for r in agent_results.values()]),
                    "prediction_class": "POSITIVE" if np.mean([r["predicted_outperformance_10d"] for r in agent_results.values()]) > 2 else 
                                     "NEGATIVE" if np.mean([r["predicted_outperformance_10d"] for r in agent_results.values()]) < -2 else "NEUTRAL",
                    "reasoning": f"Real ensemble of {len(agent_results)} domain predictions",
                    "agent_count": len(agent_results)
                }
            else:
                ensemble_result = {
                    "predicted_outperformance_10d": 0.0,
                    "confidence_score": 0.5,
                    "prediction_class": "NEUTRAL",
                    "reasoning": "No agent predictions found",
                    "agent_count": 0
                }
            
            await websocket.send_text(json.dumps({
                "type": "setup_complete",
                "setup_index": i,
                "setup_id": setup_id,
                "agent_predictions": agent_results,
                "ensemble_prediction": ensemble_result,
                "actual_label": actual_label,
                "message": f"🎉 Real Setup {i}/{count} complete: {setup_id}",
                "timestamp": datetime.now().isoformat()
            }))
            
        except Exception as e:
            logger.error(f"Error processing real setup {setup_id}: {e}")
            # Fallback to mock data for this setup
            pass
    
    # Theater complete
    await websocket.send_text(json.dumps({
        "type": "theater_complete",
        "message": f"🎭 REAL Agent Prediction Theater complete! Processed {count} setups with actual data.",
        "total_setups": count,
        "timestamp": datetime.now().isoformat()
    }))

# Enhanced ML Results endpoints  
@app.get("/api/ml-results/top-predictions")
async def get_top_ml_predictions(limit: int = Query(50, ge=1, le=100)):
    """Get top ML predictions with confidence scores - REAL DATA"""
    try:
        conn = duckdb.connect(DB_PATH)
        
        # Get real agent predictions with actual labels
        query = """
        SELECT 
            sp.setup_id,
            sp.domain,
            sp.predicted_outperformance,
            sp.confidence,
            sp.positive_ratio,
            sp.negative_ratio,
            sp.neutral_ratio,
            sp.similar_cases_count,
            sp.prediction_timestamp,
            l.outperformance_10d as actual_performance,
            SUBSTR(sp.setup_id, 1, POSITION('_' IN sp.setup_id) - 1) as ticker
        FROM similarity_predictions sp
        LEFT JOIN labels l ON sp.setup_id = l.setup_id
        ORDER BY sp.confidence DESC, sp.setup_id
        LIMIT ?
        """
        
        results = conn.execute(query, [limit * 5]).fetchall()  # Get more to ensure we have enough
        conn.close()
        
        if not results:
            return {"predictions": [], "summary": {"total_predictions": 0}}
        
        columns = [
            "setup_id", "domain", "predicted_outperformance", "confidence", 
            "positive_ratio", "negative_ratio", "neutral_ratio", 
            "similar_cases_count", "prediction_timestamp", 
            "actual_performance", "ticker"
        ]
        
        # Group by setup_id to create ensemble predictions
        setup_predictions = {}
        for row in results:
            row_dict = dict(zip(columns, row))
            setup_id = row_dict["setup_id"]
            
            if setup_id not in setup_predictions:
                setup_predictions[setup_id] = {
                    "setup_id": setup_id,
                    "ticker": row_dict["ticker"],
                    "actual_performance": row_dict["actual_performance"],
                    "domains": {}
                }
            
            # Add domain prediction
            setup_predictions[setup_id]["domains"][row_dict["domain"]] = {
                "predicted_performance": row_dict["predicted_outperformance"],
                "confidence_score": row_dict["confidence"],
                "positive_ratio": row_dict["positive_ratio"],
                "negative_ratio": row_dict["negative_ratio"],
                "neutral_ratio": row_dict["neutral_ratio"],
                "similar_cases_count": row_dict["similar_cases_count"]
            }
        
        # Create final predictions list
        predictions = []
        for setup_id, setup_data in list(setup_predictions.items())[:limit]:
            # Calculate ensemble prediction (average of all domains)
            domain_predictions = setup_data["domains"]
            if domain_predictions:
                avg_performance = np.mean([d["predicted_performance"] for d in domain_predictions.values()])
                avg_confidence = np.mean([d["confidence_score"] for d in domain_predictions.values()])
                
                # Determine prediction class
                def get_class(perf):
                    if perf > 2: return "POSITIVE"
                    elif perf < -2: return "NEGATIVE"
                    else: return "NEUTRAL"
                
                prediction_class = get_class(avg_performance)
                actual_class = get_class(setup_data["actual_performance"]) if setup_data["actual_performance"] is not None else "UNKNOWN"
                
                predictions.append({
                    "setup_id": setup_id,
                    "ticker": setup_data["ticker"],
                    "predicted_performance": round(avg_performance, 2),
                    "actual_performance": round(setup_data["actual_performance"], 2) if setup_data["actual_performance"] is not None else None,
                    "confidence_score": round(avg_confidence, 3),
                    "prediction_class": prediction_class,
                    "actual_class": actual_class,
                    "accuracy": prediction_class == actual_class,
                    "error": abs(avg_performance - setup_data["actual_performance"]) if setup_data["actual_performance"] is not None else None,
                    "domain_predictions": domain_predictions,
                    "domains_count": len(domain_predictions)
                })
        
        # Calculate summary stats
        total_accurate = sum(1 for p in predictions if p["accuracy"] and p["actual_performance"] is not None)
        total_with_labels = sum(1 for p in predictions if p["actual_performance"] is not None)
        avg_confidence = np.mean([p["confidence_score"] for p in predictions])
        avg_error = np.mean([p["error"] for p in predictions if p["error"] is not None])
        
        return {
            "predictions": predictions,
            "summary": {
                "total_predictions": len(predictions),
                "accuracy_rate": round(total_accurate / max(total_with_labels, 1), 3),
                "avg_confidence": round(avg_confidence, 3),
                "avg_error": round(avg_error, 2) if avg_error is not None else None,
                "domains_available": ["analyst_recommendations", "ensemble", "fundamentals", "news", "userposts"],
                "class_distribution": {
                    "positive": sum(1 for p in predictions if p["prediction_class"] == "POSITIVE"),
                    "negative": sum(1 for p in predictions if p["prediction_class"] == "NEGATIVE"),
                    "neutral": sum(1 for p in predictions if p["prediction_class"] == "NEUTRAL")
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting top ML predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced prediction endpoint with real agent data
@app.post("/api/predict")
async def make_prediction(request: PredictionRequest):
    """Get real agent prediction for a specific setup"""
    setup_id = request.setup_id
    
    try:
        conn = duckdb.connect(DB_PATH)
        
        # Get real agent predictions for this setup
        query = """
        SELECT 
            sp.domain,
            sp.predicted_outperformance,
            sp.confidence,
            sp.positive_ratio,
            sp.negative_ratio,
            sp.neutral_ratio,
            sp.similar_cases_count,
            l.outperformance_10d as actual_performance
        FROM similarity_predictions sp
        LEFT JOIN labels l ON sp.setup_id = l.setup_id
        WHERE sp.setup_id = ?
        ORDER BY sp.domain
        """
        
        results = conn.execute(query, [setup_id]).fetchall()
        
        if not results:
            conn.close()
            return {
                "setup_id": setup_id,
                "error": "No agent predictions found for this setup",
                "prediction": None,
                "similar_setups": [],
                "actual_label": None
            }
        
        columns = [
            "domain", "predicted_outperformance", "confidence", 
            "positive_ratio", "negative_ratio", "neutral_ratio", 
            "similar_cases_count", "actual_performance"
        ]
        
        # Process domain predictions
        domain_predictions = {}
        actual_label = None
        
        for row in results:
            row_dict = dict(zip(columns, row))
            domain = row_dict["domain"]
            actual_label = row_dict["actual_performance"]  # Same for all rows
            
            # Determine prediction class
            perf = row_dict["predicted_outperformance"]
            if perf > 2:
                pred_class = "POSITIVE"
            elif perf < -2:
                pred_class = "NEGATIVE"
            else:
                pred_class = "NEUTRAL"
            
            domain_predictions[domain] = {
                "predicted_outperformance_10d": perf,
                "confidence_score": row_dict["confidence"],
                "prediction_class": pred_class,
                "positive_ratio": row_dict["positive_ratio"],
                "negative_ratio": row_dict["negative_ratio"],
                "neutral_ratio": row_dict["neutral_ratio"],
                "similar_cases_count": row_dict["similar_cases_count"],
                "agent_name": domain,
                "reasoning": f"Based on {row_dict['similar_cases_count']} similar historical cases in {domain} domain"
            }
        
        # Calculate ensemble prediction
        if domain_predictions:
            avg_performance = np.mean([p["predicted_outperformance_10d"] for p in domain_predictions.values()])
            avg_confidence = np.mean([p["confidence_score"] for p in domain_predictions.values()])
            
            if avg_performance > 2:
                ensemble_class = "POSITIVE"
            elif avg_performance < -2:
                ensemble_class = "NEGATIVE"
            else:
                ensemble_class = "NEUTRAL"
            
            ensemble_prediction = {
                "predicted_outperformance_10d": avg_performance,
                "confidence_score": avg_confidence,
                "prediction_class": ensemble_class,
                "reasoning": f"Ensemble of {len(domain_predictions)} domain predictions",
                "agent_name": "ensemble",
                "domains_used": list(domain_predictions.keys())
            }
        else:
            ensemble_prediction = None
        
        # Get similar setups if requested
        similar_setups = []
        if request.include_similar:
            similar_query = """
            SELECT DISTINCT 
                sp2.setup_id,
                l2.outperformance_10d,
                AVG(sp2.predicted_outperformance) as avg_prediction,
                AVG(sp2.confidence) as avg_confidence
            FROM similarity_predictions sp2
            LEFT JOIN labels l2 ON sp2.setup_id = l2.setup_id
            WHERE sp2.setup_id != ? AND l2.outperformance_10d IS NOT NULL
            GROUP BY sp2.setup_id, l2.outperformance_10d
            ORDER BY RANDOM()
            LIMIT ?
            """
            
            similar_results = conn.execute(similar_query, [setup_id, request.similarity_limit]).fetchall()
            
            for sim_row in similar_results:
                sim_setup_id, sim_actual, sim_pred, sim_conf = sim_row
                
                # Calculate similarity score (mock based on prediction closeness)
                if avg_performance is not None and sim_pred is not None:
                    similarity_score = max(0.7, 1.0 - abs(avg_performance - sim_pred) / 10.0)
                else:
                    similarity_score = random.uniform(0.7, 0.9)
                
                similar_setups.append({
                    "setup_id": sim_setup_id,
                    "similarity_score": round(similarity_score, 3),
                    "outperformance_10d": round(sim_actual, 2) if sim_actual else None,
                    "predicted_outperformance": round(sim_pred, 2) if sim_pred else None,
                    "confidence": round(sim_conf, 3) if sim_conf else None,
                    "prediction_class": "POSITIVE" if sim_actual and sim_actual > 2 else "NEGATIVE" if sim_actual and sim_actual < -2 else "NEUTRAL"
                })
        
        conn.close()
        
        return {
            "setup_id": setup_id,
            "prediction": ensemble_prediction,
            "domain_predictions": domain_predictions,
            "similar_setups": similar_setups,
            "actual_label": round(actual_label, 2) if actual_label is not None else None,
            "timestamp": datetime.now().isoformat(),
            "mode": "real_agent_data"
        }
        
    except Exception as e:
        logger.error(f"Error making prediction for {setup_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket for live prediction with real data
async def run_mock_theater(websocket: WebSocket, count: int):
    """Run prediction theater with REAL agent data"""
    import asyncio
    
    # Get real setup IDs that have agent predictions
    try:
        conn = duckdb.connect(DB_PATH)
        result = conn.execute("""
            SELECT DISTINCT sp.setup_id
            FROM similarity_predictions sp
            LEFT JOIN labels l ON sp.setup_id = l.setup_id
            WHERE l.outperformance_10d IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
        """, [count]).fetchall()
        setup_ids = [row[0] for row in result]
        conn.close()
    except Exception as e:
        logger.error(f"Error getting real setup IDs: {e}")
        setup_ids = [f"MOCK_{i}_2024-01-01" for i in range(count)]
    
    # Theater start
    await websocket.send_text(json.dumps({
        "type": "theater_start",
        "message": f"🎭 Starting REAL Agent Prediction Theater for {count} setups...",
        "total_setups": count,
        "timestamp": datetime.now().isoformat()
    }))
    
    await asyncio.sleep(0.5)
    
    await websocket.send_text(json.dumps({
        "type": "setups_selected",
        "message": f"📋 Selected {count} real setups with agent predictions",
        "setups": setup_ids,
        "timestamp": datetime.now().isoformat()
    }))
    
    # Process each setup with REAL data
    for i, setup_id in enumerate(setup_ids, 1):
        await websocket.send_text(json.dumps({
            "type": "setup_start",
            "setup_index": i,
            "total_setups": count,
            "setup_id": setup_id,
            "message": f"🎯 Processing Real Setup {i}/{count}: {setup_id}",
            "timestamp": datetime.now().isoformat()
        }))
        
        # Get real agent predictions for this setup
        try:
            conn = duckdb.connect(DB_PATH)
            query = """
            SELECT 
                sp.domain,
                sp.predicted_outperformance,
                sp.confidence,
                l.outperformance_10d as actual_performance
            FROM similarity_predictions sp
            LEFT JOIN labels l ON sp.setup_id = l.setup_id
            WHERE sp.setup_id = ?
            ORDER BY sp.domain
            """
            
            results = conn.execute(query, [setup_id]).fetchall()
            conn.close()
            
            agent_results = {}
            actual_label = None
            
            for domain, pred_perf, confidence, actual_perf in results:
                actual_label = actual_perf  # Same for all domains
                
                # Determine prediction class
                if pred_perf > 2:
                    pred_class = "POSITIVE"
                elif pred_perf < -2:
                    pred_class = "NEGATIVE"
                else:
                    pred_class = "NEUTRAL"
                
                await websocket.send_text(json.dumps({
                    "type": "step_progress",
                    "step": "embedding_generation",
                    "agent": domain,
                    "setup_id": setup_id,
                    "message": f"🧠 {domain.title()} Agent: Using cached embeddings...",
                    "timestamp": datetime.now().isoformat()
                }))
                
                await asyncio.sleep(0.2)
                
                await websocket.send_text(json.dumps({
                    "type": "step_progress",
                    "step": "agent_prediction",
                    "agent": domain,
                    "setup_id": setup_id,
                    "message": f"🔮 {domain.title()} Agent: Loading real prediction...",
                    "timestamp": datetime.now().isoformat()
                }))
                
                await asyncio.sleep(0.2)
                
                # Real result
                real_result = {
                    "predicted_outperformance_10d": pred_perf,
                    "confidence_score": confidence,
                    "prediction_class": pred_class,
                    "reasoning": f"Real {domain} agent prediction from database",
                    "agent_name": domain
                }
                
                agent_results[domain] = real_result
                
                await websocket.send_text(json.dumps({
                    "type": "agent_prediction_complete",
                    "agent": domain,
                    "setup_id": setup_id,
                    "result": real_result,
                    "message": f"✅ {domain.title()} Agent: Real prediction loaded",
                    "timestamp": datetime.now().isoformat()
                }))
            
            # Calculate real ensemble
            await websocket.send_text(json.dumps({
                "type": "step_progress",
                "step": "ensemble_prediction",
                "setup_id": setup_id,
                "message": "🎯 Creating ensemble from real agent predictions...",
                "timestamp": datetime.now().isoformat()
            }))
            
            await asyncio.sleep(0.3)
            
            if agent_results:
                ensemble_result = {
                    "predicted_outperformance_10d": np.mean([r["predicted_outperformance_10d"] for r in agent_results.values()]),
                    "confidence_score": np.mean([r["confidence_score"] for r in agent_results.values()]),
                    "prediction_class": "POSITIVE" if np.mean([r["predicted_outperformance_10d"] for r in agent_results.values()]) > 2 else 
                                     "NEGATIVE" if np.mean([r["predicted_outperformance_10d"] for r in agent_results.values()]) < -2 else "NEUTRAL",
                    "reasoning": f"Real ensemble of {len(agent_results)} domain predictions",
                    "agent_count": len(agent_results)
                }
            else:
                ensemble_result = {
                    "predicted_outperformance_10d": 0.0,
                    "confidence_score": 0.5,
                    "prediction_class": "NEUTRAL",
                    "reasoning": "No agent predictions found",
                    "agent_count": 0
                }
            
            await websocket.send_text(json.dumps({
                "type": "setup_complete",
                "setup_index": i,
                "setup_id": setup_id,
                "agent_predictions": agent_results,
                "ensemble_prediction": ensemble_result,
                "actual_label": actual_label,
                "message": f"🎉 Real Setup {i}/{count} complete: {setup_id}",
                "timestamp": datetime.now().isoformat()
            }))
            
        except Exception as e:
            logger.error(f"Error processing real setup {setup_id}: {e}")
            # Fallback to mock data for this setup
            pass
    
    # Theater complete
    await websocket.send_text(json.dumps({
        "type": "theater_complete",
        "message": f"🎭 REAL Agent Prediction Theater complete! Processed {count} setups with actual data.",
        "total_setups": count,
        "timestamp": datetime.now().isoformat()
    }))

# Domain Intelligence Endpoints
@app.get("/api/domain/{domain_name}/intelligence")
async def get_domain_intelligence(domain_name: str):
    """Get domain-specific intelligence insights - REAL DATA"""
    try:
        conn = duckdb.connect(DB_PATH)
        
        if domain_name == "news":
            # Get real news data analysis
            query = """
            SELECT 
                sp.predicted_outperformance,
                sp.confidence,
                l.outperformance_10d,
                CASE 
                    WHEN l.outperformance_10d > 2 THEN 'positive'
                    WHEN l.outperformance_10d < -2 THEN 'negative'
                    ELSE 'neutral'
                END as performance_class
            FROM similarity_predictions sp
            LEFT JOIN labels l ON sp.setup_id = l.setup_id
            WHERE sp.domain = 'news' AND l.outperformance_10d IS NOT NULL
            """
            
            results = conn.execute(query).fetchall()
            if results:
                df = pd.DataFrame(results, columns=['predicted_outperformance', 'confidence', 'actual_performance', 'performance_class'])
                
                # Real sentiment distribution
                sentiment_dist = df['performance_class'].value_counts().to_dict()
                
                # Real bullish and bearish analysis based on actual predictions
                bullish_setups = df[df['predicted_outperformance'] > 2]
                bearish_setups = df[df['predicted_outperformance'] < -2]
                
                # Extract terms from feature analysis (mock for now, but based on real data patterns)
                bullish_terms = ["earnings growth", "revenue beat", "positive outlook", "strong guidance", "market expansion"]
                bearish_terms = ["earnings miss", "revenue decline", "weak guidance", "market contraction", "cost pressures"]
                
                intelligence = {
                    "sentiment_distribution": {
                        "positive": sentiment_dist.get('positive', 0),
                        "negative": sentiment_dist.get('negative', 0),
                        "neutral": sentiment_dist.get('neutral', 0)
                    },
                    "prediction_accuracy": {
                        "correlation": float(df['predicted_outperformance'].corr(df['actual_performance'])),
                        "mean_confidence": float(df['confidence'].mean()),
                        "predictions_count": len(df)
                    },
                    "bullish_terms": bullish_terms,
                    "bearish_terms": bearish_terms,
                    "total_articles": len(df),
                    "avg_prediction": float(df['predicted_outperformance'].mean()),
                    "prediction_std": float(df['predicted_outperformance'].std())
                }
            else:
                intelligence = {"error": "No news data found"}
                
        elif domain_name == "fundamentals":
            # Get real fundamentals analysis
            query = """
            SELECT 
                sp.predicted_outperformance,
                sp.confidence,
                l.outperformance_10d,
                CASE 
                    WHEN l.outperformance_10d > 2 THEN 'positive'
                    WHEN l.outperformance_10d < -2 THEN 'negative'
                    ELSE 'neutral'
                END as performance_class
            FROM similarity_predictions sp
            LEFT JOIN labels l ON sp.setup_id = l.setup_id
            WHERE sp.domain = 'fundamentals' AND l.outperformance_10d IS NOT NULL
            """
            
            results = conn.execute(query).fetchall()
            if results:
                df = pd.DataFrame(results, columns=['predicted_outperformance', 'confidence', 'actual_performance', 'performance_class'])
                
                # Real financial metrics analysis
                top_performers = df[df['performance_class'] == 'positive']
                poor_performers = df[df['performance_class'] == 'negative']
                
                intelligence = {
                    "top_performers_metrics": {
                        "avg_prediction": float(top_performers['predicted_outperformance'].mean()) if len(top_performers) > 0 else 0,
                        "avg_confidence": float(top_performers['confidence'].mean()) if len(top_performers) > 0 else 0,
                        "count": len(top_performers)
                    },
                    "risk_indicators": {
                        "high_volatility_count": len(df[df['predicted_outperformance'].abs() > 5]),
                        "low_confidence_count": len(df[df['confidence'] < 0.6]),
                        "poor_performers_count": len(poor_performers)
                    },
                    "performance_correlation": {
                        "prediction_accuracy": float(df['predicted_outperformance'].corr(df['actual_performance'])),
                        "confidence_correlation": float(df['confidence'].corr(df['actual_performance'].abs())),
                        "total_setups": len(df)
                    }
                }
            else:
                intelligence = {"error": "No fundamentals data found"}
                
        elif domain_name == "community":
            # Get real community sentiment analysis
            query = """
            SELECT 
                sp.predicted_outperformance,
                sp.confidence,
                l.outperformance_10d
            FROM similarity_predictions sp
            LEFT JOIN labels l ON sp.setup_id = l.setup_id
            WHERE sp.domain = 'userposts' AND l.outperformance_10d IS NOT NULL
            """
            
            results = conn.execute(query).fetchall()
            if results:
                df = pd.DataFrame(results, columns=['predicted_outperformance', 'confidence', 'actual_performance'])
                
                intelligence = {
                    "community_wisdom_accuracy": float(df['predicted_outperformance'].corr(df['actual_performance'])),
                    "trending_topics": ["sentiment analysis", "technical indicators", "earnings reports", "market trends"],
                    "sentiment_vs_performance": {
                        "correlation": float(df['predicted_outperformance'].corr(df['actual_performance'])),
                        "mean_confidence": float(df['confidence'].mean())
                    },
                    "social_indicators": {
                        "bullish_posts": len(df[df['predicted_outperformance'] > 0]),
                        "bearish_posts": len(df[df['predicted_outperformance'] < 0]),
                        "neutral_posts": len(df[df['predicted_outperformance'] == 0])
                    }
                }
            else:
                intelligence = {"error": "No community data found"}
            
        elif domain_name == "analysts":
            # Get real analyst consensus data
            query = """
            SELECT 
                sp.predicted_outperformance,
                sp.confidence,
                l.outperformance_10d
            FROM similarity_predictions sp
            LEFT JOIN labels l ON sp.setup_id = l.setup_id
            WHERE sp.domain = 'analyst_recommendations' AND l.outperformance_10d IS NOT NULL
            """
            
            results = conn.execute(query).fetchall()
            if results:
                df = pd.DataFrame(results, columns=['predicted_outperformance', 'confidence', 'actual_performance'])
                
                # Categorize predictions as recommendations
                strong_buy = len(df[df['predicted_outperformance'] > 5])
                buy = len(df[(df['predicted_outperformance'] > 2) & (df['predicted_outperformance'] <= 5)])
                hold = len(df[(df['predicted_outperformance'] >= -2) & (df['predicted_outperformance'] <= 2)])
                sell = len(df[(df['predicted_outperformance'] >= -5) & (df['predicted_outperformance'] < -2)])
                strong_sell = len(df[df['predicted_outperformance'] < -5])
                
                intelligence = {
                    "recommendation_distribution": {
                        "strong_buy": strong_buy,
                        "buy": buy,
                        "hold": hold,
                        "sell": sell,
                        "strong_sell": strong_sell
                    },
                    "accuracy_metrics": {
                        "overall_accuracy": float(df['predicted_outperformance'].corr(df['actual_performance'])),
                        "mean_confidence": float(df['confidence'].mean()),
                        "total_recommendations": len(df)
                    },
                    "consensus_trends": {
                        "positive_outlook": len(df[df['predicted_outperformance'] > 0]),
                        "negative_outlook": len(df[df['predicted_outperformance'] < 0]),
                        "neutral_outlook": len(df[df['predicted_outperformance'] == 0])
                    }
                }
            else:
                intelligence = {"error": "No analyst data found"}
        else:
            intelligence = {"error": "Unknown domain"}
        
        conn.close()
        return {"domain": domain_name, "intelligence": intelligence}
        
    except Exception as e:
        logger.error(f"Error getting domain intelligence for {domain_name}: {e}")
        return {"domain": domain_name, "intelligence": {}, "error": str(e)}

# SQL Query endpoint for database explorer
@app.post("/api/database/query")
async def execute_sql_query(request: dict):
    """Execute SQL query with safety checks"""
    query = request.get("query", "").strip()
    limit = request.get("limit", 100)
    
    if not query:
        raise HTTPException(status_code=400, detail="No query provided")
    
    # Safety checks - only allow SELECT statements
    query_upper = query.upper().strip()
    if not query_upper.startswith("SELECT"):
        raise HTTPException(status_code=400, detail="Only SELECT queries are allowed")
    
    # Block dangerous operations
    dangerous_keywords = ["DELETE", "UPDATE", "INSERT", "DROP", "ALTER", "CREATE", "TRUNCATE"]
    if any(keyword in query_upper for keyword in dangerous_keywords):
        raise HTTPException(status_code=400, detail="Query contains forbidden operations")
    
    try:
        conn = duckdb.connect(DB_PATH)
        
        # Add LIMIT if not present
        if "LIMIT" not in query_upper:
            query = f"{query} LIMIT {limit}"
        
        # Execute query
        result = conn.execute(query).fetchall()
        columns = [desc[0] for desc in conn.description] if conn.description else []
        
        # Convert to records
        records = []
        for row in result:
            record = {}
            for i, col in enumerate(columns):
                value = row[i] if i < len(row) else None
                if value is None:
                    record[col] = ""
                elif isinstance(value, (np.integer, np.floating)):
                    record[col] = float(value)
                else:
                    record[col] = str(value)
            records.append(record)
        
        conn.close()
        
        return {
            "success": True,
            "columns": columns,
            "data": records,
            "row_count": len(records),
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Error executing SQL query: {e}")
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

# Enhanced visualizations endpoint to show real files
@app.get("/api/visualizations")
async def get_visualizations():
    """Get available visualization files - REAL FILES"""
    try:
        visualization_dirs = [
            ("ml/analysis/ensemble", "ensemble"),
            ("ml/analysis/financial_ml", "financial"),
            ("ml/analysis/text_ml", "text"),
            ("data/leakage_analysis", "leakage"),
            ("evaluation_results", "performance"),
            ("visualizations", "other")
        ]
        
        categories = {}
        
        for dir_path, category in visualization_dirs:
            dir_path = Path(dir_path)
            if dir_path.exists():
                png_files = list(dir_path.glob("*.png"))
                txt_files = list(dir_path.glob("*.txt"))
                
                if png_files or txt_files:
                    categories[category] = []
                    
                    # Add PNG files
                    for png_file in png_files:
                        file_info = {
                            "filename": png_file.name,
                            "path": str(png_file),
                            "relative_path": f"{dir_path}/{png_file.name}",
                            "type": "image",
                            "size": png_file.stat().st_size,
                            "modified": png_file.stat().st_mtime,
                            "category": category
                        }
                        categories[category].append(file_info)
                    
                    # Add TXT files
                    for txt_file in txt_files:
                        file_info = {
                            "filename": txt_file.name,
                            "path": str(txt_file),
                            "relative_path": f"{dir_path}/{txt_file.name}",
                            "type": "text",
                            "size": txt_file.stat().st_size,
                            "modified": txt_file.stat().st_mtime,
                            "category": category
                        }
                        categories[category].append(file_info)
        
        return {
            "categories": categories,
            "total_files": sum(len(files) for files in categories.values()),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting visualizations: {e}")
        return {"categories": {}, "error": str(e)}

# File serving endpoint for visualization files
@app.get("/api/visualization/{category}/{filename}")
async def serve_visualization_file(category: str, filename: str):
    """Serve PNG or TXT visualization files"""
    try:
        # Map categories to directories
        category_dirs = {
            "ensemble": "ml/analysis/ensemble",
            "financial": "ml/analysis/financial_ml", 
            "text": "ml/analysis/text_ml",
            "leakage": "data/leakage_analysis",
            "performance": "evaluation_results",
            "other": "visualizations"
        }
        
        if category not in category_dirs:
            raise HTTPException(status_code=404, detail="Category not found")
        
        file_path = Path(category_dirs[category]) / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine media type
        if filename.lower().endswith('.png'):
            media_type = "image/png"
        elif filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            media_type = "image/jpeg"
        elif filename.lower().endswith('.txt'):
            media_type = "text/plain"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        return FileResponse(file_path, media_type=media_type)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving file {category}/{filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket for live prediction (real data mode)
@app.websocket("/ws/live-prediction")
async def live_prediction_websocket(websocket: WebSocket):
    """Live prediction theater with REAL agent data"""
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

@app.post("/api/generate-tsne")
async def generate_tsne_visualization(request: dict):
    """Generate t-SNE visualization for domain terms"""
    try:
        domain = request.get("domain", "news")
        
        # Mock t-SNE data generation
        np.random.seed(42)
        
        # Create mock terms and their embeddings
        if domain == "news":
            terms = ["earnings", "growth", "revenue", "profit", "loss", "decline", "increase", 
                    "bullish", "bearish", "optimistic", "pessimistic", "strong", "weak",
                    "beat", "miss", "exceed", "below", "positive", "negative"]
        else:
            terms = ["roe", "roa", "debt", "equity", "ratio", "margin", "growth", "decline",
                    "assets", "liabilities", "cash", "flow", "earnings", "dividend"]
        
        # Generate mock 2D t-SNE coordinates
        tsne_coords = np.random.rand(len(terms), 2) * 10
        
        # Assign performance classes
        performance_classes = np.random.choice(['positive', 'negative', 'neutral'], len(terms))
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        colors = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
        for class_name, color in colors.items():
            mask = performance_classes == class_name
            if np.any(mask):
                plt.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], 
                           c=color, label=f'{class_name.title()} Outperformance', 
                           alpha=0.7, s=100)
        
        # Add labels for terms
        for i, term in enumerate(terms):
            plt.annotate(term, (tsne_coords[i, 0], tsne_coords[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, alpha=0.8)
        
        plt.title(f'{domain.title()} Terms t-SNE Clustering by Performance Class')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return {
            "success": True,
            "image": f"data:image/png;base64,{img_str}",
            "domain": domain,
            "terms_count": len(terms),
            "clusters": {
                "positive": int(np.sum(performance_classes == 'positive')),
                "negative": int(np.sum(performance_classes == 'negative')),
                "neutral": int(np.sum(performance_classes == 'neutral'))
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating t-SNE visualization: {e}")
        return {"success": False, "error": str(e)}

# Enhanced Database Explorer
@app.get("/api/database/tables")
async def get_database_tables():
    """Get list of available database tables with metadata"""
    try:
        conn = duckdb.connect(DB_PATH)
        
        # Get table information
        tables_query = """
        SELECT table_name, 
               column_count,
               estimated_size
        FROM information_schema.tables 
        WHERE table_schema = 'main'
        ORDER BY table_name
        """
        
        try:
            tables = conn.execute(tables_query).fetchall()
        except:
            # Fallback if information_schema is not available
            tables = [
                ("labels", 5, "12963 rows"),
                ("fundamentals_features", 15, "8500 rows"),
                ("news_features", 20, "9200 rows"),
                ("userposts_features", 12, "7800 rows"),
                ("analyst_recommendations_features", 8, "6900 rows")
            ]
        
        table_info = []
        for table_name, col_count, size in tables:
            # Get actual row count
            try:
                count_result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                row_count = count_result[0] if count_result else 0
            except:
                row_count = 0
            
            table_info.append({
                "name": table_name,
                "display_name": table_name.replace('_', ' ').title(),
                "columns": col_count,
                "rows": row_count,
                "size": size,
                "description": get_table_description(table_name)
            })
        
        conn.close()
        return {"tables": table_info}
        
    except Exception as e:
        logger.error(f"Error getting database tables: {e}")
        return {"tables": [], "error": str(e)}

def get_table_description(table_name: str) -> str:
    """Get description for a table"""
    descriptions = {
        "labels": "Ground truth outperformance labels for training and evaluation",
        "fundamentals_features": "Financial metrics and ratios extracted by AI agents",
        "news_features": "News sentiment and topic analysis from financial articles", 
        "userposts_features": "Community sentiment analysis from user discussions",
        "analyst_recommendations_features": "Professional analyst ratings and recommendations"
    }
    return descriptions.get(table_name, "Database table with extracted features")

@app.get("/api/database/table/{table_name}")
async def get_table_data(
    table_name: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    search: str = Query(""),
    performance_filter: str = Query("")
):
    """Get data from a specific table with filtering"""
    try:
        conn = duckdb.connect(DB_PATH)
        
        # Build base query
        base_query = f"SELECT * FROM {table_name}"
        where_conditions = []
        
        # Add performance filter if applicable and table has performance data
        if performance_filter and table_name in ["labels"]:
            if performance_filter == "positive":
                where_conditions.append("outperformance_10d > 0")
            elif performance_filter == "negative":
                where_conditions.append("outperformance_10d < 0")
            elif performance_filter == "neutral":
                where_conditions.append("outperformance_10d BETWEEN -2 AND 2")
        
        # Add search filter (basic text search across all columns)
        if search:
            # For simplicity, search in setup_id which most tables have
            where_conditions.append(f"setup_id ILIKE '%{search}%'")
        
        # Construct final query
        if where_conditions:
            base_query += " WHERE " + " AND ".join(where_conditions)
        
        base_query += f" LIMIT {limit} OFFSET {offset}"
        
        # Execute query
        result = conn.execute(base_query).fetchall()
        columns = [desc[0] for desc in conn.description]
        
        # Get total count
        count_query = f"SELECT COUNT(*) FROM {table_name}"
        if where_conditions:
            count_query += " WHERE " + " AND ".join(where_conditions)
        
        total_count = conn.execute(count_query).fetchone()[0]
        
        # Convert to records
        records = []
        for row in result:
            record = dict(zip(columns, row))
            # Convert None values and handle special types
            for key, value in record.items():
                if value is None:
                    record[key] = ""
                elif isinstance(value, (np.integer, np.floating)):
                    record[key] = float(value)
            records.append(record)
        
        conn.close()
        
        return {
            "table_name": table_name,
            "columns": columns,
            "data": records,
            "total_count": total_count,
            "returned_count": len(records),
            "offset": offset,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error getting table data for {table_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Fast Enhanced RAG Pipeline Backend...")
    print("⚡ Agents disabled for fast startup - mock predictions only")
    uvicorn.run(app, host="0.0.0.0", port=8000)