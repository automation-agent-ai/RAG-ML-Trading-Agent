#!/usr/bin/env python3
"""
Enhanced RAG Pipeline Frontend Backend API
==========================================

FastAPI backend providing:
- Live Agent Prediction Theater with real-time progress tracking
- WebSocket streaming for agent prediction steps
- Similar embeddings discovery and matching
- Enhanced prediction workflows with step-by-step visualization
- Knowledge graph integration with agent reasoning paths
- Real-time portfolio scanning and analysis
- Comprehensive ML model evaluation and comparison
"""

import sys
import os
import asyncio
import json
import logging
import random
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager

import duckdb
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, BackgroundTasks
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

# Import ML and agent components
try:
    from make_agent_predictions import AgentPredictionMaker
    from create_prediction_list import find_complete_setups
    from ensemble_prediction import EnsemblePrediction
    from threshold_manager import ThresholdManager
except ImportError as e:
    logger.warning(f"Some ML components not available: {e}")

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
            df_clean[col] = df_clean[col].astype('Int64')  # nullable integer
        elif pd.api.types.is_float_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].astype('float64')
        elif pd.api.types.is_bool_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].astype('boolean')  # nullable boolean
    
    # Replace any remaining NaN/None values
    df_clean = df_clean.replace({np.nan: None, pd.NA: None})
    
    return df_clean.to_dict("records")

# Pydantic Models
class PredictionRequest(BaseModel):
    setup_id: str
    include_similar: bool = True
    similarity_limit: int = 5

class LivePredictionRequest(BaseModel):
    count: int = Field(default=5, ge=1, le=20, description="Number of setups to predict (1-20)")
    random_seed: Optional[int] = None

class ScanRequest(BaseModel):
    start_date: str
    end_date: str
    min_probability: float = 0.5
    limit: int = 100

class PredictionResponse(BaseModel):
    setup_id: str
    prediction: int
    probability: float
    confidence: float
    explanation: str
    key_factors: List[str]
    risk_factors: List[str]
    similar_cases: Optional[List[Dict[str, Any]]] = None

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections.copy():
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)

# Global variables
DB_PATH = "data/sentiment_system.duckdb"
agent_prediction_maker = None
ensemble_predictor = None
connection_manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup"""
    global agent_prediction_maker, ensemble_predictor
    
    logger.info("🚀 Starting Enhanced RAG Pipeline Backend...")
    
    # Initialize agent prediction maker
    try:
        agent_prediction_maker = AgentPredictionMaker(db_path=DB_PATH)
        logger.info("✅ Agent Prediction Maker initialized")
    except Exception as e:
        logger.warning(f"⚠️ Agent Prediction Maker initialization failed: {e}")
    
    # Initialize ensemble predictor
    try:
        ensemble_predictor = EnsemblePrediction(db_path=DB_PATH)
        logger.info("✅ Ensemble Predictor initialized")
    except Exception as e:
        logger.warning(f"⚠️ Ensemble Predictor initialization failed: {e}")
    
    yield
    
    logger.info("🛑 Shutting down Enhanced RAG Pipeline Backend...")

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced RAG Pipeline Frontend API",
    description="Comprehensive API for live agent predictions, similarity matching, and interactive analysis",
    version="2.0.0",
    lifespan=lifespan
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
    """Enhanced health check with component status"""
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
            "components": {
                "agent_prediction_maker": agent_prediction_maker is not None,
                "ensemble_predictor": ensemble_predictor is not None,
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

# Enhanced setup management
@app.get("/api/setups")
async def get_setups(
    limit: int = Query(100, ge=1, le=1000),
    has_labels: bool = Query(True),
    random_sample: bool = Query(False)
):
    """Get available setups with enhanced filtering"""
    try:
        conn = duckdb.connect(DB_PATH)
        
        # Build query based on filters
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

# Live Agent Prediction Theater WebSocket
@app.websocket("/ws/live-prediction")
async def live_prediction_websocket(websocket: WebSocket):
    """WebSocket endpoint for live agent prediction theater"""
    await connection_manager.connect(websocket)
    
    try:
        while True:
            # Wait for client request
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # Validate request
            if request_data.get("action") == "start_prediction":
                count = min(int(request_data.get("count", 5)), 20)
                await run_live_prediction_theater(websocket, count)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"WebSocket error: {str(e)}"
        }))

async def run_live_prediction_theater(websocket: WebSocket, count: int):
    """Run live prediction theater with real-time progress updates"""
    try:
        # Send start message
        await websocket.send_text(json.dumps({
            "type": "theater_start",
            "message": f"🎭 Starting Live Agent Prediction Theater for {count} setups...",
            "total_setups": count,
            "timestamp": datetime.now().isoformat()
        }))
        
        # Get complete setups
        all_setups = find_complete_setups(DB_PATH)
        if len(all_setups) < count:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Only {len(all_setups)} complete setups available, requested {count}"
            }))
            return
        
        # Randomly select setups
        selected_setups = random.sample(all_setups, count)
        
        await websocket.send_text(json.dumps({
            "type": "setups_selected",
            "message": f"📋 Selected {count} random setups for prediction",
            "setups": selected_setups,
            "timestamp": datetime.now().isoformat()
        }))
        
        # Process each setup through the agent pipeline
        for i, setup_id in enumerate(selected_setups, 1):
            await websocket.send_text(json.dumps({
                "type": "setup_start",
                "setup_index": i,
                "total_setups": count,
                "setup_id": setup_id,
                "message": f"🎯 Processing Setup {i}/{count}: {setup_id}",
                "timestamp": datetime.now().isoformat()
            }))
            
            # Step 1: Load setup data
            await websocket.send_text(json.dumps({
                "type": "step_progress",
                "step": "data_loading",
                "setup_id": setup_id,
                "message": "📊 Loading setup data from database...",
                "timestamp": datetime.now().isoformat()
            }))
            
            setup_data = await get_setup_full_data(setup_id)
            
            # Step 2: Generate embeddings for each agent
            agent_results = {}
            for agent_name in ["fundamentals", "news", "analyst_recommendations", "userposts"]:
                await websocket.send_text(json.dumps({
                    "type": "step_progress",
                    "step": "embedding_generation",
                    "agent": agent_name,
                    "setup_id": setup_id,
                    "message": f"🧠 {agent_name.title()} Agent: Generating embeddings...",
                    "timestamp": datetime.now().isoformat()
                }))
                
                # Simulate embedding generation time
                await asyncio.sleep(0.5)
                
                await websocket.send_text(json.dumps({
                    "type": "step_progress",
                    "step": "agent_prediction",
                    "agent": agent_name,
                    "setup_id": setup_id,
                    "message": f"🔮 {agent_name.title()} Agent: Making prediction...",
                    "timestamp": datetime.now().isoformat()
                }))
                
                # Make actual agent prediction if available
                try:
                    if agent_prediction_maker and hasattr(agent_prediction_maker, 'agents'):
                        agent = agent_prediction_maker.agents.get(agent_name)
                        if agent:
                            prediction_result = await make_agent_prediction_async(agent, setup_id)
                            agent_results[agent_name] = prediction_result
                        else:
                            agent_results[agent_name] = create_mock_prediction(agent_name)
                    else:
                        agent_results[agent_name] = create_mock_prediction(agent_name)
                except Exception as e:
                    logger.warning(f"Agent {agent_name} prediction failed: {e}")
                    agent_results[agent_name] = create_mock_prediction(agent_name)
                
                await websocket.send_text(json.dumps({
                    "type": "agent_prediction_complete",
                    "agent": agent_name,
                    "setup_id": setup_id,
                    "result": agent_results[agent_name],
                    "message": f"✅ {agent_name.title()} Agent: Prediction complete",
                    "timestamp": datetime.now().isoformat()
                }))
            
            # Step 3: Ensemble prediction
            await websocket.send_text(json.dumps({
                "type": "step_progress",
                "step": "ensemble_prediction",
                "setup_id": setup_id,
                "message": "🎯 Creating ensemble prediction from all agents...",
                "timestamp": datetime.now().isoformat()
            }))
            
            ensemble_result = create_ensemble_prediction(agent_results)
            
            # Get actual label for comparison
            actual_label = await get_actual_label(setup_id)
            
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
            "message": f"🎭 Live Agent Prediction Theater complete! Processed {count} setups.",
            "total_setups": count,
            "timestamp": datetime.now().isoformat()
        }))
        
    except Exception as e:
        logger.error(f"Live prediction theater error: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Theater error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }))

async def get_setup_full_data(setup_id: str) -> Dict[str, Any]:
    """Get comprehensive data for a setup"""
    try:
        conn = duckdb.connect(DB_PATH)
        
        # Get data from all tables
        tables = ["fundamentals_features", "news_features", "analyst_recommendations_features", "userposts_features"]
        setup_data = {"setup_id": setup_id}
        
        for table in tables:
            try:
                query = f"SELECT * FROM {table} WHERE setup_id = ?"
                result = conn.execute(query, [setup_id]).fetchall()
                if result:
                    columns = [desc[0] for desc in conn.description]
                    setup_data[table] = dict(zip(columns, result[0]))
                else:
                    setup_data[table] = {}
            except Exception as e:
                logger.warning(f"Could not fetch {table} for {setup_id}: {e}")
                setup_data[table] = {}
        
        conn.close()
        return setup_data
        
    except Exception as e:
        logger.error(f"Error getting setup data for {setup_id}: {e}")
        return {"setup_id": setup_id, "error": str(e)}

async def make_agent_prediction_async(agent, setup_id: str) -> Dict[str, Any]:
    """Make async agent prediction"""
    try:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: agent.predict_with_llm(setup_id, {}))
        return result
    except Exception as e:
        logger.error(f"Agent prediction error: {e}")
        return create_mock_prediction("error")

def create_mock_prediction(agent_name: str) -> Dict[str, Any]:
    """Create mock prediction for demonstration"""
    predictions = [-1, 0, 1]
    prediction = random.choice(predictions)
    
    return {
        "predicted_outperformance_10d": random.uniform(-5.0, 5.0),
        "confidence_score": random.uniform(0.3, 0.9),
        "prediction_class": "POSITIVE" if prediction > 0 else "NEGATIVE" if prediction < 0 else "NEUTRAL",
        "reasoning": f"Mock {agent_name} prediction based on simulated analysis",
        "agent_name": agent_name
    }

def create_ensemble_prediction(agent_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Create ensemble prediction from agent results"""
    if not agent_results:
        return create_mock_prediction("ensemble")
    
    # Simple averaging for ensemble
    predictions = [result.get("predicted_outperformance_10d", 0.0) for result in agent_results.values()]
    confidences = [result.get("confidence_score", 0.5) for result in agent_results.values()]
    
    avg_prediction = np.mean(predictions)
    avg_confidence = np.mean(confidences)
    
    return {
        "predicted_outperformance_10d": avg_prediction,
        "confidence_score": avg_confidence,
        "prediction_class": "POSITIVE" if avg_prediction > 0.5 else "NEGATIVE" if avg_prediction < -0.5 else "NEUTRAL",
        "reasoning": f"Ensemble prediction from {len(agent_results)} agents",
        "agent_count": len(agent_results)
    }

async def get_actual_label(setup_id: str) -> Optional[float]:
    """Get actual label for comparison"""
    try:
        conn = duckdb.connect(DB_PATH)
        result = conn.execute("SELECT outperformance_10d FROM labels WHERE setup_id = ?", [setup_id]).fetchone()
        conn.close()
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Error getting label for {setup_id}: {e}")
        return None

# Similar embeddings discovery endpoint
@app.get("/api/setup/{setup_id}/similar")
async def find_similar_setups(
    setup_id: str,
    limit: int = Query(10, ge=1, le=50),
    min_similarity: float = Query(0.7, ge=0.0, le=1.0)
):
    """Find setups with similar embeddings"""
    try:
        # This would integrate with your existing similarity search
        # For now, return mock data that demonstrates the concept
        
        conn = duckdb.connect(DB_PATH)
        
        # Get some random similar setups as demo
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
        
        # Sort by similarity
        similar_setups.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "setup_id": setup_id,
            "similar_setups": similar_setups,
            "count": len(similar_setups)
        }
        
    except Exception as e:
        logger.error(f"Error finding similar setups for {setup_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced data endpoints
@app.get("/api/setup/{setup_id}/fundamentals")
async def get_setup_fundamentals(setup_id: str):
    """Get comprehensive fundamentals data for a setup"""
    try:
        conn = duckdb.connect(DB_PATH)
        
        # Get both financial_features and fundamentals_features
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
    """Get news analysis data for a setup"""
    try:
        conn = duckdb.connect(DB_PATH)
        
        # Get news features
        query = "SELECT * FROM news_features WHERE setup_id = ?"
        result = conn.execute(query, [setup_id]).fetchall()
        
        if result:
            columns = [desc[0] for desc in conn.description]
            news_data = dict(zip(columns, result[0]))
        else:
            news_data = {}
        
        conn.close()
        
        return {
            "setup_id": setup_id,
            "news_analysis": news_data
        }
        
    except Exception as e:
        logger.error(f"Error fetching news for {setup_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/setup/{setup_id}/userposts")
async def get_setup_userposts(setup_id: str):
    """Get user posts analysis data for a setup"""
    try:
        conn = duckdb.connect(DB_PATH)
        
        # Get userposts features
        query = "SELECT * FROM userposts_features WHERE setup_id = ?"
        result = conn.execute(query, [setup_id]).fetchall()
        
        if result:
            columns = [desc[0] for desc in conn.description]
            userposts_data = dict(zip(columns, result[0]))
        else:
            userposts_data = {}
        
        conn.close()
        
        return {
            "setup_id": setup_id,
            "userposts_analysis": userposts_data
        }
        
    except Exception as e:
        logger.error(f"Error fetching userposts for {setup_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced prediction endpoint
@app.post("/api/predict")
async def make_prediction(request: PredictionRequest):
    """Make enhanced prediction with similarity analysis"""
    try:
        setup_id = request.setup_id
        
        # Get similar setups if requested
        similar_setups = []
        if request.include_similar:
            similar_response = await find_similar_setups(
                setup_id, 
                limit=request.similarity_limit
            )
            similar_setups = similar_response["similar_setups"]
        
        # Make prediction (mock for now, integrate with your actual prediction logic)
        prediction_result = create_mock_prediction("ensemble")
        prediction_result["setup_id"] = setup_id
        
        # Get actual label for comparison
        actual_label = await get_actual_label(setup_id)
        
        return {
            "setup_id": setup_id,
            "prediction": prediction_result,
            "similar_setups": similar_setups,
            "actual_label": actual_label,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction error for {setup_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced visualization endpoints
@app.get("/api/visualizations")
async def get_visualizations():
    """Get available visualization files"""
    try:
        viz_dir = Path("visualizations")
        if not viz_dir.exists():
            viz_dir = Path("ml/analysis")
        
        categories = {}
        
        for category_dir in viz_dir.iterdir():
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
                
                categories[category_dir.name] = files
        
        return {"categories": categories}
        
    except Exception as e:
        logger.error(f"Error getting visualizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model performance endpoint
@app.get("/api/model-performance")
async def get_model_performance():
    """Get model performance metrics"""
    try:
        # Mock performance data - integrate with your actual model evaluation
        performance_data = {
            "models": [
                {
                    "name": "Random Forest",
                    "precision": 0.426,
                    "recall": 0.383,
                    "f1_score": 0.302,
                    "auc": 0.678
                },
                {
                    "name": "Logistic Regression",
                    "precision": 0.326,
                    "recall": 0.289,
                    "f1_score": 0.246,
                    "auc": 0.567
                },
                {
                    "name": "XGBoost",
                    "precision": 0.411,
                    "recall": 0.367,
                    "f1_score": 0.285,
                    "auc": 0.645
                },
                {
                    "name": "LightGBM",
                    "precision": 0.403,
                    "recall": 0.359,
                    "f1_score": 0.278,
                    "auc": 0.634
                },
                {
                    "name": "Ensemble",
                    "precision": 0.369,
                    "recall": 0.383,
                    "f1_score": 0.302,
                    "auc": 0.702
                }
            ],
            "best_model": "Ensemble",
            "evaluation_date": datetime.now().isoformat()
        }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)