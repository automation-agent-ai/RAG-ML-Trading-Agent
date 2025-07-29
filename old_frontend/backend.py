#!/usr/bin/env python3
"""
RAG Pipeline Frontend Backend API
=================================

FastAPI backend providing:
- Database API Layer: Serve data from DuckDB
- Real-time Predictions: ML ensemble model predictions  
- Knowledge Explainer: LLM-based setup explanations
- Visualization Integration: PNG/TXT file serving
- Scanner Integration: New setup discovery
"""

import sys
import os
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import ML components
from ml.ensemble_precision_analyzer import EnsemblePrecisionAnalyzer
from ml.precision_analyzer import PrecisionFeatureAnalyzer
from ml.text_precision_analyzer import TextPrecisionAnalyzer

# Initialize FastAPI app
app = FastAPI(
    title="RAG Pipeline Frontend API",
    description="Comprehensive API for stock setup analysis and prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include Knowledge Graph router
try:
    from knowledge_graph_api import kg_router, init_kg_on_startup
    app.include_router(kg_router)
    logger.info("✅ Knowledge Graph API routes added")
except ImportError as e:
    logger.warning(f"⚠️  Knowledge Graph API not available: {e}")
    kg_router = None
    init_kg_on_startup = None

# Global variables
DB_PATH = "../data/sentiment_system.duckdb"
ensemble_analyzer = None
knowledge_explainer = None

# Utility function to handle NaN values
def clean_nan_values(obj):
    """Recursively clean NaN values from dictionaries and lists for JSON serialization"""
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(v) for v in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    else:
        return obj

# Pydantic models
class SetupPredictionRequest(BaseModel):
    setup_id: str

class SetupPredictionResponse(BaseModel):
    setup_id: str
    prediction: int
    probability: float
    confidence: float
    threshold_used: float = 0.5
    model_type: str = "ensemble_lr_calibrated"
    explanation: str
    key_factors: List[str]
    risk_factors: List[str]
    # Ground truth validation fields
    ground_truth: str = None
    ground_truth_label: int = None
    prediction_correct: bool = None
    outperformance_10d: float = None

class ScannerRequest(BaseModel):
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    min_probability: float = 0.7

class PerfectSetupResponse(BaseModel):
    optimal_features: Dict[str, str]
    probability_range: str
    description: str

# ====== DATABASE API LAYER ======

@app.on_event("startup")
async def startup_event():
    """Initialize ML models and database connections"""
    global ensemble_analyzer, knowledge_explainer
    
    try:
        print("🚀 Initializing FastAPI backend...")
        
        # Initialize ensemble analyzer
        ensemble_analyzer = EnsemblePrecisionAnalyzer(
            db_path=DB_PATH,
            output_dir="analysis_ml_ensemble"
        )
        
        # Load existing models if available
        if ensemble_analyzer.load_and_prepare_ensemble_data():
            print("✅ Ensemble analyzer loaded successfully")
        else:
            print("⚠️  Ensemble analyzer initialization failed")
        
        # Initialize knowledge explainer
        knowledge_explainer = KnowledgeExplainer(ensemble_analyzer)
        
        # Initialize knowledge graph
        if init_kg_on_startup:
            print("🧠 Starting Knowledge Graph initialization...")
            init_kg_on_startup()
        
        print("✅ Backend initialization complete")
        
    except Exception as e:
        print(f"❌ Backend initialization failed: {e}")
        traceback.print_exc()

def get_db_connection():
    """Get database connection"""
    return duckdb.connect(DB_PATH)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected" if Path(DB_PATH).exists() else "disconnected",
        "ml_models": "loaded" if ensemble_analyzer else "not_loaded"
    }

@app.get("/api/setups")
async def get_all_setups(
    limit: int = Query(50, description="Number of setups to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Get all confirmed setups with basic information"""
    try:
        conn = get_db_connection()
        query = """
        SELECT 
            l.setup_id,
            l.stock_ticker,
            l.setup_date,
            l.outperformance_10d,
            CASE WHEN l.outperformance_10d > 0 THEN true ELSE false END as outperformed,
            c.company_name,
            c.sector,
            c.market_cap
        FROM labels l
        LEFT JOIN company_info c ON l.stock_ticker = c.ticker
        WHERE l.outperformance_10d IS NOT NULL
        ORDER BY l.setup_date DESC
        LIMIT ? OFFSET ?
        """
        
        df = conn.execute(query, [limit, offset]).df()
        conn.close()
        
        return {
            "setups": safe_df_to_dict(df),
            "total": len(df),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/setup/{setup_id}/fundamentals")
async def get_setup_fundamentals(setup_id: str):
    """Get fundamentals data for a specific setup"""
    try:
        conn = get_db_connection()
        query = """
        SELECT *
        FROM fundamentals f
        JOIN labels l ON f.ticker = l.stock_ticker
        WHERE l.setup_id = ?
        """
        
        df = conn.execute(query, [setup_id]).df()
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Setup fundamentals not found")
        
        records = safe_df_to_dict(df)
        return {
            "setup_id": setup_id,
            "fundamentals": records[0] if records else {}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/setup/{setup_id}/news")
async def get_setup_news(setup_id: str):
    """Get news data for a specific setup"""
    try:
        conn = get_db_connection()
        query = """
        SELECT *
        FROM rns_announcements r
        JOIN labels l ON r.ticker = l.stock_ticker
        WHERE l.setup_id = ?
        ORDER BY r.rns_date DESC
        """
        
        df = conn.execute(query, [setup_id]).df()
        conn.close()
        
        records = safe_df_to_dict(df)
        return {
            "setup_id": setup_id,
            "news": records,
            "count": len(records)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/setup/{setup_id}/userposts")
async def get_setup_userposts(setup_id: str):
    """Get user posts data for a specific setup"""
    try:
        conn = get_db_connection()
        query = """
        SELECT *
        FROM user_posts u
        JOIN labels l ON u.ticker = l.stock_ticker
        WHERE l.setup_id = ?
        ORDER BY u.post_date DESC
        """
        
        df = conn.execute(query, [setup_id]).df()
        conn.close()
        
        records = safe_df_to_dict(df)
        return {
            "setup_id": setup_id,
            "userposts": records,
            "count": len(records)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# ====== REAL-TIME PREDICTIONS ======

@app.post("/api/predict", response_model=SetupPredictionResponse)
async def predict_setup(request: SetupPredictionRequest):
    """Generate prediction for a specific setup"""
    if not ensemble_analyzer:
        raise HTTPException(status_code=503, detail="ML models not loaded")
    
    try:
        setup_id = request.setup_id
        
        # Get prediction from ensemble model
        prediction_result = ensemble_analyzer.predict_single_setup(setup_id)
        
        if "error" in prediction_result:
            raise HTTPException(status_code=404, detail=prediction_result["error"])
        
        # Generate explanation using knowledge explainer
        explanation_result = knowledge_explainer.explain_setup_prediction(
            setup_id, prediction_result
        )
        
        # Clean NaN values from results
        prediction_result = clean_nan_values(prediction_result)
        explanation_result = clean_nan_values(explanation_result)
        
        return SetupPredictionResponse(
            setup_id=setup_id,
            prediction=prediction_result["prediction"],
            probability=prediction_result["probability"],
            confidence=prediction_result.get("confidence", 0.0),
            threshold_used=prediction_result.get("threshold_used", 0.5),
            model_type=prediction_result.get("model_type", "ensemble_lr_calibrated"),
            explanation=explanation_result["explanation"],
            key_factors=explanation_result["key_factors"],
            risk_factors=explanation_result["risk_factors"],
            # Ground truth validation
            ground_truth=prediction_result.get("ground_truth"),
            ground_truth_label=prediction_result.get("ground_truth_label"),
            prediction_correct=prediction_result.get("prediction_correct"),
            outperformance_10d=prediction_result.get("outperformance_10d")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/batch_predict")
async def batch_predict_setups(
    limit: int = Query(10, description="Number of setups to predict"),
    min_probability: float = Query(0.5, description="Minimum probability threshold")
):
    """Run batch predictions on multiple setups"""
    if not ensemble_analyzer:
        raise HTTPException(status_code=503, detail="ML models not loaded")
    
    try:
        # Get recent setups
        conn = get_db_connection()
        query = """
        SELECT id as setup_id, stock_ticker, setup_date
        FROM setups
        WHERE confirmed = true
        ORDER BY setup_date DESC
        LIMIT ?
        """
        
        df = conn.execute(query, [limit]).df()
        conn.close()
        
        predictions = []
        for _, row in df.iterrows():
            try:
                setup_id = row['setup_id']
                prediction_result = ensemble_analyzer.predict_single_setup(setup_id)
                
                if "error" not in prediction_result and prediction_result["probability"] >= min_probability:
                    # Clean NaN values from prediction result
                    clean_result = clean_nan_values(prediction_result)
                    predictions.append({
                        "setup_id": setup_id,
                        "stock_ticker": row['stock_ticker'],
                        "setup_date": row['setup_date'],
                        "prediction": clean_result["prediction"],
                        "probability": clean_result["probability"]
                    })
            except:
                continue
        
        return {
            "predictions": predictions,
            "total_processed": len(df),
            "above_threshold": len(predictions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# ====== KNOWLEDGE EXPLAINER ======

class KnowledgeExplainer:
    """LLM-based knowledge explainer for setup predictions"""
    
    def __init__(self, ensemble_analyzer):
        self.analyzer = ensemble_analyzer
    
    def explain_setup_prediction(self, setup_id: str, prediction_result: Dict) -> Dict[str, Any]:
        """Generate human-readable explanation for a setup prediction"""
        try:
            # Get setup features
            features = self._get_setup_features(setup_id)
            
            # Get feature importance
            feature_importance = self._get_feature_importance()
            
            # Generate explanation components
            explanation = self._generate_explanation(
                setup_id, prediction_result, features, feature_importance
            )
            
            return explanation
            
        except Exception as e:
            return {
                "explanation": f"Could not generate explanation: {str(e)}",
                "key_factors": [],
                "risk_factors": []
            }
    
    def _get_setup_features(self, setup_id: str) -> Dict:
        """Get feature values for a specific setup"""
        try:
            conn = get_db_connection()
            
            # Get fundamentals features using correct schema
            query = """
            WITH latest_fundamentals AS (
                SELECT f.*, 
                       ROW_NUMBER() OVER (PARTITION BY f.ticker ORDER BY f.period_end DESC) as rn
                FROM fundamentals f
            )
            SELECT lf.*, l.stock_ticker
            FROM labels l
            LEFT JOIN latest_fundamentals lf ON l.stock_ticker = lf.ticker AND lf.rn = 1
            WHERE l.setup_id = ?
            """
            fundamentals = conn.execute(query, [setup_id]).df()
            
            # Get text features if available
            news_features = pd.DataFrame()
            userposts_features = pd.DataFrame()
            
            try:
                query = "SELECT * FROM news_features WHERE setup_id = ?"
                news_features = conn.execute(query, [setup_id]).df()
            except:
                pass
            
            try:
                query = "SELECT * FROM userposts_features WHERE setup_id = ?"
                userposts_features = conn.execute(query, [setup_id]).df()
            except:
                pass
            
            conn.close()
            
            # Handle NaN values using numpy
            if not fundamentals.empty:
                fundamentals = fundamentals.replace({np.nan: None})
            if not news_features.empty:
                news_features = news_features.replace({np.nan: None})
            if not userposts_features.empty:
                userposts_features = userposts_features.replace({np.nan: None})
            
            fundamentals_records = safe_df_to_dict(fundamentals)
            news_records = safe_df_to_dict(news_features)
            userposts_records = safe_df_to_dict(userposts_features)
            
            return {
                "fundamentals": fundamentals_records[0] if fundamentals_records else {},
                "news_features": news_records[0] if news_records else {},
                "userposts_features": userposts_records[0] if userposts_records else {}
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_feature_importance(self) -> Dict:
        """Get feature importance from ensemble model"""
        if hasattr(self.analyzer, 'combined_feature_importance'):
            return self.analyzer.combined_feature_importance
        return {}
    
    def _generate_explanation(self, setup_id: str, prediction: Dict, features: Dict, importance: Dict) -> Dict:
        """Generate natural language explanation"""
        
        probability = prediction.get("probability", 0.0)
        prediction_label = prediction.get("prediction", 0)
        
        # Base explanation
        confidence_level = "high" if probability > 0.8 else "moderate" if probability > 0.6 else "low"
        outcome = "likely to outperform" if prediction_label == 1 else "unlikely to outperform"
        
        explanation = f"Based on the analysis, this setup is {outcome} with {confidence_level} confidence ({probability:.1%} probability)."
        
        # Key factors analysis
        key_factors = []
        risk_factors = []
        
        fundamentals = features.get("fundamentals", {})
        
        # Analyze key financial metrics
        if "operating_margin" in fundamentals:
            margin = fundamentals["operating_margin"]
            if margin and margin > 0.1:  # 10%
                key_factors.append(f"Strong operating margin ({margin:.1%})")
            elif margin and margin < 0.05:  # 5%
                risk_factors.append(f"Low operating margin ({margin:.1%})")
        
        if "debt_to_equity" in fundamentals:
            debt_ratio = fundamentals["debt_to_equity"]
            if debt_ratio and debt_ratio < 0.3:
                key_factors.append(f"Low debt-to-equity ratio ({debt_ratio:.2f})")
            elif debt_ratio and debt_ratio > 0.7:
                risk_factors.append(f"High debt-to-equity ratio ({debt_ratio:.2f})")
        
        if "roe" in fundamentals:
            roe = fundamentals["roe"]
            if roe and roe > 0.15:  # 15%
                key_factors.append(f"Excellent return on equity ({roe:.1%})")
            elif roe and roe < 0.05:  # 5%
                risk_factors.append(f"Low return on equity ({roe:.1%})")
        
        # Analyze text features
        news_features = features.get("news_features", {})
        if "sentiment_score" in news_features:
            sentiment = news_features["sentiment_score"]
            if sentiment and sentiment > 0.6:
                key_factors.append(f"Positive news sentiment ({sentiment:.2f})")
            elif sentiment and sentiment < 0.4:
                risk_factors.append(f"Negative news sentiment ({sentiment:.2f})")
        
        # Default factors if none found
        if not key_factors:
            key_factors = ["Market position", "Financial stability"]
        if not risk_factors:
            risk_factors = ["Market volatility", "Sector risks"]
        
        return {
            "explanation": explanation,
            "key_factors": key_factors[:5],  # Limit to top 5
            "risk_factors": risk_factors[:3]  # Limit to top 3
        }

# ====== VISUALIZATION INTEGRATION ======

@app.get("/api/visualizations")
async def get_available_visualizations():
    """Get list of available visualization files"""
    viz_dirs = ["analysis_ml_fundamentals", "analysis_ml_text", "analysis_ml_ensemble"]
    visualizations = {}
    
    for viz_dir in viz_dirs:
        # Fix path to look in parent directory since backend runs from frontend/
        viz_path = Path("..") / viz_dir
        if viz_path.exists():
            png_files = list(viz_path.glob("*.png"))
            txt_files = list(viz_path.glob("*.txt"))
            
            visualizations[viz_dir] = {
                "png_files": [f.name for f in png_files],
                "txt_files": [f.name for f in txt_files],
                "total_files": len(png_files) + len(txt_files)
            }
    
    return visualizations

@app.get("/api/visualization/{category}/{filename}")
async def get_visualization_file(category: str, filename: str):
    """Serve PNG or TXT visualization files"""
    
    allowed_categories = ["analysis_ml_fundamentals", "analysis_ml_text", "analysis_ml_ensemble"]
    if category not in allowed_categories:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # Fix path to look in parent directory since backend runs from frontend/
    file_path = Path("..") / category / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    if filename.endswith('.png'):
        return FileResponse(file_path, media_type="image/png")
    elif filename.endswith('.txt'):
        return FileResponse(file_path, media_type="text/plain")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

@app.get("/api/reports/{category}")
async def get_analysis_report(category: str):
    """Get parsed analysis report content"""
    
    allowed_categories = ["analysis_ml_fundamentals", "analysis_ml_text", "analysis_ml_ensemble"]
    if category not in allowed_categories:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # Fix path to look in parent directory since backend runs from frontend/
    report_path = Path("..") / category
    report_files = list(report_path.glob("*report*.txt"))
    
    if not report_files:
        raise HTTPException(status_code=404, detail="Report file not found")
    
    try:
        with open(report_files[0], 'r') as f:
            content = f.read()
        
        # Parse report content
        parsed_report = _parse_analysis_report(content)
        
        return {
            "category": category,
            "report_file": report_files[0].name,
            "content": content,
            "parsed": parsed_report
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading report: {str(e)}")

def _parse_analysis_report(content: str) -> Dict:
    """Parse analysis report content into structured data"""
    lines = content.split('\n')
    
    parsed = {
        "summary": "",
        "metrics": {},
        "recommendations": []
    }
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        if "SUMMARY" in line.upper():
            current_section = "summary"
        elif "METRICS" in line.upper() or "PERFORMANCE" in line.upper():
            current_section = "metrics"
        elif "RECOMMENDATION" in line.upper():
            current_section = "recommendations"
        elif line and current_section:
            if current_section == "summary":
                parsed["summary"] += line + " "
            elif current_section == "recommendations" and line.startswith("-"):
                parsed["recommendations"].append(line[1:].strip())
    
    return parsed

# ====== SCANNER INTEGRATION ======

@app.post("/api/scan")
async def scan_new_setups(request: ScannerRequest):
    """Scan for new potential setups"""
    try:
        # Get recent setups within date range
        conn = get_db_connection()
        
        date_filter = ""
        params = []
        
        if request.date_from:
            date_filter += " AND setup_date >= ?"
            params.append(request.date_from)
        if request.date_to:
            date_filter += " AND setup_date <= ?"
            params.append(request.date_to)
        
        query = f"""
        SELECT setup_id, stock_ticker, setup_date, 
               CASE WHEN outperformance_10d IS NOT NULL THEN true ELSE false END as confirmed
        FROM labels
        WHERE 1=1 {date_filter}
        ORDER BY setup_date DESC
        LIMIT 100
        """
        
        df = conn.execute(query, params).df()
        conn.close()
        
        # Run predictions on these setups
        potential_setups = []
        
        for _, row in df.iterrows():
            try:
                setup_id = row['setup_id']
                prediction_result = ensemble_analyzer.predict_single_setup(setup_id)
                
                if "error" not in prediction_result:
                    probability = prediction_result["probability"]
                    
                    if probability >= request.min_probability:
                        potential_setups.append({
                            "setup_id": setup_id,
                            "stock_ticker": row['stock_ticker'],
                            "setup_date": row['setup_date'],
                            "confirmed": row['confirmed'],
                            "probability": probability,
                            "prediction": prediction_result["prediction"]
                        })
            except:
                continue
        
        # Sort by probability
        potential_setups.sort(key=lambda x: x["probability"], reverse=True)
        
        return {
            "scanned_setups": len(df),
            "potential_setups": potential_setups,
            "threshold": request.min_probability,
            "scan_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scanner error: {str(e)}")

# ====== PERFECT SETUP ANALYSIS ======

@app.get("/api/perfect_setup", response_model=PerfectSetupResponse)
async def get_perfect_setup():
    """Get optimal feature values for highest probability setups"""
    try:
        # Analyze top performing setups
        conn = get_db_connection()
        
        query = """
        SELECT f.*, l.outperformance_10d
        FROM fundamentals f
        JOIN labels l ON f.ticker = l.stock_ticker
        WHERE l.outperformance_10d > 0
        """
        
        df = conn.execute(query).df()
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No outperforming setups found")
        
        # Calculate optimal ranges
        optimal_features = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['operating_margin', 'net_margin', 'roe', 'debt_to_equity', 'current_ratio']:
                values = df[col].dropna()
                if not values.empty:
                    if col == 'debt_to_equity':
                        # Lower is better for debt
                        optimal_features[col] = f"< {values.quantile(0.25):.2f}"
                    else:
                        # Higher is better for margins and ratios
                        optimal_features[col] = f"> {values.quantile(0.75):.2f}"
        
        return PerfectSetupResponse(
            optimal_features=optimal_features,
            probability_range="> 85%",
            description="These feature ranges represent the top quartile of historically outperforming setups"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Perfect setup analysis error: {str(e)}")

# ====== STATIC FILE SERVING ======

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    """Serve the main frontend page"""
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RAG Pipeline Frontend Backend...")
    print("📊 Frontend will be available at: http://localhost:8000")
    print("🔗 API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "backend:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    ) 