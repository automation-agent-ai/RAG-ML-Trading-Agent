#!/usr/bin/env python3
"""
Enhanced Fundamentals Agent with DuckDB Integration
==================================================

This agent handles both:
1. Structured fundamental metrics (ROE, D/E, margins, etc.) - no LLM needed
2. LLM-extracted features moved from news agent:
   - Financial Results (7 features): earnings, profit warnings, etc.
   - Corporate Actions (2 features): capital_raise_present, synthetic_summary

This follows the smart feature split strategy where quarterly/annual fundamental 
data is processed by the fundamentals agent instead of the news agent.

Author: Enhanced Fundamentals Agent
Date: 2025-01-11
"""

import os
import sys
import json
import hashlib
import time
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import duckdb
import lancedb
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
import numpy as np

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import label converter
from core.label_converter import LabelConverter, get_class_distribution

# Import setup validator
from tools.setup_validator_duckdb import SetupValidatorDuckDB

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FundamentalsFeatureSchema(BaseModel):
    """Pydantic schema for Fundamentals feature extraction"""
    
    # Metadata
    setup_id: str
    extraction_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    llm_model: str = "gpt-4o-mini"
    
    # Structured fundamental metrics (no LLM needed)
    roa: Optional[float] = Field(default=None, description="Return on Assets")
    roe: Optional[float] = Field(default=None, description="Return on Equity")
    debt_to_equity: Optional[float] = Field(default=None, description="Debt to Equity ratio")
    current_ratio: Optional[float] = Field(default=None, description="Current ratio")
    gross_margin: Optional[float] = Field(default=None, description="Gross margin %")
    net_margin: Optional[float] = Field(default=None, description="Net margin %")
    revenue_growth: Optional[float] = Field(default=None, description="Revenue growth %")
    
    # LLM-extracted features (moved from news agent - Financial Results group)
    count_financial_results: int = Field(ge=0, default=0)
    max_severity_financial_results: float = Field(ge=0.0, le=1.0, default=0.0)
    avg_headline_spin_financial_results: str = Field(default="neutral", pattern="^(positive|negative|neutral|uncertain)$")
    sentiment_score_financial_results: float = Field(ge=-1.0, le=1.0, default=0.0)
    profit_warning_present: bool = Field(default=False)
    synthetic_summary_financial_results: str = Field(max_length=240, default="")
    cot_explanation_financial_results: str = Field(default="")
    
    # LLM-extracted features (moved from news agent - Corporate Actions subset)
    capital_raise_present: bool = Field(default=False)
    synthetic_summary_corporate_actions: str = Field(max_length=240, default="")


class EnhancedFundamentalsAgentDuckDB:
    """
    Enhanced Fundamentals Agent with DuckDB integration
    
    Combines structured fundamental metrics with LLM-extracted features
    from earnings-related RNS announcements.
    """
    
    def __init__(
        self,
        db_path: str = "../data/sentiment_system.duckdb",
        lancedb_dir: str = "../lancedb_store",
        table_name: str = "fundamentals_embeddings",
        llm_model: str = "gpt-4o-mini",
        mode: str = "training"  # Either "training" or "prediction"
    ):
        self.db_path = db_path
        self.lancedb_dir = lancedb_dir
        self.table_name = table_name
        self.llm_model = llm_model
        self.mode = mode
        
        # Initialize setup validator
        self.setup_validator = SetupValidatorDuckDB(db_path)
        
        # Initialize OpenAI client
        self.openai_client = OpenAI()
        
        # Connect to LanceDB
        self._connect_to_lancedb()
        
        # Initialize DuckDB feature storage
        self._init_duckdb_feature_storage()
        
        logger.info(f"Enhanced Fundamentals Agent (DuckDB) initialized successfully in {mode} mode")
    
    def _connect_to_lancedb(self):
        """Connect to LanceDB table"""
        try:
            db = lancedb.connect(self.lancedb_dir)
            
            # In prediction mode, we don't need to open any tables
            if hasattr(self, 'mode') and self.mode == 'prediction':
                logger.info("Prediction mode: Skipping table opening")
                self.table = None
                self.training_table = None
                return
            
            # Try to open the table
            try:
                self.table = db.open_table(self.table_name)
                logger.info(f"Connected to LanceDB table: {self.table_name}")
            except Exception as e:
                logger.warning(f"Could not open table {self.table_name}: {e}")
                self.table = None
            
            # Initialize training table for similarity search
            self.training_table = None
            try:
                # Try to open with new table name first, then fall back to old name
                try:
                    self.training_table = db.open_table("fundamentals_embeddings_training")
                    logger.info("Connected to fundamentals_embeddings_training table for similarity search")
                except Exception:
                    # Fall back to old table name for backward compatibility
                    try:
                        self.training_table = db.open_table("fundamentals_embeddings")
                        logger.info("Connected to fundamentals_embeddings table for similarity search (legacy name)")
                    except Exception as e:
                        logger.warning(f"Could not open training embeddings table: {e}")
            except Exception as e:
                logger.warning(f"Could not open training embeddings table: {e}")
        except Exception as e:
            logger.error(f"Error connecting to LanceDB: {e}")
            # In prediction mode, we can continue without LanceDB
            if hasattr(self, 'mode') and self.mode == 'prediction':
                logger.warning("Continuing in prediction mode without LanceDB connection")
                self.db = None
                self.table = None
                self.training_table = None
            else:
                raise
    
    def _init_duckdb_feature_storage(self):
        """Initialize DuckDB tables for feature storage"""
        conn = duckdb.connect(self.db_path)
        
        # Create fundamentals_features table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS fundamentals_features (
                setup_id TEXT PRIMARY KEY,
                -- Structured metrics (from fundamentals.csv)
                roa REAL DEFAULT NULL,
                roe REAL DEFAULT NULL,
                debt_to_equity REAL DEFAULT NULL,
                current_ratio REAL DEFAULT NULL,
                gross_margin REAL DEFAULT NULL,
                net_margin REAL DEFAULT NULL,
                revenue_growth REAL DEFAULT NULL,
                -- Financial Results (moved from news agent - 7 features)
                count_financial_results INTEGER DEFAULT 0,
                max_severity_financial_results REAL DEFAULT 0.0,
                avg_headline_spin_financial_results TEXT DEFAULT 'neutral',
                sentiment_score_financial_results REAL DEFAULT 0.0,
                profit_warning_present BOOLEAN DEFAULT FALSE,
                synthetic_summary_financial_results TEXT DEFAULT '',
                cot_explanation_financial_results TEXT DEFAULT '',
                -- Corporate Actions subset (moved from news agent - 2 features)
                capital_raise_present BOOLEAN DEFAULT FALSE,
                synthetic_summary_corporate_actions TEXT DEFAULT '',
                -- Metadata
                extraction_timestamp TIMESTAMP,
                llm_model TEXT,
                CONSTRAINT chk_headline_spin_fr CHECK (avg_headline_spin_financial_results IN ('positive', 'negative', 'neutral', 'uncertain'))
            )
        ''')
        
        conn.close()
        logger.info("DuckDB fundamentals_features table initialized")
    
    def extract_structured_metrics(self, setup_id: str) -> Dict[str, float]:
        """
        Extract structured fundamental metrics from DuckDB
        These don't need LLM processing, just direct database lookup
        """
        conn = duckdb.connect(self.db_path)
        
        try:
            # Get setup info for ticker matching
            setup_query = "SELECT lse_ticker FROM setups WHERE setup_id = ?"
            setup_result = conn.execute(setup_query, [setup_id]).fetchone()
            
            if not setup_result:
                logger.warning(f"Setup {setup_id} not found")
                return {}
                
            lse_ticker = setup_result[0]
            
            # Try different ticker formats for matching
            ticker_variants = [lse_ticker]
            if lse_ticker.endswith('.L'):
                ticker_variants.append(lse_ticker[:-2])  # Remove .L suffix
            else:
                ticker_variants.append(f"{lse_ticker}.L")  # Add .L suffix
            
            # Get fundamental metrics for this setup with ticker matching
            query = """
                SELECT 
                    -- Calculate ratios from raw data
                    CASE WHEN f.total_assets > 0 THEN f.net_income::FLOAT / f.total_assets ELSE NULL END as roa,
                    CASE WHEN f.total_equity > 0 THEN f.net_income::FLOAT / f.total_equity ELSE NULL END as roe,
                    CASE WHEN f.total_equity > 0 THEN f.total_debt::FLOAT / f.total_equity ELSE NULL END as debt_to_equity,
                    CASE WHEN f.current_liabilities > 0 THEN f.current_assets::FLOAT / f.current_liabilities ELSE NULL END as current_ratio,
                    CASE WHEN f.total_revenue > 0 THEN f.gross_profit::FLOAT / f.total_revenue * 100 ELSE NULL END as gross_margin,
                    CASE WHEN f.total_revenue > 0 THEN f.net_income::FLOAT / f.total_revenue * 100 ELSE NULL END as net_margin,
                    NULL as revenue_growth  -- Would need historical data to calculate
                FROM fundamentals f
                WHERE f.ticker = ANY(?)
                ORDER BY f.date DESC
                LIMIT 1
            """
            
            result = conn.execute(query, [ticker_variants]).fetchone()
            
            if result:
                metrics = {
                    'roa': result[0],
                    'roe': result[1], 
                    'debt_to_equity': result[2],
                    'current_ratio': result[3],
                    'gross_margin': result[4],
                    'net_margin': result[5],
                    'revenue_growth': result[6]
                }
                logger.info(f"Extracted structured metrics for setup {setup_id}")
                return metrics
            else:
                logger.warning(f"No fundamental metrics found for setup {setup_id} (tried tickers: {ticker_variants})")
                return {}
                
        except Exception as e:
            logger.error(f"Error extracting structured metrics for {setup_id}: {e}")
            return {}
        finally:
            conn.close()
    
    def retrieve_financial_results_news(self, setup_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve financial results RNS news for LLM processing
        These are earnings, profit warnings, financial performance announcements
        """
        conn = duckdb.connect(self.db_path)
        
        try:
            # Get financial results RNS news
            query = """
                SELECT headline, text
                FROM rns_announcements r
                JOIN setups s ON r.ticker = s.lse_ticker
                WHERE s.setup_id = ?
                AND r.rns_date <= s.spike_timestamp
                AND r.rns_date >= (s.spike_timestamp - INTERVAL '90 days')
                ORDER BY r.rns_date DESC
            """
            
            results = conn.execute(query, [setup_id]).fetchall()
            
            news_items = []
            for row in results:
                news_items.append({
                    'headline': row[0],
                    'text': row[1],
                    'category': 'RNS',  # Default category since column doesn't exist
                    'severity': 0.0     # Default severity since column doesn't exist
                })
            
            logger.info(f"Retrieved {len(news_items)} financial results items for setup {setup_id}")
            return news_items
            
        except Exception as e:
            logger.error(f"Error retrieving financial results news for {setup_id}: {e}")
            return []
        finally:
            conn.close()
    
    def extract_llm_features(self, setup_id: str, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract LLM features from financial results RNS news
        """
        if not news_items:
            return self._create_empty_llm_features()
        
        # Create context for LLM
        context = self._format_news_for_llm(news_items)
        
        prompt = f"""You are analyzing financial results and earnings-related RNS announcements for trading setup {setup_id}.

**Context:** {context}

Extract the following features as JSON:

{{
  "count_financial_results": <number of financial results announcements>,
  "max_severity_financial_results": <highest severity score 0.0-1.0>,
  "avg_headline_spin_financial_results": "<positive|negative|neutral|uncertain>",
  "sentiment_score_financial_results": <overall sentiment -1.0 to 1.0>,
  "profit_warning_present": <true if any profit warnings>,
  "synthetic_summary_financial_results": "<summary ≤240 chars>",
  "cot_explanation_financial_results": "<brief explanation>",
  "capital_raise_present": <true if capital raising mentioned>,
  "synthetic_summary_corporate_actions": "<capital actions summary ≤240 chars>"
}}

Focus only on financial performance, earnings, profit warnings, and capital structure changes."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if result_text.startswith('```json'):
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif result_text.startswith('```'):
                result_text = result_text.split('```')[1].split('```')[0].strip()
            
            llm_features = json.loads(result_text)
            logger.info(f"Extracted LLM features for setup {setup_id}")
            return llm_features
            
        except Exception as e:
            logger.error(f"Error extracting LLM features for {setup_id}: {e}")
            return self._create_empty_llm_features()
    
    def _format_news_for_llm(self, news_items: List[Dict[str, Any]]) -> str:
        """Format news items for LLM context"""
        context_parts = []
        for i, item in enumerate(news_items[:10], 1):  # Limit to 10 items
            context_parts.append(f"{i}. [{item['category']}] {item['headline']}")
            if item['text']:
                context_parts.append(f"   Text: {item['text'][:200]}...")
        
        return "\n".join(context_parts)
    
    def _create_empty_llm_features(self) -> Dict[str, Any]:
        """Create empty LLM features when no data available"""
        return {
            "count_financial_results": 0,
            "max_severity_financial_results": 0.0,
            "avg_headline_spin_financial_results": "neutral",
            "sentiment_score_financial_results": 0.0,
            "profit_warning_present": False,
            "synthetic_summary_financial_results": "No financial results announcements found",
            "cot_explanation_financial_results": "No relevant RNS data available",
            "capital_raise_present": False,
            "synthetic_summary_corporate_actions": "No capital actions identified"
        }
    
    def process_setup(self, setup_id: str, mode: str = None) -> Optional[FundamentalsFeatureSchema]:
        """
        Process a single setup ID to extract and store features
        
        Args:
            setup_id: Setup ID to process
            mode: Either 'training' or 'prediction' (overrides self.mode if provided)
            
        Returns:
            Extracted features or None if processing failed
        """
        # Use provided mode or fall back to instance mode
        current_mode = mode if mode is not None else self.mode
        logger.info(f"Processing fundamentals for setup: {setup_id} in {current_mode} mode")
        
        try:
            # Step 1: Extract structured metrics
            structured_metrics = self.extract_structured_metrics(setup_id)
            
            # Step 2: Retrieve financial results news
            news_items = self.retrieve_financial_results_news(setup_id)
            
            # Step 3: Extract LLM features
            llm_features = self.extract_llm_features(setup_id, news_items)
            
            # Step 4: Combine features
            features = FundamentalsFeatureSchema(
                setup_id=setup_id,
                extraction_timestamp=datetime.now().isoformat(),
                llm_model=self.llm_model,
                **structured_metrics,
                **llm_features
            )
            
            # Step 5: Store features
            self.store_features(features)
            
            # Step 6: In prediction mode, generate similarity-based predictions
            if current_mode == 'prediction' and hasattr(self, 'training_table') and self.training_table is not None:
                # Create content for similarity search
                financial_summary = f"Financial metrics: ROA={features.roa}, ROE={features.roe}, " + \
                                   f"Debt/Equity={features.debt_to_equity}, Current Ratio={features.current_ratio}, " + \
                                   f"Gross Margin={features.gross_margin}%, Net Margin={features.net_margin}%, " + \
                                   f"Revenue Growth={features.revenue_growth}%"
                
                if features.synthetic_summary_financial_results:
                    financial_summary += f" | {features.synthetic_summary_financial_results}"
                
                # Generate prediction
                if hasattr(self, 'predict_via_similarity'):
                    prediction = self.predict_via_similarity(financial_summary)
                    if prediction:
                        prediction['setup_id'] = setup_id
                        prediction['domain'] = 'fundamentals'
                        prediction['prediction_timestamp'] = datetime.now().isoformat()
                        logger.info(f"Generated similarity prediction for {setup_id}: {prediction.get('predicted_outperformance', 'N/A')}")
                        return features, prediction
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing fundamentals for setup {setup_id}: {e}")
            return None
    
    def store_features(self, features: FundamentalsFeatureSchema):
        """Store features in DuckDB using proper DuckDB syntax"""
        conn = duckdb.connect(self.db_path)
        
        try:
            # Use DuckDB ON CONFLICT syntax instead of INSERT OR REPLACE
            conn.execute('''
                INSERT INTO fundamentals_features (
                    setup_id, roa, roe, debt_to_equity, current_ratio,
                    gross_margin, net_margin, revenue_growth,
                    count_financial_results, max_severity_financial_results,
                    avg_headline_spin_financial_results, sentiment_score_financial_results,
                    profit_warning_present, synthetic_summary_financial_results,
                    cot_explanation_financial_results, capital_raise_present,
                    synthetic_summary_corporate_actions, extraction_timestamp, llm_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (setup_id) DO UPDATE SET
                    roa = EXCLUDED.roa,
                    roe = EXCLUDED.roe,
                    debt_to_equity = EXCLUDED.debt_to_equity,
                    current_ratio = EXCLUDED.current_ratio,
                    gross_margin = EXCLUDED.gross_margin,
                    net_margin = EXCLUDED.net_margin,
                    revenue_growth = EXCLUDED.revenue_growth,
                    count_financial_results = EXCLUDED.count_financial_results,
                    max_severity_financial_results = EXCLUDED.max_severity_financial_results,
                    avg_headline_spin_financial_results = EXCLUDED.avg_headline_spin_financial_results,
                    sentiment_score_financial_results = EXCLUDED.sentiment_score_financial_results,
                    profit_warning_present = EXCLUDED.profit_warning_present,
                    synthetic_summary_financial_results = EXCLUDED.synthetic_summary_financial_results,
                    cot_explanation_financial_results = EXCLUDED.cot_explanation_financial_results,
                    capital_raise_present = EXCLUDED.capital_raise_present,
                    synthetic_summary_corporate_actions = EXCLUDED.synthetic_summary_corporate_actions,
                    extraction_timestamp = EXCLUDED.extraction_timestamp,
                    llm_model = EXCLUDED.llm_model
            ''', (
                features.setup_id, features.roa, features.roe, features.debt_to_equity,
                features.current_ratio, features.gross_margin, features.net_margin,
                features.revenue_growth, features.count_financial_results,
                features.max_severity_financial_results, features.avg_headline_spin_financial_results,
                features.sentiment_score_financial_results, features.profit_warning_present,
                features.synthetic_summary_financial_results, features.cot_explanation_financial_results,
                features.capital_raise_present, features.synthetic_summary_corporate_actions,
                features.extraction_timestamp, features.llm_model
            ))
            
            conn.close()
            logger.info(f"Stored fundamentals features for setup {features.setup_id}")
            
        except Exception as e:
            logger.error(f"Error storing features for {features.setup_id}: {e}")
            conn.close()

    def batch_process_setups(self, setup_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple setups in batch with proper progress tracking
        """
        results = []
        total_setups = len(setup_ids)
        successful = 0
        failed = 0
        
        logger.info(f"Starting batch processing of {total_setups} setups")
        start_time = time.time()
        
        for i, setup_id in enumerate(setup_ids, 1):
            try:
                logger.info(f"Processing {i}/{total_setups}: {setup_id}")
                
                # Process the setup
                features = self.process_setup(setup_id)
                
                if features:
                    results.append({
                        'setup_id': setup_id,
                        'success': True,
                        'features': features.dict(),
                        'timestamp': datetime.now().isoformat()
                    })
                    successful += 1
                else:
                    results.append({
                        'setup_id': setup_id,
                        'success': False,
                        'error': 'Processing returned None',
                        'timestamp': datetime.now().isoformat()
                    })
                    failed += 1
                
                # Progress logging every 50 setups
                if i % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed
                    remaining = (total_setups - i) / rate if rate > 0 else 0
                    logger.info(f"Progress: {i}/{total_setups} ({i/total_setups*100:.1f}%) - "
                              f"{successful} successful, {failed} failed - "
                              f"Rate: {rate:.1f} setups/sec - ETA: {remaining/60:.1f} min")
                
            except Exception as e:
                logger.error(f"Error processing setup {setup_id}: {e}")
                results.append({
                    'setup_id': setup_id,
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                failed += 1
        
        total_time = time.time() - start_time
        logger.info(f"Batch processing completed: {successful} successful, {failed} failed in {total_time/60:.1f} minutes")
        
        return results

    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'setup_validator'):
                self.setup_validator.close()
            logger.info("Fundamentals agent cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def find_similar_training_embeddings(self, query: Union[np.ndarray, str], limit: int = 10) -> List[Dict]:
        """
        Find similar training embeddings with labels
        
        Args:
            query: Either a numpy array embedding or text content to embed
            limit: Maximum number of similar cases to return
            
        Returns:
            List of similar cases with their metadata and labels
        """
        if not hasattr(self, 'training_table') or self.training_table is None:
            logger.warning("Training embeddings table not available")
            return []
        
        try:
            # Handle text input by creating embedding
            if isinstance(query, str):
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                query = model.encode(query)
            
            # Ensure numpy array
            if not isinstance(query, np.ndarray):
                query = np.array(query)
                
            # Search for similar cases
            results = self.training_table.search(query).limit(limit).to_pandas()
            return results.to_dict('records')
        except Exception as e:
            logger.error(f"Error searching similar embeddings: {e}")
            return []
    
    def compute_similarity_features(self, similar_cases: List[Dict]) -> Dict[str, float]:
        """
        Compute similarity-based features from similar cases
        
        Args:
            similar_cases: List of similar cases with their metadata and labels
            
        Returns:
            Dictionary of similarity-based features
        """
        if not similar_cases:
            return {
                'positive_signal_strength': 0.0,
                'negative_risk_score': 0.0,
                'neutral_probability': 0.0,
                'historical_pattern_confidence': 0.0,
                'similar_cases_count': 0
            }
        
        # Use label converter for classification
        label_converter = LabelConverter()
        performance_values = [c.get('outperformance_10d', 0) for c in similar_cases]
        
        # Calculate performance-based features
        positive_cases = [c for c in similar_cases if label_converter.outperformance_to_class_int(c.get('outperformance_10d', 0)) == 1]
        negative_cases = [c for c in similar_cases if label_converter.outperformance_to_class_int(c.get('outperformance_10d', 0)) == -1]
        neutral_cases = [c for c in similar_cases if label_converter.outperformance_to_class_int(c.get('outperformance_10d', 0)) == 0]
        
        total_cases = len(similar_cases)
        
        # Calculate weighted features based on similarity scores
        similarity_scores = [c.get('_distance', 1.0) for c in similar_cases]  # Lower distance = higher similarity
        max_distance = max(similarity_scores) if similarity_scores else 1.0
        similarity_weights = [1 - (score / max_distance) for score in similarity_scores]
        
        # Calculate weighted ratios (with safety check)
        sum_weights = sum(similarity_weights) if similarity_weights else 1.0
        if sum_weights == 0:
            sum_weights = 1.0
            
        # Use label converter thresholds
        positive_ratio = sum(w for c, w in zip(similar_cases, similarity_weights) 
                           if label_converter.outperformance_to_class_int(c.get('outperformance_10d', 0)) == 1) / sum_weights
        
        negative_ratio = sum(w for c, w in zip(similar_cases, similarity_weights)
                           if label_converter.outperformance_to_class_int(c.get('outperformance_10d', 0)) == -1) / sum_weights
        
        # Calculate pattern confidence based on consistency
        performance_std = np.std(performance_values) if performance_values else 1.0
        pattern_confidence = 1.0 / (1.0 + performance_std)  # Higher std = lower confidence
        
        return {
            'positive_signal_strength': float(positive_ratio),
            'negative_risk_score': float(negative_ratio),
            'neutral_probability': float(len(neutral_cases) / total_cases),
            'historical_pattern_confidence': float(pattern_confidence),
            'similar_cases_count': total_cases
        }
    
    def predict_via_similarity(self, query: Union[np.ndarray, str]) -> Dict[str, Any]:
        """
        Direct prediction using similarity to training embeddings
        
        Args:
            query: Either a numpy array embedding or text content to embed
            
        Returns:
            Dictionary with prediction results
        """
        # Find similar cases
        similar_cases = self.find_similar_training_embeddings(query, limit=10)
        
        if not similar_cases:
            logger.warning("No similar cases found for prediction")
            return {}
            
        # Extract labels from similar cases
        similar_labels = [case.get('outperformance_10d', 0.0) for case in similar_cases]
        
        # Compute weighted average prediction
        similarity_scores = [1 - case.get('_distance', 1.0) for case in similar_cases]
        sum_scores = sum(similarity_scores) if similarity_scores else 1.0
        if sum_scores == 0:
            sum_scores = 1.0
            
        weighted_prediction = sum(label * score for label, score in zip(similar_labels, similarity_scores)) / sum_scores
        
        # Compute confidence metrics
        prediction = {
            'predicted_outperformance': float(weighted_prediction),
            'confidence': float(1.0 / (1.0 + np.std(similar_labels))),
            'positive_ratio': sum(1 for l in similar_labels if l > 0.02) / len(similar_labels),
            'negative_ratio': sum(1 for l in similar_labels if l < -0.02) / len(similar_labels),
            'neutral_ratio': sum(1 for l in similar_labels if abs(l) <= 0.02) / len(similar_labels),
            'similar_cases_count': len(similar_cases)
        }
        
        return prediction


if __name__ == "__main__":
    # Example usage
    agent = EnhancedFundamentalsAgentDuckDB()
    
    # Test with a setup
    features = agent.process_setup("KZG_2024-10-16")
    if features:
        print(f"Processed fundamentals for {features.setup_id}")
        print(f"ROE: {features.roe}")
        print(f"Financial Results Count: {features.count_financial_results}")
        print(f"Profit Warning: {features.profit_warning_present}") 