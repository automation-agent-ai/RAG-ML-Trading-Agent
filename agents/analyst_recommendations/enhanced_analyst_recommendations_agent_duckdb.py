#!/usr/bin/env python3
"""
Enhanced Analyst Recommendations Agent with DuckDB Integration

Processes analyst recommendations data and extracts sophisticated features
for price target analysis, consensus ratings, and recommendation momentum.
"""

import duckdb
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from openai import OpenAI
import lancedb
from sentence_transformers import SentenceTransformer
import time
import os

# Add project root to path for proper imports
import sys
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.setup_validator_duckdb import SetupValidatorDuckDB
from core.label_converter import LabelConverter, get_class_distribution
# Force offline mode for model loading
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_CACHE'] = os.path.join('models', 'cache')
os.environ['HF_HOME'] = os.path.join('models', 'hub')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join('models', 'sentence_transformers')


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedAnalystRecommendationsAgentDuckDB:
    """
    Enhanced Analyst Recommendations Agent with DuckDB integration
    
    Processes analyst recommendations data from DuckDB and extracts features for:
    - Price target analysis and momentum
    - Consensus rating changes and sentiment
    - Analyst coverage breadth and conviction
    - Recommendation timing and momentum shifts
    """
    
    def __init__(
        self,
        db_path: str = "../data/sentiment_system.duckdb",
        lancedb_dir: str = "../lancedb_store", 
        table_name: str = "analyst_recommendations_embeddings",
        llm_model: str = "gpt-4o-mini",
        mode: str = "training",  # Either "training" or "prediction"
        use_cached_models: bool = False,
        local_files_only: bool = False
    ):
        """
        Initialize the Enhanced Analyst Recommendations Agent
        
        Args:
            db_path: Path to DuckDB database
            lancedb_dir: Path to LanceDB directory
            table_name: Name of the embedding table
            llm_model: LLM model to use
            mode: Either "training" or "prediction"
            use_cached_models: Whether to use cached models
            local_files_only: Whether to only use local files
        """
        self.db_path = db_path
        self.lancedb_dir = lancedb_dir
        self.table_name = table_name
        self.llm_model = llm_model
        self.mode = mode
        self.use_cached_models = use_cached_models
        self.local_files_only = local_files_only
        
        if self.use_cached_models:
            logger.info('Using cached models (offline mode)')
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
        
        # Initialize setup validator
        self.setup_validator = SetupValidatorDuckDB(db_path)
        
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Connect to LanceDB
        self._connect_to_lancedb()
        
        # Initialize DuckDB feature storage
        self._init_duckdb_feature_storage()
        
        logger.info(f"Enhanced Analyst Recommendations Agent (DuckDB) initialized successfully in {mode} mode")
    
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
                
            # Try to open existing table or note that it needs to be created
            try:
                self.table = db.open_table(self.table_name)
                logger.info(f"Connected to existing LanceDB table: {self.table_name}")
            except Exception as e:
                logger.info(f"LanceDB table {self.table_name} will be created during embedding process")
                self.table = None
                
            # Initialize training table for similarity search
            self.training_table = None
            try:
                # Try to open with new table name first, then fall back to old name
                try:
                    self.training_table = db.open_table("analyst_recommendations_embeddings_training")
                    logger.info("Connected to analyst_recommendations_embeddings_training table for similarity search")
                except Exception:
                    # Fall back to old table name for backward compatibility
                    try:
                        self.training_table = db.open_table("analyst_recommendations_embeddings")
                        logger.info("Connected to analyst_recommendations_embeddings table for similarity search (legacy name)")
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
        """Initialize DuckDB tables for analyst recommendations feature storage"""
        conn = duckdb.connect(self.db_path)
        
        # Create analyst_recommendations_features table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS analyst_recommendations_features (
                setup_id TEXT PRIMARY KEY,
                -- Non-LLM features (computed from data)
                recommendation_count INTEGER DEFAULT 0,
                buy_recommendations INTEGER DEFAULT 0,
                sell_recommendations INTEGER DEFAULT 0,
                hold_recommendations INTEGER DEFAULT 0,
                avg_price_target REAL DEFAULT NULL,
                price_target_vs_current REAL DEFAULT NULL,
                price_target_spread REAL DEFAULT NULL,
                coverage_breadth INTEGER DEFAULT 0,
                -- LLM-derived features
                consensus_rating REAL DEFAULT 3.0,
                recent_upgrades INTEGER DEFAULT 0,
                recent_downgrades INTEGER DEFAULT 0,
                analyst_conviction_score REAL DEFAULT 0.5,
                recommendation_momentum TEXT DEFAULT 'stable',
                synthetic_analyst_summary TEXT DEFAULT '',
                cot_explanation_analyst TEXT DEFAULT '',
                predicted_outperformance_10d REAL DEFAULT 0.0,
                -- Metadata
                extraction_timestamp TIMESTAMP,
                llm_model TEXT,
                CONSTRAINT chk_consensus_rating CHECK (consensus_rating >= 1.0 AND consensus_rating <= 5.0),
                CONSTRAINT chk_recommendation_momentum CHECK (recommendation_momentum IN ('improving', 'stable', 'deteriorating'))
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("DuckDB analyst_recommendations_features table initialized")
    
    def get_analyst_recommendations_for_setup(self, setup_id: str) -> pd.DataFrame:
        """Retrieve analyst recommendations data for a specific setup"""
        conn = duckdb.connect(self.db_path)
        
        try:
            # Get setup details first
            setup_query = """
                SELECT lse_ticker, yahoo_ticker, spike_timestamp 
                FROM setups 
                WHERE setup_id = ?
            """
            setup_result = conn.execute(setup_query, [setup_id]).fetchone()
            
            if not setup_result:
                logger.warning(f"Setup {setup_id} not found")
                return pd.DataFrame()
            
            lse_ticker, yahoo_ticker, spike_timestamp = setup_result
            spike_date = pd.to_datetime(spike_timestamp)
            
            # Get analyst recommendations for this ticker
            # For historical analysis, use all available analyst data regardless of date
            # (analyst data is recent snapshots applied to historical setups for training)
            
            # Handle ticker format variations: try both plain and .L suffix matching
            recommendations_query = """
                SELECT *
                FROM analyst_recommendations 
                WHERE (ticker = ? OR ticker = ? OR ticker = ? OR ticker = ?)
                ORDER BY created_at DESC
            """
            
            # Try multiple ticker variations
            ticker_variations = [
                lse_ticker,
                lse_ticker + '.L' if lse_ticker else None,
                yahoo_ticker,
                yahoo_ticker + '.L' if yahoo_ticker else None
            ]
            ticker_variations = [t for t in ticker_variations if t]  # Remove None values
            
            # Pad with empty strings if we have fewer than 4 variations
            while len(ticker_variations) < 4:
                ticker_variations.append('')
            
            recommendations_df = conn.execute(
                recommendations_query, 
                ticker_variations
            ).df()
            
            logger.info(f"Found {len(recommendations_df)} analyst recommendations for {setup_id} ({lse_ticker})")
            return recommendations_df
            
        except Exception as e:
            logger.error(f"Error retrieving analyst recommendations for {setup_id}: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def extract_structured_metrics(self, setup_id: str, recommendations_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract structured (non-LLM) metrics from analyst recommendations data"""
        
        if recommendations_df.empty:
            return {
                'recommendation_count': 0,
                'buy_recommendations': 0,
                'sell_recommendations': 0, 
                'hold_recommendations': 0,
                'avg_price_target': None,
                'price_target_vs_current': None,
                'price_target_spread': None,
                'coverage_breadth': 0
            }
        
        # Calculate recommendation counts
        total_recs = len(recommendations_df)
        buy_count = recommendations_df['strong_buy'].sum() + recommendations_df['buy'].sum()
        sell_count = recommendations_df['strong_sell'].sum() + recommendations_df['sell'].sum()
        hold_count = recommendations_df['hold'].sum()
        
        # Calculate price target metrics
        mean_ratings = recommendations_df['mean_rating'].dropna()
        avg_rating = mean_ratings.mean() if not mean_ratings.empty else None
        
        # For price targets, we'd need current price - placeholder calculation
        # In real implementation, you'd get current price from price_data table
        price_target_vs_current = None  # (avg_target - current_price) / current_price
        price_target_spread = None      # max_target - min_target
        
        # Coverage breadth (number of unique analysts/periods)
        coverage_breadth = recommendations_df['period'].nunique() if 'period' in recommendations_df.columns else total_recs
        
        return {
            'recommendation_count': total_recs,
            'buy_recommendations': int(buy_count),
            'sell_recommendations': int(sell_count),
            'hold_recommendations': int(hold_count),
            'avg_price_target': float(avg_rating) if avg_rating else None,
            'price_target_vs_current': price_target_vs_current,
            'price_target_spread': price_target_spread,
            'coverage_breadth': int(coverage_breadth)
        }
    
    def create_llm_prompt(self, setup_id: str, recommendations_df: pd.DataFrame) -> str:
        """Create LLM prompt for analyst recommendations feature extraction with few-shot learning"""
        
        if recommendations_df.empty:
            return f"""
Analyze analyst recommendations for setup {setup_id}:

NO ANALYST RECOMMENDATIONS AVAILABLE

Please provide analysis based on lack of coverage:
- How might limited analyst coverage affect stock sentiment?
- What does absence of recommendations suggest about market attention?
"""
        
        # Prepare recommendations summary
        recent_recs = recommendations_df.head(10)  # Most recent 10 recommendations
        
        recs_summary = ""
        for _, rec in recent_recs.iterrows():
            period = rec.get('period', 'Unknown')
            strong_buy = rec.get('strong_buy', 0)
            buy = rec.get('buy', 0) 
            hold = rec.get('hold', 0)
            sell = rec.get('sell', 0)
            strong_sell = rec.get('strong_sell', 0)
            mean_rating = rec.get('mean_rating', 'N/A')
            
            recs_summary += f"""
Period: {period}
- Strong Buy: {strong_buy}, Buy: {buy}, Hold: {hold}, Sell: {sell}, Strong Sell: {strong_sell}
- Mean Rating: {mean_rating}
"""
        
        # Calculate basic stats
        total_recs = len(recommendations_df)
        recent_period = recommendations_df.iloc[0]['period'] if len(recommendations_df) > 0 else 'Unknown'
        oldest_period = recommendations_df.iloc[-1]['period'] if len(recommendations_df) > 0 else 'Unknown'
        
        # Get similar training examples for few-shot learning
        context_text = f"Analyst recommendations: {recs_summary}"
        similar_cases = self.find_similar_training_embeddings(context_text, limit=3)
        
        # Create few-shot examples section
        examples_section = ""
        if similar_cases:
            examples_section = "\n\n**LEARNING FROM SIMILAR CASES:**\n"
            for i, case in enumerate(similar_cases, 1):
                outcome = case.get('outperformance_10d', 0.0)
                text = case.get('text_content', case.get('chunk_text', ''))[:200]
                examples_section += f"""
Example {i}:
Analyst Data: "{text}..."
Outcome: Stock {'outperformed' if outcome > 0 else 'underperformed' if outcome < 0 else 'neutral performance'} ({outcome:+.1f}% vs sector in 10 days)
"""
            examples_section += "\nNow analyze the current analyst data and extract features, learning from these patterns:\n"
        
        return f"""
Analyze analyst recommendations for setup {setup_id}:{examples_section}

**CURRENT ANALYST RECOMMENDATIONS DATA:**
Total recommendations analyzed: {total_recs}
Time range: {oldest_period} to {recent_period}

Recent Recommendations Summary:
{recs_summary}

Please extract the following features based on this analyst data:

1. CONSENSUS_RATING (1.0-5.0): Overall analyst sentiment where 1=very bullish, 3=neutral, 5=very bearish
2. RECENT_UPGRADES (integer): Count of positive rating changes in recent periods
3. RECENT_DOWNGRADES (integer): Count of negative rating changes in recent periods  
4. ANALYST_CONVICTION_SCORE (0.0-1.0): How confident/decisive analysts seem about their recommendations
5. RECOMMENDATION_MOMENTUM (improving/stable/deteriorating): Trend in analyst sentiment over time
6. SYNTHETIC_ANALYST_SUMMARY (≤240 chars): Brief summary of analyst sentiment and key themes
7. COT_EXPLANATION_ANALYST: Your reasoning for the above assessments

Focus on:
- Changes in recommendation patterns over time
- Strength of analyst conviction vs uncertainty
- Any emerging themes or consensus shifts
- How recent the recommendations are

Return JSON format:
{{
    "consensus_rating": 2.5,
    "recent_upgrades": 0,
    "recent_downgrades": 0,
    "analyst_conviction_score": 0.5,
    "recommendation_momentum": "stable",
    "synthetic_analyst_summary": "Brief summary here",
    "cot_explanation_analyst": "Your reasoning here"
}}
"""
    
    def extract_llm_features(self, setup_id: str, recommendations_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract LLM-based features from analyst recommendations"""
        
        prompt = self.create_llm_prompt(setup_id, recommendations_df)
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert financial analyst specializing in analyst recommendation analysis. Extract features accurately and return valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response with better handling
            try:
                # Clean up the content - remove markdown code blocks
                cleaned_content = content.strip()
                if cleaned_content.startswith('```json'):
                    cleaned_content = cleaned_content.split('```json')[1].split('```')[0].strip()
                elif cleaned_content.startswith('```'):
                    cleaned_content = cleaned_content.split('```')[1].split('```')[0].strip()
                
                features = json.loads(cleaned_content)
                
                # Validate and clean features
                features['consensus_rating'] = max(1.0, min(5.0, float(features.get('consensus_rating', 3.0))))
                features['recent_upgrades'] = max(0, int(features.get('recent_upgrades', 0)))
                features['recent_downgrades'] = max(0, int(features.get('recent_downgrades', 0)))
                features['analyst_conviction_score'] = max(0.0, min(1.0, float(features.get('analyst_conviction_score', 0.5))))
                
                momentum = features.get('recommendation_momentum', 'stable').lower()
                if momentum not in ['improving', 'stable', 'deteriorating']:
                    momentum = 'stable'
                features['recommendation_momentum'] = momentum
                
                # Truncate text fields
                features['synthetic_analyst_summary'] = str(features.get('synthetic_analyst_summary', ''))[:240]
                features['cot_explanation_analyst'] = str(features.get('cot_explanation_analyst', ''))[:1000]
                
                return features
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for {setup_id}: {e}")
                logger.error(f"Raw content: {content}")
                return self._get_default_llm_features()
                
        except Exception as e:
            logger.error(f"LLM extraction error for {setup_id}: {e}")
            return self._get_default_llm_features()
    
    def _get_default_llm_features(self) -> Dict[str, Any]:
        """Get default LLM features when extraction fails"""
        return {
            'consensus_rating': 3.0,
            'recent_upgrades': 0,
            'recent_downgrades': 0,
            'analyst_conviction_score': 0.5,
            'recommendation_momentum': 'stable',
            'synthetic_analyst_summary': 'Analysis unavailable',
            'cot_explanation_analyst': 'LLM extraction failed'
        }
    
    def process_setup(self, setup_id: str, mode: str = None) -> Dict[str, Any]:
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
        logger.info(f"Processing analyst recommendations for setup: {setup_id} in {current_mode} mode")
        
        try:
            # Step 1: Retrieve analyst recommendations data
            recommendations_df = self.get_analyst_recommendations_for_setup(setup_id)
            
            if len(recommendations_df) == 0:
                logger.warning(f"No analyst recommendations found for setup {setup_id}")
                features = self._get_default_llm_features()
                features['setup_id'] = setup_id
                features['extraction_timestamp'] = datetime.now().isoformat()
                features['llm_model'] = self.llm_model
                self.store_features(setup_id, features)
                return features
            
            # Step 2: Extract structured metrics
            structured_metrics = self.extract_structured_metrics(setup_id, recommendations_df)
            
            # Step 3: Extract LLM features
            llm_features = self.extract_llm_features(setup_id, recommendations_df)
            
            # Step 4: Combine all features
            features = {**structured_metrics, **llm_features}
            features['setup_id'] = setup_id
            features['extraction_timestamp'] = datetime.now().isoformat()
            features['llm_model'] = self.llm_model
            
            # Step 5: Store features
            self.store_features(setup_id, features)
            
            # Step 6: In prediction mode, generate similarity-based predictions
            if current_mode == 'prediction' and hasattr(self, 'training_table') and self.training_table is not None:
                # Create content for similarity search
                analyst_summary = f"Analyst recommendations: {features.get('buy_recommendations', 0)} Buy, " + \
                                 f"{features.get('hold_recommendations', 0)} Hold, " + \
                                 f"{features.get('sell_recommendations', 0)} Sell | " + \
                                 f"Price target: {features.get('avg_price_target', 0)} | " + \
                                 f"Consensus: {features.get('consensus_rating', 3.0)}"
                
                if features.get('synthetic_analyst_summary'):
                    analyst_summary += f" | {features['synthetic_analyst_summary']}"
                
                # Generate prediction
                if hasattr(self, 'predict_via_similarity'):
                    prediction = self.predict_via_similarity(analyst_summary)
                    if prediction:
                        prediction['setup_id'] = setup_id
                        prediction['domain'] = 'analyst'
                        prediction['prediction_timestamp'] = datetime.now().isoformat()
                        logger.info(f"Generated similarity prediction for {setup_id}: {prediction.get('predicted_outperformance_10d', 'N/A')}")
                        return features, prediction
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing analyst recommendations for setup {setup_id}: {e}")
            return None
    
    def store_features(self, setup_id: str, features: Dict[str, Any]) -> bool:
        """Store extracted features in DuckDB"""
        
        try:
            conn = duckdb.connect(self.db_path)
            
            # Prepare values for insertion
            values = (
                setup_id,
                features['recommendation_count'],
                features['buy_recommendations'],
                features['sell_recommendations'],
                features['hold_recommendations'],
                features['avg_price_target'],
                features['price_target_vs_current'],
                features['price_target_spread'],
                features['coverage_breadth'],
                features['consensus_rating'],
                features['recent_upgrades'],
                features['recent_downgrades'],
                features['analyst_conviction_score'],
                features['recommendation_momentum'],
                features['synthetic_analyst_summary'],
                features['cot_explanation_analyst'],
                features['predicted_outperformance_10d'],
                features['extraction_timestamp'],
                features['llm_model']
            )
            
            # Insert or replace features
            insert_query = '''
                INSERT OR REPLACE INTO analyst_recommendations_features (
                    setup_id, recommendation_count, buy_recommendations, sell_recommendations,
                    hold_recommendations, avg_price_target, price_target_vs_current, 
                    price_target_spread, coverage_breadth, consensus_rating, recent_upgrades,
                    recent_downgrades, analyst_conviction_score, recommendation_momentum,
                    synthetic_analyst_summary, cot_explanation_analyst, predicted_outperformance_10d, extraction_timestamp, llm_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            conn.execute(insert_query, values)
            conn.commit()
            conn.close()
            
            logger.info(f"Stored analyst features for setup: {setup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing features for {setup_id}: {e}")
            return False
    
    def process_multiple_setups(self, setup_ids: List[str]) -> Dict[str, bool]:
        """Process multiple setups for analyst recommendations features"""
        
        results = {}
        
        for i, setup_id in enumerate(setup_ids):
            try:
                logger.info(f"Processing setup {i+1}/{len(setup_ids)}: {setup_id}")
                
                # Process setup
                features = self.process_setup(setup_id)
                
                # Store features
                success = self.store_features(setup_id, features)
                results[setup_id] = success
                
                if success:
                    logger.info(f"✅ Successfully processed: {setup_id}")
                else:
                    logger.error(f"❌ Failed to store: {setup_id}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {setup_id}: {e}")
                results[setup_id] = False
        
        # Summary
        successful = sum(results.values())
        total = len(setup_ids)
        logger.info(f"Analyst recommendations processing complete: {successful}/{total} successful")
        
        return results

    def get_stored_features(self, setup_id: str) -> Optional[Dict]:
        """Retrieve stored analyst features for a setup from DuckDB"""
        try:
            conn = duckdb.connect(self.db_path)
            
            result = conn.execute(
                "SELECT * FROM analyst_recommendations_features WHERE setup_id = ?",
                (setup_id,)
            ).fetchone()
            
            if result:
                columns = [desc[0] for desc in conn.description]
                feature_dict = dict(zip(columns, result))
                return feature_dict
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving stored analyst features from DuckDB: {e}")
            return None
        finally:
            conn.close()

    def batch_process_setups(self, setup_ids: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Process multiple setups in batch, skipping already-processed setups"""
        results = {}
        skipped_count = 0
        
        for i, setup_id in enumerate(setup_ids, 1):
            # Check if features already exist
            existing_features = self.get_stored_features(setup_id)
            if existing_features:
                logger.info(f"⏭️ Skipping setup {i}/{len(setup_ids)}: {setup_id} (already processed)")
                results[setup_id] = existing_features  # Return existing features
                skipped_count += 1
                continue
                
            logger.info(f"Processing setup {i}/{len(setup_ids)}: {setup_id}")
            
            try:
                # Process the setup
                features = self.process_setup(setup_id)
                
                if features:
                    # Store features in database
                    success = self.store_features(setup_id, features)
                    if success:
                        results[setup_id] = features
                        logger.info(f"✅ Successfully processed: {setup_id}")
                    else:
                        results[setup_id] = None
                        logger.error(f"❌ Failed to store: {setup_id}")
                else:
                    results[setup_id] = None
                    logger.error(f"❌ Processing failed: {setup_id}")
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing setup {setup_id}: {e}")
                results[setup_id] = None
        
        if skipped_count > 0:
            logger.info(f"✅ Skipped {skipped_count} already-processed setups")
        
        return results

    def find_similar_training_embeddings(self, query: Union[np.ndarray, str], limit: int = 10) -> List[Dict]:
        """
        Find similar training embeddings with enhanced two-stage retrieval and re-ranking
        
        Args:
            query: Either a numpy array embedding or text content to embed
            limit: Maximum number of similar cases to return
            
        Returns:
            List of re-ranked similar cases with their metadata and labels
        """
        if not hasattr(self, 'training_table') or self.training_table is None:
            logger.warning("Training embeddings table not available")
            return []
        
        try:
            # Handle text input by creating embedding
            if isinstance(query, str):
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder="models/sentence_transformers", local_files_only=True)
                query = model.encode(query)
            
            # Ensure numpy array
            if not isinstance(query, np.ndarray):
                query = np.array(query)
                
            # Stage 1: Retrieve more candidates for re-ranking (3x the final limit)
            candidate_limit = min(limit * 3, 30)
            results = self.training_table.search(query).limit(candidate_limit).to_pandas()
            
            if len(results) == 0:
                return []
            
            # Stage 2: Re-rank candidates based on similarity only (EQUAL WEIGHTING)
            scored_results = []
            
            for _, row in results.iterrows():
                # Use only similarity score - no bias factors
                similarity = 1 - row.get('_distance', 1.0)
                score = similarity  # Equal weighting = just use similarity
                
                scored_results.append((score, row.to_dict()))
            
            # Sort by score and return top results
            scored_results.sort(key=lambda x: x[0], reverse=True)
            final_results = [result[1] for result in scored_results[:limit]]
            
            logger.info(f"Re-ranked {len(results)} candidates to return top {len(final_results)} similar cases (equal weighting)")
            
            return final_results
            
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
        
        # Use label converter
        label_converter = LabelConverter()
            
        # Extract labels from similar cases
        similar_labels = [case.get('outperformance_10d', 0.0) for case in similar_cases]
        
        # Compute weighted average prediction
        similarity_scores = [1 - case.get('_distance', 1.0) for case in similar_cases]
        sum_scores = sum(similarity_scores) if similarity_scores else 1.0
        if sum_scores == 0:
            sum_scores = 1.0
            
        weighted_prediction = sum(label * score for label, score in zip(similar_labels, similarity_scores)) / sum_scores
        
        # Get class distribution using the label converter
        class_distribution = get_class_distribution(similar_labels)
        
        # Compute confidence metrics
        prediction = {
            'predicted_outperformance': float(weighted_prediction),
            'confidence': float(1.0 / (1.0 + np.std(similar_labels))),
            'positive_ratio': class_distribution['positive'],
            'negative_ratio': class_distribution['negative'],
            'neutral_ratio': class_distribution['neutral'],
            'similar_cases_count': len(similar_cases)
        }
        
        return prediction

    def predict_with_llm(self, setup_id: str, current_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make LLM-based prediction using current analyst features and similar historical cases
        
        Args:
            setup_id: Current setup ID
            current_features: Extracted analyst features for current setup
            
        Returns:
            Dictionary with LLM prediction results
        """
        try:
            # Create context from current analyst features
            context = f"""Setup {setup_id} Analyst Analysis:
- Consensus Rating: {current_features.get('consensus_rating', 3.0)}
- Recent Upgrades: {current_features.get('recent_upgrades', 0)}
- Recent Downgrades: {current_features.get('recent_downgrades', 0)}
- Analyst Conviction: {current_features.get('analyst_conviction_score', 0.5)}
- Momentum: {current_features.get('recommendation_momentum', 'stable')}
- Summary: {current_features.get('synthetic_analyst_summary', 'N/A')}"""
            
            # Get similar training examples for prediction context
            similar_cases = self.find_similar_training_embeddings(context, limit=5)
            
            # Create prediction prompt with few-shot examples
            examples_section = ""
            if similar_cases:
                examples_section = "\n\n**HISTORICAL EXAMPLES TO LEARN FROM:**\n"
                for i, case in enumerate(similar_cases[:3], 1):
                    outcome = case.get('outperformance_10d', 0.0)
                    case_text = case.get('text_content', case.get('chunk_text', ''))[:150]
                    examples_section += f"""
Example {i}:
Analyst Context: "{case_text}..."
Actual Outcome: {outcome:+.1f}% outperformance vs sector (10 days)
Result: {'POSITIVE' if outcome > 0 else 'NEGATIVE' if outcome < 0 else 'NEUTRAL'}
"""
                examples_section += "\nBased on these historical analyst patterns, analyze the current setup:\n"
            
            prediction_prompt = f"""You are an expert analyst coverage specialist making investment predictions.

{examples_section}

**CURRENT SETUP TO PREDICT:**
{context}

Based on the analyst analysis and historical patterns above, predict the 10-day stock outperformance vs sector.

Return your prediction as JSON:
{{
    "predicted_outperformance_10d": <float between -15.0 and +15.0>,
    "confidence_score": <float between 0.0 and 1.0>,
    "prediction_class": "<POSITIVE|NEGATIVE|NEUTRAL>",
    "reasoning": "<brief explanation of your prediction logic>"
}}

Focus on:
- Analyst consensus and conviction levels
- Recent rating changes momentum
- Upgrade/downgrade patterns
- Historical analyst accuracy patterns"""

            # Make LLM prediction call
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert analyst coverage specialist specializing in stock performance prediction. Analyze analyst data and make accurate predictions based on historical patterns."},
                    {"role": "user", "content": prediction_prompt}
                ],
                temperature=0.1,
                max_tokens=400
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if result_text.startswith('```json'):
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif result_text.startswith('```'):
                result_text = result_text.split('```')[1].split('```')[0].strip()
            
            import json
            prediction_result = json.loads(result_text)
            
            # Add metadata
            prediction_result['setup_id'] = setup_id
            prediction_result['prediction_method'] = 'llm_few_shot'
            prediction_result['similar_cases_used'] = len(similar_cases)
            
            logger.info(f"Analyst LLM prediction for {setup_id}: {prediction_result.get('predicted_outperformance_10d', 0.0):.2f}% (confidence: {prediction_result.get('confidence_score', 0.0):.2f})")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error in analyst LLM prediction for {setup_id}: {e}")
            return {
                'setup_id': setup_id,
                'predicted_outperformance_10d': 0.0,
                'confidence_score': 0.1,
                'prediction_class': 'NEUTRAL',
                'reasoning': 'Error in prediction - defaulting to neutral',
                'prediction_method': 'llm_few_shot_error',
                'similar_cases_used': 0
            }

    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'setup_validator'):
                self.setup_validator.close()
            logger.info("Analyst Recommendations agent cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    # Example usage
    agent = EnhancedAnalystRecommendationsAgentDuckDB()
    
    # Test with a few setups
    test_setups = ["BARC_2024-10-25", "BOOM_2024-11-15", "BNC_2024-12-20"]
    results = agent.process_multiple_setups(test_setups)
    
    print("Processing results:", results) 