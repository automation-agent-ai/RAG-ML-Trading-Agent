#!/usr/bin/env python3
"""
Complete ML Pipeline for Stock Performance Prediction

This script orchestrates the complete ML pipeline:
1. Create embeddings (with/without labels based on mode)
2. Extract features from raw data (with similarity enhancement in prediction mode)
3. Merge features into ML feature tables
4. Train/predict ML models
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompletePipeline:
    """Complete pipeline for feature extraction and ML feature table creation"""
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        lancedb_dir: str = "data/lancedb_store",
        prediction_db_path: str = "data/prediction_features.duckdb"
    ):
        self.db_path = db_path
        self.lancedb_dir = lancedb_dir
        self.prediction_db_path = prediction_db_path
        
        # Initialize feature merger
        from core.ml_feature_merger import MLFeatureMerger
        self.feature_merger = MLFeatureMerger(db_path=db_path)
        
        # Initialize agents and embedders lazily
        self.news_agent = None
        self.fundamentals_agent = None
        self.analyst_agent = None
        self.userposts_agent = None
        
    def _init_agents(self):
        """Initialize agents if not already initialized"""
        if self.news_agent is None:
            from agents.news.enhanced_news_agent_duckdb import EnhancedNewsAgentDuckDB
            self.news_agent = EnhancedNewsAgentDuckDB(
                db_path=str(self.db_path),
                lancedb_dir=str(self.lancedb_dir)
            )
            
        if self.fundamentals_agent is None:
            from agents.fundamentals.enhanced_fundamentals_agent_duckdb import EnhancedFundamentalsAgentDuckDB
            self.fundamentals_agent = EnhancedFundamentalsAgentDuckDB(
                db_path=str(self.db_path),
                lancedb_dir=str(self.lancedb_dir)
            )
            
        if self.analyst_agent is None:
            from agents.analyst_recommendations.enhanced_analyst_recommendations_agent_duckdb import EnhancedAnalystRecommendationsAgentDuckDB
            self.analyst_agent = EnhancedAnalystRecommendationsAgentDuckDB(
                db_path=str(self.db_path),
                lancedb_dir=str(self.lancedb_dir)
            )
            
        if self.userposts_agent is None:
            from agents.userposts.enhanced_userposts_agent_complete import EnhancedUserPostsAgentComplete
            self.userposts_agent = EnhancedUserPostsAgentComplete(
                db_path=str(self.db_path),
                lancedb_dir=str(self.lancedb_dir)
            )
    
    def create_embeddings(self, setup_ids: List[str], mode: str = 'training') -> Dict[str, bool]:
        """Create embeddings with proper training/prediction mode separation"""
        logger.info(f"🎰 CREATING EMBEDDINGS ({mode.upper()} MODE)")
        logger.info("=" * 60)
        
        results = {}
        include_labels = (mode == 'training')
        
        try:
            # 1. News Embeddings
            logger.info("📰 Creating news embeddings...")
            from embeddings.embed_news_duckdb import NewsEmbeddingPipelineDuckDB
            news_embedder = NewsEmbeddingPipelineDuckDB(
                db_path=self.db_path,
                lancedb_dir=self.lancedb_dir,
                include_labels=include_labels,
                mode=mode
            )
            news_result = news_embedder.process_setups(setup_ids)
            results['news_embeddings'] = news_result
            logger.info(f"✅ News embeddings: {'Success' if news_result else 'Failed'}")
            
            # 2. Fundamentals Embeddings  
            logger.info("📊 Creating fundamentals embeddings...")
            from embeddings.embed_fundamentals_duckdb import FundamentalsEmbedderDuckDB
            fundamentals_embedder = FundamentalsEmbedderDuckDB(
                db_path=self.db_path,
                lancedb_dir=self.lancedb_dir,
                include_labels=include_labels,
                mode=mode
            )
            fundamentals_result = fundamentals_embedder.embed_fundamentals(setup_ids)
            results['fundamentals_embeddings'] = fundamentals_result
            logger.info(f"✅ Fundamentals embeddings: {'Success' if fundamentals_result else 'Failed'}")
            
            # 3. Analyst Recommendations Embeddings
            logger.info("📈 Creating analyst recommendations embeddings...")
            from embeddings.embed_analyst_recommendations_duckdb import AnalystRecommendationsEmbedderDuckDB
            analyst_embedder = AnalystRecommendationsEmbedderDuckDB(
                db_path=self.db_path,
                lancedb_dir=self.lancedb_dir,
                include_labels=include_labels,
                mode=mode
            )
            analyst_result = analyst_embedder.process_setups(setup_ids)
            results['analyst_embeddings'] = analyst_result
            logger.info(f"✅ Analyst embeddings: {'Success' if analyst_result else 'Failed'}")
            
            # 4. UserPosts Embeddings
            logger.info("💬 Creating userposts embeddings...")
            from embeddings.embed_userposts_duckdb import UserPostsEmbedderDuckDB
            userposts_embedder = UserPostsEmbedderDuckDB(
                db_path=self.db_path,
                lancedb_dir=self.lancedb_dir,
                include_labels=include_labels,
                mode=mode
            )
            userposts_result = userposts_embedder.process_setups(setup_ids)
            results['userposts_embeddings'] = userposts_result
            logger.info(f"✅ UserPosts embeddings: {'Success' if userposts_result else 'Failed'}")
            
        except Exception as e:
            logger.error(f"❌ Embedding creation failed: {e}")
            results = {domain: False for domain in ['news_embeddings', 'fundamentals_embeddings', 'analyst_embeddings', 'userposts_embeddings']}
        
        return results
        
    def extract_features(self, setup_ids: List[str], mode: str = 'training') -> Dict[str, bool]:
        """Extract features for all domains with similarity enhancement in prediction mode"""
        logger.info(f"🔧 RUNNING FEATURE EXTRACTION ({mode.upper()} MODE)")
        logger.info("=" * 60)
        
        # Initialize agents if not already done
        self._init_agents()
        
        results = {}
        similarity_predictions = {}
        
        try:
            # 1. Fundamentals Features
            logger.info("📊 Extracting fundamentals features...")
            
            # Override to target setups only
            self.fundamentals_agent.setup_validator.confirmed_setup_ids = set(setup_ids)
            
            fundamentals_results = {}
            for setup_id in setup_ids:
                try:
                    # Process setup with mode parameter
                    features = self.fundamentals_agent.process_setup(setup_id, mode=mode)
                    fundamentals_results[setup_id] = features is not None
                    
                    # In prediction mode, also get direct similarity predictions
                    if mode == 'prediction' and features is not None:
                        # Create embedding for similarity search
                        embedding = self._create_temporary_embedding(
                            setup_id, 
                            features.financial_summary if hasattr(features, 'financial_summary') else "",
                            domain="fundamentals"
                        )
                        
                        if embedding is not None:
                            # Get similarity prediction
                            sim_prediction = self.fundamentals_agent.predict_via_similarity(embedding)
                            if sim_prediction:
                                if setup_id not in similarity_predictions:
                                    similarity_predictions[setup_id] = {}
                                similarity_predictions[setup_id]['fundamentals'] = sim_prediction
                except Exception as e:
                    logger.error(f"Error processing fundamentals for setup {setup_id}: {e}")
                    fundamentals_results[setup_id] = False
            
            successful = sum(1 for r in fundamentals_results.values() if r)
            results['fundamentals_features'] = successful > 0
            logger.info(f"✅ Fundamentals features: {successful}/{len(setup_ids)} successful")
            
        except Exception as e:
            logger.error(f"❌ Fundamentals features failed: {e}")
            results['fundamentals_features'] = False
            
        try:
            # 2. News Features
            logger.info("📰 Extracting news features...")
            
            # Override to target setups only
            self.news_agent.setup_validator.confirmed_setup_ids = set(setup_ids)
            
            news_results = {}
            for setup_id in setup_ids:
                try:
                    # Process setup with mode parameter
                    features = self.news_agent.process_setup(setup_id, mode=mode)
                    news_results[setup_id] = features is not None
                    
                    # In prediction mode, also get direct similarity predictions
                    if mode == 'prediction' and features is not None:
                        # Get all news content for this setup
                        news_df = self.news_agent.retrieve_news_by_setup_id(setup_id)
                        if len(news_df) > 0:
                            # Create embedding for similarity search
                            combined_content = "\n\n".join(news_df['content'].tolist()[:5])  # Limit to first 5 news items
                            embedding = self._create_temporary_embedding(setup_id, combined_content, domain="news")
                            
                            if embedding is not None:
                                # Get similarity prediction
                                sim_prediction = self.news_agent.predict_via_similarity(embedding)
                                if sim_prediction:
                                    if setup_id not in similarity_predictions:
                                        similarity_predictions[setup_id] = {}
                                    similarity_predictions[setup_id]['news'] = sim_prediction
                except Exception as e:
                    logger.error(f"Error processing news for setup {setup_id}: {e}")
                    news_results[setup_id] = False
            
            successful = sum(1 for r in news_results.values() if r)
            results['news_features'] = successful > 0
            logger.info(f"✅ News features: {successful}/{len(setup_ids)} successful")
            
        except Exception as e:
            logger.error(f"❌ News features failed: {e}")
            results['news_features'] = False
            
        # Add other domains (analyst recommendations, userposts) here
        # ...
        
        # Store similarity predictions if in prediction mode
        if mode == 'prediction' and similarity_predictions:
            self._store_similarity_predictions(similarity_predictions)
            
        return results
    
    def _create_temporary_embedding(self, setup_id: str, content: str, domain: str) -> Optional[np.ndarray]:
        """Create a temporary embedding for similarity search"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model.encode(content)
        except Exception as e:
            logger.error(f"Error creating temporary embedding for {domain} setup {setup_id}: {e}")
            return None
    
    def _store_similarity_predictions(self, predictions: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """Store similarity-based predictions in DuckDB"""
        try:
            import duckdb
            
            # Prepare records for insertion
            records = []
            for setup_id, domains in predictions.items():
                record = {
                    'setup_id': setup_id,
                    'prediction_timestamp': datetime.now().isoformat()
                }
                
                # Add domain-specific predictions
                for domain, prediction in domains.items():
                    for key, value in prediction.items():
                        record[f"{domain}_{key}"] = value
                
                records.append(record)
            
            if not records:
                return
                
            # Create DataFrame
            df = pd.DataFrame(records)
            
            # Connect to DuckDB
            conn = duckdb.connect(self.db_path)
            
            # Create table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS similarity_predictions (
                    setup_id VARCHAR,
                    prediction_timestamp TIMESTAMP,
                    
                    -- News domain predictions
                    news_predicted_outperformance DOUBLE,
                    news_confidence DOUBLE,
                    news_positive_ratio DOUBLE,
                    news_negative_ratio DOUBLE,
                    news_neutral_ratio DOUBLE,
                    
                    -- Fundamentals domain predictions
                    fundamentals_predicted_outperformance DOUBLE,
                    fundamentals_confidence DOUBLE,
                    fundamentals_positive_ratio DOUBLE,
                    fundamentals_negative_ratio DOUBLE,
                    fundamentals_neutral_ratio DOUBLE,
                    
                    -- Analyst domain predictions
                    analyst_predicted_outperformance DOUBLE,
                    analyst_confidence DOUBLE,
                    analyst_positive_ratio DOUBLE,
                    analyst_negative_ratio DOUBLE,
                    analyst_neutral_ratio DOUBLE,
                    
                    -- UserPosts domain predictions
                    userposts_predicted_outperformance DOUBLE,
                    userposts_confidence DOUBLE,
                    userposts_positive_ratio DOUBLE,
                    userposts_negative_ratio DOUBLE,
                    userposts_neutral_ratio DOUBLE
                )
            """)
            
            # Insert records
            conn.register('predictions_df', df)
            conn.execute("INSERT INTO similarity_predictions SELECT * FROM predictions_df")
            
            # Close connection
            conn.close()
            
            logger.info(f"Stored {len(records)} similarity predictions")
            
        except Exception as e:
            logger.error(f"Error storing similarity predictions: {e}")
    
    def create_ml_features(self, setup_ids: List[str], mode: str = 'training') -> Dict[str, Dict[str, int]]:
        """Create ML feature tables by merging domain features"""
        logger.info(f"\n🔄 Creating {mode} ML feature tables...")
        
        # Use the MLFeatureMerger to create comprehensive ML feature tables
        results = self.feature_merger.merge_all_features(setup_ids, mode)
        
        return results
    
    def run_complete_pipeline(self, setup_ids: List[str], mode: str = 'training') -> Dict[str, Any]:
        """Run complete pipeline: embedding creation, feature extraction, and ML feature table creation"""
        logger.info("🚀 STARTING COMPLETE PIPELINE")
        logger.info("=" * 70)
        
        start_time = datetime.now()
        
        # Step 1: Create embeddings (with proper training/prediction mode handling)
        embedding_results = self.create_embeddings(setup_ids, mode)
        
        # Step 2: Extract features (with similarity enhancement in prediction mode)
        extraction_results = self.extract_features(setup_ids, mode)
        
        # Step 3: Create ML feature tables
        ml_features_results = self.create_ml_features(setup_ids, mode)
        
        # Generate summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return {
            'status': 'success',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'setups_processed': len(setup_ids),
            'embedding_results': embedding_results,
            'extraction_results': extraction_results,
            'ml_features_results': ml_features_results
        }

def main():
    """Run the complete pipeline"""
    import argparse
    import duckdb
    
    parser = argparse.ArgumentParser(description='Run complete ML pipeline')
    parser.add_argument('--mode', choices=['training', 'prediction'], default='training',
                       help='Pipeline mode: training or prediction')
    parser.add_argument('--setup-ids', nargs='+', help='List of setup_ids to process (required for prediction mode)')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--prediction-db-path', default='data/prediction_features.duckdb',
                       help='Path to store prediction features')
    parser.add_argument('--lancedb-dir', default='storage/lancedb_store',
                       help='Path to LanceDB directory')
    parser.add_argument('--step', choices=['all', 'embeddings', 'features', 'ml_tables'], default='all',
                       help='Pipeline step to run (default: all)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'prediction' and not args.setup_ids:
        parser.error("--setup-ids is required for prediction mode")
    
    # Initialize pipeline
    pipeline = CompletePipeline(
        db_path=args.db_path,
        prediction_db_path=args.prediction_db_path,
        lancedb_dir=args.lancedb_dir
    )
    
    if args.mode == 'training':
        # For training mode, get all setups with complete data
        conn = duckdb.connect(args.db_path)
        
        # Find setups that have all required features
        complete_setups_query = """
        WITH required_features AS (
            SELECT setup_id 
            FROM fundamentals_features
            INTERSECT
            SELECT setup_id 
            FROM news_features
            INTERSECT
            SELECT setup_id 
            FROM userposts_features
            INTERSECT
            SELECT setup_id 
            FROM analyst_recommendations_features
            INTERSECT
            SELECT setup_id 
            FROM labels
            WHERE outperformance_10d IS NOT NULL
        )
        SELECT setup_id 
        FROM required_features
        ORDER BY setup_id
        """
        setup_ids = [row[0] for row in conn.execute(complete_setups_query).fetchall()]
        conn.close()
        
        logger.info(f"Found {len(setup_ids)} setups with complete data for training")
    else:
        # For prediction mode, use provided setup_ids
        setup_ids = args.setup_ids
    
    # Run specific pipeline step or all steps
    start_time = datetime.now()
    
    if args.step == 'all' or args.step == 'embeddings':
        pipeline.create_embeddings(setup_ids, mode=args.mode)
        
    if args.step == 'all' or args.step == 'features':
        pipeline.extract_features(setup_ids, mode=args.mode)
        
    if args.step == 'all' or args.step == 'ml_tables':
        pipeline.create_ml_features(setup_ids, mode=args.mode)
    
    # Print summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n🎉 Pipeline Complete!")
    logger.info(f"Step: {args.step}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Duration: {duration:.1f}s")
    logger.info(f"Setups processed: {len(setup_ids)}")

if __name__ == "__main__":
    main() 