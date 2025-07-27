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
        lancedb_dir: str = "lancedb_store",
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
        
    def _init_agents(self, mode: str = "training"):
        """Initialize all agents"""
        if not hasattr(self, 'news_agent') or self.news_agent is None:
            from agents.news.enhanced_news_agent_duckdb import EnhancedNewsAgentDuckDB
            self.news_agent = EnhancedNewsAgentDuckDB(
                db_path=self.db_path,
                lancedb_dir=self.lancedb_dir,
                mode=mode
            )
            
        if not hasattr(self, 'fundamentals_agent') or self.fundamentals_agent is None:
            from agents.fundamentals.enhanced_fundamentals_agent_duckdb import EnhancedFundamentalsAgentDuckDB
            self.fundamentals_agent = EnhancedFundamentalsAgentDuckDB(
                db_path=self.db_path,
                lancedb_dir=self.lancedb_dir,
                mode=mode
            )
            
        if not hasattr(self, 'analyst_agent') or self.analyst_agent is None:
            from agents.analyst_recommendations.enhanced_analyst_recommendations_agent_duckdb import EnhancedAnalystRecommendationsAgentDuckDB
            self.analyst_agent = EnhancedAnalystRecommendationsAgentDuckDB(
                db_path=self.db_path,
                lancedb_dir=self.lancedb_dir,
                mode=mode
            )
            
        if not hasattr(self, 'userposts_agent') or self.userposts_agent is None:
            from agents.userposts.enhanced_userposts_agent_complete import EnhancedUserPostsAgentComplete
            self.userposts_agent = EnhancedUserPostsAgentComplete(
                db_path=self.db_path,
                lancedb_dir=self.lancedb_dir,
                mode=mode
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
        
    def extract_features(self, setup_ids: List[str], mode: str = "training"):
        """
        Extract features for setup_ids
        
        Args:
            setup_ids: List of setup IDs to process
            mode: Either 'training' or 'prediction'
        
        Returns:
            Dictionary with success status for each agent
        """
        logger.info(f"🔧 RUNNING FEATURE EXTRACTION ({mode.upper()} MODE)")
        logger.info("=" * 60)
        
        # Initialize agents if not already done
        self._init_agents(mode)
        
        results = {}
        similarity_predictions = {}
        
        # Process each setup ID
        for setup_id in setup_ids:
            similarity_predictions[setup_id] = {}
            
            # Process with news agent
            try:
                logger.info(f"Calling news_agent.process_setup for {setup_id}")
                news_result = self.news_agent.process_setup(setup_id, mode)
                logger.info(f"News result type: {type(news_result)}")
                
                if isinstance(news_result, tuple) and len(news_result) == 2:
                    # Unpack features and prediction
                    features, prediction = news_result
                    logger.info(f"News prediction: {prediction}")
                    similarity_predictions[setup_id]['news'] = prediction
                else:
                    logger.info("News agent did not return a prediction")
                    
                results['news'] = True
            except Exception as e:
                logger.error(f"Error processing news for {setup_id}: {e}")
                results['news'] = False
            
            # Process with fundamentals agent
            try:
                logger.info(f"Calling fundamentals_agent.process_setup for {setup_id}")
                fundamentals_result = self.fundamentals_agent.process_setup(setup_id, mode)
                logger.info(f"Fundamentals result type: {type(fundamentals_result)}")
                
                if isinstance(fundamentals_result, tuple) and len(fundamentals_result) == 2:
                    # Unpack features and prediction
                    features, prediction = fundamentals_result
                    logger.info(f"Fundamentals prediction: {prediction}")
                    similarity_predictions[setup_id]['fundamentals'] = prediction
                else:
                    logger.info("Fundamentals agent did not return a prediction")
                    
                results['fundamentals'] = True
            except Exception as e:
                logger.error(f"Error processing fundamentals for {setup_id}: {e}")
                results['fundamentals'] = False
            
            # Process with analyst agent
            try:
                logger.info(f"Calling analyst_agent.process_setup for {setup_id}")
                analyst_result = self.analyst_agent.process_setup(setup_id, mode)
                logger.info(f"Analyst result type: {type(analyst_result)}")
                
                if isinstance(analyst_result, tuple) and len(analyst_result) == 2:
                    # Unpack features and prediction
                    features, prediction = analyst_result
                    logger.info(f"Analyst prediction: {prediction}")
                    similarity_predictions[setup_id]['analyst'] = prediction
                else:
                    logger.info("Analyst agent did not return a prediction")
                    
                results['analyst'] = True
            except Exception as e:
                logger.error(f"Error processing analyst recommendations for {setup_id}: {e}")
                results['analyst'] = False
            
            # Process with userposts agent
            try:
                logger.info(f"Calling userposts_agent.process_setup for {setup_id}")
                userposts_result = self.userposts_agent.process_setup(setup_id, mode)
                logger.info(f"UserPosts result type: {type(userposts_result)}")
                
                if isinstance(userposts_result, tuple) and len(userposts_result) == 2:
                    # Unpack features and prediction
                    features, prediction = userposts_result
                    logger.info(f"UserPosts prediction: {prediction}")
                    similarity_predictions[setup_id]['userposts'] = prediction
                else:
                    logger.info("UserPosts agent did not return a prediction")
                    
                results['userposts'] = True
            except Exception as e:
                logger.error(f"Error processing user posts for {setup_id}: {e}")
                results['userposts'] = False
        
        # Store similarity predictions if in prediction mode
        logger.info(f"Similarity predictions: {similarity_predictions}")
        if mode == "prediction" and any(similarity_predictions.values()):
            logger.info("Storing similarity predictions")
            self._store_similarity_predictions(similarity_predictions)
        else:
            logger.info("No similarity predictions to store or not in prediction mode")
        
        # Print summary
        for agent_name, success in results.items():
            logger.info(f"✅ {agent_name.capitalize()} features: {'Success' if success else 'Failed'}")
        
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
        """
        Store similarity-based predictions in DuckDB
        
        Args:
            predictions: Dictionary of predictions by setup_id and domain
        """
        if not predictions:
            logger.info("No similarity predictions to store")
            return
        
        # Debug: Print predictions structure
        logger.info(f"Predictions structure: {predictions}")
            
        try:
            # Connect to DuckDB
            conn = duckdb.connect(self.db_path)
            
            # Create table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS similarity_predictions (
                    setup_id VARCHAR,
                    domain VARCHAR,
                    predicted_outperformance DOUBLE,
                    confidence DOUBLE,
                    positive_ratio DOUBLE,
                    negative_ratio DOUBLE,
                    neutral_ratio DOUBLE,
                    similar_cases_count INTEGER,
                    prediction_timestamp VARCHAR,
                    PRIMARY KEY (setup_id, domain)
                )
            """)
            
            # Convert predictions to rows
            rows = []
            for setup_id, domains in predictions.items():
                logger.info(f"Processing setup_id: {setup_id}, domains: {list(domains.keys())}")
                for domain, prediction in domains.items():
                    if prediction:  # Skip empty predictions
                        logger.info(f"Processing domain: {domain}, prediction: {prediction}")
                        row = (
                            setup_id,
                            domain,
                            prediction.get('predicted_outperformance', 0.0),
                            prediction.get('confidence', 0.0),
                            prediction.get('positive_ratio', 0.0),
                            prediction.get('negative_ratio', 0.0),
                            prediction.get('neutral_ratio', 0.0),
                            prediction.get('similar_cases_count', 0),
                            prediction.get('prediction_timestamp', datetime.now().isoformat())
                        )
                        rows.append(row)
            
            # Insert rows
            if rows:
                logger.info(f"Inserting {len(rows)} rows into similarity_predictions table")
                conn.executemany("""
                    INSERT OR REPLACE INTO similarity_predictions
                    (setup_id, domain, predicted_outperformance, confidence, 
                     positive_ratio, negative_ratio, neutral_ratio, 
                     similar_cases_count, prediction_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rows)
                
                # Verify insertion
                count = conn.execute("SELECT COUNT(*) FROM similarity_predictions").fetchone()[0]
                logger.info(f"Verified {count} records in similarity_predictions table")
            else:
                logger.info("No valid similarity predictions to store")
                
            # Commit and close
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing similarity predictions: {e}")
            # Continue without failing the pipeline
    
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
    parser.add_argument('--lancedb-dir', default='lancedb_store',
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