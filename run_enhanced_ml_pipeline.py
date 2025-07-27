#!/usr/bin/env python3
"""
Enhanced ML Pipeline for Stock Performance Prediction

This script orchestrates the enhanced ML pipeline with proper training/prediction separation:
1. Uses separate embedding tables for training and prediction
2. Extracts features with appropriate mode (training vs. prediction)
3. Merges features into ML feature tables
4. Trains/predicts ML models with no data leakage

Usage:
    python run_enhanced_ml_pipeline.py --mode training
    python run_enhanced_ml_pipeline.py --mode prediction --setup-ids SETUP1 SETUP2
"""

import os
import sys
import argparse
import logging
import duckdb
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path for proper imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.ml_feature_merger import MLFeatureMerger

class EnhancedPipeline:
    """Enhanced pipeline with proper training/prediction separation"""
    
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
        self.feature_merger = MLFeatureMerger(db_path=db_path)
        
        # Initialize agents lazily
        self.news_agent = None
        self.fundamentals_agent = None
        self.analyst_agent = None
        self.userposts_agent = None
        
        # Domain tables
        self.domains = ["news", "fundamentals", "analyst_recommendations", "userposts"]
        
    def _init_agents(self, mode: str = "training"):
        """Initialize all agents with appropriate mode"""
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
    
    def find_training_setups(self) -> List[str]:
        """Find setups for training (with complete data and labels)"""
        conn = duckdb.connect(self.db_path)
        
        # Find setups with complete data and labels
        query = """
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
        
        setup_ids = [row[0] for row in conn.execute(query).fetchall()]
        conn.close()
        
        logger.info(f"Found {len(setup_ids)} setups for training")
        return setup_ids
    
    def load_setup_list(self, filename: str) -> List[str]:
        """Load a list of setup IDs from a file"""
        if not os.path.exists(filename):
            logger.warning(f"Setup list file {filename} not found")
            return []
        
        with open(filename, 'r') as f:
            setup_ids = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(setup_ids)} setups from {filename}")
        return setup_ids
    
    def extract_features(self, setup_ids: List[str], mode: str = "training", domains: List[str] = ['all']) -> Dict[str, Dict[str, Any]]:
        """
        Extract features for the given setup IDs
        
        Args:
            setup_ids: List of setup IDs to process
            mode: Either 'training' or 'prediction'
            domains: List of domains to process ('all', 'news', 'fundamentals', 'analyst', 'userposts')
            
        Returns:
            Dictionary with results for each domain
        """
        logger.info(f"🔧 EXTRACTING FEATURES ({mode.upper()} MODE)")
        logger.info(f"Processing {len(setup_ids)} setups")
        
        # Initialize agents
        self._init_agents(mode)
        
        results = {}
        process_all = 'all' in domains
        
        # Process news domain
        if process_all or 'news' in domains:
            logger.info("📰 Processing news features...")
            try:
                news_results = {}
                for setup_id in setup_ids:
                    news_result = self.news_agent.process_setup(setup_id, mode)
                    news_results[setup_id] = news_result
                results['news'] = news_results
                logger.info(f"✅ News features: {len(news_results)} processed")
            except Exception as e:
                logger.error(f"❌ Error processing news features: {e}")
                results['news'] = {}
        
        # Process fundamentals domain
        if process_all or 'fundamentals' in domains:
            logger.info("📊 Processing fundamentals features...")
            try:
                fundamentals_results = {}
                for setup_id in setup_ids:
                    fundamentals_result = self.fundamentals_agent.process_setup(setup_id, mode)
                    fundamentals_results[setup_id] = fundamentals_result
                results['fundamentals'] = fundamentals_results
                logger.info(f"✅ Fundamentals features: {len(fundamentals_results)} processed")
            except Exception as e:
                logger.error(f"❌ Error processing fundamentals features: {e}")
                results['fundamentals'] = {}
        
        # Process analyst recommendations domain
        if process_all or 'analyst' in domains:
            logger.info("📈 Processing analyst recommendations features...")
            try:
                analyst_results = {}
                for setup_id in setup_ids:
                    analyst_result = self.analyst_agent.process_setup(setup_id, mode)
                    analyst_results[setup_id] = analyst_result
                results['analyst'] = analyst_results
                logger.info(f"✅ Analyst recommendations features: {len(analyst_results)} processed")
            except Exception as e:
                logger.error(f"❌ Error processing analyst recommendations features: {e}")
                results['analyst'] = {}
        
        # Process userposts domain
        if process_all or 'userposts' in domains:
            logger.info("💬 Processing userposts features...")
            try:
                userposts_results = {}
                for setup_id in setup_ids:
                    userposts_result = self.userposts_agent.process_setup(setup_id, mode)
                    userposts_results[setup_id] = userposts_result
                results['userposts'] = userposts_results
                logger.info(f"✅ Userposts features: {len(userposts_results)} processed")
            except Exception as e:
                logger.error(f"❌ Error processing userposts features: {e}")
                results['userposts'] = {}
        
        return results
    
    def create_ml_features(self, setup_ids: List[str], mode: str = 'training') -> Dict[str, Dict[str, int]]:
        """Create ML feature tables by merging domain features"""
        logger.info(f"\n🔄 Creating {mode} ML feature tables...")
        
        # Use the MLFeatureMerger to create comprehensive ML feature tables
        results = self.feature_merger.merge_all_features(setup_ids, mode)
        
        return results
    
    def run_pipeline(self, setup_ids: List[str], mode: str = 'training', domains: List[str] = ['all']) -> Dict[str, Any]:
        """
        Run the enhanced pipeline
        
        Args:
            setup_ids: List of setup IDs to process
            mode: Either 'training' or 'prediction'
            domains: List of domains to process ('all', 'news', 'fundamentals', 'analyst', 'userposts')
            
        Returns:
            Dictionary with results summary
        """
        logger.info("🚀 STARTING ENHANCED PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Mode: {mode}")
        logger.info(f"Domains: {', '.join(domains)}")
        logger.info(f"Setups: {len(setup_ids)}")
        
        start_time = datetime.now()
        
        # Step 1: Extract features with appropriate mode
        extraction_results = self.extract_features(setup_ids, mode, domains)
        
        # Step 2: Create ML feature tables
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
            'domains_processed': domains,
            'extraction_results': extraction_results,
            'ml_features_results': ml_features_results
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run enhanced ML pipeline')
    parser.add_argument('--mode', choices=['training', 'prediction'], default='training',
                       help='Pipeline mode: training or prediction')
    parser.add_argument('--setup-ids', nargs='+', help='List of setup_ids to process (optional)')
    parser.add_argument('--setup-list', help='File containing setup IDs to process (one per line)')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--prediction-db-path', default='data/prediction_features.duckdb',
                       help='Path to store prediction features')
    parser.add_argument('--lancedb-dir', default='lancedb_store',
                       help='Path to LanceDB directory')
    parser.add_argument('--domains', nargs='+', choices=['all', 'news', 'fundamentals', 'analyst', 'userposts'], 
                       default=['all'], help='Domains to process (default: all)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = EnhancedPipeline(
        db_path=args.db_path,
        prediction_db_path=args.prediction_db_path,
        lancedb_dir=args.lancedb_dir
    )
    
    # Determine setup IDs to process
    setup_ids = []
    
    if args.setup_ids:
        # Use provided setup IDs
        setup_ids = args.setup_ids
        logger.info(f"Using {len(setup_ids)} provided setup IDs")
    elif args.setup_list:
        # Load setup IDs from file
        setup_ids = pipeline.load_setup_list(args.setup_list)
        logger.info(f"Loaded {len(setup_ids)} setup IDs from {args.setup_list}")
    elif args.mode == 'training':
        # For training mode, find setups with complete data
        setup_ids = pipeline.find_training_setups()
        logger.info(f"Found {len(setup_ids)} setups for training")
    else:
        # For prediction mode, setup IDs are required
        parser.error("Setup IDs are required for prediction mode. Use --setup-ids or --setup-list")
    
    if not setup_ids:
        logger.error("No setup IDs to process")
        return
    
    # Run the pipeline
    results = pipeline.run_pipeline(setup_ids, args.mode, args.domains)
    
    # Print summary
    logger.info("\n🎉 Pipeline Complete!")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Domains: {', '.join(args.domains)}")
    logger.info(f"Duration: {results['duration_seconds']:.1f}s")
    logger.info(f"Setups processed: {len(setup_ids)}")


if __name__ == "__main__":
    main() 