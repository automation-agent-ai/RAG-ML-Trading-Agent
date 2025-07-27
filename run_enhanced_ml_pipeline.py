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
        self.feature_merger = MLFeatureMerger(
            db_path=db_path,
            prediction_db_path=prediction_db_path
        )
        
        # Agents will be initialized later with the appropriate mode
        self.news_agent = None
        self.fundamentals_agent = None
        self.analyst_agent = None
        self.userposts_agent = None

    def _init_agents(self, mode: str = "training"):
        """Initialize agents with the appropriate mode"""
        # Initialize News Agent
        from agents.news.enhanced_news_agent_duckdb import EnhancedNewsAgentDuckDB
        self.news_agent = EnhancedNewsAgentDuckDB(
            db_path=self.db_path,
            lancedb_dir=self.lancedb_dir,
            mode=mode
        )
        
        # Initialize Fundamentals Agent
        from agents.fundamentals.enhanced_fundamentals_agent_duckdb import EnhancedFundamentalsAgentDuckDB
        self.fundamentals_agent = EnhancedFundamentalsAgentDuckDB(
            db_path=self.db_path,
            lancedb_dir=self.lancedb_dir,
            mode=mode
        )
        
        # Initialize Analyst Recommendations Agent
        from agents.analyst_recommendations.enhanced_analyst_recommendations_agent_duckdb import EnhancedAnalystRecommendationsAgentDuckDB
        self.analyst_agent = EnhancedAnalystRecommendationsAgentDuckDB(
            db_path=self.db_path,
            lancedb_dir=self.lancedb_dir,
            mode=mode
        )
        
        # Initialize UserPosts Agent
        from agents.userposts.enhanced_userposts_agent_complete import EnhancedUserPostsAgentComplete
        self.userposts_agent = EnhancedUserPostsAgentComplete(
            db_path=self.db_path,
            lancedb_dir=self.lancedb_dir,
            mode=mode
        )

    def find_training_setups(self) -> List[str]:
        """Find setups with complete data for training"""
        conn = duckdb.connect(self.db_path)
        
        try:
            # Find setups with complete data
            query = """
            SELECT s.setup_id
            FROM setups s
            JOIN labels l ON s.setup_id = l.setup_id
            WHERE l.outperformance_10d IS NOT NULL
            """
            
            setup_ids = [row[0] for row in conn.execute(query).fetchall()]
            logger.info(f"Found {len(setup_ids)} setups with complete data")
            
            return setup_ids
        except Exception as e:
            logger.error(f"Error finding training setups: {e}")
            return []
        finally:
            conn.close()
    
    def load_setup_list(self, filename: str) -> List[str]:
        """Load setup IDs from a file"""
        try:
            with open(filename, 'r') as f:
                setup_ids = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Loaded {len(setup_ids)} setup IDs from {filename}")
            return setup_ids
        except Exception as e:
            logger.error(f"Error loading setup list: {e}")
            return []

    def extract_features(self, setup_ids: List[str], mode: str = "training", domains: List[str] = ['all']) -> Dict[str, Dict[str, Any]]:
        """
        Extract features for the specified setups
        
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
    
    def _store_similarity_predictions(self, extraction_results: Dict[str, Dict[str, Any]], mode: str = "prediction") -> None:
        """
        Store similarity-based predictions in the similarity_predictions table
        
        Args:
            extraction_results: Results from extract_features
            mode: Either 'training' or 'prediction'
        """
        if mode != "prediction":
            logger.info("Skipping similarity prediction storage in non-prediction mode")
            return
            
        logger.info("💾 Storing similarity predictions...")
        
        try:
            # Connect to database
            conn = duckdb.connect(self.db_path)
            
            # Ensure table exists
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
            
            # Extract predictions from each domain
            rows = []
            
            for domain, setups in extraction_results.items():
                for setup_id, result in setups.items():
                    # Get prediction method based on domain
                    if domain == 'news' and self.news_agent:
                        content = result.synthetic_summary_financial_results if hasattr(result, 'synthetic_summary_financial_results') else ""
                        if content:
                            prediction = self.news_agent.predict_via_similarity(content)
                    elif domain == 'fundamentals' and self.fundamentals_agent:
                        content = f"ROA: {result.roa if hasattr(result, 'roa') else 'N/A'}, ROE: {result.roe if hasattr(result, 'roe') else 'N/A'}"
                        prediction = self.fundamentals_agent.predict_via_similarity(content)
                    elif domain == 'analyst' and self.analyst_agent:
                        content = result.synthetic_summary if hasattr(result, 'synthetic_summary') else ""
                        if content:
                            prediction = self.analyst_agent.predict_via_similarity(content)
                    elif domain == 'userposts' and self.userposts_agent:
                        content = result.synthetic_summary if hasattr(result, 'synthetic_summary') else ""
                        if content:
                            prediction = self.userposts_agent.predict_via_similarity(content)
                    else:
                        continue
                    
                    # Create row if prediction exists
                    if prediction:
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
        
        # Step 2: Store similarity predictions (in prediction mode)
        if mode == "prediction":
            self._store_similarity_predictions(extraction_results, mode)
        
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
    parser.add_argument('--similarity-only', action='store_true',
                       help='Only extract similarity features and store predictions')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = EnhancedPipeline(
        db_path=args.db_path,
        prediction_db_path=args.prediction_db_path,
        lancedb_dir=args.lancedb_dir
    )
    
    # Get setup IDs
    setup_ids = []
    
    if args.setup_ids:
        setup_ids = args.setup_ids
        logger.info(f"Using {len(setup_ids)} setup IDs from command line")
    elif args.setup_list:
        setup_ids = pipeline.load_setup_list(args.setup_list)
    else:
        if args.mode == 'training':
            setup_ids = pipeline.find_training_setups()
        else:
            logger.error("In prediction mode, you must provide setup IDs or a setup list file")
            sys.exit(1)
    
    if not setup_ids:
        logger.error("No setup IDs to process")
        sys.exit(1)
    
    # Run pipeline
    if args.similarity_only:
        logger.info("🔍 SIMILARITY-ONLY MODE: Only extracting similarity features and storing predictions")
        # Initialize pipeline
        pipeline._init_agents(args.mode)
        
        # Get existing features from database
        conn = duckdb.connect(args.db_path)
        results = {}
        
        for domain in args.domains if 'all' not in args.domains else ['news', 'fundamentals', 'analyst', 'userposts']:
            domain_table = f"{domain}_features"
            query = f"""
            SELECT * FROM {domain_table}
            WHERE setup_id IN (
                SELECT UNNEST(?::VARCHAR[])
            )
            """
            try:
                df = conn.execute(query, [setup_ids]).df()
                if not df.empty:
                    results[domain] = {row.setup_id: row for _, row in df.iterrows()}
                    logger.info(f"Loaded {len(df)} existing {domain} features")
                else:
                    logger.warning(f"No {domain} features found for the specified setup IDs")
            except Exception as e:
                logger.error(f"Error loading {domain} features: {e}")
        
        conn.close()
        
        # Store similarity predictions
        pipeline._store_similarity_predictions(results, args.mode)
        
        logger.info("✅ Similarity predictions stored successfully")
    else:
        result = pipeline.run_pipeline(setup_ids, args.mode, args.domains)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("📊 PIPELINE SUMMARY")
        logger.info("="*50)
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Domains: {', '.join(args.domains)}")
        logger.info(f"Setups processed: {result['setups_processed']}")
        logger.info(f"Duration: {result['duration_seconds']:.2f} seconds")

if __name__ == "__main__":
    main() 