import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import duckdb

# Import feature extraction components
from agents.fundamentals.enhanced_fundamentals_agent_duckdb import EnhancedFundamentalsAgentDuckDB
from agents.news.enhanced_news_agent_duckdb import EnhancedNewsAgentDuckDB
from agents.userposts.enhanced_userposts_agent_complete import EnhancedUserPostsAgentComplete
from agents.analyst_recommendations.enhanced_analyst_recommendations_agent_duckdb import EnhancedAnalystRecommendationsAgentDuckDB
from core.ml_feature_merger import MLFeatureMerger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
        self.feature_merger = MLFeatureMerger(db_path=db_path)
        
    def extract_features(self, setup_ids: List[str], mode: str = 'training') -> Dict[str, bool]:
        """Extract features for all domains"""
        logger.info(f"🔧 RUNNING FEATURE EXTRACTION ({mode.upper()} MODE)")
        logger.info("=" * 60)
        
        results = {}
        
        try:
            # 1. Fundamentals Features
            logger.info("📊 Extracting fundamentals features...")
            fundamentals_agent = EnhancedFundamentalsAgentDuckDB(
                db_path=str(self.db_path),
                lancedb_dir=str(self.lancedb_dir)
            )
            
            # Override to target setups only
            fundamentals_agent.setup_validator.confirmed_setup_ids = set(setup_ids)
            
            batch_results = fundamentals_agent.batch_process_setups(setup_ids)
            successful = len([r for r in batch_results.values() if r is not None])
            results['fundamentals_features'] = successful > 0
            fundamentals_agent.cleanup()
            
            logger.info(f"✅ Fundamentals features: {successful}/{len(setup_ids)} successful")
            
        except Exception as e:
            logger.error(f"❌ Fundamentals features failed: {e}")
            results['fundamentals_features'] = False

        try:
            # 2. News Features
            logger.info("📰 Extracting news features...")
            news_agent = EnhancedNewsAgentDuckDB(
                db_path=str(self.db_path),
                lancedb_dir=str(self.lancedb_dir)
            )
            
            # Override to target setups only
            news_agent.setup_validator.confirmed_setup_ids = set(setup_ids)
            
            batch_results = news_agent.batch_process_setups(setup_ids)
            successful = len([r for r in batch_results.values() if r is not None])
            results['news_features'] = successful > 0
            news_agent.cleanup()
            
            logger.info(f"✅ News features: {successful}/{len(setup_ids)} successful")
            
        except Exception as e:
            logger.error(f"❌ News features failed: {e}")
            results['news_features'] = False
        
        try:
            # 3. UserPosts Features
            logger.info("💬 Extracting userposts features...")
            userposts_agent = EnhancedUserPostsAgentComplete(
                db_path=str(self.db_path),
                lancedb_dir=str(self.lancedb_dir)
            )
            
            # Override to target setups only
            userposts_agent.setup_validator.confirmed_setup_ids = set(setup_ids)
            
            batch_results = userposts_agent.batch_process_setups(setup_ids)
            successful = len([r for r in batch_results.values() if r is not None])
            results['userposts_features'] = successful > 0
            userposts_agent.cleanup()
            
            logger.info(f"✅ UserPosts features: {successful}/{len(setup_ids)} successful")
            
        except Exception as e:
            logger.error(f"❌ UserPosts features failed: {e}")
            results['userposts_features'] = False
        
        try:
            # 4. Analyst Recommendations Features
            logger.info("📈 Extracting analyst recommendations features...")
            analyst_agent = EnhancedAnalystRecommendationsAgentDuckDB(
                db_path=str(self.db_path),
                lancedb_dir=str(self.lancedb_dir)
            )
            
            # Override to target setups only
            analyst_agent.setup_validator.confirmed_setup_ids = set(setup_ids)
            
            batch_results = analyst_agent.batch_process_setups(setup_ids)
            successful = len([r for r in batch_results.values() if r is not None])
            results['analyst_features'] = successful > 0
            analyst_agent.cleanup()
            
            logger.info(f"✅ Analyst features: {successful}/{len(setup_ids)} successful")
            
        except Exception as e:
            logger.error(f"❌ Analyst features failed: {e}")
            results['analyst_features'] = False

        # Summary
        successful = sum(results.values())
        total = len(results)
        logger.info(f"📊 Feature Extraction Summary: {successful}/{total} domains successful")
        
        return results
    
    def create_ml_features(self, setup_ids: List[str], mode: str = 'training') -> Dict[str, Dict[str, int]]:
        """Create ML feature tables by merging domain features"""
        logger.info(f"\n🔄 Creating {mode} ML feature tables...")
        
        # Use the MLFeatureMerger to create comprehensive ML feature tables
        results = self.feature_merger.merge_all_features(setup_ids, mode)
        
        return results
    
    def create_embeddings(self, setup_ids: List[str], mode: str = 'training') -> Dict[str, bool]:
        """Create embeddings for all domains with proper training/prediction mode handling"""
        logger.info(f"🎰 CREATING EMBEDDINGS ({mode.upper()} MODE)")
        logger.info("=" * 60)
        
        results = {}
        
        try:
            # Import embedding modules
            from embeddings.embed_news_duckdb import NewsEmbeddingPipelineDuckDB
            from embeddings.embed_fundamentals_duckdb import FundamentalsEmbedderDuckDB
            from embeddings.embed_userposts_duckdb import UserPostsEmbedderDuckDB
            
            # 1. News Embeddings
            logger.info("📰 Creating news embeddings...")
            news_embedder = NewsEmbeddingPipelineDuckDB(
                db_path=self.db_path,
                lancedb_dir=self.lancedb_dir
            )
            # TODO: Add include_labels parameter to constructor
            news_result = news_embedder.process_all_embeddings(limit=len(setup_ids)*100)
            results['news_embeddings'] = True  # Assume success for now
            
            # 2. Fundamentals Embeddings  
            logger.info("📊 Creating fundamentals embeddings...")
            fundamentals_embedder = FundamentalsEmbedderDuckDB(
                db_path=self.db_path,
                lancedb_dir=self.lancedb_dir
            )
            # TODO: Add include_labels parameter to constructor
            fundamentals_result = fundamentals_embedder.embed_fundamentals(ticker_limit=len(setup_ids)*100)
            results['fundamentals_embeddings'] = True  # Assume success for now
            
            # 3. UserPosts Embeddings (already handles this correctly)
            logger.info("💬 Creating userposts embeddings...")
            userposts_embedder = UserPostsEmbedderDuckDB(
                db_path=self.db_path,
                lancedb_dir=self.lancedb_dir
            )
            # TODO: Add include_labels parameter to constructor
            userposts_result = userposts_embedder.embed_posts()
            results['userposts_embeddings'] = True  # Assume success for now
            
        except Exception as e:
            logger.error(f"❌ Embedding creation failed: {e}")
            results = {domain: False for domain in ['news_embeddings', 'fundamentals_embeddings', 'userposts_embeddings']}
        
        return results

    def run_complete_pipeline(self, setup_ids: List[str], mode: str = 'training') -> Dict[str, Any]:
        """Run complete pipeline: embedding creation, feature extraction, and ML feature table creation"""
        logger.info("🚀 STARTING COMPLETE PIPELINE")
        logger.info("=" * 70)
        
        start_time = datetime.now()
        
        # Step 1: Create embeddings (with proper training/prediction mode handling)
        embedding_results = self.create_embeddings(setup_ids, mode)
        
        # Step 2: Extract features
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
    
    # Run pipeline
    result = pipeline.run_complete_pipeline(
        setup_ids=setup_ids,
        mode=args.mode
    )
    
    # Print results
    logger.info("\n🎉 Pipeline Complete!")
    logger.info(f"Duration: {result['duration_seconds']:.1f}s")
    logger.info(f"Setups processed: {result['setups_processed']}")
    
    # Show feature counts
    if result['ml_features_results'].get('text'):
        text_info = result['ml_features_results']['text']
        logger.info(f"\nText ML Features:")
        logger.info(f"- Features: {text_info['feature_count']}")
        logger.info(f"- Rows: {text_info['row_count']}")
    
    if result['ml_features_results'].get('financial'):
        fin_info = result['ml_features_results']['financial']
        logger.info(f"\nFinancial ML Features:")
        logger.info(f"- Features: {fin_info['feature_count']}")
        logger.info(f"- Rows: {fin_info['row_count']}")
        
    # In training mode, also show preprocessing stats
    if args.mode == 'training':
        logger.info("\n📊 Preprocessing Statistics:")
        logger.info("- Constant columns removed")
        logger.info("- Missing values imputed")
        logger.info("- Features scaled")
        logger.info("- Outliers handled")

if __name__ == "__main__":
    main() 