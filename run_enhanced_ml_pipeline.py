#!/usr/bin/env python3
"""
Enhanced ML Pipeline Runner with Similarity Search

This script runs the enhanced ML pipeline that includes:
1. Training mode: Create embeddings with labels and store in LanceDB
2. Prediction mode: Create embeddings without labels, use for similarity search
3. Enhanced feature extraction with similarity-based features
4. Direct similarity-based predictions
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import duckdb
from typing import List, Dict

from run_complete_ml_pipeline import MLPipelineRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_training_setups(db_path: str) -> List[str]:
    """Get all setups with complete data for training"""
    conn = duckdb.connect(db_path)
    
    # Find setups that have all required features and labels
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
    
    return setup_ids

def main():
    """Run the enhanced ML pipeline"""
    parser = argparse.ArgumentParser(description='Run enhanced ML pipeline with similarity search')
    
    parser.add_argument('--mode', choices=['training', 'prediction'], default='training',
                       help='Pipeline mode: training or prediction')
    parser.add_argument('--setup-ids', nargs='+', help='List of setup_ids to process (required for prediction mode)')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--lancedb-dir', default='storage/lancedb_store',
                       help='Path to LanceDB directory')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'prediction' and not args.setup_ids:
        parser.error("--setup-ids is required for prediction mode")
    
    # Initialize pipeline runner
    pipeline = MLPipelineRunner(
        db_path=args.db_path,
        lancedb_dir=args.lancedb_dir,
        mode=args.mode
    )
    
    # Get setup IDs
    if args.mode == 'training':
        setup_ids = get_training_setups(args.db_path)
        logger.info(f"Found {len(setup_ids)} setups with complete data for training")
    else:
        setup_ids = args.setup_ids
        logger.info(f"Processing {len(setup_ids)} setups in prediction mode")
    
    # Run pipeline and time it
    start_time = datetime.now()
    
    try:
        pipeline.run_pipeline(setup_ids)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("ENHANCED ML PIPELINE SUMMARY")
        logger.info("="*50)
        logger.info(f"Mode: {args.mode.upper()}")
        logger.info(f"Setups processed: {len(setup_ids)}")
        logger.info(f"Duration: {duration:.1f} seconds")
        
        # Show additional stats in prediction mode
        if args.mode == 'prediction':
            # Count similarity predictions
            with duckdb.connect(args.db_path) as conn:
                pred_count = conn.execute("""
                    SELECT COUNT(*) FROM similarity_predictions 
                    WHERE setup_id IN (SELECT UNNEST(?))
                """, [setup_ids]).fetchone()[0]
                
                logger.info(f"\nSimilarity Predictions:")
                logger.info(f"- Total predictions: {pred_count}")
                logger.info(f"- Coverage: {(pred_count/len(setup_ids))*100:.1f}%")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 