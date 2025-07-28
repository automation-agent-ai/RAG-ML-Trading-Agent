#!/usr/bin/env python3
"""
test_pipeline.py - Test the ML pipeline with a specific setup ID in prediction mode

This script tests the ML pipeline with a specific setup ID in prediction mode
to verify that the pipeline works correctly with the fixed embedders and agents.
"""

import os
import sys
import logging
from pathlib import Path
import duckdb
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_available_setup_ids(db_path: str = "data/sentiment_system.duckdb") -> list:
    """Get a list of available setup IDs from the database"""
    try:
        conn = duckdb.connect(db_path)
        setups = conn.execute("SELECT setup_id FROM setups LIMIT 10").fetchall()
        conn.close()
        return [setup[0] for setup in setups]
    except Exception as e:
        logger.error(f"Error getting setup IDs: {e}")
        return []

def test_pipeline_prediction(setup_id: str):
    """Test the ML pipeline with a specific setup ID in prediction mode"""
    logger.info(f"Testing ML pipeline with setup ID: {setup_id}")
    
    # Import the pipeline
    from run_complete_ml_pipeline import CompletePipeline
    
    # Initialize the pipeline
    pipeline = CompletePipeline(
        db_path="data/sentiment_system.duckdb",
        lancedb_dir="lancedb_store"
    )
    
    # Run the pipeline in prediction mode
    start_time = datetime.now()
    result = pipeline.run_complete_pipeline([setup_id], mode='prediction')
    end_time = datetime.now()
    
    # Print results
    logger.info(f"Pipeline completed in {(end_time - start_time).total_seconds():.2f} seconds")
    logger.info(f"Result: {result}")
    
    # Check if similarity predictions were created
    try:
        conn = duckdb.connect("data/sentiment_system.duckdb")
        
        # Check if similarity_predictions table exists
        table_exists = conn.execute("""
            SELECT count(*) FROM information_schema.tables 
            WHERE table_name = 'similarity_predictions'
        """).fetchone()[0]
        
        if table_exists:
            # Get predictions for this setup
            predictions = conn.execute(f"""
                SELECT * FROM similarity_predictions
                WHERE setup_id = '{setup_id}'
            """).fetchdf()
            
            if len(predictions) > 0:
                logger.info(f"Found {len(predictions)} similarity predictions for {setup_id}")
                logger.info(f"Predictions: {predictions.to_dict('records')}")
            else:
                logger.warning(f"No similarity predictions found for {setup_id}")
        else:
            logger.warning("similarity_predictions table does not exist")
            
        conn.close()
    except Exception as e:
        logger.error(f"Error checking similarity predictions: {e}")

def main():
    """Main function"""
    # Get available setup IDs
    setup_ids = get_available_setup_ids()
    
    if not setup_ids:
        logger.error("No setup IDs found in the database")
        return
    
    # Use the first setup ID for testing
    setup_id = setup_ids[0]
    logger.info(f"Using setup ID: {setup_id}")
    
    # Test the pipeline
    test_pipeline_prediction(setup_id)

if __name__ == "__main__":
    main() 