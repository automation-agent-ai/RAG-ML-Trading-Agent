#!/usr/bin/env python3
"""
Test the ML feature merger with a specific setup ID

This script tests the ML feature merger with a specific setup ID to see if it can find the feature tables.
"""

import os
import sys
import logging
from pathlib import Path
import duckdb

# Add project root to path for proper imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.ml_feature_merger import MLFeatureMerger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_ml_feature_merger(setup_id: str, db_path: str = "data/sentiment_system.duckdb"):
    """
    Test the ML feature merger with a specific setup ID
    
    Args:
        setup_id: Setup ID to test with
        db_path: Path to DuckDB database
    """
    logger.info(f"Testing ML feature merger with setup ID: {setup_id}")
    
    # Initialize the ML feature merger
    merger = MLFeatureMerger(db_path=db_path)
    
    # Check table existence first
    conn = duckdb.connect(db_path)
    try:
        # Check if feature tables exist
        for table in ['news_features', 'userposts_features', 'analyst_recommendations_features', 'setups']:
            exists = conn.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table}'").fetchone()[0]
            logger.info(f"Table '{table}' exists: {exists > 0}")
            
            if exists > 0:
                count = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE setup_id = '{setup_id}'").fetchone()[0]
                logger.info(f"Records in '{table}' for setup_id '{setup_id}': {count}")
    finally:
        conn.close()
    
    # Test merging text features
    logger.info("\nTesting merge_text_features...")
    text_result = merger.merge_text_features([setup_id], mode='prediction')
    logger.info(f"Text features result: {text_result}")
    
    # Test merging financial features
    logger.info("\nTesting merge_financial_features...")
    financial_result = merger.merge_financial_features([setup_id], mode='prediction')
    logger.info(f"Financial features result: {financial_result}")
    
    # Test merging all features
    logger.info("\nTesting merge_all_features...")
    all_result = merger.merge_all_features([setup_id], mode='prediction')
    logger.info(f"All features result: {all_result}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ML feature merger')
    parser.add_argument('--setup-id', default='AFN_2023-11-20',
                      help='Setup ID to test with')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                      help='Path to DuckDB database')
    
    args = parser.parse_args()
    
    test_ml_feature_merger(args.setup_id, args.db_path)

if __name__ == "__main__":
    main() 