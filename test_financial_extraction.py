#!/usr/bin/env python3
"""
Test Financial Features Extraction
"""

import duckdb
import logging
from extract_financial_features import FinancialFeaturesExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Connect to the database
    conn = duckdb.connect('data/sentiment_system.duckdb')
    
    # Get a few setup IDs
    setup_ids = conn.execute('SELECT DISTINCT setup_id FROM setups LIMIT 10').df()['setup_id'].tolist()
    conn.close()
    
    logger.info(f"Using {len(setup_ids)} setup IDs: {setup_ids}")
    
    # Initialize extractor
    extractor = FinancialFeaturesExtractor(
        db_path='data/sentiment_system.duckdb',
        output_dir='data/ml_features'
    )
    
    # Extract features
    try:
        result = extractor.extract_financial_features(setup_ids, 'training')
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("📊 FINANCIAL FEATURES EXTRACTION SUMMARY")
        logger.info("="*50)
        logger.info(f"✅ Success")
        logger.info(f"- Table: {result['table_name']}")
        logger.info(f"- Features: {result['feature_count']}")
        logger.info(f"- Rows: {result['row_count']}")
        logger.info(f"- Output: {result['output_file']}")
        logger.info("="*50)
    except Exception as e:
        logger.error(f"❌ Error extracting financial features: {str(e)}")

if __name__ == '__main__':
    main() 