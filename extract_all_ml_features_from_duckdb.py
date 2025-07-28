#!/usr/bin/env python3
"""
Extract All ML Features From DuckDB

This script runs both financial and text feature extractions to create comprehensive ML feature tables.

Usage:
    python extract_all_ml_features_from_duckdb.py --mode [training|prediction] --setup-list [file]
"""

import os
import sys
import argparse
import logging
import duckdb
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import feature extractors
from extract_financial_features_from_duckdb import FinancialFeaturesExtractor
from extract_text_features_from_duckdb import TextFeaturesExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Extract all ML features from DuckDB')
    parser.add_argument('--mode', choices=['training', 'prediction'], default='training',
                       help='Mode: training or prediction')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--output-dir', default='data/ml_features',
                       help='Directory to save output CSV files')
    parser.add_argument('--setup-list', help='File containing setup IDs to process (one per line)')
    parser.add_argument('--features', choices=['text', 'financial', 'all'], default='all',
                       help='Which features to extract')
    
    args = parser.parse_args()
    
    # Get setup IDs if provided
    setup_ids = None
    if args.setup_list:
        with open(args.setup_list, 'r') as f:
            setup_ids = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(setup_ids)} setup IDs from {args.setup_list}")
    else:
        # Use all setup IDs
        conn = duckdb.connect(args.db_path)
        setup_ids = conn.execute('SELECT DISTINCT setup_id FROM setups LIMIT 1000').df()['setup_id'].tolist()
        conn.close()
        logger.info(f"Using {len(setup_ids)} setup IDs from database")
    
    results = {}
    
    # Extract text features if requested
    if args.features in ['text', 'all']:
        try:
            text_extractor = TextFeaturesExtractor(
                db_path=args.db_path,
                output_dir=args.output_dir
            )
            results['text'] = text_extractor.extract_text_features(setup_ids, args.mode)
        except Exception as e:
            logger.error(f"❌ Error extracting text features: {str(e)}")
            results['text'] = {"error": str(e)}
    
    # Extract financial features if requested
    if args.features in ['financial', 'all']:
        try:
            financial_extractor = FinancialFeaturesExtractor(
                db_path=args.db_path,
                output_dir=args.output_dir
            )
            results['financial'] = financial_extractor.extract_financial_features(setup_ids, args.mode)
        except Exception as e:
            logger.error(f"❌ Error extracting financial features: {str(e)}")
            results['financial'] = {"error": str(e)}
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("📊 ML FEATURES EXTRACTION SUMMARY")
    logger.info("="*50)
    
    for feature_type, result in results.items():
        if "error" in result:
            logger.info(f"{feature_type.upper()} FEATURES: ❌ Error: {result['error']}")
        else:
            logger.info(f"{feature_type.upper()} FEATURES: ✅ Success")
            logger.info(f"- Table: {result['table_name']}")
            logger.info(f"- Features: {result['feature_count']}")
            logger.info(f"- Rows: {result['row_count']}")
            logger.info(f"- Output: {result['output_file']}")
        logger.info("-"*50)
    
    logger.info("✅ ML Features Extraction Complete!")

if __name__ == "__main__":
    main() 