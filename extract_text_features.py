#!/usr/bin/env python3
"""
Extract Text Features

This script extracts text features from the news_features, userposts_features, and analyst_recommendations_features tables
and creates a comprehensive text_ml_features table.

Usage:
    python extract_text_features.py --mode [training|prediction] --setup-list [file]
"""

import os
import sys
import argparse
import logging
import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextFeaturesExtractor:
    """Extracts text features from news_features, userposts_features, and analyst_recommendations_features tables"""
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        output_dir: str = "data/ml_features"
    ):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def extract_text_features(self, setup_ids: List[str], mode: str = 'training') -> Dict[str, Any]:
        """
        Extract text features for the given setup IDs
        
        Args:
            setup_ids: List of setup IDs to process
            mode: Either 'training' or 'prediction'
            
        Returns:
            Dictionary with feature and row counts
        """
        logger.info(f"🔄 Extracting text features for {len(setup_ids)} setups in {mode} mode...")
        
        conn = None
        try:
            conn = duckdb.connect(self.db_path)
            
            # Create text features query
            text_features_query = """
            -- Create a comprehensive text features table
            WITH setup_data AS (
                SELECT 
                    s.setup_id
                FROM setups s
                WHERE s.setup_id = ANY(?)
            ),
            -- Combine all text features
            combined_text_data AS (
                SELECT 
                    sd.setup_id,
                    
                    -- News features
                    n.count_financial_results,
                    n.max_severity_financial_results,
                    n.sentiment_score_financial_results,
                    n.profit_warning_present::INTEGER as profit_warning,
                    n.count_corporate_actions,
                    n.max_severity_corporate_actions,
                    n.sentiment_score_corporate_actions,
                    n.capital_raise_present::INTEGER as capital_raise,
                    n.count_governance,
                    n.max_severity_governance,
                    n.sentiment_score_governance,
                    n.board_change_present::INTEGER as board_change,
                    n.count_corporate_events,
                    n.max_severity_corporate_events,
                    n.sentiment_score_corporate_events,
                    n.contract_award_present::INTEGER as contract_award,
                    n.merger_or_acquisition_present::INTEGER as merger_acquisition,
                    n.count_other_signals,
                    n.max_severity_other_signals,
                    n.sentiment_score_other_signals,
                    n.broker_recommendation_present::INTEGER as broker_recommendation,
                    n.credit_rating_change_present::INTEGER as credit_rating_change,
                    
                    -- User Posts features
                    u.avg_sentiment as posts_avg_sentiment,
                    u.post_count,
                    u.community_sentiment_score,
                    u.bull_bear_ratio,
                    u.rumor_intensity,
                    u.trusted_user_sentiment,
                    u.relevance_score as posts_relevance,
                    u.engagement_score,
                    u.unique_users,
                    CAST(u.contrarian_signal AS INTEGER) as contrarian_signal_numeric,
                    
                    -- Analyst features
                    a.recommendation_count,
                    a.buy_recommendations,
                    a.sell_recommendations,
                    a.hold_recommendations,
                    a.coverage_breadth,
                    a.consensus_rating,
                    a.recent_upgrades,
                    a.recent_downgrades,
                    a.analyst_conviction_score,
                    
                    -- Label for training mode
                    CASE 
                        WHEN '{mode}' = 'training' THEN l.outperformance_10d 
                        ELSE NULL 
                    END as label
                
                FROM setup_data sd
                LEFT JOIN news_features n ON sd.setup_id = n.setup_id
                LEFT JOIN userposts_features u ON sd.setup_id = u.setup_id
                LEFT JOIN analyst_recommendations_features a ON sd.setup_id = a.setup_id
                LEFT JOIN (
                    -- Calculate average outperformance for first 10 days
                    SELECT 
                        setup_id, 
                        AVG(outperformance_day) as outperformance_10d
                    FROM daily_labels
                    WHERE day_number <= 10
                    GROUP BY setup_id
                    HAVING COUNT(*) >= 5  -- Relaxed: require at least 5 days
                ) l ON sd.setup_id = l.setup_id
            )
            
            SELECT * FROM combined_text_data
            """
            
            # Create or replace the table
            table_name = f"text_ml_features_{mode}"
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"CREATE TABLE {table_name} AS {text_features_query}", [setup_ids])
            
            # Get table info
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            feature_count = len(columns)
            
            # Export to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"{table_name}_{timestamp}.csv"
            
            df = conn.execute(f"SELECT * FROM {table_name}").df()
            df.to_csv(output_file, index=False)
            
            logger.info(f"✅ Text features extracted and exported:")
            logger.info(f"- Table: {table_name}")
            logger.info(f"- Features: {feature_count}")
            logger.info(f"- Rows: {row_count}")
            logger.info(f"- Exported to: {output_file}")
            
            # Check for non-null values in key columns
            news_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE count_financial_results IS NOT NULL").fetchone()[0]
            logger.info(f"- Rows with non-null count_financial_results: {news_count} ({news_count/row_count:.1%})")
            
            posts_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE post_count IS NOT NULL").fetchone()[0]
            logger.info(f"- Rows with non-null post_count: {posts_count} ({posts_count/row_count:.1%})")
            
            analyst_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE recommendation_count IS NOT NULL").fetchone()[0]
            logger.info(f"- Rows with non-null recommendation_count: {analyst_count} ({analyst_count/row_count:.1%})")
            
            return {
                "table_name": table_name,
                "feature_count": feature_count,
                "row_count": row_count,
                "output_file": str(output_file)
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting text features: {str(e)}")
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Extract text features')
    parser.add_argument('--mode', choices=['training', 'prediction'], default='training',
                       help='Mode: training or prediction')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--output-dir', default='data/ml_features',
                       help='Directory to save output CSV files')
    parser.add_argument('--setup-list', help='File containing setup IDs to process (one per line)')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = TextFeaturesExtractor(
        db_path=args.db_path,
        output_dir=args.output_dir
    )
    
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
    
    # Extract features
    result = extractor.extract_text_features(setup_ids, args.mode)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("📊 TEXT FEATURES EXTRACTION SUMMARY")
    logger.info("="*50)
    logger.info(f"✅ Success")
    logger.info(f"- Table: {result['table_name']}")
    logger.info(f"- Features: {result['feature_count']}")
    logger.info(f"- Rows: {result['row_count']}")
    logger.info(f"- Output: {result['output_file']}")
    logger.info("="*50)

if __name__ == "__main__":
    main() 