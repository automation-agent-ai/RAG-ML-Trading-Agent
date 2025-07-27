#!/usr/bin/env python3
"""
Check if a setup ID exists in feature tables

This script checks if a setup ID exists in the news_features, userposts_features,
and analyst_recommendations_features tables.
"""

import duckdb
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_features(setup_id: str, db_path: str = "data/sentiment_system.duckdb"):
    """
    Check if a setup ID exists in feature tables
    
    Args:
        setup_id: Setup ID to check
        db_path: Path to DuckDB database
    """
    conn = duckdb.connect(db_path)
    
    try:
        # Check news_features
        news_count = conn.execute(
            f"SELECT COUNT(*) FROM news_features WHERE setup_id = '{setup_id}'"
        ).fetchone()[0]
        logger.info(f"News features for {setup_id}: {news_count}")
        
        if news_count > 0:
            news_features = conn.execute(
                f"SELECT * FROM news_features WHERE setup_id = '{setup_id}'"
            ).fetchdf()
            logger.info(f"News features columns: {news_features.columns.tolist()}")
            logger.info(f"News features non-null columns: {news_features.count()[news_features.count() > 0].index.tolist()}")
        
        # Check userposts_features
        userposts_count = conn.execute(
            f"SELECT COUNT(*) FROM userposts_features WHERE setup_id = '{setup_id}'"
        ).fetchone()[0]
        logger.info(f"Userposts features for {setup_id}: {userposts_count}")
        
        if userposts_count > 0:
            userposts_features = conn.execute(
                f"SELECT * FROM userposts_features WHERE setup_id = '{setup_id}'"
            ).fetchdf()
            logger.info(f"Userposts features columns: {userposts_features.columns.tolist()}")
            logger.info(f"Userposts features non-null columns: {userposts_features.count()[userposts_features.count() > 0].index.tolist()}")
        
        # Check analyst_recommendations_features
        analyst_count = conn.execute(
            f"SELECT COUNT(*) FROM analyst_recommendations_features WHERE setup_id = '{setup_id}'"
        ).fetchone()[0]
        logger.info(f"Analyst recommendations features for {setup_id}: {analyst_count}")
        
        if analyst_count > 0:
            analyst_features = conn.execute(
                f"SELECT * FROM analyst_recommendations_features WHERE setup_id = '{setup_id}'"
            ).fetchdf()
            logger.info(f"Analyst features columns: {analyst_features.columns.tolist()}")
            logger.info(f"Analyst features non-null columns: {analyst_features.count()[analyst_features.count() > 0].index.tolist()}")
        
        # Check setups table
        setups_count = conn.execute(
            f"SELECT COUNT(*) FROM setups WHERE setup_id = '{setup_id}'"
        ).fetchone()[0]
        logger.info(f"Setups table for {setup_id}: {setups_count}")
        
        if setups_count > 0:
            setups = conn.execute(
                f"SELECT * FROM setups WHERE setup_id = '{setup_id}'"
            ).fetchdf()
            logger.info(f"Setups columns: {setups.columns.tolist()}")
            
        # Check fundamentals_features table
        fundamentals_count = conn.execute(
            f"SELECT COUNT(*) FROM fundamentals_features WHERE setup_id = '{setup_id}'"
        ).fetchone()[0]
        logger.info(f"Fundamentals features for {setup_id}: {fundamentals_count}")
        
        if fundamentals_count > 0:
            fundamentals_features = conn.execute(
                f"SELECT * FROM fundamentals_features WHERE setup_id = '{setup_id}'"
            ).fetchdf()
            logger.info(f"Fundamentals features columns: {fundamentals_features.columns.tolist()}")
        
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description='Check if a setup ID exists in feature tables')
    parser.add_argument('--setup-id', default='AFN_2023-11-20',
                      help='Setup ID to check')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                      help='Path to DuckDB database')
    
    args = parser.parse_args()
    
    check_features(args.setup_id, args.db_path)

if __name__ == "__main__":
    main() 