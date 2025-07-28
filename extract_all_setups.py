#!/usr/bin/env python3
"""
Extract All Setup IDs From DuckDB

This script extracts all setup IDs from the database and saves them to a file.
It can also filter setups based on various criteria.

Usage:
    python extract_all_setups.py --output data/all_setups.txt
    python extract_all_setups.py --output data/training_setups.txt --limit 1000
    python extract_all_setups.py --output data/prediction_setups.txt --limit 50 --random
"""

import os
import sys
import argparse
import logging
import duckdb
import random
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_setup_ids(
    db_path: str = "data/sentiment_system.duckdb",
    limit: int = None,
    random_sample: bool = False,
    require_labels: bool = False,
    require_fundamentals: bool = False,
    require_news: bool = False,
    require_userposts: bool = False,
    require_analyst: bool = False
) -> list:
    """
    Extract setup IDs from the database
    
    Args:
        db_path: Path to DuckDB database
        limit: Maximum number of setup IDs to extract
        random_sample: Whether to randomly sample setup IDs
        require_labels: Whether to require setups to have labels
        require_fundamentals: Whether to require setups to have fundamentals data
        require_news: Whether to require setups to have news features
        require_userposts: Whether to require setups to have userposts features
        require_analyst: Whether to require setups to have analyst features
        
    Returns:
        List of setup IDs
    """
    logger.info("Extracting setup IDs from database...")
    
    conn = duckdb.connect(db_path)
    
    # Build query based on requirements
    query_parts = ["SELECT DISTINCT s.setup_id FROM setups s"]
    where_parts = []
    
    # Add joins based on requirements
    if require_labels:
        query_parts.append("JOIN daily_labels l ON s.setup_id = l.setup_id")
        where_parts.append("l.day_number <= 10")
        
    if require_fundamentals:
        query_parts.append("""
        JOIN (
            SELECT DISTINCT s_inner.setup_id
            FROM setups s_inner
            JOIN fundamentals f ON s_inner.lse_ticker || '.L' = f.ticker AND f.date <= s_inner.spike_timestamp
        ) f_exists ON s.setup_id = f_exists.setup_id
        """)
        
    if require_news:
        query_parts.append("JOIN news_features nf ON s.setup_id = nf.setup_id")
        
    if require_userposts:
        query_parts.append("JOIN userposts_features uf ON s.setup_id = uf.setup_id")
        
    if require_analyst:
        query_parts.append("JOIN analyst_recommendations_features af ON s.setup_id = af.setup_id")
    
    # Add WHERE clause if needed
    if where_parts:
        query_parts.append("WHERE " + " AND ".join(where_parts))
    
    # Add GROUP BY and HAVING if requiring labels
    if require_labels:
        query_parts.append("GROUP BY s.setup_id")
        query_parts.append("HAVING COUNT(DISTINCT l.day_number) >= 5")
    
    # Add ORDER BY and LIMIT
    if random_sample:
        query_parts.append("ORDER BY RANDOM()")
    else:
        query_parts.append("ORDER BY s.setup_id")
        
    if limit:
        query_parts.append(f"LIMIT {limit}")
    
    # Execute query
    query = " ".join(query_parts)
    logger.info(f"Query: {query}")
    
    setup_ids = conn.execute(query).df()['setup_id'].tolist()
    conn.close()
    
    logger.info(f"Extracted {len(setup_ids)} setup IDs")
    return setup_ids

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Extract setup IDs from database')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--output', default='data/all_setups.txt',
                       help='Output file path')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of setup IDs to extract')
    parser.add_argument('--random', action='store_true',
                       help='Randomly sample setup IDs')
    parser.add_argument('--require-labels', action='store_true',
                       help='Require setups to have labels')
    parser.add_argument('--require-fundamentals', action='store_true',
                       help='Require setups to have fundamentals data')
    parser.add_argument('--require-news', action='store_true',
                       help='Require setups to have news features')
    parser.add_argument('--require-userposts', action='store_true',
                       help='Require setups to have userposts features')
    parser.add_argument('--require-analyst', action='store_true',
                       help='Require setups to have analyst features')
    parser.add_argument('--require-all', action='store_true',
                       help='Require setups to have all data types')
    
    args = parser.parse_args()
    
    # If require-all is specified, set all individual requirements to True
    if args.require_all:
        args.require_labels = True
        args.require_fundamentals = True
        args.require_news = True
        args.require_userposts = True
        args.require_analyst = True
    
    # Extract setup IDs
    setup_ids = extract_setup_ids(
        db_path=args.db_path,
        limit=args.limit,
        random_sample=args.random,
        require_labels=args.require_labels,
        require_fundamentals=args.require_fundamentals,
        require_news=args.require_news,
        require_userposts=args.require_userposts,
        require_analyst=args.require_analyst
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save to file
    with open(args.output, 'w') as f:
        f.write('\n'.join(setup_ids))
    
    logger.info(f"Saved {len(setup_ids)} setup IDs to {args.output}")

if __name__ == '__main__':
    main() 