#!/usr/bin/env python3
"""
Prediction Setup List Creator

This script creates a list of setup IDs to use for prediction:
1. Finds setups with complete data across all domains
2. Randomly selects a specified number of setups
3. Saves the list to a file

Usage:
    python create_prediction_list.py --count 100 --output data/prediction_setups.txt
"""

import os
import sys
import argparse
import random
import logging
import duckdb
from pathlib import Path
from typing import List, Set
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_complete_setups(db_path: str) -> List[str]:
    """Find setups with complete data across all domains"""
    try:
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
        
        logger.info(f"Found {len(setup_ids)} setups with complete data")
        return setup_ids
    
    except Exception as e:
        logger.error(f"Error finding complete setups: {e}")
        return []

def select_prediction_setups(
    all_setups: List[str], 
    count: int = 100, 
    random_seed: int = 42
) -> List[str]:
    """Randomly select setups for prediction"""
    random.seed(random_seed)
    
    # Ensure we don't try to select more setups than available
    count = min(count, len(all_setups))
    
    # Randomly select setups
    prediction_setups = random.sample(all_setups, count)
    
    logger.info(f"Selected {len(prediction_setups)} setups for prediction")
    return prediction_setups

def save_setup_list(setup_ids: List[str], output_file: str) -> None:
    """Save setup IDs to a file"""
    try:
        # Create directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        # Write setup IDs to file
        with open(output_file, 'w') as f:
            for setup_id in sorted(setup_ids):
                f.write(f"{setup_id}\n")
        
        logger.info(f"Saved {len(setup_ids)} setup IDs to {output_file}")
    
    except Exception as e:
        logger.error(f"Error saving setup list: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Create a list of setups for prediction")
    parser.add_argument("--count", type=int, default=100,
                      help="Number of setups to select for prediction")
    parser.add_argument("--db-path", default="data/sentiment_system.duckdb",
                      help="Path to DuckDB database")
    parser.add_argument("--output", default=None,
                      help="Output file for prediction setup IDs")
    parser.add_argument("--random-seed", type=int, default=42,
                      help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Generate default output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"data/prediction_setups_{timestamp}.txt"
    
    # Find complete setups
    all_setups = find_complete_setups(args.db_path)
    
    if not all_setups:
        logger.error("No complete setups found")
        return
    
    # Select prediction setups
    prediction_setups = select_prediction_setups(
        all_setups, 
        count=args.count, 
        random_seed=args.random_seed
    )
    
    # Save setup list
    save_setup_list(prediction_setups, args.output)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("PREDICTION SETUP SELECTION SUMMARY")
    logger.info("="*50)
    logger.info(f"Total complete setups: {len(all_setups)}")
    logger.info(f"Selected for prediction: {len(prediction_setups)}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Random seed: {args.random_seed}")

if __name__ == "__main__":
    main() 