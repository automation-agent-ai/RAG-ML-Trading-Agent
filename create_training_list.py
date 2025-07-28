#!/usr/bin/env python3
"""
Create Training Setup List

This script creates a list of setup IDs for training by extracting all setups from the database
and then excluding the prediction setups.

Usage:
    python create_training_list.py --prediction-file data/prediction_setups.txt --output data/training_setups.txt
"""

import os
import sys
import argparse
import logging
import duckdb
from pathlib import Path
from typing import List, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_all_setups(db_path: str) -> Set[str]:
    """
    Get all setup IDs from the database
    
    Args:
        db_path: Path to DuckDB database
        
    Returns:
        Set of setup IDs
    """
    conn = None
    try:
        conn = duckdb.connect(db_path)
        
        # Query all setup IDs from daily_labels table
        query = "SELECT DISTINCT setup_id FROM daily_labels"
        result = conn.execute(query).fetchall()
        
        # Extract setup IDs
        setup_ids = {row[0] for row in result}
        
        logger.info(f"Found {len(setup_ids)} total setup IDs in the database")
        return setup_ids
    
    finally:
        if conn:
            conn.close()

def load_prediction_setups(file_path: str) -> Set[str]:
    """
    Load prediction setup IDs from file
    
    Args:
        file_path: Path to prediction setups file
        
    Returns:
        Set of prediction setup IDs
    """
    with open(file_path, 'r') as f:
        prediction_setups = {line.strip() for line in f if line.strip()}
    
    logger.info(f"Loaded {len(prediction_setups)} prediction setup IDs from {file_path}")
    return prediction_setups

def create_training_setups(all_setups: Set[str], prediction_setups: Set[str]) -> Set[str]:
    """
    Create training setup IDs by excluding prediction setups
    
    Args:
        all_setups: Set of all setup IDs
        prediction_setups: Set of prediction setup IDs
        
    Returns:
        Set of training setup IDs
    """
    training_setups = all_setups - prediction_setups
    logger.info(f"Created {len(training_setups)} training setup IDs")
    return training_setups

def save_training_setups(training_setups: Set[str], output_file: str) -> None:
    """
    Save training setup IDs to file
    
    Args:
        training_setups: Set of training setup IDs
        output_file: Path to output file
    """
    # Create directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Sort setup IDs for consistency
    sorted_setups = sorted(training_setups)
    
    # Save to file
    with open(output_file, 'w') as f:
        for setup_id in sorted_setups:
            f.write(f"{setup_id}\n")
    
    logger.info(f"Saved {len(training_setups)} training setup IDs to {output_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create training setup list')
    parser.add_argument('--prediction-file', default='data/prediction_setups.txt',
                       help='Path to prediction setups file')
    parser.add_argument('--output', default='data/training_setups.txt',
                       help='Path to output training setups file')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    
    args = parser.parse_args()
    
    # Get all setups from database
    all_setups = get_all_setups(args.db_path)
    
    # Load prediction setups
    prediction_setups = load_prediction_setups(args.prediction_file)
    
    # Create training setups
    training_setups = create_training_setups(all_setups, prediction_setups)
    
    # Save training setups
    save_training_setups(training_setups, args.output)
    
    logger.info("Training setup list created successfully")

if __name__ == '__main__':
    main() 