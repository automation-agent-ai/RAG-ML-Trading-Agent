#!/usr/bin/env python3
"""
Check Specific Setup

This script checks predictions for a specific setup ID.
"""

import os
import sys
import logging
import duckdb
import pandas as pd
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_setup(setup_id, db_path="data/sentiment_system.duckdb"):
    """
    Check predictions for a specific setup ID
    
    Args:
        setup_id: The setup ID to check
        db_path: Path to the DuckDB database
    """
    try:
        # Connect to DuckDB
        conn = duckdb.connect(db_path)
        
        # Check if the setup exists in the similarity_predictions table
        query = f"SELECT * FROM similarity_predictions WHERE setup_id = '{setup_id}'"
        df = conn.execute(query).fetchdf()
        
        if len(df) == 0:
            print(f"No predictions found for setup_id: {setup_id}")
            return
        
        print(f"Found {len(df)} predictions for setup_id: {setup_id}")
        print(df)
        
        # Check if the setup exists in the news_features table
        query = f"SELECT * FROM news_features WHERE setup_id = '{setup_id}'"
        df = conn.execute(query).fetchdf()
        
        if len(df) == 0:
            print(f"No news features found for setup_id: {setup_id}")
        else:
            print(f"Found news features for setup_id: {setup_id}")
            # Check for Director Dealings
            governance_cols = [col for col in df.columns if 'governance' in col]
            if governance_cols:
                print("Governance columns:")
                for col in governance_cols:
                    print(f"  {col}: {df[col].iloc[0]}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error checking setup: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Check predictions for a specific setup ID")
    parser.add_argument("setup_id", help="Setup ID to check")
    parser.add_argument("--db", default="data/sentiment_system.duckdb",
                        help="Path to DuckDB database (default: data/sentiment_system.duckdb)")
    
    args = parser.parse_args()
    
    print(f"Checking predictions for setup_id: {args.setup_id}")
    check_setup(args.setup_id, args.db)

if __name__ == "__main__":
    main() 