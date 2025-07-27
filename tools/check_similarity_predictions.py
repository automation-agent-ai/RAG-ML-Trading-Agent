#!/usr/bin/env python3
"""
check_similarity_predictions.py - Utility to check similarity predictions in the DuckDB database

This script checks if the similarity_predictions table exists in the DuckDB database
and displays its contents.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_similarity_predictions(db_path: str = "data/sentiment_system.duckdb", setup_id: str = None):
    """
    Check if the similarity_predictions table exists and display its contents
    
    Args:
        db_path: Path to DuckDB database
        setup_id: Optional setup_id to filter by
    """
    try:
        import duckdb
        
        # Connect to DuckDB
        conn = duckdb.connect(db_path)
        
        # Check if the similarity_predictions table exists
        table_exists = conn.execute("""
            SELECT count(*) FROM information_schema.tables 
            WHERE table_name = 'similarity_predictions'
        """).fetchone()[0]
        
        if not table_exists:
            logger.error("The similarity_predictions table does not exist in the database.")
            return False
        
        # Get the predictions
        if setup_id:
            query = f"""
                SELECT * FROM similarity_predictions
                WHERE setup_id = '{setup_id}'
                ORDER BY domain
            """
            logger.info(f"Checking similarity predictions for setup_id: {setup_id}")
        else:
            query = """
                SELECT * FROM similarity_predictions
                ORDER BY setup_id, domain
            """
            logger.info("Checking all similarity predictions")
        
        predictions_df = conn.execute(query).fetchdf()
        
        if len(predictions_df) == 0:
            logger.warning("No similarity predictions found.")
            return True
        
        # Display the predictions
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\nSimilarity Predictions:")
        print("=" * 80)
        print(predictions_df)
        print("=" * 80)
        
        # Calculate summary statistics
        if len(predictions_df) > 0:
            print("\nSummary Statistics:")
            print("=" * 80)
            
            # Group by domain
            domain_stats = predictions_df.groupby('domain').agg({
                'predicted_outperformance': ['mean', 'std', 'min', 'max'],
                'confidence': 'mean',
                'positive_ratio': 'mean',
                'negative_ratio': 'mean',
                'setup_id': 'count'
            })
            
            print(domain_stats)
            print("=" * 80)
            
            # Overall stats
            overall_mean = predictions_df['predicted_outperformance'].mean()
            overall_std = predictions_df['predicted_outperformance'].std()
            overall_confidence = predictions_df['confidence'].mean()
            
            print(f"\nOverall Mean Predicted Outperformance: {overall_mean:.4f}")
            print(f"Overall Std Dev of Predictions: {overall_std:.4f}")
            print(f"Overall Mean Confidence: {overall_confidence:.4f}")
            print("=" * 80)
        
        return True
    except Exception as e:
        logger.error(f"Error checking similarity predictions: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Check similarity predictions in DuckDB')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--setup-id', help='Setup ID to filter by')
    
    args = parser.parse_args()
    
    check_similarity_predictions(args.db_path, args.setup_id)

if __name__ == "__main__":
    main() 