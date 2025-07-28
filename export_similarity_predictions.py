#!/usr/bin/env python3
"""
Export Similarity Predictions

This script exports the similarity_predictions table from DuckDB to a CSV file.
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

def export_similarity_predictions(db_path="data/sentiment_system.duckdb", output_file="similarity_predictions.csv"):
    """
    Export the similarity_predictions table to a CSV file
    
    Args:
        db_path: Path to the DuckDB database
        output_file: Path to the output CSV file
    """
    try:
        # Connect to DuckDB
        conn = duckdb.connect(db_path)
        
        # Check if the table exists
        result = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='similarity_predictions'").fetchone()
        if not result:
            logger.error(f"Table 'similarity_predictions' not found in {db_path}")
            return False
        
        # Get the total number of rows
        count = conn.execute("SELECT COUNT(*) FROM similarity_predictions").fetchone()[0]
        logger.info(f"Found {count} rows in similarity_predictions table")
        
        # Get the domain distribution
        domain_counts = conn.execute("SELECT domain, COUNT(*) FROM similarity_predictions GROUP BY domain ORDER BY COUNT(*) DESC").fetchdf()
        logger.info("Domain distribution:")
        for _, row in domain_counts.iterrows():
            logger.info(f"  {row['domain']}: {row['count_star()']} records")
        
        # Export all rows
        logger.info(f"Exporting all rows to {output_file}...")
        df = conn.execute("SELECT * FROM similarity_predictions ORDER BY setup_id, domain").fetchdf()
        df.to_csv(output_file, index=False)
        logger.info(f"Successfully exported {len(df)} rows to {output_file}")
        
        # Also create a pivot table for easier viewing
        logger.info("Creating pivot table for easier viewing...")
        
        # Get unique setup_ids
        setup_ids = df['setup_id'].unique()
        logger.info(f"Found {len(setup_ids)} unique setup_ids")
        
        # Create a pivot table with setup_id as rows, domain as columns, and predicted_outperformance as values
        pivot_df = pd.pivot_table(
            df, 
            values='predicted_outperformance', 
            index='setup_id', 
            columns='domain',
            aggfunc='first'  # Take the first value if there are duplicates
        )
        
        # Add confidence columns
        for domain in df['domain'].unique():
            domain_df = df[df['domain'] == domain]
            confidence_values = domain_df.set_index('setup_id')['confidence']
            pivot_df[f"{domain}_confidence"] = confidence_values
        
        # Reorder columns as requested
        desired_order = [
            'analyst_recommendations', 'analyst_recommendations_confidence',
            'fundamentals', 'fundamentals_confidence',
            'news', 'news_confidence',
            'userposts', 'userposts_confidence',
            'ensemble', 'ensemble_confidence'
        ]
        
        # Only include columns that exist in the DataFrame
        ordered_columns = [col for col in desired_order if col in pivot_df.columns]
        
        # Reorder columns
        pivot_df = pivot_df[ordered_columns]
        
        # Export the pivot table
        pivot_output = output_file.replace('.csv', '_pivot.csv')
        pivot_df.to_csv(pivot_output)
        logger.info(f"Successfully exported pivot table to {pivot_output}")
        
        # Show a sample of the pivot table
        logger.info("Sample of pivot table:")
        logger.info(pivot_df.head())
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error exporting similarity_predictions: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Export similarity_predictions table to CSV")
    parser.add_argument("--db", default="data/sentiment_system.duckdb",
                        help="Path to DuckDB database (default: data/sentiment_system.duckdb)")
    parser.add_argument("--output", default="similarity_predictions.csv",
                        help="Path to output CSV file (default: similarity_predictions.csv)")
    
    args = parser.parse_args()
    
    print(f"Exporting similarity_predictions from {args.db} to {args.output}...")
    
    success = export_similarity_predictions(args.db, args.output)
    
    if success:
        print(f"\n✅ Successfully exported similarity_predictions to {args.output}")
        print(f"  Also created pivot table at {args.output.replace('.csv', '_pivot.csv')}")
    else:
        print("\n❌ Failed to export similarity_predictions")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 