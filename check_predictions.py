#!/usr/bin/env python3
"""
Check predictions for test setups.
"""

import duckdb
import pandas as pd

def main():
    # Connect to the database
    conn = duckdb.connect('data/sentiment_system.duckdb')
    
    # Get the list of test setups
    test_setups = ['BGO_2025-01-20', 'BLND_2025-05-23', 'BNC_2025-03-20']
    
    # Query predictions for test setups
    query = f"""
        SELECT setup_id, domain, predicted_outperformance, confidence 
        FROM similarity_predictions 
        WHERE setup_id IN ({', '.join([f"'{setup}'" for setup in test_setups])})
        ORDER BY setup_id, domain
    """
    
    predictions = conn.execute(query).df()
    
    print("Predictions for test setups:")
    print(predictions)
    
    # Count predictions by domain
    print("\nCount of predictions by domain:")
    print(conn.execute("SELECT domain, COUNT(*) FROM similarity_predictions GROUP BY domain").df())
    
    # Check if we have predictions for all domains
    print("\nChecking if we have predictions for all domains:")
    domains = ['news', 'fundamentals', 'analyst_recommendations', 'userposts', 'ensemble']
    for domain in domains:
        count = conn.execute(f"SELECT COUNT(*) FROM similarity_predictions WHERE domain = '{domain}'").fetchone()[0]
        print(f"Domain {domain}: {count} predictions")
    
    # Check feature tables to see if they have data
    print("\nChecking feature tables:")
    feature_tables = ['news_features', 'fundamentals_features', 'analyst_recommendations_features', 'userposts_features']
    for table in feature_tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"Table {table}: {count} rows")
            if count > 0:
                # Check a sample row
                sample = conn.execute(f"SELECT * FROM {table} LIMIT 1").df()
                print(f"Sample columns: {', '.join(sample.columns.tolist())}")
        except Exception as e:
            print(f"Error checking {table}: {e}")
    
    conn.close()

if __name__ == "__main__":
    main() 