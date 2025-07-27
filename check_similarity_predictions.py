#!/usr/bin/env python3
"""
Check if similarity predictions were stored in the database
"""

import duckdb
import pandas as pd

def main():
    # Connect to DuckDB
    conn = duckdb.connect('data/sentiment_system.duckdb')
    
    # Check if similarity_predictions table exists
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [t[0] for t in tables]
    
    print("Available tables:")
    for table in table_names:
        print(f"- {table}")
    
    # Check if similarity_predictions table exists
    if 'similarity_predictions' in table_names:
        print("\nSimilarity predictions table exists!")
        
        # Get column names
        columns = conn.execute("PRAGMA table_info(similarity_predictions)").fetchall()
        column_names = [c[1] for c in columns]
        
        print("\nColumns:")
        for column in column_names:
            print(f"- {column}")
        
        # Get data
        data = conn.execute("SELECT * FROM similarity_predictions").fetchall()
        
        print(f"\nFound {len(data)} prediction records:")
        for row in data:
            print(row)
    else:
        print("\nSimilarity predictions table does not exist")
    
    # Close connection
    conn.close()

if __name__ == "__main__":
    main() 