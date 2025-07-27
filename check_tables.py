#!/usr/bin/env python3
"""
Check tables in the database
"""

import duckdb
import pandas as pd

def main():
    # Connect to the database
    conn = duckdb.connect('data/sentiment_system.duckdb')
    
    # Get all tables
    tables = conn.execute("SHOW TABLES").df()
    
    # Print tables with 'feature' in the name
    feature_tables = tables[tables['name'].str.contains('feature', case=False)]
    print("Tables with 'feature' in the name:")
    print(feature_tables['name'].tolist())
    
    # Check if financial_features table exists
    if 'financial_features' in tables['name'].tolist():
        print("\nFinancial features sample:")
        print(conn.execute("SELECT * FROM financial_features LIMIT 5").df())
    
    # Close the connection
    conn.close()

if __name__ == '__main__':
    main() 