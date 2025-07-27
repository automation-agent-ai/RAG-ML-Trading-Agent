#!/usr/bin/env python3
"""
Extract all setup IDs from the database and save to a file
"""

import duckdb
import os
from pathlib import Path

def main():
    # Connect to the database
    conn = duckdb.connect('data/sentiment_system.duckdb')
    
    # Get all setup IDs
    setup_ids = conn.execute('SELECT DISTINCT setup_id FROM setups LIMIT 5000').df()['setup_id'].tolist()
    
    # Create the data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to file
    with open('data/all_setups.txt', 'w') as f:
        f.write('\n'.join(setup_ids))
    
    print(f'Saved {len(setup_ids)} setup IDs to data/all_setups.txt')

if __name__ == '__main__':
    main() 