#!/usr/bin/env python3
"""
Check Fundamentals Data
"""

import duckdb
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Check fundamentals data"""
    # Connect to the database
    conn = duckdb.connect('data/sentiment_system.duckdb')
    
    # Get a sample of setup IDs
    setup_sample = conn.execute('''
        SELECT setup_id, lse_ticker, spike_timestamp 
        FROM setups 
        WHERE setup_id IN (
            SELECT setup_id FROM setups LIMIT 10
        )
    ''').df()
    
    print("Setup sample:")
    print(setup_sample)
    
    # Check fundamentals data for these tickers
    print("\nFundamentals data for these tickers:")
    for idx, row in setup_sample.iterrows():
        ticker = row['lse_ticker']
        setup_id = row['setup_id']
        spike_timestamp = row['spike_timestamp']
        
        # Count all fundamentals for this ticker
        count = conn.execute(f"SELECT COUNT(*) FROM fundamentals WHERE ticker = '{ticker}'").fetchone()[0]
        print(f"Ticker {ticker} (setup_id {setup_id}): {count} fundamentals records")
        
        # Check if there are any fundamentals records before the spike timestamp
        before_count = conn.execute(f"SELECT COUNT(*) FROM fundamentals WHERE ticker = '{ticker}' AND date <= '{spike_timestamp}'").fetchone()[0]
        print(f"  - Records before spike timestamp: {before_count}")
        
        # If there are records, show a sample
        if before_count > 0:
            sample = conn.execute(f"""
                SELECT date, total_revenue, net_income, total_assets, total_equity
                FROM fundamentals 
                WHERE ticker = '{ticker}' AND date <= '{spike_timestamp}'
                ORDER BY date DESC
                LIMIT 1
            """).df()
            print("  - Latest record:")
            print(sample)
    
    # Check if the JOIN in our query works
    print("\nTesting JOIN between setups and fundamentals:")
    join_test = conn.execute('''
        SELECT s.setup_id, s.lse_ticker, f.date, f.total_revenue
        FROM setups s
        JOIN fundamentals f ON s.lse_ticker = f.ticker AND f.date <= s.spike_timestamp
        LIMIT 10
    ''').df()
    print(f"Found {len(join_test)} rows with JOIN")
    if len(join_test) > 0:
        print(join_test)
    
    # Check financial ratios
    print("\nTesting JOIN between setups and financial_ratios:")
    ratios_test = conn.execute('''
        SELECT s.setup_id, s.lse_ticker, fr.period_end, fr.current_ratio, fr.debt_to_equity
        FROM setups s
        JOIN financial_ratios fr ON s.lse_ticker = fr.ticker AND fr.period_end <= s.spike_timestamp
        LIMIT 10
    ''').df()
    print(f"Found {len(ratios_test)} rows with JOIN")
    if len(ratios_test) > 0:
        print(ratios_test)
    
    # Close the connection
    conn.close()

if __name__ == '__main__':
    main() 