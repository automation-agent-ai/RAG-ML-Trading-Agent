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
    
    # Check ticker formats in each table
    print("Setup tickers sample:")
    setup_tickers = conn.execute('SELECT DISTINCT lse_ticker FROM setups LIMIT 20').df()
    print(setup_tickers)
    
    print("\nFundamentals tickers sample:")
    fund_tickers = conn.execute('SELECT DISTINCT ticker FROM fundamentals LIMIT 20').df()
    print(fund_tickers)
    
    print("\nFinancial ratios tickers sample:")
    ratio_tickers = conn.execute('SELECT DISTINCT ticker FROM financial_ratios LIMIT 20').df()
    print(ratio_tickers)
    
    # Check if there's any overlap between setup tickers and fundamentals tickers
    setup_ticker_list = setup_tickers['lse_ticker'].tolist()
    fund_ticker_list = fund_tickers['ticker'].tolist()
    
    print("\nChecking for ticker matches:")
    matches = [ticker for ticker in setup_ticker_list if ticker in fund_ticker_list]
    print(f"Found {len(matches)} matches between setup tickers and fundamentals tickers")
    if matches:
        print("Matches:", matches)
    
    # Check if there's a pattern difference (e.g., .L suffix)
    print("\nChecking for pattern differences:")
    for setup_ticker in setup_ticker_list[:5]:  # Check first 5
        print(f"Setup ticker: {setup_ticker}")
        # Try with .L suffix
        with_l = f"{setup_ticker}.L"
        count = conn.execute(f"SELECT COUNT(*) FROM fundamentals WHERE ticker = '{with_l}'").fetchone()[0]
        print(f"  - With .L suffix ({with_l}): {count} records")
        
        # Try without .L suffix
        without_l = setup_ticker.replace('.L', '')
        count = conn.execute(f"SELECT COUNT(*) FROM fundamentals WHERE ticker = '{without_l}'").fetchone()[0]
        print(f"  - Without .L suffix ({without_l}): {count} records")
    
    # Close the connection
    conn.close()

if __name__ == '__main__':
    main() 