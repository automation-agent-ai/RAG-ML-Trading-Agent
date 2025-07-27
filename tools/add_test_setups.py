#!/usr/bin/env python3
"""
Add Test Setups - Utility to add new test setup IDs to the database

This tool helps create new setup IDs for testing the prediction pipeline:
1. Adds new setup IDs to the setups table
2. Optionally adds corresponding data to other tables (news, fundamentals, etc.)
3. Verifies the setup IDs are safe for prediction
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import duckdb
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple
import random

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.check_setup_embeddings import SetupEmbeddingsChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestSetupCreator:
    """Utility to add new test setup IDs to the database"""
    
    def __init__(self, db_path: str = "data/sentiment_system.duckdb"):
        """Initialize the creator with DuckDB database path"""
        self.db_path = Path(db_path)
        
        # Connect to DuckDB
        if not self.db_path.exists():
            raise FileNotFoundError(f"DuckDB database not found: {self.db_path}")
            
        self.conn = duckdb.connect(str(self.db_path))
        
        # Get existing setup IDs
        self.existing_setups = self._get_existing_setups()
        logger.info(f"Found {len(self.existing_setups)} existing setup IDs")
        
        # Get tickers
        self.tickers = self._get_tickers()
        logger.info(f"Found {len(self.tickers)} tickers")
    
    def _get_existing_setups(self) -> Set[str]:
        """Get existing setup IDs from the database"""
        try:
            result = self.conn.execute("SELECT setup_id FROM setups").fetchall()
            return {row[0] for row in result}
        except Exception as e:
            logger.error(f"Error getting existing setup IDs: {e}")
            return set()
    
    def _get_tickers(self) -> List[Dict[str, str]]:
        """Get tickers from the database"""
        try:
            # Check if company_name column exists
            columns_query = "SELECT column_name FROM information_schema.columns WHERE table_name = 'setups'"
            columns = [row[0] for row in self.conn.execute(columns_query).fetchall()]
            
            # Adjust query based on available columns
            if 'company_name' in columns:
                result = self.conn.execute("""
                    SELECT DISTINCT lse_ticker, yahoo_ticker, company_name 
                    FROM setups 
                    WHERE lse_ticker IS NOT NULL 
                      AND yahoo_ticker IS NOT NULL
                      AND company_name IS NOT NULL
                """).fetchall()
                
                return [
                    {
                        'lse_ticker': row[0],
                        'yahoo_ticker': row[1],
                        'company_name': row[2]
                    }
                    for row in result
                ]
            else:
                # Fallback query without company_name
                result = self.conn.execute("""
                    SELECT DISTINCT lse_ticker, yahoo_ticker
                    FROM setups 
                    WHERE lse_ticker IS NOT NULL 
                      AND yahoo_ticker IS NOT NULL
                """).fetchall()
                
                return [
                    {
                        'lse_ticker': row[0],
                        'yahoo_ticker': row[1],
                        'company_name': f"{row[0]} Company"  # Use ticker as company name
                    }
                    for row in result
                ]
                
        except Exception as e:
            logger.error(f"Error getting tickers: {e}")
            
            # Create some default tickers as fallback
            return [
                {
                    'lse_ticker': 'AAPL.L',
                    'yahoo_ticker': 'AAPL',
                    'company_name': 'Apple Inc'
                },
                {
                    'lse_ticker': 'MSFT.L',
                    'yahoo_ticker': 'MSFT',
                    'company_name': 'Microsoft Corp'
                },
                {
                    'lse_ticker': 'GOOGL.L',
                    'yahoo_ticker': 'GOOGL',
                    'company_name': 'Alphabet Inc'
                }
            ]
    
    def create_test_setups(self, count: int = 5, prefix: str = "TEST") -> List[str]:
        """
        Create new test setup IDs
        
        Args:
            count: Number of test setups to create
            prefix: Prefix for setup IDs
            
        Returns:
            List of created setup IDs
        """
        if not self.tickers:
            logger.error("No tickers found in the database")
            return []
        
        created_setups = []
        current_date = datetime.now()
        
        # Check if company_name column exists
        columns_query = "SELECT column_name FROM information_schema.columns WHERE table_name = 'setups'"
        columns = [row[0] for row in self.conn.execute(columns_query).fetchall()]
        has_company_name = 'company_name' in columns
        
        for i in range(count):
            # Generate unique setup ID
            ticker_info = random.choice(self.tickers)
            ticker = ticker_info['lse_ticker'].split('.')[0]  # Remove .L if present
            setup_date = current_date - timedelta(days=i)
            setup_id = f"{prefix}_{ticker}_{setup_date.strftime('%Y-%m-%d')}"
            
            # Skip if setup ID already exists
            if setup_id in self.existing_setups:
                logger.warning(f"Setup ID {setup_id} already exists, skipping")
                continue
            
            # Add to setups table
            try:
                if has_company_name:
                    self.conn.execute("""
                        INSERT INTO setups (setup_id, spike_timestamp, yahoo_ticker, lse_ticker, company_name)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        setup_id,
                        setup_date.isoformat(),
                        ticker_info['yahoo_ticker'],
                        ticker_info['lse_ticker'],
                        ticker_info['company_name']
                    ))
                else:
                    self.conn.execute("""
                        INSERT INTO setups (setup_id, spike_timestamp, yahoo_ticker, lse_ticker)
                        VALUES (?, ?, ?, ?)
                    """, (
                        setup_id,
                        setup_date.isoformat(),
                        ticker_info['yahoo_ticker'],
                        ticker_info['lse_ticker']
                    ))
                
                created_setups.append(setup_id)
                logger.info(f"Created setup ID: {setup_id}")
                
            except Exception as e:
                logger.error(f"Error creating setup ID {setup_id}: {e}")
        
        # Commit changes
        self.conn.commit()
        
        return created_setups
    
    def add_sample_data(self, setup_ids: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Add sample data for setup IDs
        
        Args:
            setup_ids: List of setup IDs to add data for
            
        Returns:
            Dictionary with counts of added data by domain and setup ID
        """
        results = {setup_id: {} for setup_id in setup_ids}
        
        for setup_id in setup_ids:
            # Get ticker information
            try:
                # Check if company_name column exists
                columns_query = "SELECT column_name FROM information_schema.columns WHERE table_name = 'setups'"
                columns = [row[0] for row in self.conn.execute(columns_query).fetchall()]
                
                if 'company_name' in columns:
                    ticker_info = self.conn.execute("""
                        SELECT lse_ticker, yahoo_ticker, company_name
                        FROM setups
                        WHERE setup_id = ?
                    """, (setup_id,)).fetchone()
                else:
                    ticker_info = self.conn.execute("""
                        SELECT lse_ticker, yahoo_ticker
                        FROM setups
                        WHERE setup_id = ?
                    """, (setup_id,)).fetchone()
                    
                    if ticker_info:
                        # Add a placeholder company name
                        ticker_info = (ticker_info[0], ticker_info[1], f"{ticker_info[0]} Company")
                
                if not ticker_info:
                    logger.error(f"Setup ID {setup_id} not found")
                    continue
                    
                lse_ticker, yahoo_ticker, company_name = ticker_info
                
                # Add sample news
                news_count = self._add_sample_news(setup_id, lse_ticker, yahoo_ticker)
                results[setup_id]['news'] = news_count
                
                # Add sample fundamentals
                fundamentals_count = self._add_sample_fundamentals(setup_id, yahoo_ticker)
                results[setup_id]['fundamentals'] = fundamentals_count
                
                # Add sample analyst recommendations
                analyst_count = self._add_sample_analyst_recommendations(setup_id, yahoo_ticker)
                results[setup_id]['analyst'] = analyst_count
                
                # Add sample user posts
                userposts_count = self._add_sample_userposts(setup_id, yahoo_ticker, company_name)
                results[setup_id]['userposts'] = userposts_count
                
            except Exception as e:
                logger.error(f"Error adding sample data for {setup_id}: {e}")
        
        # Commit changes
        self.conn.commit()
        
        return results
    
    def _add_sample_news(self, setup_id: str, lse_ticker: str, yahoo_ticker: str) -> int:
        """Add sample news for a setup ID"""
        try:
            # Check if news table exists
            self.conn.execute("CREATE TABLE IF NOT EXISTS rns_announcements (id INTEGER, setup_id VARCHAR, ticker VARCHAR, headline VARCHAR, text VARCHAR, rns_date DATE, rns_time TIME, url VARCHAR, scraped_at TIMESTAMP)")
            
            # Add 3 sample news items
            sample_news = [
                (
                    f"{setup_id}_NEWS_{i}",
                    setup_id,
                    lse_ticker,
                    f"Sample news headline {i} for {lse_ticker}",
                    f"This is sample news content {i} for testing the prediction pipeline with {lse_ticker}.",
                    datetime.now().date(),
                    datetime.now().time(),
                    f"https://example.com/news/{lse_ticker}/{i}",
                    datetime.now()
                )
                for i in range(1, 4)
            ]
            
            self.conn.executemany("""
                INSERT INTO rns_announcements (id, setup_id, ticker, headline, text, rns_date, rns_time, url, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, sample_news)
            
            return len(sample_news)
            
        except Exception as e:
            logger.error(f"Error adding sample news for {setup_id}: {e}")
            return 0
    
    def _add_sample_fundamentals(self, setup_id: str, ticker: str) -> int:
        """Add sample fundamentals for a setup ID"""
        try:
            # Check if fundamentals table exists
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS fundamentals (
                    id INTEGER, 
                    setup_id VARCHAR, 
                    ticker VARCHAR,
                    market_cap DOUBLE,
                    revenue DOUBLE,
                    net_income DOUBLE,
                    total_assets DOUBLE,
                    debt_to_equity DOUBLE,
                    current_ratio DOUBLE,
                    roa DOUBLE,
                    roe DOUBLE,
                    gross_margin_pct DOUBLE,
                    net_margin_pct DOUBLE,
                    revenue_growth_pct DOUBLE,
                    period_end DATE,
                    report_type VARCHAR
                )
            """)
            
            # Add sample fundamentals
            sample_fundamentals = [
                (
                    f"{setup_id}_FUND",
                    setup_id,
                    ticker,
                    random.uniform(1e9, 1e11),  # market_cap
                    random.uniform(1e8, 1e10),  # revenue
                    random.uniform(1e7, 1e9),   # net_income
                    random.uniform(1e9, 1e11),  # total_assets
                    random.uniform(0.1, 2.0),   # debt_to_equity
                    random.uniform(0.8, 3.0),   # current_ratio
                    random.uniform(0.01, 0.2),  # roa
                    random.uniform(0.05, 0.3),  # roe
                    random.uniform(20, 80),     # gross_margin_pct
                    random.uniform(5, 30),      # net_margin_pct
                    random.uniform(-10, 30),    # revenue_growth_pct
                    datetime.now().date(),      # period_end
                    random.choice(['annual', 'quarterly', 'interim'])  # report_type
                )
            ]
            
            self.conn.executemany("""
                INSERT INTO fundamentals (
                    id, setup_id, ticker, market_cap, revenue, net_income, total_assets,
                    debt_to_equity, current_ratio, roa, roe, gross_margin_pct,
                    net_margin_pct, revenue_growth_pct, period_end, report_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, sample_fundamentals)
            
            return len(sample_fundamentals)
            
        except Exception as e:
            logger.error(f"Error adding sample fundamentals for {setup_id}: {e}")
            return 0
    
    def _add_sample_analyst_recommendations(self, setup_id: str, ticker: str) -> int:
        """Add sample analyst recommendations for a setup ID"""
        try:
            # Check if analyst_recommendations table exists
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS analyst_recommendations (
                    id INTEGER, 
                    setup_id VARCHAR, 
                    ticker VARCHAR,
                    period VARCHAR,
                    strong_buy INTEGER,
                    buy INTEGER,
                    hold INTEGER,
                    sell INTEGER,
                    strong_sell INTEGER,
                    mean_rating DOUBLE
                )
            """)
            
            # Add sample analyst recommendations
            strong_buy = random.randint(0, 5)
            buy = random.randint(0, 8)
            hold = random.randint(0, 10)
            sell = random.randint(0, 3)
            strong_sell = random.randint(0, 2)
            
            total = strong_buy + buy + hold + sell + strong_sell
            if total > 0:
                mean_rating = (strong_buy*1 + buy*2 + hold*3 + sell*4 + strong_sell*5) / total
            else:
                mean_rating = 3.0
            
            sample_recommendations = [
                (
                    f"{setup_id}_ANALYST",
                    setup_id,
                    ticker,
                    "current",
                    strong_buy,
                    buy,
                    hold,
                    sell,
                    strong_sell,
                    mean_rating
                )
            ]
            
            self.conn.executemany("""
                INSERT INTO analyst_recommendations (
                    id, setup_id, ticker, period, strong_buy, buy, hold, sell, strong_sell, mean_rating
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, sample_recommendations)
            
            return len(sample_recommendations)
            
        except Exception as e:
            logger.error(f"Error adding sample analyst recommendations for {setup_id}: {e}")
            return 0
    
    def _add_sample_userposts(self, setup_id: str, ticker: str, company_name: str) -> int:
        """Add sample user posts for a setup ID"""
        try:
            # Check if user_posts table exists
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS user_posts (
                    post_id VARCHAR,
                    setup_id VARCHAR,
                    ticker VARCHAR,
                    user_handle VARCHAR,
                    post_content TEXT,
                    post_date TIMESTAMP,
                    post_url VARCHAR,
                    scraping_timestamp TIMESTAMP
                )
            """)
            
            # Add 5 sample user posts
            sample_posts = [
                (
                    f"{setup_id}_POST_{i}",
                    setup_id,
                    ticker,
                    f"user{random.randint(100, 999)}",
                    f"This is a sample post {i} about {company_name} ({ticker}). " + 
                    random.choice([
                        "I think this stock is going to do well.",
                        "Not sure about the recent news, seems concerning.",
                        "Earnings were better than expected!",
                        "The market is undervaluing this company.",
                        "Recent analyst upgrades look promising."
                    ]),
                    datetime.now() - timedelta(hours=random.randint(1, 48)),
                    f"https://example.com/posts/{ticker}/{i}",
                    datetime.now()
                )
                for i in range(1, 6)
            ]
            
            self.conn.executemany("""
                INSERT INTO user_posts (
                    post_id, setup_id, ticker, user_handle, post_content,
                    post_date, post_url, scraping_timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, sample_posts)
            
            return len(sample_posts)
            
        except Exception as e:
            logger.error(f"Error adding sample user posts for {setup_id}: {e}")
            return 0
    
    def verify_setups(self, setup_ids: List[str], lancedb_dir: str = "lancedb_store") -> None:
        """Verify that setup IDs are safe for prediction"""
        try:
            checker = SetupEmbeddingsChecker(lancedb_dir=lancedb_dir)
            results = checker.check_setup_ids(setup_ids)
            checker.display_results(results)
        except Exception as e:
            logger.error(f"Error verifying setup IDs: {e}")
    
    def cleanup(self) -> None:
        """Close database connection"""
        try:
            self.conn.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Add test setup IDs to the database')
    parser.add_argument('--count', type=int, default=5,
                      help='Number of test setups to create')
    parser.add_argument('--prefix', default='TEST',
                      help='Prefix for setup IDs')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                      help='Path to DuckDB database')
    parser.add_argument('--lancedb-dir', default='lancedb_store',
                      help='Path to LanceDB directory')
    parser.add_argument('--add-data', action='store_true',
                      help='Add sample data for the created setup IDs')
    parser.add_argument('--verify', action='store_true',
                      help='Verify that setup IDs are safe for prediction')
    
    args = parser.parse_args()
    
    try:
        creator = TestSetupCreator(db_path=args.db_path)
        
        # Create test setups
        created_setups = creator.create_test_setups(count=args.count, prefix=args.prefix)
        
        if not created_setups:
            logger.error("No setup IDs created")
            return 1
        
        # Add sample data if requested
        if args.add_data:
            logger.info("Adding sample data for created setup IDs...")
            results = creator.add_sample_data(created_setups)
            
            # Print summary
            print("\n" + "="*80)
            print("SAMPLE DATA SUMMARY")
            print("="*80)
            
            for setup_id, domains in results.items():
                print(f"\nSetup ID: {setup_id}")
                print("-" * 50)
                
                for domain, count in domains.items():
                    print(f"  {domain.upper()}: {count} records")
            
            print("\n" + "="*80)
        
        # Verify setups if requested
        if args.verify:
            logger.info("Verifying setup IDs...")
            creator.verify_setups(created_setups, lancedb_dir=args.lancedb_dir)
        
        # Clean up
        creator.cleanup()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 