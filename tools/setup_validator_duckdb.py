#!/usr/bin/env python3
"""
setup_validator_duckdb.py - DuckDB-based Setup Validation Utility

Provides validation functions to ensure the RAG pipeline only uses confirmed setups
from the DuckDB database and proper date filtering for historical data.
"""

import duckdb
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SetupValidatorDuckDB:
    """
    DuckDB-based setup validation for the RAG pipeline
    
    Ensures that only confirmed setups from the setups table are used
    and provides date validation for historical data filtering.
    """
    
    def __init__(self, db_path: str = "../data/sentiment_system.duckdb"):
        """
        Initialize the setup validator
        
        Args:
            db_path: Path to the DuckDB database file
        """
        self.db_path = Path(db_path)
        self.conn = None
        self.confirmed_setup_ids = set()
        self.setup_dates = {}
        self.setup_tickers = {}
        
        self.connect_and_load()
    
    def connect_and_load(self) -> None:
        """Connect to DuckDB and load confirmed setups"""
        if not self.db_path.exists():
            raise FileNotFoundError(f"DuckDB file not found at {self.db_path}")
        
        try:
            self.conn = duckdb.connect(str(self.db_path))
            self.load_confirmed_setups()
        except Exception as e:
            logger.error(f"Error connecting to DuckDB: {e}")
            raise
    
    def load_confirmed_setups(self) -> None:
        """Load confirmed setups from the setups table"""
        try:
            # Query setups table
            query = """
            SELECT setup_id, spike_timestamp, lse_ticker, yahoo_ticker
            FROM setups
            WHERE setup_id IS NOT NULL
            ORDER BY setup_id
            """
            
            setups_df = self.conn.execute(query).fetchdf()
            logger.info(f"Loaded {len(setups_df)} confirmed setups from DuckDB")
            
            # Extract setup IDs, dates, and tickers
            for _, row in setups_df.iterrows():
                setup_id = row.get('setup_id', '')
                spike_timestamp = row.get('spike_timestamp', '')
                lse_ticker = row.get('lse_ticker', '')
                yahoo_ticker = row.get('yahoo_ticker', '')
                
                if setup_id:
                    self.confirmed_setup_ids.add(setup_id)
                    self.setup_tickers[setup_id] = {
                        'lse_ticker': lse_ticker,
                        'yahoo_ticker': yahoo_ticker
                    }
                    
                    # Parse date
                    try:
                        if pd.notna(spike_timestamp):
                            if isinstance(spike_timestamp, str):
                                # Try different date formats
                                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                                    try:
                                        parsed_date = datetime.strptime(spike_timestamp, fmt)
                                        self.setup_dates[setup_id] = parsed_date
                                        break
                                    except ValueError:
                                        continue
                            else:
                                # Already a datetime object
                                self.setup_dates[setup_id] = spike_timestamp
                    except Exception as e:
                        logger.warning(f"Could not parse date for setup {setup_id}: {spike_timestamp}")
                        continue
            
            logger.info(f"Confirmed setup IDs: {len(self.confirmed_setup_ids)}")
            logger.info(f"Setup dates parsed: {len(self.setup_dates)}")
            logger.info(f"Sample setups: {list(sorted(self.confirmed_setup_ids))[:5]}")
            
        except Exception as e:
            logger.error(f"Error loading confirmed setups: {e}")
            raise
    
    def is_confirmed_setup(self, setup_id: str) -> bool:
        """
        Check if a setup_id is in the confirmed setups
        
        Args:
            setup_id: The setup ID to validate
            
        Returns:
            True if the setup is confirmed, False otherwise
        """
        return setup_id in self.confirmed_setup_ids
    
    def get_confirmed_setup_ids(self) -> Set[str]:
        """
        Get all confirmed setup IDs
        
        Returns:
            Set of confirmed setup IDs
        """
        return self.confirmed_setup_ids.copy()
    
    def get_setup_date(self, setup_id: str) -> Optional[datetime]:
        """
        Get the setup date for a given setup_id
        
        Args:
            setup_id: The setup ID
            
        Returns:
            datetime object of the setup date, or None if not found
        """
        return self.setup_dates.get(setup_id)
    
    def get_setup_ticker(self, setup_id: str) -> Dict[str, str]:
        """
        Get ticker information for a setup
        
        Args:
            setup_id: The setup ID
            
        Returns:
            Dictionary with lse_ticker and yahoo_ticker
        """
        return self.setup_tickers.get(setup_id, {})
    
    def is_historical_data(self, setup_id: str, data_date: str) -> bool:
        """
        Check if data is historical (before the setup date)
        
        Args:
            setup_id: The setup ID
            data_date: The date of the data (string format)
            
        Returns:
            True if data is historical (before setup date), False otherwise
        """
        if not self.is_confirmed_setup(setup_id):
            return False
        
        setup_date = self.get_setup_date(setup_id)
        if not setup_date:
            return False
        
        try:
            # Parse data date - handle different formats
            if isinstance(data_date, str):
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S.%f']:
                    try:
                        parsed_data_date = datetime.strptime(data_date, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    # Try pandas parsing as fallback
                    parsed_data_date = pd.to_datetime(data_date)
            else:
                parsed_data_date = pd.to_datetime(data_date)
            
            # Data is historical if it's before the setup date
            return parsed_data_date < setup_date
            
        except Exception as e:
            logger.warning(f"Could not parse data date {data_date}: {e}")
            return False
    
    def get_labels_for_confirmed_setups(self) -> pd.DataFrame:
        """
        Get labels data filtered by confirmed setups
        
        Returns:
            DataFrame with labels for confirmed setups only
        """
        try:
            query = """
            SELECT l.*
            FROM labels l
            WHERE l.setup_id IN ({})
            ORDER BY l.setup_id
            """.format(','.join([f"'{sid}'" for sid in self.confirmed_setup_ids]))
            
            labels_df = self.conn.execute(query).fetchdf()
            logger.info(f"Retrieved {len(labels_df)} labels for confirmed setups")
            return labels_df
            
        except Exception as e:
            logger.error(f"Error getting labels: {e}")
            return pd.DataFrame()
    
    def get_fundamentals_for_setup(self, setup_id: str) -> pd.DataFrame:
        """
        Get fundamentals data for a specific setup with date filtering
        
        Args:
            setup_id: The setup ID
            
        Returns:
            DataFrame with historical fundamentals data
        """
        if not self.is_confirmed_setup(setup_id):
            logger.warning(f"Setup {setup_id} is not confirmed")
            return pd.DataFrame()
        
        setup_date = self.get_setup_date(setup_id)
        if not setup_date:
            logger.warning(f"No setup date found for {setup_id}")
            return pd.DataFrame()
        
        # Get ticker information
        ticker_info = self.get_setup_ticker(setup_id)
        lse_ticker = ticker_info.get('lse_ticker', '')
        yahoo_ticker = ticker_info.get('yahoo_ticker', '')
        
        try:
            # Query fundamentals with date filtering
            # Try multiple ticker formats: plain, .L suffix, and both lse/yahoo variants
            query = """
            SELECT *
            FROM fundamentals
            WHERE (ticker = ? OR ticker = ? OR ticker = ? OR ticker = ?)
            AND date < ?
            ORDER BY date DESC
            """
            
            # Try all possible ticker combinations
            ticker_variants = [
                lse_ticker,           # e.g., "GROW"
                lse_ticker + '.L',    # e.g., "GROW.L"  
                yahoo_ticker,         # e.g., "GROW"
                yahoo_ticker + '.L'   # e.g., "GROW.L"
            ]
            
            fundamentals_df = self.conn.execute(query, ticker_variants + [setup_date]).fetchdf()
            logger.info(f"Retrieved {len(fundamentals_df)} historical fundamentals for {setup_id}")
            return fundamentals_df
            
        except Exception as e:
            logger.error(f"Error getting fundamentals for {setup_id}: {e}")
            return pd.DataFrame()
    
    def get_news_for_confirmed_setups(self, news_type: str = 'rns') -> pd.DataFrame:
        """
        Get news data for confirmed setups
        
        Args:
            news_type: Type of news ('rns' or 'enhanced')
            
        Returns:
            DataFrame with news data for confirmed setups
        """
        try:
            if news_type == 'rns':
                table_name = 'rns_announcements'
            elif news_type == 'enhanced':
                table_name = 'stock_news_enhanced'
            else:
                raise ValueError(f"Unknown news type: {news_type}")
            
            query = f"""
            SELECT *
            FROM {table_name}
            WHERE setup_id IN ({','.join([f"'{sid}'" for sid in self.confirmed_setup_ids])})
            ORDER BY setup_id
            """
            
            news_df = self.conn.execute(query).fetchdf()
            logger.info(f"Retrieved {len(news_df)} {news_type} news records for confirmed setups")
            return news_df
            
        except Exception as e:
            logger.error(f"Error getting {news_type} news: {e}")
            return pd.DataFrame()
    
    def get_user_posts_for_confirmed_setups(self) -> pd.DataFrame:
        """
        Get user posts data for confirmed setups
        
        Returns:
            DataFrame with user posts for confirmed setups
        """
        try:
            query = """
            SELECT *
            FROM user_posts
            WHERE setup_id IN ({})
            ORDER BY setup_id, post_date
            """.format(','.join([f"'{sid}'" for sid in self.confirmed_setup_ids]))
            
            posts_df = self.conn.execute(query).fetchdf()
            logger.info(f"Retrieved {len(posts_df)} user posts for confirmed setups")
            return posts_df
            
        except Exception as e:
            logger.error(f"Error getting user posts: {e}")
            return pd.DataFrame()
    
    def get_financial_ratios_for_setup(self, setup_id: str) -> pd.DataFrame:
        """
        Get financial ratios for a specific setup
        
        Args:
            setup_id: The setup ID
            
        Returns:
            DataFrame with financial ratios data
        """
        if not self.is_confirmed_setup(setup_id):
            return pd.DataFrame()
        
        setup_date = self.get_setup_date(setup_id)
        if not setup_date:
            return pd.DataFrame()
        
        ticker_info = self.get_setup_ticker(setup_id)
        lse_ticker = ticker_info.get('lse_ticker', '')
        yahoo_ticker = ticker_info.get('yahoo_ticker', '')
        
        try:
            query = """
            SELECT *
            FROM financial_ratios
            WHERE (ticker = ? OR ticker = ? OR ticker = ? OR ticker = ?)
            AND period_end < ?
            ORDER BY period_end DESC
            """
            
            # Try all possible ticker combinations
            ticker_variants = [
                lse_ticker,           # e.g., "GROW"
                lse_ticker + '.L',    # e.g., "GROW.L"  
                yahoo_ticker,         # e.g., "GROW"
                yahoo_ticker + '.L'   # e.g., "GROW.L"
            ]
            
            ratios_df = self.conn.execute(query, ticker_variants + [setup_date]).fetchdf()
            logger.info(f"Retrieved {len(ratios_df)} financial ratios for {setup_id}")
            return ratios_df
            
        except Exception as e:
            logger.error(f"Error getting financial ratios for {setup_id}: {e}")
            return pd.DataFrame()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about confirmed setups
        
        Returns:
            Dictionary containing summary statistics
        """
        try:
            # Get setup statistics
            setup_query = """
            SELECT 
                COUNT(*) as total_setups,
                COUNT(DISTINCT lse_ticker) as unique_tickers,
                MIN(spike_timestamp) as earliest_date,
                MAX(spike_timestamp) as latest_date
            FROM setups
            WHERE setup_id IS NOT NULL
            """
            
            setup_stats = self.conn.execute(setup_query).fetchone()
            
            # Get data counts
            data_counts = {}
            for table in ['labels', 'fundamentals', 'user_posts', 'rns_announcements', 'stock_news_enhanced']:
                try:
                    count_query = f"SELECT COUNT(*) FROM {table}"
                    count = self.conn.execute(count_query).fetchone()[0]
                    data_counts[table] = count
                except:
                    data_counts[table] = 0
            
            stats = {
                'total_confirmed_setups': len(self.confirmed_setup_ids),
                'confirmed_setup_ids': sorted(list(self.confirmed_setup_ids)),
                'unique_tickers': setup_stats[1] if setup_stats else 0,
                'date_range': {
                    'earliest': setup_stats[2] if setup_stats else None,
                    'latest': setup_stats[3] if setup_stats else None
                },
                'data_counts': data_counts
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting summary stats: {e}")
            return {}
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Ensure connection is closed when object is destroyed"""
        self.close()


def get_setup_validator_duckdb(db_path: str = "data/sentiment_system.duckdb") -> SetupValidatorDuckDB:
    """
    Factory function to get a DuckDB setup validator instance
    
    Args:
        db_path: Path to the DuckDB database file
        
    Returns:
        SetupValidatorDuckDB instance
    """
    return SetupValidatorDuckDB(db_path=db_path)


if __name__ == "__main__":
    # Test the DuckDB setup validator
    validator = SetupValidatorDuckDB(db_path="../data/sentiment_system.duckdb")
    stats = validator.get_summary_stats()
    
    print("DuckDB Setup Validator Summary:")
    print(f"Total confirmed setups: {stats['total_confirmed_setups']}")
    print(f"Unique tickers: {stats['unique_tickers']}")
    print(f"Date range: {stats['date_range']}")
    print(f"Data counts: {stats['data_counts']}")
    
    # Show first few setup IDs
    print(f"\nFirst 10 setup IDs: {stats['confirmed_setup_ids'][:10]}")
    
    # Test specific functionality
    if stats['confirmed_setup_ids']:
        test_setup = stats['confirmed_setup_ids'][0]
        print(f"\nTesting setup validation for {test_setup}:")
        print(f"Is confirmed: {validator.is_confirmed_setup(test_setup)}")
        print(f"Setup date: {validator.get_setup_date(test_setup)}")
        print(f"Ticker info: {validator.get_setup_ticker(test_setup)}")
    
    validator.close() 