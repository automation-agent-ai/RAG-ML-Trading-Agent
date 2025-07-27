#!/usr/bin/env python3
"""
CLI tool for Enhanced Fundamentals Agent Feature Extraction

This tool provides command-line interface for extracting fundamental features
using the enhanced fundamentals agent with DuckDB integration.

Usage:
    python cli_extract_fundamentals_features.py --discover
    python cli_extract_fundamentals_features.py --setup KZG_2024-10-16
    python cli_extract_fundamentals_features.py --all
    python cli_extract_fundamentals_features.py --batch setup1,setup2,setup3
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Add project root and agents directory to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "agents" / "fundamentals"))
from agents.fundamentals.enhanced_fundamentals_agent_duckdb import EnhancedFundamentalsAgentDuckDB

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FundamentalsCLI:
    """Command-line interface for fundamentals feature extraction"""
    
    def __init__(self, db_path: str = None, lancedb_dir: str = None):
        """Initialize CLI with database paths"""
        # Default paths (adjust for CLI usage)
        self.db_path = db_path or "../data/sentiment_system.duckdb"
        self.lancedb_dir = lancedb_dir or "../storage/lancedb_store"
        
        try:
            self.agent = EnhancedFundamentalsAgentDuckDB(
                db_path=self.db_path,
                lancedb_dir=self.lancedb_dir
            )
            logger.info("Enhanced Fundamentals Agent CLI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            sys.exit(1)
    
    def discover_setups(self) -> List[str]:
        """Discover available trading setups in the database"""
        import duckdb
        
        try:
            conn = duckdb.connect(self.db_path)
            
            # Get confirmed setups with fundamental data availability
            query = """
                SELECT DISTINCT s.setup_id, s.ticker, s.setup_date,
                       COUNT(f.ticker) as fundamental_records,
                       COUNT(r.ticker) as rns_records
                FROM setups s
                LEFT JOIN fundamentals f ON s.ticker = f.ticker
                LEFT JOIN rns_announcements r ON s.ticker = r.ticker 
                    AND r.date_published <= s.setup_date
                    AND r.date_published >= date(s.setup_date, '-90 days')
                    AND r.rns_category IN ('Trading Update', 'Final Results', 'Interim Results', 
                                         'Quarterly Results', 'Profit Warning', 'Trading Statement')
                WHERE s.confirmed = true
                GROUP BY s.setup_id, s.ticker, s.setup_date
                HAVING fundamental_records > 0 OR rns_records > 0
                ORDER BY s.setup_date DESC
            """
            
            results = conn.execute(query).fetchall()
            conn.close()
            
            setups = []
            print(f"\n{'Setup ID':<20} {'Ticker':<8} {'Date':<12} {'Fundamentals':<12} {'RNS Items':<10}")
            print("-" * 70)
            
            for row in results:
                setup_id, ticker, setup_date, fund_count, rns_count = row
                setups.append(setup_id)
                print(f"{setup_id:<20} {ticker:<8} {setup_date:<12} {fund_count:<12} {rns_count:<10}")
            
            print(f"\nFound {len(setups)} setups with fundamental/RNS data")
            return setups
            
        except Exception as e:
            logger.error(f"Error discovering setups: {e}")
            return []
    
    def process_single_setup(self, setup_id: str) -> bool:
        """Process a single trading setup"""
        logger.info(f"Processing fundamentals for setup: {setup_id}")
        
        try:
            features = self.agent.process_setup(setup_id)
            
            if features:
                print(f"\n✅ Successfully processed fundamentals for {setup_id}")
                print(f"📊 Structured Metrics:")
                print(f"   ROE: {features.roe}")
                print(f"   ROA: {features.roa}")
                print(f"   Debt/Equity: {features.debt_to_equity}")
                print(f"   Current Ratio: {features.current_ratio}")
                print(f"📰 Financial Results:")
                print(f"   Count: {features.count_financial_results}")
                print(f"   Profit Warning: {features.profit_warning_present}")
                print(f"   Capital Raise: {features.capital_raise_present}")
                print(f"   Summary: {features.synthetic_summary_financial_results[:100]}...")
                return True
            else:
                print(f"❌ Failed to process fundamentals for {setup_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing setup {setup_id}: {e}")
            print(f"❌ Error processing {setup_id}: {e}")
            return False
    
    def process_all_setups(self) -> None:
        """Process all available setups"""
        setups = self.discover_setups()
        
        if not setups:
            print("No setups found with fundamental data")
            return
        
        print(f"\nProcessing {len(setups)} setups...")
        
        success_count = 0
        failed_setups = []
        
        for i, setup_id in enumerate(setups, 1):
            print(f"\n[{i}/{len(setups)}] Processing {setup_id}...")
            
            try:
                if self.process_single_setup(setup_id):
                    success_count += 1
                else:
                    failed_setups.append(setup_id)
            except KeyboardInterrupt:
                print("\n⚠️ Processing interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error processing {setup_id}: {e}")
                failed_setups.append(setup_id)
        
        # Summary
        print(f"\n📊 Processing Summary:")
        print(f"✅ Successfully processed: {success_count}/{len(setups)}")
        if failed_setups:
            print(f"❌ Failed setups: {', '.join(failed_setups)}")
    
    def process_batch(self, setup_ids: List[str]) -> None:
        """Process a batch of specific setups"""
        print(f"Processing batch of {len(setup_ids)} setups...")
        
        success_count = 0
        failed_setups = []
        
        for i, setup_id in enumerate(setup_ids, 1):
            print(f"\n[{i}/{len(setup_ids)}] Processing {setup_id}...")
            
            try:
                if self.process_single_setup(setup_id):
                    success_count += 1
                else:
                    failed_setups.append(setup_id)
            except KeyboardInterrupt:
                print("\n⚠️ Processing interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error processing {setup_id}: {e}")
                failed_setups.append(setup_id)
        
        # Summary
        print(f"\n📊 Batch Processing Summary:")
        print(f"✅ Successfully processed: {success_count}/{len(setup_ids)}")
        if failed_setups:
            print(f"❌ Failed setups: {', '.join(failed_setups)}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Enhanced Fundamentals Agent Feature Extraction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Discover available setups
    python cli_extract_fundamentals_features.py --discover
    
    # Process single setup
    python cli_extract_fundamentals_features.py --setup KZG_2024-10-16
    
    # Process all setups
    python cli_extract_fundamentals_features.py --all
    
    # Process specific batch
    python cli_extract_fundamentals_features.py --batch "setup1,setup2,setup3"
    
    # Custom database paths
    python cli_extract_fundamentals_features.py --setup KZG_2024-10-16 --db ../data/sentiment_system.duckdb --lancedb ../lancedb_store
        """
    )
    
    # Action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        '--discover', 
        action='store_true',
        help='Discover available trading setups with fundamental data'
    )
    action_group.add_argument(
        '--setup', 
        type=str,
        help='Process a single trading setup'
    )
    action_group.add_argument(
        '--all', 
        action='store_true',
        help='Process all available setups'
    )
    action_group.add_argument(
        '--batch', 
        type=str,
        help='Process comma-separated list of setup IDs'
    )
    
    # Configuration arguments
    parser.add_argument(
        '--db', 
        type=str,
        help='Path to DuckDB database (default: ../data/sentiment_system.duckdb)'
    )
    parser.add_argument(
        '--lancedb', 
        type=str,
        help='Path to LanceDB directory (default: ../storage/lancedb_store)'
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize CLI
    try:
        cli = FundamentalsCLI(db_path=args.db, lancedb_dir=args.lancedb)
    except Exception as e:
        print(f"❌ Failed to initialize CLI: {e}")
        sys.exit(1)
    
    # Execute action
    try:
        if args.discover:
            cli.discover_setups()
            
        elif args.setup:
            success = cli.process_single_setup(args.setup)
            sys.exit(0 if success else 1)
            
        elif args.all:
            cli.process_all_setups()
            
        elif args.batch:
            setup_ids = [s.strip() for s in args.batch.split(',') if s.strip()]
            if not setup_ids:
                print("❌ No valid setup IDs provided in batch")
                sys.exit(1)
            cli.process_batch(setup_ids)
    
    except KeyboardInterrupt:
        print("\n⚠️ Operation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 