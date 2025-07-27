#!/usr/bin/env python3
"""
Check Setup Embeddings - Utility to verify setup_id status in embedding stores

This tool helps identify if a setup_id:
1. Already exists in any embedding tables
2. Contains performance labels (potential data leakage)
3. Was processed in training or prediction mode
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import lancedb
import pandas as pd
from typing import Dict, List, Set, Tuple

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SetupEmbeddingsChecker:
    """Utility to check setup_id status in embedding stores"""
    
    def __init__(self, lancedb_dir: str = "lancedb_store"):
        """Initialize the checker with LanceDB directory"""
        self.lancedb_dir = Path(lancedb_dir)
        
        # Connect to LanceDB
        if not self.lancedb_dir.exists():
            raise FileNotFoundError(f"LanceDB directory not found: {self.lancedb_dir}")
            
        self.db = lancedb.connect(str(self.lancedb_dir))
        
        # Get all available tables
        self.tables = self.db.table_names()
        logger.info(f"Found {len(self.tables)} tables in LanceDB")
        
        # Domain-specific tables
        self.domain_tables = {
            'news': [t for t in self.tables if 'news' in t.lower()],
            'fundamentals': [t for t in self.tables if 'fundamental' in t.lower()],
            'analyst': [t for t in self.tables if 'analyst' in t.lower()],
            'userposts': [t for t in self.tables if 'userposts' in t.lower() or 'user_posts' in t.lower()]
        }
    
    def check_setup_ids(self, setup_ids: List[str]) -> Dict[str, Dict[str, Dict[str, bool]]]:
        """
        Check if setup_ids exist in embedding tables and if they contain labels
        
        Args:
            setup_ids: List of setup_ids to check
            
        Returns:
            Dictionary with status for each setup_id by domain and table
        """
        results = {}
        
        for setup_id in setup_ids:
            results[setup_id] = {}
            
            for domain, tables in self.domain_tables.items():
                results[setup_id][domain] = {}
                
                for table_name in tables:
                    try:
                        table = self.db.open_table(table_name)
                        
                        # Query for this setup_id
                        query = f"setup_id = '{setup_id}'"
                        matches = table.search(query).to_pandas()
                        
                        if len(matches) > 0:
                            # Check if records contain labels
                            has_labels = False
                            has_performance_labels = False
                            
                            if 'has_performance_labels' in matches.columns:
                                has_performance_labels = matches['has_performance_labels'].any()
                            
                            if 'outperformance_10d' in matches.columns:
                                has_labels = matches['outperformance_10d'].abs().max() > 0
                            
                            results[setup_id][domain][table_name] = {
                                'exists': True,
                                'has_labels': has_labels or has_performance_labels,
                                'record_count': len(matches)
                            }
                        else:
                            results[setup_id][domain][table_name] = {
                                'exists': False,
                                'has_labels': False,
                                'record_count': 0
                            }
                    except Exception as e:
                        logger.error(f"Error checking table {table_name}: {e}")
                        results[setup_id][domain][table_name] = {
                            'exists': False,
                            'has_labels': False,
                            'record_count': 0,
                            'error': str(e)
                        }
        
        return results
    
    def get_safe_prediction_setups(self, setup_ids: List[str]) -> List[str]:
        """
        Get list of setup_ids that are safe to use for prediction
        (not present in any embedding table with labels)
        
        Args:
            setup_ids: List of setup_ids to check
            
        Returns:
            List of setup_ids that are safe for prediction
        """
        results = self.check_setup_ids(setup_ids)
        safe_setups = []
        
        for setup_id, domains in results.items():
            is_safe = True
            
            for domain, tables in domains.items():
                for table_name, status in tables.items():
                    if status['exists'] and status['has_labels']:
                        is_safe = False
                        break
                
                if not is_safe:
                    break
            
            if is_safe:
                safe_setups.append(setup_id)
        
        return safe_setups
    
    def display_results(self, results: Dict[str, Dict[str, Dict[str, bool]]]) -> None:
        """Display results in a readable format"""
        print("\n" + "="*80)
        print("SETUP EMBEDDINGS STATUS")
        print("="*80)
        
        for setup_id, domains in results.items():
            print(f"\nSetup ID: {setup_id}")
            print("-" * 50)
            
            setup_is_safe = True
            setup_exists_anywhere = False
            
            for domain, tables in domains.items():
                print(f"  {domain.upper()}:")
                
                for table_name, status in tables.items():
                    if status['exists']:
                        setup_exists_anywhere = True
                        label_status = "⚠️ WITH LABELS" if status['has_labels'] else "✅ NO LABELS"
                        if status['has_labels']:
                            setup_is_safe = False
                        print(f"    - {table_name}: {status['record_count']} records {label_status}")
                    else:
                        print(f"    - {table_name}: Not present")
            
            # Overall status
            if not setup_exists_anywhere:
                print(f"\n  OVERALL: ✅ SAFE FOR PREDICTION (not in any embedding table)")
            elif setup_is_safe:
                print(f"\n  OVERALL: ✅ SAFE FOR PREDICTION (exists but no labels)")
            else:
                print(f"\n  OVERALL: ⚠️ NOT SAFE FOR PREDICTION (has labels in some tables)")
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        safe_count = sum(1 for setup_id, domains in results.items() 
                       if all(not any(status['exists'] and status['has_labels'] 
                                    for status in tables.values())
                            for domain, tables in domains.items()))
        
        print(f"Total setup_ids checked: {len(results)}")
        print(f"Safe for prediction: {safe_count}")
        print(f"Not safe (has labels): {len(results) - safe_count}")
        print("="*80)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Check setup_id status in embedding stores')
    parser.add_argument('--setup-ids', nargs='+', required=True,
                      help='List of setup_ids to check')
    parser.add_argument('--lancedb-dir', default='lancedb_store',
                      help='Path to LanceDB directory')
    parser.add_argument('--safe-only', action='store_true',
                      help='Only output setup_ids that are safe for prediction')
    
    args = parser.parse_args()
    
    try:
        checker = SetupEmbeddingsChecker(lancedb_dir=args.lancedb_dir)
        
        if args.safe_only:
            safe_setups = checker.get_safe_prediction_setups(args.setup_ids)
            print("Safe setup_ids for prediction:")
            for setup_id in safe_setups:
                print(setup_id)
            print(f"\nFound {len(safe_setups)} safe setup_ids out of {len(args.setup_ids)}")
        else:
            results = checker.check_setup_ids(args.setup_ids)
            checker.display_results(results)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 