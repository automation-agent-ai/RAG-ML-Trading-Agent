#!/usr/bin/env python3
"""
Clean Labeled Embeddings - Utility to remove setup_ids with labels from embedding tables

This tool helps prevent data leakage by:
1. Identifying setup_ids with performance labels in embedding tables
2. Removing those records from the tables
3. Optionally recreating the embeddings in prediction mode
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import lancedb
import pandas as pd
from typing import Dict, List, Set, Tuple
import numpy as np

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

class LabeledEmbeddingsCleaner:
    """Utility to clean setup_ids with labels from embedding tables"""
    
    def __init__(self, lancedb_dir: str = "lancedb_store"):
        """Initialize the cleaner with LanceDB directory"""
        self.lancedb_dir = Path(lancedb_dir)
        
        # Connect to LanceDB
        if not self.lancedb_dir.exists():
            raise FileNotFoundError(f"LanceDB directory not found: {self.lancedb_dir}")
            
        self.db = lancedb.connect(str(self.lancedb_dir))
        
        # Initialize checker
        self.checker = SetupEmbeddingsChecker(lancedb_dir=lancedb_dir)
        
    def clean_setup_ids(self, setup_ids: List[str], dry_run: bool = True) -> Dict[str, Dict[str, int]]:
        """
        Remove setup_ids with labels from embedding tables
        
        Args:
            setup_ids: List of setup_ids to clean
            dry_run: If True, only report what would be removed without making changes
            
        Returns:
            Dictionary with removal counts by domain and table
        """
        # First check which setup_ids have labels
        check_results = self.checker.check_setup_ids(setup_ids)
        
        # Track removal counts
        removal_counts = {}
        
        for setup_id, domains in check_results.items():
            removal_counts[setup_id] = {}
            
            for domain, tables in domains.items():
                removal_counts[setup_id][domain] = {}
                
                for table_name, status in tables.items():
                    if status['exists'] and status['has_labels']:
                        logger.info(f"Found {status['record_count']} records with labels for {setup_id} in {table_name}")
                        
                        if not dry_run:
                            try:
                                table = self.db.open_table(table_name)
                                
                                # Get current records
                                all_records = table.to_pandas()
                                
                                # Filter out records for this setup_id
                                filtered_records = all_records[all_records['setup_id'] != setup_id]
                                
                                # Count removed records
                                removed_count = len(all_records) - len(filtered_records)
                                
                                if removed_count > 0:
                                    # Drop and recreate table
                                    self.db.drop_table(table_name)
                                    
                                    # Convert vector column to numpy arrays
                                    filtered_records['vector'] = filtered_records['vector'].apply(
                                        lambda x: x if isinstance(x, np.ndarray) else np.array(x, dtype=np.float32)
                                    )
                                    
                                    # Create new table
                                    self.db.create_table(table_name, filtered_records)
                                    
                                    logger.info(f"Removed {removed_count} records for {setup_id} from {table_name}")
                                    removal_counts[setup_id][domain][table_name] = removed_count
                                else:
                                    logger.warning(f"No records found to remove for {setup_id} in {table_name}")
                                    removal_counts[setup_id][domain][table_name] = 0
                                    
                            except Exception as e:
                                logger.error(f"Error removing {setup_id} from {table_name}: {e}")
                                removal_counts[setup_id][domain][table_name] = -1  # Error flag
                        else:
                            logger.info(f"[DRY RUN] Would remove {status['record_count']} records for {setup_id} from {table_name}")
                            removal_counts[setup_id][domain][table_name] = status['record_count']
                    else:
                        removal_counts[setup_id][domain][table_name] = 0
        
        return removal_counts
    
    def recreate_embeddings(self, setup_ids: List[str]) -> Dict[str, bool]:
        """
        Recreate embeddings for setup_ids in prediction mode
        
        Args:
            setup_ids: List of setup_ids to recreate
            
        Returns:
            Dictionary with success status by setup_id
        """
        from run_complete_ml_pipeline import CompletePipeline
        
        results = {}
        
        try:
            pipeline = CompletePipeline()
            
            # Run embedding creation in prediction mode
            embedding_results = pipeline.create_embeddings(setup_ids, mode='prediction')
            
            for setup_id in setup_ids:
                results[setup_id] = all(embedding_results.values())
                
            return results
            
        except Exception as e:
            logger.error(f"Error recreating embeddings: {e}")
            return {setup_id: False for setup_id in setup_ids}
    
    def display_results(self, removal_counts: Dict[str, Dict[str, Dict[str, int]]], dry_run: bool = True) -> None:
        """Display results in a readable format"""
        print("\n" + "="*80)
        print(f"{'[DRY RUN] ' if dry_run else ''}LABELED EMBEDDINGS CLEANING RESULTS")
        print("="*80)
        
        total_removed = 0
        
        for setup_id, domains in removal_counts.items():
            setup_total = 0
            setup_has_removals = False
            
            print(f"\nSetup ID: {setup_id}")
            print("-" * 50)
            
            for domain, tables in domains.items():
                domain_total = sum(count for count in tables.values() if count > 0)
                
                if domain_total > 0:
                    setup_has_removals = True
                    setup_total += domain_total
                    
                    print(f"  {domain.upper()}: {domain_total} records removed")
                    
                    for table_name, count in tables.items():
                        if count > 0:
                            print(f"    - {table_name}: {count} records")
                        elif count == -1:
                            print(f"    - {table_name}: ERROR")
            
            if setup_has_removals:
                print(f"\n  Total for {setup_id}: {setup_total} records removed")
                total_removed += setup_total
            else:
                print(f"\n  No labeled records found for {setup_id}")
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        print(f"Total setup_ids processed: {len(removal_counts)}")
        print(f"Total records {'that would be removed' if dry_run else 'removed'}: {total_removed}")
        print("="*80)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Clean setup_ids with labels from embedding tables')
    parser.add_argument('--setup-ids', nargs='+', required=True,
                      help='List of setup_ids to clean')
    parser.add_argument('--lancedb-dir', default='lancedb_store',
                      help='Path to LanceDB directory')
    parser.add_argument('--dry-run', action='store_true',
                      help='Only report what would be removed without making changes')
    parser.add_argument('--recreate', action='store_true',
                      help='Recreate embeddings in prediction mode after cleaning')
    
    args = parser.parse_args()
    
    try:
        cleaner = LabeledEmbeddingsCleaner(lancedb_dir=args.lancedb_dir)
        
        # Clean setup_ids
        removal_counts = cleaner.clean_setup_ids(args.setup_ids, dry_run=args.dry_run)
        cleaner.display_results(removal_counts, dry_run=args.dry_run)
        
        # Recreate embeddings if requested
        if args.recreate and not args.dry_run:
            logger.info("Recreating embeddings in prediction mode...")
            results = cleaner.recreate_embeddings(args.setup_ids)
            
            success_count = sum(1 for success in results.values() if success)
            logger.info(f"Successfully recreated embeddings for {success_count}/{len(args.setup_ids)} setup_ids")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 