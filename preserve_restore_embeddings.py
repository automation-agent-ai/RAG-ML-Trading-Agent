#!/usr/bin/env python3
"""
Preserve and Restore Embeddings

This script handles two important tasks:
1. Preserving prediction embeddings for future use
2. Restoring prediction embeddings with labels back to the training set

Usage:
    # Preserve prediction embeddings
    python preserve_restore_embeddings.py --action preserve --domains all --setup-list data/prediction_setups.txt
    
    # Restore embeddings to training set (after labels are available)
    python preserve_restore_embeddings.py --action restore --domains all --setup-list data/prediction_setups.txt
"""

import os
import sys
import logging
import duckdb
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Class to manage embeddings preservation and restoration"""
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        lancedb_dir: str = "lancedb_store",
        backup_dir: str = "data/embeddings_backup"
    ):
        """
        Initialize the embedding manager
        
        Args:
            db_path: Path to DuckDB database
            lancedb_dir: Path to LanceDB directory
            backup_dir: Path to backup directory
        """
        self.db_path = db_path
        self.lancedb_dir = lancedb_dir
        self.backup_dir = backup_dir
        
        # Create backup directory if it doesn't exist
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def load_setup_ids(self, setup_list_file: str) -> List[str]:
        """
        Load setup IDs from file
        
        Args:
            setup_list_file: Path to setup list file
            
        Returns:
            List of setup IDs
        """
        with open(setup_list_file, 'r') as f:
            setup_ids = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(setup_ids)} setup IDs from {setup_list_file}")
        return setup_ids
    
    def preserve_embeddings(self, domains: List[str], setup_ids: List[str]) -> None:
        """
        Preserve prediction embeddings for future use
        
        Args:
            domains: List of domains to preserve
            setup_ids: List of setup IDs to preserve
        """
        conn = duckdb.connect(self.db_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for domain in domains:
            # Check if domain has embeddings table
            embedding_table = f"{domain}_embeddings_prediction"
            
            try:
                # Check if table exists
                result = conn.execute(f"SELECT * FROM {embedding_table} LIMIT 1").fetchone()
                
                # Export embeddings for this domain
                logger.info(f"Exporting {domain} embeddings...")
                
                # Query embeddings for specified setup IDs
                query = f"""
                    SELECT * 
                    FROM {embedding_table}
                    WHERE setup_id = ANY(?)
                """
                
                embeddings_df = conn.execute(query, [setup_ids]).fetchdf()
                
                if len(embeddings_df) == 0:
                    logger.warning(f"No {domain} embeddings found for the specified setup IDs")
                    continue
                
                # Save to backup directory
                backup_file = os.path.join(self.backup_dir, f"{domain}_embeddings_{timestamp}.parquet")
                embeddings_df.to_parquet(backup_file)
                
                logger.info(f"Saved {len(embeddings_df)} {domain} embeddings to {backup_file}")
                
            except Exception as e:
                logger.error(f"Error preserving {domain} embeddings: {e}")
        
        conn.close()
        logger.info("Embeddings preservation completed")
    
    def restore_embeddings(self, domains: List[str], setup_ids: List[str]) -> None:
        """
        Restore prediction embeddings to training set
        
        Args:
            domains: List of domains to restore
            setup_ids: List of setup IDs to restore
        """
        conn = duckdb.connect(self.db_path)
        
        for domain in domains:
            # Check if domain has embeddings table
            prediction_table = f"{domain}_embeddings_prediction"
            training_table = f"{domain}_embeddings"
            
            try:
                # Check if tables exist
                try:
                    conn.execute(f"SELECT * FROM {prediction_table} LIMIT 1").fetchone()
                    conn.execute(f"SELECT * FROM {training_table} LIMIT 1").fetchone()
                except:
                    logger.error(f"Tables {prediction_table} or {training_table} do not exist")
                    continue
                
                # Check if labels are available for these setup IDs
                label_query = """
                    SELECT setup_id, outperformance_10d
                    FROM labels
                    WHERE setup_id = ANY(?)
                    AND outperformance_10d IS NOT NULL
                """
                
                labels_df = conn.execute(label_query, [setup_ids]).fetchdf()
                
                if len(labels_df) == 0:
                    logger.warning(f"No labels found for the specified setup IDs")
                    continue
                
                # Get setup IDs with labels
                labeled_setup_ids = labels_df['setup_id'].tolist()
                
                logger.info(f"Found labels for {len(labeled_setup_ids)} setup IDs")
                
                # Get embeddings for these setup IDs
                embeddings_query = f"""
                    SELECT *
                    FROM {prediction_table}
                    WHERE setup_id = ANY(?)
                """
                
                embeddings_df = conn.execute(embeddings_query, [labeled_setup_ids]).fetchdf()
                
                if len(embeddings_df) == 0:
                    logger.warning(f"No {domain} embeddings found for setup IDs with labels")
                    continue
                
                # Merge embeddings with labels
                embeddings_df = pd.merge(
                    embeddings_df,
                    labels_df,
                    on='setup_id',
                    how='left'
                )
                
                logger.info(f"Merging {len(embeddings_df)} {domain} embeddings with labels")
                
                # Insert into training table
                # First, create a temporary table
                temp_table = f"temp_{domain}_embeddings"
                conn.execute(f"CREATE OR REPLACE TABLE {temp_table} AS SELECT * FROM embeddings_df")
                
                # Then, insert into training table
                insert_query = f"""
                    INSERT INTO {training_table}
                    SELECT * FROM {temp_table}
                """
                
                conn.execute(insert_query)
                
                # Drop temporary table
                conn.execute(f"DROP TABLE {temp_table}")
                
                # Count rows in training table
                count = conn.execute(f"SELECT COUNT(*) FROM {training_table}").fetchone()[0]
                
                logger.info(f"Successfully restored {len(embeddings_df)} {domain} embeddings to training set")
                logger.info(f"Training table {training_table} now has {count} rows")
                
                # Also update LanceDB table if applicable
                if domain in ['news', 'userposts', 'analyst_recommendations']:
                    logger.info(f"Updating LanceDB table for {domain}...")
                    try:
                        # This would require LanceDB-specific code
                        # For now, just log a message
                        logger.info(f"LanceDB update for {domain} would happen here")
                    except Exception as e:
                        logger.error(f"Error updating LanceDB for {domain}: {e}")
                
            except Exception as e:
                logger.error(f"Error restoring {domain} embeddings: {e}")
        
        conn.close()
        logger.info("Embeddings restoration completed")
    
    def run(self, action: str, domains: List[str], setup_list_file: str) -> None:
        """
        Run the embedding manager
        
        Args:
            action: Action to perform ('preserve' or 'restore')
            domains: List of domains to process
            setup_list_file: Path to setup list file
        """
        # Load setup IDs
        setup_ids = self.load_setup_ids(setup_list_file)
        
        # Process all domains if 'all' is specified
        if 'all' in domains:
            domains = ['news', 'fundamentals', 'analyst_recommendations', 'userposts']
        
        # Perform action
        if action == 'preserve':
            logger.info(f"Preserving embeddings for {len(setup_ids)} setup IDs across {len(domains)} domains")
            self.preserve_embeddings(domains, setup_ids)
        elif action == 'restore':
            logger.info(f"Restoring embeddings for {len(setup_ids)} setup IDs across {len(domains)} domains")
            self.restore_embeddings(domains, setup_ids)
                        else:
            logger.error(f"Unknown action: {action}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Preserve and restore embeddings')
    parser.add_argument('--action', choices=['preserve', 'restore'], required=True,
                       help='Action to perform')
    parser.add_argument('--domains', nargs='+', default=['all'],
                       help='Domains to process')
    parser.add_argument('--setup-list', required=True,
                       help='Path to setup list file')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--lancedb-dir', default='lancedb_store',
                       help='Path to LanceDB directory')
    parser.add_argument('--backup-dir', default='data/embeddings_backup',
                       help='Path to backup directory')
    
    args = parser.parse_args()
    
    # Initialize embedding manager
    embedding_manager = EmbeddingManager(
        db_path=args.db_path,
        lancedb_dir=args.lancedb_dir,
        backup_dir=args.backup_dir
    )
    
    # Run embedding manager
    embedding_manager.run(args.action, args.domains, args.setup_list)

if __name__ == "__main__":
    main() 