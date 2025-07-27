#!/usr/bin/env python3
"""
Embedding and Feature Preservation Tool

This script provides functions to:
1. Preserve embeddings and features for prediction setups
2. Remove prediction setups from training tables
3. Restore preserved embeddings and features back to training tables

This approach avoids unnecessary re-computation by preserving original data.

Usage:
    # Preserve embeddings and features for prediction setups
    python preserve_restore_embeddings.py preserve --prediction-setup-file data/prediction_setups_20230101.txt
    
    # Remove prediction setups from training tables
    python preserve_restore_embeddings.py remove --prediction-setup-file data/prediction_setups.txt
    
    # Restore preserved embeddings and features back to training
    python preserve_restore_embeddings.py restore --preserved-file data/preserved_data_20230101.pkl
"""

import os
import sys
import argparse
import logging
import pickle
import pandas as pd
import duckdb
import lancedb
from pathlib import Path
from typing import List, Dict, Set, Any, Optional, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataManager:
    """Manages embedding and feature preservation and restoration"""
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        lancedb_dir: str = "lancedb_store",
        domains: List[str] = ["news", "fundamentals", "analyst_recommendations", "userposts"]
    ):
        self.db_path = db_path
        self.lancedb_dir = lancedb_dir
        self.domains = domains
        self.db = lancedb.connect(lancedb_dir)
        self.duckdb_conn = duckdb.connect(db_path)
        
    def load_setup_ids(self, file_path: str) -> List[str]:
        """Load setup IDs from a file"""
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    def preserve_data(self, prediction_setup_file: str) -> str:
        """
        Extract embeddings and features for prediction setups and save them
        
        Args:
            prediction_setup_file: File containing prediction setup IDs
            
        Returns:
            Path to the preserved data file
        """
        # Load prediction setup IDs
        prediction_setups = self.load_setup_ids(prediction_setup_file)
        logger.info(f"Loaded {len(prediction_setups)} prediction setups")
        
        # Create dictionary to store preserved data
        preserved_data = {
            "embeddings": {},
            "features": {}
        }
        
        # Process embeddings for each domain
        for domain in self.domains:
            logger.info(f"Processing embeddings for domain: {domain}")
            
            # Table name
            table_name = f"{domain}_embeddings"
            
            try:
                # Open table
                table = self.db.open_table(table_name)
                
                # Extract embeddings for prediction setups
                df = table.to_pandas()
                prediction_df = df[df['setup_id'].isin(prediction_setups)]
                
                if prediction_df.empty:
                    logger.warning(f"No embeddings found for {domain} prediction setups")
                else:
                    # Store embeddings
                    preserved_data["embeddings"][domain] = prediction_df
                    logger.info(f"Preserved {len(prediction_df)} embeddings for {domain}")
                
            except Exception as e:
                logger.error(f"Error preserving embeddings for {domain}: {e}")
        
        # Process features for each domain
        for domain in self.domains:
            logger.info(f"Processing features for domain: {domain}")
            
            # Table name
            feature_table = f"{domain}_features"
            
            try:
                # Query features
                query = f"""
                SELECT * FROM {feature_table}
                WHERE setup_id IN (
                    SELECT UNNEST(?::VARCHAR[])
                )
                """
                
                # Execute query
                features_df = self.duckdb_conn.execute(query, [prediction_setups]).df()
                
                if features_df.empty:
                    logger.warning(f"No features found for {domain} prediction setups")
                else:
                    # Store features
                    preserved_data["features"][domain] = features_df
                    logger.info(f"Preserved {len(features_df)} feature rows for {domain}")
                
            except Exception as e:
                logger.error(f"Error preserving features for {domain}: {e}")
        
        # Save preserved data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"preserved_data_{timestamp}.pkl"
        
        with open(output_file, 'wb') as f:
            pickle.dump(preserved_data, f)
        
        logger.info(f"Saved preserved data to {output_file}")
        return str(output_file)
    
    def restore_data(self, preserved_file: str) -> Dict[str, Dict[str, int]]:
        """
        Restore preserved embeddings and features back to training tables
        
        Args:
            preserved_file: File containing preserved data
            
        Returns:
            Dictionary with counts of restored items by domain and type
        """
        # Load preserved data
        with open(preserved_file, 'rb') as f:
            preserved_data = pickle.load(f)
        
        logger.info(f"Loaded preserved data from {preserved_file}")
        
        # Track restoration counts
        restored_counts = {
            "embeddings": {},
            "features": {}
        }
        
        # Restore embeddings
        for domain, embeddings_df in preserved_data.get("embeddings", {}).items():
            logger.info(f"Restoring embeddings for domain: {domain}")
            
            # Table name
            training_table_name = f"{domain}_embeddings_training"
            
            try:
                # Open training table
                training_table = self.db.open_table(training_table_name)
                
                # Add preserved embeddings back to training table
                training_table.add(embeddings_df)
                
                # Record count
                restored_counts["embeddings"][domain] = len(embeddings_df)
                logger.info(f"Restored {len(embeddings_df)} embeddings to {training_table_name}")
                
            except Exception as e:
                logger.error(f"Error restoring embeddings for {domain}: {e}")
                restored_counts["embeddings"][domain] = 0
        
        # Restore features
        for domain, features_df in preserved_data.get("features", {}).items():
            logger.info(f"Restoring features for domain: {domain}")
            
            # Table name
            feature_table = f"{domain}_features"
            
            try:
                # First, delete any existing features for these setup IDs
                setup_ids = features_df['setup_id'].tolist()
                delete_query = f"""
                DELETE FROM {feature_table}
                WHERE setup_id IN (
                    SELECT UNNEST(?::VARCHAR[])
                )
                """
                self.duckdb_conn.execute(delete_query, [setup_ids])
                
                # Insert preserved features
                self.duckdb_conn.register("temp_features", features_df)
                insert_query = f"""
                INSERT INTO {feature_table}
                SELECT * FROM temp_features
                """
                self.duckdb_conn.execute(insert_query)
                
                # Record count
                restored_counts["features"][domain] = len(features_df)
                logger.info(f"Restored {len(features_df)} feature rows to {feature_table}")
                
            except Exception as e:
                logger.error(f"Error restoring features for {domain}: {e}")
                restored_counts["features"][domain] = 0
        
        # Commit changes
        self.duckdb_conn.commit()
        
        return restored_counts
    
    def remove_prediction_setups(self, prediction_setup_file: str) -> Dict[str, Dict[str, int]]:
        """
        Remove prediction setups from training tables (both embeddings and features)
        
        Args:
            prediction_setup_file: File containing prediction setup IDs
            
        Returns:
            Dictionary with counts of removed items by domain and type
        """
        # Load prediction setup IDs
        prediction_setups = self.load_setup_ids(prediction_setup_file)
        logger.info(f"Loaded {len(prediction_setups)} prediction setups")
        
        # Track removal counts
        removed_counts = {
            "embeddings": {},
            "features": {}
        }
        
        # Remove embeddings
        for domain in self.domains:
            logger.info(f"Removing embeddings for domain: {domain}")
            
            # Table names
            table_name = f"{domain}_embeddings"
            
            try:
                # Open table
                table = self.db.open_table(table_name)
                
                # Get current count
                before_count = len(table.to_pandas())
                
                # Remove prediction setups
                for setup_id in prediction_setups:
                    table.delete(f"setup_id = '{setup_id}'")
                
                # Get new count
                after_count = len(table.to_pandas())
                removed = before_count - after_count
                
                # Record count
                removed_counts["embeddings"][domain] = removed
                logger.info(f"Removed {removed} embeddings from {table_name}")
                
            except Exception as e:
                logger.error(f"Error removing embeddings for {domain}: {e}")
                removed_counts["embeddings"][domain] = 0
        
        # Remove features
        for domain in self.domains:
            logger.info(f"Removing features for domain: {domain}")
            
            # Table name
            feature_table = f"{domain}_features"
            
            try:
                # Get current count
                count_query = f"SELECT COUNT(*) FROM {feature_table}"
                before_count = self.duckdb_conn.execute(count_query).fetchone()[0]
                
                # Delete features for prediction setups
                delete_query = f"""
                DELETE FROM {feature_table}
                WHERE setup_id IN (
                    SELECT UNNEST(?::VARCHAR[])
                )
                """
                self.duckdb_conn.execute(delete_query, [prediction_setups])
                
                # Get new count
                after_count = self.duckdb_conn.execute(count_query).fetchone()[0]
                removed = before_count - after_count
                
                # Record count
                removed_counts["features"][domain] = removed
                logger.info(f"Removed {removed} feature rows from {feature_table}")
                
            except Exception as e:
                logger.error(f"Error removing features for {domain}: {e}")
                removed_counts["features"][domain] = 0
        
        # Commit changes
        self.duckdb_conn.commit()
        
        return removed_counts
    
    def remove_similarity_features(self, prediction_setup_file: str) -> Dict[str, int]:
        """
        Remove only similarity-based features for prediction setups
        This allows keeping raw features but recalculating similarity-based ones
        
        Args:
            prediction_setup_file: File containing prediction setup IDs
            
        Returns:
            Dictionary with counts of affected rows by domain
        """
        # Load prediction setup IDs
        prediction_setups = self.load_setup_ids(prediction_setup_file)
        logger.info(f"Loaded {len(prediction_setups)} prediction setups")
        
        # Track affected counts
        affected_counts = {}
        
        # Process each domain
        for domain in self.domains:
            logger.info(f"Resetting similarity features for domain: {domain}")
            
            # Table name
            feature_table = f"{domain}_features"
            
            try:
                # Get all columns for this table
                columns_query = f"PRAGMA table_info({feature_table})"
                columns_df = self.duckdb_conn.execute(columns_query).df()
                all_columns = columns_df['name'].tolist()
                
                # Get setup_id rows that need updating
                setup_query = f"""
                SELECT * FROM {feature_table}
                WHERE setup_id IN (
                    SELECT UNNEST(?::VARCHAR[])
                )
                LIMIT 1
                """
                
                try:
                    # Try to get one row to check column values
                    sample_row = self.duckdb_conn.execute(setup_query, [prediction_setups]).df()
                    
                    if not sample_row.empty:
                        # Identify similarity-related columns by checking for non-null values
                        # We'll use a simple heuristic: columns that might contain similarity features
                        # typically have names containing these keywords
                        similarity_keywords = ['similar', 'confidence', 'probability', 'signal', 'risk', 'score', 'prediction']
                        
                        # Filter columns that might be similarity-related
                        potential_similarity_columns = []
                        for col in all_columns:
                            # Skip setup_id and timestamp columns
                            if col == 'setup_id' or 'timestamp' in col or 'model' in col:
                                continue
                                
                            # Check if column name contains any similarity keyword
                            if any(keyword in col.lower() for keyword in similarity_keywords):
                                potential_similarity_columns.append(col)
                        
                        if potential_similarity_columns:
                            # Build update query to reset similarity features
                            set_clauses = [f"{col} = NULL" for col in potential_similarity_columns]
                            update_query = f"""
                            UPDATE {feature_table}
                            SET {', '.join(set_clauses)}
                            WHERE setup_id IN (
                                SELECT UNNEST(?::VARCHAR[])
                            )
                            """
                            
                            # Execute update
                            self.duckdb_conn.execute(update_query, [prediction_setups])
                            
                            # Get count of affected rows
                            affected_count = self.duckdb_conn.execute(
                                f"SELECT COUNT(*) FROM {feature_table} WHERE setup_id IN (SELECT UNNEST(?::VARCHAR[]))",
                                [prediction_setups]
                            ).fetchone()[0]
                            
                            # Record count
                            affected_counts[domain] = affected_count
                            logger.info(f"Reset {len(potential_similarity_columns)} similarity features for {affected_count} rows in {feature_table}")
                            logger.info(f"Reset columns: {', '.join(potential_similarity_columns)}")
                        else:
                            logger.warning(f"No similarity-related columns found in {feature_table}")
                            affected_counts[domain] = 0
                    else:
                        logger.warning(f"No matching rows found in {feature_table} for the prediction setups")
                        affected_counts[domain] = 0
                        
                except Exception as e:
                    logger.error(f"Error analyzing table structure for {domain}: {e}")
                    affected_counts[domain] = 0
                
            except Exception as e:
                logger.error(f"Error resetting similarity features for {domain}: {e}")
                affected_counts[domain] = 0
        
        # Commit changes
        self.duckdb_conn.commit()
        
        return affected_counts
    
    def cleanup(self):
        """Close connections"""
        if hasattr(self, 'duckdb_conn'):
            self.duckdb_conn.close()
        logger.info("Connections closed")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Preserve and restore embeddings and features for prediction setups"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Preserve command
    preserve_parser = subparsers.add_parser(
        "preserve", help="Preserve embeddings and features for prediction setups"
    )
    preserve_parser.add_argument(
        "--prediction-setup-file", required=True,
        help="File containing prediction setup IDs"
    )
    preserve_parser.add_argument(
        "--db-path", default="data/sentiment_system.duckdb",
        help="Path to DuckDB database"
    )
    preserve_parser.add_argument(
        "--lancedb-dir", default="lancedb_store",
        help="LanceDB directory"
    )
    preserve_parser.add_argument(
        "--domains", nargs="+", 
        default=["news", "fundamentals", "analyst_recommendations", "userposts"],
        help="Domains to process"
    )
    
    # Restore command
    restore_parser = subparsers.add_parser(
        "restore", help="Restore preserved embeddings and features to training tables"
    )
    restore_parser.add_argument(
        "--preserved-file", required=True,
        help="File containing preserved data"
    )
    restore_parser.add_argument(
        "--db-path", default="data/sentiment_system.duckdb",
        help="Path to DuckDB database"
    )
    restore_parser.add_argument(
        "--lancedb-dir", default="lancedb_store",
        help="LanceDB directory"
    )
    restore_parser.add_argument(
        "--domains", nargs="+", 
        default=["news", "fundamentals", "analyst_recommendations", "userposts"],
        help="Domains to process"
    )
    
    # Remove command
    remove_parser = subparsers.add_parser(
        "remove", help="Remove prediction setups from training tables"
    )
    remove_parser.add_argument(
        "--prediction-setup-file", required=True,
        help="File containing prediction setup IDs"
    )
    remove_parser.add_argument(
        "--db-path", default="data/sentiment_system.duckdb",
        help="Path to DuckDB database"
    )
    remove_parser.add_argument(
        "--lancedb-dir", default="lancedb_store",
        help="LanceDB directory"
    )
    remove_parser.add_argument(
        "--domains", nargs="+", 
        default=["news", "fundamentals", "analyst_recommendations", "userposts"],
        help="Domains to process"
    )
    
    # Reset similarity features command
    reset_parser = subparsers.add_parser(
        "reset-similarity", help="Reset similarity-based features for prediction setups"
    )
    reset_parser.add_argument(
        "--prediction-setup-file", required=True,
        help="File containing prediction setup IDs"
    )
    reset_parser.add_argument(
        "--db-path", default="data/sentiment_system.duckdb",
        help="Path to DuckDB database"
    )
    reset_parser.add_argument(
        "--lancedb-dir", default="lancedb_store",
        help="LanceDB directory"
    )
    reset_parser.add_argument(
        "--domains", nargs="+", 
        default=["news", "fundamentals", "analyst_recommendations", "userposts"],
        help="Domains to process"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize data manager
    manager = DataManager(
        db_path=args.db_path,
        lancedb_dir=args.lancedb_dir,
        domains=args.domains
    )
    
    try:
        # Execute command
        if args.command == "preserve":
            preserved_file = manager.preserve_data(args.prediction_setup_file)
            logger.info(f"Preserved data saved to: {preserved_file}")
        
        elif args.command == "restore":
            restored_counts = manager.restore_data(args.preserved_file)
            logger.info("Restoration complete:")
            for data_type, domains in restored_counts.items():
                logger.info(f"  {data_type.capitalize()}:")
                for domain, count in domains.items():
                    logger.info(f"    {domain}: {count} items restored")
        
        elif args.command == "remove":
            removed_counts = manager.remove_prediction_setups(args.prediction_setup_file)
            logger.info("Removal complete:")
            for data_type, domains in removed_counts.items():
                logger.info(f"  {data_type.capitalize()}:")
                for domain, count in domains.items():
                    logger.info(f"    {domain}: {count} items removed")
        
        elif args.command == "reset-similarity":
            affected_counts = manager.remove_similarity_features(args.prediction_setup_file)
            logger.info("Similarity features reset complete:")
            for domain, count in affected_counts.items():
                logger.info(f"  {domain}: {count} rows affected")
    
    finally:
        # Clean up
        manager.cleanup()

if __name__ == "__main__":
    main() 