#!/usr/bin/env python3
"""
Training/Prediction Data Splitter

This script:
1. Identifies setups with complete data across all domains
2. Randomly splits them into training and prediction sets
3. Creates separate embedding tables for each
4. Ensures no data leakage between training and prediction

Usage:
    python split_train_prediction_data.py --prediction-ratio 0.2
"""

import os
import sys
import argparse
import random
import logging
import duckdb
import pandas as pd
import lancedb
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainPredictionSplitter:
    """Split data into training and prediction sets"""
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        lancedb_dir: str = "lancedb_store",
        prediction_ratio: float = 0.2,
        random_seed: int = 42
    ):
        self.db_path = db_path
        self.lancedb_dir = lancedb_dir
        self.prediction_ratio = prediction_ratio
        self.random_seed = random_seed
        random.seed(random_seed)
        
        # Connect to databases
        self.duckdb_conn = duckdb.connect(db_path)
        self.lancedb = lancedb.connect(lancedb_dir)
        
        # Domain tables
        self.domains = ["news", "fundamentals", "analyst_recommendations", "userposts"]
        
        # Track setups
        self.all_setups = set()
        self.complete_setups = set()
        self.training_setups = set()
        self.prediction_setups = set()
    
    def find_complete_setups(self) -> Set[str]:
        """Find setups with data in all domains and labels"""
        try:
            # Find setups that have all required features and labels
            complete_setups_query = """
            WITH required_features AS (
                SELECT setup_id 
                FROM fundamentals_features
                INTERSECT
                SELECT setup_id 
                FROM news_features
                INTERSECT
                SELECT setup_id 
                FROM userposts_features
                INTERSECT
                SELECT setup_id 
                FROM analyst_recommendations_features
                INTERSECT
                SELECT setup_id 
                FROM labels
                WHERE outperformance_10d IS NOT NULL
            )
            SELECT setup_id 
            FROM required_features
            ORDER BY setup_id
            """
            
            setup_ids = [row[0] for row in self.duckdb_conn.execute(complete_setups_query).fetchall()]
            self.complete_setups = set(setup_ids)
            logger.info(f"Found {len(self.complete_setups)} setups with complete data")
            
            return self.complete_setups
        
        except Exception as e:
            logger.error(f"Error finding complete setups: {e}")
            return set()
    
    def split_setups(self) -> Tuple[Set[str], Set[str]]:
        """Split complete setups into training and prediction sets"""
        if not self.complete_setups:
            self.find_complete_setups()
        
        if not self.complete_setups:
            logger.error("No complete setups found")
            return set(), set()
        
        # Convert to list for random sampling
        setup_list = list(self.complete_setups)
        
        # Calculate number of prediction samples
        n_prediction = max(1, int(len(setup_list) * self.prediction_ratio))
        
        # Randomly sample for prediction
        prediction_setups = set(random.sample(setup_list, n_prediction))
        
        # Rest goes to training
        training_setups = self.complete_setups - prediction_setups
        
        self.training_setups = training_setups
        self.prediction_setups = prediction_setups
        
        logger.info(f"Split data: {len(training_setups)} training, {len(prediction_setups)} prediction")
        
        return training_setups, prediction_setups
    
    def create_split_tables(self) -> Dict[str, bool]:
        """Create separate embedding tables for training and prediction"""
        if not self.training_setups or not self.prediction_setups:
            self.split_setups()
        
        results = {}
        
        # Process each domain
        for domain in self.domains:
            table_name = f"{domain}_embeddings"
            training_table_name = f"{domain}_embeddings_training"
            prediction_table_name = f"{domain}_embeddings_prediction"
            
            try:
                # Check if source table exists
                try:
                    source_table = self.lancedb.open_table(table_name)
                except Exception as e:
                    logger.error(f"Source table {table_name} not found: {e}")
                    results[domain] = False
                    continue
                
                # Read all data
                df = source_table.to_pandas()
                if df.empty:
                    logger.warning(f"Source table {table_name} is empty")
                    results[domain] = False
                    continue
                
                # Split data
                training_df = df[df['setup_id'].isin(self.training_setups)]
                prediction_df = df[df['setup_id'].isin(self.prediction_setups)]
                
                # Remove labels from prediction data
                if 'outperformance_10d' in prediction_df.columns:
                    # Save labels separately for evaluation
                    prediction_labels = prediction_df[['setup_id', 'outperformance_10d']].copy()
                    prediction_labels_file = f"data/prediction_labels_{domain}.csv"
                    prediction_labels.to_csv(prediction_labels_file, index=False)
                    logger.info(f"Saved prediction labels to {prediction_labels_file}")
                    
                    # Remove labels from prediction data
                    prediction_df['outperformance_10d'] = 0.0
                
                # Create training table
                if not training_df.empty:
                    try:
                        # Drop existing table if it exists
                        try:
                            self.lancedb.drop_table(training_table_name)
                        except:
                            pass
                        
                        # Create new table
                        self.lancedb.create_table(
                            training_table_name,
                            data=training_df,
                            mode="overwrite"
                        )
                        logger.info(f"Created training table {training_table_name} with {len(training_df)} records")
                    except Exception as e:
                        logger.error(f"Error creating training table {training_table_name}: {e}")
                
                # Create prediction table
                if not prediction_df.empty:
                    try:
                        # Drop existing table if it exists
                        try:
                            self.lancedb.drop_table(prediction_table_name)
                        except:
                            pass
                        
                        # Create new table
                        self.lancedb.create_table(
                            prediction_table_name,
                            data=prediction_df,
                            mode="overwrite"
                        )
                        logger.info(f"Created prediction table {prediction_table_name} with {len(prediction_df)} records")
                    except Exception as e:
                        logger.error(f"Error creating prediction table {prediction_table_name}: {e}")
                
                results[domain] = True
                
            except Exception as e:
                logger.error(f"Error processing {domain}: {e}")
                results[domain] = False
        
        return results
    
    def save_setup_lists(self) -> None:
        """Save lists of training and prediction setups"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save training setups
        training_file = f"data/training_setups_{timestamp}.txt"
        with open(training_file, 'w') as f:
            for setup_id in sorted(self.training_setups):
                f.write(f"{setup_id}\n")
        logger.info(f"Saved {len(self.training_setups)} training setups to {training_file}")
        
        # Save prediction setups
        prediction_file = f"data/prediction_setups_{timestamp}.txt"
        with open(prediction_file, 'w') as f:
            for setup_id in sorted(self.prediction_setups):
                f.write(f"{setup_id}\n")
        logger.info(f"Saved {len(self.prediction_setups)} prediction setups to {prediction_file}")
    
    def cleanup(self) -> None:
        """Close connections"""
        if hasattr(self, 'duckdb_conn'):
            self.duckdb_conn.close()
        logger.info("Cleanup complete")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Split data into training and prediction sets')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--lancedb-dir', default='lancedb_store',
                       help='Path to LanceDB directory')
    parser.add_argument('--prediction-ratio', type=float, default=0.2,
                       help='Ratio of data to use for prediction (0.0-1.0)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.prediction_ratio <= 0.0 or args.prediction_ratio >= 1.0:
        parser.error("Prediction ratio must be between 0.0 and 1.0")
    
    # Initialize splitter
    splitter = TrainPredictionSplitter(
        db_path=args.db_path,
        lancedb_dir=args.lancedb_dir,
        prediction_ratio=args.prediction_ratio,
        random_seed=args.random_seed
    )
    
    # Find complete setups
    complete_setups = splitter.find_complete_setups()
    logger.info(f"Found {len(complete_setups)} complete setups")
    
    # Split setups
    training_setups, prediction_setups = splitter.split_setups()
    logger.info(f"Split into {len(training_setups)} training and {len(prediction_setups)} prediction setups")
    
    # Create split tables
    results = splitter.create_split_tables()
    for domain, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        logger.info(f"{status}: {domain}")
    
    # Save setup lists
    splitter.save_setup_lists()
    
    # Cleanup
    splitter.cleanup()
    
    logger.info("Data splitting complete")


if __name__ == "__main__":
    main() 