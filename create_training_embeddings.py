#!/usr/bin/env python3
"""
Create Training Embeddings

This script creates training embeddings for a set of setup IDs.
These embeddings will be used for similarity search in prediction mode.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List

# Add project root to path for proper imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_training_embeddings(setup_ids: List[str], db_path: str = "data/sentiment_system.duckdb", lancedb_dir: str = "lancedb_store"):
    """
    Create training embeddings for the given setup IDs
    
    Args:
        setup_ids: List of setup IDs to process
        db_path: Path to DuckDB database
        lancedb_dir: Directory for LanceDB vector store
    """
    from run_complete_ml_pipeline import CompletePipeline
    
    # Initialize pipeline
    pipeline = CompletePipeline(db_path=db_path, lancedb_dir=lancedb_dir)
    
    # Create embeddings in training mode
    logger.info(f"Creating training embeddings for {len(setup_ids)} setup IDs")
    pipeline.create_embeddings(setup_ids, mode="training")
    
    # Verify that embeddings were created
    import lancedb
    db = lancedb.connect(lancedb_dir)
    
    tables = db.table_names()
    logger.info(f"Available tables in LanceDB: {tables}")
    
    for table_name in tables:
        try:
            table = db.open_table(table_name)
            count = len(table.to_pandas())
            logger.info(f"Table {table_name}: {count} records")
        except Exception as e:
            logger.error(f"Error opening table {table_name}: {e}")

def get_training_setup_ids(db_path: str = "data/sentiment_system.duckdb", limit: int = 10) -> List[str]:
    """
    Get a list of setup IDs with complete features and labels
    
    Args:
        db_path: Path to DuckDB database
        limit: Maximum number of setup IDs to return
        
    Returns:
        List of setup IDs
    """
    import duckdb
    
    conn = duckdb.connect(db_path)
    
    # Get setup IDs with labels
    query = f"""
        SELECT DISTINCT s.setup_id
        FROM setups s
        JOIN labels l ON s.setup_id = l.setup_id
        WHERE l.outperformance_10d IS NOT NULL
        LIMIT {limit}
    """
    
    try:
        result = conn.execute(query).fetchall()
        setup_ids = [row[0] for row in result]
        conn.close()
        return setup_ids
    except Exception as e:
        logger.error(f"Error getting training setup IDs: {e}")
        conn.close()
        return []

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create training embeddings')
    parser.add_argument('--setup-ids', nargs='+',
                      help='List of setup IDs to process (if not provided, will use random setup IDs with labels)')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                      help='Path to DuckDB database')
    parser.add_argument('--lancedb-dir', default='lancedb_store',
                      help='Directory for LanceDB vector store')
    parser.add_argument('--limit', type=int, default=10,
                      help='Maximum number of setup IDs to process if --setup-ids is not provided')
    
    args = parser.parse_args()
    
    # Get setup IDs
    setup_ids = args.setup_ids
    if not setup_ids:
        logger.info(f"No setup IDs provided, getting up to {args.limit} random setup IDs with labels")
        setup_ids = get_training_setup_ids(args.db_path, args.limit)
        
    if not setup_ids:
        logger.error("No setup IDs found")
        return 1
        
    logger.info(f"Creating training embeddings for {len(setup_ids)} setup IDs: {setup_ids}")
    
    # Create training embeddings
    create_training_embeddings(setup_ids, args.db_path, args.lancedb_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 