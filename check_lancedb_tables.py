#!/usr/bin/env python3
"""
check_lancedb_tables.py - Utility to check if LanceDB tables exist and are accessible

This script checks if LanceDB tables exist and are accessible, and prints their metadata
"""

import os
import sys
import logging
from pathlib import Path
import lancedb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_lancedb_tables(lancedb_dir: str = "lancedb_store"):
    """
    Check if LanceDB tables exist and are accessible
    
    Args:
        lancedb_dir: Path to LanceDB directory
    """
    logger.info(f"Checking LanceDB tables in {lancedb_dir}")
    
    # Check if directory exists
    if not os.path.exists(lancedb_dir):
        logger.error(f"LanceDB directory {lancedb_dir} does not exist")
        return False
    
    try:
        # Connect to LanceDB
        db = lancedb.connect(lancedb_dir)
        
        # List all tables
        tables = db.table_names()
        logger.info(f"Found {len(tables)} tables: {tables}")
        
        # Check each table
        for table_name in tables:
            try:
                table = db.open_table(table_name)
                schema = table.schema
                count = table.count_rows()
                logger.info(f"Table {table_name}: {count} rows, schema: {schema}")
                
                # Check if table has vector data
                if count > 0:
                    try:
                        # Try to get first row
                        first_row = table.search().limit(1).to_pandas()
                        logger.info(f"First row columns: {first_row.columns.tolist()}")
                        
                        # Check if it has embedding vector
                        if 'vector' in first_row.columns:
                            vector_shape = first_row['vector'].iloc[0].shape
                            logger.info(f"Vector shape: {vector_shape}")
                        else:
                            logger.warning(f"Table {table_name} does not have 'vector' column")
                            
                        # Check if it has setup_id
                        if 'setup_id' in first_row.columns:
                            logger.info(f"Example setup_id: {first_row['setup_id'].iloc[0]}")
                        else:
                            logger.warning(f"Table {table_name} does not have 'setup_id' column")
                            
                        # Check if it has outperformance_10d (for training tables)
                        if 'outperformance_10d' in first_row.columns:
                            logger.info(f"Has outperformance_10d label: {first_row['outperformance_10d'].iloc[0]}")
                        else:
                            logger.warning(f"Table {table_name} does not have 'outperformance_10d' column")
                    except Exception as e:
                        logger.error(f"Error accessing first row of table {table_name}: {e}")
            except Exception as e:
                logger.error(f"Error opening table {table_name}: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Error connecting to LanceDB: {e}")
        return False

def check_with_absolute_path():
    """Check with absolute path to LanceDB directory"""
    abs_path = os.path.abspath("lancedb_store")
    logger.info(f"Checking with absolute path: {abs_path}")
    return check_lancedb_tables(abs_path)

def main():
    """Main function"""
    logger.info("Checking LanceDB tables")
    
    # Check with relative path
    success = check_lancedb_tables()
    
    # If failed, try with absolute path
    if not success:
        success = check_with_absolute_path()
    
    if success:
        logger.info("✅ LanceDB tables check completed successfully")
    else:
        logger.error("❌ LanceDB tables check failed")

if __name__ == "__main__":
    main() 