#!/usr/bin/env python3
"""
Check Labels in DuckDB Database

This script checks the label columns in the DuckDB database.
"""

import os
import sys
import duckdb
import pandas as pd

def check_db_labels(db_path="data/sentiment_system.duckdb"):
    """Check labels in the DuckDB database"""
    print(f"Checking labels in {db_path}")
    print("-" * 50)
    
    # Connect to the database
    conn = duckdb.connect(db_path)
    
    try:
        # Check tables
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"Total tables: {len(tables)}")
        print("Tables:")
        for table in tables:
            print(f"- {table[0]}")
        
        # Check daily_labels table
        print("\nChecking daily_labels table...")
        try:
            # Get row count
            row_count = conn.execute("SELECT COUNT(*) FROM daily_labels").fetchone()[0]
            print(f"Total rows in daily_labels: {row_count}")
            
            # Get sample rows
            if row_count > 0:
                print("\nSample rows from daily_labels:")
                sample = conn.execute("SELECT * FROM daily_labels LIMIT 5").df()
                print(sample)
                
                # Get unique setup_ids
                setup_count = conn.execute("SELECT COUNT(DISTINCT setup_id) FROM daily_labels").fetchone()[0]
                print(f"\nUnique setup_ids in daily_labels: {setup_count}")
                
                # Get outperformance statistics
                print("\noutperformance_day statistics:")
                stats = conn.execute("SELECT MIN(outperformance_day), MAX(outperformance_day), AVG(outperformance_day) FROM daily_labels").fetchone()
                print(f"Min: {stats[0]}, Max: {stats[1]}, Avg: {stats[2]}")
                
                # Check for setups with complete labels (days 1-10)
                complete_setups = conn.execute("""
                    SELECT setup_id, COUNT(DISTINCT day_number) as days
                    FROM daily_labels
                    WHERE day_number <= 10
                    GROUP BY setup_id
                    HAVING COUNT(DISTINCT day_number) >= 10
                """).df()
                
                print(f"\nSetups with complete labels (days 1-10): {len(complete_setups)}")
                
                if len(complete_setups) > 0:
                    # Get average outperformance for these setups
                    avg_outperformance = conn.execute("""
                        SELECT setup_id, AVG(outperformance_day) as avg_outperformance
                        FROM daily_labels
                        WHERE day_number <= 10
                        AND setup_id IN (
                            SELECT setup_id
                            FROM daily_labels
                            WHERE day_number <= 10
                            GROUP BY setup_id
                            HAVING COUNT(DISTINCT day_number) >= 10
                        )
                        GROUP BY setup_id
                    """).df()
                    
                    print("\nAverage outperformance statistics:")
                    print(avg_outperformance['avg_outperformance'].describe())
                    
                    # Calculate class distribution based on thresholds
                    pos_threshold = 0.02
                    neg_threshold = -0.02
                    
                    pos_count = len(avg_outperformance[avg_outperformance['avg_outperformance'] >= pos_threshold])
                    neg_count = len(avg_outperformance[avg_outperformance['avg_outperformance'] <= neg_threshold])
                    neutral_count = len(avg_outperformance[(avg_outperformance['avg_outperformance'] > neg_threshold) & (avg_outperformance['avg_outperformance'] < pos_threshold)])
                    
                    print("\nClass distribution based on thresholds:")
                    print(f"Positive (>= {pos_threshold}): {pos_count} ({pos_count/len(avg_outperformance):.1%})")
                    print(f"Neutral (between {neg_threshold} and {pos_threshold}): {neutral_count} ({neutral_count/len(avg_outperformance):.1%})")
                    print(f"Negative (<= {neg_threshold}): {neg_count} ({neg_count/len(avg_outperformance):.1%})")
        except Exception as e:
            print(f"Error checking daily_labels: {str(e)}")
        
        # Check if there's a labels table
        label_tables = [table[0] for table in tables if 'label' in table[0].lower()]
        if label_tables:
            print("\nFound label-related tables:")
            for table in label_tables:
                print(f"- {table}")
                try:
                    row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    print(f"  Total rows: {row_count}")
                    
                    if row_count > 0:
                        # Get column names
                        columns = conn.execute(f"PRAGMA table_info({table})").df()['name'].tolist()
                        print(f"  Columns: {columns}")
                        
                        # Get sample rows
                        print(f"\n  Sample rows from {table}:")
                        sample = conn.execute(f"SELECT * FROM {table} LIMIT 5").df()
                        print(sample)
                except Exception as e:
                    print(f"  Error checking {table}: {str(e)}")
        
        # Check for outperformance in other tables
        print("\nChecking for outperformance in other tables...")
        for table in tables:
            table_name = table[0]
            try:
                columns = conn.execute(f"PRAGMA table_info({table_name})").df()['name'].tolist()
                outperformance_cols = [col for col in columns if 'outperformance' in col.lower()]
                
                if outperformance_cols:
                    print(f"\nFound outperformance columns in {table_name}:")
                    for col in outperformance_cols:
                        print(f"- {col}")
                        
                        # Get statistics
                        stats = conn.execute(f"SELECT MIN({col}), MAX({col}), AVG({col}) FROM {table_name} WHERE {col} IS NOT NULL").fetchone()
                        print(f"  Min: {stats[0]}, Max: {stats[1]}, Avg: {stats[2]}")
                        
                        # Get non-null count
                        non_null = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NOT NULL").fetchone()[0]
                        total = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                        print(f"  Non-null values: {non_null}/{total} ({non_null/total:.1%})")
            except Exception as e:
                # Skip errors for system tables
                if not table_name.startswith('sqlite_'):
                    print(f"Error checking {table_name}: {str(e)}")
    
    finally:
        # Close the connection
        conn.close()

def main():
    """Main function"""
    db_path = "data/sentiment_system.duckdb"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    check_db_labels(db_path)

if __name__ == '__main__':
    main() 