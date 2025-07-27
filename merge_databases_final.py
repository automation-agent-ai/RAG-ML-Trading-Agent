import duckdb
import os
import sys
import shutil
from datetime import datetime

def backup_current_db(current_db_path):
    """Create a backup of the current database"""
    backup_path = current_db_path.replace('.duckdb', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.duckdb')
    print(f"Creating backup of current database at: {backup_path}")
    if os.path.exists(current_db_path):
        shutil.copy2(current_db_path, backup_path)
    return backup_path

def get_table_list(db_path):
    """Get list of tables in a database"""
    conn = duckdb.connect(db_path)
    tables = [r[0] for r in conn.execute('PRAGMA show_tables').fetchall()]
    conn.close()
    return tables

def merge_databases(current_db_path, new_db_path, output_db_path=None):
    """Merge two DuckDB databases"""
    if output_db_path is None:
        output_db_path = current_db_path.replace('.duckdb', '_merged.duckdb')
    
    # Create backup of current database
    backup_path = backup_current_db(current_db_path)
    
    # Get table lists
    current_tables = get_table_list(current_db_path)
    new_tables = get_table_list(new_db_path)
    
    # Tables only in current database (keep as is)
    current_only_tables = set(current_tables) - set(new_tables)
    # Tables only in new database (copy to output)
    new_only_tables = set(new_tables) - set(current_tables)
    # Common tables (need to check schema)
    common_tables = set(current_tables) & set(new_tables)
    
    print(f"\nTables only in current database (will be preserved): {sorted(current_only_tables)}")
    print(f"\nTables only in new database (will be copied): {sorted(new_only_tables)}")
    print(f"\nCommon tables (will be updated with new data): {sorted(common_tables)}")
    
    # Create a new output database
    if os.path.exists(output_db_path):
        os.remove(output_db_path)
    
    # Copy the current database to the output path
    shutil.copy2(current_db_path, output_db_path)
    
    # Connect to output database
    conn_out = duckdb.connect(output_db_path)
    conn_new = duckdb.connect(new_db_path)
    
    # Copy tables only in new database
    for table in new_only_tables:
        print(f"Copying table {table} from new database")
        # Get the CREATE TABLE statement
        create_stmt = conn_new.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'").fetchone()
        if create_stmt and create_stmt[0]:
            conn_out.execute(create_stmt[0])
            # Copy data
            data = conn_new.execute(f"SELECT * FROM {table}").fetchall()
            if data:
                columns = conn_new.execute(f"PRAGMA table_info({table})").fetchall()
                column_names = [col[1] for col in columns]
                placeholders = ", ".join(["?" for _ in column_names])
                conn_out.executemany(f"INSERT INTO {table} VALUES ({placeholders})", data)
            print(f"  - Copied {len(data) if data else 0} rows")
    
    # Update common tables with data from new database
    for table in common_tables:
        print(f"Updating table {table} with data from new database")
        # Drop the existing table
        conn_out.execute(f"DROP TABLE IF EXISTS {table}")
        
        # Get the CREATE TABLE statement
        create_stmt = conn_new.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'").fetchone()
        if create_stmt and create_stmt[0]:
            conn_out.execute(create_stmt[0])
            # Copy data
            data = conn_new.execute(f"SELECT * FROM {table}").fetchall()
            if data:
                columns = conn_new.execute(f"PRAGMA table_info({table})").fetchall()
                column_names = [col[1] for col in columns]
                placeholders = ", ".join(["?" for _ in column_names])
                conn_out.executemany(f"INSERT INTO {table} VALUES ({placeholders})", data)
            print(f"  - Updated with {len(data) if data else 0} rows")
    
    # Close connections
    conn_out.close()
    conn_new.close()
    
    print(f"\nDatabase merge complete. Output saved at: {output_db_path}")
    print(f"Backup of original database saved at: {backup_path}")
    return output_db_path

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_databases_final.py <current_db_path> <new_db_path> [output_db_path]")
        sys.exit(1)
    
    current_db_path = sys.argv[1]
    new_db_path = sys.argv[2]
    output_db_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    merge_databases(current_db_path, new_db_path, output_db_path) 