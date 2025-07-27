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

def get_table_list(conn, db_name=None):
    """Get list of tables in a database"""
    if db_name:
        tables = [r[0] for r in conn.execute(f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{db_name}'").fetchall()]
    else:
        tables = [r[0] for r in conn.execute('PRAGMA show_tables').fetchall()]
    return tables

def merge_databases(current_db_path, new_db_path, output_db_path=None):
    """Merge two DuckDB databases using ATTACH DATABASE"""
    if output_db_path is None:
        output_db_path = current_db_path.replace('.duckdb', '_merged.duckdb')
    
    # Create backup of current database
    backup_path = backup_current_db(current_db_path)
    
    # Create a new output database
    if os.path.exists(output_db_path):
        os.remove(output_db_path)
    
    # Connect to output database
    conn = duckdb.connect(output_db_path)
    
    # Attach both source databases
    conn.execute(f"ATTACH DATABASE '{current_db_path}' AS current")
    conn.execute(f"ATTACH DATABASE '{new_db_path}' AS new")
    
    # Get table lists
    current_tables = get_table_list(conn, "current")
    new_tables = get_table_list(conn, "new")
    
    # Tables only in current database (keep as is)
    current_only_tables = set(current_tables) - set(new_tables)
    # Tables only in new database (copy to output)
    new_only_tables = set(new_tables) - set(current_tables)
    # Common tables (need to check schema)
    common_tables = set(current_tables) & set(new_tables)
    
    print(f"\nTables only in current database (will be preserved): {sorted(current_only_tables)}")
    print(f"\nTables only in new database (will be copied): {sorted(new_only_tables)}")
    print(f"\nCommon tables (will be updated with new data): {sorted(common_tables)}")
    
    # Copy tables only in current database
    for table in current_only_tables:
        print(f"Copying table {table} from current database")
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM current.{table}")
    
    # Copy tables only in new database
    for table in new_only_tables:
        print(f"Copying table {table} from new database")
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM new.{table}")
    
    # Copy common tables from new database (replacing current data)
    for table in common_tables:
        print(f"Updating table {table} with data from new database")
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM new.{table}")
    
    # Detach databases
    conn.execute("DETACH DATABASE current")
    conn.execute("DETACH DATABASE new")
    
    # Close connection
    conn.close()
    
    print(f"\nDatabase merge complete. Output saved at: {output_db_path}")
    print(f"Backup of original database saved at: {backup_path}")
    return output_db_path

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_databases_simple.py <current_db_path> <new_db_path> [output_db_path]")
        sys.exit(1)
    
    current_db_path = sys.argv[1]
    new_db_path = sys.argv[2]
    output_db_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    merge_databases(current_db_path, new_db_path, output_db_path) 