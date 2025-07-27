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
        # Copy the file using shutil
        shutil.copy2(current_db_path, backup_path)
    return backup_path

def get_table_list(db_path):
    """Get list of tables in a database"""
    conn = duckdb.connect(db_path)
    tables = [r[0] for r in conn.execute('PRAGMA show_tables').fetchall()]
    conn.close()
    return tables

def get_table_schema(db_path, table_name):
    """Get schema of a table"""
    conn = duckdb.connect(db_path)
    schema = conn.execute(f'DESCRIBE {table_name}').fetchall()
    conn.close()
    return schema

def get_column_names(schema):
    """Extract column names from schema"""
    return [col[0] for col in schema]

def get_column_types(schema):
    """Extract column types from schema"""
    return {col[0]: col[1] for col in schema}

def copy_table(source_db, target_db, table_name, overwrite=True):
    """Copy a table from source to target database"""
    source_conn = duckdb.connect(source_db)
    target_conn = duckdb.connect(target_db)
    
    # Check if table exists in target
    target_tables = get_table_list(target_db)
    if table_name in target_tables and overwrite:
        target_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    
    # Get data from source
    data = source_conn.execute(f"SELECT * FROM {table_name}").fetchall()
    
    if data:
        # Get column names and types
        schema = get_table_schema(source_db, table_name)
        columns = get_column_names(schema)
        
        # Create table in target
        column_defs = []
        for col in schema:
            name, type_, nullable, key, default, extra = col
            nullable_str = "NULL" if nullable == "YES" else "NOT NULL"
            
            # Handle default values carefully
            default_str = ""
            if default is not None:
                # Skip CAST expressions in defaults as they cause syntax errors
                if "CAST" in str(default):
                    if "'f'" in str(default):
                        default_str = "DEFAULT FALSE"
                    elif "'t'" in str(default):
                        default_str = "DEFAULT TRUE"
                    else:
                        # Skip problematic default
                        default_str = ""
                else:
                    default_str = f"DEFAULT {default}"
                    
            column_defs.append(f"{name} {type_} {nullable_str} {default_str}".strip())
        
        create_stmt = f"CREATE TABLE {table_name} ({', '.join(column_defs)})"
        target_conn.execute(create_stmt)
        
        # Insert data
        placeholders = ", ".join(["?" for _ in columns])
        target_conn.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", data)
        
        print(f"Copied table {table_name} with {len(data)} rows")
    else:
        print(f"Table {table_name} is empty, creating empty table")
        # Create empty table with same schema
        schema = get_table_schema(source_db, table_name)
        columns = get_column_names(schema)
        column_defs = []
        for col in schema:
            name, type_, nullable, key, default, extra = col
            nullable_str = "NULL" if nullable == "YES" else "NOT NULL"
            
            # Handle default values carefully
            default_str = ""
            if default is not None:
                # Skip CAST expressions in defaults as they cause syntax errors
                if "CAST" in str(default):
                    if "'f'" in str(default):
                        default_str = "DEFAULT FALSE"
                    elif "'t'" in str(default):
                        default_str = "DEFAULT TRUE"
                    else:
                        # Skip problematic default
                        default_str = ""
                else:
                    default_str = f"DEFAULT {default}"
                    
            column_defs.append(f"{name} {type_} {nullable_str} {default_str}".strip())
        
        create_stmt = f"CREATE TABLE {table_name} ({', '.join(column_defs)})"
        target_conn.execute(create_stmt)
    
    source_conn.close()
    target_conn.close()

def merge_tables_with_schema_diff(current_db, new_db, table_name, output_db):
    """Merge tables with schema differences"""
    current_conn = duckdb.connect(current_db)
    new_conn = duckdb.connect(new_db)
    output_conn = duckdb.connect(output_db)
    
    # Get schemas
    current_schema = get_table_schema(current_db, table_name)
    new_schema = get_table_schema(new_db, table_name)
    
    current_columns = get_column_names(current_schema)
    new_columns = get_column_names(new_schema)
    
    # Identify common columns
    common_columns = list(set(current_columns) & set(new_columns))
    
    # Create a temporary table with the new schema
    column_defs = []
    for col in new_schema:
        name, type_, nullable, key, default, extra = col
        nullable_str = "NULL" if nullable == "YES" else "NOT NULL"
        
        # Handle default values carefully
        default_str = ""
        if default is not None:
            # Skip CAST expressions in defaults as they cause syntax errors
            if "CAST" in str(default):
                if "'f'" in str(default):
                    default_str = "DEFAULT FALSE"
                elif "'t'" in str(default):
                    default_str = "DEFAULT TRUE"
                else:
                    # Skip problematic default
                    default_str = ""
            else:
                default_str = f"DEFAULT {default}"
                
        column_defs.append(f"{name} {type_} {nullable_str} {default_str}".strip())
    
    output_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    output_conn.execute(f"CREATE TABLE {table_name} ({', '.join(column_defs)})")
    
    # Copy data from new database
    common_cols_str = ", ".join(common_columns)
    new_data = new_conn.execute(f"SELECT {common_cols_str} FROM {table_name}").fetchall()
    
    if new_data:
        placeholders = ", ".join(["?" for _ in common_columns])
        output_conn.executemany(f"INSERT INTO {table_name} ({common_cols_str}) VALUES ({placeholders})", new_data)
        print(f"Merged table {table_name} with {len(new_data)} rows from new database")
    
    current_conn.close()
    new_conn.close()
    output_conn.close()

def merge_databases(current_db_path, new_db_path, output_db_path=None):
    """Merge two DuckDB databases"""
    if output_db_path is None:
        output_db_path = current_db_path
    
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
    print(f"\nCommon tables (will be merged): {sorted(common_tables)}")
    
    # Create output database (if different from current)
    if output_db_path != current_db_path:
        if os.path.exists(output_db_path):
            os.remove(output_db_path)
        
        # Copy current database to output
        for table in current_tables:
            copy_table(current_db_path, output_db_path, table)
    
    # Copy tables only in new database
    for table in new_only_tables:
        copy_table(new_db_path, output_db_path, table)
    
    # Merge common tables
    for table in common_tables:
        current_schema = get_table_schema(current_db_path, table)
        new_schema = get_table_schema(new_db_path, table)
        
        current_columns = get_column_names(current_schema)
        new_columns = get_column_names(new_schema)
        
        # Check if schemas are identical
        if set(current_columns) == set(new_columns):
            print(f"Table {table} has identical schema, replacing with new data")
            copy_table(new_db_path, output_db_path, table, overwrite=True)
        else:
            print(f"Table {table} has schema differences:")
            print(f"  - Columns only in current: {set(current_columns) - set(new_columns)}")
            print(f"  - Columns only in new: {set(new_columns) - set(current_columns)}")
            merge_tables_with_schema_diff(current_db_path, new_db_path, table, output_db_path)
    
    print(f"\nDatabase merge complete. Backup saved at: {backup_path}")
    return output_db_path

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_databases.py <current_db_path> <new_db_path> [output_db_path]")
        sys.exit(1)
    
    current_db_path = sys.argv[1]
    new_db_path = sys.argv[2]
    output_db_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    merge_databases(current_db_path, new_db_path, output_db_path) 