#!/usr/bin/env python3
"""
Export features tables from DuckDB to CSV files
"""

import duckdb
import pandas as pd
import os
from datetime import datetime

def export_features_tables():
    """Export all features tables from DuckDB to CSV"""
    
    db_path = "../data/sentiment_system.duckdb"
    output_dir = "../data/csv_export"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Connect to DuckDB
    conn = duckdb.connect(db_path)
    
    try:
        # Get list of all tables
        tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        tables_df = conn.execute(tables_query).df()
        
        print(f"📊 Found {len(tables_df)} tables in DuckDB:")
        for table in tables_df['table_name']:
            print(f"  - {table}")
        
        # Export news features
        print(f"\n📰 Exporting news features...")
        try:
            news_features = conn.execute("SELECT * FROM news_features ORDER BY setup_id").df()
            output_file = os.path.join(output_dir, f"news_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            news_features.to_csv(output_file, index=False)
            print(f"  ✅ Exported {len(news_features)} news feature records to: {output_file}")
            
            # Show sample
            print(f"  📋 Sample columns: {list(news_features.columns[:10])}")
            print(f"  📋 Shape: {news_features.shape}")
            
        except Exception as e:
            print(f"  ❌ Error exporting news features: {e}")
        
        # Export user posts features (if exists)
        print(f"\n👥 Exporting user posts features...")
        try:
            userposts_features = conn.execute("SELECT * FROM userposts_features ORDER BY setup_id").df()
            output_file = os.path.join(output_dir, f"userposts_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            userposts_features.to_csv(output_file, index=False)
            print(f"  ✅ Exported {len(userposts_features)} userposts feature records to: {output_file}")
            
            # Show sample
            print(f"  📋 Sample columns: {list(userposts_features.columns[:10])}")
            print(f"  📋 Shape: {userposts_features.shape}")
            
        except Exception as e:
            print(f"  ❌ Error exporting userposts features: {e}")
        
        # Export fundamentals features (if exists)
        print(f"\n💰 Exporting fundamentals features...")
        try:
            fundamentals_features = conn.execute("SELECT * FROM fundamentals_features ORDER BY setup_id").df()
            output_file = os.path.join(output_dir, f"fundamentals_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            fundamentals_features.to_csv(output_file, index=False)
            print(f"  ✅ Exported {len(fundamentals_features)} fundamentals feature records to: {output_file}")
            
            # Show sample
            print(f"  📋 Sample columns: {list(fundamentals_features.columns[:10])}")
            print(f"  📋 Shape: {fundamentals_features.shape}")
            
        except Exception as e:
            print(f"  ❌ Error exporting fundamentals features: {e}")
        
        # Show detailed news features sample
        print(f"\n🔍 NEWS FEATURES SAMPLE:")
        try:
            sample_news = conn.execute("""
                SELECT setup_id, 
                       count_financial_results, count_corporate_actions, count_governance, 
                       count_corporate_events, count_other_signals,
                       avg_headline_spin_financial_results, sentiment_score_financial_results,
                       synthetic_summary_financial_results
                FROM news_features 
                WHERE count_financial_results > 0 OR count_corporate_actions > 0 OR count_governance > 0
                ORDER BY setup_id 
                LIMIT 5
            """).df()
            
            if len(sample_news) > 0:
                print(sample_news.to_string(index=False))
            else:
                print("  📋 No news features with content found")
                
        except Exception as e:
            print(f"  ❌ Error getting news sample: {e}")
        
    finally:
        conn.close()
    
    print(f"\n✅ Export completed! Files saved to: {output_dir}")

if __name__ == "__main__":
    export_features_tables() 