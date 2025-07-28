#!/usr/bin/env python3
"""
Test News Agent

This script tests the news agent with a few setup IDs to see if it can handle
the "Director Dealings" category correctly.
"""

import os
import sys
import logging
import duckdb
from pathlib import Path
import pandas as pd
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from the news agent
from agents.news.enhanced_news_agent_duckdb import EnhancedNewsAgentDuckDB, CategoryClassifier

def get_setup_ids_with_director_dealings(db_path="data/sentiment_system.duckdb", limit=5):
    """Find setup IDs with 'Director Dealings' headlines"""
    try:
        conn = duckdb.connect(db_path)
        
        # Query RNS announcements for director dealings
        query = """
            SELECT 
                rns.setup_id,
                rns.ticker,
                rns.headline,
                rns.rns_date,
                rns.rns_time
            FROM rns_announcements rns
            WHERE LOWER(rns.headline) LIKE '%director%' OR LOWER(rns.headline) LIKE '%pdmr%'
            ORDER BY rns.rns_date DESC, rns.rns_time DESC
            LIMIT ?
        """
        
        results = conn.execute(query, [limit]).fetchdf()
        conn.close()
        
        if len(results) == 0:
            print("No RNS announcements with 'Director Dealings' found")
            return []
        
        print(f"Found {len(results)} RNS announcements with potential 'Director Dealings':")
        for i, row in results.iterrows():
            print(f"{i+1}. [{row['rns_date']}] {row['headline']} (Setup: {row['setup_id']})")
        
        return results['setup_id'].unique().tolist()
        
    except Exception as e:
        print(f"Error querying RNS data: {e}")
        return []

def test_news_agent_classification(headline):
    """Test the news agent's classification of a headline"""
    print(f"\nTesting classification of: '{headline}'")
    
    # Initialize the news agent
    agent = EnhancedNewsAgentDuckDB(
        db_path="data/sentiment_system.duckdb",
        mode="prediction",
        use_cached_models=True,
        local_files_only=True
    )
    
    # Classify the headline
    category = agent.classifier.classify_headline(headline)
    group = agent.classifier.category_counts
    
    print(f"Classification: '{category}'")
    print(f"Category counts: {group}")
    
    return category

def process_setup_with_news_agent(setup_id):
    """Process a setup with the news agent"""
    print(f"\nProcessing setup: {setup_id}")
    
    # Initialize the news agent
    agent = EnhancedNewsAgentDuckDB(
        db_path="data/sentiment_system.duckdb",
        mode="prediction",
        use_cached_models=True,
        local_files_only=True
    )
    
    try:
        # Retrieve news for the setup
        news_df = agent.retrieve_news_by_setup_id(setup_id)
        
        if len(news_df) == 0:
            print(f"No news found for setup {setup_id}")
            return
        
        print(f"Retrieved {len(news_df)} news items for setup {setup_id}")
        
        # Apply quality filtering
        filtered_news_df = agent._filter_quality_news(news_df)
        print(f"After quality filtering: {len(filtered_news_df)} news items")
        
        # Classify and group news
        news_groups = agent.classify_and_group_news(filtered_news_df)
        
        # Print group counts
        for group_name, items in news_groups.items():
            print(f"Group '{group_name}': {len(items)} items")
            
            # Print headlines in each group
            for i, item in enumerate(items):
                print(f"  {i+1}. {item['headline']} (Category: {item['category']})")
        
        # Check if any items were classified as "Director Dealings"
        director_dealings_items = []
        for group_name, items in news_groups.items():
            for item in items:
                if item['category'] == "Director Dealings":
                    director_dealings_items.append(item)
        
        if director_dealings_items:
            print(f"\nFound {len(director_dealings_items)} items classified as 'Director Dealings':")
            for i, item in enumerate(director_dealings_items):
                print(f"  {i+1}. {item['headline']} (Group: {CATEGORY_TO_GROUP.get(item['category'], 'unknown')})")
        else:
            print("\nNo items classified as 'Director Dealings'")
        
    except Exception as e:
        print(f"Error processing setup {setup_id}: {e}")
    finally:
        # Clean up
        agent.cleanup()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test news agent with director dealings")
    parser.add_argument("--db", default="data/sentiment_system.duckdb",
                        help="Path to DuckDB database (default: data/sentiment_system.duckdb)")
    parser.add_argument("--limit", type=int, default=5,
                        help="Limit the number of setups to test (default: 5)")
    parser.add_argument("--setup", default=None,
                        help="Specific setup ID to test (optional)")
    
    args = parser.parse_args()
    
    print("Testing news agent with 'Director Dealings' category...")
    
    if args.setup:
        # Test specific setup
        process_setup_with_news_agent(args.setup)
    else:
        # Find setups with director dealings
        setup_ids = get_setup_ids_with_director_dealings(args.db, args.limit)
        
        if not setup_ids:
            print("No setups found with 'Director Dealings'")
            return
        
        # Test each setup
        for setup_id in setup_ids:
            process_setup_with_news_agent(setup_id)
    
    # Test classification of specific headlines
    test_headlines = [
        "Director Dealings - Purchase of Shares",
        "Director Share Purchase",
        "PDMR Shareholding",
        "Directorate Change",
        "Board Changes"
    ]
    
    print("\nTesting classification of specific headlines:")
    for headline in test_headlines:
        test_news_agent_classification(headline)
    
    print("\nDone!")

if __name__ == "__main__":
    # Import here to avoid circular import
    from agents.news.news_categories import CATEGORY_TO_GROUP
    main() 