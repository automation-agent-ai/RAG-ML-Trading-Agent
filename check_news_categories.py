#!/usr/bin/env python3
"""
Check News Categories

This script checks how the news agent handles different news categories,
particularly focusing on "Director Dealings" which is causing errors.
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
from agents.news.enhanced_news_agent_duckdb import CategoryClassifier
from agents.news.news_categories import RNS_CATEGORIES, CATEGORY_TO_GROUP

def check_category_mapping():
    """Check if 'Director Dealings' is properly mapped in CATEGORY_TO_GROUP"""
    print("\n1. Checking category mapping...")
    if "Director Dealings" in RNS_CATEGORIES:
        print(f"✅ 'Director Dealings' is in RNS_CATEGORIES")
    else:
        print(f"❌ 'Director Dealings' is NOT in RNS_CATEGORIES")
    
    group = CATEGORY_TO_GROUP.get("Director Dealings")
    if group:
        print(f"✅ 'Director Dealings' is mapped to group: '{group}'")
    else:
        print(f"❌ 'Director Dealings' is NOT mapped in CATEGORY_TO_GROUP")

def test_classification():
    """Test the classification of 'Director Dealings' headlines"""
    print("\n2. Testing classification...")
    
    # Create a dummy OpenAI client for testing
    class DummyOpenAIClient:
        def __init__(self):
            self.chat = self

        class Completions:
            def create(self, model, messages, temperature, max_tokens):
                class Response:
                    def __init__(self, content):
                        self.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': content})})]
                
                # Return "Director Dealings" for any message containing "director"
                message = messages[0]["content"].lower()
                if "director" in message:
                    return Response("Director Dealings")
                return Response("Investment Update")  # Default fallback
        
        def __init__(self):
            self.completions = self.Completions()
    
    # Initialize classifier with dummy client
    dummy_client = DummyOpenAIClient()
    classifier = CategoryClassifier(dummy_client)
    
    # Test headlines
    test_headlines = [
        "Director Dealings - Purchase of Shares",
        "Director Share Purchase",
        "PDMR Share Transaction",
        "Board Director Acquires Shares",
        "CEO Purchases Company Stock"
    ]
    
    for headline in test_headlines:
        category = classifier._fast_pattern_match(headline)
        if category:
            method = "fast pattern match"
        else:
            category = classifier._fuzzy_match(headline)
            if category:
                method = "fuzzy match"
            else:
                category = classifier._llm_classify(headline)
                method = "LLM classification"
        
        group = CATEGORY_TO_GROUP.get(category, "unknown")
        print(f"'{headline}' → '{category}' (via {method}) → group: '{group}'")

def check_rns_data(db_path="data/sentiment_system.duckdb", setup_id="BGO_2025-01-20"):
    """Check the actual RNS data for the given setup_id"""
    print(f"\n3. Checking RNS data for setup_id: {setup_id}...")
    
    try:
        conn = duckdb.connect(db_path)
        
        # Query RNS announcements
        query = """
            SELECT 
                rns.setup_id,
                rns.ticker,
                rns.headline,
                rns.rns_date,
                rns.rns_time
            FROM rns_announcements rns
            WHERE rns.setup_id = ?
                AND rns.headline IS NOT NULL
            ORDER BY rns.rns_date DESC, rns.rns_time DESC
        """
        
        results = conn.execute(query, [setup_id]).fetchdf()
        conn.close()
        
        if len(results) == 0:
            print(f"No RNS announcements found for setup_id: {setup_id}")
            return
        
        print(f"Found {len(results)} RNS announcements:")
        for i, row in results.iterrows():
            print(f"{i+1}. [{row['rns_date']}] {row['headline']}")
        
    except Exception as e:
        print(f"Error querying RNS data: {e}")

def check_news_features(db_path="data/sentiment_system.duckdb", setup_id="BGO_2025-01-20"):
    """Check if there are any news features stored for the setup_id"""
    print(f"\n4. Checking news features for setup_id: {setup_id}...")
    
    try:
        conn = duckdb.connect(db_path)
        
        # Query news features
        query = "SELECT * FROM news_features WHERE setup_id = ?"
        
        results = conn.execute(query, [setup_id]).fetchdf()
        conn.close()
        
        if len(results) == 0:
            print(f"No news features found for setup_id: {setup_id}")
            return
        
        print(f"Found news features:")
        
        # Print governance-related features
        print("\nGovernance features:")
        governance_cols = [col for col in results.columns if 'governance' in col]
        for col in governance_cols:
            print(f"  {col}: {results[col].iloc[0]}")
        
    except Exception as e:
        print(f"Error querying news features: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Check news category handling")
    parser.add_argument("setup_id", nargs="?", default="BGO_2025-01-20", 
                        help="Setup ID to check (default: BGO_2025-01-20)")
    parser.add_argument("--db", default="data/sentiment_system.duckdb",
                        help="Path to DuckDB database (default: data/sentiment_system.duckdb)")
    
    args = parser.parse_args()
    
    print(f"Checking 'Director Dealings' category handling for setup_id: {args.setup_id}")
    
    # 1. Check if the category is properly mapped
    check_category_mapping()
    
    # 2. Test classification of director dealings headlines
    test_classification()
    
    # 3. Check actual RNS data for the specified setup
    check_rns_data(args.db, args.setup_id)
    
    # 4. Check if there are any news features stored
    check_news_features(args.db, args.setup_id)
    
    print("\nDone!")

if __name__ == "__main__":
    main() 