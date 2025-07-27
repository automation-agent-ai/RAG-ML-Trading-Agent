#!/usr/bin/env python3
"""
Debug script to show what RNS news items get filtered out during quality filtering
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import lancedb
from setup_validator_duckdb import SetupValidatorDuckDB
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_filtering_for_setup(setup_id: str):
    """Show filtering details for a specific setup"""
    
    # Connect to LanceDB
    db = lancedb.connect("../lancedb_store")
    table = db.open_table("news_embeddings")
    
    # Get all data for setup
    all_data = table.to_pandas()
    results = all_data[
        (all_data['setup_id'] == setup_id) & 
        (all_data['source_type'] == 'rns_announcement')
    ].copy()
    
    if len(results) == 0:
        print(f"❌ No RNS announcements found for setup_id: {setup_id}")
        return
    
    print(f"\n🔍 RNS Filtering Analysis for {setup_id}")
    print("=" * 60)
    
    original_count = len(results)
    print(f"📊 Original count: {original_count} items")
    
    # Show all original headlines
    print(f"\n📰 ALL Original Headlines:")
    for i, row in results.iterrows():
        headline = row['headline']
        date = row.get('rns_date', 'Unknown')
        print(f"  {len(headline):2d} chars: [{date}] {headline}")
    
    # Apply filtering step by step
    print(f"\n🔧 FILTERING PROCESS:")
    
    # Step 1: Filter out very short headlines
    step1_before = len(results)
    short_headlines = results[results['headline'].str.len() < 10]
    results_step1 = results[results['headline'].str.len() >= 10]
    step1_after = len(results_step1)
    
    print(f"\n  Step 1 - Remove short headlines (< 10 chars):")
    print(f"    Before: {step1_before}, After: {step1_after}, Removed: {step1_before - step1_after}")
    
    if len(short_headlines) > 0:
        print(f"    📋 FILTERED OUT (too short):")
        for _, row in short_headlines.iterrows():
            headline = row['headline']
            print(f"      {len(headline):2d} chars: {headline}")
    
    # Step 2: Remove duplicates
    step2_before = len(results_step1)
    seen_headlines = set()
    filtered_indices = []
    duplicates = []
    
    for idx, row in results_step1.iterrows():
        headline = row['headline'].lower().strip()
        
        if headline not in seen_headlines:
            seen_headlines.add(headline)
            filtered_indices.append(idx)
        else:
            duplicates.append(row['headline'])
    
    results_final = results_step1.loc[filtered_indices]
    step2_after = len(results_final)
    
    print(f"\n  Step 2 - Remove duplicates:")
    print(f"    Before: {step2_before}, After: {step2_after}, Removed: {step2_before - step2_after}")
    
    if duplicates:
        print(f"    📋 FILTERED OUT (duplicates):")
        for dup in duplicates:
            print(f"      {dup}")
    
    # Final summary
    print(f"\n✅ FINAL RESULT:")
    print(f"   Original: {original_count} → Final: {step2_after} ({original_count - step2_after} filtered out)")
    
    if len(results_final) > 0:
        print(f"\n📰 FINAL Headlines (kept):")
        for _, row in results_final.iterrows():
            headline = row['headline']
            date = row.get('rns_date', 'Unknown')
            print(f"  ✅ [{date}] {headline}")
    
    print("=" * 60)

def main():
    """Main function to analyze filtering across multiple setups"""
    
    # Load confirmed setups
    validator = SetupValidatorDuckDB()
    confirmed_setups = validator.get_confirmed_setup_ids()
    
    print(f"🔍 Analyzing RNS filtering across {len(confirmed_setups)} confirmed setups...")
    
    # Check some example setups
    example_setups = [
        "HWDN_2024-07-23",  # We know this one has news
        "BLND_2025-04-22",  # We know this one has news
        "WTB_2025-01-06",   # We know this one has news
        "KZG_2024-12-16"    # We know this one has news
    ]
    
    for setup_id in example_setups:
        if setup_id in confirmed_setups:
            debug_filtering_for_setup(setup_id)
        else:
            print(f"⚠️  Setup {setup_id} not in confirmed setups")

if __name__ == "__main__":
    main() 