#!/usr/bin/env python3
"""
CLI Tool: Extract News Features
===============================

Command-line interface for extracting features from RNS news using the Enhanced News Agent.
This is the production tool for news feature extraction that writes to the feature storage system.

Usage:
    python cli_extract_news_features.py --setup-id AAPL_2024Q2
    python cli_extract_news_features.py --batch BLND_2024-09-19 HWDN_2024-07-09
    python cli_extract_news_features.py --all-setups
    python cli_extract_news_features.py --discover

Author: Enhanced News Agent
Date: 2025-01-06
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add project root and agents directory to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "agents" / "news"))

from agents.news.enhanced_news_agent_duckdb import EnhancedNewsAgentDuckDB, NewsFeatureSchema


def create_default_features(setup_id: str) -> NewsFeatureSchema:
    """Create default features for setups without news"""
    return NewsFeatureSchema(
        setup_id=setup_id,
        count_financial_results=0,
        count_corporate_actions=0,
        count_governance=0,
        count_corporate_events=0,
        count_other_signals=0,
        sentiment_score_financial=0.0,
        sentiment_score_corporate=0.0,
        sentiment_score_governance=0.0,
        sentiment_score_events=0.0,
        sentiment_score_other=0.0,
        headline_spin_financial='neutral',
        headline_spin_corporate='neutral',
        headline_spin_governance='neutral',
        headline_spin_events='neutral',
        headline_spin_other='neutral',
        max_severity_financial=0,
        max_severity_corporate=0,
        max_severity_governance=0,
        max_severity_events=0,
        max_severity_other=0,
        profit_warning_present=False,
        capital_raise_present=False,
        board_change_present=False,
        contract_award_present=False,
        merger_acquisition_present=False,
        broker_rec_present=False,
        credit_rating_present=False,
        cot_explanation_news_grouped="No news announcements found for this setup.",
        llm_model="default",
        extraction_timestamp=time.time()
    )


def extract_single_setup(
    setup_id: str,
    agent: EnhancedNewsAgentDuckDB,
    verbose: bool = False
) -> bool:
    """
    Extract news features for a single setup
    
    Args:
        setup_id: Setup identifier
        agent: Enhanced News agent
        verbose: Whether to show detailed output
        
    Returns:
        True if successful, False otherwise
    """
    if verbose:
        print(f"\n📰 Processing news for setup: {setup_id}")
        print("-" * 50)
    
    start_time = time.time()
    
    try:
        # Extract features - this calls the real LLM and stores to database
        features = agent.process_setup(setup_id)
        
        # If no features were extracted, create default ones
        if not features:
            features = create_default_features(setup_id)
            agent.store_features(features)  # Store default features
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        total_news = (features.count_financial_results + features.count_corporate_actions + 
                     features.count_governance + features.count_corporate_events + 
                     features.count_other_signals)
        
        if verbose:
            print(f"✅ SUCCESS! Extracted features for {total_news} news items in {processing_time:.2f}s")
            print(f"   Setup ID: {features.setup_id}")
            print(f"   Financial Results: {features.count_financial_results} items")
            print(f"   Corporate Actions: {features.count_corporate_actions} items")
            print(f"   Governance: {features.count_governance} items")
            print(f"   Corporate Events: {features.count_corporate_events} items")
            print(f"   Other Signals: {features.count_other_signals} items")
            print(f"   LLM Model: {features.llm_model}")
            print(f"   Global Summary: {features.cot_explanation_news_grouped[:150]}...")
        else:
            print(f"✅ {setup_id}: {total_news} news items processed")
        
        return True
            
    except Exception as e:
        print(f"❌ {setup_id}: Error - {e}")
        return False


def extract_batch_setups(
    setup_ids: List[str],
    agent: EnhancedNewsAgentDuckDB,
    verbose: bool = False
) -> dict:
    """
    Extract news features for multiple setups
    
    Args:
        setup_ids: List of setup identifiers
        agent: Enhanced News agent
        verbose: Whether to show detailed output
        
    Returns:
        Dictionary with results summary
    """
    print(f"\n🔄 BATCH PROCESSING: {len(setup_ids)} setups")
    print("=" * 60)
    
    start_time = time.time()
    successful = 0
    failed = 0
    
    # Process each setup
    for i, setup_id in enumerate(setup_ids, 1):
        if verbose:
            print(f"\n[{i}/{len(setup_ids)}] Processing {setup_id}")
        
        success = extract_single_setup(setup_id, agent, verbose)
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Summary
    print(f"\n📈 BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {successful/(successful+failed)*100:.1f}%")
    print(f"⏱️ Total Time: {total_time:.2f}s")
    print(f"📊 Average Time: {total_time/len(setup_ids):.2f}s per setup")
    
    return {
        'successful': successful,
        'failed': failed,
        'total_time': total_time,
        'setup_ids': setup_ids
    }


def discover_available_setups(agent: EnhancedNewsAgentDuckDB, limit: int = 20) -> List[str]:
    """
    Discover available setups that have RNS news
    
    Args:
        agent: Enhanced News agent
        limit: Maximum number of setups to return
        
    Returns:
        List of available setup IDs
    """
    print(f"\n🔍 DISCOVERING AVAILABLE SETUPS WITH NEWS")
    print("=" * 60)
    
    try:
        # Get all RNS announcements data from LanceDB
        all_data = agent.table.to_pandas()
        rns_data = all_data[all_data['source_type'] == 'rns_announcement']
        
        setup_ids = rns_data['setup_id'].unique().tolist()
        
        # Count news per setup
        setup_counts = rns_data['setup_id'].value_counts().to_dict()
        
        print(f"Found {len(setup_ids)} setups with RNS news:")
        print()
        
        for setup_id in setup_ids[:limit]:
            count = setup_counts.get(setup_id, 0)
            print(f"  📰 {setup_id}: {count} RNS announcements")
        
        if len(setup_ids) > limit:
            print(f"  ... and {len(setup_ids) - limit} more")
        
        print(f"\n💡 To extract features for a specific setup:")
        print(f"   python {Path(__file__).name} --setup-id {setup_ids[0]}")
        print(f"\n💡 To extract features for multiple setups:")
        print(f"   python {Path(__file__).name} --batch {' '.join(setup_ids[:3])}")
        
        return setup_ids
        
    except Exception as e:
        print(f"❌ Error discovering setups: {e}")
        return []


def extract_all_setups(agent: EnhancedNewsAgentDuckDB, max_setups: int = 50) -> dict:
    """
    Extract features for all available setups
    
    Args:
        agent: Enhanced News agent
        max_setups: Maximum number of setups to process
        
    Returns:
        Dictionary with results summary
    """
    print(f"\n🚀 EXTRACTING ALL AVAILABLE SETUPS (max {max_setups})")
    print("=" * 60)
    
    # Discover all setups
    all_setup_ids = discover_available_setups(agent, max_setups)
    
    if not all_setup_ids:
        print("❌ No setups found with news data")
        return {'successful': 0, 'failed': 0, 'total_time': 0, 'setup_ids': []}
    
    # Limit to max_setups
    setup_ids_to_process = all_setup_ids[:max_setups]
    
    print(f"\n🔄 Processing {len(setup_ids_to_process)} setups...")
    
    # Process in batch
    return extract_batch_setups(setup_ids_to_process, agent, verbose=False)


def check_feature_storage(agent: EnhancedNewsAgentDuckDB, setup_id: str):
    """Check if features are already stored for a setup"""
    try:
        stored_features = agent.get_stored_features(setup_id)
        
        if stored_features:
            print(f"\n✅ STORED FEATURES FOUND for {setup_id}:")
            print(f"   Extraction Time: {stored_features.get('extraction_timestamp', 'Unknown')}")
            print(f"   Model Used: {stored_features.get('llm_model', 'Unknown')}")
            print(f"   Total News Items: {stored_features.get('count_financial_results', 0) + stored_features.get('count_corporate_actions', 0) + stored_features.get('count_governance', 0) + stored_features.get('count_corporate_events', 0) + stored_features.get('count_other_signals', 0)}")
            
            # Show group breakdown
            groups = [
                ('Financial Results', stored_features.get('count_financial_results', 0)),
                ('Corporate Actions', stored_features.get('count_corporate_actions', 0)),
                ('Governance', stored_features.get('count_governance', 0)),
                ('Corporate Events', stored_features.get('count_corporate_events', 0)),
                ('Other Signals', stored_features.get('count_other_signals', 0))
            ]
            
            for group_name, count in groups:
                if count > 0:
                    print(f"   {group_name}: {count} items")
            
            return True
        else:
            print(f"\n❌ No stored features found for {setup_id}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking stored features: {e}")
        return False


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Extract RNS news features using Enhanced News Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single setup
  python cli_extract_news_features.py --setup-id AAPL_2024Q2

  # Process multiple setups
  python cli_extract_news_features.py --batch BLND_2024-09-19 HWDN_2024-07-09 KZG_2024-10-16

  # Process all available setups  
  python cli_extract_news_features.py --all-setups --max-setups 20

  # Discover available setups
  python cli_extract_news_features.py --discover

  # Check stored features
  python cli_extract_news_features.py --check AAPL_2024Q2
        """
    )
    
    # Command options (mutually exclusive)
    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument('--setup-id', type=str, help='Process single setup ID')
    command_group.add_argument('--batch', nargs='+', help='Process multiple setup IDs')
    command_group.add_argument('--all-setups', action='store_true', help='Process all available setups')
    command_group.add_argument('--discover', action='store_true', help='Discover available setups')
    command_group.add_argument('--check', type=str, help='Check stored features for setup ID')
    
    # Options
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--max-setups', type=int, default=50, help='Maximum setups to process (for --all-setups)')
    parser.add_argument('--db-path', type=str, default="../data/sentiment_system.duckdb", help='Path to DuckDB database')
    parser.add_argument('--lancedb-dir', type=str, default="../lancedb_store", help='LanceDB directory')
    
    args = parser.parse_args()
    
    # Header
    print("🔧 Enhanced News Feature Extraction CLI")
    print("=" * 60)
    
    # Check environment
    import os
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Initialize agent
    try:
        print(f"🔌 Initializing Enhanced News Agent...")
        print(f"   Database: {args.db_path}")
        print(f"   LanceDB: {args.lancedb_dir}")
        
        agent = EnhancedNewsAgentDuckDB(
            db_path=args.db_path,
            lancedb_dir=args.lancedb_dir
        )
        
        print("✅ Agent initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        sys.exit(1)
    
    try:
        # Execute command
        if args.setup_id:
            extract_single_setup(args.setup_id, agent, args.verbose)
            
        elif args.batch:
            extract_batch_setups(args.batch, agent, args.verbose)
            
        elif args.all_setups:
            extract_all_setups(agent, args.max_setups)
            
        elif args.discover:
            discover_available_setups(agent)
            
        elif args.check:
            check_feature_storage(agent, args.check)
        
        print(f"\n📊 Classification Statistics:")
        stats = agent.get_classification_stats()
        if stats['total_classifications'] > 0:
            print(f"   Classifications: {stats['total_classifications']}")
            print(f"   Categories: {list(stats['category_distribution'].keys())}")
            print(f"   Groups: {stats['group_distribution']}")
        else:
            print("   No classifications performed")
            
    except KeyboardInterrupt:
        print(f"\n⚡ Interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        
    finally:
        # Cleanup
        try:
            agent.cleanup()
        except:
            pass
        
        print("\n🏁 Done")


if __name__ == "__main__":
    main() 