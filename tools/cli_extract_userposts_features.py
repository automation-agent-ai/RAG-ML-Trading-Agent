#!/usr/bin/env python3
"""
CLI Tool: Extract UserPosts Features
=====================================

Command-line interface for extracting features from user posts using the Enhanced UserPosts Agent.
This is the production tool for feature extraction that writes to the feature storage system.

Usage:
    python cli_extract_userposts_features.py --setup-id KZG_2024-10-16
    python cli_extract_userposts_features.py --batch BLND_2024-09-19 HWDN_2024-07-09
    python cli_extract_userposts_features.py --all-setups
    python cli_extract_userposts_features.py --discover

Author: Enhanced UserPosts Agent
Date: 2025-01-06
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Add project root and agents directory to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "agents" / "userposts"))

from agents.userposts.enhanced_userposts_agent_complete import EnhancedUserPostsAgentComplete as EnhancedUserPostsAgent, UserPostsFeatureSchema


def create_default_features(setup_id: str) -> UserPostsFeatureSchema:
    """Create default features for setups without user posts"""
    return UserPostsFeatureSchema(
        setup_id=setup_id,
        post_count=0,
        unique_users=0,
        avg_sentiment=0.0,
        consensus_level='low',  # Changed from 'no_posts'
        consensus_topics=[],
        controversial_topics=[],
        rumor_intensity=0.0,
        trusted_user_sentiment=0.0,
        community_sentiment_score=0.0,
        bull_bear_ratio=1.0,
        contrarian_signal=False,
        relevance_score=0.0,
        recent_sentiment_shift='unknown',  # Changed from 'neutral'
        sentiment_distribution={'bullish': 0, 'bearish': 0, 'neutral': 0},  # Added proper structure
        engagement_score=0.0,
        coherence='low',  # Changed from 'no_posts'
        synthetic_post="No user posts found for this setup.",
        cot_explanation="No user posts available for analysis.",  # Added required field
        llm_model="default",
        extraction_timestamp=datetime.now().isoformat()  # Changed from timestamp float
    )


def extract_single_setup(
    setup_id: str,
    agent: EnhancedUserPostsAgent,
    query: Optional[str] = None,
    verbose: bool = False
) -> bool:
    """
    Extract features for a single setup
    
    Args:
        setup_id: Setup identifier
        agent: Enhanced UserPosts agent
        query: Optional custom query for reranking
        verbose: Whether to show detailed output
        
    Returns:
        True if successful, False otherwise
    """
    if verbose:
        print(f"\n📊 Processing setup: {setup_id}")
        print("-" * 50)
    
    start_time = time.time()
    
    try:
        # Check if setup has any posts
        has_posts = agent.check_setup_has_posts(setup_id)
        
        if not has_posts:
            # Create and store default features
            features = create_default_features(setup_id)
            agent.store_features(features)
            if verbose:
                print(f"ℹ️ No posts found for {setup_id}, using default features")
            else:
                print(f"ℹ️ {setup_id}: No posts found")
            return True
        
        # Extract features - this calls the real LLM and stores to database
        features = agent.process_setup(setup_id)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if verbose:
            print(f"✅ SUCCESS! Extracted {len(features.model_dump())} features in {processing_time:.2f}s")
            print(f"   Setup ID: {features.setup_id}")
            print(f"   Posts Analyzed: {features.post_count}")
            print(f"   Community Sentiment: {features.community_sentiment_score:.3f}")
            print(f"   Consensus Level: {features.consensus_level}")
            print(f"   Rumor Intensity: {features.rumor_intensity:.3f}")
            print(f"   LLM Model: {features.llm_model}")
            print(f"   Summary: {features.synthetic_post[:100]}...")
        else:
            print(f"✅ {setup_id}: {features.post_count} posts, sentiment {features.community_sentiment_score:.3f}")
        
        return True
            
    except Exception as e:
        print(f"❌ {setup_id}: Error - {e}")
        return False


def extract_batch_setups(
    setup_ids: List[str],
    agent: EnhancedUserPostsAgent,
    query: Optional[str] = None,
    verbose: bool = False
) -> dict:
    """
    Extract features for multiple setups
    
    Args:
        setup_ids: List of setup identifiers
        agent: Enhanced UserPosts agent
        query: Optional custom query for reranking
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
        
        success = extract_single_setup(setup_id, agent, None, verbose)
        
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


def discover_available_setups(agent: EnhancedUserPostsAgent, limit: int = 20) -> List[str]:
    """
    Discover available setups that have user posts
    
    Args:
        agent: Enhanced UserPosts agent
        limit: Maximum number of setups to return
        
    Returns:
        List of available setup IDs
    """
    print(f"\n🔍 DISCOVERING AVAILABLE SETUPS")
    print("=" * 60)
    
    try:
        # Get all data from LanceDB
        all_data = agent.table.to_pandas()
        setup_ids = all_data['setup_id'].unique().tolist()
        
        # Count posts per setup
        setup_counts = all_data['setup_id'].value_counts().to_dict()
        
        print(f"Found {len(setup_ids)} setups with user posts:")
        print()
        
        for setup_id in setup_ids[:limit]:
            count = setup_counts.get(setup_id, 0)
            print(f"  📊 {setup_id}: {count} posts")
        
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


def extract_all_setups(agent: EnhancedUserPostsAgent, max_setups: int = 50) -> dict:
    """
    Extract features for all available setups
    
    Args:
        agent: Enhanced UserPosts agent
        max_setups: Maximum number of setups to process
        
    Returns:
        Dictionary with results summary
    """
    print(f"\n🚀 EXTRACTING ALL AVAILABLE SETUPS")
    print("=" * 60)
    
    # First discover available setups
    available_setups = discover_available_setups(agent, limit=max_setups)
    
    if not available_setups:
        print("❌ No setups available for processing")
        return {'successful': 0, 'failed': 0, 'total_time': 0, 'setup_ids': []}
    
    # Limit to max_setups
    if len(available_setups) > max_setups:
        print(f"⚠️ Limiting to first {max_setups} setups (out of {len(available_setups)} available)")
        available_setups = available_setups[:max_setups]
    
    print(f"\n🔄 Processing {len(available_setups)} setups...")
    
    # Process all setups
    return extract_batch_setups(available_setups, agent, verbose=True)


def check_feature_storage(agent: EnhancedUserPostsAgent, setup_id: str):
    """
    Check if features are already stored for a setup
    
    Args:
        agent: Enhanced UserPosts agent
        setup_id: Setup identifier
    """
    print(f"\n🔍 CHECKING STORED FEATURES: {setup_id}")
    print("=" * 60)
    
    features = agent.get_stored_features(setup_id)
    
    if features:
        print(f"✅ Found stored features for {setup_id}")
        print(f"   Community Sentiment: {features['community_sentiment_score']:.3f}")
        print(f"   Consensus Level: {features['consensus_level']}")
        print(f"   Post Count: {features['post_count']}")
        print(f"   Extraction Time: {features['extraction_timestamp']}")
        print(f"   LLM Model: {features['llm_model']}")
        print(f"   Summary: {features['synthetic_post'][:100]}...")
    else:
        print(f"❌ No stored features found for {setup_id}")
        print(f"💡 To extract features: python {Path(__file__).name} --setup-id {setup_id}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Extract UserPosts features using Enhanced UserPosts Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract features for a single setup
  python cli_extract_userposts_features.py --setup-id KZG_2024-10-16
  
  # Extract features for multiple setups
  python cli_extract_userposts_features.py --batch BLND_2024-09-19 HWDN_2024-07-09
  
  # Extract features for all available setups
  python cli_extract_userposts_features.py --all-setups
  
  # Discover available setups
  python cli_extract_userposts_features.py --discover
  
  # Check if features are already stored
  python cli_extract_userposts_features.py --check KZG_2024-10-16
  
  # Use custom query for reranking
  python cli_extract_userposts_features.py --setup-id KZG_2024-10-16 --query "mining discussion sentiment"
        """
    )
    
    # Main operation modes
    parser.add_argument("--setup-id", help="Extract features for a specific setup ID")
    parser.add_argument("--batch", nargs="+", help="Extract features for multiple setup IDs")
    parser.add_argument("--all-setups", action="store_true", help="Extract features for all available setups")
    parser.add_argument("--discover", action="store_true", help="Discover available setups")
    parser.add_argument("--check", help="Check if features are already stored for a setup ID")
    
    # Options
    parser.add_argument("--query", help="Custom query for reranking posts")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--max-setups", type=int, default=50, help="Maximum setups to process with --all-setups")
    
    # Database configuration
    parser.add_argument("--db-path", default="data/sentiment_system.duckdb", help="Path to DuckDB database")
    parser.add_argument("--lancedb-dir", default="lancedb_store", help="LanceDB directory")
    parser.add_argument("--feature-storage", default="data/userposts_features.db", help="Feature storage database path")
    parser.add_argument("--cache-path", default="cache/userposts_cache.db", help="Cache database path")
    
    # Model configuration
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--max-retrieve", type=int, default=20, help="Maximum posts to retrieve")
    parser.add_argument("--max-llm-input", type=int, default=5, help="Maximum posts to send to LLM")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.setup_id, args.batch, args.all_setups, args.discover, args.check]):
        parser.print_help()
        sys.exit(1)
    
    # Initialize agent
    print(f"🔧 INITIALIZING ENHANCED USERPOSTS AGENT")
    print("=" * 60)
    print(f"  Database: {args.db_path}")
    print(f"  LanceDB: {args.lancedb_dir}")
    print(f"  Feature Storage: {args.feature_storage}")
    print(f"  Cache: {args.cache_path}")
    print(f"  LLM Model: {args.llm_model}")
    print(f"  Max Retrieve: {args.max_retrieve}")
    print(f"  Max LLM Input: {args.max_llm_input}")
    
    try:
        agent = EnhancedUserPostsAgent(
            db_path=args.db_path,
            lancedb_dir=args.lancedb_dir,
            table_name="userposts_embeddings",
            llm_model=args.llm_model,
            max_group_size=args.max_llm_input
        )
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure OpenAI API key is set: $env:OPENAI_API_KEY = 'your-key'")
        print("  2. Check if DuckDB database exists at specified path")
        print("  3. Check if LanceDB store exists with userposts embeddings")
        print("  4. Verify all required dependencies are installed")
        sys.exit(1)
    
    try:
        # Execute the requested operation
        if args.discover:
            discover_available_setups(agent, limit=args.max_setups)
        
        elif args.check:
            check_feature_storage(agent, args.check)
        
        elif args.setup_id:
            success = extract_single_setup(args.setup_id, agent, args.query, args.verbose)
            sys.exit(0 if success else 1)
        
        elif args.batch:
            results = extract_batch_setups(args.batch, agent, args.query, args.verbose)
            sys.exit(0 if results['failed'] == 0 else 1)
        
        elif args.all_setups:
            results = extract_all_setups(agent, args.max_setups)
            sys.exit(0 if results['failed'] == 0 else 1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Clean up
        if 'agent' in locals():
            agent.cleanup()


if __name__ == "__main__":
    main() 