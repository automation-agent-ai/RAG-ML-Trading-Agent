#!/usr/bin/env python3
"""
YOUR PRODUCTION USAGE: Enhanced UserPosts Agent
================================================

This is EXACTLY how you'll use the Enhanced UserPosts Agent in your 
production trading system. Copy this code into your own applications.

Author: Your Production System
Date: 2025-01-06
"""

import sys
from pathlib import Path

# Add the scripts directory to your path
sys.path.append('scripts')  # Adjust path as needed

from enhanced_userposts_agent import EnhancedUserPostsAgent

def extract_features_for_trading(setup_id: str):
    """
    PRODUCTION FUNCTION: Extract features for a single trading setup
    
    This is the main function you'll call in your trading system.
    """
    # Initialize the agent (do this once, reuse the agent)
    agent = EnhancedUserPostsAgent()
    
    # Extract all 19 features using real LLM calls
    features = agent.process_setup(setup_id)
    
    if features:
        # You now have ALL 19 features synchronized with feature_plan.md
        return {
            'setup_id': features.setup_id,
            'community_sentiment_score': features.community_sentiment_score,
            'consensus_level': features.consensus_level,
            'rumor_intensity': features.rumor_intensity,
            'trusted_user_sentiment': features.trusted_user_sentiment,
            'bull_bear_ratio': features.bull_bear_ratio,
            'relevance_score': features.relevance_score,
            'engagement_score': features.engagement_score,
            'contrarian_signal': features.contrarian_signal,
            'recent_sentiment_shift': features.recent_sentiment_shift,
            'coherence': features.coherence,
            'consensus_topics': features.consensus_topics,
            'controversial_topics': features.controversial_topics,
            'sentiment_distribution': features.sentiment_distribution,
            'unique_users': features.unique_users,
            'post_count': features.post_count,
            'avg_sentiment': features.avg_sentiment,
            'outperformance_pred': features.outperformance_pred,
            'synthetic_post': features.synthetic_post,
            'cot_explanation': features.cot_explanation,
            'extraction_timestamp': features.extraction_timestamp,
            'llm_model': features.llm_model
        }
    else:
        return None

def batch_extract_features(setup_ids: list):
    """
    PRODUCTION FUNCTION: Extract features for multiple setups efficiently
    
    Use this for processing multiple setups at once.
    """
    agent = EnhancedUserPostsAgent()
    
    # Batch process all setups
    results = agent.batch_process_setups(setup_ids)
    
    # Convert to simple dictionary format
    processed_results = {}
    for setup_id, features in results.items():
        if features:
            processed_results[setup_id] = extract_feature_dict(features)
        else:
            processed_results[setup_id] = None
    
    return processed_results

def extract_feature_dict(features):
    """Helper function to convert features to dictionary"""
    return {
        'setup_id': features.setup_id,
        'community_sentiment_score': features.community_sentiment_score,
        'consensus_level': features.consensus_level,
        'rumor_intensity': features.rumor_intensity,
        'relevance_score': features.relevance_score,
        'post_count': features.post_count,
        # Add all other fields as needed
    }

def generate_trading_signals(features_dict):
    """
    PRODUCTION FUNCTION: Generate trading signals from extracted features
    
    This shows how to use the features for actual trading decisions.
    """
    signals = []
    
    # Sentiment-based signals
    sentiment = features_dict['community_sentiment_score']
    if sentiment > 0.3:
        signals.append(('BULLISH_SENTIMENT', sentiment))
    elif sentiment < -0.3:
        signals.append(('BEARISH_SENTIMENT', sentiment))
    
    # Consensus-based signals
    if features_dict['consensus_level'] == 'high' and features_dict['coherence'] == 'high':
        signals.append(('HIGH_CONVICTION', features_dict['relevance_score']))
    
    # Contrarian signals
    if features_dict['contrarian_signal']:
        signals.append(('CONTRARIAN_OPPORTUNITY', features_dict['relevance_score']))
    
    # Rumor-based signals
    if features_dict['rumor_intensity'] > 0.7:
        signals.append(('HIGH_RUMOR_ACTIVITY', features_dict['rumor_intensity']))
    
    return signals

def main_production_example():
    """
    MAIN PRODUCTION EXAMPLE: This is how you'd use it in your trading system
    """
    print("🚀 PRODUCTION EXAMPLE: Enhanced UserPosts Agent")
    print("=" * 60)
    
    # Example 1: Single setup analysis
    print("\n📊 Example 1: Single Setup Analysis")
    setup_id = "KZG_2024-10-16"
    features = extract_features_for_trading(setup_id)
    
    if features:
        print(f"✅ Extracted features for {setup_id}")
        print(f"   Sentiment: {features['community_sentiment_score']:.3f}")
        print(f"   Consensus: {features['consensus_level']}")
        print(f"   Rumor Intensity: {features['rumor_intensity']:.3f}")
        
        # Generate trading signals
        signals = generate_trading_signals(features)
        print(f"   Trading Signals: {len(signals)} generated")
        for signal_type, strength in signals:
            print(f"     - {signal_type}: {strength:.3f}")
    
    # Example 2: Batch processing
    print("\n📊 Example 2: Batch Processing")
    setup_ids = ["BLND_2024-09-19", "HWDN_2024-07-09"]
    batch_results = batch_extract_features(setup_ids)
    
    print(f"✅ Processed {len(batch_results)} setups")
    for setup_id, features in batch_results.items():
        if features:
            sentiment = features['community_sentiment_score']
            print(f"   {setup_id}: sentiment {sentiment:.3f}")
    
    print("\n🎯 READY FOR PRODUCTION!")
    print("Copy the functions above into your trading system.")

if __name__ == "__main__":
    main_production_example() 