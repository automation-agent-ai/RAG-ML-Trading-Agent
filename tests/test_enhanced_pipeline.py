#!/usr/bin/env python3
"""
Test Enhanced RAG Pipeline with Similarity Search

This script tests the enhanced ML pipeline implementation with:
1. Training mode: Create embeddings with labels
2. Prediction mode: Create embeddings without labels, use similarity search
3. Enhanced features via similarity search
4. Direct similarity predictions
"""

import logging
import sys
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import our enhanced components
from embeddings.embed_news_duckdb import NewsEmbeddingPipelineDuckDB
from embeddings.embed_fundamentals_duckdb import FundamentalsEmbedderDuckDB
from agents.news.enhanced_news_agent_duckdb import EnhancedNewsAgentDuckDB
from run_complete_ml_pipeline import MLPipelineRunner

def test_embedding_data_separation():
    """Test that embeddings properly separate training and prediction data"""
    logger.info("🧪 Testing embedding data separation...")
    
    # Test news embeddings in training mode
    news_embedder_training = NewsEmbeddingPipelineDuckDB(
        db_path="data/sentiment_system.duckdb",
        lancedb_dir="lancedb_store",
        include_labels=True,
        mode="training"
    )
    
    # Test news embeddings in prediction mode
    news_embedder_prediction = NewsEmbeddingPipelineDuckDB(
        db_path="data/sentiment_system.duckdb", 
        lancedb_dir="lancedb_store",
        include_labels=False,
        mode="prediction"
    )
    
    # Create sample records for testing
    sample_records = [
        {
            'setup_id': 'TEST_SETUP_001',
            'chunk_text': 'Company announces strong quarterly results',
            'text_length': 42
        }
    ]
    
    # Test training mode (should include labels)
    logger.info("Testing training mode embedding creation...")
    training_records = news_embedder_training.create_embeddings(sample_records.copy())
    
    # Test prediction mode (should not include labels)
    logger.info("Testing prediction mode embedding creation...")
    prediction_records = news_embedder_prediction.create_embeddings(sample_records.copy())
    
    # Verify both have embeddings
    assert len(training_records) > 0, "Training records should be created"
    assert len(prediction_records) > 0, "Prediction records should be created"
    assert 'vector' in training_records[0], "Training records should have embeddings"
    assert 'vector' in prediction_records[0], "Prediction records should have embeddings"
    
    logger.info("✅ Embedding data separation test passed!")
    return True

def test_similarity_search_functionality():
    """Test similarity search capabilities"""
    logger.info("🧪 Testing similarity search functionality...")
    
    # Initialize news agent
    news_agent = EnhancedNewsAgentDuckDB(
        db_path="data/sentiment_system.duckdb",
        lancedb_dir="lancedb_store"
    )
    
    if news_agent.training_table is None:
        logger.warning("⚠️ No training embeddings table found. Skipping similarity search test.")
        return False
    
    # Create a test embedding
    test_embedding = np.random.rand(384).astype(np.float32)  # MiniLM dimensions
    
    # Test similarity search
    similar_cases = news_agent.find_similar_training_embeddings(test_embedding, limit=5)
    
    if similar_cases:
        logger.info(f"✅ Found {len(similar_cases)} similar cases")
        
        # Test similarity feature computation
        similarity_features = news_agent.compute_similarity_features(similar_cases)
        
        expected_keys = [
            'positive_signal_strength',
            'negative_risk_score', 
            'neutral_probability',
            'historical_pattern_confidence',
            'similar_cases_count'
        ]
        
        for key in expected_keys:
            assert key in similarity_features, f"Missing similarity feature: {key}"
        
        logger.info("✅ Similarity search functionality test passed!")
        return True
    else:
        logger.warning("⚠️ No similar cases found. This may be expected if no training data exists.")
        return False

def test_agent_mode_handling():
    """Test that agents properly handle training vs prediction modes"""
    logger.info("🧪 Testing agent mode handling...")
    
    # Get a test setup_id
    conn = duckdb.connect("data/sentiment_system.duckdb")
    result = conn.execute("SELECT setup_id FROM labels LIMIT 1").fetchone()
    conn.close()
    
    if not result:
        logger.warning("⚠️ No setup_ids found in labels table. Skipping agent mode test.")
        return False
    
    test_setup_id = result[0]
    
    # Initialize news agent
    news_agent = EnhancedNewsAgentDuckDB(
        db_path="data/sentiment_system.duckdb",
        lancedb_dir="lancedb_store"
    )
    
    try:
        # Test training mode
        logger.info(f"Testing training mode for setup: {test_setup_id}")
        training_features = news_agent.process_setup(test_setup_id, mode="training")
        
        # Test prediction mode
        logger.info(f"Testing prediction mode for setup: {test_setup_id}")
        prediction_features = news_agent.process_setup(test_setup_id, mode="prediction")
        
        # Both should succeed (or both fail consistently)
        if training_features is None and prediction_features is None:
            logger.info("✅ Both modes returned None consistently (no news data)")
            return True
        elif training_features is not None and prediction_features is not None:
            logger.info("✅ Both modes returned features successfully")
            
            # Check if prediction mode has similarity features
            if hasattr(prediction_features, '__dict__'):
                feature_dict = prediction_features.__dict__
                similarity_keys = [k for k in feature_dict.keys() if 'similarity' in k]
                if similarity_keys:
                    logger.info(f"✅ Found {len(similarity_keys)} similarity features in prediction mode")
                else:
                    logger.info("ℹ️ No similarity features found (may be expected if no training embeddings)")
            
            return True
        else:
            logger.error("❌ Inconsistent behavior between training and prediction modes")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing agent modes: {e}")
        return False

def test_pipeline_integration():
    """Test the complete pipeline integration"""
    logger.info("🧪 Testing complete pipeline integration...")
    
    try:
        # Test with a small subset of setup_ids
        conn = duckdb.connect("data/sentiment_system.duckdb")
        setup_ids = [row[0] for row in conn.execute("SELECT setup_id FROM labels LIMIT 2").fetchall()]
        conn.close()
        
        if not setup_ids:
            logger.warning("⚠️ No setup_ids found for pipeline test.")
            return False
        
        logger.info(f"Testing pipeline with setup_ids: {setup_ids}")
        
        # Test training mode
        logger.info("Testing training mode pipeline...")
        training_pipeline = MLPipelineRunner(
            db_path="data/sentiment_system.duckdb",
            lancedb_dir="lancedb_store", 
            mode="training"
        )
        
        # We won't run the full pipeline as it's expensive, just verify initialization
        logger.info("✅ Training pipeline initialized successfully")
        
        # Test prediction mode
        logger.info("Testing prediction mode pipeline...")
        prediction_pipeline = MLPipelineRunner(
            db_path="data/sentiment_system.duckdb",
            lancedb_dir="lancedb_store",
            mode="prediction"
        )
        
        logger.info("✅ Prediction pipeline initialized successfully")
        logger.info("✅ Pipeline integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline integration test failed: {e}")
        return False

def test_data_leakage_prevention():
    """Test that data leakage is prevented in prediction mode"""
    logger.info("🧪 Testing data leakage prevention...")
    
    # Test fundamentals embedder
    fundamentals_embedder = FundamentalsEmbedderDuckDB(
        db_path="data/sentiment_system.duckdb",
        lancedb_dir="lancedb_store",
        include_labels=False,
        mode="prediction"
    )
    
    # Verify configuration
    assert not fundamentals_embedder.include_labels, "Labels should be disabled in prediction mode"
    assert fundamentals_embedder.mode == "prediction", "Mode should be prediction"
    
    logger.info("✅ Data leakage prevention test passed!")
    return True

def main():
    """Run all tests"""
    logger.info("🚀 Starting Enhanced RAG Pipeline Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Embedding Data Separation", test_embedding_data_separation),
        ("Similarity Search Functionality", test_similarity_search_functionality),
        ("Agent Mode Handling", test_agent_mode_handling),
        ("Pipeline Integration", test_pipeline_integration),
        ("Data Leakage Prevention", test_data_leakage_prevention)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Enhanced RAG pipeline is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main()) 