#!/usr/bin/env python3
"""
Complete ML Pipeline Workflow

This script demonstrates the complete end-to-end workflow from 
feature extraction to ML predictions using your existing components.
"""
import sys
import logging
from pathlib import Path

# Add parent directory to path to access ml module
sys.path.append('..')

from run_complete_ml_pipeline import CompletePipeline
from ml.multiclass_predictor import MultiClassPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_complete_workflow(setup_ids=None, limit=10):
    """
    Run the complete ML workflow
    
    Args:
        setup_ids: List of setup IDs to process. If None, uses first 'limit' setups
        limit: Number of setups to process if setup_ids is None
    
    Returns:
        Tuple of (pipeline_results, trained_model, sample_prediction)
    """
    
    print("🚀 STARTING COMPLETE ML WORKFLOW")
    print("=" * 50)
    
    # Step 1: Initialize pipeline
    logger.info("Step 1: Initializing pipeline...")
    pipeline = CompletePipeline(
        db_path='data/sentiment_system.duckdb',
        lancedb_dir='data/lancedb_store'
    )
    
    # Step 2: Get setup IDs if not provided
    if setup_ids is None:
        import duckdb
        logger.info("Step 2: Getting sample setup IDs...")
        conn = duckdb.connect('data/sentiment_system.duckdb')
        setup_ids = [row[0] for row in conn.execute(
            f"SELECT setup_id FROM setups WHERE outperformance_10d IS NOT NULL LIMIT {limit}"
        ).fetchall()]
        conn.close()
        logger.info(f"Found {len(setup_ids)} setups for processing")
    
    print(f"📊 Processing {len(setup_ids)} setups: {setup_ids[:3]}{'...' if len(setup_ids) > 3 else ''}")
    
    # Step 3: Extract features and create ML tables
    logger.info("Step 3: Running feature extraction and merging...")
    print("\n🔧 FEATURE EXTRACTION & MERGING")
    print("-" * 40)
    
    try:
        results = pipeline.run_complete_pipeline(setup_ids, mode='training')
        
        # Display results
        print(f"✅ Pipeline completed in {results['duration_seconds']:.1f}s")
        if 'text' in results['ml_features_results']:
            text_info = results['ml_features_results']['text']
            print(f"📝 Text features: {text_info['feature_count']} features, {text_info['row_count']} rows")
        
        if 'financial' in results['ml_features_results']:
            fin_info = results['ml_features_results']['financial']
            print(f"💰 Financial features: {fin_info['feature_count']} features, {fin_info['row_count']} rows")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return None, None, None
    
    # Step 4: Train ML model
    logger.info("Step 4: Training ML model...")
    print("\n🎯 MACHINE LEARNING TRAINING")
    print("-" * 40)
    
    try:
        predictor = MultiClassPredictor('data/sentiment_system.duckdb')
        model = predictor.train_quick_model()
        print("✅ ML model trained successfully")
        
    except Exception as e:
        logger.error(f"ML training failed: {e}")
        return results, None, None
    
    # Step 5: Make sample predictions
    logger.info("Step 5: Making sample predictions...")
    print("\n🔮 MAKING PREDICTIONS")
    print("-" * 40)
    
    try:
        # Predict on first setup
        sample_setup = setup_ids[0]
        prediction = predictor.predict_setup(sample_setup)
        
        print(f"Sample Prediction ({sample_setup}):")
        print(f"  Class: {prediction[0]}")
        print(f"  Confidence: {prediction[1]:.3f}")
        print(f"  Probabilities:")
        for class_name, prob in prediction[2]['probabilities'].items():
            print(f"    {class_name}: {prob:.3f}")
        
        # Try batch predictions on first 3 setups
        if len(setup_ids) >= 3:
            print(f"\nBatch Predictions (first 3 setups):")
            batch_predictions = predictor.batch_predict(setup_ids[:3])
            for setup_id, pred in batch_predictions.items():
                print(f"  {setup_id}: Class {pred[0]} ({pred[1]:.3f} confidence)")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        prediction = None
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎉 WORKFLOW COMPLETE!")
    print("=" * 50)
    print(f"✅ Duration: {results['duration_seconds']:.1f}s")
    print(f"✅ Setups processed: {len(setup_ids)}")
    print(f"✅ Features extracted and merged")
    print(f"✅ ML model trained")
    print(f"✅ Predictions generated")
    
    return results, model, prediction

def run_step_by_step_demo():
    """
    Demonstrate each step of the pipeline individually
    """
    print("🔬 STEP-BY-STEP PIPELINE DEMONSTRATION")
    print("=" * 50)
    
    # Initialize
    pipeline = CompletePipeline()
    
    # Get sample setups
    import duckdb
    conn = duckdb.connect('data/sentiment_system.duckdb')
    setup_ids = [row[0] for row in conn.execute(
        "SELECT setup_id FROM setups LIMIT 3"
    ).fetchall()]
    conn.close()
    
    print(f"Demo setups: {setup_ids}")
    
    # Step 1: Feature extraction
    print("\n📊 Step 1: Feature Extraction")
    extraction_results = pipeline.extract_features(setup_ids)
    for domain, success in extraction_results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {domain}")
    
    # Step 2: Feature merging
    print("\n🔧 Step 2: Feature Merging")
    ml_results = pipeline.create_ml_features(setup_ids, mode='training')
    for feature_type, info in ml_results.items():
        print(f"  ✅ {feature_type}: {info['feature_count']} features, {info['row_count']} rows")
    
    # Step 3: ML Training
    print("\n🎯 Step 3: ML Training")
    predictor = MultiClassPredictor('data/sentiment_system.duckdb')
    model = predictor.train_quick_model()
    print("  ✅ Model trained")
    
    # Step 4: Prediction
    print("\n🔮 Step 4: Prediction")
    prediction = predictor.predict_setup(setup_ids[0])
    print(f"  ✅ Prediction for {setup_ids[0]}: Class {prediction[0]} ({prediction[1]:.3f})")
    
    print("\n🎉 Step-by-step demo complete!")

def main():
    """Main function with options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete ML Pipeline Workflow')
    parser.add_argument('--demo', action='store_true', 
                       help='Run step-by-step demonstration')
    parser.add_argument('--setup-ids', nargs='+', 
                       help='Specific setup IDs to process')
    parser.add_argument('--limit', type=int, default=10,
                       help='Number of setups to process (default: 10)')
    
    args = parser.parse_args()
    
    if args.demo:
        run_step_by_step_demo()
    else:
        run_complete_workflow(
            setup_ids=args.setup_ids,
            limit=args.limit
        )

if __name__ == "__main__":
    main() 