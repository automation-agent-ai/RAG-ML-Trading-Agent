#!/usr/bin/env python3
"""
Simple Majority Vote Ensemble Implementation

This script implements a simple confidence-weighted majority voting ensemble
that doesn't require training an additional model. It weights each model's 
prediction by its confidence score.

Usage:
    python simple_ensemble_voting.py --input-dir ml/prediction --output-file ml/prediction/simple_ensemble_predictions.csv
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_predictions(input_dir):
    """Load predictions from all models in text_ml and financial_ml directories"""
    model_predictions = {}
    model_types = ['text_ml', 'financial_ml']
    
    for model_type in model_types:
        model_dir = os.path.join(input_dir, model_type)
        if not os.path.exists(model_dir):
            logger.warning(f"Directory {model_dir} not found. Skipping.")
            continue
        
        # Find all prediction CSV files
        prediction_files = glob.glob(os.path.join(model_dir, f"*_predictions.csv"))
        
        for pred_file in prediction_files:
            model_name = os.path.basename(pred_file).replace('_predictions.csv', '')
            logger.info(f"Loading predictions from {model_name}")
            
            try:
                predictions = pd.read_csv(pred_file)
                if 'setup_id' in predictions.columns and 'prediction' in predictions.columns and 'confidence' in predictions.columns:
                    model_predictions[model_name] = predictions
                else:
                    logger.warning(f"File {pred_file} doesn't have required columns. Skipping.")
            except Exception as e:
                logger.error(f"Error loading {pred_file}: {e}")
    
    return model_predictions

def simple_confidence_weighted_voting(model_predictions):
    """Implement simple confidence-weighted majority voting"""
    if not model_predictions:
        logger.error("No valid model predictions found.")
        return None
    
    # Get list of all setup_ids
    all_setup_ids = set()
    for model_name, preds in model_predictions.items():
        all_setup_ids.update(preds['setup_id'].tolist())
    
    logger.info(f"Found {len(all_setup_ids)} unique setup IDs across all models")
    
    # Initialize result dataframe
    results = pd.DataFrame({'setup_id': list(all_setup_ids)})
    results.set_index('setup_id', inplace=True)
    
    # Initialize arrays for weighted voting
    weighted_votes = {setup_id: {-1: 0.0, 0: 0.0, 1: 0.0} for setup_id in all_setup_ids}
    total_confidence = {setup_id: 0.0 for setup_id in all_setup_ids}
    
    # Collect weighted votes from each model
    for model_name, preds in model_predictions.items():
        for _, row in preds.iterrows():
            setup_id = row['setup_id']
            prediction = row['prediction']
            confidence = row['confidence']
            
            # Add weighted vote
            weighted_votes[setup_id][prediction] += confidence
            total_confidence[setup_id] += confidence
    
    # Calculate final predictions
    final_predictions = []
    final_confidences = []
    final_outperformances = []
    
    # Mapping from prediction classes to outperformance values
    class_to_outperformance = {-1: -5.0, 0: 0.0, 1: 5.0}  # Approximate mapping
    
    for setup_id in results.index:
        # Get class with highest weighted vote
        votes = weighted_votes[setup_id]
        
        if total_confidence[setup_id] > 0:
            # Normalize votes by total confidence
            normalized_votes = {cls: vote/total_confidence[setup_id] for cls, vote in votes.items()}
            
            # Calculate weighted prediction value
            weighted_prediction = sum(cls * weight for cls, weight in normalized_votes.items())
            
            # Determine final class
            if weighted_prediction >= 0.33:
                final_pred = 1    # Positive
            elif weighted_prediction <= -0.33:
                final_pred = -1   # Negative
            else:
                final_pred = 0    # Neutral
                
            # Calculate confidence as ratio of winning class votes to total
            winning_class_votes = max(votes.values())
            final_conf = winning_class_votes / total_confidence[setup_id]
            
            # Calculate weighted outperformance
            weighted_outperformance = sum(class_to_outperformance[cls] * weight 
                                         for cls, weight in normalized_votes.items())
        else:
            # Default values if no predictions available
            final_pred = 0
            final_conf = 0.33
            weighted_outperformance = 0.0
        
        final_predictions.append(final_pred)
        final_confidences.append(final_conf)
        final_outperformances.append(weighted_outperformance)
    
    # Add results to dataframe
    results['prediction'] = final_predictions
    results['confidence'] = final_confidences
    results['predicted_outperformance_10d'] = final_outperformances
    
    # Add detailed voting information
    for cls in [-1, 0, 1]:
        results[f'votes_{cls}'] = [weighted_votes[setup_id][cls] for setup_id in results.index]
    
    results['total_confidence'] = [total_confidence[setup_id] for setup_id in results.index]
    
    # Reset index to make setup_id a column
    results = results.reset_index()
    
    return results

def generate_summary(results, output_file):
    """Generate summary statistics and save to a text file"""
    summary_file = output_file.replace('.csv', '_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("SIMPLE CONFIDENCE-WEIGHTED ENSEMBLE SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Prediction distribution
        f.write("PREDICTION DISTRIBUTION\n")
        f.write("-"*30 + "\n")
        pred_counts = results['prediction'].value_counts().sort_index()
        for pred, count in pred_counts.items():
            pct = count / len(results) * 100
            label = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}[pred]
            f.write(f"{label} ({pred}): {count} predictions ({pct:.1f}%)\n")
        f.write("\n")
        
        # Confidence statistics
        f.write("CONFIDENCE STATISTICS\n")
        f.write("-"*30 + "\n")
        f.write(f"Mean confidence: {results['confidence'].mean():.3f}\n")
        f.write(f"Median confidence: {results['confidence'].median():.3f}\n")
        f.write(f"Min confidence: {results['confidence'].min():.3f}\n")
        f.write(f"Max confidence: {results['confidence'].max():.3f}\n\n")
        
        # Outperformance statistics
        f.write("PREDICTED OUTPERFORMANCE STATISTICS\n")
        f.write("-"*30 + "\n")
        f.write(f"Mean predicted outperformance: {results['predicted_outperformance_10d'].mean():.3f}%\n")
        f.write(f"Median predicted outperformance: {results['predicted_outperformance_10d'].median():.3f}%\n")
        f.write(f"Min predicted outperformance: {results['predicted_outperformance_10d'].min():.3f}%\n")
        f.write(f"Max predicted outperformance: {results['predicted_outperformance_10d'].max():.3f}%\n\n")
        
        # Voting statistics
        f.write("VOTING STATISTICS\n")
        f.write("-"*30 + "\n")
        f.write(f"Mean total confidence: {results['total_confidence'].mean():.3f}\n")
        f.write(f"Mean negative votes: {results['votes_-1'].mean():.3f}\n")
        f.write(f"Mean neutral votes: {results['votes_0'].mean():.3f}\n")
        f.write(f"Mean positive votes: {results['votes_1'].mean():.3f}\n\n")
        
        f.write("="*50 + "\n")
    
    logger.info(f"Summary saved to {summary_file}")
    return summary_file

def main():
    parser = argparse.ArgumentParser(description='Simple confidence-weighted majority voting ensemble')
    parser.add_argument('--input-dir', default='ml/prediction', 
                      help='Directory containing model prediction folders')
    parser.add_argument('--output-file', default='ml/prediction/simple_ensemble_predictions.csv',
                      help='Path to save ensemble predictions')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Load model predictions
    model_predictions = load_model_predictions(args.input_dir)
    logger.info(f"Loaded predictions from {len(model_predictions)} models")
    
    # Generate ensemble predictions
    results = simple_confidence_weighted_voting(model_predictions)
    
    if results is not None:
        # Save results
        results.to_csv(args.output_file, index=False)
        logger.info(f"Ensemble predictions saved to {args.output_file}")
        
        # Generate summary
        summary_file = generate_summary(results, args.output_file)
        
        # Print summary
        logger.info(f"Generated {len(results)} ensemble predictions")
        logger.info(f"Prediction distribution: {results['prediction'].value_counts().to_dict()}")
        logger.info(f"Mean confidence: {results['confidence'].mean():.3f}")
        logger.info(f"Mean predicted outperformance: {results['predicted_outperformance_10d'].mean():.3f}%")
    else:
        logger.error("Failed to generate ensemble predictions")

if __name__ == '__main__':
    main() 