#!/usr/bin/env python3
"""
Visualization Script for Individual Model Predictions

This script generates consistent visualizations for each model in the text_ml, 
financial_ml, and ensemble directories. It creates the same set of visualizations 
for each model to enable easy comparison.

Usage:
    python visualize_model_predictions.py --input-dir ml/prediction --output-dir ml/analysis/visualizations
"""

import os
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_dirs(output_dir):
    """Create output directories if they don't exist"""
    model_types = ['text_ml', 'financial_ml', 'ensemble']
    for model_type in model_types:
        os.makedirs(os.path.join(output_dir, model_type), exist_ok=True)
    return

def load_actual_labels(results_file):
    """Load actual labels from results table"""
    if not os.path.exists(results_file):
        logger.warning(f"Results file {results_file} not found. Visualizations requiring actual labels will be skipped.")
        return None
    
    results = pd.read_csv(results_file)
    # Create a mapping from setup_id to actual_label
    if 'setup_id' in results.columns and 'actual_label' in results.columns:
        return dict(zip(results['setup_id'], results['actual_label']))
    else:
        logger.warning("Results file doesn't contain required columns (setup_id, actual_label)")
        return None

def visualize_model_predictions(input_dir, output_dir, results_file=None):
    """Generate visualizations for each model prediction file"""
    model_types = ['text_ml', 'financial_ml', 'ensemble']
    actual_labels_map = load_actual_labels(results_file) if results_file else None
    
    for model_type in model_types:
        model_dir = os.path.join(input_dir, model_type)
        if not os.path.exists(model_dir):
            logger.warning(f"Directory {model_dir} not found. Skipping.")
            continue
        
        # Find all prediction CSV files
        prediction_files = glob.glob(os.path.join(model_dir, f"*_predictions.csv"))
        
        for pred_file in prediction_files:
            model_name = os.path.basename(pred_file).replace('_predictions.csv', '')
            logger.info(f"Visualizing predictions for {model_name}")
            
            # Load predictions
            try:
                predictions = pd.read_csv(pred_file)
                
                # Create output directory for this model
                model_output_dir = os.path.join(output_dir, model_type)
                os.makedirs(model_output_dir, exist_ok=True)
                
                # 1. Prediction Distribution
                plt.figure(figsize=(10, 6))
                pred_counts = predictions['prediction'].value_counts().sort_index()
                pred_counts.plot(kind='bar')
                plt.title(f'{model_name} - Prediction Distribution')
                plt.xlabel('Class')
                plt.ylabel('Count')
                plt.savefig(os.path.join(model_output_dir, f"{model_name}_prediction_distribution.png"))
                plt.close()
                
                # 2. Confidence Distribution
                plt.figure(figsize=(10, 6))
                sns.histplot(predictions['confidence'], bins=20)
                plt.title(f'{model_name} - Confidence Distribution')
                plt.xlabel('Confidence')
                plt.ylabel('Count')
                plt.savefig(os.path.join(model_output_dir, f"{model_name}_confidence_distribution.png"))
                plt.close()
                
                # 3. Confidence by Class
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='prediction', y='confidence', data=predictions)
                plt.title(f'{model_name} - Confidence by Prediction Class')
                plt.xlabel('Predicted Class')
                plt.ylabel('Confidence')
                plt.savefig(os.path.join(model_output_dir, f"{model_name}_confidence_by_class.png"))
                plt.close()
                
                # 4. Probability Distribution (if available)
                if all(col in predictions.columns for col in ['prob_negative', 'prob_neutral', 'prob_positive']):
                    plt.figure(figsize=(12, 8))
                    
                    # Melt the dataframe for seaborn
                    prob_cols = ['prob_negative', 'prob_neutral', 'prob_positive']
                    melted_probs = pd.melt(predictions, value_vars=prob_cols, 
                                          var_name='Probability Type', value_name='Probability')
                    
                    # Map to cleaner names
                    melted_probs['Probability Type'] = melted_probs['Probability Type'].map({
                        'prob_negative': 'Negative (-1)', 
                        'prob_neutral': 'Neutral (0)', 
                        'prob_positive': 'Positive (1)'
                    })
                    
                    sns.boxplot(x='Probability Type', y='Probability', data=melted_probs)
                    plt.title(f'{model_name} - Class Probability Distributions')
                    plt.savefig(os.path.join(model_output_dir, f"{model_name}_probability_distribution.png"))
                    plt.close()
                
                # 5. Confusion Matrix (if actual labels available)
                if actual_labels_map and 'setup_id' in predictions.columns:
                    # Add actual labels to predictions
                    predictions['actual_label'] = predictions['setup_id'].map(actual_labels_map)
                    valid_preds = predictions.dropna(subset=['actual_label'])
                    
                    if len(valid_preds) > 0:
                        cm = confusion_matrix(valid_preds['actual_label'], valid_preds['prediction'])
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                        plt.title(f'{model_name} - Confusion Matrix')
                        plt.xlabel('Predicted')
                        plt.ylabel('Actual')
                        plt.savefig(os.path.join(model_output_dir, f"{model_name}_confusion_matrix.png"))
                        plt.close()
                        
                        # 6. Classification Report
                        report = classification_report(valid_preds['actual_label'], 
                                                      valid_preds['prediction'],
                                                      output_dict=True)
                        
                        # Save as text file
                        with open(os.path.join(model_output_dir, f"{model_name}_classification_report.txt"), 'w') as f:
                            f.write(f"Classification Report for {model_name}\n")
                            f.write("="*50 + "\n\n")
                            
                            # Overall metrics
                            f.write("Overall Metrics:\n")
                            f.write(f"Accuracy: {report['accuracy']:.4f}\n")
                            f.write(f"Macro Avg Precision: {report['macro avg']['precision']:.4f}\n")
                            f.write(f"Macro Avg Recall: {report['macro avg']['recall']:.4f}\n")
                            f.write(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}\n")
                            f.write(f"Weighted Avg Precision: {report['weighted avg']['precision']:.4f}\n")
                            f.write(f"Weighted Avg Recall: {report['weighted avg']['recall']:.4f}\n")
                            f.write(f"Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.4f}\n\n")
                            
                            # Class metrics
                            f.write("Class Metrics:\n")
                            for cls in sorted([c for c in report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]):
                                f.write(f"Class {cls}:\n")
                                f.write(f"  Precision: {report[cls]['precision']:.4f}\n")
                                f.write(f"  Recall: {report[cls]['recall']:.4f}\n")
                                f.write(f"  F1-Score: {report[cls]['f1-score']:.4f}\n")
                                f.write(f"  Support: {report[cls]['support']}\n\n")
                
                # 7. Summary statistics
                with open(os.path.join(model_output_dir, f"{model_name}_summary.txt"), 'w') as f:
                    f.write(f"Summary Statistics for {model_name}\n")
                    f.write("="*50 + "\n\n")
                    
                    # Prediction distribution
                    f.write("Prediction Distribution:\n")
                    for cls, count in pred_counts.items():
                        f.write(f"Class {cls}: {count} ({count/len(predictions)*100:.1f}%)\n")
                    f.write("\n")
                    
                    # Confidence statistics
                    f.write("Confidence Statistics:\n")
                    f.write(f"Mean: {predictions['confidence'].mean():.4f}\n")
                    f.write(f"Median: {predictions['confidence'].median():.4f}\n")
                    f.write(f"Min: {predictions['confidence'].min():.4f}\n")
                    f.write(f"Max: {predictions['confidence'].max():.4f}\n")
                    f.write(f"Std Dev: {predictions['confidence'].std():.4f}\n\n")
                    
                    # Confidence by class
                    f.write("Confidence by Class:\n")
                    for cls in sorted(predictions['prediction'].unique()):
                        cls_conf = predictions[predictions['prediction'] == cls]['confidence']
                        f.write(f"Class {cls}:\n")
                        f.write(f"  Mean: {cls_conf.mean():.4f}\n")
                        f.write(f"  Median: {cls_conf.median():.4f}\n")
                        f.write(f"  Count: {len(cls_conf)}\n\n")
                
                logger.info(f"Visualizations for {model_name} saved to {model_output_dir}")
                
            except Exception as e:
                logger.error(f"Error processing {pred_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate visualizations for model predictions')
    parser.add_argument('--input-dir', default='ml/prediction', 
                      help='Directory containing model prediction folders')
    parser.add_argument('--output-dir', default='ml/analysis/visualizations',
                      help='Directory to save visualizations')
    parser.add_argument('--results-file', default='ml/prediction/results_table_final.csv',
                      help='Path to results table with actual labels')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    setup_dirs(args.output_dir)
    
    # Generate visualizations
    visualize_model_predictions(args.input_dir, args.output_dir, args.results_file)
    
    logger.info(f"All visualizations completed and saved to {args.output_dir}")

if __name__ == '__main__':
    main() 