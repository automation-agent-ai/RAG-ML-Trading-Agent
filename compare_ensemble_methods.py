#!/usr/bin/env python3
"""
Ensemble Methods Comparison

This script compares the performance of different ensemble methods:
1. Trained ensemble model (from the 3-stage ML pipeline)
2. Simple confidence-weighted majority voting (without training)

Usage:
    python compare_ensemble_methods.py --trained-ensemble ml/prediction/final_predictions_*.csv 
                                      --simple-ensemble ml/prediction/simple_ensemble_predictions.csv
                                      --results-table ml/prediction/results_table_final.csv
                                      --output-dir ml/analysis/ensemble_comparison
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_predictions(file_path):
    """Load predictions from a CSV file, handling glob patterns"""
    if '*' in file_path:
        files = glob.glob(file_path)
        if not files:
            logger.error(f"No files found matching pattern: {file_path}")
            return None
        file_path = max(files, key=os.path.getmtime)  # Get most recent file
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

def compare_ensemble_methods(trained_ensemble_file, simple_ensemble_file, results_table_file, output_dir):
    """Compare different ensemble methods"""
    # Load predictions
    trained_ensemble = load_predictions(trained_ensemble_file)
    simple_ensemble = load_predictions(simple_ensemble_file)
    results_table = load_predictions(results_table_file)
    
    if trained_ensemble is None or simple_ensemble is None or results_table is None:
        logger.error("Failed to load required files")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge predictions with actual labels
    merged = pd.merge(
        trained_ensemble[['setup_id', 'prediction', 'confidence']].rename(
            columns={'prediction': 'trained_prediction', 'confidence': 'trained_confidence'}
        ),
        simple_ensemble[['setup_id', 'prediction', 'confidence']].rename(
            columns={'prediction': 'simple_prediction', 'confidence': 'simple_confidence'}
        ),
        on='setup_id'
    )
    
    # Add actual labels
    merged = pd.merge(
        merged,
        results_table[['setup_id', 'actual_label']],
        on='setup_id',
        how='left'
    )
    
    # Calculate performance metrics
    metrics = {}
    
    # Filter to rows with actual labels
    valid_rows = merged.dropna(subset=['actual_label'])
    
    if len(valid_rows) == 0:
        logger.warning("No valid rows with actual labels found")
        return
    
    # Calculate metrics for trained ensemble
    metrics['trained'] = {
        'accuracy': accuracy_score(valid_rows['actual_label'], valid_rows['trained_prediction']),
        'precision_macro': precision_score(valid_rows['actual_label'], valid_rows['trained_prediction'], average='macro'),
        'recall_macro': recall_score(valid_rows['actual_label'], valid_rows['trained_prediction'], average='macro'),
        'f1_macro': f1_score(valid_rows['actual_label'], valid_rows['trained_prediction'], average='macro'),
        'precision_weighted': precision_score(valid_rows['actual_label'], valid_rows['trained_prediction'], average='weighted'),
        'recall_weighted': recall_score(valid_rows['actual_label'], valid_rows['trained_prediction'], average='weighted'),
        'f1_weighted': f1_score(valid_rows['actual_label'], valid_rows['trained_prediction'], average='weighted'),
    }
    
    # Calculate metrics for simple ensemble
    metrics['simple'] = {
        'accuracy': accuracy_score(valid_rows['actual_label'], valid_rows['simple_prediction']),
        'precision_macro': precision_score(valid_rows['actual_label'], valid_rows['simple_prediction'], average='macro'),
        'recall_macro': recall_score(valid_rows['actual_label'], valid_rows['simple_prediction'], average='macro'),
        'f1_macro': f1_score(valid_rows['actual_label'], valid_rows['simple_prediction'], average='macro'),
        'precision_weighted': precision_score(valid_rows['actual_label'], valid_rows['simple_prediction'], average='weighted'),
        'recall_weighted': recall_score(valid_rows['actual_label'], valid_rows['simple_prediction'], average='weighted'),
        'f1_weighted': f1_score(valid_rows['actual_label'], valid_rows['simple_prediction'], average='weighted'),
    }
    
    # Calculate agreement between methods
    agreement = (merged['trained_prediction'] == merged['simple_prediction']).mean()
    metrics['agreement'] = agreement
    
    # Generate visualizations
    
    # 1. Confusion matrices
    plt.figure(figsize=(16, 7))
    
    plt.subplot(1, 2, 1)
    cm_trained = confusion_matrix(valid_rows['actual_label'], valid_rows['trained_prediction'])
    sns.heatmap(cm_trained, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Trained Ensemble Confusion Matrix\nAccuracy: {metrics["trained"]["accuracy"]:.3f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.subplot(1, 2, 2)
    cm_simple = confusion_matrix(valid_rows['actual_label'], valid_rows['simple_prediction'])
    sns.heatmap(cm_simple, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Simple Ensemble Confusion Matrix\nAccuracy: {metrics["simple"]["accuracy"]:.3f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
    plt.close()
    
    # 2. Prediction distribution comparison
    plt.figure(figsize=(12, 6))
    
    # Prepare data for grouped bar chart
    labels = ['Negative (-1)', 'Neutral (0)', 'Positive (1)']
    trained_counts = [
        (merged['trained_prediction'] == -1).sum(),
        (merged['trained_prediction'] == 0).sum(),
        (merged['trained_prediction'] == 1).sum()
    ]
    simple_counts = [
        (merged['simple_prediction'] == -1).sum(),
        (merged['simple_prediction'] == 0).sum(),
        (merged['simple_prediction'] == 1).sum()
    ]
    actual_counts = [
        (valid_rows['actual_label'] == -1).sum(),
        (valid_rows['actual_label'] == 0).sum(),
        (valid_rows['actual_label'] == 1).sum()
    ]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, trained_counts, width, label='Trained Ensemble')
    rects2 = ax.bar(x, simple_counts, width, label='Simple Ensemble')
    rects3 = ax.bar(x + width, actual_counts, width, label='Actual Distribution')
    
    ax.set_ylabel('Count')
    ax.set_title('Prediction Distribution Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'))
    plt.close()
    
    # 3. Confidence comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(merged['trained_confidence'], bins=20)
    plt.title(f'Trained Ensemble Confidence\nMean: {merged["trained_confidence"].mean():.3f}')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    sns.histplot(merged['simple_confidence'], bins=20)
    plt.title(f'Simple Ensemble Confidence\nMean: {merged["simple_confidence"].mean():.3f}')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_comparison.png'))
    plt.close()
    
    # 4. Performance metrics comparison
    plt.figure(figsize=(12, 8))
    
    # Prepare data for bar chart
    metric_names = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1 (macro)', 
                   'Precision (weighted)', 'Recall (weighted)', 'F1 (weighted)']
    trained_values = [
        metrics['trained']['accuracy'],
        metrics['trained']['precision_macro'],
        metrics['trained']['recall_macro'],
        metrics['trained']['f1_macro'],
        metrics['trained']['precision_weighted'],
        metrics['trained']['recall_weighted'],
        metrics['trained']['f1_weighted']
    ]
    simple_values = [
        metrics['simple']['accuracy'],
        metrics['simple']['precision_macro'],
        metrics['simple']['recall_macro'],
        metrics['simple']['f1_macro'],
        metrics['simple']['precision_weighted'],
        metrics['simple']['recall_weighted'],
        metrics['simple']['f1_weighted']
    ]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, trained_values, width, label='Trained Ensemble')
    rects2 = ax.bar(x + width/2, simple_values, width, label='Simple Ensemble')
    
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_metrics.png'))
    plt.close()
    
    # 5. Agreement analysis
    plt.figure(figsize=(10, 6))
    
    # Create agreement matrix
    agreement_matrix = pd.crosstab(merged['trained_prediction'], merged['simple_prediction'])
    
    sns.heatmap(agreement_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Prediction Agreement Matrix\nAgreement Rate: {agreement:.1%}')
    plt.xlabel('Simple Ensemble Prediction')
    plt.ylabel('Trained Ensemble Prediction')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agreement_matrix.png'))
    plt.close()
    
    # Generate summary report
    with open(os.path.join(output_dir, 'comparison_summary.txt'), 'w') as f:
        f.write("ENSEMBLE METHODS COMPARISON SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write("PERFORMANCE METRICS\n")
        f.write("-"*30 + "\n")
        f.write(f"{'Metric':<20} {'Trained Ensemble':<20} {'Simple Ensemble':<20} {'Difference':<20}\n")
        f.write("-"*80 + "\n")
        
        for metric_name, trained_key in [
            ('Accuracy', 'accuracy'),
            ('Precision (macro)', 'precision_macro'),
            ('Recall (macro)', 'recall_macro'),
            ('F1 (macro)', 'f1_macro'),
            ('Precision (weighted)', 'precision_weighted'),
            ('Recall (weighted)', 'recall_weighted'),
            ('F1 (weighted)', 'f1_weighted')
        ]:
            trained_val = metrics['trained'][trained_key]
            simple_val = metrics['simple'][trained_key]
            diff = trained_val - simple_val
            diff_str = f"{diff:.4f} ({'better' if diff > 0 else 'worse'})"
            
            f.write(f"{metric_name:<20} {trained_val:.4f}{' ':<20} {simple_val:.4f}{' ':<20} {diff_str}\n")
        
        f.write("\n")
        
        f.write("PREDICTION DISTRIBUTION\n")
        f.write("-"*30 + "\n")
        f.write(f"{'Class':<10} {'Trained Ensemble':<20} {'Simple Ensemble':<20} {'Actual':<20}\n")
        f.write("-"*70 + "\n")
        
        for i, label in enumerate(['Negative (-1)', 'Neutral (0)', 'Positive (1)']):
            cls_val = i - 1  # Convert to -1, 0, 1
            trained_count = (merged['trained_prediction'] == cls_val).sum()
            trained_pct = trained_count / len(merged) * 100
            
            simple_count = (merged['simple_prediction'] == cls_val).sum()
            simple_pct = simple_count / len(merged) * 100
            
            actual_count = (valid_rows['actual_label'] == cls_val).sum()
            actual_pct = actual_count / len(valid_rows) * 100
            
            f.write(f"{label:<10} {trained_count} ({trained_pct:.1f}%){' ':<5} {simple_count} ({simple_pct:.1f}%){' ':<5} {actual_count} ({actual_pct:.1f}%)\n")
        
        f.write("\n")
        
        f.write("AGREEMENT ANALYSIS\n")
        f.write("-"*30 + "\n")
        f.write(f"Agreement rate: {agreement:.1%}\n\n")
        
        f.write("Agreement matrix:\n")
        f.write(str(agreement_matrix) + "\n\n")
        
        f.write("CONFIDENCE COMPARISON\n")
        f.write("-"*30 + "\n")
        f.write(f"Trained ensemble mean confidence: {merged['trained_confidence'].mean():.4f}\n")
        f.write(f"Simple ensemble mean confidence: {merged['simple_confidence'].mean():.4f}\n")
        f.write(f"Trained ensemble median confidence: {merged['trained_confidence'].median():.4f}\n")
        f.write(f"Simple ensemble median confidence: {merged['simple_confidence'].median():.4f}\n")
        
        f.write("\n")
        f.write("CONCLUSION\n")
        f.write("-"*30 + "\n")
        
        better_metrics = sum(1 for k in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 
                                        'precision_weighted', 'recall_weighted', 'f1_weighted'] 
                            if metrics['trained'][k] > metrics['simple'][k])
        
        if better_metrics >= 4:
            f.write("The trained ensemble model performs better overall.\n")
        elif better_metrics <= 3:
            f.write("The simple ensemble model performs better overall.\n")
        else:
            f.write("Both ensemble methods perform similarly overall.\n")
        
        # Highlight specific strengths
        if metrics['trained']['precision_weighted'] > metrics['simple']['precision_weighted']:
            f.write("The trained ensemble has better precision, which is important for your use case.\n")
        else:
            f.write("The simple ensemble has better precision, which is important for your use case.\n")
            
        f.write("\n")
        f.write("RECOMMENDATION\n")
        f.write("-"*30 + "\n")
        
        if metrics['trained']['precision_weighted'] > metrics['simple']['precision_weighted'] and better_metrics >= 4:
            f.write("Continue using the trained ensemble model as it provides better overall performance and precision.\n")
        elif metrics['simple']['precision_weighted'] > metrics['trained']['precision_weighted'] and better_metrics <= 3:
            f.write("Consider switching to the simple ensemble model as it provides better precision and comparable overall performance without requiring additional training.\n")
        else:
            f.write("Both methods have strengths. Consider using both and combining their predictions for maximum robustness.\n")
    
    logger.info(f"Comparison results saved to {output_dir}")
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Compare ensemble methods')
    parser.add_argument('--trained-ensemble', default='ml/prediction/final_predictions_*.csv', 
                      help='Path to trained ensemble predictions')
    parser.add_argument('--simple-ensemble', default='ml/prediction/simple_ensemble_predictions.csv',
                      help='Path to simple ensemble predictions')
    parser.add_argument('--results-table', default='ml/prediction/results_table_final.csv',
                      help='Path to results table with actual labels')
    parser.add_argument('--output-dir', default='ml/analysis/ensemble_comparison',
                      help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    # Compare ensemble methods
    metrics = compare_ensemble_methods(
        args.trained_ensemble, 
        args.simple_ensemble, 
        args.results_table, 
        args.output_dir
    )
    
    if metrics:
        # Print summary
        logger.info("Ensemble Methods Comparison Summary:")
        logger.info(f"Trained Ensemble Accuracy: {metrics['trained']['accuracy']:.4f}")
        logger.info(f"Simple Ensemble Accuracy: {metrics['simple']['accuracy']:.4f}")
        logger.info(f"Agreement Rate: {metrics['agreement']:.1%}")
        
        # Print recommendation
        better_metrics = sum(1 for k in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 
                                        'precision_weighted', 'recall_weighted', 'f1_weighted'] 
                            if metrics['trained'][k] > metrics['simple'][k])
        
        if better_metrics >= 4:
            logger.info("Recommendation: The trained ensemble model performs better overall.")
        elif better_metrics <= 3:
            logger.info("Recommendation: The simple ensemble model performs better overall.")
        else:
            logger.info("Recommendation: Both ensemble methods perform similarly overall.")

if __name__ == '__main__':
    main() 