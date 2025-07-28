#!/usr/bin/env python3
"""
Simple Visualization Script for 3-Stage ML Pipeline Results

Creates visualizations for the corrected 3-stage ML pipeline results.

Usage:
    python simple_visualize.py --predictions data/predictions_corrected/final_predictions_*.csv --results data/results_table_corrected.csv
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class SimpleVisualizer:
    """Simple visualizer for 3-stage ML pipeline results"""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def load_predictions(self, file_path: str) -> pd.DataFrame:
        """Load predictions from CSV file"""
        logger.info(f"Loading predictions from: {file_path}")
        
        # Find the file if pattern is provided
        if '*' in file_path:
            files = glob.glob(file_path)
            if not files:
                raise ValueError(f"No files found matching pattern: {file_path}")
            file_path = sorted(files)[-1]  # Use most recent file
            logger.info(f"Using file: {file_path}")
        
        # Load predictions
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} predictions")
        
        return df
        
    def load_results(self, file_path: str) -> pd.DataFrame:
        """Load results table from CSV file"""
        logger.info(f"Loading results from: {file_path}")
        
        # Find the file if pattern is provided
        if '*' in file_path:
            files = glob.glob(file_path)
            if not files:
                raise ValueError(f"No files found matching pattern: {file_path}")
            file_path = sorted(files)[-1]  # Use most recent file
            logger.info(f"Using file: {file_path}")
        
        # Load results
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} results")
        
        return df
        
    def plot_prediction_distribution(self, predictions: pd.DataFrame):
        """Plot prediction distribution"""
        logger.info("Plotting prediction distribution")
        
        plt.figure(figsize=(10, 6))
        
        # Count predictions
        pred_counts = predictions['prediction'].value_counts().sort_index()
        
        # Create bar plot
        ax = sns.barplot(x=pred_counts.index, y=pred_counts.values)
        
        # Add value labels
        for i, count in enumerate(pred_counts.values):
            ax.text(i, count + 0.5, str(count), ha='center')
            
        # Add labels
        plt.title('Prediction Distribution')
        plt.xlabel('Prediction (-1: Negative, 0: Neutral, 1: Positive)')
        plt.ylabel('Count')
        
        # Save figure
        output_path = self.output_dir / 'prediction_distribution.png'
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved prediction distribution to {output_path}")
        
    def plot_confidence_distribution(self, predictions: pd.DataFrame):
        """Plot confidence distribution"""
        logger.info("Plotting confidence distribution")
        
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        sns.histplot(predictions['confidence'], bins=20, kde=True)
        
        # Add labels
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        
        # Save figure
        output_path = self.output_dir / 'confidence_distribution.png'
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved confidence distribution to {output_path}")
        
    def plot_confusion_matrix(self, results: pd.DataFrame):
        """Plot confusion matrix"""
        logger.info("Plotting confusion matrix")
        
        # Check if required columns exist
        if 'actual_label' not in results.columns or 'predicted_label_ML' not in results.columns:
            logger.warning("Missing required columns for confusion matrix")
            return
            
        # Drop missing values
        valid_results = results.dropna(subset=['actual_label', 'predicted_label_ML'])
        
        if len(valid_results) == 0:
            logger.warning("No valid data for confusion matrix")
            return
            
        # Create confusion matrix
        cm = confusion_matrix(valid_results['actual_label'], valid_results['predicted_label_ML'])
        
        plt.figure(figsize=(10, 8))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
        # Add labels
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save figure
        output_path = self.output_dir / 'confusion_matrix.png'
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved confusion matrix to {output_path}")
        
    def plot_confidence_by_class(self, predictions: pd.DataFrame):
        """Plot confidence by class"""
        logger.info("Plotting confidence by class")
        
        plt.figure(figsize=(10, 6))
        
        # Create box plot
        sns.boxplot(x='prediction', y='confidence', data=predictions)
        
        # Add labels
        plt.title('Confidence by Prediction Class')
        plt.xlabel('Prediction Class')
        plt.ylabel('Confidence Score')
        
        # Save figure
        output_path = self.output_dir / 'confidence_by_class.png'
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved confidence by class plot to {output_path}")
        
    def generate_performance_metrics(self, results: pd.DataFrame):
        """Generate performance metrics"""
        logger.info("Generating performance metrics")
        
        # Check if required columns exist
        if 'actual_label' not in results.columns or 'predicted_label_ML' not in results.columns:
            logger.warning("Missing required columns for performance metrics")
            return
            
        # Drop missing values
        valid_results = results.dropna(subset=['actual_label', 'predicted_label_ML'])
        
        if len(valid_results) == 0:
            logger.warning("No valid data for performance metrics")
            return
            
        # Calculate metrics
        accuracy = accuracy_score(valid_results['actual_label'], valid_results['predicted_label_ML'])
        precision = precision_score(valid_results['actual_label'], valid_results['predicted_label_ML'], average='macro')
        recall = recall_score(valid_results['actual_label'], valid_results['predicted_label_ML'], average='macro')
        f1 = f1_score(valid_results['actual_label'], valid_results['predicted_label_ML'], average='macro')
        
        # Create metrics report
        report = f"""
PERFORMANCE METRICS
==================

Accuracy:  {accuracy:.4f}
Precision: {precision:.4f}
Recall:    {recall:.4f}
F1 Score:  {f1:.4f}

Number of valid predictions: {len(valid_results)}
"""
        
        # Save report
        output_path = self.output_dir / 'performance_metrics.txt'
        with open(output_path, 'w') as f:
            f.write(report)
            
        logger.info(f"Saved performance metrics to {output_path}")
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
        
        # Create bar plot
        plt.bar(metrics.keys(), metrics.values())
        
        # Add value labels
        for i, (metric, value) in enumerate(metrics.items()):
            plt.text(i, value + 0.02, f'{value:.4f}', ha='center')
            
        # Add labels
        plt.title('Performance Metrics')
        plt.ylim(0, 1.0)
        
        # Save figure
        output_path = self.output_dir / 'performance_metrics.png'
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved performance metrics chart to {output_path}")
        
    def visualize_all(self, predictions_file: str, results_file: str):
        """Generate all visualizations"""
        logger.info("Generating all visualizations")
        
        # Load data
        predictions = self.load_predictions(predictions_file)
        results = self.load_results(results_file)
        
        # Generate visualizations
        self.plot_prediction_distribution(predictions)
        self.plot_confidence_distribution(predictions)
        self.plot_confusion_matrix(results)
        self.plot_confidence_by_class(predictions)
        self.generate_performance_metrics(results)
        
        logger.info("✅ Visualization completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Simple Visualization for 3-Stage ML Pipeline Results")
    parser.add_argument("--predictions", required=True,
                       help="Path to predictions CSV file")
    parser.add_argument("--results", required=True,
                       help="Path to results table CSV file")
    parser.add_argument("--output-dir", default="visualizations",
                       help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    try:
        # Initialize visualizer
        visualizer = SimpleVisualizer(args.output_dir)
        
        # Generate visualizations
        visualizer.visualize_all(args.predictions, args.results)
        
    except Exception as e:
        logger.error(f"❌ Visualization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 