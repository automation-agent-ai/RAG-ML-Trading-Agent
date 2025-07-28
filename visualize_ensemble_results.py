#!/usr/bin/env python3
"""
Visualize Ensemble Results

This script generates visualizations for ensemble prediction results.

Usage:
    python visualize_ensemble_results.py --input ensemble_predictions.csv --output-dir visualizations
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResultsVisualizer:
    """Class for visualizing ensemble results"""
    
    def __init__(
        self,
        output_dir: str = "visualizations"
    ):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def load_predictions(self, file_path: str) -> pd.DataFrame:
        """
        Load predictions from CSV
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Loading predictions from {file_path}")
        df = pd.read_csv(file_path)
        
        # Basic info
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        return df
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: List[str],
        output_path: str,
        title: str = 'Confusion Matrix'
    ) -> None:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            classes: Class labels
            output_path: Path to save the plot
            title: Plot title
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Convert to percentage
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot confusion matrix
        sns.heatmap(
            cm_percent,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes
        )
        
        # Add labels and title
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved confusion matrix to {output_path}")
    
    def plot_class_distribution(
        self,
        df: pd.DataFrame,
        label_col: str,
        output_path: str,
        title: str = 'Class Distribution'
    ) -> None:
        """
        Plot class distribution
        
        Args:
            df: DataFrame with labels
            label_col: Column name for labels
            output_path: Path to save the plot
            title: Plot title
        """
        # Count classes
        class_counts = df[label_col].value_counts().sort_index()
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot class distribution
        ax = sns.barplot(
            x=class_counts.index,
            y=class_counts.values
        )
        
        # Add value labels
        for i, count in enumerate(class_counts.values):
            ax.text(i, count + 5, str(count), ha='center')
        
        # Add labels and title
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title(title)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved class distribution to {output_path}")
    
    def generate_performance_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: str,
        model_name: str = 'Ensemble'
    ) -> Dict[str, float]:
        """
        Generate performance report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_path: Path to save the report
            model_name: Model name
            
        Returns:
            Dictionary with performance metrics
        """
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Get classification report
        report = classification_report(y_true, y_pred)
        
        # Create report
        with open(output_path, 'w') as f:
            f.write(f"Performance Report for {model_name}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision (weighted): {precision:.4f}\n")
            f.write(f"Recall (weighted): {recall:.4f}\n")
            f.write(f"F1 Score (weighted): {f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        
        logger.info(f"Saved performance report to {output_path}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def visualize_results(
        self,
        predictions_file: str,
        output_dir: str = None
    ) -> Dict[str, Any]:
        """
        Visualize ensemble results
        
        Args:
            predictions_file: Path to predictions CSV
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary with paths to visualizations
        """
        # Load predictions
        df = self.load_predictions(predictions_file)
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Check required columns
        required_cols = ['true_label', 'ensemble_prediction']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return {}
        
        # Get true and predicted labels
        y_true = df['true_label'].values
        y_pred = df['ensemble_prediction'].values
        
        # Get unique classes
        classes = sorted(list(set(np.concatenate([y_true, y_pred]))))
        class_names = [str(cls) for cls in classes]
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot confusion matrix
        cm_path = self.output_dir / f"confusion_matrix_ensemble_{timestamp}.png"
        self.plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            classes=class_names,
            output_path=str(cm_path),
            title='Ensemble Confusion Matrix'
        )
        
        # Plot class distribution
        dist_path = self.output_dir / f"class_distribution_{timestamp}.png"
        self.plot_class_distribution(
            df=df,
            label_col='true_label',
            output_path=str(dist_path),
            title='True Class Distribution'
        )
        
        # Generate performance report
        report_path = self.output_dir / f"performance_report_{timestamp}.txt"
        metrics = self.generate_performance_report(
            y_true=y_true,
            y_pred=y_pred,
            output_path=str(report_path),
            model_name='Ensemble'
        )
        
        # Plot individual model predictions if available
        model_metrics = {}
        for col in df.columns:
            if col.endswith('_prediction') and col != 'ensemble_prediction':
                model_name = col.replace('_prediction', '')
                
                # Skip if column has NaN values
                if df[col].isna().any():
                    continue
                
                # Get model predictions
                model_pred = df[col].values
                
                # Plot confusion matrix
                model_cm_path = self.output_dir / f"confusion_matrix_{model_name}_{timestamp}.png"
                self.plot_confusion_matrix(
                    y_true=y_true,
                    y_pred=model_pred,
                    classes=class_names,
                    output_path=str(model_cm_path),
                    title=f'{model_name.capitalize()} Confusion Matrix'
                )
                
                # Generate performance report
                model_report_path = self.output_dir / f"performance_report_{model_name}_{timestamp}.txt"
                model_metrics[model_name] = self.generate_performance_report(
                    y_true=y_true,
                    y_pred=model_pred,
                    output_path=str(model_report_path),
                    model_name=model_name.capitalize()
                )
        
        # Generate summary CSV
        summary_path = self.output_dir / f"model_comparison_{timestamp}.csv"
        
        # Create summary DataFrame
        summary_data = {
            'model': ['ensemble'] + list(model_metrics.keys()),
            'accuracy': [metrics['accuracy']] + [m['accuracy'] for m in model_metrics.values()],
            'precision': [metrics['precision']] + [m['precision'] for m in model_metrics.values()],
            'recall': [metrics['recall']] + [m['recall'] for m in model_metrics.values()],
            'f1': [metrics['f1']] + [m['f1'] for m in model_metrics.values()]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_path, index=False)
        
        logger.info(f"Saved model comparison to {summary_path}")
        
        return {
            'confusion_matrix': str(cm_path),
            'class_distribution': str(dist_path),
            'performance_report': str(report_path),
            'model_comparison': str(summary_path)
        }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualize ensemble results')
    parser.add_argument('--input', required=True,
                       help='Path to predictions CSV')
    parser.add_argument('--output-dir', default='visualizations',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = ResultsVisualizer(
        output_dir=args.output_dir
    )
    
    # Visualize results
    visualizer.visualize_results(
        predictions_file=args.input,
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main() 