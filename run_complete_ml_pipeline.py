#!/usr/bin/env python3
"""
Run Complete ML Pipeline

This script runs the complete ML pipeline, from feature extraction to model training and prediction.

Usage:
    python run_complete_ml_pipeline.py --mode [training|prediction|all]
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompletePipeline:
    """Class for running the complete ML pipeline"""
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        output_dir: str = "data/ml_pipeline_output",
        conda_env: str = "sts"
    ):
        """
        Initialize the pipeline
        
        Args:
            db_path: Path to DuckDB database
            output_dir: Directory to save output files
            conda_env: Conda environment to use for running commands
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.conda_env = conda_env
        
        # Create subdirectories
        self.features_dir = self.output_dir / "features"
        self.labeled_dir = self.output_dir / "labeled"
        self.balanced_dir = self.output_dir / "balanced"
        self.models_dir = self.output_dir / "models"
        self.predictions_dir = self.output_dir / "predictions"
        self.results_dir = self.output_dir / "results"
        
        for directory in [self.features_dir, self.labeled_dir, self.balanced_dir, 
                         self.models_dir, self.predictions_dir, self.results_dir]:
            directory.mkdir(exist_ok=True, parents=True)
    
    def run_command(self, command: str) -> Tuple[int, str]:
        """
        Run a shell command
        
        Args:
            command: Command to run
            
        Returns:
            Tuple of (return code, output)
        """
        # Add conda environment activation if specified
        if self.conda_env:
            command = f"conda run -n {self.conda_env} {command}"
        
        logger.info(f"Running command: {command}")
        
        # Run command and capture output
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Read and log output in real-time
        output = []
        for line in process.stdout:
            line = line.rstrip()
            logger.info(f"  {line}")
            output.append(line)
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code != 0:
            logger.error(f"Command failed with return code {return_code}")
        else:
            logger.info(f"Command completed successfully")
        
        return return_code, "\n".join(output)
    
    def extract_features(self, mode: str = "training") -> Dict[str, str]:
        """
        Extract ML features from DuckDB
        
        Args:
            mode: Mode ('training' or 'prediction')
            
        Returns:
            Dictionary with paths to extracted feature files
        """
        logger.info(f"Extracting {mode} features")
        
        # Extract text features
        text_setup_file = f"data/{mode}_setups.txt"
        text_features_cmd = f"python extract_text_features.py --mode {mode} --setup-list {text_setup_file} --output-dir {self.features_dir}"
        self.run_command(text_features_cmd)
        
        # Extract financial features
        financial_features_cmd = f"python extract_financial_features_from_duckdb.py --mode {mode} --setup-list {text_setup_file} --output-dir {self.features_dir}"
        self.run_command(financial_features_cmd)
        
        # Find the most recent feature files
        text_pattern = f"text_ml_features_{mode}_*.csv"
        financial_pattern = f"financial_ml_features_{mode}_*.csv"
        
        text_files = sorted(list(self.features_dir.glob(text_pattern)), reverse=True)
        financial_files = sorted(list(self.features_dir.glob(financial_pattern)), reverse=True)
        
        if not text_files or not financial_files:
            logger.error(f"Failed to find extracted feature files")
            return {}
        
        text_file = text_files[0]
        financial_file = financial_files[0]
        
        logger.info(f"Extracted text features: {text_file}")
        logger.info(f"Extracted financial features: {financial_file}")
        
        return {
            "text_features": str(text_file),
            "financial_features": str(financial_file)
        }
    
    def add_labels(self, feature_files: Dict[str, str], mode: str = "training") -> Dict[str, str]:
        """
        Add labels to ML features
        
        Args:
            feature_files: Dictionary with paths to feature files
            mode: Mode ('training' or 'prediction')
            
        Returns:
            Dictionary with paths to labeled feature files
        """
        logger.info(f"Adding labels to {mode} features")
        
        text_file = feature_files.get("text_features")
        financial_file = feature_files.get("financial_features")
        
        if not text_file or not financial_file:
            logger.error("Missing feature files")
            return {}
        
        # Add labels to text features
        text_labeled_file = str(self.labeled_dir / f"text_ml_features_{mode}_labeled.csv")
        text_labels_cmd = f"python add_labels_to_features.py --input {text_file} --output {text_labeled_file} --mode {mode}"
        self.run_command(text_labels_cmd)
        
        # Add labels to financial features
        financial_labeled_file = str(self.labeled_dir / f"financial_ml_features_{mode}_labeled.csv")
        financial_labels_cmd = f"python add_labels_to_features.py --input {financial_file} --output {financial_labeled_file} --mode {mode}"
        self.run_command(financial_labels_cmd)
        
        return {
            "text_labeled": text_labeled_file,
            "financial_labeled": financial_labeled_file
        }
    
    def balance_datasets(self, labeled_files: Dict[str, str]) -> Dict[str, str]:
        """
        Balance datasets to ensure consistent setup_ids and labels
        
        Args:
            labeled_files: Dictionary with paths to labeled feature files
            
        Returns:
            Dictionary with paths to balanced feature files
        """
        logger.info("Balancing datasets")
        
        text_train = labeled_files.get("text_labeled_train")
        financial_train = labeled_files.get("financial_labeled_train")
        text_predict = labeled_files.get("text_labeled_predict")
        financial_predict = labeled_files.get("financial_labeled_predict")
        
        if not text_train or not financial_train or not text_predict or not financial_predict:
            logger.error("Missing labeled files")
            return {}
        
        # Balance datasets
        balance_cmd = (
            f"python balance_ml_datasets.py "
            f"--text-train {text_train} "
            f"--financial-train {financial_train} "
            f"--text-predict {text_predict} "
            f"--financial-predict {financial_predict} "
            f"--output-dir {self.balanced_dir}"
        )
        self.run_command(balance_cmd)
        
        # Find the most recent balanced files
        text_train_pattern = "text_ml_features_training_balanced_*.csv"
        financial_train_pattern = "financial_ml_features_training_balanced_*.csv"
        text_predict_pattern = "text_ml_features_prediction_balanced_*.csv"
        financial_predict_pattern = "financial_ml_features_prediction_balanced_*.csv"
        
        text_train_files = sorted(list(self.balanced_dir.glob(text_train_pattern)), reverse=True)
        financial_train_files = sorted(list(self.balanced_dir.glob(financial_train_pattern)), reverse=True)
        text_predict_files = sorted(list(self.balanced_dir.glob(text_predict_pattern)), reverse=True)
        financial_predict_files = sorted(list(self.balanced_dir.glob(financial_predict_pattern)), reverse=True)
        
        if not text_train_files or not financial_train_files or not text_predict_files or not financial_predict_files:
            logger.error("Failed to find balanced files")
            return {}
        
        return {
            "text_balanced_train": str(text_train_files[0]),
            "financial_balanced_train": str(financial_train_files[0]),
            "text_balanced_predict": str(text_predict_files[0]),
            "financial_balanced_predict": str(financial_predict_files[0])
        }
    
    def train_models(self, balanced_files: Dict[str, str]) -> Dict[str, str]:
        """
        Train ML models
        
        Args:
            balanced_files: Dictionary with paths to balanced feature files
            
        Returns:
            Dictionary with paths to trained model directories
        """
        logger.info("Training ML models")
        
        text_train = balanced_files.get("text_balanced_train")
        financial_train = balanced_files.get("financial_balanced_train")
        
        if not text_train or not financial_train:
            logger.error("Missing balanced training files")
            return {}
        
        # Train text models
        text_models_dir = str(self.models_dir / "text")
        text_train_cmd = f"python train_domain_models_cv.py --input {text_train} --domain text --output-dir {text_models_dir}"
        self.run_command(text_train_cmd)
        
        # Train financial models
        financial_models_dir = str(self.models_dir / "financial")
        financial_train_cmd = f"python train_domain_models_cv.py --input {financial_train} --domain financial --output-dir {financial_models_dir}"
        self.run_command(financial_train_cmd)
        
        return {
            "text_models": text_models_dir,
            "financial_models": financial_models_dir
        }
    
    def make_predictions(self, balanced_files: Dict[str, str], model_dirs: Dict[str, str]) -> Dict[str, str]:
        """
        Make predictions using trained models
        
        Args:
            balanced_files: Dictionary with paths to balanced feature files
            model_dirs: Dictionary with paths to trained model directories
            
        Returns:
            Dictionary with paths to prediction files
        """
        logger.info("Making predictions")
        
        text_predict = balanced_files.get("text_balanced_predict")
        financial_predict = balanced_files.get("financial_balanced_predict")
        text_models = model_dirs.get("text_models")
        financial_models = model_dirs.get("financial_models")
        
        if not text_predict or not financial_predict or not text_models or not financial_models:
            logger.error("Missing balanced prediction files or model directories")
            return {}
        
        # Make domain predictions
        domain_predictions_cmd = (
            f"python ensemble_domain_predictions.py "
            f"--text-input {text_predict} "
            f"--financial-input {financial_predict} "
            f"--text-models-dir {text_models} "
            f"--financial-models-dir {financial_models} "
            f"--output-dir {self.predictions_dir}"
        )
        self.run_command(domain_predictions_cmd)
        
        # Find the most recent ensemble predictions file
        ensemble_pattern = "ensemble_predictions_*.csv"
        ensemble_files = sorted(list(self.predictions_dir.glob(ensemble_pattern)), reverse=True)
        
        if not ensemble_files:
            logger.error("Failed to find ensemble predictions file")
            return {}
        
        ensemble_file = ensemble_files[0]
        logger.info(f"Ensemble predictions: {ensemble_file}")
        
        return {
            "ensemble_predictions": str(ensemble_file)
        }
    
    def generate_results(self, prediction_files: Dict[str, str]) -> Dict[str, str]:
        """
        Generate final results table and visualizations
        
        Args:
            prediction_files: Dictionary with paths to prediction files
            
        Returns:
            Dictionary with paths to results files
        """
        logger.info("Generating results")
        
        ensemble_predictions = prediction_files.get("ensemble_predictions")
        
        if not ensemble_predictions:
            logger.error("Missing ensemble predictions file")
            return {}
        
        # Generate results table
        results_table_file = str(self.results_dir / "results_table.csv")
        results_cmd = f"python generate_results_table.py --input {ensemble_predictions} --output {results_table_file}"
        self.run_command(results_cmd)
        
        # Generate visualizations
        visualizations_dir = str(self.results_dir / "visualizations")
        Path(visualizations_dir).mkdir(exist_ok=True, parents=True)
        
        viz_cmd = f"python visualize_ensemble_results.py --input {ensemble_predictions} --output-dir {visualizations_dir}"
        self.run_command(viz_cmd)
        
        return {
            "results_table": results_table_file,
            "visualizations_dir": visualizations_dir
        }
    
    def run_pipeline(self, mode: str = "all") -> Dict[str, Any]:
        """
        Run the complete ML pipeline
        
        Args:
            mode: Mode ('training', 'prediction', or 'all')
            
        Returns:
            Dictionary with paths to all output files
        """
        logger.info(f"Running complete ML pipeline in {mode} mode")
        
        results = {}
        
        # Extract features
        if mode in ["training", "all"]:
            training_features = self.extract_features(mode="training")
            results["training_features"] = training_features
        
        if mode in ["prediction", "all"]:
            prediction_features = self.extract_features(mode="prediction")
            results["prediction_features"] = prediction_features
        
        # Add labels
        if mode in ["training", "all"]:
            training_labeled = self.add_labels(results["training_features"], mode="training")
            results["training_labeled"] = training_labeled
        
        if mode in ["prediction", "all"]:
            prediction_labeled = self.add_labels(results["prediction_features"], mode="prediction")
            results["prediction_labeled"] = prediction_labeled
        
        # Balance datasets
        if mode == "all":
            labeled_files = {
                "text_labeled_train": results["training_labeled"]["text_labeled"],
                "financial_labeled_train": results["training_labeled"]["financial_labeled"],
                "text_labeled_predict": results["prediction_labeled"]["text_labeled"],
                "financial_labeled_predict": results["prediction_labeled"]["financial_labeled"]
            }
            balanced_files = self.balance_datasets(labeled_files)
            results["balanced_files"] = balanced_files
            
            # Train models
            model_dirs = self.train_models(balanced_files)
            results["model_dirs"] = model_dirs
            
            # Make predictions
            prediction_files = self.make_predictions(balanced_files, model_dirs)
            results["prediction_files"] = prediction_files
            
            # Generate results
            results_files = self.generate_results(prediction_files)
            results["results_files"] = results_files
        
        logger.info("Pipeline completed successfully")
        return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run complete ML pipeline')
    parser.add_argument('--mode', choices=['training', 'prediction', 'all'], default='all',
                       help='Pipeline mode')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--output-dir', default='data/ml_pipeline_output',
                       help='Directory to save output files')
    parser.add_argument('--conda-env', default='sts',
                       help='Conda environment to use')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CompletePipeline(
        db_path=args.db_path,
        output_dir=args.output_dir,
        conda_env=args.conda_env
    )
    
    # Run pipeline
    results = pipeline.run_pipeline(args.mode)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("📊 ML PIPELINE SUMMARY")
    logger.info("="*50)
    
    # Print results
    for category, items in results.items():
        logger.info(f"\n{category.upper()}:")
        for key, value in items.items():
            logger.info(f"- {key}: {value}")
    
    logger.info("\n" + "="*50)
    logger.info("✅ Pipeline completed successfully")
    logger.info("="*50)

if __name__ == "__main__":
    main() 