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
import time

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
        self.visualizations_dir = self.output_dir / "visualizations"
        
        for directory in [self.features_dir, self.labeled_dir, self.balanced_dir, 
                         self.models_dir, self.predictions_dir, self.results_dir,
                         self.visualizations_dir]:
            directory.mkdir(exist_ok=True, parents=True)
    
    def check_database_access(self, max_retries: int = 3, retry_delay: int = 5) -> bool:
        """
        Check if the database is accessible
        
        Args:
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            
        Returns:
            True if database is accessible, False otherwise
        """
        logger.info(f"Checking database access: {self.db_path}")
        
        for attempt in range(max_retries):
            try:
                # Try to connect to the database
                conn = duckdb.connect(self.db_path, read_only=True)
                
                # Run a simple query to verify access
                conn.execute("SELECT 1").fetchone()
                
                # Close connection
                conn.close()
                
                logger.info("✅ Database is accessible")
                return True
            
            except Exception as e:
                logger.warning(f"❌ Database access failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        
        logger.error("❌ Database is not accessible after maximum retries")
        return False
    
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
    
    def create_setup_lists(self) -> Dict[str, str]:
        """
        Create setup lists for training and prediction
        
        Returns:
            Dictionary with paths to setup list files
        """
        logger.info("Creating setup lists for training and prediction")
        
        # Check if prediction setups file exists
        prediction_file = "data/prediction_setups.txt"
        if not os.path.exists(prediction_file):
            # Create prediction setups file
            logger.info("Creating prediction setups file")
            create_prediction_cmd = f"python create_prediction_list.py --count 100 --output {prediction_file}"
            self.run_command(create_prediction_cmd)
        else:
            logger.info(f"Using existing prediction setups file: {prediction_file}")
        
        # Create training setups file
        training_file = "data/training_setups.txt"
        logger.info("Creating training setups file")
        create_training_cmd = f"python create_training_list.py --prediction-file {prediction_file} --output {training_file}"
        self.run_command(create_training_cmd)
        
        return {
            "training_setups": training_file,
            "prediction_setups": prediction_file
        }
    
    def extract_features(self, mode: str = "training", setup_file: str = None) -> Dict[str, str]:
        """
        Extract ML features from DuckDB
        
        Args:
            mode: Mode ('training' or 'prediction')
            setup_file: Path to setup list file
            
        Returns:
            Dictionary with paths to extracted feature files
        """
        logger.info(f"Extracting {mode} features")
        
        # Use provided setup file or default
        if setup_file is None:
            setup_file = f"data/{mode}_setups.txt"
        
        # Extract text features
        text_features_cmd = f"python extract_text_features_from_duckdb.py --mode {mode} --setup-list {setup_file} --output-dir {self.features_dir}"
        self.run_command(text_features_cmd)
        
        # Extract financial features
        financial_features_cmd = f"python extract_financial_features_from_duckdb.py --mode {mode} --setup-list {setup_file} --output-dir {self.features_dir}"
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
        Add labels to ML features using dynamic percentile thresholds for balanced classes
        
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
        
        logger.info(f"Labels added using dynamic thresholds for balanced classes (-1, 0, 1 format)")
        
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
        
        logger.info("Datasets balanced with consistent setup_ids and label format")
        
        return {
            "text_balanced_train": str(text_train_files[0]),
            "financial_balanced_train": str(financial_train_files[0]),
            "text_balanced_predict": str(text_predict_files[0]),
            "financial_balanced_predict": str(financial_predict_files[0])
        }
    
    def train_models(self, balanced_files: Dict[str, str]) -> Dict[str, str]:
        """
        Train ML models using cross-validation
        
        Args:
            balanced_files: Dictionary with paths to balanced feature files
            
        Returns:
            Dictionary with paths to trained model directories
        """
        logger.info("Training ML models with cross-validation")
        
        text_train = balanced_files.get("text_balanced_train")
        financial_train = balanced_files.get("financial_balanced_train")
        
        if not text_train or not financial_train:
            logger.error("Missing balanced training files")
            return {}
        
        # Train text models
        text_models_dir = str(self.models_dir / "text")
        text_train_cmd = f"python train_domain_models_cv.py --input {text_train} --domain text --output-dir {text_models_dir} --exclude-cols outperformance_10d --cv-folds 5"
        self.run_command(text_train_cmd)
        
        # Train financial models
        financial_models_dir = str(self.models_dir / "financial")
        financial_train_cmd = f"python train_domain_models_cv.py --input {financial_train} --domain financial --output-dir {financial_models_dir} --exclude-cols outperformance_10d --cv-folds 5"
        self.run_command(financial_train_cmd)
        
        logger.info("Models trained with cross-validation, excluding outperformance_10d to prevent target leakage")
        
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
        logger.info("Making ensemble predictions from domain models")
        
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
        Generate comprehensive results table combining ML predictions, 
        agent ensemble predictions, and actual labels
        
        Args:
            prediction_files: Dictionary with paths to prediction files
            
        Returns:
            Dictionary with paths to results files
        """
        logger.info("Generating comprehensive results table")
        
        ensemble_predictions = prediction_files.get("ensemble_predictions")
        
        if not ensemble_predictions:
            logger.error("Missing ensemble predictions file")
            return {}
        
        # Generate results table
        results_table_file = str(self.results_dir / "results_table.csv")
        results_cmd = f"python generate_results_table.py --input {ensemble_predictions} --output {results_table_file} --db-path {self.db_path}"
        self.run_command(results_cmd)
        
        logger.info(f"Comprehensive results table generated with ML predictions and agent predictions")
        
        return {
            "results_table": results_table_file
        }
    
    def visualize_results(self, prediction_files: Dict[str, str]) -> Dict[str, str]:
        """
        Generate visualizations for ensemble results
        
        Args:
            prediction_files: Dictionary with paths to prediction files
            
        Returns:
            Dictionary with paths to visualization files
        """
        logger.info("Generating visualizations for ensemble results")
        
        ensemble_predictions = prediction_files.get("ensemble_predictions")
        
        if not ensemble_predictions:
            logger.error("Missing ensemble predictions file")
            return {}
        
        # Generate visualizations
        viz_cmd = f"python visualize_ensemble_results.py --input {ensemble_predictions} --output-dir {self.visualizations_dir}"
        self.run_command(viz_cmd)
        
        logger.info(f"Visualizations generated in {self.visualizations_dir}")
        
        return {
            "visualizations_dir": str(self.visualizations_dir)
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
        
        # Check database access
        if not self.check_database_access():
            logger.error("Database is not accessible, aborting pipeline")
            return {"error": "Database not accessible"}
        
        results = {}
        
        # Create setup lists for training and prediction
        if mode == "all":
            setup_files = self.create_setup_lists()
            results["setup_files"] = setup_files
            
            # Extract features
            training_features = self.extract_features(mode="training", setup_file=setup_files["training_setups"])
            results["training_features"] = training_features
            
            prediction_features = self.extract_features(mode="prediction", setup_file=setup_files["prediction_setups"])
            results["prediction_features"] = prediction_features
        else:
            # Extract features for specific mode
            if mode == "training":
                training_features = self.extract_features(mode="training")
                results["training_features"] = training_features
            
            if mode == "prediction":
                prediction_features = self.extract_features(mode="prediction")
                results["prediction_features"] = prediction_features
        
        # Add labels
        if mode in ["training", "all"] and "training_features" in results and results["training_features"]:
            training_labeled = self.add_labels(results["training_features"], mode="training")
            results["training_labeled"] = training_labeled
        
        if mode in ["prediction", "all"] and "prediction_features" in results and results["prediction_features"]:
            prediction_labeled = self.add_labels(results["prediction_features"], mode="prediction")
            results["prediction_labeled"] = prediction_labeled
        
        # Balance datasets
        if mode == "all":
            # Check if we have both training and prediction labeled files
            if "training_labeled" in results and "prediction_labeled" in results:
                labeled_files = {
                    "text_labeled_train": results["training_labeled"]["text_labeled"],
                    "financial_labeled_train": results["training_labeled"]["financial_labeled"],
                    "text_labeled_predict": results["prediction_labeled"]["text_labeled"],
                    "financial_labeled_predict": results["prediction_labeled"]["financial_labeled"]
                }
                balanced_files = self.balance_datasets(labeled_files)
                results["balanced_files"] = balanced_files
            else:
                logger.warning("Missing labeled files, skipping dataset balancing")
                return results
            
            # Train models
            model_dirs = self.train_models(balanced_files)
            results["model_dirs"] = model_dirs
            
            # Make predictions
            prediction_files = self.make_predictions(balanced_files, model_dirs)
            results["prediction_files"] = prediction_files
            
            # Generate results
            results_files = self.generate_results(prediction_files)
            results["results_files"] = results_files
            
            # Visualize results
            visualization_files = self.visualize_results(prediction_files)
            results["visualization_files"] = visualization_files
        
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
    
    # Print column consistency reminder
    logger.info("\nIMPORTANT REMINDER:")
    logger.info("Ensure that training and prediction feature tables have the same columns in the same order.")
    logger.info("The balance_ml_datasets.py script helps maintain this consistency.")
    logger.info("If you encounter column mismatch errors during prediction, check that the feature extraction")
    logger.info("process completed successfully and the datasets were properly balanced.")

if __name__ == "__main__":
    main() 