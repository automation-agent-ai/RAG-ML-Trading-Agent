#!/usr/bin/env python3
"""
Run Complete Pipeline

This script runs the complete ML pipeline with all the improvements:
1. Balance text and financial ML datasets
2. Create balanced classes with configurable thresholds
3. Integrate agent ensemble predictions
4. Train ML models with cross-validation
5. Generate comprehensive results table

Usage:
    python run_complete_pipeline.py --text-train data/ml_features/text_ml_features_training_labeled.csv 
                                   --financial-train data/ml_features/financial_ml_features_training_labeled.csv
                                   --text-predict data/ml_features/text_ml_features_prediction_labeled.csv
                                   --financial-predict data/ml_features/financial_ml_features_prediction_labeled.csv
                                   --output-dir data/ml_pipeline_output
"""

import os
import sys
import argparse
import logging
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLPipeline:
    """Class for running the complete ML pipeline"""
    
    def __init__(
        self,
        output_dir: str = "data/ml_pipeline_output",
        conda_env: str = "sts",
        target_size: int = 600,
        class_ratio: List[float] = None,
        random_seed: int = 42
    ):
        """
        Initialize the pipeline
        
        Args:
            output_dir: Directory to save pipeline outputs
            conda_env: Conda environment to use
            target_size: Target number of setup_ids for each dataset
            class_ratio: Ratio of classes (negative, neutral, positive)
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.conda_env = conda_env
        self.target_size = target_size
        
        # Default to equal class distribution if not provided
        if class_ratio is None:
            self.class_ratio = [0.33, 0.33, 0.34]  # Negative, Neutral, Positive
        else:
            self.class_ratio = class_ratio
            
        self.random_seed = random_seed
        
        # Create subdirectories
        self.balanced_dir = self.output_dir / "balanced"
        self.balanced_classes_dir = self.output_dir / "balanced_classes"
        self.integrated_dir = self.output_dir / "integrated"
        self.models_dir = self.output_dir / "models"
        self.results_dir = self.output_dir / "results"
        
        for directory in [self.balanced_dir, self.balanced_classes_dir, self.integrated_dir, 
                         self.models_dir, self.results_dir]:
            directory.mkdir(exist_ok=True, parents=True)
    
    def run_command(self, command: str) -> int:
        """
        Run a shell command
        
        Args:
            command: Command to run
            
        Returns:
            Return code from command
        """
        logger.info(f"Running command: {command}")
        
        if self.conda_env:
            command = f"conda run -n {self.conda_env} {command}"
        
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            logger.info(line.strip())
        
        # Wait for process to complete
        process.wait()
        
        # Check for errors
        if process.returncode != 0:
            for line in process.stderr:
                logger.error(line.strip())
            logger.error(f"Command failed with return code {process.returncode}")
        
        return process.returncode
    
    def balance_datasets(
        self,
        text_train_file: str,
        financial_train_file: str,
        text_predict_file: str,
        financial_predict_file: str
    ) -> Dict[str, str]:
        """
        Balance text and financial ML datasets
        
        Args:
            text_train_file: Path to text training CSV
            financial_train_file: Path to financial training CSV
            text_predict_file: Path to text prediction CSV
            financial_predict_file: Path to financial prediction CSV
            
        Returns:
            Dictionary with paths to balanced datasets
        """
        logger.info("Step 1: Balancing datasets")
        
        command = (
            f"python balance_ml_datasets.py "
            f"--text-train {text_train_file} "
            f"--financial-train {financial_train_file} "
            f"--text-predict {text_predict_file} "
            f"--financial-predict {financial_predict_file} "
            f"--output-dir {self.balanced_dir} "
            f"--target-size {self.target_size} "
            f"--random-seed {self.random_seed}"
        )
        
        if self.run_command(command) != 0:
            logger.error("Failed to balance datasets")
            return {}
        
        # Find the latest balanced files
        text_train_balanced = self._find_latest_file(self.balanced_dir, "text_ml_features_training_balanced_*.csv")
        financial_train_balanced = self._find_latest_file(self.balanced_dir, "financial_ml_features_training_balanced_*.csv")
        text_predict_balanced = self._find_latest_file(self.balanced_dir, "text_ml_features_prediction_balanced_*.csv")
        financial_predict_balanced = self._find_latest_file(self.balanced_dir, "financial_ml_features_prediction_balanced_*.csv")
        
        return {
            "text_train": text_train_balanced,
            "financial_train": financial_train_balanced,
            "text_predict": text_predict_balanced,
            "financial_predict": financial_predict_balanced
        }
    
    def create_balanced_classes(self, balanced_files: Dict[str, str]) -> Dict[str, str]:
        """
        Create balanced classes with configurable thresholds
        
        Args:
            balanced_files: Dictionary with paths to balanced datasets
            
        Returns:
            Dictionary with paths to balanced classes datasets
        """
        logger.info("Step 2: Creating balanced classes")
        
        # Process training files first to calculate thresholds
        class_ratio_str = ",".join([str(r) for r in self.class_ratio])
        
        # Process text training file
        text_train_command = (
            f"python create_balanced_classes.py "
            f"--input {balanced_files['text_train']} "
            f"--output {self.balanced_classes_dir}/text_ml_features_training_balanced_classes.csv "
            f"--class-ratio {class_ratio_str} "
            f"--mode training"
        )
        
        if self.run_command(text_train_command) != 0:
            logger.error("Failed to create balanced classes for text training data")
            return {}
        
        # Process financial training file
        financial_train_command = (
            f"python create_balanced_classes.py "
            f"--input {balanced_files['financial_train']} "
            f"--output {self.balanced_classes_dir}/financial_ml_features_training_balanced_classes.csv "
            f"--class-ratio {class_ratio_str} "
            f"--mode training"
        )
        
        if self.run_command(financial_train_command) != 0:
            logger.error("Failed to create balanced classes for financial training data")
            return {}
        
        # Process prediction files using the same thresholds
        text_predict_command = (
            f"python create_balanced_classes.py "
            f"--input {balanced_files['text_predict']} "
            f"--output {self.balanced_classes_dir}/text_ml_features_prediction_balanced_classes.csv "
            f"--mode prediction"
        )
        
        if self.run_command(text_predict_command) != 0:
            logger.error("Failed to create balanced classes for text prediction data")
            return {}
        
        financial_predict_command = (
            f"python create_balanced_classes.py "
            f"--input {balanced_files['financial_predict']} "
            f"--output {self.balanced_classes_dir}/financial_ml_features_prediction_balanced_classes.csv "
            f"--mode prediction"
        )
        
        if self.run_command(financial_predict_command) != 0:
            logger.error("Failed to create balanced classes for financial prediction data")
            return {}
        
        return {
            "text_train": f"{self.balanced_classes_dir}/text_ml_features_training_balanced_classes.csv",
            "financial_train": f"{self.balanced_classes_dir}/financial_ml_features_training_balanced_classes.csv",
            "text_predict": f"{self.balanced_classes_dir}/text_ml_features_prediction_balanced_classes.csv",
            "financial_predict": f"{self.balanced_classes_dir}/financial_ml_features_prediction_balanced_classes.csv"
        }
    
    def integrate_agent_predictions(self, balanced_classes_files: Dict[str, str]) -> Dict[str, str]:
        """
        Integrate agent ensemble predictions
        
        Args:
            balanced_classes_files: Dictionary with paths to balanced classes datasets
            
        Returns:
            Dictionary with paths to integrated datasets
        """
        logger.info("Step 3: Integrating agent predictions")
        
        integrated_files = {}
        
        for file_type, file_path in balanced_classes_files.items():
            output_path = f"{self.integrated_dir}/{Path(file_path).name.replace('_balanced_classes', '_with_agents')}"
            
            command = (
                f"python integrate_agent_predictions.py "
                f"--input {file_path} "
                f"--output {output_path} "
                f"--output-dir {self.integrated_dir}"
            )
            
            if self.run_command(command) != 0:
                logger.error(f"Failed to integrate agent predictions for {file_type}")
                continue
            
            integrated_files[file_type] = output_path
        
        return integrated_files
    
    def train_models_with_cv(self, integrated_files: Dict[str, str]) -> Dict[str, str]:
        """
        Train ML models with cross-validation
        
        Args:
            integrated_files: Dictionary with paths to integrated datasets
            
        Returns:
            Dictionary with paths to trained models
        """
        logger.info("Step 4: Training models with cross-validation")
        
        # Train text domain models
        text_models_dir = f"{self.models_dir}/text"
        text_command = (
            f"python train_domain_models_cv.py "
            f"--input {integrated_files['text_train']} "
            f"--output-dir {text_models_dir} "
            f"--cv 5 "
            f"--random-state {self.random_seed} "
            f"--exclude-cols setup_id outperformance_10d"
        )
        
        if self.run_command(text_command) != 0:
            logger.error("Failed to train text domain models")
            return {}
        
        # Train financial domain models
        financial_models_dir = f"{self.models_dir}/financial"
        financial_command = (
            f"python train_domain_models_cv.py "
            f"--input {integrated_files['financial_train']} "
            f"--output-dir {financial_models_dir} "
            f"--cv 5 "
            f"--random-state {self.random_seed} "
            f"--exclude-cols setup_id outperformance_10d"
        )
        
        if self.run_command(financial_command) != 0:
            logger.error("Failed to train financial domain models")
            return {}
        
        return {
            "text_models_dir": text_models_dir,
            "financial_models_dir": financial_models_dir
        }
    
    def make_ensemble_predictions(
        self,
        integrated_files: Dict[str, str],
        models_dirs: Dict[str, str]
    ) -> str:
        """
        Make ensemble predictions
        
        Args:
            integrated_files: Dictionary with paths to integrated datasets
            models_dirs: Dictionary with paths to model directories
            
        Returns:
            Path to ensemble predictions file
        """
        logger.info("Step 5: Making ensemble predictions")
        
        ensemble_predictions_file = f"{self.results_dir}/ensemble_predictions.csv"
        
        command = (
            f"python ensemble_domain_predictions.py "
            f"--text-data {integrated_files['text_predict']} "
            f"--financial-data {integrated_files['financial_predict']} "
            f"--text-models-dir {models_dirs['text_models_dir']} "
            f"--financial-models-dir {models_dirs['financial_models_dir']} "
            f"--ensemble-method weighted_voting "
            f"--output-file {ensemble_predictions_file}"
        )
        
        if self.run_command(command) != 0:
            logger.error("Failed to make ensemble predictions")
            return ""
        
        return ensemble_predictions_file
    
    def generate_results_table(self, ensemble_predictions_file: str) -> str:
        """
        Generate comprehensive results table
        
        Args:
            ensemble_predictions_file: Path to ensemble predictions file
            
        Returns:
            Path to results table file
        """
        logger.info("Step 6: Generating results table")
        
        results_table_file = f"{self.results_dir}/final_results_table.csv"
        
        command = (
            f"python generate_results_table.py "
            f"--ml-predictions {ensemble_predictions_file} "
            f"--output {results_table_file} "
            f"--output-dir {self.results_dir}"
        )
        
        if self.run_command(command) != 0:
            logger.error("Failed to generate results table")
            return ""
        
        return results_table_file
    
    def visualize_results(self, ensemble_predictions_file: str) -> str:
        """
        Visualize ensemble results
        
        Args:
            ensemble_predictions_file: Path to ensemble predictions file
            
        Returns:
            Path to visualizations directory
        """
        logger.info("Step 7: Visualizing results")
        
        visualizations_dir = f"{self.results_dir}/visualizations"
        Path(visualizations_dir).mkdir(exist_ok=True, parents=True)
        
        command = (
            f"python visualize_ensemble_results.py "
            f"--predictions {ensemble_predictions_file} "
            f"--output-dir {visualizations_dir}"
        )
        
        if self.run_command(command) != 0:
            logger.error("Failed to visualize results")
            return ""
        
        return visualizations_dir
    
    def run_pipeline(
        self,
        text_train_file: str,
        financial_train_file: str,
        text_predict_file: str,
        financial_predict_file: str
    ) -> Dict[str, str]:
        """
        Run the complete ML pipeline
        
        Args:
            text_train_file: Path to text training CSV
            financial_train_file: Path to financial training CSV
            text_predict_file: Path to text prediction CSV
            financial_predict_file: Path to financial prediction CSV
            
        Returns:
            Dictionary with paths to pipeline outputs
        """
        logger.info("Starting ML pipeline")
        
        # Step 1: Balance datasets
        balanced_files = self.balance_datasets(
            text_train_file,
            financial_train_file,
            text_predict_file,
            financial_predict_file
        )
        
        if not balanced_files:
            logger.error("Failed to balance datasets, aborting pipeline")
            return {}
        
        # Step 2: Create balanced classes
        balanced_classes_files = self.create_balanced_classes(balanced_files)
        
        if not balanced_classes_files:
            logger.error("Failed to create balanced classes, aborting pipeline")
            return balanced_files
        
        # Step 3: Integrate agent predictions
        integrated_files = self.integrate_agent_predictions(balanced_classes_files)
        
        if not integrated_files:
            logger.error("Failed to integrate agent predictions, aborting pipeline")
            return {**balanced_files, **balanced_classes_files}
        
        # Step 4: Train models with cross-validation
        models_dirs = self.train_models_with_cv(integrated_files)
        
        if not models_dirs:
            logger.error("Failed to train models, aborting pipeline")
            return {**balanced_files, **balanced_classes_files, **integrated_files}
        
        # Step 5: Make ensemble predictions
        ensemble_predictions_file = self.make_ensemble_predictions(integrated_files, models_dirs)
        
        if not ensemble_predictions_file:
            logger.error("Failed to make ensemble predictions, aborting pipeline")
            return {**balanced_files, **balanced_classes_files, **integrated_files, **models_dirs}
        
        # Step 6: Generate results table
        results_table_file = self.generate_results_table(ensemble_predictions_file)
        
        # Step 7: Visualize results
        visualizations_dir = self.visualize_results(ensemble_predictions_file)
        
        # Return all outputs
        outputs = {
            **balanced_files,
            **balanced_classes_files,
            **integrated_files,
            **models_dirs,
            "ensemble_predictions_file": ensemble_predictions_file,
            "results_table_file": results_table_file,
            "visualizations_dir": visualizations_dir
        }
        
        logger.info("ML pipeline completed successfully")
        logger.info("\nPipeline outputs:")
        for output_name, output_path in outputs.items():
            logger.info(f"- {output_name}: {output_path}")
        
        return outputs
    
    def _find_latest_file(self, directory: Path, pattern: str) -> str:
        """
        Find the latest file matching a pattern
        
        Args:
            directory: Directory to search
            pattern: Glob pattern to match
            
        Returns:
            Path to latest file
        """
        files = list(directory.glob(pattern))
        if not files:
            return ""
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return str(files[0])

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run complete ML pipeline')
    parser.add_argument('--text-train', required=True,
                       help='Path to text training CSV')
    parser.add_argument('--financial-train', required=True,
                       help='Path to financial training CSV')
    parser.add_argument('--text-predict', required=True,
                       help='Path to text prediction CSV')
    parser.add_argument('--financial-predict', required=True,
                       help='Path to financial prediction CSV')
    parser.add_argument('--output-dir', default='data/ml_pipeline_output',
                       help='Directory to save pipeline outputs')
    parser.add_argument('--conda-env', default='sts',
                       help='Conda environment to use')
    parser.add_argument('--target-size', type=int, default=600,
                       help='Target number of setup_ids for each dataset')
    parser.add_argument('--class-ratio', default='0.33,0.33,0.34',
                       help='Ratio of classes (negative,neutral,positive)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Parse class ratio
    class_ratio = [float(r) for r in args.class_ratio.split(',')]
    
    # Initialize pipeline
    pipeline = MLPipeline(
        output_dir=args.output_dir,
        conda_env=args.conda_env,
        target_size=args.target_size,
        class_ratio=class_ratio,
        random_seed=args.random_seed
    )
    
    # Run pipeline
    pipeline.run_pipeline(
        text_train_file=args.text_train,
        financial_train_file=args.financial_train,
        text_predict_file=args.text_predict,
        financial_predict_file=args.financial_predict
    )

if __name__ == '__main__':
    main() 