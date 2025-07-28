#!/usr/bin/env python3
"""
Integrate Agent Predictions

This script integrates agent ensemble predictions from the similarity_predictions table
into ML feature datasets as additional features.

Usage:
    python integrate_agent_predictions.py --input data/ml_features/text_ml_features_training_balanced_classes.csv 
                                         --output data/ml_features/text_ml_features_training_with_agents.csv
                                         --db-path data/sentiment_system.duckdb
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentIntegrator:
    """Class for integrating agent predictions into ML features"""
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        output_dir: str = "data/ml_features/integrated"
    ):
        """
        Initialize the integrator
        
        Args:
            db_path: Path to DuckDB database
            output_dir: Directory to save integrated features
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load ML features from CSV
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with ML features
        """
        logger.info(f"Loading ML features from {file_path}")
        df = pd.read_csv(file_path)
        
        # Basic info
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        return df
    
    def get_agent_predictions(self, setup_ids: List[str]) -> pd.DataFrame:
        """
        Get agent predictions from the database
        
        Args:
            setup_ids: List of setup IDs
            
        Returns:
            DataFrame with agent predictions
        """
        logger.info(f"Getting agent predictions for {len(setup_ids)} setup IDs")
        
        conn = None
        try:
            conn = duckdb.connect(self.db_path)
            
            # Get agent predictions from similarity_predictions table
            agent_df = conn.execute("""
                SELECT 
                    setup_id,
                    domain,
                    predicted_outperformance,
                    confidence,
                    positive_ratio,
                    negative_ratio,
                    neutral_ratio,
                    similar_cases_count
                FROM similarity_predictions
                WHERE setup_id = ANY(?)
            """, [setup_ids]).df()
            
            logger.info(f"Found {len(agent_df)} agent predictions")
            
            # Check domains
            domains = agent_df['domain'].unique()
            logger.info(f"Found predictions for domains: {domains}")
            
            # Pivot the data to have one row per setup_id with columns for each domain
            pivot_df = pd.DataFrame(index=setup_ids)
            
            for domain in domains:
                domain_df = agent_df[agent_df['domain'] == domain]
                
                # Add domain-specific columns
                pivot_df[f'{domain}_predicted_outperformance'] = domain_df.set_index('setup_id')['predicted_outperformance']
                pivot_df[f'{domain}_confidence'] = domain_df.set_index('setup_id')['confidence']
                pivot_df[f'{domain}_positive_ratio'] = domain_df.set_index('setup_id')['positive_ratio']
                pivot_df[f'{domain}_negative_ratio'] = domain_df.set_index('setup_id')['negative_ratio']
                pivot_df[f'{domain}_neutral_ratio'] = domain_df.set_index('setup_id')['neutral_ratio']
                pivot_df[f'{domain}_similar_cases_count'] = domain_df.set_index('setup_id')['similar_cases_count']
            
            # Add setup_id as a column
            pivot_df = pivot_df.reset_index().rename(columns={'index': 'setup_id'})
            
            # Fill NaN values with 0
            pivot_df = pivot_df.fillna(0)
            
            # Add aggregate features across domains
            for domain in domains:
                # Count how many domains have predictions for each setup_id
                if f'{domain}_predicted_outperformance' in pivot_df.columns:
                    pivot_df['domains_count'] = (pivot_df[f'{domain}_predicted_outperformance'] != 0).astype(int)
                    
                    # If we have multiple domains, add more aggregate features
                    if len(domains) > 1:
                        # Calculate weighted average predictions
                        pivot_df['agent_ensemble_prediction'] = pivot_df[[f'{d}_predicted_outperformance' for d in domains]].mean(axis=1)
                        
                        # Calculate weighted confidence
                        confidence_cols = [f'{d}_confidence' for d in domains if f'{d}_confidence' in pivot_df.columns]
                        if confidence_cols:
                            pivot_df['agent_ensemble_confidence'] = pivot_df[confidence_cols].mean(axis=1)
                        
                        # Calculate aggregate ratios
                        pivot_df['agent_ensemble_positive_ratio'] = pivot_df[[f'{d}_positive_ratio' for d in domains if f'{d}_positive_ratio' in pivot_df.columns]].mean(axis=1)
                        pivot_df['agent_ensemble_negative_ratio'] = pivot_df[[f'{d}_negative_ratio' for d in domains if f'{d}_negative_ratio' in pivot_df.columns]].mean(axis=1)
                        pivot_df['agent_ensemble_neutral_ratio'] = pivot_df[[f'{d}_neutral_ratio' for d in domains if f'{d}_neutral_ratio' in pivot_df.columns]].mean(axis=1)
                    else:
                        # If we only have one domain, use its values directly
                        pivot_df['agent_ensemble_prediction'] = pivot_df[f'{domain}_predicted_outperformance']
                        pivot_df['agent_ensemble_confidence'] = pivot_df[f'{domain}_confidence']
                        pivot_df['agent_ensemble_positive_ratio'] = pivot_df[f'{domain}_positive_ratio']
                        pivot_df['agent_ensemble_negative_ratio'] = pivot_df[f'{domain}_negative_ratio']
                        pivot_df['agent_ensemble_neutral_ratio'] = pivot_df[f'{domain}_neutral_ratio']
            
            # Add predicted class based on predicted_outperformance
            pivot_df['agent_predicted_class'] = pivot_df['agent_ensemble_prediction'].apply(
                lambda x: 0 if x < -0.02 else (2 if x > 0.02 else 1)
            )
            
            return pivot_df
        
        finally:
            if conn:
                conn.close()
    
    def integrate_agent_predictions(self, df: pd.DataFrame, agent_df: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate agent predictions into ML features
        
        Args:
            df: DataFrame with ML features
            agent_df: DataFrame with agent predictions
            
        Returns:
            DataFrame with integrated features
        """
        logger.info("Integrating agent predictions into ML features")
        
        # Merge features with agent predictions
        df_integrated = pd.merge(df, agent_df, on='setup_id', how='left')
        
        # Fill NaN values with 0
        agent_cols = [col for col in agent_df.columns if col != 'setup_id']
        df_integrated[agent_cols] = df_integrated[agent_cols].fillna(0)
        
        return df_integrated
    
    def process_file(self, input_file: str, output_file: str = None) -> str:
        """
        Process a ML features file
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            
        Returns:
            Path to output CSV file
        """
        # Load data
        df = self.load_data(input_file)
        
        # Get setup IDs
        setup_ids = df['setup_id'].tolist()
        
        # Get agent predictions from database
        agent_df = self.get_agent_predictions(setup_ids)
        
        # Integrate agent predictions into features
        df_integrated = self.integrate_agent_predictions(df, agent_df)
        
        # Generate output file name if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_path = Path(input_file)
            output_file = self.output_dir / f"{input_path.stem}_with_agents_{timestamp}.csv"
        
        # Save integrated features
        df_integrated.to_csv(output_file, index=False)
        logger.info(f"Saved integrated features to {output_file}")
        
        # Print summary
        logger.info("\nIntegrated Features Summary:")
        logger.info(f"- Input rows: {len(df)}")
        logger.info(f"- Output rows: {len(df_integrated)}")
        logger.info(f"- Added agent features: {len(agent_df.columns) - 1}")
        
        # List added agent features
        agent_cols = [col for col in agent_df.columns if col != 'setup_id']
        logger.info("\nAdded agent features:")
        for col in agent_cols:
            logger.info(f"- {col}")
        
        return str(output_file)
    
    def process_multiple_files(
        self,
        files_dict: Dict[str, str],
        output_suffix: str = "with_agents"
    ) -> Dict[str, str]:
        """
        Process multiple files
        
        Args:
            files_dict: Dictionary mapping file types to file paths
            output_suffix: Suffix to add to output file names
            
        Returns:
            Dictionary with paths to processed files
        """
        output_dict = {}
        
        for file_type, file_path in files_dict.items():
            output_path = Path(file_path).parent / f"{Path(file_path).stem}_{output_suffix}.csv"
            output_dict[file_type] = self.process_file(file_path, str(output_path))
        
        return output_dict

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Integrate agent predictions')
    parser.add_argument('--input', required=True,
                       help='Path to input ML features CSV')
    parser.add_argument('--output',
                       help='Path to output integrated features CSV')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--output-dir', default='data/ml_features/integrated',
                       help='Directory to save integrated features')
    
    args = parser.parse_args()
    
    # Initialize integrator
    integrator = AgentIntegrator(
        db_path=args.db_path,
        output_dir=args.output_dir
    )
    
    # Process file
    integrator.process_file(
        input_file=args.input,
        output_file=args.output
    )

if __name__ == '__main__':
    main() 