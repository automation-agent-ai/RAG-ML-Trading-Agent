#!/usr/bin/env python3
"""
Generate Results Table

This script generates a comprehensive results table by combining ML predictions,
agent ensemble predictions, and actual labels.

Usage:
    python generate_results_table.py --input ensemble_predictions.csv --output results_table.csv
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

class ResultsTableGenerator:
    """Class for generating comprehensive results table"""
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        output_dir: str = "data"
    ):
        """
        Initialize the generator
        
        Args:
            db_path: Path to DuckDB database
            output_dir: Directory to save output files
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def load_ml_predictions(self, file_path: str) -> pd.DataFrame:
        """
        Load ML predictions from CSV
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with ML predictions
        """
        logger.info(f"Loading ML predictions from {file_path}")
        df = pd.read_csv(file_path)
        
        # Basic info
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        return df
    
    def get_actual_labels(self, setup_ids: List[str]) -> pd.DataFrame:
        """
        Get actual labels from the database
        
        Args:
            setup_ids: List of setup IDs
            
        Returns:
            DataFrame with actual labels
        """
        logger.info(f"Getting actual labels for {len(setup_ids)} setup IDs")
        
        conn = None
        try:
            conn = duckdb.connect(self.db_path)
            
            # Get average outperformance for first 10 days
            labels_df = conn.execute("""
                SELECT 
                    setup_id, 
                    AVG(outperformance_day) as outperformance_10d,
                    CASE
                        WHEN AVG(outperformance_day) >= 0.02 THEN 1  -- Positive (1)
                        WHEN AVG(outperformance_day) <= -0.02 THEN -1  -- Negative (-1)
                        ELSE 0  -- Neutral (0)
                    END as actual_label
                FROM daily_labels
                WHERE day_number <= 10
                AND setup_id = ANY(?)
                GROUP BY setup_id
                HAVING COUNT(DISTINCT day_number) >= 2
            """, [setup_ids]).df()
            
            logger.info(f"Found {len(labels_df)} actual labels")
            
            return labels_df
        
        finally:
            if conn:
                conn.close()
    
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
                pivot_df[f'{domain}_similar_cases_count'] = domain_df.set_index('setup_id')['similar_cases_count']
            
            # Add setup_id as a column
            pivot_df = pivot_df.reset_index().rename(columns={'index': 'setup_id'})
            
            # Fill NaN values with 0
            pivot_df = pivot_df.fillna(0)
            
            # Add aggregate features across domains
            domains_with_data = [d for d in domains if f'{d}_predicted_outperformance' in pivot_df.columns]
            if domains_with_data:
                # Count domains with predictions for each setup_id
                pivot_df['domains_count'] = 0
                for domain in domains_with_data:
                    pivot_df['domains_count'] += (pivot_df[f'{domain}_predicted_outperformance'] != 0).astype(int)
                
                # Calculate weighted average predictions
                if len(domains_with_data) > 1:
                    # Multiple domains - calculate ensemble
                    pivot_df['agent_predicted_outperformance'] = pivot_df[[f'{d}_predicted_outperformance' for d in domains_with_data]].mean(axis=1)
                    pivot_df['agent_confidence_score'] = pivot_df[[f'{d}_confidence' for d in domains_with_data if f'{d}_confidence' in pivot_df.columns]].mean(axis=1)
                else:
                    # Single domain - use its values directly
                    domain = domains_with_data[0]
                    pivot_df['agent_predicted_outperformance'] = pivot_df[f'{domain}_predicted_outperformance']
                    pivot_df['agent_confidence_score'] = pivot_df[f'{domain}_confidence']
                
                # Add agent predicted class
                pivot_df['agent_ensemble_prediction'] = pivot_df['agent_predicted_outperformance'].apply(
                    lambda x: 1 if x >= 0.02 else (-1 if x <= -0.02 else 0)
                )
            
            return pivot_df
        
        finally:
            if conn:
                conn.close()
    
    def generate_results_table(
        self,
        ml_predictions_file: str,
        output_file: str = None
    ) -> str:
        """
        Generate comprehensive results table
        
        Args:
            ml_predictions_file: Path to ML predictions CSV
            output_file: Path to output CSV file
            
        Returns:
            Path to output CSV file
        """
        # Load ML predictions
        ml_df = self.load_ml_predictions(ml_predictions_file)
        
        # Get setup IDs
        setup_ids = ml_df['setup_id'].tolist()
        
        # Get actual labels
        labels_df = self.get_actual_labels(setup_ids)
        
        # Get agent predictions
        agent_df = self.get_agent_predictions(setup_ids)
        
        # Create results table
        results_df = pd.DataFrame({'setup_id': setup_ids})
        
        # Add actual labels and outperformance
        if not labels_df.empty:
            results_df = pd.merge(results_df, labels_df[['setup_id', 'actual_label', 'outperformance_10d']], 
                                 on='setup_id', how='left')
        
        # Add ML predictions
        if 'ensemble_prediction' in ml_df.columns:
            results_df = pd.merge(results_df, ml_df[['setup_id', 'ensemble_prediction']], 
                                 on='setup_id', how='left')
            results_df = results_df.rename(columns={'ensemble_prediction': 'predicted_label_ML'})
        
        # Add confidence score (placeholder)
        results_df['confidence_score_ml'] = 0.8  # This would ideally come from the ML models
        
        # Add agent predictions
        if not agent_df.empty:
            agent_cols = ['setup_id', 'agent_ensemble_prediction', 'agent_predicted_outperformance', 
                         'agent_confidence_score', 'domains_count']
            agent_cols = [col for col in agent_cols if col in agent_df.columns]
            results_df = pd.merge(results_df, agent_df[agent_cols], on='setup_id', how='left')
        
        # Fill NaN values
        results_df = results_df.fillna({
            'agent_ensemble_prediction': 0,
            'agent_predicted_outperformance': 0.0,
            'agent_confidence_score': 0.0,
            'domains_count': 0
        })
        
        # Rename columns to match requested format
        column_mapping = {
            'agent_ensemble_prediction': 'Agent_Ensemble_Prediction',
            'agent_predicted_outperformance': 'Agent_Predicted_Outperformance',
            'agent_confidence_score': 'Agent_Confidence_Score',
            'domains_count': 'Domains_Count'
        }
        results_df = results_df.rename(columns=column_mapping)
        
        # Reorder columns
        ordered_cols = [
            'setup_id', 
            'actual_label', 
            'outperformance_10d', 
            'predicted_label_ML', 
            'confidence_score_ml',
            'Agent_Ensemble_Prediction',
            'Agent_Predicted_Outperformance',
            'Agent_Confidence_Score',
            'Domains_Count'
        ]
        ordered_cols = [col for col in ordered_cols if col in results_df.columns]
        remaining_cols = [col for col in results_df.columns if col not in ordered_cols]
        ordered_cols.extend(remaining_cols)
        results_df = results_df[ordered_cols]
        
        # Generate output file name if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"results_table_{timestamp}.csv"
        
        # Save results table
        results_df.to_csv(output_file, index=False)
        logger.info(f"Saved results table to {output_file}")
        
        # Print summary
        logger.info("\nResults Table Summary:")
        logger.info(f"- Rows: {len(results_df)}")
        logger.info(f"- Columns: {len(results_df.columns)}")
        
        # Print column list
        logger.info("\nColumns:")
        for col in results_df.columns:
            logger.info(f"- {col}")
        
        return str(output_file)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate results table')
    parser.add_argument('--input', required=True,
                       help='Path to ML predictions CSV')
    parser.add_argument('--output',
                       help='Path to output CSV file')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--output-dir', default='data',
                       help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = ResultsTableGenerator(
        db_path=args.db_path,
        output_dir=args.output_dir
    )
    
    # Generate results table
    generator.generate_results_table(
        ml_predictions_file=args.input,
        output_file=args.output
    )

if __name__ == '__main__':
    main() 