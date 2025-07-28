#!/usr/bin/env python3
"""
Make Agent Predictions

This script makes agent predictions using the consistent thresholds
determined during the label balancing step. It ensures that both
ML models and domain agents use the same thresholds for classification.

Usage:
    python make_agent_predictions.py --setup-list data/prediction_setups.txt
"""

import os
import sys
import argparse
import logging
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Import threshold manager
from threshold_manager import ThresholdManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentPredictionMaker:
    """Class for making agent predictions with consistent thresholds"""
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        threshold_file: str = "data/label_thresholds.json"
    ):
        """
        Initialize the agent prediction maker
        
        Args:
            db_path: Path to DuckDB database
            threshold_file: Path to threshold file
        """
        self.db_path = db_path
        self.threshold_manager = ThresholdManager(threshold_file=threshold_file)
    
    def load_setup_ids(self, setup_list_file: str) -> List[str]:
        """
        Load setup IDs from file
        
        Args:
            setup_list_file: Path to setup list file
            
        Returns:
            List of setup IDs
        """
        with open(setup_list_file, 'r') as f:
            setup_ids = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(setup_ids)} setup IDs from {setup_list_file}")
        return setup_ids
    
    def get_domain_predictions(self, setup_ids: List[str], domain: str) -> pd.DataFrame:
        """
        Get predictions from a specific domain
        
        Args:
            setup_ids: List of setup IDs
            domain: Domain name ('news', 'fundamentals', 'analyst_recommendations', 'userposts')
            
        Returns:
            DataFrame with domain predictions
        """
        conn = None
        try:
            conn = duckdb.connect(self.db_path)
            
            # Query to get domain predictions
            query = f"""
                SELECT 
                    setup_id,
                    positive_signal_strength,
                    negative_risk_score,
                    weighted_avg_outperformance
                FROM {domain}_features
                WHERE setup_id = ANY(?)
            """
            
            domain_df = conn.execute(query, [setup_ids]).df()
            
            if len(domain_df) == 0:
                logger.warning(f"No predictions found for domain: {domain}")
                return pd.DataFrame(columns=['setup_id', 'predicted_label', 'predicted_outperformance', 'confidence_score'])
            
            # Load thresholds
            neg_threshold, pos_threshold = self.threshold_manager.get_thresholds_for_prediction()
            
            # Calculate confidence score
            domain_df['confidence_score'] = domain_df.apply(
                lambda row: max(row['positive_signal_strength'], row['negative_risk_score']),
                axis=1
            )
            
            # Calculate predicted label using consistent thresholds
            domain_df['predicted_label'] = domain_df['weighted_avg_outperformance'].apply(
                lambda x: 1 if x >= pos_threshold else (-1 if x <= neg_threshold else 0)
            )
            
            # Select relevant columns
            result_df = domain_df[['setup_id', 'predicted_label', 'weighted_avg_outperformance', 'confidence_score']]
            result_df = result_df.rename(columns={'weighted_avg_outperformance': 'predicted_outperformance'})
            
            logger.info(f"Found {len(result_df)} predictions for domain: {domain}")
            
            # Log class distribution
            if len(result_df) > 0:
                class_counts = result_df['predicted_label'].value_counts()
                logger.info(f"Class distribution for domain {domain}:")
                for cls, count in class_counts.items():
                    logger.info(f"- Class {cls}: {count} ({count/len(result_df):.1%})")
            
            return result_df
        
        finally:
            if conn:
                conn.close()
    
    def make_ensemble_predictions(self, setup_ids: List[str]) -> pd.DataFrame:
        """
        Make ensemble predictions across all domains
        
        Args:
            setup_ids: List of setup IDs
            
        Returns:
            DataFrame with ensemble predictions
        """
        logger.info("Making ensemble predictions across all domains")
        
        # Get predictions from each domain
        domains = ['news', 'fundamentals', 'analyst_recommendations', 'userposts']
        domain_predictions = {}
        
        for domain in domains:
            domain_predictions[domain] = self.get_domain_predictions(setup_ids, domain)
        
        # Create a DataFrame with all setup IDs
        ensemble_df = pd.DataFrame({'setup_id': setup_ids})
        
        # Merge predictions from all domains
        for domain in domains:
            if len(domain_predictions[domain]) > 0:
                domain_df = domain_predictions[domain].copy()
                domain_df = domain_df.rename(columns={
                    'predicted_label': f'{domain}_predicted_label',
                    'predicted_outperformance': f'{domain}_predicted_outperformance',
                    'confidence_score': f'{domain}_confidence_score'
                })
                ensemble_df = pd.merge(ensemble_df, domain_df, on='setup_id', how='left')
        
        # Calculate ensemble prediction
        for col in ['predicted_label', 'predicted_outperformance', 'confidence_score']:
            domain_cols = [f'{domain}_{col}' for domain in domains]
            valid_cols = [col for col in domain_cols if col in ensemble_df.columns]
            
            if not valid_cols:
                ensemble_df[col] = None
                continue
            
            if col == 'predicted_label':
                # Weighted voting for label
                weights = {}
                for domain in domains:
                    confidence_col = f'{domain}_confidence_score'
                    if confidence_col in ensemble_df.columns:
                        weights[f'{domain}_predicted_label'] = ensemble_df[confidence_col]
                
                # Calculate weighted sum for each setup
                ensemble_df['weighted_sum'] = 0
                for label_col, weight_col in weights.items():
                    if label_col in ensemble_df.columns:
                        ensemble_df['weighted_sum'] += ensemble_df[label_col] * weight_col
                
                # Convert weighted sum to label
                neg_threshold, pos_threshold = self.threshold_manager.get_thresholds_for_prediction()
                ensemble_df[col] = ensemble_df['weighted_sum'].apply(
                    lambda x: 1 if x >= pos_threshold else (-1 if x <= neg_threshold else 0)
                )
                ensemble_df = ensemble_df.drop(columns=['weighted_sum'])
            
            elif col == 'predicted_outperformance':
                # Weighted average for outperformance
                weights = {}
                for domain in domains:
                    confidence_col = f'{domain}_confidence_score'
                    if confidence_col in ensemble_df.columns:
                        weights[f'{domain}_predicted_outperformance'] = ensemble_df[confidence_col]
                
                # Calculate weighted average
                weighted_sum = 0
                weight_sum = 0
                for outperformance_col, weight_col in weights.items():
                    if outperformance_col in ensemble_df.columns:
                        weighted_sum += ensemble_df[outperformance_col] * weight_col
                        weight_sum += weight_col
                
                ensemble_df[col] = weighted_sum / weight_sum.replace(0, 1)
            
            elif col == 'confidence_score':
                # Maximum confidence across domains
                ensemble_df[col] = ensemble_df[valid_cols].max(axis=1)
        
        # Count domains with predictions
        domain_count_cols = [f'{domain}_predicted_label' for domain in domains]
        ensemble_df['domains_count'] = ensemble_df[domain_count_cols].notna().sum(axis=1)
        
        # Log ensemble prediction stats
        logger.info(f"Made ensemble predictions for {len(ensemble_df)} setups")
        
        # Log class distribution
        if len(ensemble_df) > 0:
            class_counts = ensemble_df['predicted_label'].value_counts()
            logger.info("Class distribution for ensemble predictions:")
            for cls, count in class_counts.items():
                if pd.notna(cls):
                    logger.info(f"- Class {cls}: {count} ({count/len(ensemble_df):.1%})")
        
        return ensemble_df
    
    def save_predictions_to_db(self, predictions_df: pd.DataFrame) -> None:
        """
        Save predictions to database
        
        Args:
            predictions_df: DataFrame with predictions
        """
        conn = None
        try:
            conn = duckdb.connect(self.db_path)
            
            # Create similarity_predictions table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS similarity_predictions (
                    setup_id VARCHAR,
                    domain VARCHAR,
                    predicted_label INTEGER,
                    predicted_outperformance DOUBLE,
                    confidence_score DOUBLE,
                    PRIMARY KEY (setup_id, domain)
                )
            """)
            
            # Delete existing predictions for these setups
            setup_ids = predictions_df['setup_id'].tolist()
            conn.execute("DELETE FROM similarity_predictions WHERE setup_id = ANY(?)", [setup_ids])
            
            # Insert ensemble predictions
            ensemble_data = predictions_df[['setup_id', 'predicted_label', 'predicted_outperformance', 'confidence_score']].copy()
            ensemble_data['domain'] = 'ensemble'
            
            conn.execute("""
                INSERT INTO similarity_predictions (setup_id, domain, predicted_label, predicted_outperformance, confidence_score)
                SELECT setup_id, domain, predicted_label, predicted_outperformance, confidence_score
                FROM ensemble_data
            """)
            
            # Insert domain predictions
            domains = ['news', 'fundamentals', 'analyst_recommendations', 'userposts']
            for domain in domains:
                label_col = f'{domain}_predicted_label'
                outperformance_col = f'{domain}_predicted_outperformance'
                confidence_col = f'{domain}_confidence_score'
                
                if label_col in predictions_df.columns and outperformance_col in predictions_df.columns and confidence_col in predictions_df.columns:
                    domain_data = predictions_df[['setup_id', label_col, outperformance_col, confidence_col]].copy()
                    domain_data = domain_data.rename(columns={
                        label_col: 'predicted_label',
                        outperformance_col: 'predicted_outperformance',
                        confidence_col: 'confidence_score'
                    })
                    domain_data['domain'] = domain
                    domain_data = domain_data.dropna(subset=['predicted_label'])
                    
                    if len(domain_data) > 0:
                        conn.execute("""
                            INSERT INTO similarity_predictions (setup_id, domain, predicted_label, predicted_outperformance, confidence_score)
                            SELECT setup_id, domain, predicted_label, predicted_outperformance, confidence_score
                            FROM domain_data
                        """)
            
            logger.info(f"Saved predictions to similarity_predictions table")
            
        finally:
            if conn:
                conn.close()
    
    def run(self, setup_list_file: str) -> None:
        """
        Run the agent prediction maker
        
        Args:
            setup_list_file: Path to setup list file
        """
        # Load setup IDs
        setup_ids = self.load_setup_ids(setup_list_file)
        
        # Make ensemble predictions
        predictions_df = self.make_ensemble_predictions(setup_ids)
        
        # Save predictions to database
        self.save_predictions_to_db(predictions_df)
        
        logger.info("Agent predictions completed successfully")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Make agent predictions with consistent thresholds')
    parser.add_argument('--setup-list', required=True,
                       help='Path to setup list file')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--threshold-file', default='data/label_thresholds.json',
                       help='Path to threshold file')
    
    args = parser.parse_args()
    
    # Initialize agent prediction maker
    prediction_maker = AgentPredictionMaker(
        db_path=args.db_path,
        threshold_file=args.threshold_file
    )
    
    # Run prediction maker
    prediction_maker.run(args.setup_list)

if __name__ == '__main__':
    main() 