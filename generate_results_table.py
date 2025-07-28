#!/usr/bin/env python3
"""
Generate Comprehensive Results Table

Combines ML predictions, agent ensemble predictions, and actual labels into a single table.

Usage:
    python generate_results_table.py --input data/predictions/final_predictions_*.csv --output data/results_table.csv
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import duckdb
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class ResultsTableGenerator:
    """Generate comprehensive results table combining ML and agent predictions"""
    
    def __init__(self, db_path: str = "data/sentiment_system.duckdb"):
        self.db_path = db_path
        
    def load_ml_predictions(self, input_file: str) -> pd.DataFrame:
        """Load ML predictions from final predictions CSV"""
        logger.info(f"Loading ML predictions from: {input_file}")
        
        try:
            df = pd.read_csv(input_file, index_col=0)  # setup_id as index
            logger.info(f"Loaded {len(df)} ML predictions")
            
            # Ensure we have the required columns
            required_cols = ['prediction', 'confidence']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
                    
            return df
            
        except Exception as e:
            logger.error(f"Error loading ML predictions: {e}")
            raise
            
    def load_actual_labels(self, setup_ids: List[str]) -> pd.DataFrame:
        """Load actual labels and outperformance values from DuckDB"""
        logger.info(f"Loading actual labels for {len(setup_ids)} setup IDs from DuckDB")
        
        try:
            # Connect to DuckDB
            conn = duckdb.connect(self.db_path)
            
            # Create a temporary table with setup_ids
            setup_ids_df = pd.DataFrame({'setup_id': setup_ids})
            conn.register('temp_setup_ids', setup_ids_df)
            
            # Query to get actual labels and outperformance
            query = """
            SELECT 
                setup_id,
                outperformance_10d,
                CASE 
                    WHEN outperformance_10d <= -5 THEN -1
                    WHEN outperformance_10d >= 5 THEN 1
                    ELSE 0
                END as actual_label
            FROM labels l
            WHERE l.setup_id IN (SELECT setup_id FROM temp_setup_ids)
            """
            
            df = conn.execute(query).df()
            conn.close()
            
            logger.info(f"Loaded actual labels for {len(df)} setup IDs")
            return df.set_index('setup_id')
            
        except Exception as e:
            logger.error(f"Error loading actual labels: {e}")
            # Return empty DataFrame if DuckDB fails
            return pd.DataFrame(columns=['outperformance_10d', 'actual_label']).set_index(pd.Index([], name='setup_id'))
            
    def load_agent_predictions(self, setup_ids: List[str]) -> pd.DataFrame:
        """Load agent predictions from similarity_predictions table"""
        logger.info(f"Loading agent predictions for {len(setup_ids)} setup IDs")
        
        try:
            # Connect to DuckDB
            conn = duckdb.connect(self.db_path)
            
            # Create a temporary table with setup_ids
            setup_ids_df = pd.DataFrame({'setup_id': setup_ids})
            conn.register('temp_setup_ids', setup_ids_df)
            
            # Query to get agent predictions - pivot by domain
            query = """
            SELECT 
                setup_id,
                MAX(CASE WHEN domain = 'analyst_recommendations' THEN predicted_outperformance END) as analyst_recommendations_prediction,
                MAX(CASE WHEN domain = 'analyst_recommendations' THEN confidence END) as analyst_recommendations_confidence,
                MAX(CASE WHEN domain = 'fundamentals' THEN predicted_outperformance END) as fundamentals_prediction,
                MAX(CASE WHEN domain = 'fundamentals' THEN confidence END) as fundamentals_confidence,
                MAX(CASE WHEN domain = 'news' THEN predicted_outperformance END) as news_prediction,
                MAX(CASE WHEN domain = 'news' THEN confidence END) as news_confidence,
                MAX(CASE WHEN domain = 'userposts' THEN predicted_outperformance END) as userposts_prediction,
                MAX(CASE WHEN domain = 'userposts' THEN confidence END) as userposts_confidence,
                MAX(CASE WHEN domain = 'ensemble' THEN predicted_outperformance END) as ensemble_prediction,
                MAX(CASE WHEN domain = 'ensemble' THEN confidence END) as ensemble_confidence,
                COUNT(DISTINCT CASE WHEN domain != 'ensemble' AND predicted_outperformance IS NOT NULL THEN domain END) as domains_count
            FROM similarity_predictions sp
            WHERE sp.setup_id IN (SELECT setup_id FROM temp_setup_ids)
            GROUP BY setup_id
            """
            
            df = conn.execute(query).df()
            conn.close()
            
            logger.info(f"Loaded agent predictions for {len(df)} setup IDs")
            return df.set_index('setup_id')
            
        except Exception as e:
            logger.error(f"Error loading agent predictions: {e}")
            # Return empty DataFrame if no agent predictions
            return pd.DataFrame(columns=[
                'analyst_recommendations_prediction', 'analyst_recommendations_confidence',
                'fundamentals_prediction', 'fundamentals_confidence',
                'news_prediction', 'news_confidence',
                'userposts_prediction', 'userposts_confidence',
                'ensemble_prediction', 'ensemble_confidence',
                'domains_count'
            ]).set_index(pd.Index([], name='setup_id'))
            
    def generate_results_table(self, ml_predictions: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive results table"""
        logger.info("Generating comprehensive results table")
        
        setup_ids = list(ml_predictions.index)
        
        # Load actual labels
        actual_labels = self.load_actual_labels(setup_ids)
        
        # Load agent predictions
        agent_predictions = self.load_agent_predictions(setup_ids)
        
        # Create results table
        results = []
        
        for setup_id in setup_ids:
            row = {'setup_id': setup_id}
            
            # ML predictions
            if setup_id in ml_predictions.index:
                ml_data = ml_predictions.loc[setup_id]
                row['predicted_label_ML'] = ml_data['prediction']
                row['confidence_score_ml'] = ml_data['confidence']
            else:
                row['predicted_label_ML'] = None
                row['confidence_score_ml'] = None
                
            # Actual labels
            if setup_id in actual_labels.index:
                actual_data = actual_labels.loc[setup_id]
                row['actual_label'] = actual_data['actual_label']
                row['outperformance_10d'] = actual_data['outperformance_10d']
            else:
                row['actual_label'] = None
                row['outperformance_10d'] = None
                
            # Agent predictions
            if setup_id in agent_predictions.index:
                agent_data = agent_predictions.loc[setup_id]
                row['Agent_Ensemble_Prediction'] = agent_data['ensemble_prediction']
                row['Agent_Predicted_Outperformance'] = None  # Not directly available
                row['Agent_Confidence_Score'] = agent_data['ensemble_confidence']
                row['Domains_Count'] = agent_data['domains_count']
                
                # Individual domain predictions (optional, for analysis)
                row['analyst_predictions'] = agent_data['analyst_recommendations_prediction']
                row['fundamentals_predictions'] = agent_data['fundamentals_prediction']
                row['news_predictions'] = agent_data['news_prediction']
                row['userposts_predictions'] = agent_data['userposts_prediction']
            else:
                row['Agent_Ensemble_Prediction'] = None
                row['Agent_Predicted_Outperformance'] = None
                row['Agent_Confidence_Score'] = None
                row['Domains_Count'] = 0
                row['analyst_predictions'] = None
                row['fundamentals_predictions'] = None
                row['news_predictions'] = None
                row['userposts_predictions'] = None
                
            results.append(row)
            
        results_df = pd.DataFrame(results)
        
        # Reorder columns as specified in the workflow
        column_order = [
            'setup_id', 'actual_label', 'outperformance_10d',
            'predicted_label_ML', 'confidence_score_ml',
            'Agent_Ensemble_Prediction', 'Agent_Predicted_Outperformance', 
            'Agent_Confidence_Score', 'Domains_Count',
            'analyst_predictions', 'fundamentals_predictions', 
            'news_predictions', 'userposts_predictions'
        ]
        
        # Only include columns that exist
        existing_columns = [col for col in column_order if col in results_df.columns]
        results_df = results_df[existing_columns]
        
        logger.info(f"Generated results table with {len(results_df)} rows and {len(results_df.columns)} columns")
        return results_df
        
    def save_results(self, results_df: pd.DataFrame, output_file: str):
        """Save results table to CSV"""
        logger.info(f"Saving results table to: {output_file}")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        results_df.to_csv(output_file, index=False)
        
        # Generate summary
        logger.info("\nRESULTS TABLE SUMMARY:")
        logger.info(f"Total records: {len(results_df)}")
        
        if 'actual_label' in results_df.columns:
            actual_counts = results_df['actual_label'].value_counts().sort_index()
            logger.info(f"Actual label distribution: {dict(actual_counts)}")
            
        if 'predicted_label_ML' in results_df.columns:
            ml_counts = results_df['predicted_label_ML'].value_counts().sort_index()
            logger.info(f"ML prediction distribution: {dict(ml_counts)}")
            
        if 'Agent_Ensemble_Prediction' in results_df.columns:
            agent_counts = results_df['Agent_Ensemble_Prediction'].value_counts().sort_index()
            logger.info(f"Agent prediction distribution: {dict(agent_counts)}")
            
        if 'Domains_Count' in results_df.columns:
            domain_counts = results_df['Domains_Count'].value_counts().sort_index()
            logger.info(f"Domain coverage: {dict(domain_counts)}")
            
        # Calculate agreement where both predictions exist
        if 'actual_label' in results_df.columns and 'predicted_label_ML' in results_df.columns:
            valid_ml = results_df.dropna(subset=['actual_label', 'predicted_label_ML'])
            if len(valid_ml) > 0:
                ml_accuracy = (valid_ml['actual_label'] == valid_ml['predicted_label_ML']).mean()
                logger.info(f"ML accuracy: {ml_accuracy:.3f} ({len(valid_ml)} valid predictions)")
                
        if 'actual_label' in results_df.columns and 'Agent_Ensemble_Prediction' in results_df.columns:
            valid_agent = results_df.dropna(subset=['actual_label', 'Agent_Ensemble_Prediction'])
            if len(valid_agent) > 0:
                agent_accuracy = (valid_agent['actual_label'] == valid_agent['Agent_Ensemble_Prediction']).mean()
                logger.info(f"Agent accuracy: {agent_accuracy:.3f} ({len(valid_agent)} valid predictions)")
                

def main():
    parser = argparse.ArgumentParser(description="Generate Comprehensive Results Table")
    parser.add_argument("--input", required=True,
                       help="Input file pattern for ML predictions (e.g., data/predictions/final_predictions_*.csv)")
    parser.add_argument("--output", default="data/results_table.csv",
                       help="Output CSV file for results table")
    parser.add_argument("--db-path", default="data/sentiment_system.duckdb",
                       help="Path to DuckDB database")
    
    args = parser.parse_args()
    
    try:
        # Find input files
        input_files = glob.glob(args.input)
        if not input_files:
            raise ValueError(f"No files found matching pattern: {args.input}")
            
        # Use the most recent file if multiple matches
        input_file = sorted(input_files)[-1]
        logger.info(f"Using ML predictions file: {input_file}")
        
        # Initialize generator
        generator = ResultsTableGenerator(args.db_path)
        
        # Load ML predictions
        ml_predictions = generator.load_ml_predictions(input_file)
        
        # Generate results table
        results_df = generator.generate_results_table(ml_predictions)
        
        # Save results
        generator.save_results(results_df, args.output)
        
        logger.info("✅ Results table generation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Results table generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 