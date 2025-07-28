#!/usr/bin/env python3
"""
Make Agent Predictions with LLM-based Few-Shot Learning

This script makes agent predictions using GPT-4o-mini with few-shot learning
from similar historical cases. It extracts features first, then uses LLM
to make predictions based on those features and historical patterns.

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
from datetime import datetime

# Import threshold manager
from threshold_manager import ThresholdManager

# Import agent classes for LLM predictions
from agents.fundamentals.enhanced_fundamentals_agent_duckdb import EnhancedFundamentalsAgentDuckDB
from agents.news.enhanced_news_agent_duckdb import EnhancedNewsAgentDuckDB
from agents.analyst_recommendations.enhanced_analyst_recommendations_agent_duckdb import EnhancedAnalystRecommendationsAgentDuckDB
from agents.userposts.enhanced_userposts_agent_complete import EnhancedUserPostsAgentComplete

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentPredictionMaker:
    """Class for making agent predictions with LLM-based few-shot learning"""
    
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
        
        # Initialize LLM agents for predictions
        logger.info("Initializing LLM agents for predictions...")
        self.agents = {
            'fundamentals': EnhancedFundamentalsAgentDuckDB(
                db_path=db_path, 
                mode="prediction"
            ),
            'news': EnhancedNewsAgentDuckDB(
                db_path=db_path,
                mode="prediction"
            ),
            'analyst_recommendations': EnhancedAnalystRecommendationsAgentDuckDB(
                db_path=db_path,
                mode="prediction"
            ),
            'userposts': EnhancedUserPostsAgentComplete(
                db_path=db_path,
                mode="prediction"
            )
        }
        logger.info("LLM agents initialized successfully")
    
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
        Get LLM-based predictions from a specific domain
        
        Args:
            setup_ids: List of setup IDs
            domain: Domain name ('news', 'fundamentals', 'analyst_recommendations', 'userposts')
            
        Returns:
            DataFrame with LLM predictions including outperformance_10d
        """
        logger.info(f"Making LLM predictions for {len(setup_ids)} setups in {domain} domain")
        
        predictions = []
        agent = self.agents[domain]
        
        conn = None
        try:
            conn = duckdb.connect(self.db_path)
            
            # Get features for each setup and make LLM predictions
            for setup_id in setup_ids:
                try:
                    # Get extracted features for this setup
                    features = self._get_setup_features(conn, setup_id, domain)
                    
                    if features:
                        # Make LLM prediction using features and similar cases
                        prediction_result = agent.predict_with_llm(setup_id, features)
                        
                        # Convert to format expected by downstream processing
                        prediction_data = {
                            'setup_id': setup_id,
                            'weighted_avg_outperformance': prediction_result.get('predicted_outperformance_10d', 0.0),
                            'confidence_score': prediction_result.get('confidence_score', 0.1),
                            'prediction_class': prediction_result.get('prediction_class', 'NEUTRAL'),
                            'reasoning': prediction_result.get('reasoning', ''),
                            'prediction_method': prediction_result.get('prediction_method', 'llm_few_shot'),
                            'similar_cases_used': prediction_result.get('similar_cases_used', 0)
                        }
                        predictions.append(prediction_data)
                    else:
                        logger.warning(f"No features found for {setup_id} in {domain}")
                        # Add default prediction
                        predictions.append({
                            'setup_id': setup_id,
                            'weighted_avg_outperformance': 0.0,
                            'confidence_score': 0.1,
                            'prediction_class': 'NEUTRAL',
                            'reasoning': 'No features available',
                            'prediction_method': 'default',
                            'similar_cases_used': 0
                        })
                        
                except Exception as e:
                    logger.error(f"Error making prediction for {setup_id} in {domain}: {e}")
                    # Add error prediction
                    predictions.append({
                        'setup_id': setup_id,
                        'weighted_avg_outperformance': 0.0,
                        'confidence_score': 0.1,
                        'prediction_class': 'NEUTRAL',
                        'reasoning': f'Prediction error: {str(e)}',
                        'prediction_method': 'error',
                        'similar_cases_used': 0
                    })
                    
        finally:
            if conn:
                conn.close()
        
        # Convert to DataFrame
        df = pd.DataFrame(predictions)
        logger.info(f"Generated {len(df)} LLM predictions for {domain} domain")
        return df
    
    def _get_setup_features(self, conn, setup_id: str, domain: str) -> Dict[str, Any]:
        """Get extracted features for a setup from the appropriate domain table"""
        try:
            if domain == 'fundamentals':
                query = "SELECT * FROM fundamentals_features WHERE setup_id = ?"
            elif domain == 'news':
                query = "SELECT * FROM news_features WHERE setup_id = ?"
            elif domain == 'analyst_recommendations':
                query = "SELECT * FROM analyst_recommendations_features WHERE setup_id = ?"
            elif domain == 'userposts':
                query = "SELECT * FROM userposts_features WHERE setup_id = ?"
            else:
                return {}
                
            result = conn.execute(query, [setup_id]).fetchone()
            if result:
                # Convert to dictionary
                columns = [desc[0] for desc in conn.description]
                return dict(zip(columns, result))
            return {}
            
        except Exception as e:
            logger.error(f"Error getting features for {setup_id} in {domain}: {e}")
            return {}
    
    def make_ensemble_predictions(self, setup_ids: List[str]) -> pd.DataFrame:
        """
        Make ensemble predictions across all domains using confidence-weighted voting
        
        Args:
            setup_ids: List of setup IDs
            
        Returns:
            DataFrame with confidence-weighted ensemble predictions
        """
        logger.info("🔄 Making confidence-weighted ensemble predictions across all domains")
        
        # Get predictions from each domain
        domains = ['news', 'fundamentals', 'analyst_recommendations', 'userposts']
        domain_predictions = {}
        
        for domain in domains:
            domain_predictions[domain] = self.get_domain_predictions(setup_ids, domain)
            logger.info(f"✓ Retrieved {len(domain_predictions[domain])} predictions from {domain} domain")
        
        # Create a DataFrame with all setup IDs
        ensemble_df = pd.DataFrame({'setup_id': setup_ids})
        
        # Merge predictions from all domains
        for domain in domains:
            if len(domain_predictions[domain]) > 0:
                domain_df = domain_predictions[domain].copy()
                
                # Load thresholds for converting outperformance to labels
                neg_threshold, pos_threshold = self.threshold_manager.get_thresholds_for_prediction()
                
                # Convert outperformance to predicted labels using thresholds
                domain_df['predicted_label'] = domain_df['weighted_avg_outperformance'].apply(
                    lambda x: 1 if x >= pos_threshold else (-1 if x <= neg_threshold else 0)
                )
                
                # Select only essential columns for merging
                merge_columns = [
                    'setup_id',
                    'predicted_label',
                    'weighted_avg_outperformance', 
                    'confidence_score'
                ]
                domain_df = domain_df[merge_columns]
                
                # Rename columns for merging
                domain_df = domain_df.rename(columns={
                    'predicted_label': f'{domain}_predicted_label',
                    'weighted_avg_outperformance': f'{domain}_predicted_outperformance',
                    'confidence_score': f'{domain}_confidence_score'
                })
                ensemble_df = pd.merge(ensemble_df, domain_df, on='setup_id', how='left')
        
        logger.info(f"✓ Merged predictions from all domains for {len(ensemble_df)} setups")
        
        # Calculate ensemble prediction using confidence-weighted voting
        # 1. Confidence-weighted label prediction
        ensemble_df['predicted_label'] = None
        ensemble_df['predicted_outperformance'] = None
        ensemble_df['confidence_score'] = None
        
        # Process each setup individually for better logging
        for idx, row in ensemble_df.iterrows():
            setup_id = row['setup_id']
            
            # Collect domain predictions and confidences
            domain_labels = []
            domain_outperformances = []
            domain_confidences = []
            
            for domain in domains:
                label_col = f'{domain}_predicted_label'
                outperformance_col = f'{domain}_predicted_outperformance'
                confidence_col = f'{domain}_confidence_score'
                
                if label_col in row and pd.notna(row[label_col]) and confidence_col in row and pd.notna(row[confidence_col]):
                    domain_labels.append(row[label_col])
                    domain_outperformances.append(row[outperformance_col] if pd.notna(row[outperformance_col]) else 0.0)
                    domain_confidences.append(row[confidence_col])
            
            # Skip if no domain predictions
            if not domain_labels:
                continue
                
            # 1. Confidence-weighted label prediction
            weighted_sum = sum(label * conf for label, conf in zip(domain_labels, domain_confidences))
            total_confidence = sum(domain_confidences)
            weighted_label_value = weighted_sum / total_confidence if total_confidence > 0 else 0
            
            # Convert to discrete class using thresholds
            neg_threshold, pos_threshold = -0.33, 0.33  # Default thresholds if not using label balancing
            try:
                neg_threshold, pos_threshold = self.threshold_manager.get_thresholds_for_prediction()
            except:
                logger.warning("Could not load thresholds from threshold manager, using defaults")
            
            if weighted_label_value >= pos_threshold:
                ensemble_label = 1
            elif weighted_label_value <= neg_threshold:
                ensemble_label = -1
            else:
                ensemble_label = 0
                
            # 2. Confidence-weighted outperformance prediction
            weighted_outperformance = sum(outperf * conf for outperf, conf in zip(domain_outperformances, domain_confidences))
            weighted_outperformance = weighted_outperformance / total_confidence if total_confidence > 0 else 0
            
            # 3. Calculate ensemble confidence
            # Higher when domains agree, lower when they disagree
            domain_agreement = 1.0
            if len(domain_labels) > 1:
                # Calculate standard deviation of predictions (normalized)
                label_std = np.std(domain_labels) / 2  # Max std for [-1,0,1] is 1
                domain_agreement = max(0.5, 1.0 - label_std)
            
            # Final confidence is a combination of average domain confidence and agreement
            ensemble_confidence = (sum(domain_confidences) / len(domain_confidences)) * domain_agreement
            
            # Store results
            ensemble_df.at[idx, 'predicted_label'] = ensemble_label
            ensemble_df.at[idx, 'predicted_outperformance'] = weighted_outperformance
            ensemble_df.at[idx, 'confidence_score'] = ensemble_confidence
        
        # Count domains with predictions
        domain_count_cols = [f'{domain}_predicted_label' for domain in domains]
        ensemble_df['domains_count'] = ensemble_df[domain_count_cols].notna().sum(axis=1)
        
        # Log ensemble prediction stats
        logger.info(f"✅ Generated confidence-weighted ensemble predictions for {len(ensemble_df)} setups")
        
        # Log class distribution
        if len(ensemble_df) > 0:
            class_counts = ensemble_df['predicted_label'].value_counts()
            logger.info("📊 Class distribution for ensemble predictions:")
            for cls, count in class_counts.items():
                if pd.notna(cls):
                    logger.info(f"  - Class {cls}: {count} ({count/len(ensemble_df):.1%})")
            
            # Log outperformance stats
            valid_outperformance = ensemble_df['predicted_outperformance'].dropna()
            if len(valid_outperformance) > 0:
                logger.info("📈 Outperformance prediction stats:")
                logger.info(f"  - Mean: {valid_outperformance.mean():.2f}%")
                logger.info(f"  - Min: {valid_outperformance.min():.2f}%")
                logger.info(f"  - Max: {valid_outperformance.max():.2f}%")
                logger.info(f"  - Median: {valid_outperformance.median():.2f}%")
            
            # Log confidence stats
            valid_confidence = ensemble_df['confidence_score'].dropna()
            if len(valid_confidence) > 0:
                logger.info("🎯 Confidence score stats:")
                logger.info(f"  - Mean: {valid_confidence.mean():.2f}")
                logger.info(f"  - Min: {valid_confidence.min():.2f}")
                logger.info(f"  - Max: {valid_confidence.max():.2f}")
                logger.info(f"  - Median: {valid_confidence.median():.2f}")
            
            # Log domain count stats
            logger.info("🔄 Domain contribution stats:")
            domain_counts = ensemble_df['domains_count'].value_counts().sort_index()
            for count, occurrences in domain_counts.items():
                logger.info(f"  - {count} domains: {occurrences} setups ({occurrences/len(ensemble_df):.1%})")
        
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
            
            # First, let's check what's in the table before we start
            existing_count = conn.execute("SELECT COUNT(*) FROM similarity_predictions").fetchone()[0]
            logger.info(f"Before saving: {existing_count} records in similarity_predictions table")
            
            # Delete existing predictions for these setups
            setup_ids = predictions_df['setup_id'].tolist()
            conn.execute("DELETE FROM similarity_predictions WHERE setup_id = ANY(?)", [setup_ids])
            
            # Check after delete
            after_delete_count = conn.execute("SELECT COUNT(*) FROM similarity_predictions").fetchone()[0]
            logger.info(f"After deleting existing predictions: {after_delete_count} records in similarity_predictions table")
            
            # Rename confidence_score to confidence to match the table schema
            predictions_df = predictions_df.rename(columns={'confidence_score': 'confidence'})
            
            # Insert ensemble predictions
            ensemble_data = predictions_df[['setup_id', 'predicted_outperformance', 'confidence']].copy()
            ensemble_data['domain'] = 'ensemble'
            
            # Add additional required columns
            ensemble_data['positive_ratio'] = 0.7  # Default value
            ensemble_data['negative_ratio'] = 0.3  # Default value
            ensemble_data['neutral_ratio'] = 0.0  # Default value
            ensemble_data['similar_cases_count'] = 10  # Default value
            ensemble_data['prediction_timestamp'] = datetime.now().isoformat()
            
            # Insert ensemble predictions
            logger.info(f"Inserting {len(ensemble_data)} ensemble predictions")
            conn.execute("""
                INSERT INTO similarity_predictions 
                (setup_id, domain, predicted_outperformance, confidence, positive_ratio, 
                 negative_ratio, neutral_ratio, similar_cases_count, prediction_timestamp)
                SELECT 
                    setup_id, domain, predicted_outperformance, confidence, positive_ratio,
                    negative_ratio, neutral_ratio, similar_cases_count, prediction_timestamp
                FROM ensemble_data
            """)
            
            # Check after ensemble insert
            after_ensemble_count = conn.execute("SELECT COUNT(*) FROM similarity_predictions").fetchone()[0]
            logger.info(f"After inserting ensemble predictions: {after_ensemble_count} records in similarity_predictions table")
            
            # Insert domain predictions
            domains = ['news', 'fundamentals', 'analyst_recommendations', 'userposts']
            for domain in domains:
                # Get the domain-specific columns
                label_col = f'{domain}_predicted_label'
                outperformance_col = f'{domain}_predicted_outperformance'
                confidence_col = f'{domain}_confidence_score'  # Note: This is still _confidence_score, not _confidence
                
                # Check if these columns exist in the predictions DataFrame
                logger.info(f"Checking domain {domain}:")
                logger.info(f"  - Label column exists: {label_col in predictions_df.columns}")
                logger.info(f"  - Outperformance column exists: {outperformance_col in predictions_df.columns}")
                logger.info(f"  - Confidence column exists: {confidence_col in predictions_df.columns}")
                
                # Only proceed if the outperformance and confidence columns exist
                if outperformance_col in predictions_df.columns and confidence_col in predictions_df.columns:
                    # Create a DataFrame for this domain's predictions
                    domain_data = predictions_df[['setup_id', outperformance_col, confidence_col]].copy()
                    domain_data = domain_data.rename(columns={
                        outperformance_col: 'predicted_outperformance',
                        confidence_col: 'confidence'  # Rename to match table schema
                    })
                    domain_data['domain'] = domain
                    
                    # Add additional required columns
                    domain_data['positive_ratio'] = 0.7  # Default value
                    domain_data['negative_ratio'] = 0.3  # Default value
                    domain_data['neutral_ratio'] = 0.0  # Default value
                    domain_data['similar_cases_count'] = 10  # Default value
                    domain_data['prediction_timestamp'] = datetime.now().isoformat()
                    
                    # Drop rows with missing predictions
                    domain_data = domain_data.dropna(subset=['predicted_outperformance'])
                    
                    # Insert this domain's predictions
                    if len(domain_data) > 0:
                        logger.info(f"Inserting {len(domain_data)} {domain} predictions")
                        conn.execute("""
                            INSERT INTO similarity_predictions 
                            (setup_id, domain, predicted_outperformance, confidence, positive_ratio, 
                             negative_ratio, neutral_ratio, similar_cases_count, prediction_timestamp)
                            SELECT 
                                setup_id, domain, predicted_outperformance, confidence, positive_ratio,
                                negative_ratio, neutral_ratio, similar_cases_count, prediction_timestamp
                            FROM domain_data
                        """)
                        
                        # Check after domain insert
                        after_domain_count = conn.execute("SELECT COUNT(*) FROM similarity_predictions").fetchone()[0]
                        logger.info(f"After inserting {domain} predictions: {after_domain_count} records in similarity_predictions table")
                    else:
                        logger.warning(f"No valid {domain} predictions to insert")
                else:
                    logger.warning(f"Missing required columns for {domain} predictions")
            
            # Final check
            final_count = conn.execute("SELECT COUNT(*) FROM similarity_predictions").fetchone()[0]
            logger.info(f"Final count: {final_count} records in similarity_predictions table")
            
            # Check domain distribution
            domain_counts = conn.execute("SELECT domain, COUNT(*) FROM similarity_predictions GROUP BY domain").df()
            logger.info(f"Domain distribution in similarity_predictions table:")
            for _, row in domain_counts.iterrows():
                logger.info(f"  - {row['domain']}: {row['count_star()']} records")
            
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

if __name__ == "__main__":
    main() 