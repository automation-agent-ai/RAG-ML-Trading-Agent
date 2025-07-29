#!/usr/bin/env python3
"""
3-Stage ML Prediction Pipeline with Confidence-Weighted Ensemble

Stage 1: Text ML predictions (Random Forest, XGBoost, LightGBM, Logistic Regression)
Stage 2: Financial ML predictions (Random Forest, XGBoost, LightGBM, Logistic Regression)  
Stage 3: Ensemble ML predictions using confidence-weighted voting

Usage:
    python predict_3stage_ml_pipeline.py --input-dir data/ml_features/balanced --model-dir models --output-dir data/predictions
"""

import os
import sys
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Suppress LightGBM warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

from corrected_csv_feature_loader import CorrectedCSVFeatureLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThreeStageMLPredictor:
    """3-Stage ML Predictor with Confidence-Weighted Ensemble"""
    
    def __init__(self, model_dir: str = "models", output_dir: str = "data/predictions"):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        
        # Create prediction output directories
        self.text_pred_dir = self.output_dir / "text_ml"
        self.financial_pred_dir = self.output_dir / "financial_ml"
        self.ensemble_pred_dir = self.output_dir / "ensemble"
        
        for dir_path in [self.text_pred_dir, self.financial_pred_dir, self.ensemble_pred_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize feature loader
        self.feature_loader = CorrectedCSVFeatureLoader()
        
        # Model names
        self.model_names = ['random_forest', 'logistic_regression', 'xgboost', 'lightgbm']
        
        # Load models
        self.text_models = {}
        self.financial_models = {}
        self.ensemble_models = {}
        
        self.load_all_models()
        
    def load_all_models(self):
        """Load all trained models"""
        logger.info("📂 Loading trained models...")
        
        # Load text models
        for model_name in self.model_names:
            model_path = self.model_dir / "text" / f"text_{model_name}.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.text_models[model_name] = pickle.load(f)
                logger.info(f"  ✅ Loaded text {model_name}")
            else:
                logger.warning(f"  ⚠️ Text {model_name} model not found at {model_path}")
                
        # Load financial models
        for model_name in self.model_names:
            model_path = self.model_dir / "financial" / f"financial_{model_name}.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.financial_models[model_name] = pickle.load(f)
                logger.info(f"  ✅ Loaded financial {model_name}")
            else:
                logger.warning(f"  ⚠️ Financial {model_name} model not found at {model_path}")
                
        # Load ensemble models
        for model_name in self.model_names:
            model_path = self.model_dir / "ensemble" / f"ensemble_{model_name}.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.ensemble_models[model_name] = pickle.load(f)
                logger.info(f"  ✅ Loaded ensemble {model_name}")
            else:
                logger.warning(f"  ⚠️ Ensemble {model_name} model not found at {model_path}")
                
        logger.info(f"📊 Loaded {len(self.text_models)} text, {len(self.financial_models)} financial, {len(self.ensemble_models)} ensemble models")
        
    def predict_stage1_text(self, X_text: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Stage 1: Make text-based ML predictions"""
        logger.info("🔤 Stage 1: Making text-based predictions...")
        
        text_predictions = {}
        individual_results = {}
        
        for model_name, model in self.text_models.items():
            logger.info(f"  🎯 Predicting with text {model_name}")
            
            # Get predictions and probabilities
            predictions = model.predict(X_text)
            probabilities = model.predict_proba(X_text)
            
            # Convert back to -1, 0, 1 format
            predictions_converted = predictions - 1  # 0,1,2 -> -1,0,1
            
            # Calculate confidence (max probability)
            confidences = np.max(probabilities, axis=1)
            
            # Store predictions
            text_predictions[f'text_{model_name}_pred'] = predictions_converted
            text_predictions[f'text_{model_name}_confidence'] = confidences
            text_predictions[f'text_{model_name}_prob_0'] = probabilities[:, 0]  # Negative
            text_predictions[f'text_{model_name}_prob_1'] = probabilities[:, 1]  # Neutral
            text_predictions[f'text_{model_name}_prob_2'] = probabilities[:, 2]  # Positive
            
            # Individual model results
            individual_results[model_name] = pd.DataFrame({
                'prediction': predictions_converted,
                'confidence': confidences,
                'prob_negative': probabilities[:, 0],
                'prob_neutral': probabilities[:, 1],
                'prob_positive': probabilities[:, 2]
            }, index=X_text.index)
            
            # Save individual model results
            output_file = self.text_pred_dir / f"text_{model_name}_predictions.csv"
            individual_results[model_name].to_csv(output_file)
            logger.info(f"    💾 Saved to {output_file}")
        
        return text_predictions, individual_results
        
    def predict_stage2_financial(self, X_financial: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Stage 2: Make financial-based ML predictions"""
        logger.info("💰 Stage 2: Making financial-based predictions...")
        
        financial_predictions = {}
        individual_results = {}
        
        for model_name, model in self.financial_models.items():
            logger.info(f"  🎯 Predicting with financial {model_name}")
            
            # Get predictions and probabilities
            predictions = model.predict(X_financial)
            probabilities = model.predict_proba(X_financial)
            
            # Convert back to -1, 0, 1 format
            predictions_converted = predictions - 1  # 0,1,2 -> -1,0,1
            
            # Calculate confidence (max probability)
            confidences = np.max(probabilities, axis=1)
            
            # Store predictions
            financial_predictions[f'financial_{model_name}_pred'] = predictions_converted
            financial_predictions[f'financial_{model_name}_confidence'] = confidences
            financial_predictions[f'financial_{model_name}_prob_0'] = probabilities[:, 0]  # Negative
            financial_predictions[f'financial_{model_name}_prob_1'] = probabilities[:, 1]  # Neutral
            financial_predictions[f'financial_{model_name}_prob_2'] = probabilities[:, 2]  # Positive
            
            # Individual model results
            individual_results[model_name] = pd.DataFrame({
                'prediction': predictions_converted,
                'confidence': confidences,
                'prob_negative': probabilities[:, 0],
                'prob_neutral': probabilities[:, 1],
                'prob_positive': probabilities[:, 2]
            }, index=X_financial.index)
            
            # Save individual model results
            output_file = self.financial_pred_dir / f"financial_{model_name}_predictions.csv"
            individual_results[model_name].to_csv(output_file)
            logger.info(f"    💾 Saved to {output_file}")
        
        return financial_predictions, individual_results
        
    def create_ensemble_features(self, text_predictions: Dict, financial_predictions: Dict, setup_ids: List[str]) -> pd.DataFrame:
        """Create ensemble features from text and financial predictions"""
        logger.info("🔗 Creating ensemble features...")
        
        # Load the training feature order to ensure exact match
        try:
            with open(self.model_dir / "ensemble" / "ensemble_data.pkl", 'rb') as f:
                training_data = pickle.load(f)
                training_feature_names = training_data['feature_names']
            logger.info(f"Loaded training feature order: {len(training_feature_names)} features")
        except:
            logger.warning("Could not load training feature order, using default order")
            training_feature_names = None
        
        # Create ensemble features in the exact same order as training
        ensemble_features = pd.DataFrame(index=range(len(setup_ids)))
        
        if training_feature_names:
            # Use exact training order
            for feature_name in training_feature_names:
                if feature_name in text_predictions:
                    ensemble_features[feature_name] = text_predictions[feature_name]
                elif feature_name in financial_predictions:
                    ensemble_features[feature_name] = financial_predictions[feature_name]
                else:
                    # This is a derived feature - calculate it
                    ensemble_features[feature_name] = self._calculate_derived_feature(
                        feature_name, ensemble_features, text_predictions, financial_predictions
                    )
        else:
            # Fallback: create features in expected order
            # Based on training pattern: text_prob, financial_prob for each model, then derived features
            for model_name in self.model_names:
                # Add probability features
                for class_idx in [0, 1, 2]:
                    # Text probabilities first
                    text_prob_key = f'text_{model_name}_prob_{class_idx}'
                    if text_prob_key in text_predictions:
                        ensemble_features[text_prob_key] = text_predictions[text_prob_key]
                    
                    # Then financial probabilities
                    financial_prob_key = f'financial_{model_name}_prob_{class_idx}'
                    if financial_prob_key in financial_predictions:
                        ensemble_features[financial_prob_key] = financial_predictions[financial_prob_key]
            
            # Add confidence features
            for model_name in self.model_names:
                text_conf_key = f'text_{model_name}_confidence'
                if text_conf_key in text_predictions:
                    ensemble_features[text_conf_key] = text_predictions[text_conf_key]
                
                financial_conf_key = f'financial_{model_name}_confidence'
                if financial_conf_key in financial_predictions:
                    ensemble_features[financial_conf_key] = financial_predictions[financial_conf_key]
            
            # Add derived features
            for model_name in self.model_names:
                text_conf_key = f'text_{model_name}_confidence'
                financial_conf_key = f'financial_{model_name}_confidence'
                
                if text_conf_key in ensemble_features.columns and financial_conf_key in ensemble_features.columns:
                    # Confidence difference
                    ensemble_features[f'{model_name}_confidence_diff'] = np.abs(
                        ensemble_features[text_conf_key] - 
                        ensemble_features[financial_conf_key]
                    )
                    
                    # Add agreement measures
                    for i in range(3):  # For each class (0, 1, 2)
                        text_prob_key = f'text_{model_name}_prob_{i}'
                        financial_prob_key = f'financial_{model_name}_prob_{i}'
                        
                        if text_prob_key in ensemble_features.columns and financial_prob_key in ensemble_features.columns:
                            text_prob = ensemble_features[text_prob_key]
                            financial_prob = ensemble_features[financial_prob_key]
                            ensemble_features[f'{model_name}_class{i}_agreement'] = 1 - np.abs(text_prob - financial_prob)
        
        logger.info(f"Created ensemble features shape: {ensemble_features.shape}")
        logger.info(f"Feature order matches training: {training_feature_names is not None}")
        return ensemble_features
        
    def _calculate_derived_feature(self, feature_name: str, ensemble_features: pd.DataFrame, 
                                 text_predictions: Dict, financial_predictions: Dict) -> pd.Series:
        """Calculate derived features (confidence differences, agreements)"""
        
        # Extract model name from feature name
        parts = feature_name.split('_')
        
        if 'confidence_diff' in feature_name:
            # Extract model name (everything before '_confidence_diff')
            model_name = feature_name.replace('_confidence_diff', '')
            text_conf_key = f'text_{model_name}_confidence'
            financial_conf_key = f'financial_{model_name}_confidence'
            
            if text_conf_key in text_predictions and financial_conf_key in financial_predictions:
                return np.abs(text_predictions[text_conf_key] - financial_predictions[financial_conf_key])
        
        elif 'agreement' in feature_name:
            # Extract model name and class (e.g., 'random_forest_class0_agreement')
            parts = feature_name.split('_')
            class_idx = int(parts[-2][-1])  # Extract the number from 'class0', 'class1', etc.
            model_name = '_'.join(parts[:-2])  # Everything before '_classX_agreement'
            
            text_prob_key = f'text_{model_name}_prob_{class_idx}'
            financial_prob_key = f'financial_{model_name}_prob_{class_idx}'
            
            if text_prob_key in text_predictions and financial_prob_key in financial_predictions:
                return 1 - np.abs(text_predictions[text_prob_key] - financial_predictions[financial_prob_key])
        
        # Default: return zeros if we can't calculate
        return pd.Series(np.zeros(len(ensemble_features)), index=ensemble_features.index)
        
    def predict_stage3_ensemble(self, ensemble_features: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Stage 3: Make ensemble predictions with confidence weighting"""
        logger.info("🏆 Stage 3: Making ensemble predictions...")
        
        ensemble_predictions = {}
        individual_results = {}
        
        for model_name, model in self.ensemble_models.items():
            logger.info(f"  🎯 Predicting with ensemble {model_name}")
            
            # Get predictions and probabilities
            predictions = model.predict(ensemble_features)
            probabilities = model.predict_proba(ensemble_features)
            
            # Convert back to -1, 0, 1 format
            predictions_converted = predictions - 1  # 0,1,2 -> -1,0,1
            
            # Calculate confidence (max probability)
            confidences = np.max(probabilities, axis=1)
            
            # Store predictions
            ensemble_predictions[f'ensemble_{model_name}_pred'] = predictions_converted
            ensemble_predictions[f'ensemble_{model_name}_confidence'] = confidences
            ensemble_predictions[f'ensemble_{model_name}_prob_0'] = probabilities[:, 0]  # Negative
            ensemble_predictions[f'ensemble_{model_name}_prob_1'] = probabilities[:, 1]  # Neutral
            ensemble_predictions[f'ensemble_{model_name}_prob_2'] = probabilities[:, 2]  # Positive
            
            # Individual model results
            individual_results[model_name] = pd.DataFrame({
                'prediction': predictions_converted,
                'confidence': confidences,
                'prob_negative': probabilities[:, 0],
                'prob_neutral': probabilities[:, 1],
                'prob_positive': probabilities[:, 2]
            }, index=ensemble_features.index)
            
            # Save individual model results
            output_file = self.ensemble_pred_dir / f"ensemble_{model_name}_predictions.csv"
            individual_results[model_name].to_csv(output_file)
            logger.info(f"    💾 Saved to {output_file}")
        
        return ensemble_predictions, individual_results
        
    def confidence_weighted_ensemble(self, ensemble_predictions: Dict, setup_ids: List[str]) -> pd.DataFrame:
        """Create confidence-weighted final predictions"""
        logger.info("⚖️ Computing confidence-weighted ensemble predictions...")
        
        final_predictions = []
        final_confidences = []
        final_outperformances = []
        
        # Mapping from prediction classes to outperformance values
        class_to_outperformance = {-1: -5.0, 0: 0.0, 1: 5.0}  # Approximate mapping
        
        for i in range(len(setup_ids)):
            # Collect predictions and confidences
            predictions = []
            confidences = []
            
            for model_name in self.model_names:
                pred_key = f'ensemble_{model_name}_pred'
                conf_key = f'ensemble_{model_name}_confidence'
                
                if pred_key in ensemble_predictions and conf_key in ensemble_predictions:
                    predictions.append(ensemble_predictions[pred_key][i])
                    confidences.append(ensemble_predictions[conf_key][i])
            
            if len(predictions) == 0:
                final_predictions.append(0)
                final_confidences.append(0.33)
                final_outperformances.append(0.0)
                continue
            
            # Method 1: Confidence-weighted voting for classification
            weighted_sum = 0
            total_weight = 0
            
            for pred, conf in zip(predictions, confidences):
                weighted_sum += pred * conf
                total_weight += conf
            
            weighted_prediction = weighted_sum / total_weight if total_weight > 0 else 0
            
            # Convert to discrete class
            if weighted_prediction >= 0.33:
                final_pred = 1    # Positive
            elif weighted_prediction <= -0.33:
                final_pred = -1   # Negative
            else:
                final_pred = 0    # Neutral
            
            # Method 2: Confidence-weighted outperformance prediction
            weighted_outperformance = sum(class_to_outperformance[pred] * conf 
                                        for pred, conf in zip(predictions, confidences))
            weighted_outperformance /= total_weight if total_weight > 0 else 1
            
            # Final confidence as weighted average of individual confidences
            final_conf = total_weight / len(confidences) if len(confidences) > 0 else 0.33
            
            final_predictions.append(final_pred)
            final_confidences.append(final_conf)
            final_outperformances.append(weighted_outperformance)
        
        # Create final results DataFrame
        final_results = pd.DataFrame({
            'prediction': final_predictions,
            'confidence': final_confidences,
            'predicted_outperformance_10d': final_outperformances
        }, index=setup_ids)
        
        logger.info(f"✅ Generated {len(final_predictions)} confidence-weighted predictions")
        logger.info(f"📊 Prediction distribution: {pd.Series(final_predictions).value_counts().to_dict()}")
        logger.info(f"📊 Mean confidence: {np.mean(final_confidences):.3f}")
        logger.info(f"📊 Mean predicted outperformance: {np.mean(final_outperformances):.3f}%")
        
        return final_results
        
    def run_complete_prediction(self) -> Dict[str, pd.DataFrame]:
        """Run the complete 3-stage prediction pipeline"""
        logger.info("🚀 Starting Complete 3-Stage ML Prediction Pipeline")
        
        # Load prediction data
        X_text, _ = self.feature_loader.load_text_features("prediction")
        X_financial, _ = self.feature_loader.load_financial_features("prediction")
        
        if X_text is None or X_financial is None:
            raise ValueError("Failed to load prediction features")
        
        # Get common setup IDs
        common_ids = list(set(X_text.index) & set(X_financial.index))
        logger.info(f"📊 Found {len(common_ids)} common setup IDs for prediction")
        
        # Filter to common IDs
        X_text_common = X_text.loc[common_ids]
        X_financial_common = X_financial.loc[common_ids]
        
        # Stage 1: Text predictions
        text_predictions, text_individual = self.predict_stage1_text(X_text_common)
        
        # Stage 2: Financial predictions
        financial_predictions, financial_individual = self.predict_stage2_financial(X_financial_common)
        
        # Stage 3: Create ensemble features
        ensemble_features = self.create_ensemble_features(text_predictions, financial_predictions, common_ids)
        
        # Stage 3: Ensemble predictions
        ensemble_predictions, ensemble_individual = self.predict_stage3_ensemble(ensemble_features)
        
        # Final: Confidence-weighted ensemble
        final_results = self.confidence_weighted_ensemble(ensemble_predictions, common_ids)
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_file = self.output_dir / f"final_predictions_{timestamp}.csv"
        final_results.to_csv(final_output_file)
        logger.info(f"💾 Saved final predictions to {final_output_file}")
        
        # Create summary report
        self._generate_prediction_summary(text_individual, financial_individual, ensemble_individual, final_results)
        
        logger.info("✅ Complete 3-Stage ML Prediction Pipeline finished successfully!")
        
        return {
            'text': text_individual,
            'financial': financial_individual, 
            'ensemble': ensemble_individual,
            'final': final_results
        }
        
    def _generate_prediction_summary(self, text_results: Dict, financial_results: Dict, 
                                   ensemble_results: Dict, final_results: pd.DataFrame):
        """Generate prediction summary report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"prediction_summary_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("3-STAGE ML PREDICTION SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset overview
            f.write("PREDICTION DATASET OVERVIEW\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total predictions: {len(final_results)}\n")
            f.write(f"Text models: {len(text_results)}\n")
            f.write(f"Financial models: {len(financial_results)}\n")
            f.write(f"Ensemble models: {len(ensemble_results)}\n\n")
            
            # Final prediction distribution
            f.write("FINAL PREDICTION DISTRIBUTION\n")
            f.write("-" * 30 + "\n")
            pred_counts = final_results['prediction'].value_counts()
            for pred, count in pred_counts.items():
                pct = count / len(final_results) * 100
                label = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}[pred]
                f.write(f"{label}: {count} predictions ({pct:.1f}%)\n")
            f.write("\n")
            
            # Confidence statistics
            f.write("CONFIDENCE STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Mean confidence: {final_results['confidence'].mean():.3f}\n")
            f.write(f"Median confidence: {final_results['confidence'].median():.3f}\n")
            f.write(f"Min confidence: {final_results['confidence'].min():.3f}\n")
            f.write(f"Max confidence: {final_results['confidence'].max():.3f}\n\n")
            
            # Output files
            f.write("OUTPUT FILES\n")
            f.write("-" * 30 + "\n")
            f.write("Individual Model Predictions:\n")
            f.write("- ml/prediction/text_ml/text_[model]_predictions.csv\n")
            f.write("- ml/prediction/financial_ml/financial_[model]_predictions.csv\n")
            f.write("- ml/prediction/ensemble/ensemble_[model]_predictions.csv\n")
            f.write("\nFinal Results:\n")
            f.write(f"- {final_results.index.name}_predictions_{timestamp}.csv\n\n")
            
            f.write("=" * 50 + "\n")
            
        logger.info(f"📄 Prediction summary saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='3-Stage ML Prediction Pipeline')
    parser.add_argument('--input-dir', default='data/ml_features/balanced', 
                       help='Directory containing prediction feature CSV files')
    parser.add_argument('--model-dir', default='models/3stage_fixed', 
                       help='Directory containing trained models')
    parser.add_argument('--output-dir', default='ml/prediction', 
                       help='Output directory for predictions')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ThreeStageMLPredictor(
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
    
    # Update feature loader with input directory
    predictor.feature_loader = CorrectedCSVFeatureLoader(args.input_dir)
    
    try:
        results = predictor.run_complete_prediction()
        logger.info("🎉 Prediction pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Prediction pipeline failed with error: {e}")
        raise


if __name__ == '__main__':
    main() 