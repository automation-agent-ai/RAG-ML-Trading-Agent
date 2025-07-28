#!/usr/bin/env python3
"""
3-Stage ML Pipeline Prediction Script

Makes predictions using the trained 3-stage ML pipeline:
1. Apply text models to text features
2. Apply financial models to financial features  
3. Apply ensemble meta-models to prediction vectors

Usage:
    python predict_3stage_ml_pipeline.py --input-dir data/ml_features/balanced --models-dir models_3stage --output-dir data/predictions
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

from corrected_csv_feature_loader import CorrectedCSVFeatureLoader

class ThreeStageMLPredictor:
    """3-Stage ML Pipeline Predictor"""
    
    def __init__(self, models_dir: str, input_dir: str = "data/ml_features/balanced"):
        self.models_dir = Path(models_dir)
        self.input_dir = input_dir
        self.feature_loader = CorrectedCSVFeatureLoader(input_dir)
        
        # Define model names
        self.model_names = ['random_forest', 'logistic_regression', 'xgboost', 'lightgbm']
        
        # Storage for loaded models
        self.text_models = {}
        self.financial_models = {}
        self.ensemble_models = {}
        
    def load_all_models(self):
        """Load all trained models from the 3-stage pipeline"""
        logger.info("Loading all trained models...")
        
        # Load text models
        text_dir = self.models_dir / "text"
        for model_name in self.model_names:
            model_path = text_dir / f"text_{model_name}.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.text_models[model_name] = pickle.load(f)
                logger.info(f"  ✅ Loaded text model: {model_name}")
            else:
                logger.warning(f"  ⚠️ Text model not found: {model_path}")
        
        # Load financial models
        financial_dir = self.models_dir / "financial"
        for model_name in self.model_names:
            model_path = financial_dir / f"financial_{model_name}.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.financial_models[model_name] = pickle.load(f)
                logger.info(f"  ✅ Loaded financial model: {model_name}")
            else:
                logger.warning(f"  ⚠️ Financial model not found: {model_path}")
        
        # Load ensemble models
        ensemble_dir = self.models_dir / "ensemble"
        for model_name in self.model_names:
            model_path = ensemble_dir / f"ensemble_{model_name}.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.ensemble_models[model_name] = pickle.load(f)
                logger.info(f"  ✅ Loaded ensemble model: {model_name}")
            else:
                logger.warning(f"  ⚠️ Ensemble model not found: {model_path}")
        
        logger.info(f"Loaded {len(self.text_models)} text, {len(self.financial_models)} financial, {len(self.ensemble_models)} ensemble models")
        
    def predict_stage1_text(self, X_text: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Stage 1: Make predictions using text models"""
        logger.info("🔤 Stage 1: Making text-based predictions...")
        
        text_predictions = {}
        
        for model_name, model in self.text_models.items():
            # Get predictions and probabilities
            predictions = model.predict(X_text)
            probabilities = model.predict_proba(X_text)
            
            text_predictions[f'text_{model_name}_pred'] = predictions
            text_predictions[f'text_{model_name}_prob_0'] = probabilities[:, 0]
            text_predictions[f'text_{model_name}_prob_1'] = probabilities[:, 1]
            text_predictions[f'text_{model_name}_prob_2'] = probabilities[:, 2]
            
            logger.info(f"  ✅ Text {model_name}: {len(predictions)} predictions")
        
        return text_predictions
        
    def predict_stage2_financial(self, X_financial: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Stage 2: Make predictions using financial models"""
        logger.info("💰 Stage 2: Making financial-based predictions...")
        
        financial_predictions = {}
        
        for model_name, model in self.financial_models.items():
            # Get predictions and probabilities
            predictions = model.predict(X_financial)
            probabilities = model.predict_proba(X_financial)
            
            financial_predictions[f'financial_{model_name}_pred'] = predictions
            financial_predictions[f'financial_{model_name}_prob_0'] = probabilities[:, 0]
            financial_predictions[f'financial_{model_name}_prob_1'] = probabilities[:, 1]
            financial_predictions[f'financial_{model_name}_prob_2'] = probabilities[:, 2]
            
            logger.info(f"  ✅ Financial {model_name}: {len(predictions)} predictions")
        
        return financial_predictions
        
    def predict_stage3_ensemble(self, ensemble_features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Stage 3: Make predictions using ensemble meta-models"""
        logger.info("🏆 Stage 3: Making ensemble meta-predictions...")
        
        ensemble_predictions = {}
        
        for model_name, model in self.ensemble_models.items():
            # Get predictions and probabilities
            predictions = model.predict(ensemble_features)
            probabilities = model.predict_proba(ensemble_features)
            
            ensemble_predictions[f'ensemble_{model_name}_pred'] = predictions
            ensemble_predictions[f'ensemble_{model_name}_prob_0'] = probabilities[:, 0]
            ensemble_predictions[f'ensemble_{model_name}_prob_1'] = probabilities[:, 1]
            ensemble_predictions[f'ensemble_{model_name}_prob_2'] = probabilities[:, 2]
            
            logger.info(f"  ✅ Ensemble {model_name}: {len(predictions)} predictions")
        
        return ensemble_predictions
        
    def create_ensemble_features(self, text_predictions: Dict, financial_predictions: Dict, setup_ids: List[str]) -> pd.DataFrame:
        """Create rich ensemble features from text and financial predictions"""
        
        # Collect all probability columns (24 features total)
        ensemble_data = {}
        
        # Add text probabilities (4 models × 3 classes = 12 features)
        for model_name in self.model_names:
            if f'text_{model_name}_prob_0' in text_predictions:
                ensemble_data[f'text_{model_name}_prob_0'] = text_predictions[f'text_{model_name}_prob_0']
                ensemble_data[f'text_{model_name}_prob_1'] = text_predictions[f'text_{model_name}_prob_1']
                ensemble_data[f'text_{model_name}_prob_2'] = text_predictions[f'text_{model_name}_prob_2']
        
        # Add financial probabilities (4 models × 3 classes = 12 features)
        for model_name in self.model_names:
            if f'financial_{model_name}_prob_0' in financial_predictions:
                ensemble_data[f'financial_{model_name}_prob_0'] = financial_predictions[f'financial_{model_name}_prob_0']
                ensemble_data[f'financial_{model_name}_prob_1'] = financial_predictions[f'financial_{model_name}_prob_1']
                ensemble_data[f'financial_{model_name}_prob_2'] = financial_predictions[f'financial_{model_name}_prob_2']
        
        # Create DataFrame
        ensemble_features = pd.DataFrame(ensemble_data, index=setup_ids)
        
        # Add enhanced ensemble features
        
        # Add confidence measures (max probability)
        for model_name in self.model_names:
            if f'text_{model_name}_prob_0' in ensemble_features.columns:
                # Text model confidence
                ensemble_features[f'text_{model_name}_confidence'] = ensemble_features[[
                    f'text_{model_name}_prob_0', 
                    f'text_{model_name}_prob_1', 
                    f'text_{model_name}_prob_2'
                ]].max(axis=1)
                
                # Financial model confidence
                ensemble_features[f'financial_{model_name}_confidence'] = ensemble_features[[
                    f'financial_{model_name}_prob_0', 
                    f'financial_{model_name}_prob_1', 
                    f'financial_{model_name}_prob_2'
                ]].max(axis=1)
                
                # Confidence difference
                ensemble_features[f'{model_name}_confidence_diff'] = np.abs(
                    ensemble_features[f'text_{model_name}_confidence'] - 
                    ensemble_features[f'financial_{model_name}_confidence']
                )
                
                # Add agreement measures
                for i in range(3):  # For each class (0, 1, 2)
                    text_prob = ensemble_features[f'text_{model_name}_prob_{i}']
                    fund_prob = ensemble_features[f'financial_{model_name}_prob_{i}']
                    ensemble_features[f'{model_name}_class{i}_agreement'] = 1 - np.abs(text_prob - fund_prob)
        
        logger.info(f"Created ensemble features shape: {ensemble_features.shape}")
        return ensemble_features
        
    def run_complete_prediction(self) -> Dict[str, pd.DataFrame]:
        """Run the complete 3-stage prediction pipeline"""
        logger.info("🚀 Starting Complete 3-Stage ML Prediction Pipeline")
        
        # Load prediction data
        X_text, _ = self.feature_loader.load_text_features("prediction")
        X_financial, _ = self.feature_loader.load_financial_features("prediction")
        
        if X_text is None or X_financial is None:
            raise ValueError("Failed to load prediction data")
        
        # Ensure same setup IDs
        common_ids = list(set(X_text.index) & set(X_financial.index))
        X_text = X_text.loc[common_ids]
        X_financial = X_financial.loc[common_ids]
        
        logger.info(f"Using {len(common_ids)} common setup IDs for prediction")
        
        # Load all models
        self.load_all_models()
        
        # Stage 1: Text predictions
        text_predictions = self.predict_stage1_text(X_text)
        
        # Stage 2: Financial predictions
        financial_predictions = self.predict_stage2_financial(X_financial)
        
        # Create ensemble features
        ensemble_features = self.create_ensemble_features(text_predictions, financial_predictions, common_ids)
        
        # Stage 3: Ensemble predictions
        ensemble_predictions = self.predict_stage3_ensemble(ensemble_features)
        
        # Create result DataFrames
        results = {}
        
        # Text predictions DataFrame
        text_df_data = {col: values for col, values in text_predictions.items()}
        results['text'] = pd.DataFrame(text_df_data, index=common_ids)
        
        # Financial predictions DataFrame
        financial_df_data = {col: values for col, values in financial_predictions.items()}
        results['financial'] = pd.DataFrame(financial_df_data, index=common_ids)
        
        # Ensemble predictions DataFrame
        ensemble_df_data = {col: values for col, values in ensemble_predictions.items()}
        results['ensemble'] = pd.DataFrame(ensemble_df_data, index=common_ids)
        
        # Final predictions (average of ensemble models, converted back to -1, 0, 1)
        final_predictions = []
        final_confidences = []
        
        for i in range(len(common_ids)):
            # Average probabilities across ensemble models
            probs = np.zeros(3)
            for model_name in self.model_names:
                if f'ensemble_{model_name}_prob_0' in ensemble_predictions:
                    probs[0] += ensemble_predictions[f'ensemble_{model_name}_prob_0'][i]
                    probs[1] += ensemble_predictions[f'ensemble_{model_name}_prob_1'][i]
                    probs[2] += ensemble_predictions[f'ensemble_{model_name}_prob_2'][i]
            
            probs /= len(self.ensemble_models)  # Average
            
            # Get final prediction and confidence
            final_pred = np.argmax(probs) - 1  # Convert 0,1,2 back to -1,0,1
            final_conf = np.max(probs)
            
            final_predictions.append(final_pred)
            final_confidences.append(final_conf)
        
        # Final results DataFrame
        results['final'] = pd.DataFrame({
            'prediction': final_predictions,
            'confidence': final_confidences,
            'prob_negative': [ensemble_predictions[f'ensemble_{self.model_names[0]}_prob_0'][i] for i in range(len(common_ids))],
            'prob_neutral': [ensemble_predictions[f'ensemble_{self.model_names[0]}_prob_1'][i] for i in range(len(common_ids))], 
            'prob_positive': [ensemble_predictions[f'ensemble_{self.model_names[0]}_prob_2'][i] for i in range(len(common_ids))]
        }, index=common_ids)
        
        logger.info("✅ Complete 3-Stage ML Prediction Pipeline finished successfully!")
        return results
        
    def save_results(self, results: Dict[str, pd.DataFrame], output_dir: str):
        """Save prediction results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual result files
        for stage, df in results.items():
            filename = f"{stage}_predictions_{timestamp}.csv"
            file_path = output_path / filename
            df.to_csv(file_path)
            logger.info(f"💾 Saved {stage} predictions to {file_path}")
        
        # Create comprehensive summary report
        report_path = output_path / f"prediction_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write("3-STAGE ML PIPELINE PREDICTION REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overview
            f.write("PREDICTION OVERVIEW\n")
            f.write("-"*30 + "\n")
            f.write(f"Total setup IDs: {len(results['final'])}\n")
            f.write(f"Models directory: {self.models_dir}\n")
            f.write(f"Input directory: {self.input_dir}\n\n")
            
            # Final predictions
            if 'final' in results:
                f.write("FINAL PREDICTIONS\n")
                f.write("-"*30 + "\n")
                pred_counts = results['final']['prediction'].value_counts().sort_index()
                f.write(f"Prediction distribution:\n")
                
                # Convert 0-based to -1, 0, 1 labels for display
                class_names = ['Negative (-1)', 'Neutral (0)', 'Positive (1)']
                for i, count in pred_counts.items():
                    pct = count / len(results['final']) * 100
                    # Adjust index for display (prediction is already in -1, 0, 1 format)
                    class_name = class_names[i+1] if i >= -1 and i <= 1 else f"Unknown ({i})"
                    f.write(f"  {class_name}: {count} ({pct:.1f}%)\n")
                
                f.write(f"\nAverage confidence: {results['final']['confidence'].mean():.3f}\n\n")
                
                # Calculate high confidence predictions
                high_conf = results['final'][results['final']['confidence'] > 0.8]
                f.write(f"High confidence predictions (>0.8): {len(high_conf)} ({len(high_conf)/len(results['final'])*100:.1f}%)\n\n")
            
            # Domain-specific predictions
            for stage in ['text', 'financial', 'ensemble']:
                if stage in results:
                    f.write(f"{stage.upper()} MODEL PREDICTIONS\n")
                    f.write("-"*30 + "\n")
                    f.write(f"Number of models: {len([c for c in results[stage].columns if c.endswith('_pred')])}\n")
                    
                    # Calculate average predictions across models
                    pred_cols = [c for c in results[stage].columns if c.endswith('_pred')]
                    if pred_cols:
                        avg_preds = results[stage][pred_cols].mode(axis=1)[0]
                        pred_counts = avg_preds.value_counts().sort_index()
                        f.write(f"Average prediction distribution:\n")
                        for i, count in pred_counts.items():
                            pct = count / len(avg_preds) * 100
                            f.write(f"  Class {i}: {count} ({pct:.1f}%)\n")
                    
                    # Calculate average confidence
                    conf_cols = [c for c in results[stage].columns if 'confidence' in c]
                    if conf_cols:
                        avg_conf = results[stage][conf_cols].mean().mean()
                        f.write(f"Average confidence: {avg_conf:.3f}\n")
                    
                    f.write("\n")
            
            # Agreement analysis
            f.write("DOMAIN AGREEMENT ANALYSIS\n")
            f.write("-"*30 + "\n")
            
            try:
                # Get predictions from each domain's best model
                text_best = [c for c in results['text'].columns if c.endswith('_pred')][0]
                financial_best = [c for c in results['financial'].columns if c.endswith('_pred')][0]
                
                text_preds = results['text'][text_best]
                financial_preds = results['financial'][financial_best]
                
                # Calculate agreement
                agreement = (text_preds == financial_preds).mean()
                f.write(f"Text-Financial agreement: {agreement:.3f} ({agreement*100:.1f}%)\n")
                
                # Calculate disagreement types
                text_pos_fin_neg = ((text_preds == 2) & (financial_preds == 0)).sum()
                text_neg_fin_pos = ((text_preds == 0) & (financial_preds == 2)).sum()
                
                f.write(f"Text positive, Financial negative: {text_pos_fin_neg} cases\n")
                f.write(f"Text negative, Financial positive: {text_neg_fin_pos} cases\n\n")
            except:
                f.write("Could not calculate domain agreement\n\n")
            
            # Output files
            f.write("OUTPUT FILES\n")
            f.write("-"*30 + "\n")
            for stage, df in results.items():
                f.write(f"{stage}_predictions_{timestamp}.csv\n")
            f.write("\n")
            
            f.write("="*50 + "\n")
        
        logger.info(f"📄 Saved prediction report to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="3-Stage ML Pipeline Prediction")
    parser.add_argument("--input-dir", default="data/ml_features/balanced",
                       help="Directory containing prediction data CSV files")
    parser.add_argument("--models-dir", required=True,
                       help="Directory containing trained models")
    parser.add_argument("--output-dir", default="data/predictions",
                       help="Directory to save prediction results")
    parser.add_argument("--high-confidence-threshold", type=float, default=0.8,
                       help="Threshold for high confidence predictions (default: 0.8)")
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = ThreeStageMLPredictor(args.models_dir, args.input_dir)
        
        # Run predictions
        results = predictor.run_complete_prediction()
        
        # Save results
        predictor.save_results(results, args.output_dir)
        
        logger.info("🎉 Prediction pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 