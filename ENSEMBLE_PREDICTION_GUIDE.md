# Ensemble Prediction Guide

This guide explains how to use domain-specific models (text and financial) to make ensemble predictions.

## Overview

The ensemble prediction system combines predictions from multiple domain-specific models to make more accurate predictions. The system consists of three main components:

1. **Domain-specific models**: Models trained on text and financial features separately.
2. **Ensemble predictor**: Combines predictions from domain-specific models using weighted voting.
3. **Visualization tools**: Visualize ensemble predictions and model performance.

## Workflow

1. **Train domain-specific models**:
   - Train text models using `train_domain_models.py` with text features.
   - Train financial models using `train_domain_models.py` with financial features.

2. **Make ensemble predictions**:
   - Use `ensemble_domain_predictions.py` to combine predictions from domain-specific models.

3. **Visualize results**:
   - Use `visualize_ensemble_results.py` to generate visualizations and reports.

## Usage

### 1. Train Domain-Specific Models

```bash
python train_domain_models.py --text-data data/ml_features/text_ml_features_training_labeled.csv --financial-data data/ml_features/financial_ml_features_training_labeled.csv --text-test-data data/ml_features/text_ml_features_prediction_labeled.csv --financial-test-data data/ml_features/financial_ml_features_prediction_labeled.csv --output-dir models
```

This will train the following models for each domain:
- Random Forest
- XGBoost
- Logistic Regression

### 2. Make Ensemble Predictions

```bash
python ensemble_domain_predictions.py --text-data data/ml_features/text_ml_features_prediction_labeled.csv --financial-data data/ml_features/financial_ml_features_prediction_labeled.csv --text-models-dir models/text --financial-models-dir models/financial --output-file data/ensemble_predictions.csv
```

This will:
- Load the latest trained models from each domain
- Make predictions with each model
- Combine predictions using weighted voting
- Save the ensemble predictions to a CSV file

### 3. Visualize Results

```bash
python visualize_ensemble_results.py --predictions data/ensemble_predictions.csv --output-dir models/ensemble
```

This will generate:
- Confusion matrix for ensemble predictions
- Model accuracy comparison
- Prediction distribution
- Model agreement matrix
- Summary report

## Results

After adjusting the ensemble weights to give more importance to the text XGBoost model, the ensemble model now achieves:

- **Accuracy**: 72.00% (up from 64.00%)
- **Precision**: 76.12% (down slightly from 78.09%)
- **Recall**: 72.00% (up from 64.00%)
- **F1 Score**: 71.41% (up significantly from 55.92%)

The text XGBoost model still performs slightly better with:

- **Accuracy**: 74.00%
- **Precision**: 77.71%
- **Recall**: 74.00%
- **F1 Score**: 73.78%

However, the ensemble model now performs much closer to the best individual model while potentially providing more robustness across different scenarios.

## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| text_xgboost | 74.00% | 77.71% | 74.00% | 73.78% |
| ensemble (optimized) | 72.00% | 76.12% | 72.00% | 71.41% |
| text_random_forest | 64.00% | 65.12% | 64.00% | 64.38% |
| financial_logistic_regression | 56.00% | 53.65% | 56.00% | 46.12% |
| financial_random_forest | 56.00% | 53.50% | 56.00% | 43.46% |
| financial_xgboost | 54.00% | 50.11% | 54.00% | 46.96% |
| text_logistic_regression | 52.00% | 57.53% | 52.00% | 52.28% |
| ensemble (initial) | 64.00% | 78.09% | 64.00% | 55.92% |

## Recommendations

1. **Further optimize ensemble weights**: Continue experimenting with different weight combinations.
2. **Feature engineering**: Improve financial features to enhance financial model performance.
3. **Model tuning**: Fine-tune hyperparameters for all models, especially the financial models.
4. **Data augmentation**: Collect more labeled data to improve model training.
5. **Advanced ensemble methods**: Consider implementing stacking or blending approaches.

## Next Steps

1. ✅ Update the ensemble weights in `ensemble_domain_predictions.py` to give more weight to the text XGBoost model.
2. Experiment with different ensemble methods (e.g., stacking) to improve performance.
3. Implement cross-validation to get more robust model performance estimates.
4. Improve financial features to enhance financial model performance.
5. Deploy the ensemble model to production. 