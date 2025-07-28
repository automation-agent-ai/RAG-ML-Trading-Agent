# Ensemble Prediction Pipeline Improvements

This document summarizes the improvements made to the ensemble prediction pipeline to address data quality, model robustness, and reporting requirements.

## Key Improvements

1. **Balanced Dataset Sizes**
   - Ensured text and financial ML datasets have the same setup_ids and sample sizes
   - Target size of 600 setup_ids for both domains
   - Implemented in `balance_ml_datasets.py`

2. **Balanced Class Distribution**
   - Dynamic thresholds for class balancing based on outperformance_10d values
   - Configurable class ratios (default: 33% negative, 33% neutral, 34% positive)
   - Thresholds stored centrally and applied consistently to both training and prediction data
   - Implemented in `create_balanced_classes.py`

3. **Agent Ensemble Integration**
   - Added agent predictions from similarity_predictions table as features for ML models
   - Includes predicted_outperformance, confidence scores, and ratio metrics
   - Handles multiple domains with appropriate aggregation
   - Implemented in `integrate_agent_predictions.py`

4. **Cross-Validation for Model Training**
   - Stratified K-fold cross-validation for robust performance estimates
   - Model selection based on precision (as requested)
   - Proper data preprocessing (imputation and scaling)
   - Removal of target leakage (outperformance_10d excluded from features)
   - Implemented in `train_domain_models_cv.py`

5. **Comprehensive Results Table**
   - Combined ML predictions, agent ensemble predictions, and actual labels
   - Includes all requested metrics:
     - setup_id
     - actual_label
     - outperformance_10d
     - predicted_label_ML
     - confidence_score_ml
     - agent_ensemble_prediction
     - agent_predicted_outperformance
     - agent_confidence_score
     - domains_count
   - Implemented in `generate_results_table.py`

6. **Unified Pipeline**
   - Single consolidated pipeline controllable via step flags
   - Detailed terminal logs showing current steps
   - Runs in the conda environment 'sts'
   - Implemented in `run_complete_pipeline.py`

## Issues Addressed

1. **Data Size Discrepancy**
   - Found: text training had 393 setup_ids while financial had 5000
   - Database has 655-743 setup_ids across domains
   - Solution: Created `balance_ml_datasets.py` to ensure consistent sample sizes

2. **Domains Count Issue**
   - Found only 'news' domain in similarity_predictions table
   - Explanation: Other domains (fundamentals, analyst, userposts) were not populated in the similarity_predictions table
   - Solution: Pipeline is designed to handle multiple domains when they become available

3. **Class Imbalance**
   - Original data had very few neutral (class 1) samples
   - Solution: Dynamic thresholds based on sorted outperformance values to achieve desired class distribution

## Usage

Run the complete pipeline with:

```bash
python run_complete_pipeline.py \
  --text-train data/ml_features/text_ml_features_training_labeled.csv \
  --financial-train data/ml_features/financial_ml_features_training_labeled.csv \
  --text-predict data/ml_features/text_ml_features_prediction_labeled.csv \
  --financial-predict data/ml_features/financial_ml_features_prediction_labeled.csv \
  --output-dir data/ml_pipeline_output \
  --class-ratio 0.33,0.33,0.34 \
  --target-size 600
```

## Pipeline Steps

1. **Balance Datasets**
   - Ensures text and financial datasets have the same setup_ids
   - Target size of 600 setup_ids (configurable)

2. **Create Balanced Classes**
   - Calculates thresholds based on training data
   - Applies same thresholds to prediction data

3. **Integrate Agent Predictions**
   - Adds agent features from similarity_predictions table

4. **Train Models with Cross-Validation**
   - Uses 5-fold stratified cross-validation
   - Selects best model based on precision

5. **Make Ensemble Predictions**
   - Combines text and financial model predictions
   - Uses weighted voting based on model performance

6. **Generate Results Table**
   - Creates comprehensive table with all metrics

7. **Visualize Results**
   - Generates confusion matrices and performance charts

## Future Improvements

1. **Multiple Domain Support**
   - When more domains are available in similarity_predictions, the pipeline will automatically use them

2. **Advanced Ensemble Methods**
   - Consider implementing stacking or blending approaches

3. **Confidence Score Calculation**
   - Replace placeholder confidence scores with actual probabilities from models

4. **Time-Series Considerations**
   - Implement time-based train-test splitting consistently 