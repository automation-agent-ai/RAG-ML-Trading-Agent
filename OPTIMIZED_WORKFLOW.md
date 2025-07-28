# Optimized ML Pipeline Workflow

This guide provides a step-by-step walkthrough of the optimized workflow for training/prediction separation in the ML pipeline, focusing on efficiency and preventing data leakage.

## Workflow Overview

The optimized workflow is designed to:

1. Maintain a clear separation between training and prediction data
2. Avoid unnecessary re-computation of features
3. Enable proper evaluation of prediction performance
4. Support ensemble prediction across multiple domains
5. Generate comprehensive ML feature tables for model training
6. Train and evaluate machine learning models
7. Ensure consistent label format (-1, 0, 1) across all datasets
8. Maintain balanced class distribution using dynamic thresholds
9. Generate comprehensive results tables with ML and agent predictions

## Workflow Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Select Setups  │────▶│ Preserve Data   │────▶│ Remove from     │
│  for Prediction │     │                 │     │ Training        │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Restore Data   │◀────│ Evaluate        │◀────│ Reset Similarity│
│  (when needed)  │     │ Predictions     │     │ Features        │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │                 │
                                               │ Create          │
                                               │ Embeddings      │
                                               │ & Features      │
                                               │                 │
                                               └────────┬────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │                 │
                                               │ Ensemble        │
                                               │ Prediction      │
                                               │                 │
                                               └────────┬────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │                 │
                                               │ Extract ML      │
                                               │ Features        │
                                               │                 │
                                               └────────┬────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐     ┌─────────────────┐
                                               │                 │     │                 │
                                               │ Add Labels &    │────▶│ Balance         │
                                               │ Balance Classes │     │ Datasets        │
                                               │                 │     │                 │
                                               └────────┬────────┘     └────────┬────────┘
                                                        │                       │
                                                        ▼                       ▼
                                               ┌─────────────────┐     ┌─────────────────┐
                                               │                 │     │                 │
                                               │ Train ML        │────▶│ Generate        │
                                               │ Models          │     │ Results Table   │
                                               │                 │     │                 │
                                               └─────────────────┘     └─────────────────┘
```

## Preparation: Offline Model Caching (Recommended)

To avoid HTTP 429 errors and rate limiting issues with Hugging Face, we recommend setting up offline model caching before running the pipeline:

```bash
# Step 1: Download and cache the model
python download_models.py

# Step 2: Patch the pipeline to use cached models
python use_cached_model.py
```

See the [Offline Model Caching Guide](OFFLINE_MODEL_GUIDE.md) for more details.

## Detailed Steps

### 1. Select Setups for Training and Prediction

First, choose which setups to use for prediction (typically a random subset of setups with complete data).

```bash
python create_prediction_list.py --count 100 --output data/prediction_setups.txt
```

**Options:**
- `--count`: Number of setups to select
- `--db-path`: Path to DuckDB database
- `--output`: Output file for setup IDs
- `--random-seed`: Random seed for reproducibility

Then, create a list of training setups by excluding the prediction setups from all available setups:

```bash
python create_training_list.py --prediction-file data/prediction_setups.txt --output data/training_setups.txt
```

**Options:**
- `--prediction-file`: Path to prediction setups file
- `--output`: Path to output training setups file
- `--db-path`: Path to DuckDB database

### 2. Preserve Data

Save the original embeddings and features for the prediction setups before removing them from training.

```bash
python preserve_restore_embeddings.py preserve --prediction-setup-file data/prediction_setups.txt
```

**What it does:**
- Extracts embeddings for prediction setups from each domain's embedding table
- Extracts features for prediction setups from each domain's feature table
- Saves both to a timestamped pickle file for later restoration

### 3. Remove from Training

Remove the prediction setups from training tables to ensure proper separation.

```bash
python preserve_restore_embeddings.py remove --prediction-setup-file data/prediction_setups.txt
```

**What it does:**
- Removes embeddings for prediction setups from each domain's embedding table
- Removes features for prediction setups from each domain's feature table
- This ensures these setups won't influence training or similarity calculations

### 4. Reset Similarity Features (Optional but Recommended)

Reset only the similarity-based features while keeping the raw features intact:

```bash
python preserve_restore_embeddings.py reset-similarity --prediction-setup-file data/prediction_setups.txt
```

**What it does:**
- Resets only the similarity-based features (like positive_signal_strength, negative_risk_score, etc.)
- Keeps the raw extracted features intact
- This is more efficient than re-extracting all features

### 5. Create Prediction Embeddings and Features

Create new embeddings and extract features for the prediction setups (without labels).

```bash
python run_enhanced_ml_pipeline.py --mode prediction --setup-list data/prediction_setups.txt --domains all
```

**What it does:**
- Creates embeddings for the specified setups in prediction mode
- Stores them in separate `{domain}_embeddings_prediction` tables
- Extracts features for these setups using the prediction embeddings
- Creates ML feature tables for prediction:
  - `text_ml_features_prediction`: Contains text-based features from all domains
  - `financial_features_prediction`: Contains financial metrics

**Note:** This step handles both embedding creation and feature extraction in a single command.

### 6. Create Ensemble Predictions

Combine predictions from all domains to create ensemble predictions.

```bash
python ensemble_prediction.py --method weighted --setup-list data/prediction_setups.txt --output data/ensemble_predictions.csv
```

**Options:**
- `--method`: Ensemble method (`majority` or `weighted`)
- `--domains`: Domains to include
- `--weights`: Custom weights for each domain

**Note:** If this step fails with "No predictions found", ensure that Step 5 completed successfully and check that the similarity_predictions table contains entries for your prediction setups.

### 7. Extract ML Features

Extract comprehensive ML features for both training and prediction sets:

```bash
# Extract training features
python extract_all_ml_features_from_duckdb.py --mode training --setup-list data/training_setups.txt

# Extract prediction features
python extract_all_ml_features_from_duckdb.py --mode prediction --setup-list data/prediction_setups.txt
```

**What it does:**
- Extracts text features from news, user posts, and analyst recommendations
- Extracts financial features from fundamentals and financial ratios
- Creates comprehensive ML feature tables:
  - `text_ml_features_training` / `text_ml_features_prediction`
  - `financial_ml_features_training` / `financial_ml_features_prediction`
- Exports these tables to CSV files with timestamps
- Ensures consistent column order and names between training and prediction datasets

### 8. Add Labels and Balance Classes

Add labels to the feature tables and ensure balanced class distribution:

```bash
# Add labels to training features
python add_labels_to_features.py --input data/ml_features/text_ml_features_training_*.csv --output data/ml_features/text_ml_features_training_labeled.csv --mode training
python add_labels_to_features.py --input data/ml_features/financial_ml_features_training_*.csv --output data/ml_features/financial_ml_features_training_labeled.csv --mode training

# Add labels to prediction features (for evaluation only)
python add_labels_to_features.py --input data/ml_features/text_ml_features_prediction_*.csv --output data/ml_features/text_ml_features_prediction_labeled.csv --mode prediction
python add_labels_to_features.py --input data/ml_features/financial_ml_features_prediction_*.csv --output data/ml_features/financial_ml_features_prediction_labeled.csv --mode prediction
```

**What it does:**
- Retrieves `outperformance_10d` values from the database
- For training mode:
  - Calculates dynamic percentile thresholds (33.33% and 66.67%) for balanced classes
  - Assigns labels based on these thresholds: -1 (negative), 0 (neutral), 1 (positive)
- For prediction mode:
  - Uses fixed thresholds based on training data
  - Assigns labels for evaluation purposes only
- Ensures consistent label format (-1, 0, 1) across all datasets

### 9. Balance Datasets

Ensure text and financial ML datasets have the same setup_ids and sample sizes:

```bash
python balance_ml_datasets.py --text-train data/ml_features/text_ml_features_training_labeled.csv --financial-train data/ml_features/financial_ml_features_training_labeled.csv --text-predict data/ml_features/text_ml_features_prediction_labeled.csv --financial-predict data/ml_features/financial_ml_features_prediction_labeled.csv --output-dir data/ml_features/balanced
```

**What it does:**
- Identifies common setup_ids between text and financial datasets
- Uses all available setup_ids from the text dataset by default
- Adds dummy rows for missing setup_ids to ensure consistency
- Ensures label consistency across both datasets
- Maintains consistent column order and names between datasets
- Converts labels to a consistent format (-1, 0, 1) if needed

### 10. Train ML Models with Cross-Validation

Train various machine learning models using cross-validation:

```bash
python train_domain_models_cv.py --input data/ml_features/balanced/text_ml_features_training_balanced_*.csv --domain text --output-dir models/text
python train_domain_models_cv.py --input data/ml_features/balanced/financial_ml_features_training_balanced_*.csv --domain financial --output-dir models/financial
```

**Options:**
- `--input`: Path to training data CSV
- `--domain`: Domain name (text or financial)
- `--output-dir`: Directory to save trained models
- `--exclude-cols`: Columns to exclude from features (default: outperformance_10d)
- `--models`: Models to train (default: random_forest, xgboost, logistic_regression)
- `--cv-folds`: Number of cross-validation folds (default: 5)

**What it does:**
- Prepares data for training by separating features and labels
- Excludes `outperformance_10d` from features to prevent target leakage
- Performs stratified k-fold cross-validation
- Trains multiple ML models (Random Forest, XGBoost, Logistic Regression)
- Evaluates models on validation and test sets
- Saves trained models, feature importances, and evaluation results

### 11. Make Ensemble Predictions

Combine predictions from all models to create ensemble predictions:

```bash
python ensemble_domain_predictions.py --text-input data/ml_features/balanced/text_ml_features_prediction_balanced_*.csv --financial-input data/ml_features/balanced/financial_ml_features_prediction_balanced_*.csv --text-models-dir models/text --financial-models-dir models/financial --output-dir data/predictions
```

**What it does:**
- Loads trained models from each domain
- Makes predictions on the balanced prediction datasets
- Combines predictions using weighted voting
- Outputs ensemble predictions to a CSV file

### 12. Generate Results Table

Generate a comprehensive results table combining ML predictions, agent ensemble predictions, and actual labels:

```bash
python generate_results_table.py --input data/predictions/ensemble_predictions_*.csv --output data/results_table.csv
```

**What it does:**
- Loads ML predictions from the ensemble predictions file
- Retrieves actual labels from the database
- Fetches agent predictions from the `similarity_predictions` table
- Combines all information into a comprehensive results table with columns:
  - `setup_id`: Setup ID
  - `actual_label`: Actual label (-1, 0, 1)
  - `outperformance_10d`: Actual outperformance value
  - `predicted_label_ML`: ML model prediction
  - `confidence_score_ml`: ML model confidence score
  - `Agent_Ensemble_Prediction`: Agent ensemble prediction
  - `Agent_Predicted_Outperformance`: Agent predicted outperformance
  - `Agent_Confidence_Score`: Agent confidence score
  - `Domains_Count`: Number of domains with agent predictions

### 13. Visualize Results

Generate visualizations for the ensemble predictions:

```bash
python visualize_ensemble_results.py --input data/predictions/ensemble_predictions_*.csv --output-dir visualizations
```

**What it does:**
- Creates confusion matrices for ensemble and individual models
- Generates class distribution plots
- Calculates and visualizes performance metrics
- Creates model comparison charts
- Outputs comprehensive performance reports

### 14. Restore Data (When Needed)

When you're ready to return to training with the full dataset:

```bash
python preserve_restore_embeddings.py restore --preserved-file data/preserved_data_20230101.pkl
```

**What it does:**
- Restores the original embeddings back to the training tables
- Restores the original features back to the feature tables
- This avoids having to re-embed or re-extract features for these setups

## Column Consistency Between Training and Prediction

Maintaining consistent column order and names between training and prediction datasets is critical for ML model training and prediction. The pipeline ensures this consistency through several mechanisms:

1. **Feature Extraction**: The `extract_all_ml_features_from_duckdb.py` script uses the same SQL queries for both training and prediction modes, ensuring consistent feature extraction.

2. **Label Addition**: The `add_labels_to_features.py` script adds labels in a consistent format (-1, 0, 1) to both training and prediction datasets.

3. **Dataset Balancing**: The `balance_ml_datasets.py` script ensures that:
   - Both text and financial datasets have the same set of setup_ids
   - Labels are consistent between datasets
   - Column order and names are preserved

4. **Column Order Preservation**: When loading datasets for training and prediction, the pipeline ensures that the same columns are used in the same order.

5. **Feature Exclusion**: The `outperformance_10d` column is consistently excluded from features during training to prevent target leakage.

6. **Model Training**: The `train_domain_models_cv.py` script records the column order used during training and ensures the same order is used during prediction.

If you encounter any issues with column mismatches, check that:
- The feature extraction process completed successfully for both training and prediction
- The dataset balancing step was performed
- No manual modifications were made to the CSV files
- The same version of the pipeline was used for both training and prediction

## Feature Tables Created

The pipeline creates several types of feature tables:

1. **Domain-specific feature tables**:
   - `news_features`: Raw and similarity-based features from news
   - `fundamentals_features`: Raw and similarity-based features from fundamentals
   - `analyst_recommendations_features`: Raw and similarity-based features from analyst recommendations
   - `userposts_features`: Raw and similarity-based features from user posts

2. **ML feature tables**:
   - `text_ml_features_training`: Combined text features for training
   - `text_ml_features_prediction`: Combined text features for prediction
   - `financial_ml_features_training`: Financial metrics for training
   - `financial_ml_features_prediction`: Financial metrics for prediction

3. **Labeled ML feature tables**:
   - `text_ml_features_training_labeled.csv`: Text features with labels for training
   - `text_ml_features_prediction_labeled.csv`: Text features with labels for prediction
   - `financial_ml_features_training_labeled.csv`: Financial features with labels for training
   - `financial_ml_features_prediction_labeled.csv`: Financial features with labels for prediction

4. **Balanced ML feature tables**:
   - `text_ml_features_training_balanced_*.csv`: Balanced text features for training
   - `text_ml_features_prediction_balanced_*.csv`: Balanced text features for prediction
   - `financial_ml_features_training_balanced_*.csv`: Balanced financial features for training
   - `financial_ml_features_prediction_balanced_*.csv`: Balanced financial features for prediction

5. **Prediction tables**:
   - `similarity_predictions`: Contains predictions from each domain
   - `ensemble_predictions_*.csv`: Contains combined predictions across domains

6. **Results tables**:
   - `results_table.csv`: Comprehensive results table with ML and agent predictions

## ML Models

The pipeline supports training and evaluation of several machine learning models:

1. **Random Forest**: An ensemble learning method that builds multiple decision trees and merges their predictions
2. **XGBoost**: A gradient boosting framework that uses decision trees and is optimized for performance
3. **Logistic Regression**: A linear model for classification problems

Each model is trained with:
- Feature imputation (using median strategy)
- Feature scaling (using StandardScaler)
- Cross-validation (using StratifiedKFold)
- Hyperparameter tuning (configurable)

The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Label Format

The pipeline now consistently uses the following label format across all datasets:

- **-1**: Negative class (outperformance below the lower threshold)
- **0**: Neutral class (outperformance between thresholds)
- **1**: Positive class (outperformance above the upper threshold)

This format is used for:
- Labels in feature tables
- Model training and prediction
- Results table
- Performance evaluation

## Running the Complete Pipeline

To run the complete pipeline in one go:

```bash
python run_complete_ml_pipeline.py --mode all
```

**Options:**
- `--mode`: Pipeline mode (`training`, `prediction`, or `all`)
- `--db-path`: Path to DuckDB database
- `--output-dir`: Directory to save output files
- `--conda-env`: Conda environment to use

This will execute all steps from feature extraction to results generation.

## Troubleshooting

### HTTP 429 Errors

If you encounter HTTP 429 errors (Too Many Requests) when downloading models from Hugging Face:

1. Set up offline model caching as described in the preparation section
2. See the [Offline Model Caching Guide](OFFLINE_MODEL_GUIDE.md) for more details

### No Predictions Found

If the ensemble prediction step fails with "No predictions found":

1. Check that Step 5 completed successfully
2. Verify that the `similarity_predictions` table exists and contains entries:
   ```bash
   python -c "import duckdb; conn = duckdb.connect('data/sentiment_system.duckdb'); print(conn.execute('SELECT COUNT(*) FROM similarity_predictions').fetchone())"
   ```
3. If the table is empty, you may need to check the agent code to ensure predictions are being saved correctly

### Column Mismatch Error

If you encounter column mismatch errors during model prediction:

1. Check that the balanced datasets have the same columns in the same order
2. Verify that the feature extraction process completed successfully
3. Ensure that the same version of the pipeline was used for both training and prediction

### Missing Import Error

If you see "NameError: name 'os' is not defined" in agent files:

```bash
echo "import os" | cat - agents/analyst_recommendations/enhanced_analyst_recommendations_agent_duckdb.py > temp && mv temp agents/analyst_recommendations/enhanced_analyst_recommendations_agent_duckdb.py
```

### Other Issues

- **Missing embeddings**: Check that the embedding tables exist and contain data
- **Feature extraction errors**: Check the logs for specific error messages
- **Prediction errors**: Verify that the prediction setups have been properly processed

## Performance Considerations

- **Embedding creation** is typically the most computationally intensive step
- **Feature extraction** time depends on the number of similar cases to process
- **Ensemble prediction** is relatively fast even with large datasets
- The optimized workflow minimizes re-computation by preserving and restoring data

## Conclusion

This optimized workflow provides a balance between:
- Proper separation of training and prediction data
- Computational efficiency
- Accurate evaluation of prediction performance
- Comprehensive ML feature extraction
- Machine learning model training and evaluation
- Consistent label format and column order
- Balanced class distribution
- Comprehensive results reporting

By following these steps, you can ensure that your predictions are not influenced by data leakage while still maintaining efficient use of computational resources. 