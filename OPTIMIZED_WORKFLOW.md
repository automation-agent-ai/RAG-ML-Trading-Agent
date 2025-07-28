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
                                               ┌─────────────────┐
                                               │                 │
                                               │ Train ML        │
                                               │ Models          │
                                               │                 │
                                               └─────────────────┘
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

### 1. Select Setups for Prediction

Choose which setups to use for prediction (typically a random subset of setups with complete data).

```bash
python create_prediction_list.py --count 100 --output data/prediction_setups.txt
```

**Options:**
- `--count`: Number of setups to select
- `--db-path`: Path to DuckDB database
- `--output`: Output file for setup IDs
- `--random-seed`: Random seed for reproducibility

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

### 8. Merge Features with Ensemble Predictions

Merge the ML features with ensemble predictions to create a comprehensive dataset for ML:

```bash
# Merge prediction features
python merge_ensemble_with_features.py --ensemble data/ensemble_predictions.csv --text-features data/ml_features/text_ml_features_prediction_*.csv --financial-features data/ml_features/financial_ml_features_prediction_*.csv --output data/merged_ml_features_prediction.csv

# Merge training features (if ensemble predictions are available for training)
python merge_ensemble_with_features.py --ensemble data/ensemble_predictions.csv --text-features data/ml_features/text_ml_features_training_*.csv --financial-features data/ml_features/financial_ml_features_training_*.csv --output data/merged_ml_features_training.csv
```

**What it does:**
- Merges text features, financial features, and ensemble predictions into a single dataset
- Creates comprehensive ML feature tables for training and prediction
- Exports these tables to CSV files

### 9. Train ML Models

Train various machine learning models using the merged features:

```bash
python train_ml_models.py --training-data data/merged_ml_features_training.csv --test-data data/merged_ml_features_prediction.csv --output-dir models
```

**Options:**
- `--training-data`: Path to training data CSV
- `--test-data`: Path to test data CSV
- `--output-dir`: Directory to save trained models
- `--label-col`: Name of the label column
- `--exclude-cols`: Columns to exclude from features
- `--models`: Models to train (default: random_forest, gradient_boosting, logistic_regression)

**What it does:**
- Prepares data for training by separating features and labels
- Trains multiple ML models (Random Forest, Gradient Boosting, Logistic Regression)
- Evaluates models on validation and test sets
- Saves trained models, feature importances, and evaluation results

### 10. Evaluate Predictions

Evaluate prediction performance against the actual labels.

```bash
python evaluate_predictions.py --predictions data/ensemble_predictions.csv
```

**What it does:**
- Loads the preserved labels for the prediction setups
- Compares predictions against actual labels
- Calculates metrics like accuracy, precision, recall, F1 score
- Generates confusion matrices and other visualizations

### 11. Restore Data (When Needed)

When you're ready to return to training with the full dataset:

```bash
python preserve_restore_embeddings.py restore --preserved-file data/preserved_data_20230101.pkl
```

**What it does:**
- Restores the original embeddings back to the training tables
- Restores the original features back to the feature tables
- This avoids having to re-embed or re-extract features for these setups

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

3. **Prediction tables**:
   - `similarity_predictions`: Contains predictions from each domain
   - `ensemble_predictions`: Contains combined predictions across domains (saved to CSV)

4. **Merged ML feature tables**:
   - `merged_ml_features_training.csv`: Combined text, financial, and ensemble features for training
   - `merged_ml_features_prediction.csv`: Combined text, financial, and ensemble features for prediction

## ML Models

The pipeline supports training and evaluation of several machine learning models:

1. **Random Forest**: An ensemble learning method that builds multiple decision trees and merges their predictions
2. **Gradient Boosting**: A boosting algorithm that builds trees sequentially, with each tree correcting errors of previous ones
3. **Logistic Regression**: A linear model for classification problems

Each model is trained with:
- Feature imputation (using median strategy)
- Feature scaling (using StandardScaler)
- Hyperparameter tuning (configurable)

The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

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

By following these steps, you can ensure that your predictions are not influenced by data leakage while still maintaining efficient use of computational resources. 