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
10. Apply consistent financial data preprocessing across training and prediction

## Workflow Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  1. Select      │────▶│  2. Preserve    │────▶│  3. Remove from │
│  Setups for     │     │     Data        │     │     Training    │
│  Prediction     │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  16. Restore    │◀────│  15. Evaluate   │◀────│  4. Reset       │
│  Data           │     │  Predictions    │     │  Similarity     │
│  (when needed)  │     │                 │     │  Features       │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │                 │
                                               │  5. Create      │
                                               │  Prediction     │
                                               │  Embeddings     │
                                               └────────┬────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │                 │
                                               │  7. Extract ML  │
                                               │  Features       │
                                               │                 │
                                               └────────┬────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐     ┌─────────────────┐
                                               │                 │     │                 │
                                               │  8. Add Labels  │────▶│  9. Balance     │
                                               │  & Balance      │     │  Datasets       │
                                               │  Classes        │     │                 │
                                               └────────┬────────┘     └────────┬────────┘
                                                        │                       │
                                                        ▼                       │
                                               ┌─────────────────┐              │
                                               │                 │              │
                                               │  10. Make Agent │              │
                                               │  Predictions    │              │
                                               │                 │              │
                                               └────────┬────────┘              │
                                                        │                       │
                                                        ▼                       ▼
                                               ┌─────────────────┐     ┌─────────────────┐
                                               │                 │     │                 │
                                               │  11. Train ML   │────▶│  12. Generate   │
                                               │  Models         │     │  Results Table  │
                                               │                 │     │                 │
                                               └─────────────────┘     └─────────────────┘
```

## Preparation: Offline Model Caching (Recommended)

To avoid HTTP 429 errors and rate limiting issues with Hugging Face, we recommend setting up offline model caching before running the pipeline:

```bash
# Step 1: Download and cache the model
conda activate sts
python download_models.py

# Step 2: Patch the pipeline to use cached models
python use_cached_model.py
```

See the [Offline Model Caching Guide](OFFLINE_MODEL_GUIDE.md) for more details.

## Detailed Steps

### 1. Select Setups for Training and Prediction

First, choose which setups to use for prediction (typically a random subset of setups with complete data).

```bash
conda activate sts
python create_prediction_list.py --count 100 --output data/prediction_setups.txt
```

**Options:**
- `--count`: Number of setups to select
- `--db-path`: Path to DuckDB database
- `--output`: Output file for setup IDs
- `--random-seed`: Random seed for reproducibility

Then, create a list of training setups by excluding the prediction setups from all available setups:

```bash
conda activate sts
python create_training_list.py --prediction-file data/prediction_setups.txt --output data/training_setups.txt
```

**Options:**
- `--prediction-file`: Path to prediction setups file
- `--output`: Path to output training setups file
- `--db-path`: Path to DuckDB database

### 2. Preserve Data

Save the original embeddings and features for the prediction setups before removing them from training.

```bash
conda activate sts
python preserve_restore_embeddings.py preserve --prediction-setup-file data/prediction_setups.txt
```

**What it does:**
- Extracts embeddings for prediction setups from each domain's embedding table
- Extracts features for prediction setups from each domain's feature table
- Saves both to a timestamped pickle file for later restoration

### 3. Remove from Training

Remove the prediction setups from training tables to ensure proper separation.

```bash
conda activate sts
python preserve_restore_embeddings.py remove --prediction-setup-file data/prediction_setups.txt
```

**What it does:**
- Removes embeddings for prediction setups from each domain's embedding table
- Removes features for prediction setups from each domain's feature table
- This ensures these setups won't influence training or similarity calculations

### 4. Reset Similarity Features (Optional but Recommended)

Reset only the similarity-based features while keeping the raw features intact:

```bash
conda activate sts
python preserve_restore_embeddings.py reset-similarity --prediction-setup-file data/prediction_setups.txt
```

**What it does:**
- Resets only the similarity-based features (like positive_signal_strength, negative_risk_score, etc.)
- Keeps the raw extracted features intact
- This is more efficient than re-extracting all features

### 5. Create Prediction Embeddings

Create new embeddings for the prediction setups (without making predictions yet).

```bash
conda activate sts
python run_enhanced_ml_pipeline.py --mode prediction --setup-list data/prediction_setups.txt --domains all --embeddings-only
```

**What it does:**
- Creates embeddings for the specified setups in prediction mode
- Stores them in separate `{domain}_embeddings_prediction` tables
- Does NOT make predictions yet (this will happen after thresholds are determined)

**Note:** This step only creates embeddings, not predictions. Predictions will be made later using consistent thresholds.

### 7. Extract ML Features

Extract comprehensive ML features for both training and prediction sets:

```bash
# Extract training features
conda activate sts
python extract_all_ml_features_from_duckdb.py --mode training --setup-list data/training_setups.txt

# Extract prediction features
python extract_all_ml_features_from_duckdb.py --mode prediction --setup-list data/prediction_setups.txt

# Extract financial features specifically (with enhanced preprocessing)
python extract_financial_features_from_duckdb.py --mode training --setup-list data/training_setups.txt
python extract_financial_features_from_duckdb.py --mode prediction --setup-list data/prediction_setups.txt
```

**What it does:**
- Extracts text features from news, user posts, and analyst recommendations
- Extracts financial features from fundamentals and financial ratios using the enhanced financial preprocessor
- Creates comprehensive ML feature tables:
  - `text_ml_features_training` / `text_ml_features_prediction`
  - `financial_ml_features_training` / `financial_ml_features_prediction`
- Exports these tables to CSV files with timestamps
- Ensures consistent column order and names between training and prediction datasets

**Enhanced Financial Preprocessing:**
- Calculates comprehensive financial ratios (P&L/revenue, balance sheet/total assets)
- Computes 1-3 year growth metrics and trend indicators
- Performs data quality validation with detailed reporting
- Handles ticker format differences properly (e.g., adding `.L` suffix for LSE tickers)
- Saves preprocessing parameters (imputation and scaling) during training
- Applies consistent preprocessing parameters during prediction
- Ensures feature alignment between training and prediction sets

### 8. Add Labels and Balance Classes

Add labels to the feature tables and ensure balanced class distribution:

```bash
# Add labels to training features
conda activate sts
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
conda activate sts
python balance_ml_datasets.py --text-train data/ml_features/text_ml_features_training_labeled.csv --financial-train data/ml_features/financial_ml_features_training_labeled.csv --text-predict data/ml_features/text_ml_features_prediction_labeled.csv --financial-predict data/ml_features/financial_ml_features_prediction_labeled.csv --output-dir data/ml_features/balanced
```

**What it does:**
- Identifies common setup_ids between text and financial datasets
- Uses all available setup_ids from the text dataset by default
- Adds dummy rows for missing setup_ids to ensure consistency
- Ensures label consistency across both datasets
- Maintains consistent column order and names between datasets
- Converts labels to a consistent format (-1, 0, 1) if needed

### 10. Make Agent Predictions with Consistent Thresholds

Make agent predictions using the same thresholds determined during the label balancing step:

```bash
conda activate sts
python make_agent_predictions.py --setup-list data/prediction_setups.txt
```

**What it does:**
- Loads the thresholds saved during the label balancing step
- Makes predictions for each domain using these consistent thresholds
- Combines predictions into an ensemble prediction
- Saves all predictions to the similarity_predictions table
- Ensures that both ML models and domain agents use the same classification thresholds

**Note:** This step is placed after label balancing to ensure that agent predictions use the same thresholds as ML models.

### 11. Train ML Models with Cross-Validation

Train various machine learning models using cross-validation:

```bash
conda activate sts
python train_domain_models_cv.py --input data/ml_features/balanced/text_ml_features_training_balanced_*.csv --domain text --output-dir models/text
python train_domain_models_cv.py --input data/ml_features/balanced/financial_ml_features_training_balanced_*.csv --domain financial --output-dir models/financial

# To disable saved preprocessing parameters (optional)
python train_domain_models_cv.py --input data/ml_features/balanced/financial_ml_features_training_balanced_*.csv --domain financial --output-dir models/financial --disable-saved-preprocessing
```

**Options:**
- `--input`: Path to training data CSV
- `--domain`: Domain name (text or financial)
- `--output-dir`: Directory to save trained models
- `--exclude-cols`: Columns to exclude from features (default: outperformance_10d)
- `--models`: Models to train (default: random_forest, xgboost, logistic_regression)
- `--cv-folds`: Number of cross-validation folds (default: 5)
- `--disable-saved-preprocessing`: Disable loading saved preprocessing parameters for financial domain

**What it does:**
- Prepares data for training by separating features and labels
- Excludes `outperformance_10d` from features to prevent target leakage
- For financial domain: loads and applies saved preprocessing parameters from `financial_preprocessor.py` (unless disabled)
- Performs stratified k-fold cross-validation
- Trains multiple ML models (Random Forest, XGBoost, Logistic Regression)
- Evaluates models on validation and test sets
- Saves trained models, feature importances, and evaluation results

### 12. Make ML Ensemble Predictions

Combine predictions from all ML models to create ensemble predictions:

```bash
conda activate sts
python ensemble_domain_predictions.py --text-input data/ml_features/balanced/text_ml_features_prediction_balanced_*.csv --financial-input data/ml_features/balanced/financial_ml_features_prediction_balanced_*.csv --text-models-dir models/text --financial-models-dir models/financial --output-dir data/predictions
```

**What it does:**
- Loads trained models from each domain
- Makes predictions on the balanced prediction datasets
- Combines predictions using weighted voting
- Outputs ensemble predictions to a CSV file

### 13. Generate Results Table

Generate a comprehensive results table combining ML predictions, agent ensemble predictions, and actual labels:

```bash
conda activate sts
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

### 14. Visualize Results

Generate visualizations for the ensemble predictions:

```bash
conda activate sts
python visualize_ensemble_results.py --input data/predictions/ensemble_predictions_*.csv --output-dir visualizations
```

**What it does:**
- Creates confusion matrices for ensemble and individual models
- Generates class distribution plots
- Calculates and visualizes performance metrics
- Creates model comparison charts

### 15. Preserve and Restore Embeddings (Optional)

If you want to preserve prediction embeddings and later restore them to the training set (after labels become available):

```bash
# First, preserve prediction embeddings
conda activate sts
python preserve_restore_embeddings.py --action preserve --domains all --setup-list data/prediction_setups.txt

# Later, once labels are available, restore embeddings to training set
conda activate sts
python preserve_restore_embeddings.py --action restore --domains all --setup-list data/prediction_setups.txt
```

**What it does:**
- **Preserve action:**
  - Extracts embeddings for prediction setups from domain-specific embedding tables
  - Saves them to parquet files in the backup directory with timestamps
  - This prevents data loss and allows future reuse

- **Restore action:**
  - Checks if labels are available for the preserved setup IDs
  - Retrieves embeddings for setups with available labels
  - Merges embeddings with labels
  - Adds them back to the training embedding tables
  - Updates LanceDB tables if applicable
  - This enables continuous learning by incorporating new labeled data into the training set

### 16. Restore Data (When Needed)

When you're ready to return to training with the full dataset:

```bash
conda activate sts
python preserve_restore_embeddings.py restore --preserved-file data/preserved_data_20230101.pkl
```

**What it does:**
- Restores the original embeddings back to the training tables
- Restores the original features back to the feature tables
- This avoids having to re-embed or re-extract features for these setups

## Column Consistency Between Training and Prediction

Maintaining consistent column order and names between training and prediction datasets is critical for ML model training and prediction. The pipeline ensures this consistency through several mechanisms:

1. **Feature Extraction**: The `extract_all_ml_features_from_duckdb.py` script uses the same SQL queries for both training and prediction modes, ensuring consistent feature extraction.

2. **Enhanced Financial Preprocessing**: The `financial_preprocessor.py` module:
   - Saves preprocessing parameters (imputer, scaler, feature columns) during training
   - Loads and applies these parameters during prediction
   - Handles feature alignment by adding missing columns and removing extra ones
   - Ensures identical preprocessing steps for both training and prediction data

3. **Label Addition**: The `add_labels_to_features.py` script adds labels in a consistent format (-1, 0, 1) to both training and prediction datasets.

4. **Dataset Balancing**: The `balance_ml_datasets.py` script ensures that:
   - Both text and financial datasets have the same set of setup_ids
   - Labels are consistent between datasets
   - Column order and names are preserved

5. **Column Order Preservation**: When loading datasets for training and prediction, the pipeline ensures that the same columns are used in the same order.

6. **Feature Exclusion**: The `outperformance_10d` column is consistently excluded from features during training to prevent target leakage.

7. **Model Training**: The `train_domain_models_cv.py` script records the column order used during training and ensures the same order is used during prediction.

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

For the financial domain, the enhanced preprocessing pipeline:
1. Extracts comprehensive financial data from DuckDB
2. Calculates advanced financial ratios and metrics
3. Applies consistent preprocessing (imputation and scaling)
4. Saves preprocessing parameters during training
5. Loads and applies these parameters during prediction
6. Handles feature alignment between training and prediction datasets

The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Label Format and Thresholds

The pipeline uses a consistent label format of -1 (negative), 0 (neutral), and 1 (positive) across all steps. This format is used for:

1. ML model training and prediction
2. Agent ensemble predictions
3. Results table

### Threshold Management

The thresholds for determining these labels are:
- Negative (-1): outperformance_10d <= [neg_threshold]
- Neutral (0): [neg_threshold] < outperformance_10d < [pos_threshold]
- Positive (1): outperformance_10d >= [pos_threshold]

Where:
- [neg_threshold] is calculated as the 33.33 percentile of outperformance_10d in the training data
- [pos_threshold] is calculated as the 66.67 percentile of outperformance_10d in the training data

These thresholds are:
1. Determined dynamically during the "Add Labels" step to ensure balanced classes
2. Saved to a JSON file (data/label_thresholds.json) by the threshold_manager.py script
3. Used consistently across both ML models and domain agents

### Threshold Consistency

To ensure consistent thresholds across the entire system:
1. The `add_labels_to_features.py` script calculates and saves thresholds during training
2. The `make_agent_predictions.py` script loads these thresholds before making predictions
3. All predictions (ML and agent) use the same thresholds for classification

This ensures that the class boundaries are identical across all prediction methods, making the results directly comparable.

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
conda activate sts
python run_complete_ml_pipeline.py --mode all
```

**Options:**
- `--mode`: Pipeline mode (`training`, `prediction`, or `all`)
- `--db-path`: Path to DuckDB database
- `--output-dir`: Directory to save output files
- `--conda-env`: Conda environment to use
- `--disable-financial-preprocessing`: Disable enhanced financial preprocessing

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
   conda activate sts
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
conda activate sts
echo "import os" | cat - agents/analyst_recommendations/enhanced_analyst_recommendations_agent_duckdb.py > temp && mv temp agents/analyst_recommendations/enhanced_analyst_recommendations_agent_duckdb.py
```

### Other Issues

- **Missing embeddings**: Check that the embedding tables exist and contain data
- **Feature extraction errors**: Check the logs for specific error messages
- **Prediction errors**: Verify that the prediction setups have been properly processed

### Feature Mismatch in Financial Preprocessing

If you encounter warnings about feature mismatches during financial preprocessing:

1. Check the logs for specific warnings about missing or extra features
2. Verify that the training and prediction datasets contain similar financial metrics
3. The financial preprocessor will attempt to handle mismatches by:
   - Adding missing features with NaN values
   - Removing extra features not present during training
   - Reordering columns to match the training order

If issues persist, you can disable the enhanced preprocessing:
```bash
conda activate sts
python train_domain_models.py --input data/ml_features/balanced/financial_ml_features_training_balanced_*.csv --domain financial --output-dir models/financial --disable-saved-preprocessing
```

## Performance Considerations

- **Embedding creation** is typically the most computationally intensive step
- **Feature extraction** time depends on the number of similar cases to process
- **Financial preprocessing** is optimized but may take longer with comprehensive metrics
- **Ensemble prediction** is relatively fast even with large datasets
- The optimized workflow minimizes re-computation by preserving and restoring data

## Financial Preprocessor Module

The pipeline includes a dedicated `financial_preprocessor.py` module that enhances financial data preprocessing:

### Key Features

1. **Comprehensive Financial Metrics**:
   - Profitability ratios (ROA, ROE, Gross Margin, Net Margin)
   - Liquidity ratios (Current Ratio, Quick Ratio)
   - Leverage ratios (Debt to Equity, Interest Coverage)
   - Asset efficiency ratios (Asset Turnover, Inventory Turnover)
   - Cash flow metrics (FCF, FCF Yield, Cash Conversion)
   - Per-share metrics (EPS, BVPS, CFPS)
   - Valuation ratios (P/E, P/B, P/S, EV/EBITDA)

2. **Multi-Year Growth Metrics**:
   - 1-year growth rates for key metrics
   - 2-year growth rates for key metrics
   - 3-year growth rates for key metrics
   - Average growth rates (1-3 years)
   - Trend indicators (acceleration/deceleration)

3. **Consistent Preprocessing**:
   - Fits imputer and scaler on training data
   - Saves preprocessing parameters to disk
   - Loads and applies parameters during prediction
   - Handles feature alignment between datasets

4. **Data Quality Validation**:
   - Validates ticker format (adds `.L` suffix for LSE tickers)
   - Calculates data quality scores based on completeness
   - Performs basic sanity checks on financial ratios
   - Reports detailed quality metrics during extraction

### Usage

The financial preprocessor is integrated into the pipeline and used automatically by:
1. `extract_financial_features_from_duckdb.py` - For feature extraction
2. `train_domain_models.py` - For consistent preprocessing during training

To use it directly (advanced usage):

```python
from financial_preprocessor import FinancialPreprocessor

# Initialize preprocessor
preprocessor = FinancialPreprocessor()

# Extract and preprocess training data
training_features = preprocessor.extract_and_preprocess(
    setup_ids=training_setup_ids,
    mode="training"
)

# Extract and preprocess prediction data (using saved parameters)
prediction_features = preprocessor.extract_and_preprocess(
    setup_ids=prediction_setup_ids,
    mode="prediction"
)
```

## Conclusion

This optimized workflow provides a balance between:
- Proper separation of training and prediction data
- Computational efficiency
- Accurate evaluation of prediction performance
- Comprehensive ML feature extraction
- Enhanced financial data preprocessing
- Machine learning model training and evaluation
- Consistent label format and column order
- Balanced class distribution
- Comprehensive results reporting

By following these steps, you can ensure that your predictions are not influenced by data leakage while still maintaining efficient use of computational resources and applying consistent preprocessing across all datasets.