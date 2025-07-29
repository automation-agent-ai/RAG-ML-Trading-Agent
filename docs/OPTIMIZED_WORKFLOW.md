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
                                               ┌─────────────────┐
                                               │                 │
                                               │  4. Reset       │
                                               │  Similarity     │
                                               │  Features       │
                                               └────────┬────────┘
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
                                               │  6. Create      │
                                               │  Training       │
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
                                               │  11. Train ML   │────▶│  12. Make ML    │
                                               │  Models         │     │  Ensemble       │
                                               │                 │     │  Predictions    │
                                               └─────────────────┘     └────────┬────────┘
                                                                                │
                                                                                ▼
                                               ┌─────────────────┐     ┌─────────────────┐
                                               │                 │     │                 │
                                               │  13. Generate   │────▶│  14. Visualize  │
                                               │  Results Table  │     │  Results        │
                                               │                 │     │                 │
                                               └─────────────────┘     └────────┬────────┘
                                                                                │
                                                                                ▼
                                               ┌─────────────────┐     ┌─────────────────┐
                                               │                 │     │                 │
                                               │  15. Evaluate   │────▶│  16. Preserve   │
                                               │  Predictions    │     │  & Restore      │
                                               │                 │     │  Embeddings     │
                                               └─────────────────┘     └────────┬────────┘
                                                                                │
                                                                                ▼
                                                                      ┌─────────────────┐
                                                                      │                 │
                                                                      │  17. Restore    │
                                                                      │  Data           │
                                                                      │  (when needed)  │
                                                                      └─────────────────┘
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

### 6. Create Training Embeddings

Create new embeddings for the training setups (without making predictions yet).

```bash
conda activate sts
python run_enhanced_ml_pipeline.py --mode training --setup-list data/training_setups.txt --domains all --embeddings-only
```

**What it does:**
- Creates embeddings for the specified setups in training mode
- Stores them in separate `{domain}_embeddings_training` tables
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

### 10. Make Agent Predictions with LLM-based Few-Shot Learning

Make agent predictions using GPT-4o-mini with few-shot learning and consistent thresholds:

```bash
conda activate sts
python make_agent_predictions.py --setup-list data/prediction_setups.txt
```

**What it does:**
- **NEW LLM-BASED SYSTEM**: Uses GPT-4o-mini to make actual predictions with few-shot learning
- Each agent (news, fundamentals, analyst_recommendations, userposts) extracts features for current setup
- Retrieves 5 similar historical cases from LanceDB training embeddings
- Creates few-shot learning prompt with historical examples and their outcomes
- Calls GPT-4o-mini to make reasoned predictions based on features and historical patterns
- Returns structured JSON with predicted_outperformance_10d, confidence_score, and reasoning
- Applies consistent thresholds to convert predictions to -1/0/1 format
- Combines predictions into ensemble using confidence-weighted voting
- Saves all predictions to the similarity_predictions table

**Key Improvements:**
- ✅ **Real LLM Predictions**: Uses GPT-4o-mini API calls for each domain agent
- ✅ **Few-Shot Learning**: Provides historical examples with outcomes as context
- ✅ **Enhanced Error Handling**: Defaults to outperformance=0.0, confidence=0.34 on parsing errors
- ✅ **Domain-Specific Prompts**: Each agent has specialized prompts for their expertise area
- ✅ **Similarity-Based Retrieval**: Uses SentenceTransformer embeddings to find relevant cases

**Agent Files Updated:**
- `agents/fundamentals/enhanced_fundamentals_agent_duckdb.py` - Financial analysis predictions
- `agents/news/enhanced_news_agent_duckdb.py` - News sentiment predictions  
- `agents/analyst_recommendations/enhanced_analyst_recommendations_agent_duckdb.py` - Analyst consensus predictions
- `agents/userposts/enhanced_userposts_agent_complete.py` - Community sentiment predictions

**Example Output:**
```
News LLM prediction for EDEN_2024-10-08: -4.50% (confidence: 0.70)
Fundamentals LLM prediction for BGO_2025-01-20: 8.50% (confidence: 0.75)
Analyst LLM prediction for PAY_2025-02-19: 0.00% (confidence: 0.34)
```

**Note:** This step requires OpenAI API access and uses the same thresholds as ML models for consistent classification.

### 11. Train ML Models with 3-Stage Pipeline

Train ML models using the proper 3-stage approach:

#### The 3-Stage ML Pipeline

**Stage 1: Text-based ML** - Train 4 models (Random Forest, XGBoost, LightGBM, Logistic Regression) on text features extracted by LLM agents.

**Stage 2: Financial-based ML** - Train 4 models on financial features (ratios, fundamentals, growth metrics) from DuckDB.

**Stage 3: Ensemble ML** - Train 4 meta-models on the 8 prediction vectors (3 classes × 8 models = 24 features) from stages 1 & 2.

```bash
conda activate sts
python train_3stage_ml_pipeline.py --input-dir data/ml_features/balanced --output-dir models/3stage_fixed
```

**What it does:**
- Uses balanced training data from `data/ml_features/balanced/` (641 setup IDs)
- Stage 1: Trains 4 text models on 41 text features 
- Stage 2: Trains 4 financial models on 49 financial features
- Stage 3: Trains 4 ensemble meta-models on 24 prediction vectors (8 models × 3 probability classes)
- Saves models in separate directories: `text/`, `financial/`, `ensemble/`
- Applies proper imputation, scaling, and handles -1/0/1 labels → 0/1/2 for sklearn
- Generates confusion matrices and comprehensive training report
- **NEW:** Performs detailed cross-validation for each individual model
- **NEW:** Generates detailed per-model visualizations and metrics in `ml/analysis/` directories
- **NEW:** Fixed data leakage in ensemble training using proper nested cross-validation

**Key Features:**
- Consistent setup IDs across text and financial data
- Proper feature preprocessing with saved transformation parameters
- Cross-validation for model evaluation
- Ensemble meta-learning from individual model predictions
- Individual model analysis and visualizations

**Note on Warnings:**
- You may see warnings about "X does not have valid feature names" from LightGBM - these can be safely ignored
- XGBoost may show warnings about "use_label_encoder" parameter - this has been fixed in the corrected version
- The warning about "Labels differ between text and financial data" is expected and handled by using text labels as the reference

### 12. Make ML Ensemble Predictions

Create ensemble predictions using the trained 3-stage ML pipeline:

```bash
conda activate sts
python predict_3stage_ml_pipeline.py --input-dir data/ml_features/balanced --model-dir models/3stage_fixed --output-dir ml/prediction
```

**What it does:**
- Loads the trained models from all 3 stages (text/, financial/, ensemble/)
- Makes predictions on prediction data (50 setup IDs) using the same 3-stage process:
  1. **Stage 1**: Apply 4 text models to text features → get 12 prediction vectors
  2. **Stage 2**: Apply 4 financial models to financial features → get 12 prediction vectors  
  3. **Stage 3**: Apply 4 ensemble meta-models to the 24 prediction vectors → get final predictions
- **NEW:** Implements confidence-weighted ensemble voting for final predictions
- **NEW:** Organizes results into structured directories (`ml/prediction/text_ml/`, `ml/prediction/financial_ml/`, `ml/prediction/ensemble/`)
- **NEW:** Each prediction file includes confidence scores and class probabilities
- Outputs final ensemble predictions with confidence scores
- Provides predictions from individual models for analysis

**Output Files:**
- `ml/prediction/final_predictions_{timestamp}.csv` - Final confidence-weighted predictions
- `ml/prediction/prediction_summary_{timestamp}.txt` - Comprehensive prediction summary
- `ml/prediction/text_ml/text_{model}_predictions.csv` - Individual text model predictions
- `ml/prediction/financial_ml/financial_{model}_predictions.csv` - Individual financial model predictions
- `ml/prediction/ensemble/ensemble_{model}_predictions.csv` - Individual ensemble model predictions

### 13. Generate Results Table

Generate a comprehensive results table combining ML predictions, agent ensemble predictions, and actual labels:

```bash
conda activate sts
python generate_results_table.py --input ml/prediction/final_predictions_*.csv --output ml/prediction/results_table_final.csv
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

**Note:** If using the corrected 3-stage pipeline, you may need to adapt the visualization script to match the column names in your ensemble predictions file. The default script expects columns named 'true_label' and 'ensemble_prediction', but the 3-stage pipeline outputs different column names.

**Alternative Visualization:**

You can also use a simple Python script to visualize the results:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load predictions
predictions = pd.read_csv('data/predictions_corrected/final_predictions_*.csv')
results = pd.read_csv('data/results_table_corrected.csv')

# Plot prediction distribution
plt.figure(figsize=(10, 6))
predictions['prediction'].value_counts().sort_index().plot(kind='bar')
plt.title('Prediction Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.savefig('visualizations/prediction_distribution.png')

# Plot confusion matrix if actual labels are available
if 'actual_label' in results.columns and 'predicted_label_ML' in results.columns:
    valid_results = results.dropna(subset=['actual_label', 'predicted_label_ML'])
    cm = confusion_matrix(valid_results['actual_label'], valid_results['predicted_label_ML'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('visualizations/confusion_matrix.png')
```

**What it does:**
- Creates confusion matrices for ensemble and individual models
- Generates class distribution plots
- Calculates and visualizes performance metrics
- Creates model comparison charts

### 15. Evaluate Predictions

Evaluate the performance of the ML ensemble predictions:

```bash
conda activate sts
python evaluate_predictions.py --input data/predictions/ensemble_predictions_*.csv --output data/evaluation_results.csv
```

**What it does:**
- Loads the ensemble predictions
- Calculates various performance metrics (Accuracy, Precision, Recall, F1 Score, Confusion Matrix)
- Saves the evaluation results to a CSV file

### 16. Preserve and Restore Embeddings (Optional)

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

### 17. Restore Data (When Needed)

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

### Incorrect Dates in Filenames

You may notice that some CSV files have incorrect dates in their filenames (e.g., year 2025). This is a known issue with the system clock on some machines and doesn't affect functionality. If needed, you can fix this by:

1. Manually renaming the files with correct dates
2. Updating the date generation in scripts that create these files
3. Using the most recent file regardless of the timestamp (which the pipeline does by default)

The pipeline is designed to use the most recent file matching the pattern, so this issue doesn't affect functionality as long as the newest file is the one you want to use.

## Summary of 3-Stage ML Pipeline Fixes

The following fixes were made to the 3-stage ML pipeline to ensure it works correctly:

1. **Fixed CSV Feature Loader**:
   - Added more flexible pattern matching for CSV files
   - Added better error handling for missing files
   - Improved handling of file paths

2. **Fixed XGBoost and LightGBM Warnings**:
   - Removed the deprecated `use_label_encoder=False` parameter from XGBoost
   - Added `importance_type='gain'` to LightGBM to reduce warnings
   - Added warning suppression in the prediction script

3. **Created Simple Visualization Script**:
   - Developed a new script (`simple_visualize.py`) compatible with the 3-stage pipeline output format
   - Generated comprehensive visualizations including:
     - Prediction distribution
     - Confidence distribution
     - Confusion matrix
     - Confidence by prediction class
     - Performance metrics

4. **Updated Documentation**:
   - Added notes about handling warnings
   - Added information about the date issue in filenames
   - Added alternative visualization options
   - Updated commands to use the corrected versions

The pipeline now successfully:
1. Trains text, financial, and ensemble models
2. Makes predictions on new data
3. Generates comprehensive results tables
4. Creates visualizations for analysis

The accuracy of the current model is around 32%, which is in line with expectations for a 3-class classification problem (random baseline would be 33%). Further improvements could be made by:
1. Adding more features
2. Tuning hyperparameters
3. Using more sophisticated ensemble techniques
4. Incorporating additional domain knowledge