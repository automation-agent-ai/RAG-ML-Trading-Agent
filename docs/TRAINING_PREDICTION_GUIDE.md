# Training/Prediction Data Separation Guide

This guide explains how to properly separate training and prediction data to avoid data leakage in the ML pipeline.

## Overview

The ML pipeline needs to maintain a clear separation between training and prediction data to avoid data leakage. This is achieved through:

1. **Separate embedding tables** for training and prediction data
2. **Different embedding processes** for training vs. prediction
3. **Clean separation of setup IDs** between training and prediction sets

## Data Flow

```
                       ┌─────────────────┐
                       │  Complete Data  │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Data Split    │
                       └────────┬────────┘
                       ┌────────┴────────┐
                       │                 │
                       ▼                 ▼
         ┌─────────────────────┐ ┌─────────────────────┐
         │   Training Data     │ │  Prediction Data    │
         │ (with labels)       │ │ (without labels)    │
         └─────────┬───────────┘ └───────────┬─────────┘
                   │                         │
                   ▼                         ▼
         ┌─────────────────────┐ ┌─────────────────────┐
         │ Training Embeddings │ │Prediction Embeddings│
         │ (include labels)    │ │ (exclude labels)    │
         └─────────┬───────────┘ └───────────┬─────────┘
                   │                         │
                   ▼                         ▼
         ┌─────────────────────┐ ┌─────────────────────┐
         │  Feature Extraction │ │ Feature Extraction  │
         │  (training mode)    │ │ (prediction mode)   │
         └─────────┬───────────┘ └───────────┬─────────┘
                   │                         │
                   ▼                         ▼
         ┌─────────────────────┐ ┌─────────────────────┐
         │   ML Features       │ │   ML Features       │
         │   (with labels)     │ │   (no labels)       │
         └─────────────────────┘ └─────────────────────┘
```

## New Tools

### 1. Data Splitting Tool

The `split_train_prediction_data.py` script:

- Identifies setups with complete data across all domains
- Randomly splits them into training and prediction sets
- Creates separate embedding tables for each
- Saves labels separately for evaluation

```bash
# Split data with 20% for prediction
python split_train_prediction_data.py --prediction-ratio 0.2
```

### 2. Enhanced ML Pipeline

The `run_enhanced_ml_pipeline.py` script:

- Uses separate embedding tables for training and prediction
- Processes features with appropriate mode
- Creates ML feature tables with proper separation

```bash
# Training mode
python run_enhanced_ml_pipeline.py --mode training

# Prediction mode
python run_enhanced_ml_pipeline.py --mode prediction --setup-ids SETUP1 SETUP2
```

### 3. Embedding Preservation Tool

The `preserve_restore_embeddings.py` script:

- Preserves embeddings before moving setups to prediction
- Restores embeddings back to training tables later
- Avoids unnecessary re-embedding

```bash
# Preserve embeddings for prediction setups
python preserve_restore_embeddings.py preserve --prediction-setup-file data/prediction_setups_20230101.txt

# Restore preserved embeddings back to training
python preserve_restore_embeddings.py restore --preserved-file data/preserved_embeddings_20230101.pkl
```

## Workflow Options

### Option 1: Standard Workflow (With Re-embedding)

1. **Split your data**:
   ```bash
   python split_train_prediction_data.py --prediction-ratio 0.2
   ```
   This creates:
   - `{domain}_embeddings_training` tables with labels
   - `{domain}_embeddings_prediction` tables without labels
   - Saved lists of training and prediction setup IDs

2. **Process training data**:
   ```bash
   python run_enhanced_ml_pipeline.py --mode training --setup-list data/training_setups_*.txt
   ```

3. **Process prediction data**:
   ```bash
   python run_enhanced_ml_pipeline.py --mode prediction --setup-list data/prediction_setups_*.txt
   ```

### Option 2: Optimized Workflow (Preserving Embeddings)

This workflow avoids re-embedding by preserving the original embeddings:

1. **Identify setups to use for prediction**:
   ```bash
   # Create a list of setups for prediction
   python create_prediction_list.py --count 100 --output data/prediction_setups.txt
   ```

2. **Preserve original embeddings**:
   ```bash
   # Save the original embeddings with labels
   python preserve_restore_embeddings.py preserve --prediction-setup-file data/prediction_setups.txt
   ```

3. **Remove prediction setups from training tables**:
   ```bash
   # Remove the setups from training tables
   python preserve_restore_embeddings.py remove --prediction-setup-file data/prediction_setups.txt
   ```

4. **Create prediction embeddings**:
   ```bash
   # Create embeddings without labels for prediction
   python run_enhanced_ml_pipeline.py --mode prediction --setup-list data/prediction_setups.txt --step embeddings
   ```

5. **Run prediction**:
   ```bash
   # Run feature extraction and prediction
   python run_enhanced_ml_pipeline.py --mode prediction --setup-list data/prediction_setups.txt --step features
   ```

6. **Evaluate predictions**:
   ```bash
   # Evaluate prediction performance
   python evaluate_predictions.py
   ```

7. **Restore original embeddings to training**:
   ```bash
   # Add the original embeddings back to training tables
   python preserve_restore_embeddings.py restore --preserved-file data/preserved_embeddings_*.pkl
   ```

This optimized workflow is particularly useful for historical data where you already have embeddings with labels.

### Domain-Specific Processing

You can process specific domains using the `--domains` flag:

```bash
# Process only news and fundamentals domains
python run_enhanced_ml_pipeline.py --mode training --domains news fundamentals
```

## Preventing Data Leakage

This approach prevents data leakage by:

1. **Physical separation**: Training and prediction data are in separate tables
2. **Label removal**: Prediction data has labels removed or zeroed out
3. **Mode-specific processing**: Different embedding and feature extraction processes for each mode

## Evaluation

To evaluate predictions:

1. Use the saved prediction labels (`data/prediction_labels_*.csv`)
2. Compare with generated predictions
3. Calculate metrics (accuracy, precision, recall, etc.)

## Best Practices

1. **Always split data first** before any embedding or feature extraction
2. **Never mix** training and prediction data
3. **Save random seeds** for reproducibility
4. **Document which setups** were used for training vs. prediction
5. **Preserve original embeddings** when possible to avoid re-embedding
6. **Use separate tables** for training and prediction embeddings

## Real-World vs. Historical Data

### For Historical Data (Backtesting)
- Use the optimized workflow to preserve and restore embeddings
- This avoids unnecessary re-embedding of data

### For Real-World New Data
- Create prediction embeddings without labels
- After labels become available, create training embeddings with labels
- Add these to your training set for future predictions 