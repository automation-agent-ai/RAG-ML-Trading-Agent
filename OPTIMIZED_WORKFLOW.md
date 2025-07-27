# Optimized Training/Prediction Workflow

This document explains the optimized workflow for separating training and prediction data without unnecessary re-embedding.

## Overview

The optimized workflow preserves original embeddings with labels, allowing you to:

1. Use some setups for prediction (without labels)
2. Later restore those same setups back to training (with labels)
3. Avoid the computational cost of re-embedding

This is particularly useful for historical data analysis and backtesting.

## Workflow Diagram

```
┌─────────────────────┐
│  Original Embeddings│
│  (with labels)      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Preserve Embeddings│
│  for Prediction     │
└─────────┬───────────┘
          │
          │    ┌───────────────────┐
          ├───▶│  Saved Embeddings │
          │    │  (with labels)    │
          │    └─────────┬─────────┘
          │              │
          ▼              │
┌─────────────────────┐  │
│  Remove from        │  │
│  Training Tables    │  │
└─────────┬───────────┘  │
          │              │
          ▼              │
┌─────────────────────┐  │
│  Create Prediction  │  │
│  Embeddings         │  │
│  (without labels)   │  │
└─────────┬───────────┘  │
          │              │
          ▼              │
┌─────────────────────┐  │
│  Run Prediction     │  │
│  & Evaluation       │  │
└─────────┬───────────┘  │
          │              │
          ▼              │
┌─────────────────────┐  │
│  Restore Original   │◀─┘
│  Embeddings to      │
│  Training Tables    │
└─────────────────────┘
```

## Step-by-Step Guide

### 1. Select Prediction Setups

First, identify which setups to use for prediction:

```bash
# Create a list of 100 random setups for prediction
python create_prediction_list.py --count 100 --output data/prediction_setups.txt
```

### 2. Preserve Original Embeddings

Before removing these setups from training, preserve their embeddings:

```bash
# Save the original embeddings with labels
python preserve_restore_embeddings.py preserve --prediction-setup-file data/prediction_setups.txt
```

This saves the embeddings to a pickle file (e.g., `data/preserved_embeddings_20230101_120000.pkl`).

### 3. Remove from Training Tables

Remove these setups from the training tables:

```bash
# Remove the setups from training tables
python preserve_restore_embeddings.py remove --prediction-setup-file data/prediction_setups.txt
```

### 4. Create Prediction Embeddings

Create embeddings for prediction (without labels):

```bash
# Create embeddings without labels for prediction
python run_enhanced_ml_pipeline.py --mode prediction --setup-list data/prediction_setups.txt --step embeddings
```

### 5. Run Prediction

Extract features and make predictions:

```bash
# Run feature extraction and prediction
python run_enhanced_ml_pipeline.py --mode prediction --setup-list data/prediction_setups.txt --step features
```

### 6. Evaluate Predictions

Evaluate the prediction performance:

```bash
# Evaluate prediction performance
python evaluate_predictions.py
```

### 7. Restore Original Embeddings

After evaluation, restore the original embeddings back to training:

```bash
# Add the original embeddings back to training tables
python preserve_restore_embeddings.py restore --preserved-file data/preserved_embeddings_*.pkl
```

## Benefits

This optimized workflow provides several advantages:

1. **Computational Efficiency**: Avoids re-embedding, which is computationally expensive
2. **Data Integrity**: Preserves the original embeddings with their labels
3. **Flexibility**: Allows you to easily move setups between training and prediction
4. **Reproducibility**: Maintains consistent embeddings across experiments

## Use Cases

### Historical Backtesting

When working with historical data where all labels are known:

1. Select different subsets for prediction in each experiment
2. Preserve/restore embeddings to avoid re-embedding
3. Compare performance across different prediction sets

### Incremental Learning

As new data becomes available:

1. Initially create prediction embeddings without labels
2. After labels become available, create training embeddings
3. Add these to your training set for future predictions

## Script Reference

### preserve_restore_embeddings.py

```bash
# Preserve embeddings
python preserve_restore_embeddings.py preserve --prediction-setup-file FILE

# Restore embeddings
python preserve_restore_embeddings.py restore --preserved-file FILE

# Remove setups from training
python preserve_restore_embeddings.py remove --prediction-setup-file FILE
```

### create_prediction_list.py

```bash
# Create a list of prediction setups
python create_prediction_list.py --count COUNT --output FILE
```

### run_enhanced_ml_pipeline.py

```bash
# Run specific steps of the pipeline
python run_enhanced_ml_pipeline.py --mode MODE --setup-list FILE --step STEP
```

### evaluate_predictions.py

```bash
# Evaluate predictions for all domains
python evaluate_predictions.py

# Evaluate predictions for a specific domain
python evaluate_predictions.py --domain DOMAIN
``` 