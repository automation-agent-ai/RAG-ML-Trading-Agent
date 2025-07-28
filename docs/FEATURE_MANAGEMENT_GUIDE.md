# Feature Management in Training/Prediction Pipeline

This guide explains how features are managed in the training/prediction pipeline, focusing on the different types of features and how they're handled when separating training and prediction data.

## Feature Types

In our pipeline, we have two main types of features:

### 1. Raw Features

These are features extracted directly from the source data without depending on other setups or labels:

- **News features**: Sentiment scores, entity counts, topic classifications
- **Fundamentals features**: Financial ratios, growth rates, balance sheet metrics
- **UserPosts features**: Sentiment analysis, engagement metrics, topic distribution
- **Analyst Recommendations features**: Rating counts, consensus metrics

### 2. Similarity-Based Features

These are features derived from comparing a setup with other similar setups:

- **Positive signal strength**: Weighted ratio of positive outcomes in similar cases
- **Negative risk score**: Weighted ratio of negative outcomes in similar cases
- **Neutral probability**: Weighted ratio of neutral outcomes in similar cases
- **Historical pattern confidence**: Confidence score based on similarity distribution
- **Similar cases count**: Number of similar cases found

## Feature Management Approaches

When separating training and prediction data, we have different options for handling features:

### Option 1: Complete Re-extraction

- **Process**:
  1. Remove both embeddings and features for prediction setups
  2. Create new embeddings without labels
  3. Extract all features from scratch in prediction mode

- **Advantages**:
  - Complete separation between training and prediction data
  - Ensures no data leakage

- **Disadvantages**:
  - Computationally expensive
  - Raw features will be identical to original extraction in most cases

### Option 2: Selective Re-extraction (Recommended)

- **Process**:
  1. Preserve both embeddings and features for prediction setups
  2. Remove embeddings from training tables
  3. Selectively reset only similarity-based features
  4. Create new embeddings without labels
  5. Re-extract only similarity-based features

- **Advantages**:
  - More efficient - avoids redundant computation
  - Maintains data separation where it matters (similarity-based features)
  - Raw features remain consistent

- **Disadvantages**:
  - Requires careful tracking of which features are similarity-based

## Implementation

We've implemented the selective re-extraction approach using the following tools:

### 1. `preserve_restore_embeddings.py`

This script provides functions to:
- Preserve embeddings and features for prediction setups
- Remove prediction setups from training tables
- Reset only similarity-based features
- Restore preserved embeddings and features back to training tables

#### Key Commands

```bash
# Preserve data
python preserve_restore_embeddings.py preserve --prediction-setup-file data/prediction_setups.txt

# Remove from training
python preserve_restore_embeddings.py remove --prediction-setup-file data/prediction_setups.txt

# Reset only similarity features
python preserve_restore_embeddings.py reset-similarity --prediction-setup-file data/prediction_setups.txt

# Restore preserved data
python preserve_restore_embeddings.py restore --preserved-file data/preserved_data_20230101.pkl
```

### 2. `run_enhanced_ml_pipeline.py`

This script runs the pipeline in either training or prediction mode:

```bash
# Training mode
python run_enhanced_ml_pipeline.py --mode training --step features

# Prediction mode
python run_enhanced_ml_pipeline.py --mode prediction --setup-list data/prediction_setups.txt --step features
```

### 3. `ensemble_prediction.py`

This script combines predictions from multiple domains:

```bash
# Create ensemble predictions
python ensemble_prediction.py --method weighted --setup-list data/prediction_setups.txt --output data/ensemble_predictions.csv
```

## Optimized Workflow

The recommended workflow for managing features is:

1. **Select prediction setups**:
   ```bash
   python create_prediction_list.py --count 100 --output data/prediction_setups.txt
   ```

2. **Preserve data**:
   ```bash
   python preserve_restore_embeddings.py preserve --prediction-setup-file data/prediction_setups.txt
   ```

3. **Remove prediction setups from training**:
   ```bash
   python preserve_restore_embeddings.py remove --prediction-setup-file data/prediction_setups.txt
   ```

4. **Create prediction embeddings**:
   ```bash
   python run_enhanced_ml_pipeline.py --mode prediction --setup-list data/prediction_setups.txt --step embeddings
   ```

5. **Reset similarity features** (optional if you want to keep raw features):
   ```bash
   python preserve_restore_embeddings.py reset-similarity --prediction-setup-file data/prediction_setups.txt
   ```

6. **Extract prediction features**:
   ```bash
   python run_enhanced_ml_pipeline.py --mode prediction --setup-list data/prediction_setups.txt --step features
   ```

7. **Create ensemble predictions**:
   ```bash
   python ensemble_prediction.py --method weighted --setup-list data/prediction_setups.txt --output data/ensemble_predictions.csv
   ```

8. **Evaluate predictions**:
   ```bash
   python evaluate_predictions.py --predictions data/ensemble_predictions.csv
   ```

9. **Restore original data** (when ready to return to training):
   ```bash
   python preserve_restore_embeddings.py restore --preserved-file data/preserved_data_*.pkl
   ```

## Feature Differences Between Training and Prediction

The key differences in features between training and prediction modes:

1. **Label-related features**: In prediction mode, we don't have access to the true labels (`outperformance_10d`), so any features derived from these labels will be different.

2. **Similarity-based features**: When we extract features in prediction mode, the similarity search is performed against the training embeddings. If some embeddings are moved from training to prediction, the similarity search results (and thus the features) will be different.

3. **Ensemble features**: The ensemble prediction combines results from all domains to create more robust predictions than any single domain could provide.

## Conclusion

By carefully managing which features are preserved, reset, and re-extracted, we can maintain a clean separation between training and prediction data while minimizing unnecessary computation. The selective re-extraction approach offers the best balance between computational efficiency and preventing data leakage. 