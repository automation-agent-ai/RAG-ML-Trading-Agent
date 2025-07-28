# Threshold Consistency Implementation

This document summarizes the changes made to ensure threshold consistency across the ML pipeline and domain agents.

## Problem Statement

Previously, the ML pipeline and domain agents used different thresholding mechanisms:

1. **ML Pipeline**: Used dynamic percentile-based thresholds (33.33% and 66.67%) for balanced classes in training, and fixed thresholds for prediction.
2. **Domain Agents**: Used fixed thresholds (typically -0.02 and 0.02) for all predictions.

This inconsistency made it difficult to compare predictions between ML models and domain agents directly.

## Solution Overview

We implemented a system-wide threshold consistency approach:

1. Calculate dynamic thresholds during label balancing (in training mode)
2. Save these thresholds to a central location
3. Make agent predictions AFTER thresholds are determined
4. Use the same thresholds for both ML models and domain agents

## Implementation Details

### 1. Threshold Manager

Created a new `threshold_manager.py` script that:
- Saves thresholds to a JSON file (`data/label_thresholds.json`)
- Loads thresholds from the file when needed
- Provides a consistent interface for threshold management

```python
# Key methods
def save_thresholds(self, neg_threshold: float, pos_threshold: float, source: str = "dynamic")
def load_thresholds(self) -> Tuple[float, float, str]
def get_thresholds_for_prediction(self) -> Tuple[float, float]
```

### 2. Modified Label Addition

Updated `add_labels_to_features.py` to:
- Save thresholds after calculating them in training mode
- Load thresholds from the threshold manager in prediction mode

### 3. Agent Prediction Integration

Created a new `make_agent_predictions.py` script that:
- Loads thresholds from the threshold manager
- Makes predictions for each domain using these consistent thresholds
- Combines predictions into an ensemble prediction
- Saves all predictions to the similarity_predictions table

### 4. Pipeline Reorganization

Modified `run_complete_ml_pipeline.py` to:
1. Extract features
2. Add labels and calculate thresholds
3. Balance datasets
4. Make agent predictions using consistent thresholds
5. Train ML models
6. Make ML ensemble predictions
7. Generate results table

### 5. Embeddings-Only Mode

Added a new `--embeddings-only` flag to `run_enhanced_ml_pipeline.py` that:
- Creates embeddings without making predictions
- Allows embeddings to be created early in the pipeline
- Defers prediction until after thresholds are determined

Added a `create_embeddings_only` method to each domain agent class.

## Updated Workflow

The updated workflow now follows this sequence:

1. Select setups for training and prediction
2. Preserve data
3. Remove from training
4. Reset similarity features
5. Create embeddings (without predictions)
6. Extract ML features
7. Add labels and calculate thresholds
8. Balance datasets
9. Make agent predictions with consistent thresholds
10. Train ML models
11. Make ML ensemble predictions
12. Generate results table
13. Visualize results

## Benefits

This implementation ensures:

1. **Consistency**: Both ML models and domain agents use the same thresholds
2. **Comparability**: Predictions can be directly compared
3. **Balance**: Class distribution is consistent across all prediction methods
4. **Adaptability**: Thresholds adapt to the data distribution
5. **Transparency**: Threshold values are stored and documented

## Usage

To use the consistent thresholds:

1. Run the pipeline with the updated workflow:
   ```bash
   python run_complete_ml_pipeline.py --mode all
   ```

2. Or run individual steps:
   ```bash
   # Extract features and add labels (calculates and saves thresholds)
   python add_labels_to_features.py --input data/ml_features/text_ml_features_training_*.csv --output data/ml_features/text_ml_features_training_labeled.csv --mode training
   
   # Make agent predictions with consistent thresholds
   python make_agent_predictions.py --setup-list data/prediction_setups.txt
   ```

3. Check the current thresholds:
   ```bash
   python threshold_manager.py load
   ``` 