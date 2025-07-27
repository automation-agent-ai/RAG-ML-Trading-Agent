# Ensemble Prediction Guide

This guide explains how to use the ensemble prediction system to combine predictions from multiple domains for more robust and accurate predictions.

## Overview

The ensemble prediction approach combines predictions from different data domains (news, fundamentals, analyst recommendations, and user posts) to create a more reliable prediction than any single domain could provide. This is based on the principle that different data sources capture different aspects of market behavior, and combining them can reduce noise and increase signal.

## Ensemble Methods

Our system supports two main ensemble methods:

### 1. Majority Voting

In this approach, each domain "votes" for a class (positive, neutral, or negative), and the class with the most votes wins. In case of a tie, the domain with the highest confidence determines the outcome.

**Advantages:**
- Simple and intuitive
- Robust to outliers
- Works well when domains have similar predictive power

**Use when:**
- You want a simple, interpretable approach
- Domains have roughly equal predictive power
- You want to reduce the impact of any single domain

### 2. Weighted Average

This approach calculates a weighted average of the predicted outperformance values from each domain, with weights determined by:
- Domain-specific weights (configurable)
- Confidence scores from each domain

**Advantages:**
- More nuanced predictions
- Can account for varying domain reliability
- Preserves more information from the original predictions

**Use when:**
- Some domains are known to be more reliable than others
- You want to incorporate confidence levels
- You need more fine-grained predictions than just class labels

## Using the Ensemble Prediction Tool

### Basic Usage

```bash
python ensemble_prediction.py --method weighted --output ensemble_predictions.csv
```

### Options

- `--method`: Ensemble method to use (`majority` or `weighted`, default: `weighted`)
- `--db-path`: Path to DuckDB database (default: `data/sentiment_system.duckdb`)
- `--setup-list`: File containing setup IDs to process (optional)
- `--output`: Output file for ensemble predictions (optional)
- `--domains`: Domains to include in ensemble (default: all domains)
- `--weights`: Weights for each domain (must match number of domains)

### Examples

**Using majority voting:**
```bash
python ensemble_prediction.py --method majority --output ensemble_majority.csv
```

**Using weighted average with custom weights:**
```bash
python ensemble_prediction.py --method weighted --domains news fundamentals analyst_recommendations userposts --weights 1.5 1.0 2.0 0.5 --output ensemble_weighted.csv
```

**Processing specific setups:**
```bash
python ensemble_prediction.py --method weighted --setup-list data/prediction_setups.txt --output ensemble_specific.csv
```

## Output

The ensemble prediction tool produces:

1. **CSV file with predictions** containing:
   - `setup_id`: Unique identifier for each setup
   - `predicted_outperformance`: Predicted outperformance value
   - `predicted_class`: Class prediction (-1, 0, 1)
   - `confidence`: Confidence score for the prediction
   - Additional metrics depending on the ensemble method

2. **Evaluation metrics** (if labels are available):
   - Accuracy, precision, recall, F1 score
   - Class-specific metrics
   - Confusion matrix (saved as image)

## Interpreting Results

### Confidence Scores

The confidence score indicates how certain the ensemble is about its prediction:
- Higher values (closer to 1.0) indicate greater confidence
- Lower values (closer to 0.0) indicate uncertainty

### Class Distribution

For majority voting, the output includes vote counts for each class, which can help understand how unanimous the prediction was.

For weighted average, the output includes weighted ratios for each class, providing insight into the distribution of predictions.

## Integration with the Pipeline

The ensemble prediction tool is designed to work with the training/prediction separation pipeline:

1. **Extract features** in prediction mode:
   ```bash
   python run_enhanced_ml_pipeline.py --mode prediction --setup-list data/prediction_setups.txt --step features
   ```

2. **Create ensemble predictions**:
   ```bash
   python ensemble_prediction.py --method weighted --setup-list data/prediction_setups.txt --output data/ensemble_predictions.csv
   ```

3. **Evaluate predictions** (if labels are available):
   ```bash
   python evaluate_predictions.py --predictions data/ensemble_predictions.csv
   ```

## Best Practices

1. **Domain Selection**: Include all domains unless you have a specific reason to exclude one.

2. **Weight Tuning**: Start with equal weights and adjust based on historical performance.

3. **Method Selection**: Use weighted average for more nuanced predictions, majority voting for more conservative predictions.

4. **Confidence Thresholds**: Consider filtering predictions based on confidence scores for production use.

5. **Evaluation**: Always evaluate ensemble performance against individual domain performance to ensure it's adding value.

## Conclusion

Ensemble prediction combines the strengths of different data domains to create more robust predictions. By carefully selecting the ensemble method and parameters, you can optimize for your specific use case and improve overall prediction accuracy. 