# Data Leakage Analysis Findings

## Summary of Issues

Our analysis has identified several potential data leakage issues in the ML pipeline:

1. **Train-Test Overlap**: We found 25 overlapping setup_ids between training and test data (50% of test data), indicating a significant data leakage problem. This explains the unrealistically high model performance.

2. **Ticker Bias**: 
   - 93-98% of tickers in the test set also appear in the training set
   - Several tickers show strong label bias (>80% of samples have the same label)

3. **Outperformance Feature**: The `outperformance_10d` column is present in both training and test data, which is a direct target leakage since this is what we're trying to predict.

4. **Feature Correlations**: Some features show moderate correlation with the target variable, but nothing suspicious enough to indicate direct leakage through other features.

## Detailed Findings

### Train-Test Overlap

The most serious issue is the direct overlap between training and test data:

- 25 out of 50 test samples (50%) are also present in the training data
- Examples: 'PAY_2024-10-23', 'CML_2024-09-27', 'HMSO_2024-09-20', 'LUCE_2024-10-24', 'BRBY_2024-08-30'

This is a critical problem that would artificially inflate model performance metrics, as the model has already seen these exact samples during training.

### Ticker Bias

We found several tickers with strong label bias:

**Text Features Dataset**:
- BGO: 83.3% are label 2.0 (n=6)
- HBR: 100.0% are label 0.0 (n=5)
- MTRO: 85.7% are label 2.0 (n=7)
- TEP: 83.3% are label 2.0 (n=6)

**Financial Features Dataset**:
- 12 tickers with strong label bias, including:
  - HBR: 100.0% are label 0.0 (n=6)
  - NWG: 100.0% are label 2.0 (n=8)
  - WINE: 100.0% are label 2.0 (n=7)

This suggests that the model could be learning ticker-specific patterns rather than generalizable features.

### Feature Correlations with Target

**Text Features**:
- capital_raise: -0.1294
- max_severity_corporate_actions: -0.0798
- count_corporate_actions: -0.0782

**Financial Features**:
- current_liabilities_to_assets: 0.0605
- calculated_roe: -0.0471
- roe: -0.0471

These correlations are moderate and expected for predictive features.

### Outperformance Feature

The `outperformance_10d` column is included in the datasets and shows correlation with some features:

**Text Features**:
- sentiment_score_corporate_actions: -0.2808
- capital_raise: -0.1040

**Financial Features**:
- current_assets_to_assets: 0.0580
- total_equity_to_assets: -0.0542

This is problematic because `outperformance_10d` is directly related to what we're trying to predict.

## Recommendations

1. **Fix Train-Test Split**:
   - Implement proper train-test splitting that ensures no overlap in setup_ids
   - Consider time-based splitting to prevent future information leakage

2. **Remove Target Leakage**:
   - Remove `outperformance_10d` from feature set during training and prediction
   - Verify no other target-derived features are included

3. **Address Ticker Bias**:
   - Implement stratified sampling by ticker to ensure balanced representation
   - Consider adding ticker as a categorical feature rather than letting it influence other features
   - Alternatively, train separate models for different ticker groups

4. **Cross-Validation Strategy**:
   - Use time-based or group-based cross-validation (grouping by ticker)
   - Ensure validation sets don't contain samples from training set

5. **Feature Engineering**:
   - Review feature creation process to ensure no leakage from target
   - Consider feature selection to remove highly correlated features

6. **Model Evaluation**:
   - Re-evaluate model performance after fixing data leakage issues
   - Expect lower but more realistic performance metrics

## Implementation Plan

1. Create a new data splitting script that ensures:
   - No overlap between train and test sets
   - Proper temporal separation (training data strictly before test data)
   - Stratification by ticker and label where possible

2. Modify feature extraction to:
   - Remove `outperformance_10d` from feature set
   - Add proper data preprocessing (scaling, imputation)

3. Update model training to:
   - Use proper cross-validation strategy
   - Implement early stopping to prevent overfitting
   - Add regularization to improve generalization

4. Re-train models and evaluate performance on clean test set 