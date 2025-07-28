# 3-Stage ML Pipeline Improvements

## Overview of Enhancements

We have significantly improved the 3-stage ML pipeline by incorporating best practices from the `old_prediction_models` files. These enhancements make the pipeline more robust, interpretable, and potentially more accurate for trading decisions.

## Key Improvements

### 1. Enhanced Feature Preprocessing

- **Robust missing value handling** with improved `SimpleImputer` configuration
- **Better feature type handling** for different data types (object, string, datetime, bool)
- **Handling of all-NaN columns** to prevent training errors
- **Clipping of extreme financial values** to prevent model bias

### 2. Comprehensive Model Evaluation

- **Class-wise metrics** (precision, recall, F1) for each prediction class
- **Detailed confusion matrices** with proper normalization
- **Focus on positive class precision** which is most important for trading
- **Enhanced cross-validation reporting** with standard deviations

### 3. Domain Prediction Correlation Analysis

- **Agreement analysis** between text and financial predictions
- **Class-wise agreement metrics** to understand where domains agree/disagree
- **Probability correlation analysis** using Pearson correlation
- **Disagreement type analysis** to identify conflicting signals

### 4. Richer Ensemble Features

- **Confidence measures** (max probability) for each model
- **Confidence difference metrics** between text and financial models
- **Agreement indicators** to capture consensus between domains
- **Class-specific agreement features** for more nuanced ensemble learning

### 5. Enhanced Visualizations

- **Feature importance plots** for each domain and the ensemble
- **Confusion matrix visualizations** with proper class labels
- **Cross-validation performance charts** with error bars
- **Class distribution visualizations** for balanced evaluation

### 6. Comprehensive Reporting

- **Detailed dataset overview** with class distribution
- **Model performance summaries** with class-specific metrics
- **Domain correlation analysis** to understand complementary signals
- **Trading interpretation** of model performance
- **Comparison to baselines** to quantify improvement

### 7. CSV-Based Feature Loading

- **Direct loading from prepared CSV files** in `data/ml_features/balanced/`
- **Proper label conversion** from -1, 0, 1 to 0, 1, 2 for sklearn compatibility
- **Consistent feature handling** between training and prediction
- **Setup ID verification** to ensure data consistency

### 8. Improved Command-Line Interface

- **Backward compatibility** with previous command formats
- **Flexible input directory specification** for CSV files
- **High-confidence threshold parameter** for prediction filtering
- **Removed hardcoded paths** for better portability

## Usage

The improved pipeline maintains compatibility with the workflow commands in `docs/OPTIMIZED_WORKFLOW.md`:

```bash
# Training
conda activate sts
python train_3stage_ml_pipeline.py --output-dir models_3stage

# Prediction
conda activate sts
python predict_3stage_ml_pipeline.py --models-dir models_3stage --output-dir data/predictions
```

The pipeline now uses the CSV files in `data/ml_features/balanced/` by default, ensuring consistent data handling across the entire workflow. 