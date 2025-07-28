# ML Features Extraction Guide

This guide explains how to extract ML features from the DuckDB database for both training and prediction modes.

## Overview

The feature extraction process involves:

1. Extracting financial features from the `fundamentals` and `financial_ratios` tables
2. Extracting text features from the `news_features`, `userposts_features`, and `analyst_recommendations_features` tables
3. Combining these features into comprehensive ML feature tables for training and prediction

## Scripts

The following scripts are available:

- **`extract_financial_features_from_duckdb.py`**: Extracts financial features
- **`extract_text_features_from_duckdb.py`**: Extracts text features
- **`extract_all_ml_features_from_duckdb.py`**: Runs both financial and text feature extractions

## Usage

### Extract All Features

To extract both financial and text features:

```bash
# For training mode
python extract_all_ml_features_from_duckdb.py --mode training --setup-list data/prediction_setups.txt

# For prediction mode
python extract_all_ml_features_from_duckdb.py --mode prediction --setup-list data/prediction_setups.txt
```

### Extract Only Financial Features

```bash
# For training mode
python extract_all_ml_features_from_duckdb.py --mode training --setup-list data/prediction_setups.txt --features financial

# For prediction mode
python extract_all_ml_features_from_duckdb.py --mode prediction --setup-list data/prediction_setups.txt --features financial
```

### Extract Only Text Features

```bash
# For training mode
python extract_all_ml_features_from_duckdb.py --mode training --setup-list data/prediction_setups.txt --features text

# For prediction mode
python extract_all_ml_features_from_duckdb.py --mode prediction --setup-list data/prediction_setups.txt --features text
```

## Parameters

- `--mode`: Either `training` or `prediction` (default: `training`)
- `--db-path`: Path to the DuckDB database (default: `data/sentiment_system.duckdb`)
- `--output-dir`: Directory to save output CSV files (default: `data/ml_features`)
- `--setup-list`: File containing setup IDs to process (one per line)
- `--features`: Which features to extract: `text`, `financial`, or `all` (default: `all`)

## Output

The scripts create the following outputs:

1. DuckDB tables:
   - `text_ml_features_training` / `text_ml_features_prediction`
   - `financial_ml_features_training` / `financial_ml_features_prediction`

2. CSV files (with timestamps) in the `data/ml_features` directory:
   - `text_ml_features_training_YYYYMMDD_HHMMSS.csv` / `text_ml_features_prediction_YYYYMMDD_HHMMSS.csv`
   - `financial_ml_features_training_YYYYMMDD_HHMMSS.csv` / `financial_ml_features_prediction_YYYYMMDD_HHMMSS.csv`

## Notes on Financial Features

The financial features extraction script handles the ticker format difference between the `setups` table and the `fundamentals`/`financial_ratios` tables:

- In the `setups` table, tickers are stored without the `.L` suffix (e.g., "TSCO")
- In the `fundamentals` and `financial_ratios` tables, tickers are stored with the `.L` suffix (e.g., "TSCO.L")

The script automatically adds the `.L` suffix when joining these tables.

## Analyzing Feature Tables

To analyze the generated feature tables, you can use the following scripts:

- **`analyze_financial_features.py`**: Analyzes financial feature tables
- **`analyze_ml_features.py`**: Analyzes both financial and text feature tables

These scripts provide information about:

- Number of rows and columns
- Columns with non-null values
- Label distribution (for training mode)
- Sample rows with non-null values

They also export clean versions of the tables containing only rows with non-null values for key columns.

## Troubleshooting

### No Financial Data

If you're not seeing any financial data in the output tables, check:

1. The ticker format in the `fundamentals` and `financial_ratios` tables
2. Whether the setup dates match the available financial data dates
3. Whether the setup tickers have any financial data at all

### Database Locked

If you encounter "File is already open" errors, make sure no other process is accessing the database, and try again.

### Missing Features

If certain features are missing, check the corresponding domain-specific feature tables to ensure the data exists. 