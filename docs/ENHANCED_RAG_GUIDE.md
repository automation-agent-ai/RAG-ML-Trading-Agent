# Enhanced RAG Pipeline Guide

This guide explains how to use the enhanced RAG (Retrieval-Augmented Generation) pipeline for stock performance prediction, which addresses data leakage issues and adds similarity-based prediction capabilities.

## Overview

The enhanced RAG pipeline has two main modes:

1. **Training Mode**: Creates embeddings with labels and stores them in LanceDB for future retrieval
2. **Prediction Mode**: Creates embeddings without labels, doesn't store them in LanceDB, and uses similarity search against training embeddings to generate predictions

## Key Components

- **Embedders**: Create vector embeddings from raw data
  - `BaseEmbedder`: Base class for all embedders
  - Domain-specific embedders: News, Fundamentals, Analyst Recommendations, UserPosts
  
- **Agents**: Extract features and generate predictions
  - Use LLMs for feature extraction
  - Use similarity search for prediction in prediction mode
  
- **Pipeline Orchestrator**: `CompletePipeline` in `run_complete_ml_pipeline.py`
  - Coordinates the entire process
  - Handles different modes and step flags

## Getting Started

### Prerequisites

1. Make sure you have the required Python packages installed:
   ```
   pip install -r requirements.txt
   ```

2. Ensure you have a properly configured DuckDB database with the necessary tables.

### Creating Training Embeddings

Before you can use the pipeline for prediction, you need to create training embeddings:

```bash
python create_training_embeddings.py --setup-ids SETUP_ID1 SETUP_ID2 ... --db-path data/sentiment_system.duckdb --lancedb-dir lancedb_store
```

This will create embeddings with labels and store them in LanceDB for future retrieval.

### Running the Pipeline in Prediction Mode

To run the pipeline in prediction mode:

```bash
python run_complete_ml_pipeline.py --mode prediction --setup-ids SETUP_ID1 SETUP_ID2 ... --db-path data/sentiment_system.duckdb --lancedb-dir lancedb_store
```

This will:
1. Create embeddings without labels (not stored in LanceDB)
2. Extract features from raw data
3. Use similarity search against training embeddings to generate predictions
4. Store predictions in the `similarity_predictions` table

### Checking Setup IDs for Data Leakage

To check if a setup_id already exists in the embeddings vector store and might cause data leakage:

```bash
python tools/check_setup_embeddings.py --setup-ids SETUP_ID1 SETUP_ID2 ...
```

This will tell you if the setup_ids have embeddings with labels that could cause data leakage.

### Cleaning Labeled Embeddings

If you need to remove setup_ids with labels from the embeddings vector store:

```bash
python tools/clean_labeled_embeddings.py --setup-ids SETUP_ID1 SETUP_ID2 ...
```

## Pipeline Steps

The pipeline consists of the following steps:

1. **Create Embeddings**:
   - In training mode: Creates embeddings with labels and stores them in LanceDB
   - In prediction mode: Creates embeddings without labels (not stored)

2. **Extract Features**:
   - In training mode: Extracts features using LLMs
   - In prediction mode: Extracts features and generates predictions using similarity search

3. **Create ML Feature Tables**:
   - Merges features from different domains into comprehensive ML feature tables
   - In training mode: Includes labels
   - In prediction mode: No labels

## Configuration Options

The pipeline can be configured with the following options:

```bash
python run_complete_ml_pipeline.py --help
```

Key options:
- `--mode`: Either 'training' or 'prediction'
- `--setup-ids`: List of setup_ids to process
- `--db-path`: Path to DuckDB database
- `--lancedb-dir`: Path to LanceDB directory
- `--step`: Pipeline step to run ('all', 'embeddings', 'features', 'ml_tables')

## Troubleshooting

### LanceDB Tables Not Found

If the agents can't find the LanceDB tables during feature extraction:

1. Check if the LanceDB directory exists and has the correct permissions
2. Verify that training embeddings were created successfully
3. Ensure the LanceDB path is consistent across all components

### Missing Feature Tables

If the ML feature merger can't find the feature tables:

1. Make sure the feature extraction step completed successfully
2. Check if the feature tables were created in the DuckDB database

## Best Practices

1. **Always create training embeddings first**: Before running the pipeline in prediction mode, make sure you have created training embeddings.

2. **Check for data leakage**: Use the `check_setup_embeddings.py` tool to ensure you're not using setup_ids that could cause data leakage.

3. **Use consistent paths**: Make sure the LanceDB directory path is consistent across all components.

4. **Monitor similarity predictions**: Check the `similarity_predictions` table to see if predictions are being generated correctly.

## Advanced Usage

### Creating Custom Embedders

To create a custom embedder for a new data source:

1. Create a new class that inherits from `BaseEmbedder`
2. Implement the required methods: `get_text_field_name`, `load_data`, etc.
3. Register the embedder in the pipeline orchestrator

### Customizing Similarity Search

To customize how similarity search works:

1. Modify the `find_similar_training_embeddings` method in the agent classes
2. Adjust the `compute_similarity_features` method to change how similarity features are calculated
3. Update the `predict_via_similarity` method to change how predictions are generated

## Example Workflow

1. Create training embeddings:
   ```bash
   python create_training_embeddings.py --setup-ids TRAIN_SETUP_ID1 TRAIN_SETUP_ID2 ...
   ```

2. Check if prediction setup_ids have data leakage:
   ```bash
   python tools/check_setup_embeddings.py --setup-ids PRED_SETUP_ID1 PRED_SETUP_ID2 ...
   ```

3. Run the pipeline in prediction mode:
   ```bash
   python run_complete_ml_pipeline.py --mode prediction --setup-ids PRED_SETUP_ID1 PRED_SETUP_ID2 ...
   ```

4. Check the similarity predictions:
   ```bash
   python tools/check_similarity_predictions.py
   ``` 