# Managing Setup IDs for Prediction

This guide explains how to properly manage setup IDs to prevent data leakage in the prediction pipeline.

## Understanding Data Leakage

Data leakage occurs when information from outside the training dataset is used to create the model. In our RAG-based ML pipeline, this can happen if:

1. Setup IDs used for prediction already have embeddings with performance labels in the vector store
2. The embeddings created during training mode are used during prediction

## Tools for Managing Setup IDs

We've created two utilities to help you manage setup IDs:

### 1. Check Setup Embeddings

The `check_setup_embeddings.py` tool helps you verify if a setup ID:
- Already exists in any embedding tables
- Contains performance labels (potential data leakage)
- Was processed in training or prediction mode

#### Usage:

```bash
# Check specific setup IDs
python tools/check_setup_embeddings.py --setup-ids SETUP_ID1 SETUP_ID2 --lancedb-dir lancedb_store

# Get only safe setup IDs for prediction
python tools/check_setup_embeddings.py --setup-ids SETUP_ID1 SETUP_ID2 --lancedb-dir lancedb_store --safe-only
```

### 2. Clean Labeled Embeddings

The `clean_labeled_embeddings.py` tool helps you:
- Remove setup IDs with labels from embedding tables
- Optionally recreate the embeddings in prediction mode

#### Usage:

```bash
# Dry run (show what would be removed without making changes)
python tools/clean_labeled_embeddings.py --setup-ids SETUP_ID1 SETUP_ID2 --lancedb-dir lancedb_store --dry-run

# Remove setup IDs with labels
python tools/clean_labeled_embeddings.py --setup-ids SETUP_ID1 SETUP_ID2 --lancedb-dir lancedb_store

# Remove and recreate embeddings in prediction mode
python tools/clean_labeled_embeddings.py --setup-ids SETUP_ID1 SETUP_ID2 --lancedb-dir lancedb_store --recreate
```

## Best Practices for Prediction

### 1. Verify Setup IDs Before Prediction

Always check if setup IDs are safe for prediction before using them:

```bash
python tools/check_setup_embeddings.py --setup-ids SETUP_ID1 SETUP_ID2 --safe-only
```

### 2. Use Separate Setup IDs for Training and Prediction

Keep a clear separation between setup IDs used for training and prediction:

- **Training Set**: Use for model training and store with labels
- **Validation Set**: Use for hyperparameter tuning and model selection
- **Test Set**: Use for final evaluation
- **Prediction Set**: New setup IDs never seen during training

### 3. Clean Up Before Reusing Setup IDs

If you need to reuse a setup ID that was previously used for training:

```bash
# Remove labels and recreate embeddings
python tools/clean_labeled_embeddings.py --setup-ids SETUP_ID1 --recreate
```

### 4. Always Run in the Correct Mode

When running the pipeline:

```bash
# For training
python run_complete_ml_pipeline.py --mode training --setup-ids TRAIN_ID1 TRAIN_ID2

# For prediction
python run_complete_ml_pipeline.py --mode prediction --setup-ids PRED_ID1 PRED_ID2
```

## Troubleshooting

### Identifying Data Leakage

Signs of potential data leakage:
- Unrealistically high model performance
- Perfect predictions on specific setup IDs
- Significant performance drop when using truly new data

### Fixing Data Leakage

If you suspect data leakage:

1. Check all setup IDs used for prediction:
   ```bash
   python tools/check_setup_embeddings.py --setup-ids PRED_ID1 PRED_ID2
   ```

2. Clean any setup IDs with labels:
   ```bash
   python tools/clean_labeled_embeddings.py --setup-ids LEAKY_ID1 LEAKY_ID2 --recreate
   ```

3. Re-run your prediction with clean setup IDs:
   ```bash
   python run_complete_ml_pipeline.py --mode prediction --setup-ids CLEAN_ID1 CLEAN_ID2
   ```

## Advanced: Adding New Setup IDs

To add new setup IDs to the database for prediction:

```python
import duckdb

conn = duckdb.connect('data/sentiment_system.duckdb')

# Add new setups
new_setups = [
    ('NEW_SETUP_001', '2025-05-01', 'AAPL', 'AAPL.L', 'Apple Inc'),
    ('NEW_SETUP_002', '2025-05-15', 'MSFT', 'MSFT.L', 'Microsoft Corp')
]

conn.executemany("""
    INSERT INTO setups (setup_id, spike_timestamp, yahoo_ticker, lse_ticker, company_name)
    VALUES (?, ?, ?, ?, ?)
""", new_setups)

conn.commit()
conn.close()
```

Then verify they're safe for prediction:

```bash
python tools/check_setup_embeddings.py --setup-ids NEW_SETUP_001 NEW_SETUP_002 --safe-only
``` 