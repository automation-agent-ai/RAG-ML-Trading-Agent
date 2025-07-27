# Offline Model Caching Guide

This guide explains how to use the offline model caching solution to avoid HTTP errors and rate limiting issues when running the pipeline.

## Problem

The pipeline uses the `sentence-transformers/all-MiniLM-L6-v2` model from Hugging Face for embedding text. When running the pipeline, you may encounter HTTP 429 errors (Too Many Requests) when trying to download the model configuration files:

```
HTTP Error 429 thrown while requesting HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/...
Retrying in 8s [Retry 5/5].
```

These errors occur because Hugging Face rate-limits requests to their servers. If the final retry fails, the pipeline may fail or produce incomplete results.

## Solution

We've created a solution that:

1. Downloads and caches the model locally
2. Modifies the pipeline to use the cached model
3. Eliminates the need for internet connectivity when running the pipeline

## Step 1: Download and Cache the Model

First, download and cache the model using the provided script:

```bash
python download_models.py
```

This script:
- Downloads the `sentence-transformers/all-MiniLM-L6-v2` model
- Caches it in the `./models` directory
- Sets up proper environment variables for caching

**Options:**
- `--model`: Specify a different model to download
- `--cache-dir`: Specify a different cache directory
- `--force`: Force re-download even if the model is already cached

## Step 2: Patch the Pipeline

Next, patch the pipeline to use the cached model:

```bash
python use_cached_model.py
```

This script:
- Patches the `SentenceTransformer` class to always use cached models
- Modifies agent files to set environment variables for caching
- Creates a model initialization script that can be imported at the start of your pipeline

## Step 3: Use the Cached Model

After patching, you can run the pipeline as usual:

```bash
python run_enhanced_ml_pipeline.py --mode prediction --setup-list data/prediction_setups.txt
```

The pipeline will now use the cached model instead of trying to download it from Hugging Face.

## Verifying the Solution

To verify that the solution is working:

1. Check the logs for messages like:
   ```
   Initializing model sentence-transformers/all-MiniLM-L6-v2 from cache
   Successfully initialized model sentence-transformers/all-MiniLM-L6-v2
   ```

2. You should no longer see HTTP 429 errors or retry messages.

3. The pipeline should run successfully even without internet connectivity.

## Troubleshooting

If you encounter issues:

1. **Model not found in cache**:
   - Run `python download_models.py --force` to re-download the model
   - Check that the `./models` directory exists and contains the model files

2. **Pipeline still trying to download model**:
   - Run `python use_cached_model.py` again to ensure all files are patched
   - Check that the environment variables are being set correctly

3. **Other errors**:
   - Check the logs for specific error messages
   - Ensure you have the required dependencies installed

## Advanced: Using Different Models

If you want to use a different model:

1. Download the model:
   ```bash
   python download_models.py --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
   ```

2. Update the model name in `model_init.py`:
   ```python
   model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
   ```

## Benefits of Offline Caching

Using offline cached models provides several benefits:

1. **Reliability**: No more HTTP errors or rate limiting issues
2. **Speed**: Faster initialization as models are loaded from disk
3. **Offline Operation**: Pipeline can run without internet connectivity
4. **Consistency**: Same model version used every time

## Conclusion

By using the offline model caching solution, you can make your pipeline more reliable and robust against network issues. The solution is easy to implement and requires minimal changes to your existing code. 