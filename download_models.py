#!/usr/bin/env python3
"""
Model Download and Cache Utility

This script downloads and caches sentence transformer models locally to avoid
HTTP errors and rate limiting issues when running the pipeline.

Usage:
    python download_models.py
"""

import os
import sys
import logging
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_model(model_name, cache_dir='./models', force=False):
    """
    Download and cache a sentence transformer model
    
    Args:
        model_name: Name of the model to download
        cache_dir: Directory to store the model
        force: Whether to force re-download even if cached
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Import here to avoid requiring the package for script help/info
        from sentence_transformers import SentenceTransformer
        import torch
        
        logger.info(f"Downloading model {model_name} to {cache_dir}...")
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set environment variables for caching
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'cache')
        os.environ['HF_HOME'] = os.path.join(cache_dir, 'hub')
        
        # Determine the model directory path
        # For sentence-transformers models, the path is typically:
        # cache_dir/sentence_transformers/model_name_normalized
        model_name_normalized = model_name.replace('/', '_')
        st_cache_dir = os.path.join(cache_dir, 'sentence_transformers')
        model_dir = os.path.join(st_cache_dir, model_name_normalized)
        
        # Force download by clearing cache if requested
        if force and os.path.exists(model_dir):
            logger.info(f"Removing existing cached model at {model_dir}")
            import shutil
            shutil.rmtree(model_dir, ignore_errors=True)
        
        # Download the model
        model = SentenceTransformer(model_name, cache_folder=st_cache_dir)
        
        # Save the model to ensure all files are downloaded
        model.save(model_dir)
        
        # Verify the model was downloaded by checking if files exist
        if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
            logger.info(f"Successfully downloaded model {model_name} to {model_dir}")
            
            # List the model files
            files = os.listdir(model_dir)
            logger.info(f"Model files: {', '.join(files)}")
            
            # Also check the modules directory
            modules_dir = os.path.join(model_dir, "0_Transformer")
            if os.path.exists(modules_dir):
                modules_files = os.listdir(modules_dir)
                logger.info(f"Transformer module files: {', '.join(modules_files)}")
            
            # Check the transformer cache directory
            transformer_cache = os.environ['TRANSFORMERS_CACHE']
            if os.path.exists(transformer_cache):
                logger.info(f"Transformer cache directory exists at {transformer_cache}")
                
            return True
        else:
            logger.error(f"Model directory not found or empty at {model_dir}")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Download and cache sentence transformer models')
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2',
                      help='Model name to download (default: sentence-transformers/all-MiniLM-L6-v2)')
    parser.add_argument('--cache-dir', default='./models',
                      help='Directory to store models (default: ./models)')
    parser.add_argument('--force', action='store_true',
                      help='Force re-download even if model is already cached')
    
    args = parser.parse_args()
    
    # Set up cache directories
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    # Download the model
    success = download_model(args.model, str(cache_dir), args.force)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 