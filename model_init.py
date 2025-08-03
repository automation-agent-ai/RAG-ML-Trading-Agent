#!/usr/bin/env python3
"""
Model Initialization Script

This script initializes sentence transformer models from the cache.
It should be imported at the beginning of your pipeline to ensure
models are loaded from cache.

Usage:
    import model_init
"""

import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables for model caching
os.environ['TRANSFORMERS_CACHE'] = 'models/cache'
os.environ['HF_HOME'] = 'models/hub'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = 'models/sentence_transformers'

def init_models():
    """Initialize models from cache"""
    try:
        from sentence_transformers import SentenceTransformer
        
        # Initialize the model from cache
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        cache_dir = 'models/sentence_transformers'
        
        logger.info(f"Initializing model {model_name} from cache")
        model = SentenceTransformer(model_name, cache_folder=cache_dir)
        logger.info(f"Successfully initialized model {model_name}")
        
        return model
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return None

# Initialize models when imported
model = init_models()
