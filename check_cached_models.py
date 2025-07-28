#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check if cached models exist and are properly configured.
This script verifies that all necessary model files are cached locally
and that the cache directories are properly configured.
"""

import os
import sys
import logging
from pathlib import Path
import json
from typing import Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Default cache directories
DEFAULT_CACHE_DIRS = [
    os.path.expanduser("~/.cache/huggingface"),
    os.path.expanduser("~/.cache/torch/transformers"),
    os.path.expanduser("~/.cache/torch/sentence_transformers"),
    "models/cached",  # Local project cache
]

# Expected model files (adjust based on your specific models)
EXPECTED_MODELS = {
    "all-MiniLM-L6-v2": [
        "config.json",
        "pytorch_model.bin",
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
    ],
    "distilbert-base-uncased": [
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "vocab.txt",
    ],
    # Add other models as needed
}

def find_model_cache_dirs() -> List[str]:
    """Find all potential model cache directories."""
    cache_dirs = []
    
    # Check environment variables
    for env_var in ["TRANSFORMERS_CACHE", "HF_HOME", "HF_CACHE_HOME"]:
        if env_var in os.environ:
            cache_dirs.append(os.environ[env_var])
    
    # Add default cache directories
    cache_dirs.extend(DEFAULT_CACHE_DIRS)
    
    # Filter to existing directories
    return [d for d in cache_dirs if os.path.isdir(d)]

def check_model_files(model_name: str, cache_dirs: List[str]) -> Tuple[bool, Optional[str]]:
    """Check if all required files for a model exist in any cache directory."""
    expected_files = EXPECTED_MODELS.get(model_name, [])
    if not expected_files:
        logger.warning(f"No file list defined for model {model_name}")
        return False, None
    
    # Search for the model in all cache directories
    for cache_dir in cache_dirs:
        # Check different possible model directory structures
        possible_model_dirs = [
            os.path.join(cache_dir, model_name),
            os.path.join(cache_dir, "models", model_name),
            os.path.join(cache_dir, "hub", model_name),
            os.path.join(cache_dir, "models--" + model_name.replace("/", "--")),
            os.path.join(cache_dir, "sentence-transformers", model_name),
        ]
        
        for model_dir in possible_model_dirs:
            if not os.path.isdir(model_dir):
                continue
            
            # Check if all expected files exist
            missing_files = []
            for file in expected_files:
                file_path = os.path.join(model_dir, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
            
            if not missing_files:
                return True, model_dir
            else:
                logger.warning(f"Model {model_name} at {model_dir} is missing files: {', '.join(missing_files)}")
    
    return False, None

def find_all_cached_models(cache_dirs: List[str]) -> Dict[str, str]:
    """Find all cached models in the cache directories."""
    cached_models = {}
    
    for cache_dir in cache_dirs:
        for root, dirs, files in os.walk(cache_dir):
            # Check if this directory contains model files
            if "config.json" in files and "pytorch_model.bin" in files:
                # Extract model name from path
                model_name = os.path.basename(root)
                cached_models[model_name] = root
    
    return cached_models

def check_embedding_code_for_cache_usage():
    """Check if embedding code is properly configured to use cached models."""
    embedding_files = [
        "run_enhanced_ml_pipeline.py",
        "extract_financial_features_from_duckdb.py",
        "extract_all_ml_features_from_duckdb.py",
    ]
    
    # Add any domain-specific embedding files
    for domain in ["news", "analyst_recommendations", "userposts", "fundamentals"]:
        domain_file = f"agents/{domain}/enhanced_{domain}_agent_duckdb.py"
        if os.path.exists(domain_file):
            embedding_files.append(domain_file)
    
    results = {}
    for file in embedding_files:
        if not os.path.exists(file):
            results[file] = "File not found"
            continue
        
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for cache-related code
        cache_indicators = [
            "cache_dir",
            "local_files_only=True",
            "use_cached_model",
            "offline=True",
            "from_pretrained",
        ]
        
        found_indicators = []
        for indicator in cache_indicators:
            if indicator in content:
                found_indicators.append(indicator)
        
        if found_indicators:
            results[file] = f"Found cache indicators: {', '.join(found_indicators)}"
        else:
            results[file] = "No cache indicators found"
    
    return results

def main():
    """Main function to check cached models."""
    logger.info("Checking cached models...")
    
    # Find cache directories
    cache_dirs = find_model_cache_dirs()
    logger.info(f"Found {len(cache_dirs)} potential cache directories:")
    for cache_dir in cache_dirs:
        logger.info(f"  - {cache_dir}")
    
    # Check expected models
    all_models_found = True
    for model_name in EXPECTED_MODELS:
        found, model_dir = check_model_files(model_name, cache_dirs)
        if found:
            logger.info(f"✅ Model {model_name} found at {model_dir}")
        else:
            logger.error(f"❌ Model {model_name} not found or incomplete")
            all_models_found = False
    
    # Find all cached models
    logger.info("\nSearching for all cached models...")
    cached_models = find_all_cached_models(cache_dirs)
    logger.info(f"Found {len(cached_models)} cached models:")
    for model_name, model_dir in cached_models.items():
        logger.info(f"  - {model_name}: {model_dir}")
    
    # Check embedding code
    logger.info("\nChecking embedding code for cache usage...")
    code_results = check_embedding_code_for_cache_usage()
    for file, result in code_results.items():
        logger.info(f"  - {file}: {result}")
    
    # Overall result
    if all_models_found:
        logger.info("\n✅ All expected models are properly cached")
    else:
        logger.error("\n❌ Some expected models are missing or incomplete")
        logger.info("\nTo fix this, run:")
        logger.info("  python download_models.py")
        logger.info("  python use_cached_model.py")

if __name__ == "__main__":
    main() 