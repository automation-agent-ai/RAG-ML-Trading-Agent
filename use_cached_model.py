#!/usr/bin/env python3
"""
Patch Script for Using Cached Models

This script modifies the SentenceTransformer initialization in the pipeline
to use locally cached models instead of downloading them from Hugging Face.

Usage:
    python use_cached_model.py
"""

import os
import sys
import logging
import importlib
import inspect
import types
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def patch_sentence_transformer():
    """
    Patch the SentenceTransformer class to always use cached models
    """
    try:
        # Import the SentenceTransformer class
        from sentence_transformers import SentenceTransformer
        
        # Store the original __init__ method
        original_init = SentenceTransformer.__init__
        
        # Define the new __init__ method
        def patched_init(self, model_name_or_path, *args, **kwargs):
            logger.info(f"Patched SentenceTransformer initialization for: {model_name_or_path}")
            
            # Always use our cache directory
            cache_dir = os.path.join("models", "sentence_transformers")
            kwargs["cache_folder"] = cache_dir
            
            # Call the original __init__ method
            original_init(self, model_name_or_path, *args, **kwargs)
        
        # Replace the __init__ method
        SentenceTransformer.__init__ = patched_init
        
        logger.info("Successfully patched SentenceTransformer to use cached models")
        return True
    
    except Exception as e:
        logger.error(f"Failed to patch SentenceTransformer: {e}")
        return False

def patch_agent_files(models_dir="./models"):
    """
    Patch agent files to use cached models
    
    This function modifies the agent files to use cached models by adding
    environment variable settings at the top of the file.
    
    Args:
        models_dir: Directory where models are cached
    """
    # Agent files to patch
    agent_files = [
        "agents/news/enhanced_news_agent_duckdb.py",
        "agents/fundamentals/enhanced_fundamentals_agent_duckdb.py",
        "agents/analyst_recommendations/enhanced_analyst_recommendations_agent_duckdb.py",
        "agents/userposts/enhanced_userposts_agent_complete.py"
    ]
    
    # Environment variables to set
    env_vars = {
        'TRANSFORMERS_CACHE': os.path.join(models_dir, 'cache'),
        'HF_HOME': os.path.join(models_dir, 'hub'),
        'SENTENCE_TRANSFORMERS_HOME': os.path.join(models_dir, 'sentence_transformers')
    }
    
    # Code to insert
    code_to_insert = "\n# Set environment variables for model caching\n"
    for var, value in env_vars.items():
        code_to_insert += f"os.environ['{var}'] = '{value}'\n"
    
    for file_path in agent_files:
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
            
            # Read the file
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check if already patched
            if "TRANSFORMERS_CACHE" in content:
                logger.info(f"File already patched: {file_path}")
                continue
            
            # Find the position to insert the code (after imports)
            import_lines = []
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    import_lines.append(i)
            
            if not import_lines:
                logger.warning(f"No import statements found in {file_path}")
                continue
            
            # Insert after the last import
            insert_position = import_lines[-1] + 1
            
            # Insert the code
            new_lines = lines[:insert_position] + [code_to_insert] + lines[insert_position:]
            new_content = '\n'.join(new_lines)
            
            # Write the file
            with open(file_path, 'w') as f:
                f.write(new_content)
            
            logger.info(f"Successfully patched file: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to patch file {file_path}: {e}")
    
def create_model_init_script(models_dir="./models"):
    """
    Create a script to initialize models at startup
    
    Args:
        models_dir: Directory where models are cached
    """
    # Default model name
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    
    script_content = """#!/usr/bin/env python3
\"\"\"
Model Initialization Script

This script initializes sentence transformer models from the cache.
It should be imported at the beginning of your pipeline to ensure
models are loaded from cache.

Usage:
    import model_init
\"\"\"

import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables for model caching
os.environ['TRANSFORMERS_CACHE'] = '{cache_dir}'
os.environ['HF_HOME'] = '{hub_dir}'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '{st_dir}'

def init_models():
    \"\"\"Initialize models from cache\"\"\"
    try:
        from sentence_transformers import SentenceTransformer
        
        # Initialize the model from cache
        model_name = '{model_name}'
        cache_dir = '{st_dir}'
        
        logger.info(f"Initializing model {{model_name}} from cache")
        model = SentenceTransformer(model_name, cache_folder=cache_dir)
        logger.info(f"Successfully initialized model {{model_name}}")
        
        return model
    except Exception as e:
        logger.error(f"Failed to initialize model: {{e}}")
        return None

# Initialize models when imported
model = init_models()
""".format(
        cache_dir=os.path.join(models_dir, 'cache'),
        hub_dir=os.path.join(models_dir, 'hub'),
        st_dir=os.path.join(models_dir, 'sentence_transformers'),
        model_name=model_name
    )
    
    # Write the script
    script_path = "model_init.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    logger.info(f"Created model initialization script: {script_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Patch pipeline to use cached models')
    parser.add_argument('--models-dir', default='./models',
                      help='Directory where models are cached (default: ./models)')
    
    args = parser.parse_args()
    
    # Set up models directory
    models_dir = Path(args.models_dir)
    models_dir.mkdir(exist_ok=True, parents=True)
    
    # Patch SentenceTransformer
    patch_sentence_transformer()
    
    # Patch agent files
    patch_agent_files(str(models_dir))
    
    # Create model initialization script
    create_model_init_script(str(models_dir))
    
    logger.info("Patching complete. Run 'python download_models.py' to download and cache models.")

if __name__ == "__main__":
    main() 