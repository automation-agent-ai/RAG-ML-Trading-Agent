#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Patch Agent Files for Cached Model Usage

This script modifies all agent files to ensure they properly use cached models.
It adds the necessary parameters to SentenceTransformer initialization and
sets environment variables to force offline mode.
"""

import os
import sys
import logging
import re
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def find_agent_files():
    """Find all agent files in the project."""
    agent_files = []
    agent_dirs = [
        "agents/news",
        "agents/fundamentals",
        "agents/analyst_recommendations",
        "agents/userposts",
        "embeddings"
    ]
    
    for agent_dir in agent_dirs:
        if os.path.exists(agent_dir):
            for root, _, files in os.walk(agent_dir):
                for file in files:
                    if file.endswith(".py"):
                        agent_files.append(os.path.join(root, file))
    
    return agent_files

def patch_sentence_transformer_init(content):
    """
    Patch SentenceTransformer initialization to use cached models.
    
    This function finds all instances of SentenceTransformer initialization
    and adds parameters to use cached models.
    """
    # Regular expression to find SentenceTransformer initialization
    pattern = r'SentenceTransformer\s*\(\s*(["\'].*?["\'])\s*\)'
    
    # Replacement with cached model parameters
    replacement = r'SentenceTransformer(\1, cache_folder="models/sentence_transformers", local_files_only=True)'
    
    # Apply the replacement
    patched_content = re.sub(pattern, replacement, content)
    
    return patched_content

def patch_huggingface_from_pretrained(content):
    """
    Patch Hugging Face from_pretrained calls to use cached models.
    
    This function finds all instances of from_pretrained calls
    and adds parameters to use cached models.
    """
    # Regular expression to find from_pretrained calls
    pattern = r'\.from_pretrained\s*\(\s*(["\'].*?["\'])\s*\)'
    
    # Replacement with cached model parameters
    replacement = r'.from_pretrained(\1, cache_dir="models/cache", local_files_only=True)'
    
    # Apply the replacement
    patched_content = re.sub(pattern, replacement, content)
    
    return patched_content

def add_env_vars(content):
    """
    Add environment variables to force offline mode.
    
    This function adds environment variable settings at the top of the file
    to force offline mode for Hugging Face models.
    """
    # Check if environment variables are already set
    if "TRANSFORMERS_OFFLINE" in content:
        return content
    
    # Find the position to insert the code (after imports)
    lines = content.split('\n')
    import_lines = []
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_lines.append(i)
    
    if not import_lines:
        logger.warning("No import statements found")
        return content
    
    # Insert after the last import
    insert_position = import_lines[-1] + 1
    
    # Environment variables to set
    env_vars = [
        "# Force offline mode for model loading",
        "import os",
        "os.environ['TRANSFORMERS_OFFLINE'] = '1'",
        "os.environ['HF_DATASETS_OFFLINE'] = '1'",
        "os.environ['TRANSFORMERS_CACHE'] = os.path.join('models', 'cache')",
        "os.environ['HF_HOME'] = os.path.join('models', 'hub')",
        "os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join('models', 'sentence_transformers')",
        ""
    ]
    
    # Insert the environment variables
    new_lines = lines[:insert_position] + env_vars + lines[insert_position:]
    patched_content = '\n'.join(new_lines)
    
    return patched_content

def add_use_cached_models_param(content):
    """
    Add use_cached_models parameter to agent initialization.
    
    This function adds a use_cached_models parameter to the agent's __init__ method
    and modifies the code to use it.
    """
    # Check if use_cached_models parameter already exists
    if "use_cached_models" in content:
        return content
    
    # Regular expression to find __init__ method signature
    pattern = r'def __init__\s*\(\s*self\s*,\s*(.*?)\s*\):'
    
    # Find the __init__ method
    match = re.search(pattern, content)
    if not match:
        logger.warning("Could not find __init__ method")
        return content
    
    # Get the current parameters
    params = match.group(1)
    
    # Add use_cached_models parameter
    if params.endswith(','):
        new_params = f"{params} use_cached_models: bool = False"
    else:
        new_params = f"{params}, use_cached_models: bool = False"
    
    # Replace the __init__ signature
    patched_content = content.replace(match.group(0), f"def __init__(self, {new_params}):")
    
    # Add code to use cached models
    # Find the end of the __init__ method
    lines = patched_content.split('\n')
    init_start = None
    init_end = None
    indent = ""
    
    for i, line in enumerate(lines):
        if "def __init__" in line:
            init_start = i
            # Extract indentation
            next_line_idx = i + 1
            while next_line_idx < len(lines) and not lines[next_line_idx].strip():
                next_line_idx += 1
            if next_line_idx < len(lines):
                indent = re.match(r'^(\s*)', lines[next_line_idx]).group(1)
        elif init_start is not None and line.startswith(f"{indent}def "):
            init_end = i
            break
    
    if init_start is not None and init_end is None:
        # If we didn't find the end, assume it's the end of the file
        init_end = len(lines)
    
    if init_start is not None and init_end is not None:
        # Add code to use cached models
        cached_models_code = [
            f"{indent}# Set up cached models if requested",
            f"{indent}self.use_cached_models = use_cached_models",
            f"{indent}if self.use_cached_models:",
            f"{indent}    logger.info('Using cached models (offline mode)')",
            f"{indent}    os.environ['TRANSFORMERS_OFFLINE'] = '1'",
            f"{indent}    os.environ['HF_DATASETS_OFFLINE'] = '1'"
        ]
        
        # Find a good position to insert the code (after initializing other attributes)
        insert_position = init_start + 1
        for i in range(init_start + 1, init_end):
            if "self." in lines[i] and "=" in lines[i]:
                insert_position = i + 1
        
        # Insert the code
        lines = lines[:insert_position] + cached_models_code + lines[insert_position:]
        patched_content = '\n'.join(lines)
    
    return patched_content

def patch_file(file_path):
    """
    Patch a single file to use cached models.
    
    This function applies all necessary patches to a file.
    """
    logger.info(f"Patching file: {file_path}")
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply patches
        patched_content = content
        patched_content = add_env_vars(patched_content)
        patched_content = patch_sentence_transformer_init(patched_content)
        patched_content = patch_huggingface_from_pretrained(patched_content)
        patched_content = add_use_cached_models_param(patched_content)
        
        # Check if the content was modified
        if patched_content != content:
            # Write the patched file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(patched_content)
            logger.info(f"✅ Successfully patched file: {file_path}")
        else:
            logger.info(f"ℹ️ No changes needed for file: {file_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Error patching file {file_path}: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Patch agent files to use cached models')
    parser.add_argument('--files', nargs='+', help='Specific files to patch')
    parser.add_argument('--all', action='store_true', help='Patch all agent files')
    
    args = parser.parse_args()
    
    if not args.files and not args.all:
        parser.error("Either --files or --all must be specified")
    
    # Files to patch
    files_to_patch = []
    
    if args.all:
        files_to_patch = find_agent_files()
    else:
        files_to_patch = args.files
    
    logger.info(f"Found {len(files_to_patch)} files to patch")
    
    # Patch each file
    success_count = 0
    for file_path in files_to_patch:
        if patch_file(file_path):
            success_count += 1
    
    logger.info(f"✅ Successfully patched {success_count}/{len(files_to_patch)} files")
    
    if success_count < len(files_to_patch):
        logger.warning("⚠️ Some files could not be patched")
    
    logger.info("\nTo ensure all models are properly cached, run:")
    logger.info("  python download_models.py")
    logger.info("  python check_cached_models.py")

if __name__ == "__main__":
    main() 