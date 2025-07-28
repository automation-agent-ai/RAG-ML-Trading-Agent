#!/usr/bin/env python3
"""
Fix Training Table Error

This script fixes the error in the news agent where it tries to access 'training_table'
which doesn't exist in prediction mode.
"""

import os
import sys
import logging
from pathlib import Path
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_news_agent():
    """Fix the 'training_table' error in the news agent"""
    
    # Path to the news agent file
    news_agent_path = Path("agents/news/enhanced_news_agent_duckdb.py")
    
    if not news_agent_path.exists():
        logger.error(f"News agent file not found at {news_agent_path}")
        return False
    
    # Read the current content
    with open(news_agent_path, 'r') as f:
        content = f.read()
    
    # Find the _extract_single_group_features method
    method_regex = r'def _extract_single_group_features\(self, group_name: str, news_items: List\[Dict\], mode: str = "training"\) -> Dict\[str, Any\]:(.*?)return features'
    match = re.search(method_regex, content, re.DOTALL)
    
    if not match:
        logger.error("Could not find _extract_single_group_features method")
        return False
    
    # Extract the method body
    method_body = match.group(1)
    
    # Check if the method already has the fix
    if "if hasattr(self, 'training_table') and self.training_table is not None" in method_body:
        logger.info("News agent already has the fix for 'training_table'")
        return True
    
    # Find the code that tries to enhance with similarity features
    similarity_code = r'# In prediction mode, enhance with similarity features\n            if mode == "prediction" and self.training_table is not None:'
    
    # Replace with a check for hasattr and is not None
    new_similarity_code = r'# In prediction mode, enhance with similarity features\n            if mode == "prediction" and hasattr(self, "training_table") and self.training_table is not None:'
    
    # Replace in the method body
    new_method_body = method_body.replace(similarity_code, new_similarity_code)
    
    # If no change was made, try another pattern
    if new_method_body == method_body:
        similarity_code = r'if mode == "prediction" and self.training_table is not None:'
        new_similarity_code = r'if mode == "prediction" and hasattr(self, "training_table") and self.training_table is not None:'
        new_method_body = method_body.replace(similarity_code, new_similarity_code)
    
    # Replace the method body in the content
    new_content = content.replace(method_body, new_method_body)
    
    # Write the patched content back to the file
    with open(news_agent_path, 'w') as f:
        f.write(new_content)
    
    logger.info("Added check for 'training_table' attribute in _extract_single_group_features method")
    
    # Also fix the process_setup method
    process_regex = r'def process_setup\(self, setup_id: str, mode: str = None\) -> Optional\[NewsFeatureSchema\]:(.*?)return features'
    match = re.search(process_regex, content, re.DOTALL)
    
    if not match:
        logger.error("Could not find process_setup method")
        return False
    
    # Extract the method body
    method_body = match.group(1)
    
    # Find the code that tries to generate a similarity-based prediction
    similarity_code = r'# In prediction mode, generate similarity-based predictions\n            if current_mode == "prediction" and self.training_table is not None:'
    
    # Replace with a check for hasattr and is not None
    new_similarity_code = r'# In prediction mode, generate similarity-based predictions\n            if current_mode == "prediction" and hasattr(self, "training_table") and self.training_table is not None:'
    
    # Replace in the method body
    new_method_body = method_body.replace(similarity_code, new_similarity_code)
    
    # If no change was made, try another pattern
    if new_method_body == method_body:
        similarity_code = r'if current_mode == "prediction" and self.training_table is not None:'
        new_similarity_code = r'if current_mode == "prediction" and hasattr(self, "training_table") and self.training_table is not None:'
        new_method_body = method_body.replace(similarity_code, new_similarity_code)
    
    # Replace the method body in the content
    new_content = new_content.replace(method_body, new_method_body)
    
    # Write the patched content back to the file
    with open(news_agent_path, 'w') as f:
        f.write(new_content)
    
    logger.info("Added check for 'training_table' attribute in process_setup method")
    
    return True

def main():
    """Main function"""
    print("Fixing 'training_table' error in the news agent...")
    
    success = fix_news_agent()
    
    if success:
        print("\n✅ Successfully fixed news agent")
    else:
        print("\n❌ Failed to fix news agent")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 