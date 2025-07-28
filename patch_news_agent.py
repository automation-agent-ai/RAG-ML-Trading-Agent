#!/usr/bin/env python3
"""
Patch News Agent

This script patches the news agent to handle "Director Dealings" correctly.
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

def patch_news_agent():
    """Patch the news agent to handle 'Director Dealings' correctly"""
    
    # Path to the news agent file
    news_agent_path = Path("agents/news/enhanced_news_agent_duckdb.py")
    
    if not news_agent_path.exists():
        logger.error(f"News agent file not found at {news_agent_path}")
        return False
    
    # Read the current content
    with open(news_agent_path, 'r') as f:
        content = f.read()
    
    # Check if the file already contains the fix
    if "director dealing" in content.lower() and "director dealings" in content.lower():
        logger.info("News agent already contains the fix for 'Director Dealings'")
    
    # Fix 1: Update the category_counts dictionary in CategoryClassifier.__init__
    # Find the CategoryClassifier.__init__ method
    init_regex = r'def __init__\(self, openai_client: OpenAI, use_cached_models: bool = False\):(.*?)self\.category_counts = {'
    init_match = re.search(init_regex, content, re.DOTALL)
    
    if not init_match:
        logger.error("Could not find CategoryClassifier.__init__ method")
        return False
    
    # Extract the method body
    init_body = init_match.group(1)
    
    # Find the category_counts dictionary initialization
    category_counts_regex = r'self\.category_counts = {(.*?)}'
    category_counts_match = re.search(category_counts_regex, content, re.DOTALL)
    
    if not category_counts_match:
        logger.error("Could not find category_counts dictionary initialization")
        return False
    
    # Extract the category_counts dictionary
    category_counts_dict = category_counts_match.group(1)
    
    # Check if "director_dealings" is already in the dictionary
    if "director_dealings" in category_counts_dict.lower():
        logger.info("'Director Dealings' already exists in category_counts dictionary")
    else:
        # Add "Director Dealings" to the dictionary
        new_category_counts_dict = category_counts_dict
        if category_counts_dict.strip().endswith(","):
            new_category_counts_dict = category_counts_dict + '\n            "director_dealings": 0'
        else:
            new_category_counts_dict = category_counts_dict + ',\n            "director_dealings": 0'
        
        # Replace the category_counts dictionary in the content
        new_content = content.replace(category_counts_dict, new_category_counts_dict)
        
        # Write the patched content back to the file
        with open(news_agent_path, 'w') as f:
            f.write(new_content)
        
        logger.info("Added 'Director Dealings' to category_counts dictionary")
        
        # Update content for next fixes
        content = new_content
    
    # Fix 2: Update the classify_headline method to handle "Director Dealings"
    # Find the classify_headline method
    classify_regex = r'def classify_headline\(self, headline: str\) -> str:(.*?)return category'
    classify_match = re.search(classify_regex, content, re.DOTALL)
    
    if not classify_match:
        logger.error("Could not find classify_headline method")
        return False
    
    # Extract the method body
    classify_body = classify_match.group(1)
    
    # Check if the method already handles "Director Dealings"
    if "director dealings" in classify_body.lower():
        logger.info("classify_headline method already handles 'Director Dealings'")
    else:
        # Add handling for "Director Dealings"
        new_classify_body = classify_body.replace(
            "self.category_counts[category] += 1",
            "if category not in self.category_counts:\n            self.category_counts[category] = 0\n        self.category_counts[category] += 1"
        )
        
        # Replace the method body in the content
        new_content = content.replace(classify_body, new_classify_body)
        
        # Write the patched content back to the file
        with open(news_agent_path, 'w') as f:
            f.write(new_content)
        
        logger.info("Updated classify_headline method to handle 'Director Dealings'")
        
        # Update content for next fixes
        content = new_content
    
    # Fix 3: Add "director dealing" to the _fast_pattern_match method
    # Find the _fast_pattern_match method
    pattern_match_regex = r'def _fast_pattern_match\(self, headline: str\) -> Optional\[str\]:(.*?)return None'
    match = re.search(pattern_match_regex, content, re.DOTALL)
    
    if not match:
        logger.error("Could not find _fast_pattern_match method")
        return False
    
    # Extract the method body
    method_body = match.group(1)
    
    # Check if "director dealings" is already in the patterns
    if "director dealing" in method_body.lower():
        logger.info("'Director Dealings' pattern already exists in _fast_pattern_match")
    else:
        # Add the pattern to the patterns dictionary
        patterns_dict_regex = r'patterns = {(.*?)}'
        patterns_match = re.search(patterns_dict_regex, method_body, re.DOTALL)
        
        if not patterns_match:
            logger.error("Could not find patterns dictionary in _fast_pattern_match method")
            return False
        
        # Extract the patterns dictionary
        patterns_dict = patterns_match.group(1)
        
        # Add the pattern for "director dealings"
        new_patterns_dict = patterns_dict + ',\n            "director dealing": "Director Dealings"'
        
        # Replace the patterns dictionary in the method body
        new_method_body = method_body.replace(patterns_dict, new_patterns_dict)
        
        # Replace the method body in the content
        new_content = content.replace(method_body, new_method_body)
        
        # Write the patched content back to the file
        with open(news_agent_path, 'w') as f:
            f.write(new_content)
        
        logger.info("Added 'Director Dealings' pattern to _fast_pattern_match method")
    
    # Also check the GOVERNANCE_PATTERNS in news_categories.py
    news_categories_path = Path("agents/news/news_categories.py")
    
    if not news_categories_path.exists():
        logger.error(f"News categories file not found at {news_categories_path}")
        return False
    
    # Read the current content
    with open(news_categories_path, 'r') as f:
        categories_content = f.read()
    
    # Check if the file already contains the pattern
    if "director dealing" in categories_content.lower():
        logger.info("News categories already contains the pattern for 'Director Dealings'")
    else:
        # Find the GOVERNANCE_PATTERNS dictionary
        governance_patterns_regex = r'GOVERNANCE_PATTERNS = {(.*?)}'
        governance_match = re.search(governance_patterns_regex, categories_content, re.DOTALL)
        
        if not governance_match:
            logger.error("Could not find GOVERNANCE_PATTERNS dictionary in news_categories.py")
            return False
        
        # Extract the governance patterns dictionary
        governance_patterns = governance_match.group(1)
        
        # Add the pattern for "director dealings"
        new_governance_patterns = governance_patterns + ',\n    "director dealing": "Director Dealings"'
        
        # Replace the governance patterns dictionary in the content
        new_categories_content = categories_content.replace(governance_patterns, new_governance_patterns)
        
        # Write the patched content back to the file
        with open(news_categories_path, 'w') as f:
            f.write(new_categories_content)
        
        logger.info("Added 'Director Dealings' pattern to GOVERNANCE_PATTERNS dictionary")
    
    return True

def main():
    """Main function"""
    print("Patching news agent to handle 'Director Dealings' correctly...")
    
    success = patch_news_agent()
    
    if success:
        print("\n✅ Successfully patched news agent")
    else:
        print("\n❌ Failed to patch news agent")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 