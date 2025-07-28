#!/usr/bin/env python3
"""
Extract Subset of Setup IDs

This script extracts a subset of setup IDs from a file.

Usage:
    python extract_subset_setups.py --input data/fundamentals_setups.txt --output data/subset_setups.txt --limit 1000 --random
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Extract subset of setup IDs')
    parser.add_argument('--input', required=True,
                       help='Input file path')
    parser.add_argument('--output', required=True,
                       help='Output file path')
    parser.add_argument('--limit', type=int, default=1000,
                       help='Maximum number of setup IDs to extract')
    parser.add_argument('--random', action='store_true',
                       help='Randomly sample setup IDs')
    
    args = parser.parse_args()
    
    # Read setup IDs from input file
    with open(args.input, 'r') as f:
        setup_ids = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Read {len(setup_ids)} setup IDs from {args.input}")
    
    # Take a subset
    if args.random:
        if args.limit < len(setup_ids):
            subset_ids = random.sample(setup_ids, args.limit)
        else:
            subset_ids = setup_ids
            logger.warning(f"Requested {args.limit} setup IDs, but only {len(setup_ids)} available")
    else:
        subset_ids = setup_ids[:args.limit]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save to file
    with open(args.output, 'w') as f:
        f.write('\n'.join(subset_ids))
    
    logger.info(f"Saved {len(subset_ids)} setup IDs to {args.output}")

if __name__ == '__main__':
    main() 