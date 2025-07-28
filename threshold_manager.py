#!/usr/bin/env python3
"""
Threshold Manager

This script manages thresholds for label classification across the system.
It provides functions to save and load thresholds to ensure consistency
between ML models and domain agents.

Usage:
    # Save thresholds
    python threshold_manager.py save --neg-threshold -0.21 --pos-threshold 0.28
    
    # Load thresholds
    python threshold_manager.py load
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ThresholdManager:
    """Class for managing label thresholds across the system"""
    
    def __init__(self, threshold_file: str = "data/label_thresholds.json"):
        """
        Initialize the threshold manager
        
        Args:
            threshold_file: Path to the threshold file
        """
        self.threshold_file = Path(threshold_file)
        self.threshold_file.parent.mkdir(exist_ok=True, parents=True)
    
    def save_thresholds(self, neg_threshold: float, pos_threshold: float, source: str = "dynamic") -> None:
        """
        Save thresholds to file
        
        Args:
            neg_threshold: Negative threshold value
            pos_threshold: Positive threshold value
            source: Source of thresholds ('dynamic', 'fixed', etc.)
        """
        thresholds = {
            "neg_threshold": neg_threshold,
            "pos_threshold": pos_threshold,
            "source": source
        }
        
        with open(self.threshold_file, 'w') as f:
            json.dump(thresholds, f, indent=2)
        
        logger.info(f"Thresholds saved to {self.threshold_file}:")
        logger.info(f"- Negative threshold: {neg_threshold:.4f}")
        logger.info(f"- Positive threshold: {pos_threshold:.4f}")
        logger.info(f"- Source: {source}")
    
    def load_thresholds(self) -> Tuple[float, float, str]:
        """
        Load thresholds from file
        
        Returns:
            Tuple of (neg_threshold, pos_threshold, source)
        """
        if not self.threshold_file.exists():
            logger.warning(f"Threshold file not found: {self.threshold_file}")
            logger.warning("Using default thresholds: -0.21, 0.28")
            return -0.21, 0.28, "default"
        
        with open(self.threshold_file, 'r') as f:
            thresholds = json.load(f)
        
        neg_threshold = thresholds.get("neg_threshold", -0.21)
        pos_threshold = thresholds.get("pos_threshold", 0.28)
        source = thresholds.get("source", "unknown")
        
        logger.info(f"Thresholds loaded from {self.threshold_file}:")
        logger.info(f"- Negative threshold: {neg_threshold:.4f}")
        logger.info(f"- Positive threshold: {pos_threshold:.4f}")
        logger.info(f"- Source: {source}")
        
        return neg_threshold, pos_threshold, source
    
    def get_thresholds_for_prediction(self) -> Tuple[float, float]:
        """
        Get thresholds for prediction
        
        Returns:
            Tuple of (neg_threshold, pos_threshold)
        """
        neg_threshold, pos_threshold, _ = self.load_thresholds()
        return neg_threshold, pos_threshold

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Manage label thresholds across the system')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Save command
    save_parser = subparsers.add_parser('save', help='Save thresholds')
    save_parser.add_argument('--neg-threshold', type=float, required=True,
                          help='Negative threshold value')
    save_parser.add_argument('--pos-threshold', type=float, required=True,
                          help='Positive threshold value')
    save_parser.add_argument('--source', default='dynamic',
                          help='Source of thresholds')
    save_parser.add_argument('--threshold-file', default='data/label_thresholds.json',
                          help='Path to threshold file')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load thresholds')
    load_parser.add_argument('--threshold-file', default='data/label_thresholds.json',
                           help='Path to threshold file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize threshold manager
    threshold_manager = ThresholdManager(
        threshold_file=args.threshold_file if hasattr(args, 'threshold_file') else 'data/label_thresholds.json'
    )
    
    # Execute command
    if args.command == 'save':
        threshold_manager.save_thresholds(
            neg_threshold=args.neg_threshold,
            pos_threshold=args.pos_threshold,
            source=args.source
        )
    elif args.command == 'load':
        neg_threshold, pos_threshold, source = threshold_manager.load_thresholds()
        print(f"Negative threshold: {neg_threshold:.4f}")
        print(f"Positive threshold: {pos_threshold:.4f}")
        print(f"Source: {source}")

if __name__ == '__main__':
    main() 