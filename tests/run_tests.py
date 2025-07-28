#!/usr/bin/env python3
"""
Test runner script for the production pipeline.

This script runs all tests in the tests directory.
"""

import unittest
import sys
import os

def run_all_tests():
    """Run all tests in the tests directory."""
    # Add the parent directory to the path so we can import modules
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Discover and run all tests
    test_suite = unittest.defaultTestLoader.discover(
        start_dir=os.path.dirname(__file__),
        pattern='test_*.py'
    )
    
    # Run the tests
    result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_all_tests()) 