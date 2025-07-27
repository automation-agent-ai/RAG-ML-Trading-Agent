"""
Test file for the label converter
"""

import unittest
import numpy as np
from core.label_converter import (
    LabelConverter, 
    PerformanceClass,
    outperformance_to_class,
    outperformance_to_class_int,
    outperformance_to_class_name,
    get_class_distribution
)


class TestLabelConverter(unittest.TestCase):
    """Test the label converter functionality"""
    
    def setUp(self):
        """Set up the test"""
        self.converter = LabelConverter(positive_threshold=0.02, negative_threshold=-0.02)
    
    def test_outperformance_to_class(self):
        """Test conversion from outperformance to class"""
        self.assertEqual(self.converter.outperformance_to_class(0.03), PerformanceClass.POSITIVE)
        self.assertEqual(self.converter.outperformance_to_class(0.01), PerformanceClass.NEUTRAL)
        self.assertEqual(self.converter.outperformance_to_class(-0.03), PerformanceClass.NEGATIVE)
    
    def test_outperformance_to_class_array(self):
        """Test conversion from outperformance array to class array"""
        values = np.array([0.03, 0.01, -0.03])
        expected = np.array([1, 0, -1])
        result = self.converter.outperformance_to_class_int(values)
        np.testing.assert_array_equal(result, expected)
    
    def test_outperformance_to_class_int(self):
        """Test conversion from outperformance to class int"""
        self.assertEqual(self.converter.outperformance_to_class_int(0.03), 1)
        self.assertEqual(self.converter.outperformance_to_class_int(0.01), 0)
        self.assertEqual(self.converter.outperformance_to_class_int(-0.03), -1)
    
    def test_outperformance_to_class_name(self):
        """Test conversion from outperformance to class name"""
        self.assertEqual(self.converter.outperformance_to_class_name(0.03), "positive")
        self.assertEqual(self.converter.outperformance_to_class_name(0.01), "neutral")
        self.assertEqual(self.converter.outperformance_to_class_name(-0.03), "negative")
    
    def test_class_to_thresholds(self):
        """Test getting thresholds from class"""
        self.assertEqual(self.converter.class_to_thresholds(PerformanceClass.POSITIVE), (0.02, float('inf')))
        self.assertEqual(self.converter.class_to_thresholds(PerformanceClass.NEUTRAL), (-0.02, 0.02))
        self.assertEqual(self.converter.class_to_thresholds(PerformanceClass.NEGATIVE), (float('-inf'), -0.02))
        
        # Test with int values
        self.assertEqual(self.converter.class_to_thresholds(1), (0.02, float('inf')))
        self.assertEqual(self.converter.class_to_thresholds(0), (-0.02, 0.02))
        self.assertEqual(self.converter.class_to_thresholds(-1), (float('-inf'), -0.02))
        
        # Test with string values
        self.assertEqual(self.converter.class_to_thresholds("positive"), (0.02, float('inf')))
        self.assertEqual(self.converter.class_to_thresholds("neutral"), (-0.02, 0.02))
        self.assertEqual(self.converter.class_to_thresholds("negative"), (float('-inf'), -0.02))
    
    def test_get_class_distribution(self):
        """Test getting class distribution"""
        values = [0.03, 0.01, -0.03, 0.05, -0.01]
        expected = {"positive": 0.4, "neutral": 0.4, "negative": 0.2}
        self.assertEqual(self.converter.get_class_distribution(values), expected)
    
    def test_get_thresholds(self):
        """Test getting thresholds"""
        expected = {"positive_threshold": 0.02, "negative_threshold": -0.02}
        self.assertEqual(self.converter.get_thresholds(), expected)
    
    def test_custom_thresholds(self):
        """Test with custom thresholds"""
        custom_converter = LabelConverter(positive_threshold=0.05, negative_threshold=-0.05)
        self.assertEqual(custom_converter.outperformance_to_class(0.03), PerformanceClass.NEUTRAL)
        self.assertEqual(custom_converter.outperformance_to_class(0.06), PerformanceClass.POSITIVE)
        self.assertEqual(custom_converter.outperformance_to_class(-0.06), PerformanceClass.NEGATIVE)
    
    def test_module_level_functions(self):
        """Test module level convenience functions"""
        self.assertEqual(outperformance_to_class(0.03), PerformanceClass.POSITIVE)
        self.assertEqual(outperformance_to_class_int(0.03), 1)
        self.assertEqual(outperformance_to_class_name(0.03), "positive")
        
        values = [0.03, 0.01, -0.03, 0.05, -0.01]
        expected = {"positive": 0.4, "neutral": 0.4, "negative": 0.2}
        self.assertEqual(get_class_distribution(values), expected)


if __name__ == "__main__":
    unittest.main() 