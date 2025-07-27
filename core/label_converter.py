"""
Label Converter for ML Pipeline

This module provides a centralized way to convert between outperformance_10d values and class labels.
It also stores the thresholds used for classification in a single place.
"""

import enum
from typing import Dict, Tuple, Union, List, Optional
import numpy as np


class PerformanceClass(enum.Enum):
    """Enum for performance classes"""
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1


class LabelConverter:
    """
    Converts between outperformance_10d values and class labels.
    Centralizes thresholds used for classification.
    """
    
    def __init__(self, 
                 positive_threshold: float = 0.02, 
                 negative_threshold: float = -0.02):
        """
        Initialize the label converter with thresholds
        
        Args:
            positive_threshold: Threshold for positive class (default: 0.02 or 2%)
            negative_threshold: Threshold for negative class (default: -0.02 or -2%)
        """
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
    
    def outperformance_to_class(self, outperformance: Union[float, np.ndarray]) -> Union[PerformanceClass, np.ndarray]:
        """
        Convert outperformance_10d value to class
        
        Args:
            outperformance: The outperformance_10d value or array of values
            
        Returns:
            The corresponding PerformanceClass enum or array of enums
        """
        if isinstance(outperformance, np.ndarray):
            result = np.zeros(outperformance.shape, dtype=int)
            result[outperformance > self.positive_threshold] = PerformanceClass.POSITIVE.value
            result[outperformance < self.negative_threshold] = PerformanceClass.NEGATIVE.value
            return result
        
        if outperformance > self.positive_threshold:
            return PerformanceClass.POSITIVE
        elif outperformance < self.negative_threshold:
            return PerformanceClass.NEGATIVE
        else:
            return PerformanceClass.NEUTRAL
    
    def outperformance_to_class_int(self, outperformance: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """
        Convert outperformance_10d value to class integer (-1, 0, 1)
        
        Args:
            outperformance: The outperformance_10d value or array of values
            
        Returns:
            The corresponding class integer or array of integers
        """
        if isinstance(outperformance, np.ndarray):
            result = np.zeros(outperformance.shape, dtype=int)
            result[outperformance > self.positive_threshold] = PerformanceClass.POSITIVE.value
            result[outperformance < self.negative_threshold] = PerformanceClass.NEGATIVE.value
            return result
        
        if outperformance > self.positive_threshold:
            return PerformanceClass.POSITIVE.value
        elif outperformance < self.negative_threshold:
            return PerformanceClass.NEGATIVE.value
        else:
            return PerformanceClass.NEUTRAL.value
    
    def outperformance_to_class_name(self, outperformance: float) -> str:
        """
        Convert outperformance_10d value to class name
        
        Args:
            outperformance: The outperformance_10d value
            
        Returns:
            The corresponding class name ('positive', 'neutral', 'negative')
        """
        performance_class = self.outperformance_to_class(outperformance)
        
        if performance_class == PerformanceClass.POSITIVE:
            return "positive"
        elif performance_class == PerformanceClass.NEGATIVE:
            return "negative"
        else:
            return "neutral"
    
    def class_to_thresholds(self, performance_class: Union[PerformanceClass, int, str]) -> Tuple[float, float]:
        """
        Get the thresholds for a given class
        
        Args:
            performance_class: The performance class (PerformanceClass enum, int, or string)
            
        Returns:
            Tuple of (min_threshold, max_threshold)
        """
        if isinstance(performance_class, str):
            if performance_class.lower() == "positive":
                performance_class = PerformanceClass.POSITIVE
            elif performance_class.lower() == "negative":
                performance_class = PerformanceClass.NEGATIVE
            else:
                performance_class = PerformanceClass.NEUTRAL
        
        if isinstance(performance_class, int):
            if performance_class == 1:
                performance_class = PerformanceClass.POSITIVE
            elif performance_class == -1:
                performance_class = PerformanceClass.NEGATIVE
            else:
                performance_class = PerformanceClass.NEUTRAL
        
        if performance_class == PerformanceClass.POSITIVE:
            return (self.positive_threshold, float('inf'))
        elif performance_class == PerformanceClass.NEGATIVE:
            return (float('-inf'), self.negative_threshold)
        else:
            return (self.negative_threshold, self.positive_threshold)
    
    def get_class_distribution(self, outperformance_values: List[float]) -> Dict[str, float]:
        """
        Get the distribution of classes in a list of outperformance values
        
        Args:
            outperformance_values: List of outperformance_10d values
            
        Returns:
            Dictionary with class names as keys and ratios as values
        """
        if not outperformance_values:
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        
        total = len(outperformance_values)
        positive = sum(1 for v in outperformance_values if v > self.positive_threshold)
        negative = sum(1 for v in outperformance_values if v < self.negative_threshold)
        neutral = total - positive - negative
        
        return {
            "positive": positive / total,
            "neutral": neutral / total,
            "negative": negative / total
        }
    
    def get_thresholds(self) -> Dict[str, float]:
        """
        Get the current thresholds
        
        Returns:
            Dictionary with threshold names and values
        """
        return {
            "positive_threshold": self.positive_threshold,
            "negative_threshold": self.negative_threshold
        }


# Default instance with standard thresholds
default_converter = LabelConverter()


def outperformance_to_class(outperformance: Union[float, np.ndarray]) -> Union[PerformanceClass, np.ndarray]:
    """Convenience function using the default converter"""
    return default_converter.outperformance_to_class(outperformance)


def outperformance_to_class_int(outperformance: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
    """Convenience function using the default converter"""
    return default_converter.outperformance_to_class_int(outperformance)


def outperformance_to_class_name(outperformance: float) -> str:
    """Convenience function using the default converter"""
    return default_converter.outperformance_to_class_name(outperformance)


def get_class_distribution(outperformance_values: List[float]) -> Dict[str, float]:
    """Convenience function using the default converter"""
    return default_converter.get_class_distribution(outperformance_values) 