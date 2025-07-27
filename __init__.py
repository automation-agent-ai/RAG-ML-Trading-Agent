"""
Production Pipeline Package

This package contains the complete, production-ready ML pipeline with:
- Enhanced historical financial features
- LLM-based feature extraction agents
- Vector embeddings generation
- Comprehensive feature merging
- ML training and prediction

Main entry point: run_complete_ml_pipeline.py
"""

from .run_complete_ml_pipeline import CompletePipeline
from .export_ml_features import export_training_features, export_prediction_features
from .merge_financial_features import merge_financial_features

__version__ = "1.0.0"
__all__ = ['CompletePipeline', 'export_training_features', 'export_prediction_features', 'merge_financial_features'] 