#!/usr/bin/env python3
"""
Base Embedder - Foundation class for all embedding pipelines

Provides common functionality for embedding creation, label handling,
and LanceDB storage to reduce code duplication across embedding classes.
"""

from pathlib import Path
import logging
import lancedb
import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

from tools.setup_validator_duckdb import SetupValidatorDuckDB
# Force offline mode for model loading
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_CACHE'] = os.path.join('models', 'cache')
os.environ['HF_HOME'] = os.path.join('models', 'hub')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join('models', 'sentence_transformers')


class BaseEmbedder:
    """Base class for all embedding pipelines"""
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        lancedb_dir: str = "lancedb_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        include_labels: bool = True,
        mode: str = "training"
    ):
        """
        Initialize base embedder
        
        Args:
            db_path: Path to DuckDB database
            lancedb_dir: Directory for LanceDB storage
            embedding_model: Model name for sentence transformer
            include_labels: Whether to include labels in embeddings
            mode: Either 'training' or 'prediction'
        """
        self.db_path = Path(db_path)
        self.lancedb_dir = Path(lancedb_dir)
        self.include_labels = include_labels
        self.mode = mode
        self.embedding_model_name = embedding_model
        
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize setup validator
        self.setup_validator = SetupValidatorDuckDB(db_path=str(self.db_path))
        self.logger.info(f"Setup validator initialized with {len(self.setup_validator.confirmed_setup_ids)} confirmed setups")
        
        # Initialize embedding model
        self.logger.info(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize LanceDB
        self.lancedb_dir.mkdir(exist_ok=True)
        self.db = lancedb.connect(str(self.lancedb_dir))
        
        # Data containers
        self.labels_data = None
    
    def load_labels(self):
        """Load labels for confirmed setups"""
        self.labels_data = self.setup_validator.get_labels_for_confirmed_setups()
        self.logger.info(f"Loaded {len(self.labels_data)} confirmed setup labels")
        
    def enrich_with_labels(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich records with performance labels if in training mode"""
        if not self.include_labels or self.mode != "training":
            self.logger.info("Skipping label enrichment (prediction mode or labels disabled)")
            return records
        
        if self.labels_data is None or self.labels_data.empty:
            self.logger.warning("No labels data available for enrichment")
            self.load_labels()
            
        if self.labels_data is None or self.labels_data.empty:
            self.logger.warning("Still no labels data available after loading")
            return records
            
        self.logger.info("Enriching records with labels...")
        
        # Create lookup dictionary for labels
        labels_lookup = {}
        for _, label_row in self.labels_data.iterrows():
            setup_id = label_row.get('setup_id', '')
            if setup_id:
                labels_lookup[setup_id] = {
                    'setup_date': str(label_row.get('setup_date', '')),
                    'stock_return_10d': float(label_row.get('stock_return_10d', 0.0)),
                    'benchmark_return_10d': float(label_row.get('benchmark_return_10d', 0.0)),
                    'outperformance_10d': float(label_row.get('outperformance_10d', 0.0)),
                    'days_outperformed_10d': int(label_row.get('days_outperformed_10d', 0)),
                    'benchmark_ticker': label_row.get('benchmark_ticker', ''),
                    'calculation_date': str(label_row.get('calculation_date', '')),
                    'actual_days_calculated': label_row.get('actual_days_calculated', 0),
                    'has_performance_labels': True
                }
        
        # Enrich records
        enriched_count = 0
        for record in records:
            setup_id = record.get('setup_id', '')
            if setup_id and setup_id in labels_lookup:
                record.update(labels_lookup[setup_id])
                enriched_count += 1
            else:
                # Add default values for consistency
                record.update({
                    'setup_date': '',
                    'stock_return_10d': 0.0,
                    'benchmark_return_10d': 0.0,
                    'outperformance_10d': 0.0,
                    'days_outperformed_10d': 0,
                    'benchmark_ticker': '',
                    'calculation_date': '',
                    'actual_days_calculated': 0,
                    'has_performance_labels': False
                })
        
        self.logger.info(f"Enriched {enriched_count} records with labels")
        return records
    
    def create_embeddings(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create embeddings for all text chunks"""
        if not records:
            return records
        
        self.logger.info("Creating embeddings...")
        
        # Extract texts for batch embedding
        texts = []
        for record in records:
            # Get text from the appropriate field (different for each embedder)
            text_field = self.get_text_field_name()
            if text_field in record:
                texts.append(record[text_field])
            else:
                self.logger.warning(f"Text field '{text_field}' not found in record")
                # Use a placeholder text to maintain alignment
                texts.append("")
        
        # Create embeddings in batches
        self.logger.info(f"Generating embeddings for {len(texts)} text chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Add embeddings to records
        for record, embedding in zip(records, embeddings):
            record['vector'] = embedding.tolist()
        
        self.logger.info("Embeddings created successfully")
        return records
    
    def get_text_field_name(self) -> str:
        """Get the name of the text field to embed (override in subclasses)"""
        return "text"  # Default field name, should be overridden
    
    def store_in_lancedb(self, records: List[Dict[str, Any]], table_name: str) -> None:
        """Store embeddings in LanceDB"""
        if not records:
            self.logger.warning("No records to store")
            return
            
        # Skip storage for prediction mode
        if self.mode == "prediction":
            self.logger.info(f"Skipping LanceDB storage in prediction mode")
            return
            
        self.logger.info(f"Storing {len(records)} records in LanceDB table: {table_name}")
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Convert vector column to proper numpy arrays
        df['vector'] = df['vector'].apply(lambda x: np.array(x, dtype=np.float32))
        
        # Handle data types
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'vector':
                df[col] = df[col].astype(str).replace('nan', '')
        
        try:
            # Drop existing table if it exists
            if table_name in self.db.table_names():
                self.db.drop_table(table_name)
                self.logger.info(f"Dropped existing table: {table_name}")
            
            # Create new table
            table = self.db.create_table(table_name, df)
            self.logger.info(f"Successfully created table '{table_name}' with {len(df)} records")
            
            # Verify table creation
            row_count = len(table.to_pandas())
            self.logger.info(f"Table verification: {row_count} records stored")
            
        except Exception as e:
            self.logger.error(f"Error storing data in LanceDB: {e}")
            raise
    
    def process_setups(self, setup_ids: List[str]) -> bool:
        """Process a list of setups (implement in subclasses)"""
        self.logger.warning("process_setups() not implemented in base class")
        return False
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.setup_validator.close()
            self.logger.info("Resources cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}") 