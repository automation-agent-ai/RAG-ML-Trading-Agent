#!/usr/bin/env python3
"""
embed_analyst_recommendations_duckdb.py - DuckDB-based Analyst Recommendations Embedding Pipeline

Processes analyst recommendations data from DuckDB database, creates embeddings,
and stores in LanceDB with performance labels for RAG retrieval.

Table Schema: id, ticker, period, strong_buy, buy, hold, sell, strong_sell, mean_rating, setup_id, created_at
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import sys
import re

import lancedb
from sentence_transformers import SentenceTransformer

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.setup_validator_duckdb import SetupValidatorDuckDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalystRecommendationsEmbedderDuckDB:
    """DuckDB-based Analyst Recommendations Domain Embedding Pipeline"""
    
    def __init__(self, db_path: str = "data/sentiment_system.duckdb", 
                 lancedb_dir: str = "lancedb_store",
                 include_labels: bool = True,
                 mode: str = "training"):
        """
        Initialize Analyst Recommendations Embedder
        
        Args:
            db_path: Path to DuckDB database
            lancedb_dir: Directory for LanceDB storage
            include_labels: Whether to include performance labels in embeddings
            mode: Either 'training' or 'prediction'
        """
        self.db_path = Path(db_path)
        self.lancedb_dir = Path(lancedb_dir)
        self.include_labels = include_labels
        self.mode = mode
        
        # Initialize setup validator
        self.setup_validator = SetupValidatorDuckDB(db_path=str(self.db_path))
        logger.info(f"Setup validator initialized with {len(self.setup_validator.confirmed_setup_ids)} confirmed setups")
        
        # Initialize models
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize LanceDB
        self.lancedb_dir.mkdir(exist_ok=True)
        self.db = lancedb.connect(str(self.lancedb_dir))
        
        # Data containers
        self.analyst_data = None
        self.labels_data = None
        
    def load_data(self):
        """Load analyst recommendations and labels data from DuckDB"""
        logger.info("Loading analyst recommendations data from DuckDB...")
        
        # Get all analyst recommendations for confirmed setups
        # Handle ticker format variations: try both plain and .L suffix matching
        analyst_query = '''
            SELECT 
                ar.*,
                s.spike_timestamp as setup_date
            FROM analyst_recommendations ar
            JOIN setups s ON (
                ar.ticker = s.lse_ticker 
                OR ar.ticker = s.lse_ticker || '.L'
                OR ar.ticker = s.yahoo_ticker
                OR ar.ticker = s.yahoo_ticker || '.L'
            ) AND ar.setup_id = s.setup_id
            WHERE s.setup_id IN ({})
                AND ar.id IS NOT NULL
            ORDER BY s.setup_id
        '''.format(','.join([f"'{sid}'" for sid in self.setup_validator.confirmed_setup_ids]))
        
        self.analyst_data = self.setup_validator.conn.execute(analyst_query).df()
        logger.info(f"Loaded {len(self.analyst_data)} analyst recommendations")
        
        # Load labels for confirmed setups
        self.labels_data = self.setup_validator.get_labels_for_confirmed_setups()
        logger.info(f"Loaded {len(self.labels_data)} confirmed setup labels")
        
    def create_analyst_summary(self, row: pd.Series) -> str:
        """Create a comprehensive text summary for an analyst recommendation"""
        parts = []
        
        # Basic info
        parts.append(f"Analyst recommendations for {row.get('ticker', 'Unknown')} (Period: {row.get('period', 'Unknown')})")
        
        # Recommendation counts
        strong_buy = row.get('strong_buy', 0) or 0
        buy = row.get('buy', 0) or 0
        hold = row.get('hold', 0) or 0
        sell = row.get('sell', 0) or 0
        strong_sell = row.get('strong_sell', 0) or 0
        
        total_recs = strong_buy + buy + hold + sell + strong_sell
        
        if total_recs > 0:
            parts.append(f"Total recommendations: {total_recs}")
            parts.append(f"Strong Buy: {strong_buy}, Buy: {buy}, Hold: {hold}, Sell: {sell}, Strong Sell: {strong_sell}")
            
            # Calculate percentages
            buy_pct = round((strong_buy + buy) / total_recs * 100, 1)
            hold_pct = round(hold / total_recs * 100, 1)
            sell_pct = round((sell + strong_sell) / total_recs * 100, 1)
            
            parts.append(f"Buy/Strong Buy: {buy_pct}%, Hold: {hold_pct}%, Sell/Strong Sell: {sell_pct}%")
        
        # Mean rating
        mean_rating = row.get('mean_rating')
        if pd.notna(mean_rating) and mean_rating > 0:
            parts.append(f"Mean rating: {mean_rating:.2f}")
            
            # Interpret rating
            if mean_rating <= 1.5:
                sentiment = "Very Bullish"
            elif mean_rating <= 2.5:
                sentiment = "Bullish" 
            elif mean_rating <= 3.5:
                sentiment = "Neutral"
            elif mean_rating <= 4.5:
                sentiment = "Bearish"
            else:
                sentiment = "Very Bearish"
            parts.append(f"Overall sentiment: {sentiment}")
        
        # Timing context
        if pd.notna(row.get('setup_date')):
            parts.append(f"Setup date: {row['setup_date']}")
            
        if pd.notna(row.get('created_at')):
            parts.append(f"Data as of: {row['created_at']}")
        
        return " | ".join(parts) if parts else f"Analyst data for {row.get('ticker', 'Unknown')}"
        
    def create_embeddings_with_labels(self):
        """Create embeddings with optional labels based on mode"""
        embeddings_data = []
        
        for idx, row in self.analyst_data.iterrows():
            # Create comprehensive text summary
            analyst_summary = self.create_analyst_summary(row)
            
            # Create unique ID
            record_id = f"analyst_{row.get('ticker', 'unknown')}_{row.get('setup_id', 'unknown')}"
            
            # Generate embedding
            embedding = self.model.encode(analyst_summary)
            
            # Base record without labels
            record = {
                'id': record_id,
                'setup_id': row['setup_id'],
                'ticker': row.get('ticker', ''),
                'analyst_summary': analyst_summary,
                'vector': embedding.tolist(),
                'embedded_at': datetime.now().isoformat(),
                'strong_buy': row.get('strong_buy', 0) or 0,
                'buy': row.get('buy', 0) or 0,
                'hold': row.get('hold', 0) or 0,
                'sell': row.get('sell', 0) or 0,
                'strong_sell': row.get('strong_sell', 0) or 0,
                'mean_target_price': float(row.get('mean_target_price', 0)) if pd.notna(row.get('mean_target_price')) else 0,
                'mean_recommendation': float(row.get('mean_recommendation', 0)) if pd.notna(row.get('mean_recommendation')) else 0
            }
            
            # Add performance labels only if in training mode
            if self.include_labels and self.mode == "training":
                setup_id = row['setup_id']
                setup_labels = self.labels_data[self.labels_data['setup_id'] == setup_id]
                
                if not setup_labels.empty:
                    latest_label = setup_labels.iloc[-1]
                    record.update({
                        'stock_return_10d': float(latest_label.get('stock_return_10d', 0)),
                        'outperformance_10d': float(latest_label.get('outperformance_10d', 0)),
                        'days_outperformed_10d': int(latest_label.get('days_outperformed_10d', 0)),
                        'benchmark_return_10d': float(latest_label.get('benchmark_return_10d', 0)),
                        'has_performance_labels': True
                    })
                else:
                    record.update({
                        'stock_return_10d': 0.0,
                        'outperformance_10d': 0.0,
                        'days_outperformed_10d': 0,
                        'benchmark_return_10d': 0.0,
                        'has_performance_labels': False
                    })
            else:
                # In prediction mode, don't include labels
                record.update({
                    'stock_return_10d': 0.0,
                    'outperformance_10d': 0.0,
                    'days_outperformed_10d': 0,
                    'benchmark_return_10d': 0.0,
                    'has_performance_labels': False
                })
            
            embeddings_data.append(record)
        
        logger.info(f"Created {len(embeddings_data)} analyst recommendation embeddings")
        return embeddings_data

    def store_embeddings(self, embeddings_data, table_name=None):
        """Store embeddings in LanceDB with mode-specific table names"""
        if not embeddings_data:
            logger.warning("No embeddings to store")
            return
        
        if table_name is None:
            table_name = "analyst_embeddings_training" if self.mode == "training" else "analyst_embeddings_prediction"
        
        logger.info(f"Storing {len(embeddings_data)} embeddings in table: {table_name}")
        
        # Only store in LanceDB if in training mode or explicitly requested
        if self.mode != "training" and not table_name.endswith("_prediction"):
            logger.info("Skipping LanceDB storage in prediction mode")
            return

        # Drop existing table if it exists
        try:
            self.db.drop_table(table_name)
            logger.info(f"Dropped existing table: {table_name}")
        except Exception:
            pass
            
        # Create DataFrame
        df = pd.DataFrame(embeddings_data)
        df['vector'] = df['vector'].apply(lambda x: np.array(x, dtype=np.float32))
        
        # Handle data types
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'vector':
                df[col] = df[col].astype(str).replace('nan', '')
                
        # Create table
        table = self.db.create_table(table_name, df)
        logger.info(f"Created LanceDB table '{table_name}' with {len(table)} records")
        
        # Verify the table
        if len(embeddings_data) > 0:
            sample_query = table.search(embeddings_data[0]['vector']).limit(3).to_pandas()
            logger.info(f"Table verification: Retrieved {len(sample_query)} sample records")
        
    def run_pipeline(self):
        """Execute the complete analyst recommendations embedding pipeline"""
        logger.info("Starting DuckDB-based Analyst Recommendations Embedding Pipeline")
        
        self.load_data()
        embeddings_data = self.create_embeddings_with_labels()
        self.store_embeddings(embeddings_data)
        
        # Summary
        logger.info("Pipeline Summary:")
        logger.info(f"  Total records: {len(embeddings_data)}")
        logger.info(f"  With performance labels: {len([r for r in embeddings_data if r.get('has_performance_labels')])}")
        logger.info(f"  Unique setups: {len(set(r['setup_id'] for r in embeddings_data))}")
        logger.info(f"  Unique tickers: {len(set(r['ticker'] for r in embeddings_data))}")
        
        # Close DuckDB connection
        self.setup_validator.close()


def main():
    """Main function to run the analyst recommendations embedding pipeline"""
    embedder = AnalystRecommendationsEmbedderDuckDB()
    embedder.run_pipeline()


if __name__ == "__main__":
    main() 