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
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        lancedb_dir: str = "storage/lancedb_store",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.db_path = Path(db_path)
        self.lancedb_dir = Path(lancedb_dir)
        
        # Initialize setup validator
        self.setup_validator = SetupValidatorDuckDB(db_path=str(self.db_path))
        logger.info(f"Setup validator initialized with {len(self.setup_validator.confirmed_setup_ids)} confirmed setups")
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        
        # Data containers
        self.analyst_data = None
        self.labels_data = None
        
        # Ensure LanceDB directory exists
        self.lancedb_dir.mkdir(parents=True, exist_ok=True)
        
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
        
    def process_analyst_recommendations(self) -> List[Dict[str, Any]]:
        """Process analyst recommendations data into embeddings"""
        if self.analyst_data is None or self.analyst_data.empty:
            logger.warning("No analyst recommendations data to process")
            return []
            
        logger.info(f"Processing {len(self.analyst_data)} analyst recommendations...")
        
        records = []
        
        for idx, row in self.analyst_data.iterrows():
            # Create comprehensive text summary
            analyst_summary = self.create_analyst_summary(row)
            
            # Create unique ID
            record_id = f"analyst_{row.get('ticker', 'unknown')}_{row.get('setup_id', 'unknown')}"
            
            # Generate embedding
            embedding = self.model.encode(analyst_summary)
            
            # Get performance labels for this setup
            setup_id = row['setup_id']
            setup_labels = self.labels_data[self.labels_data['setup_id'] == setup_id]
            
            if not setup_labels.empty:
                latest_label = setup_labels.iloc[-1]
                has_labels = True
                stock_return = float(latest_label.get('stock_return_10d', 0))
                outperformance = float(latest_label.get('outperformance_10d', 0))
                days_outperformed = int(latest_label.get('days_outperformed_10d', 0))
                benchmark_return = float(latest_label.get('benchmark_return_10d', 0))
            else:
                has_labels = False
                stock_return = outperformance = benchmark_return = 0.0
                days_outperformed = 0
            
            # Calculate derived metrics
            strong_buy = row.get('strong_buy', 0) or 0
            buy = row.get('buy', 0) or 0
            hold = row.get('hold', 0) or 0
            sell = row.get('sell', 0) or 0
            strong_sell = row.get('strong_sell', 0) or 0
            total_recs = strong_buy + buy + hold + sell + strong_sell
            
            buy_ratio = (strong_buy + buy) / total_recs if total_recs > 0 else 0
            sell_ratio = (sell + strong_sell) / total_recs if total_recs > 0 else 0
            
            # Create record
            record = {
                # Identifiers
                'id': record_id,
                'setup_id': setup_id,
                'ticker': row.get('ticker', ''),
                'source_type': 'analyst_recommendation',
                
                # Content
                'analyst_summary': analyst_summary,
                'text_length': len(analyst_summary),
                
                # Recommendation counts (raw)
                'strong_buy_count': strong_buy,
                'buy_count': buy,
                'hold_count': hold,
                'sell_count': sell,
                'strong_sell_count': strong_sell,
                'total_recommendations': total_recs,
                
                # Derived metrics
                'buy_ratio': buy_ratio,
                'sell_ratio': sell_ratio,
                'hold_ratio': hold / total_recs if total_recs > 0 else 0,
                'mean_rating': float(row.get('mean_rating', 0)) if pd.notna(row.get('mean_rating')) else 0.0,
                
                # Metadata
                'period': row.get('period', ''),
                'setup_date': row.get('setup_date', '').strftime('%Y-%m-%d') if pd.notna(row.get('setup_date')) else '',
                'created_at': row.get('created_at', '').strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row.get('created_at')) else '',
                
                # Performance labels
                'has_performance_labels': has_labels,
                'stock_return_10d': stock_return,
                'benchmark_return_10d': benchmark_return,
                'outperformance_10d': outperformance,
                'days_outperformed_10d': days_outperformed,
                
                # Embedding and metadata
                'vector': embedding.tolist(),
                'embedded_at': datetime.now().isoformat()
            }
            
            records.append(record)
                    
        logger.info(f"Created embeddings for {len(records)} analyst recommendations")
        return records
        
    def store_in_lancedb(self, records: List[Dict[str, Any]], table_name: str = "analyst_recommendations_embeddings") -> None:
        """Store embeddings in LanceDB"""
        if not records:
            logger.warning("No records to store")
            return
            
        logger.info(f"Storing {len(records)} records in LanceDB")
        
        # Connect to LanceDB
        db = lancedb.connect(str(self.lancedb_dir))
        
        # Drop existing table if it exists
        try:
            db.drop_table(table_name)
            logger.info(f"Dropped existing table: {table_name}")
        except Exception:
            pass
            
        # Create DataFrame
        df = pd.DataFrame(records)
        df['vector'] = df['vector'].apply(lambda x: np.array(x, dtype=np.float32))
        
        # Handle data types
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'vector':
                df[col] = df[col].astype(str).replace('nan', '')
                
        # Create table
        table = db.create_table(table_name, df)
        logger.info(f"Created LanceDB table '{table_name}' with {len(table)} records")
        
        # Verify the table
        if len(records) > 0:
            sample_query = table.search(records[0]['vector']).limit(3).to_pandas()
            logger.info(f"Table verification: Retrieved {len(sample_query)} sample records")
        
    def run_pipeline(self):
        """Execute the complete analyst recommendations embedding pipeline"""
        logger.info("Starting DuckDB-based Analyst Recommendations Embedding Pipeline")
        
        self.load_data()
        records = self.process_analyst_recommendations()
        self.store_in_lancedb(records)
        
        # Summary
        logger.info("Pipeline Summary:")
        logger.info(f"  Total records: {len(records)}")
        logger.info(f"  With performance labels: {len([r for r in records if r.get('has_performance_labels')])}")
        logger.info(f"  Unique setups: {len(set(r['setup_id'] for r in records))}")
        logger.info(f"  Unique tickers: {len(set(r['ticker'] for r in records))}")
        
        # Close DuckDB connection
        self.setup_validator.close()


def main():
    """Main function to run the analyst recommendations embedding pipeline"""
    embedder = AnalystRecommendationsEmbedderDuckDB()
    embedder.run_pipeline()


if __name__ == "__main__":
    main() 