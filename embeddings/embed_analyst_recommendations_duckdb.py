#!/usr/bin/env python3
"""
embed_analyst_recommendations_duckdb.py - DuckDB-based Analyst Recommendations Embedding Pipeline

Processes analyst recommendations data from DuckDB database, creates embeddings,
and stores in LanceDB with performance labels for RAG retrieval.

Table Schema: id, ticker, period, strong_buy, buy, hold, sell, strong_sell, mean_rating, setup_id, created_at
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import sys

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from embeddings.base_embedder import BaseEmbedder
from tools.setup_validator_duckdb import SetupValidatorDuckDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalystRecommendationsEmbedderDuckDB(BaseEmbedder):
    """DuckDB-based Analyst Recommendations Domain Embedding Pipeline"""
    
    def __init__(self, db_path: str = "data/sentiment_system.duckdb", 
                 lancedb_dir: str = "lancedb_store",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 include_labels: bool = True,
                 mode: str = "training"):
        """
        Initialize Analyst Recommendations Embedder
        
        Args:
            db_path: Path to DuckDB database
            lancedb_dir: Directory for LanceDB storage
            embedding_model: Model name for sentence transformer
            include_labels: Whether to include performance labels in embeddings
            mode: Either 'training' or 'prediction'
        """
        # Initialize base class
        super().__init__(
            db_path=db_path,
            lancedb_dir=lancedb_dir,
            embedding_model=embedding_model,
            include_labels=include_labels,
            mode=mode
        )
        
        # Data containers
        self.analyst_data = None
        
    def get_text_field_name(self) -> str:
        """Get the name of the text field to embed"""
        return "analyst_summary"
        
    def load_data(self):
        """Load analyst recommendations and labels data from DuckDB"""
        self.logger.info("Loading analyst recommendations data from DuckDB...")
        
        # Load labels for confirmed setups
        super().load_labels()
        
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
        self.logger.info(f"Loaded {len(self.analyst_data)} analyst recommendations")
        
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
        
    def create_embeddings_dataset(self) -> List[Dict[str, Any]]:
        """Create embeddings dataset from analyst recommendations"""
        if self.analyst_data is None or self.analyst_data.empty:
            self.logger.warning("No analyst recommendations data to process")
            return []
        
        self.logger.info("Creating embeddings dataset...")
        records = []
        
        for idx, row in self.analyst_data.iterrows():
            # Create comprehensive text summary
            analyst_summary = self.create_analyst_summary(row)
            
            # Create unique ID
            record_id = f"analyst_{row.get('ticker', 'unknown')}_{row.get('setup_id', 'unknown')}"
            
            # Base record without embedding (will be added by create_embeddings)
            record = {
                'id': record_id,
                'setup_id': row['setup_id'],
                'ticker': row.get('ticker', ''),
                'analyst_summary': analyst_summary,
                'embedded_at': datetime.now().isoformat(),
                'strong_buy': row.get('strong_buy', 0) or 0,
                'buy': row.get('buy', 0) or 0,
                'hold': row.get('hold', 0) or 0,
                'sell': row.get('sell', 0) or 0,
                'strong_sell': row.get('strong_sell', 0) or 0,
                'mean_target_price': float(row.get('mean_target_price', 0)) if pd.notna(row.get('mean_target_price')) else 0,
                'mean_recommendation': float(row.get('mean_recommendation', 0)) if pd.notna(row.get('mean_recommendation')) else 0
            }
            
            records.append(record)
        
        self.logger.info(f"Created {len(records)} analyst recommendation records")
        return records

    def run_pipeline(self):
        """Execute the complete analyst recommendations embedding pipeline"""
        self.logger.info("Starting DuckDB-based Analyst Recommendations Embedding Pipeline")
        
        # Load data
        self.load_data()
        
        # Create embeddings dataset
        records = self.create_embeddings_dataset()
        
        # Enrich with labels (only in training mode)
        enriched_records = self.enrich_with_labels(records)
        
        # Create embeddings
        final_records = self.create_embeddings(enriched_records)
        
        # Store in LanceDB (only in training mode)
        self.store_in_lancedb(final_records, "analyst_embeddings")
        
        # Display summary
        self.display_summary(final_records)
        
        # Close DuckDB connection
        self.cleanup()
    
    def process_setups(self, setup_ids: List[str]) -> bool:
        """Process specific setup IDs"""
        self.logger.info(f"Processing analyst recommendations embeddings for {len(setup_ids)} setups")
        
        # Override confirmed setup IDs
        self.setup_validator.confirmed_setup_ids = set(setup_ids)
        
        try:
            self.run_pipeline()
            return True
        except Exception as e:
            self.logger.error(f"Error processing setups: {e}")
            return False
    
    def display_summary(self, records: List[Dict[str, Any]]) -> None:
        """Display pipeline summary statistics"""
        if not records:
            self.logger.info("No records processed")
            return
        
        self.logger.info("\n" + "="*50)
        self.logger.info("ANALYST RECOMMENDATIONS EMBEDDING PIPELINE SUMMARY")
        self.logger.info("="*50)
        
        # Basic stats
        self.logger.info(f"Total records processed: {len(records)}")
        self.logger.info(f"Unique setups: {len(set(r['setup_id'] for r in records))}")
        self.logger.info(f"Unique tickers: {len(set(r['ticker'] for r in records))}")
        
        # Label stats
        labeled_records = [r for r in records if r.get('has_performance_labels', False)]
        self.logger.info(f"Records with performance labels: {len(labeled_records)}")
        
        self.logger.info("="*50)


def main():
    """Main function to run the analyst recommendations embedding pipeline"""
    embedder = AnalystRecommendationsEmbedderDuckDB()
    embedder.run_pipeline()


if __name__ == "__main__":
    main() 