#!/usr/bin/env python3
"""
embed_news_duckdb.py - DuckDB-based News Domain Embedding Pipeline

Processes RNS announcements and enhanced stock news data from DuckDB database,
creates embeddings, and stores in LanceDB with performance labels for RAG retrieval.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import sys
import re

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from embeddings.base_embedder import BaseEmbedder
from tools.setup_validator_duckdb import SetupValidatorDuckDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsEmbeddingPipelineDuckDB(BaseEmbedder):
    """
    DuckDB-based News Domain Embedding Pipeline for RAG System
    
    Processes RNS announcements and enhanced stock news data from DuckDB,
    creates semantic embeddings, and stores in LanceDB with rich metadata.
    """
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        lancedb_dir: str = "lancedb_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        max_chunk_size: int = 512,
        chunk_overlap: int = 50,
        include_labels: bool = True,
        mode: str = "training"
    ):
        """
        Initialize the News Embedding Pipeline
        
        Args:
            db_path: Path to DuckDB database file
            lancedb_dir: Directory for LanceDB storage
            embedding_model: HuggingFace model for embeddings
            max_chunk_size: Maximum tokens per text chunk
            chunk_overlap: Overlap between consecutive chunks
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
        
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Data containers
        self.rns_data = None
        self.news_data = None
        
    def get_text_field_name(self) -> str:
        """Get the name of the text field to embed"""
        return "chunk_text"
        
    def load_data(self):
        """Load news data and labels from DuckDB"""
        logger.info("Loading news data from DuckDB...")
        
        # Load labels for confirmed setups
        super().load_labels()
        
        # Load RNS announcements for confirmed setups
        rns_query = '''
            SELECT 
                r.*,
                s.spike_timestamp as setup_date
            FROM rns r
            JOIN setups s ON r.ticker = s.lse_ticker AND r.setup_id = s.setup_id
            WHERE s.setup_id IN ({})
                AND r.headline IS NOT NULL
                AND r.content IS NOT NULL
            ORDER BY s.setup_id
        '''.format(','.join([f"'{sid}'" for sid in self.setup_validator.confirmed_setup_ids]))
        
        self.rns_data = self.setup_validator.conn.execute(rns_query).df()
        logger.info(f"Loaded {len(self.rns_data)} RNS announcements")
        
        # Load enhanced news for confirmed setups
        news_query = '''
            SELECT 
                n.*,
                s.spike_timestamp as setup_date
            FROM enhanced_news n
            JOIN setups s ON n.ticker = s.yahoo_ticker AND n.setup_id = s.setup_id
            WHERE s.setup_id IN ({})
                AND n.headline IS NOT NULL
                AND n.content IS NOT NULL
            ORDER BY s.setup_id
        '''.format(','.join([f"'{sid}'" for sid in self.setup_validator.confirmed_setup_ids]))
        
        self.news_data = self.setup_validator.conn.execute(news_query).df()
        logger.info(f"Loaded {len(self.news_data)} enhanced news items")
        
    def process_rns_announcements(self) -> List[Dict[str, Any]]:
        """Process RNS announcements into embedding records"""
        if self.rns_data is None or self.rns_data.empty:
            logger.warning("No RNS data to process")
            return []
            
        logger.info(f"Processing {len(self.rns_data)} RNS announcements...")
        
        records = []
        
        for _, row in self.rns_data.iterrows():
            # Get content and headline
            content = row.get('content', '')
            headline = row.get('headline', '')
            
            if not content or not headline:
                continue
                
            # Create full text with headline
            full_text = f"{headline}\n\n{content}"
            
            # Create record
            record = {
                'setup_id': row.get('setup_id', ''),
                'ticker': row.get('ticker', ''),
                'source_type': 'rns',
                'headline': headline,
                'chunk_text': full_text,  # No chunking for RNS
                'text_length': len(full_text),
                'rns_date': str(row.get('date', '')),
                'rns_time': str(row.get('time', '')),
                'rns_category': row.get('category', ''),
                'embedded_at': datetime.now().isoformat()
            }
            
            records.append(record)
            
        logger.info(f"Created {len(records)} RNS embedding records")
        return records
        
    def process_enhanced_news(self) -> List[Dict[str, Any]]:
        """Process enhanced news into embedding records"""
        if self.news_data is None or self.news_data.empty:
            logger.warning("No enhanced news data to process")
            return []
            
        logger.info(f"Processing {len(self.news_data)} enhanced news items...")
        
        records = []
        
        for _, row in self.news_data.iterrows():
            # Get content and headline
            content = row.get('content', '')
            headline = row.get('headline', '')
            
            if not content or not headline:
                continue
                
            # Create full text with headline
            full_text = f"{headline}\n\n{content}"
            
            # Create record
            record = {
                'setup_id': row.get('setup_id', ''),
                'ticker': row.get('ticker', ''),
                'source_type': 'enhanced_news',
                'headline': headline,
                'chunk_text': full_text,  # No chunking for enhanced news
                'text_length': len(full_text),
                'news_date': str(row.get('date', '')),
                'news_source': row.get('source', ''),
                'news_url': row.get('url', ''),
                'embedded_at': datetime.now().isoformat()
            }
            
            records.append(record)
            
        logger.info(f"Created {len(records)} enhanced news embedding records")
        return records
    
    def run_pipeline(self) -> None:
        """Execute the complete news embedding pipeline"""
        logger.info("Starting DuckDB-based News Domain Embedding Pipeline")
        
        # Load data
        self.load_data()
        
        # Process both types of news data
        rns_records = self.process_rns_announcements()
        news_records = self.process_enhanced_news()
        
        # Combine all records
        all_records = rns_records + news_records
        
        # Enrich with labels
        enriched_records = self.enrich_with_labels(all_records)
        
        # Create embeddings
        final_records = self.create_embeddings(enriched_records)
        
        # Store in LanceDB (only in training mode)
        self.store_in_lancedb(final_records, "news_embeddings")
        
        # Display summary
        self.display_summary(final_records)
        
        # Close DuckDB connection
        self.cleanup()
    
    def process_setups(self, setup_ids: List[str]) -> bool:
        """Process specific setup IDs"""
        logger.info(f"Processing news embeddings for {len(setup_ids)} setups")
        
        # Override confirmed setup IDs
        self.setup_validator.confirmed_setup_ids = set(setup_ids)
        
        try:
            self.run_pipeline()
            return True
        except Exception as e:
            logger.error(f"Error processing setups: {e}")
            return False
    
    def display_summary(self, records: List[Dict[str, Any]]) -> None:
        """Display pipeline summary statistics"""
        if not records:
            logger.info("No records processed")
            return
        
        logger.info("\n" + "="*50)
        logger.info("NEWS EMBEDDING PIPELINE SUMMARY")
        logger.info("="*50)
        
        # Basic stats
        logger.info(f"Total records processed: {len(records)}")
        logger.info(f"Source types: {set(r['source_type'] for r in records)}")
        logger.info(f"Unique setups: {len(set(r['setup_id'] for r in records))}")
        logger.info(f"Unique tickers: {len(set(r['ticker'] for r in records))}")
        
        # Content stats
        avg_text_length = sum(r['text_length'] for r in records) / len(records)
        logger.info(f"Average text length: {avg_text_length:.1f} characters")
        
        # Label stats
        labeled_records = [r for r in records if r.get('stock_return_10d', 0) != 0]
        logger.info(f"Records with performance labels: {len(labeled_records)}")
        
        logger.info("="*50)

def main():
    """Main function to run the news embedding pipeline"""
    embedder = NewsEmbeddingPipelineDuckDB()
    embedder.run_pipeline()

if __name__ == "__main__":
    main() 