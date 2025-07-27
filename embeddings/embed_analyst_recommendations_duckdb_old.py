#!/usr/bin/env python3
"""
embed_analyst_recommendations_duckdb.py - DuckDB-based Analyst Recommendations Embedding Pipeline

Processes analyst recommendations data from DuckDB database, creates embeddings,
and stores in LanceDB with performance labels for RAG retrieval.
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
         analyst_query = '''
             SELECT 
                 ar.*,
                 s.setup_id,
                 s.spike_timestamp as setup_date
             FROM analyst_recommendations ar
             JOIN setups s ON ar.ticker = s.lse_ticker
                          WHERE s.setup_id IN ({})
                 AND ar.id IS NOT NULL
             ORDER BY s.setup_id
         '''.format(','.join([f"'{sid}'" for sid in self.setup_validator.confirmed_setup_ids]))
        
        self.analyst_data = self.setup_validator.conn.execute(analyst_query).df()
        logger.info(f"Loaded {len(self.analyst_data)} analyst recommendations")
        
        # Load labels for confirmed setups
        self.labels_data = self.setup_validator.get_labels_for_confirmed_setups()
        logger.info(f"Loaded {len(self.labels_data)} confirmed setup labels")
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        return text
        
    def create_analyst_summary(self, row: pd.Series) -> str:
        """Create a comprehensive text summary for an analyst recommendation"""
        parts = []
        
        # Basic recommendation info
        if pd.notna(row.get('recommendation')):
            parts.append(f"Analyst recommendation: {row['recommendation']}")
            
        if pd.notna(row.get('price_target')):
            parts.append(f"Price target: {row['price_target']}")
            
        if pd.notna(row.get('analyst_firm')):
            parts.append(f"Analyst firm: {row['analyst_firm']}")
            
        if pd.notna(row.get('analyst_name')):
            parts.append(f"Analyst: {row['analyst_name']}")
            
        # Date and timing context
        if pd.notna(row.get('date')):
            parts.append(f"Published: {row['date']}")
            
        # Recommendation details
        if pd.notna(row.get('summary')):
            summary = self.clean_text(str(row['summary']))
            if summary:
                parts.append(f"Summary: {summary}")
                
        if pd.notna(row.get('notes')):
            notes = self.clean_text(str(row['notes']))
            if notes:
                parts.append(f"Notes: {notes}")
                
        # Price target context
        if pd.notna(row.get('previous_price_target')):
            parts.append(f"Previous target: {row['previous_price_target']}")
            
        if pd.notna(row.get('previous_recommendation')):
            parts.append(f"Previous recommendation: {row['previous_recommendation']}")
            
        return " | ".join(parts) if parts else f"Analyst recommendation for {row.get('ticker', 'Unknown')}"
        
    def chunk_text(self, text: str, max_chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= max_chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + max_chunk_size, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + max_chunk_size // 2:
                    end = sentence_end + 1
                    
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            start = max(start + max_chunk_size - overlap, end)
            
        return chunks if chunks else [text[:max_chunk_size]]
        
    def process_analyst_recommendations(self) -> List[Dict[str, Any]]:
        """Process analyst recommendations data into embeddings"""
        if self.analyst_data is None or self.analyst_data.empty:
            logger.warning("No analyst recommendations data to process")
            return []
            
        logger.info(f"Processing {len(self.analyst_data)} analyst recommendations...")
        
        records = []
        total_chunks = 0
        
        for idx, row in self.analyst_data.iterrows():
            # Create comprehensive text summary
            analyst_summary = self.create_analyst_summary(row)
            
            # Chunk the text if needed
            chunks = self.chunk_text(analyst_summary, max_chunk_size=512, overlap=50)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Create unique ID for each chunk
                chunk_id = f"analyst_{row.get('ticker', 'unknown')}_{row.get('date', 'unknown')}_{chunk_idx}"
                
                # Generate embedding
                embedding = self.model.encode(chunk)
                
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
                
                # Create record
                record = {
                    # Identifiers
                    'id': chunk_id,
                    'setup_id': setup_id,
                    'ticker': row.get('ticker', ''),
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'source_type': 'analyst_recommendation',
                    
                    # Content
                    'chunk_text': chunk,
                    'full_text': analyst_summary,
                    'text_length': len(chunk),
                    'full_text_length': len(analyst_summary),
                    
                    # Analyst recommendation details
                    'recommendation': row.get('recommendation', ''),
                    'price_target': float(row.get('price_target', 0)) if pd.notna(row.get('price_target')) else 0.0,
                    'previous_price_target': float(row.get('previous_price_target', 0)) if pd.notna(row.get('previous_price_target')) else 0.0,
                    'previous_recommendation': row.get('previous_recommendation', ''),
                    'analyst_firm': row.get('analyst_firm', ''),
                    'analyst_name': row.get('analyst_name', ''),
                    'date': row.get('date', '').strftime('%Y-%m-%d') if pd.notna(row.get('date')) else '',
                    'summary': self.clean_text(str(row.get('summary', ''))),
                    'notes': self.clean_text(str(row.get('notes', ''))),
                    
                    # Setup context
                    'setup_date': row.get('setup_date', '').strftime('%Y-%m-%d') if pd.notna(row.get('setup_date')) else '',
                    
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
                total_chunks += 1
                
                if total_chunks % 50 == 0:
                    logger.info(f"Processed {total_chunks} chunks...")
                    
        logger.info(f"Created embeddings for {total_chunks} chunks from {len(self.analyst_data)} analyst recommendations")
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