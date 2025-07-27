#!/usr/bin/env python3
"""
embed_news_duckdb.py - DuckDB-based News Domain Embedding Pipeline

Processes news data from DuckDB database, creates embeddings,
and stores in LanceDB with performance labels for RAG retrieval.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re
from pathlib import Path

# LanceDB and embeddings
import lancedb
from sentence_transformers import SentenceTransformer

# Text processing
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))
# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.setup_validator_duckdb import SetupValidatorDuckDB

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

print("DuckDB News Domain Embedding Pipeline initialized")


class NewsEmbeddingPipelineDuckDB:
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
        chunk_overlap: int = 50
    ):
        """
        Initialize the News Embedding Pipeline
        
        Args:
            db_path: Path to DuckDB database file
            lancedb_dir: Directory for LanceDB storage
            embedding_model: HuggingFace model for embeddings
            max_chunk_size: Maximum tokens per text chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.db_path = Path(db_path)
        self.lancedb_dir = Path(lancedb_dir)
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize setup validator
        self.setup_validator = SetupValidatorDuckDB(db_path=str(self.db_path))
        print(f"Setup validator initialized with {len(self.setup_validator.confirmed_setup_ids)} confirmed setups")
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize LanceDB
        self.lancedb_dir.mkdir(exist_ok=True)
        self.db = lancedb.connect(str(self.lancedb_dir))
        
        # Data containers
        self.rns_data = None
        self.news_data = None
        self.labels_data = None
    
    def load_data(self) -> None:
        """Load all news domain data from DuckDB"""
        print("Loading news domain data from DuckDB...")
        
        # Load RNS announcements for confirmed setups
        self.rns_data = self.setup_validator.get_news_for_confirmed_setups(news_type='rns')
        print(f"Loaded {len(self.rns_data)} RNS announcements")
        
        # Load enhanced stock news for confirmed setups
        self.news_data = self.setup_validator.get_news_for_confirmed_setups(news_type='enhanced')
        print(f"Loaded {len(self.news_data)} enhanced news articles")
        
        # Load labels for confirmed setups
        self.labels_data = self.setup_validator.get_labels_for_confirmed_setups()
        print(f"Loaded {len(self.labels_data)} confirmed setup labels")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\"\']+', '', text)
        
        # Clean up common web artifacts
        text = re.sub(r'https?://\S+', '[URL]', text)
        text = re.sub(r'www\.\S+', '[URL]', text)
        text = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\b', '[TIME]', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, title: str = "") -> List[Dict[str, Any]]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Main text content
            title: Optional title/headline
            
        Returns:
            List of chunk dictionaries
        """
        if not text and not title:
            return []
        
        # If we have a title, include it in the first chunk
        if title:
            full_text = f"{title}. {text}" if text else title
        else:
            full_text = text
        
        # Split into sentences
        try:
            sentences = sent_tokenize(full_text)
        except:
            sentences = full_text.split('.')
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        current_chunk_idx = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed max chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk.split()) > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'chunk_text': current_chunk.strip(),
                    'chunk_idx': current_chunk_idx,
                    'chunk_type': 'content'
                })
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap*4:] if len(current_chunk) > self.chunk_overlap*4 else current_chunk
                current_chunk = overlap_text + " " + sentence
                current_chunk_idx += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'chunk_text': current_chunk.strip(),
                'chunk_idx': current_chunk_idx,
                'chunk_type': 'content'
            })
        
        return chunks
    
    def process_rns_announcements(self) -> List[Dict[str, Any]]:
        """Process RNS announcements data"""
        if self.rns_data is None or self.rns_data.empty:
            return []
        
        print("Processing RNS announcements...")
        processed_records = []
        
        for idx, row in self.rns_data.iterrows():
            setup_id = row.get('setup_id', '')
            if not setup_id:
                continue
            
            # Clean content fields
            headline = self.clean_text(row.get('headline', ''))
            text_content = self.clean_text(row.get('text', ''))
            
            if not text_content and not headline:
                continue
            
            # Create chunks
            chunks = self.chunk_text(text_content, headline)
            
            if not chunks:
                chunks = [{
                    'chunk_text': headline,
                    'chunk_idx': 0,
                    'chunk_type': 'headline'
                }]
            
            # Process each chunk
            for chunk in chunks:
                record = {
                    'id': f"rns_{row.get('id', idx)}_{chunk['chunk_idx']}",
                    'source_type': 'rns_announcement',
                    'source_id': row.get('id', idx),
                    'setup_id': setup_id,
                    'ticker': row.get('ticker', ''),
                    'headline': headline,
                    'rns_date': str(row.get('rns_date', '')),
                    'rns_time': str(row.get('rns_time', '')),
                    'url': row.get('url', ''),
                    'scraped_at': str(row.get('scraped_at', '')),
                    'content_length': row.get('content_length', 0),
                    'chunk_text': chunk['chunk_text'],
                    'chunk_idx': chunk['chunk_idx'],
                    'chunk_type': chunk['chunk_type'],
                    'text_length': len(chunk['chunk_text']),
                    'created_at': datetime.now().isoformat(),
                    'title': headline,
                    'content_summary': chunk['chunk_text'][:200] + "..." if len(chunk['chunk_text']) > 200 else chunk['chunk_text'],
                    'publisher': 'RNS',
                    'sentiment_score': 0.0,
                    'relevance_score': 1.0,
                    'article_type': 'announcement'
                }
                
                processed_records.append(record)
        
        print(f"Created {len(processed_records)} RNS chunks from confirmed setups")
        return processed_records
    
    def process_enhanced_news(self) -> List[Dict[str, Any]]:
        """Process enhanced stock news data"""
        if self.news_data is None or self.news_data.empty:
            return []
        
        print("Processing enhanced stock news...")
        processed_records = []
        
        for idx, row in self.news_data.iterrows():
            setup_id = row.get('setup_id', '')
            if not setup_id:
                continue
            
            # Clean content fields
            title = self.clean_text(row.get('title', ''))
            content_summary = self.clean_text(row.get('content_summary', ''))
            
            if not content_summary and not title:
                continue
            
            # Create chunks
            chunks = self.chunk_text(content_summary, title)
            
            if not chunks:
                chunks = [{
                    'chunk_text': title,
                    'chunk_idx': 0,
                    'chunk_type': 'title'
                }]
            
            # Process each chunk
            for chunk in chunks:
                record = {
                    'id': f"news_{row.get('id', idx)}_{chunk['chunk_idx']}",
                    'source_type': 'enhanced_news',
                    'source_id': row.get('id', idx),
                    'setup_id': setup_id,
                    'ticker': row.get('ticker', ''),
                    'title': title,
                    'headline': title,
                    'publisher': row.get('publisher', ''),
                    'link': row.get('link', ''),
                    'url': row.get('link', ''),
                    'provider_publish_time': str(row.get('provider_publish_time', '')),
                    'rns_date': str(row.get('provider_publish_time', '')),
                    'sentiment_score': float(row.get('sentiment_score', 0.0)),
                    'relevance_score': float(row.get('relevance_score', 1.0)),
                    'article_type': row.get('article_type', 'story'),
                    'setup_date_window': row.get('setup_date_window', ''),
                    'days_before_setup': row.get('days_before_setup', ''),
                    'chunk_text': chunk['chunk_text'],
                    'chunk_idx': chunk['chunk_idx'],
                    'chunk_type': chunk['chunk_type'],
                    'text_length': len(chunk['chunk_text']),
                    'created_at': datetime.now().isoformat(),
                    'content_summary': content_summary,
                    'scraped_at': str(row.get('created_at', '')),
                    'content_length': len(content_summary) if content_summary else 0,
                    'rns_time': ''
                }
                
                processed_records.append(record)
        
        print(f"Created {len(processed_records)} enhanced news chunks from confirmed setups")
        return processed_records
    
    def enrich_with_labels(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich records with label information"""
        if self.labels_data is None or self.labels_data.empty:
            print("No labels data available for enrichment")
            return records
        
        print("Enriching records with labels...")
        
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
                    'actual_days_calculated': label_row.get('actual_days_calculated', 0)
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
                    'actual_days_calculated': 0
                })
        
        print(f"Enriched {enriched_count} records with labels")
        return records
    
    def create_embeddings(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create embeddings for all text chunks"""
        if not records:
            return records
        
        print("Creating embeddings...")
        
        # Extract texts for batch embedding
        texts = [record['chunk_text'] for record in records]
        
        # Create embeddings in batches
        print(f"Generating embeddings for {len(texts)} text chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Add embeddings to records
        for record, embedding in zip(records, embeddings):
            record['vector'] = embedding.tolist()
        
        print("Embeddings created successfully")
        return records
    
    def store_in_lancedb(self, records: List[Dict[str, Any]], table_name: str = "news_embeddings") -> None:
        """Store embeddings in LanceDB"""
        if not records:
            print("No records to store")
            return
        
        print(f"Storing {len(records)} records in LanceDB table: {table_name}")
        
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
                print(f"Dropped existing table: {table_name}")
            
            # Create new table
            table = self.db.create_table(table_name, df)
            print(f"Successfully created table '{table_name}' with {len(df)} records")
            
            # Verify table creation
            row_count = len(table.to_pandas())
            print(f"Table verification: {row_count} records stored")
            
        except Exception as e:
            print(f"Error storing data in LanceDB: {e}")
            raise
    
    def run_pipeline(self) -> None:
        """Execute the complete news embedding pipeline"""
        print("Starting DuckDB-based News Domain Embedding Pipeline")
        
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
        
        # Store in LanceDB
        self.store_in_lancedb(final_records)
        
        # Display summary
        self.display_summary(final_records)
        
        # Close DuckDB connection
        self.setup_validator.close()
    
    def display_summary(self, records: List[Dict[str, Any]]) -> None:
        """Display pipeline summary statistics"""
        if not records:
            print("No records processed")
            return
        
        print("\n" + "="*50)
        print("NEWS EMBEDDING PIPELINE SUMMARY")
        print("="*50)
        
        # Basic stats
        print(f"Total records processed: {len(records)}")
        print(f"Source types: {set(r['source_type'] for r in records)}")
        print(f"Unique setups: {len(set(r['setup_id'] for r in records))}")
        print(f"Unique tickers: {len(set(r['ticker'] for r in records))}")
        
        # Content stats
        avg_text_length = sum(r['text_length'] for r in records) / len(records)
        print(f"Average text length: {avg_text_length:.1f} characters")
        
        # Label stats
        labeled_records = [r for r in records if r.get('stock_return_10d', 0) != 0]
        print(f"Records with performance labels: {len(labeled_records)}")
        
        print("="*50)


def main():
    """Main execution function"""
    pipeline = NewsEmbeddingPipelineDuckDB()
    pipeline.run_pipeline()


if __name__ == "__main__":
    main() 