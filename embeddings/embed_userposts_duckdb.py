#!/usr/bin/env python3
"""
embed_userposts_duckdb.py - DuckDB-based UserPosts Domain Embedding Pipeline

Processes user posts from DuckDB database, creates embeddings of post content,
and stores in LanceDB with metadata and performance labels for RAG retrieval.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import hashlib
import re
import sys

# LanceDB and embeddings
import lancedb
from sentence_transformers import SentenceTransformer

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))
# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.setup_validator_duckdb import SetupValidatorDuckDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_post_content(content: str) -> str:
    """Clean and normalize post content"""
    if not isinstance(content, str):
        return ""
    
    # Remove extra whitespace
    content = re.sub(r'\s+', ' ', content).strip()
    
    # Remove URLs
    content = re.sub(r'http\S+|www\S+', '[URL]', content)
    
    # Remove excessive punctuation
    content = re.sub(r'[!]{2,}', '!', content)
    content = re.sub(r'[?]{2,}', '?', content)
    content = re.sub(r'[.]{3,}', '...', content)
    
    return content


def extract_sentiment_indicators(content: str) -> Dict[str, Any]:
    """Extract sentiment indicators from post content"""
    positive_words = ['buy', 'bullish', 'up', 'good', 'great', 'excellent', 'profit', 'gain', 'strong', 'positive']
    negative_words = ['sell', 'bearish', 'down', 'bad', 'terrible', 'loss', 'weak', 'negative', 'drop', 'fall']
    uncertainty_words = ['uncertain', 'maybe', 'perhaps', 'might', 'could', 'unsure', 'risky', 'volatile']
    
    content_lower = content.lower()
    
    positive_count = sum(1 for word in positive_words if word in content_lower)
    negative_count = sum(1 for word in negative_words if word in content_lower)
    uncertainty_count = sum(1 for word in uncertainty_words if word in content_lower)
    
    # Calculate sentiment score
    total_sentiment_words = positive_count + negative_count
    if total_sentiment_words > 0:
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
    else:
        sentiment_score = 0.0
    
    return {
        'positive_indicators': positive_count,
        'negative_indicators': negative_count,
        'uncertainty_indicators': uncertainty_count,
        'sentiment_score': sentiment_score,
        'post_length': len(content)
    }


def chunk_post_content(content: str, max_chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Chunk post content into smaller pieces"""
    if not content:
        return []
    
    words = content.split()
    if len(words) <= max_chunk_size:
        return [content]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + max_chunk_size
        chunk_words = words[start:end]
        chunks.append(' '.join(chunk_words))
        
        if end >= len(words):
            break
        
        start = end - overlap
    
    return chunks


class UserPostsEmbedderDuckDB:
    """DuckDB-based User Posts Domain Embedding Pipeline"""
    
    def __init__(self, db_path: str = "data/sentiment_system.duckdb", 
                 lancedb_dir: str = "lancedb_store",
                 include_labels: bool = True,
                 mode: str = "training"):
        """
        Initialize User Posts Embedder
        
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
        
        logger.info("Loading embedding model: all-MiniLM-L6-v2")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize LanceDB
        self.lancedb_dir.mkdir(exist_ok=True)
        self.db = lancedb.connect(str(self.lancedb_dir))
        
        # Data containers
        self.posts_data = None
        self.labels_data = None
    
    def load_data(self):
        """Load user posts and labels data from DuckDB"""
        logger.info("Loading user posts data from DuckDB...")
        
        # Load user posts for target setups (specific or all confirmed)
        if self.specific_setup_ids:
            # Load only specific setup_ids
            setup_ids_str = ','.join([f"'{sid}'" for sid in self.target_setup_ids])
            
            # Query user posts directly
            try:
                query = f"""
                SELECT *
                FROM user_posts
                WHERE setup_id IN ({setup_ids_str})
                ORDER BY setup_id, post_date
                """
                self.posts_data = self.setup_validator.conn.execute(query).fetchdf()
                logger.info(f"Loaded {len(self.posts_data)} user posts for specific setups")
            except Exception as e:
                logger.error(f"Error loading user posts: {e}")
                self.posts_data = pd.DataFrame()
                
            # Query labels directly
            try:
                query = f"""
                SELECT l.*
                FROM labels l
                WHERE l.setup_id IN ({setup_ids_str})
                ORDER BY l.setup_id
                """
                self.labels_data = self.setup_validator.conn.execute(query).fetchdf()
                logger.info(f"Loaded {len(self.labels_data)} labels for specific setups")
            except Exception as e:
                logger.error(f"Error loading labels: {e}")
                self.labels_data = pd.DataFrame()
        else:
            # Load all confirmed setups (original behavior)
            self.posts_data = self.setup_validator.get_user_posts_for_confirmed_setups()
            logger.info(f"Loaded {len(self.posts_data)} user posts")
            
            self.labels_data = self.setup_validator.get_labels_for_confirmed_setups()
            logger.info(f"Loaded {len(self.labels_data)} confirmed setup labels")
    
    def process_posts(self):
        """Process user posts data"""
        if self.posts_data.empty:
            logger.warning("No user posts data to process")
            return pd.DataFrame()
        
        # Clean post content
        self.posts_data['post_content_clean'] = self.posts_data['post_content'].apply(clean_post_content)
        
        # Filter out empty posts after cleaning
        initial_count = len(self.posts_data)
        self.posts_data = self.posts_data[self.posts_data['post_content_clean'] != '']
        filtered_count = len(self.posts_data)
        logger.info(f"Filtered out {initial_count - filtered_count} empty posts, {filtered_count} remaining")
        
        # Extract sentiment indicators
        sentiment_data = self.posts_data['post_content_clean'].apply(extract_sentiment_indicators)
        sentiment_df = pd.DataFrame(sentiment_data.tolist())
        processed_df = pd.concat([self.posts_data, sentiment_df], axis=1)
        
        # Convert post_date to datetime
        if 'post_date' in processed_df.columns:
            processed_df['post_date'] = pd.to_datetime(processed_df['post_date'])
        
        # Merge with performance labels
        if not self.labels_data.empty:
            merged_df = processed_df.merge(
                self.labels_data[['setup_id', 'stock_return_10d', 'benchmark_return_10d', 'outperformance_10d', 
                               'days_outperformed_10d', 'setup_date']],
                on='setup_id',
                how='left'
            )
        else:
            merged_df = processed_df
            for col in ['stock_return_10d', 'benchmark_return_10d', 'outperformance_10d', 'days_outperformed_10d', 'setup_date']:
                merged_df[col] = 0.0 if 'return' in col or 'performance' in col else ''
        
        # Add performance label indicators
        merged_df['has_performance_labels'] = ~merged_df['stock_return_10d'].isna()
        
        # Fill NaN values for performance metrics
        performance_cols = ['stock_return_10d', 'benchmark_return_10d', 'outperformance_10d', 'days_outperformed_10d']
        for col in performance_cols:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(0.0)
        
        logger.info(f"Processed data: {len(merged_df)} posts, {merged_df['has_performance_labels'].sum()} with performance labels")
        return merged_df
    
    def create_embeddings_dataset(self, processed_df: pd.DataFrame, max_chunk_size: int = 300, overlap: int = 50) -> List[Dict[str, Any]]:
        """Create embeddings dataset from processed posts"""
        if processed_df.empty:
            return []
        
        logger.info("Creating embeddings dataset...")
        records = []
        total_chunks = 0
        
        for idx, row in processed_df.iterrows():
            # Chunk the post content
            chunks = chunk_post_content(row['post_content_clean'], max_chunk_size, overlap)
            
            if not chunks:
                continue
            
            for chunk_idx, chunk in enumerate(chunks):
                # Create unique ID for each chunk
                chunk_id = f"{row['post_id']}_chunk_{chunk_idx}"
                
                # Generate embedding
                embedding = self.model.encode(chunk)
                
                # Create record
                record = {
                    # Identifiers
                    'id': chunk_id,
                    'post_id': row['post_id'],
                    'setup_id': row['setup_id'],
                    'ticker': row['ticker'],
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    
                    # Content
                    'post_content': chunk,
                    'full_post_content': row['post_content_clean'],
                    
                    # User and timing
                    'user_handle': row['user_handle'],
                    'post_date': row['post_date'].isoformat() if pd.notna(row['post_date']) else '',
                    'post_url': row.get('post_url', ''),
                    'scraping_timestamp': str(row.get('scraping_timestamp', '')),
                    
                    # Sentiment analysis
                    'positive_indicators': int(row['positive_indicators']),
                    'negative_indicators': int(row['negative_indicators']),
                    'uncertainty_indicators': int(row['uncertainty_indicators']),
                    'sentiment_score': float(row['sentiment_score']),
                    'post_length': int(row['post_length']),
                    
                    # Performance labels (ONLY for training - exclude in prediction mode)
                    'has_performance_labels': bool(row['has_performance_labels']),
                    # NOTE: Removed performance metrics to prevent data leakage during prediction
                    # 'stock_return_10d': float(row['stock_return_10d']),
                    # 'benchmark_return_10d': float(row['benchmark_return_10d']),
                    # 'outperformance_10d': float(row['outperformance_10d']),
                    # 'days_outperformed_10d': int(row['days_outperformed_10d']),
                    'setup_date': str(row.get('setup_date', '')),
                    
                    # Embedding
                    'vector': embedding.tolist(),
                    'embedded_at': datetime.now().isoformat()
                }
                
                records.append(record)
                total_chunks += 1
                
                if total_chunks % 50 == 0:
                    logger.info(f"Processed {total_chunks} chunks...")
        
        logger.info(f"Created embeddings for {total_chunks} chunks from {len(processed_df)} posts")
        return records
    
    def store_in_lancedb(self, records: List[Dict[str, Any]], table_name: str = "userposts_embeddings") -> None:
        """Store embeddings in LanceDB"""
        if not records:
            logger.warning("No records to store")
            return
        
        logger.info(f"Storing {len(records)} records in LanceDB")
        
        # Ensure directory exists
        self.lancedb_dir.mkdir(parents=True, exist_ok=True)
        
        # Connect to LanceDB
        db = lancedb.connect(str(self.lancedb_dir))
        
        # Drop existing table if it exists
        try:
            db.drop_table(table_name)
            logger.info(f"Dropped existing table: {table_name}")
        except Exception:
            pass
        
        # Create table
        df = pd.DataFrame(records)
        df['vector'] = df['vector'].apply(lambda x: np.array(x, dtype=np.float32))
        
        table = db.create_table(table_name, df)
        
        logger.info(f"Created LanceDB table '{table_name}' with {len(table)} records")
        
        # Verify the table
        sample_query = table.search(records[0]['vector']).limit(3).to_pandas()
        logger.info(f"Table verification: Retrieved {len(sample_query)} sample records")
    
    def generate_summary_stats(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics about the embedded data"""
        if not records:
            return {}
        
        df = pd.DataFrame(records)
        
        stats = {
            'total_chunks': len(records),
            'unique_posts': df['post_id'].nunique(),
            'unique_setups': df['setup_id'].nunique(),
            'unique_tickers': df['ticker'].nunique(),
            'unique_users': df['user_handle'].nunique(),
            'avg_chunk_length': df['post_length'].mean(),
            'avg_sentiment_score': df['sentiment_score'].mean(),
            'posts_with_labels': df['has_performance_labels'].sum(),
            'avg_stock_return': df[df['has_performance_labels']]['stock_return_10d'].mean() if df['has_performance_labels'].any() else 0,
            'avg_outperformance': df[df['has_performance_labels']]['outperformance_10d'].mean() if df['has_performance_labels'].any() else 0
        }
        
        return stats
    
    def run_pipeline(self):
        """Execute the complete user posts embedding pipeline"""
        logger.info("Starting DuckDB-based User Posts Domain Embedding Pipeline")
        
        # Load data
        self.load_data()
        
        # Process posts
        processed_df = self.process_posts()
        
        # Create embeddings
        records = self.create_embeddings_dataset(processed_df)
        
        # Store in LanceDB
        self.store_in_lancedb(records)
        
        # Generate and display summary
        stats = self.generate_summary_stats(records)
        
        logger.info("Pipeline Summary:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Close DuckDB connection
        self.setup_validator.close()


def main():
    """Main execution function"""
    embedder = UserPostsEmbedderDuckDB()
    embedder.run_pipeline()


if __name__ == "__main__":
    main() 