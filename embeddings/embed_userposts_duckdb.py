#!/usr/bin/env python3
"""
embed_userposts_duckdb.py - DuckDB-based UserPosts Domain Embedding Pipeline

Processes user posts from DuckDB database, creates embeddings of post content,
and stores in LanceDB with metadata and performance labels for RAG retrieval.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import re
import sys

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from embeddings.base_embedder import BaseEmbedder
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


class UserPostsEmbedderDuckDB(BaseEmbedder):
    """DuckDB-based User Posts Domain Embedding Pipeline"""
    
    def __init__(self, db_path: str = "data/sentiment_system.duckdb", 
                 lancedb_dir: str = "lancedb_store",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 max_chunk_size: int = 300,
                 chunk_overlap: int = 50,
                 include_labels: bool = True,
                 mode: str = "training"):
        """
        Initialize User Posts Embedder
        
        Args:
            db_path: Path to DuckDB database
            lancedb_dir: Directory for LanceDB storage
            embedding_model: Model name for sentence transformer
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
        self.posts_data = None
        self.specific_setup_ids = False
        self.target_setup_ids = []
    
    def get_text_field_name(self) -> str:
        """Get the name of the text field to embed"""
        return "post_content"
    
    def load_data(self):
        """Load user posts and labels data from DuckDB"""
        self.logger.info("Loading user posts data from DuckDB...")
        
        # Load labels first (using base class method)
        super().load_labels()
        
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
                self.logger.info(f"Loaded {len(self.posts_data)} user posts for specific setups")
            except Exception as e:
                self.logger.error(f"Error loading user posts: {e}")
                self.posts_data = pd.DataFrame()
        else:
            # Load all confirmed setups (original behavior)
            self.posts_data = self.setup_validator.get_user_posts_for_confirmed_setups()
            self.logger.info(f"Loaded {len(self.posts_data)} user posts")
    
    def process_posts(self):
        """Process user posts data"""
        if self.posts_data is None or self.posts_data.empty:
            self.logger.warning("No user posts data to process")
            return pd.DataFrame()
        
        # Clean post content
        self.posts_data['post_content_clean'] = self.posts_data['post_content'].apply(clean_post_content)
        
        # Filter out empty posts after cleaning
        initial_count = len(self.posts_data)
        self.posts_data = self.posts_data[self.posts_data['post_content_clean'] != '']
        filtered_count = len(self.posts_data)
        self.logger.info(f"Filtered out {initial_count - filtered_count} empty posts, {filtered_count} remaining")
        
        # Extract sentiment indicators
        sentiment_data = self.posts_data['post_content_clean'].apply(extract_sentiment_indicators)
        sentiment_df = pd.DataFrame(sentiment_data.tolist())
        processed_df = pd.concat([self.posts_data, sentiment_df], axis=1)
        
        # Convert post_date to datetime
        if 'post_date' in processed_df.columns:
            processed_df['post_date'] = pd.to_datetime(processed_df['post_date'])
        
        # Add performance label indicators
        processed_df['has_performance_labels'] = False
        
        self.logger.info(f"Processed data: {len(processed_df)} posts")
        return processed_df
    
    def create_embeddings_dataset(self, processed_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create embeddings dataset from processed posts"""
        if processed_df.empty:
            return []
        
        self.logger.info("Creating embeddings dataset...")
        records = []
        total_chunks = 0
        
        for idx, row in processed_df.iterrows():
            # Chunk the post content
            chunks = chunk_post_content(row['post_content_clean'], self.max_chunk_size, self.chunk_overlap)
            
            if not chunks:
                continue
            
            for chunk_idx, chunk in enumerate(chunks):
                # Create unique ID for each chunk
                chunk_id = f"{row['post_id']}_chunk_{chunk_idx}"
                
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
                    
                    # Performance labels placeholder (will be filled by enrich_with_labels)
                    'has_performance_labels': False,
                    'setup_date': '',
                    
                    # Embedding will be added later
                    'embedded_at': datetime.now().isoformat()
                }
                
                records.append(record)
                total_chunks += 1
                
                if total_chunks % 50 == 0:
                    self.logger.info(f"Processed {total_chunks} chunks...")
        
        self.logger.info(f"Created {total_chunks} chunks from {len(processed_df)} posts")
        return records
    
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
            'posts_with_labels': df['has_performance_labels'].sum()
        }
        
        # Add performance metrics only if we have labels
        if 'outperformance_10d' in df.columns and df['has_performance_labels'].any():
            stats.update({
                'avg_stock_return': df[df['has_performance_labels']]['stock_return_10d'].mean(),
                'avg_outperformance': df[df['has_performance_labels']]['outperformance_10d'].mean()
            })
        
        return stats
    
    def run_pipeline(self):
        """Execute the complete user posts embedding pipeline"""
        self.logger.info("Starting DuckDB-based User Posts Domain Embedding Pipeline")
        
        # Load data
        self.load_data()
        
        # Process posts
        processed_df = self.process_posts()
        
        # Create embeddings dataset
        records = self.create_embeddings_dataset(processed_df)
        
        # Enrich with labels (only in training mode)
        enriched_records = self.enrich_with_labels(records)
        
        # Create embeddings
        final_records = self.create_embeddings(enriched_records)
        
        # Store in LanceDB (only in training mode)
        self.store_in_lancedb(final_records, "userposts_embeddings")
        
        # Generate and display summary
        stats = self.generate_summary_stats(final_records)
        
        self.logger.info("Pipeline Summary:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")
        
        # Close DuckDB connection
        self.cleanup()
    
    def process_setups(self, setup_ids: List[str]) -> bool:
        """Process specific setup IDs"""
        self.logger.info(f"Processing user posts embeddings for {len(setup_ids)} setups")
        
        # Set specific setup IDs
        self.specific_setup_ids = True
        self.target_setup_ids = setup_ids
        
        try:
            self.run_pipeline()
            return True
        except Exception as e:
            self.logger.error(f"Error processing setups: {e}")
            return False


def main():
    """Main execution function"""
    embedder = UserPostsEmbedderDuckDB()
    embedder.run_pipeline()


if __name__ == "__main__":
    main() 