#!/usr/bin/env python3
"""
Enhanced UserPosts Agent (Complete Retrieval Pattern)
Similar to enhanced_news_agent_duckdb.py but for UserPosts domain

This version removes:
- Top-K retrieval limiting
- Cross-encoder reranking
- Vector search limitations

Instead uses:
- Complete record retrieval per setup_id
- All available posts for feature extraction
- Direct LLM processing with chunking for large datasets
"""

import os
import sys
import json
import time
import hashlib
import sqlite3
import logging
import pandas as pd
import lancedb
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
import numpy as np

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
# Import label converter
from core.label_converter import LabelConverter, get_class_distribution

from tools.setup_validator_duckdb import SetupValidatorDuckDB
# Force offline mode for model loading
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_CACHE'] = os.path.join('models', 'cache')
os.environ['HF_HOME'] = os.path.join('models', 'hub')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join('models', 'sentence_transformers')


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserPostsFeatureSchema(BaseModel):
    """Pydantic schema for UserPosts feature extraction (matches feature_plan.md exactly)"""
    
    # Metadata
    setup_id: str
    extraction_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    llm_model: str = "gpt-4o-mini"
    
    # Non-LLM features (computed directly)
    avg_sentiment: float = Field(ge=-1.0, le=1.0)  # Statistical sentiment (VADER)
    post_count: int = Field(ge=0)  # Record count
    
    # LLM-extracted features
    community_sentiment_score: float = Field(ge=-1.0, le=1.0)
    bull_bear_ratio: float = Field(ge=0.0)
    rumor_intensity: float = Field(ge=0.0, le=1.0)
    trusted_user_sentiment: float = Field(ge=-1.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    engagement_score: float = Field(ge=0.0, le=1.0)
    unique_users: int = Field(ge=0)
    contrarian_signal: bool
    consensus_level: str = Field(pattern="^(high|medium|low)$")
    recent_sentiment_shift: str = Field(pattern="^(up|down|stable|unknown)$")
    coherence: str = Field(pattern="^(high|medium|low)$")
    
    # List fields - must use allowed values from feature_plan.md
    consensus_topics: List[str] = Field(default_factory=list)
    controversial_topics: List[str] = Field(default_factory=list)
    
    # Structured field
    sentiment_distribution: Dict[str, int] = Field(default_factory=dict)
    
    # Text summaries
    synthetic_post: str = Field(max_length=240)  # Dashboard summary
    cot_explanation: str  # Chain of thought explanation
    
    def model_post_init(self, __context):
        """Validate topic categories after model initialization"""
        allowed_topics = {"earnings", "dividends", "guidance", "macro", "product", "rumor", "other"}
        
        # Validate consensus_topics
        for topic in self.consensus_topics:
            if topic not in allowed_topics:
                raise ValueError(f"Topic '{topic}' not in allowed values: {allowed_topics}")
        
        # Validate controversial_topics  
        for topic in self.controversial_topics:
            if topic not in allowed_topics:
                raise ValueError(f"Topic '{topic}' not in allowed values: {allowed_topics}")

class EnhancedUserPostsAgentComplete:
    """Enhanced UserPosts Agent with complete retrieval pattern"""
    
    def __init__(
        self,
        db_path: str = "../data/sentiment_system.duckdb",
        lancedb_dir: str = "../lancedb_store",
        table_name: str = "userposts_embeddings",
        llm_model: str = "gpt-4o-mini",
        max_group_size: int = 20,  # Max posts per LLM call before chunking
        mode: str = "training",  # Either "training" or "prediction"
        use_cached_models: bool = False,
        local_files_only: bool = False
    ):
        """
        Initialize the Enhanced UserPosts Agent
        
        Args:
            db_path: Path to DuckDB database
            lancedb_dir: Path to LanceDB directory
            table_name: Name of the embedding table
            llm_model: LLM model to use
            max_group_size: Maximum number of posts per LLM call before chunking
            mode: Either "training" or "prediction"
            use_cached_models: Whether to use cached models
            local_files_only: Whether to only use local files
        """
        self.db_path = db_path
        self.lancedb_dir = lancedb_dir
        self.table_name = table_name
        self.llm_model = llm_model
        self.max_group_size = max_group_size
        self.mode = mode
        self.use_cached_models = use_cached_models
        self.local_files_only = local_files_only
        
        if self.use_cached_models:
            logger.info('Using cached models (offline mode)')
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
        
        # Initialize setup validator
        self.setup_validator = SetupValidatorDuckDB(db_path)
        
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Connect to LanceDB
        self._connect_to_lancedb()
        
        # Initialize DuckDB feature storage
        self._init_duckdb_feature_storage()
        
        logger.info(f"Enhanced UserPosts Agent initialized in {mode} mode")
    
    def check_setup_has_posts(self, setup_id: str) -> bool:
        """Check if a setup has any user posts"""
        import duckdb
        
        try:
            conn = duckdb.connect(self.db_path)
            
            # Query to check for posts
            query = """
                SELECT COUNT(*) as post_count
                FROM user_posts up
                WHERE up.setup_id = ?
            """
            
            result = conn.execute(query, [setup_id]).fetchone()
            conn.close()
            
            post_count = result[0] if result else 0
            return post_count > 0
            
        except Exception as e:
            logger.error(f"Error checking posts for setup_id '{setup_id}': {e}")
            return False
    
    def _connect_to_lancedb(self):
        """Connect to LanceDB table"""
        try:
            # Use absolute path for LanceDB connection
            abs_lancedb_dir = os.path.abspath(self.lancedb_dir)
            logger.info(f"Connecting to LanceDB at absolute path: {abs_lancedb_dir}")
            db = lancedb.connect(abs_lancedb_dir)
            
            # In prediction mode, we don't need to open any tables
            if hasattr(self, 'mode') and self.mode == 'prediction':
                logger.info("Prediction mode: Skipping table opening")
                self.table = None
                self.training_table = None
                return
                
            # Try to open existing table or note that it needs to be created
            try:
                self.table = db.open_table(self.table_name)
                logger.info(f"Connected to existing LanceDB table: {self.table_name}")
            except Exception as e:
                logger.info(f"LanceDB table {self.table_name} will be created during embedding process")
                self.table = None
                
            # Initialize training table for similarity search
            self.training_table = None
            try:
                # List available tables
                tables = db.table_names()
                logger.info(f"Available tables: {tables}")
                
                # Try to open with new table name first, then fall back to old name
                try:
                    if "userposts_embeddings_training" in tables:
                        self.training_table = db.open_table("userposts_embeddings_training")
                        logger.info("Connected to userposts_embeddings_training table for similarity search")
                    elif "userposts_embeddings" in tables:
                        # Fall back to old table name for backward compatibility
                        self.training_table = db.open_table("userposts_embeddings")
                        logger.info("Connected to userposts_embeddings table for similarity search (legacy name)")
                    else:
                        logger.warning("No userposts embeddings table found in available tables")
                except Exception as e:
                    logger.warning(f"Could not open training embeddings table: {e}")
            except Exception as e:
                logger.warning(f"Could not open training embeddings table: {e}")
        except Exception as e:
            logger.error(f"Error connecting to LanceDB: {e}")
            # In prediction mode, we can continue without LanceDB
            if hasattr(self, 'mode') and self.mode == 'prediction':
                logger.warning("Continuing in prediction mode without LanceDB connection")
                self.db = None
                self.table = None
                self.training_table = None
            else:
                raise
    
    def _init_duckdb_feature_storage(self):
        """Initialize DuckDB tables for feature storage"""
        import duckdb
        
        conn = duckdb.connect(self.db_path)
        
        # Create userposts_features table matching the schema
        conn.execute('''
            CREATE TABLE IF NOT EXISTS userposts_features (
                setup_id TEXT PRIMARY KEY,
                -- Non-LLM features (computed)
                avg_sentiment REAL DEFAULT 0.0,
                post_count INTEGER DEFAULT 0,
                -- LLM features
                community_sentiment_score REAL DEFAULT 0.0,
                bull_bear_ratio REAL DEFAULT 0.0,
                rumor_intensity REAL DEFAULT 0.0,
                trusted_user_sentiment REAL DEFAULT 0.0,
                relevance_score REAL DEFAULT 0.0,
                engagement_score REAL DEFAULT 0.0,
                unique_users INTEGER DEFAULT 0,
                contrarian_signal BOOLEAN DEFAULT FALSE,
                consensus_level TEXT DEFAULT 'medium',
                recent_sentiment_shift TEXT DEFAULT 'unknown',
                coherence TEXT DEFAULT 'medium',
                consensus_topics TEXT DEFAULT '[]',  -- JSON array
                controversial_topics TEXT DEFAULT '[]',  -- JSON array
                sentiment_distribution TEXT DEFAULT '{}',  -- JSON object
                synthetic_post TEXT DEFAULT '',
                cot_explanation TEXT DEFAULT '',
                -- Metadata
                extraction_timestamp TIMESTAMP,
                llm_model TEXT,
                CONSTRAINT chk_consensus_level CHECK (consensus_level IN ('high', 'medium', 'low')),
                CONSTRAINT chk_sentiment_shift CHECK (recent_sentiment_shift IN ('up', 'down', 'stable', 'unknown')),
                CONSTRAINT chk_coherence CHECK (coherence IN ('high', 'medium', 'low'))
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("DuckDB userposts feature storage tables initialized")
    
    def get_posts_for_setup(self, setup_id: str) -> pd.DataFrame:
        """Retrieve all posts for a given setup_id from DuckDB"""
        import duckdb
        
        try:
            conn = duckdb.connect(self.db_path)
            
            # Query without sentiment_score since it doesn't exist in the table
            query = """
                SELECT 
                    up.post_id,
                    up.setup_id,
                    up.ticker,
                    up.post_date,
                    up.user_handle,
                    up.post_content,
                    up.scraping_timestamp,
                    up.post_url
                FROM user_posts up
                WHERE up.setup_id = ?
                ORDER BY up.post_date DESC
            """
            
            results = conn.execute(query, [setup_id]).fetchdf()
            conn.close()
            
            if len(results) == 0:
                logger.warning(f"No posts found for setup_id: {setup_id}")
                return pd.DataFrame()
            
            logger.info(f"Retrieved {len(results)} posts for setup_id: {setup_id} from DuckDB source")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving posts for setup_id '{setup_id}': {e}")
            return pd.DataFrame()
    
    def _compute_non_llm_features(self, posts_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute non-LLM features as specified in feature_plan.md"""
        computed_features = {}
        
        # avg_sentiment: Compute from post_content using simple heuristics
        # Since sentiment_score column doesn't exist, compute basic sentiment
        if 'post_content' in posts_df.columns and len(posts_df) > 0:
            # Simple sentiment computation based on positive/negative keywords
            sentiment_scores = []
            for content in posts_df['post_content']:
                if pd.isna(content):
                    sentiment_scores.append(0.0)
                    continue
                    
                content_lower = str(content).lower()
                
                # Simple keyword-based sentiment
                positive_words = ['good', 'great', 'excellent', 'positive', 'buy', 'bullish', 'up', 'strong', 'growth', 'profit']
                negative_words = ['bad', 'terrible', 'negative', 'sell', 'bearish', 'down', 'weak', 'loss', 'decline', 'poor']
                
                pos_count = sum(1 for word in positive_words if word in content_lower)
                neg_count = sum(1 for word in negative_words if word in content_lower)
                
                if pos_count + neg_count > 0:
                    sentiment = (pos_count - neg_count) / (pos_count + neg_count)
                else:
                    sentiment = 0.0
                    
                sentiment_scores.append(max(-1.0, min(1.0, sentiment)))  # Clamp to [-1, 1]
            
            computed_features['avg_sentiment'] = float(sum(sentiment_scores) / len(sentiment_scores)) if sentiment_scores else 0.0
        else:
            computed_features['avg_sentiment'] = 0.0
        
        # post_count: Record count
        computed_features['post_count'] = len(posts_df)
        
        return computed_features
    
    def extract_features_with_llm(self, setup_id: str, posts_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract LLM features from posts using chunking if needed"""
        if len(posts_df) == 0:
            return self._get_default_llm_features()
        
        # Handle large groups by chunking
        if len(posts_df) > self.max_group_size:
            return self._extract_large_group_features(posts_df)
        else:
            return self._extract_single_group_features(posts_df)
    
    def _extract_single_group_features(self, posts_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract features for a group that fits in one LLM call"""
        
        # Prepare context for LLM
        context = self._prepare_posts_context(posts_df)
        
        # Load prompt template
        prompt = self._load_prompt_template(context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1200
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            # Extract JSON from response
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = response_text[json_start:json_end]
                    features = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in LLM response")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.error(f"Raw response: {response_text}")
                return self._get_default_llm_features()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting LLM features: {e}")
            return self._get_default_llm_features()
    
    def _extract_large_group_features(self, posts_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract features for large groups using chunking and aggregation"""
        
        # Split into chunks
        chunks = [posts_df.iloc[i:i + self.max_group_size] 
                 for i in range(0, len(posts_df), self.max_group_size)]
        
        chunk_features = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing UserPosts chunk {i+1}/{len(chunks)} ({len(chunk)} posts)")
            features = self._extract_single_group_features(chunk)
            chunk_features.append(features)
        
        # Aggregate chunk features
        return self._aggregate_chunk_features(chunk_features, len(posts_df))
    
    def _prepare_posts_context(self, posts_df: pd.DataFrame) -> str:
        """Prepare posts as context for LLM"""
        context_parts = []
        
        for i, (_, row) in enumerate(posts_df.iterrows(), 1):
            date_str = row.get('post_date', 'Unknown date')
            user = row.get('user_handle', 'Unknown user')
            content = row['post_content'][:300]  # Limit content length
            
            context_parts.append(f"{i}. [{date_str}] @{user}")
            context_parts.append(f"   {content}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _load_prompt_template(self, context: str) -> str:
        """Load and format the prompt template for UserPosts feature extraction"""
        
        # Try to load from file first
        prompt_path = Path("features/llm_prompts/llm_prompt_userposts.md")
        
        if prompt_path.exists():
            template = prompt_path.read_text()
            return template.replace("{insert top 2–20 user post excerpts here}", context)
        
        # Fallback embedded prompt
        template = """You are analyzing user posts for trading insights. Extract these features as JSON:

POSTS TO ANALYZE:
{context}

Extract these features and return as JSON:
{{
  "community_sentiment_score": <float between -1.0 and 1.0>,
  "bull_bear_ratio": <float >= 0.0, ratio of bullish to bearish posts>,
  "rumor_intensity": <float between 0.0 and 1.0>,
  "trusted_user_sentiment": <float between -1.0 and 1.0>,
  "relevance_score": <float between 0.0 and 1.0>,
  "engagement_score": <float between 0.0 and 1.0>,
  "unique_users": <integer >= 0>,
  "contrarian_signal": <true or false>,
  "consensus_level": <"high", "medium", or "low">,
  "recent_sentiment_shift": <"up", "down", "stable", or "unknown">,
  "coherence": <"high", "medium", or "low">,
  "consensus_topics": [list of strings from: "earnings", "dividends", "guidance", "macro", "product", "rumor", "other"],
  "controversial_topics": [list of strings from: "earnings", "dividends", "guidance", "macro", "product", "rumor", "other"],
  "sentiment_distribution": {{"bullish": <int>, "bearish": <int>, "neutral": <int>}},
  "synthetic_post": "<summary in 240 chars or less>",
  "cot_explanation": "<short reasoning for your analysis>"
}}

Return ONLY the JSON object."""
        
        return template.replace("{context}", context)
    
    def _get_default_llm_features(self) -> Dict[str, Any]:
        """Return default LLM features for empty or failed extractions"""
        return {
            'community_sentiment_score': 0.0,
            'bull_bear_ratio': 0.0,
            'rumor_intensity': 0.0,
            'trusted_user_sentiment': 0.0,
            'relevance_score': 0.0,
            'engagement_score': 0.0,
            'unique_users': 0,
            'contrarian_signal': False,
            'consensus_level': 'low',  # Changed from 'medium'
            'recent_sentiment_shift': 'unknown',
            'coherence': 'low',  # Changed from 'medium'
            'consensus_topics': [],
            'controversial_topics': [],
            'sentiment_distribution': {'bullish': 0, 'bearish': 0, 'neutral': 0},
            'synthetic_post': 'No posts available for analysis',
            'cot_explanation': 'No user posts available for analysis, using default features.'  # Added
        }
    
    def _aggregate_chunk_features(self, chunk_features: List[Dict], total_posts: int) -> Dict[str, Any]:
        """Aggregate features from multiple chunks"""
        if not chunk_features:
            return self._get_default_llm_features()
        
        # For numeric features, take weighted average
        aggregated = {}
        
        # Numeric aggregation
        numeric_fields = ['community_sentiment_score', 'bull_bear_ratio', 'rumor_intensity', 
                         'trusted_user_sentiment', 'relevance_score', 'engagement_score']
        
        for field in numeric_fields:
            values = [chunk.get(field, 0.0) for chunk in chunk_features]
            aggregated[field] = sum(values) / len(values)
        
        # Integer sum
        aggregated['unique_users'] = sum(chunk.get('unique_users', 0) for chunk in chunk_features)
        
        # Boolean - true if any chunk is true
        aggregated['contrarian_signal'] = any(chunk.get('contrarian_signal', False) for chunk in chunk_features)
        
        # Categorical - take most common
        consensus_levels = [chunk.get('consensus_level', 'medium') for chunk in chunk_features]
        aggregated['consensus_level'] = max(set(consensus_levels), key=consensus_levels.count)
        
        shifts = [chunk.get('recent_sentiment_shift', 'unknown') for chunk in chunk_features]
        aggregated['recent_sentiment_shift'] = max(set(shifts), key=shifts.count)
        
        coherence_levels = [chunk.get('coherence', 'medium') for chunk in chunk_features]
        aggregated['coherence'] = max(set(coherence_levels), key=coherence_levels.count)
        
        # Lists - combine and deduplicate
        all_consensus_topics = []
        all_controversial_topics = []
        
        for chunk in chunk_features:
            all_consensus_topics.extend(chunk.get('consensus_topics', []))
            all_controversial_topics.extend(chunk.get('controversial_topics', []))
        
        aggregated['consensus_topics'] = list(set(all_consensus_topics))
        aggregated['controversial_topics'] = list(set(all_controversial_topics))
        
        # Sentiment distribution - sum counts
        total_distribution = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        for chunk in chunk_features:
            dist = chunk.get('sentiment_distribution', {})
            for key in total_distribution:
                total_distribution[key] += dist.get(key, 0)
        aggregated['sentiment_distribution'] = total_distribution
        
        # Text - combine summaries
        summaries = [chunk.get('synthetic_post', '') for chunk in chunk_features if chunk.get('synthetic_post')]
        if summaries:
            combined_summary = '; '.join(summaries)[:240]
            aggregated['synthetic_post'] = combined_summary
        else:
            aggregated['synthetic_post'] = f'Analyzed {total_posts} posts across {len(chunk_features)} chunks'
        
        explanations = [chunk.get('cot_explanation', '') for chunk in chunk_features if chunk.get('cot_explanation')]
        if explanations:
            aggregated['cot_explanation'] = f"Aggregated from {len(chunk_features)} chunks: " + '; '.join(explanations)[:200]
        else:
            aggregated['cot_explanation'] = f'Features aggregated from {len(chunk_features)} chunks'
        
        return aggregated
    
    def store_features(self, features: UserPostsFeatureSchema):
        """Store features in DuckDB"""
        import duckdb
        
        conn = duckdb.connect(self.db_path)
        
        try:
            # Convert lists and dicts to JSON
            consensus_topics_json = json.dumps(features.consensus_topics)
            controversial_topics_json = json.dumps(features.controversial_topics)
            sentiment_distribution_json = json.dumps(features.sentiment_distribution)
            
            conn.execute('''
                INSERT OR REPLACE INTO userposts_features (
                    setup_id, avg_sentiment, post_count,
                    community_sentiment_score, bull_bear_ratio, rumor_intensity,
                    trusted_user_sentiment, relevance_score, engagement_score,
                    unique_users, contrarian_signal, consensus_level,
                    recent_sentiment_shift, coherence, consensus_topics,
                    controversial_topics, sentiment_distribution,
                    synthetic_post, cot_explanation, extraction_timestamp, llm_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                features.setup_id,
                features.avg_sentiment,
                features.post_count,
                features.community_sentiment_score,
                features.bull_bear_ratio,
                features.rumor_intensity,
                features.trusted_user_sentiment,
                features.relevance_score,
                features.engagement_score,
                features.unique_users,
                features.contrarian_signal,
                features.consensus_level,
                features.recent_sentiment_shift,
                features.coherence,
                consensus_topics_json,
                controversial_topics_json,
                sentiment_distribution_json,
                features.synthetic_post,
                features.cot_explanation,
                features.extraction_timestamp,
                features.llm_model
            ))
            
            conn.commit()
            logger.info(f"Stored features for setup_id: {features.setup_id} in DuckDB")
            
        except Exception as e:
            logger.error(f"Error storing features in DuckDB: {e}")
            raise
        finally:
            conn.close()
    
    def process_setup(self, setup_id: str, mode: str = None) -> Optional[UserPostsFeatureSchema]:
        """
        Process a single setup ID to extract and store features
        
        Args:
            setup_id: Setup ID to process
            mode: Either 'training' or 'prediction' (overrides self.mode if provided)
            
        Returns:
            Extracted features or None if processing failed
        """
        # Use provided mode or fall back to instance mode
        current_mode = mode if mode is not None else self.mode
        logger.info(f"Processing user posts for setup: {setup_id} in {current_mode} mode")
        
        try:
            # Check if setup has any posts
            if not self.check_setup_has_posts(setup_id):
                logger.warning(f"No user posts found for setup {setup_id}")
                # Create empty features with default values
                default_features = self._get_default_llm_features()
                default_features['setup_id'] = setup_id
                default_features['post_count'] = 0
                default_features['avg_sentiment'] = 0.0
                default_features['extraction_timestamp'] = datetime.now().isoformat()
                default_features['llm_model'] = self.llm_model
                
                try:
                    features = UserPostsFeatureSchema(**default_features)
                    self.store_features(features)
                    return features
                except ValidationError as e:
                    logger.error(f"Validation error for default features: {e}")
                    return None
            
            # Get all posts for this setup
            posts_df = self.get_posts_for_setup(setup_id)
            
            if len(posts_df) == 0:
                logger.warning(f"No user posts retrieved for setup {setup_id}")
                return None
            
            # Step 1: Calculate non-LLM features
            non_llm_features = self._compute_non_llm_features(posts_df)
            
            # Step 2: Extract LLM features
            llm_features = self.extract_features_with_llm(setup_id, posts_df)
            
            # Step 3: Combine features
            combined_features = {
                'setup_id': setup_id,
                'extraction_timestamp': datetime.now().isoformat(),
                'llm_model': self.llm_model,
                **non_llm_features,
                **llm_features
            }
            
            # Step 4: Validate and create feature schema
            try:
                features = UserPostsFeatureSchema(**combined_features)
                
                # Step 5: Store features
                self.store_features(features)
                
                # Step 6: In prediction mode, generate similarity-based predictions
                if current_mode == 'prediction':
                    # Create content for similarity search
                    user_posts_summary = f"User posts summary: {features.synthetic_post} | " + \
                                       f"Sentiment: {features.community_sentiment_score} | " + \
                                       f"Bull/Bear: {features.bull_bear_ratio} | " + \
                                       f"Topics: {', '.join(features.consensus_topics)}"
                    
                    # Generate prediction
                    logger.info(f"Generating similarity-based prediction for {setup_id}")
                    prediction = self.predict_via_similarity(user_posts_summary)
                    if prediction:
                        prediction['setup_id'] = setup_id
                        prediction['domain'] = 'userposts'
                        prediction['prediction_timestamp'] = datetime.now().isoformat()
                        logger.info(f"Generated similarity prediction for {setup_id}: {prediction.get('predicted_outperformance', 'N/A')}")
                        return features, prediction
                
                return features
                
            except ValidationError as e:
                logger.error(f"Validation error for setup {setup_id}: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing user posts for setup {setup_id}: {e}")
            return None
    
    def batch_process_setups(self, setup_ids: List[str]) -> Dict[str, Optional[UserPostsFeatureSchema]]:
        """Process multiple setups in batch"""
        results = {}
        successful = 0
        failed = 0
        
        for i, setup_id in enumerate(setup_ids, 1):
            logger.info(f"Processing setup {i}/{len(setup_ids)}: {setup_id}")
            
            try:
                features = self.process_setup(setup_id)
                results[setup_id] = features
                
                if features is not None:
                    successful += 1
                else:
                    failed += 1
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing setup {setup_id}: {e}")
                results[setup_id] = None
                failed += 1
        
        logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
        return results
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Close any open connections if needed
            pass
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        logger.info("Enhanced UserPosts Agent (Complete) cleaned up successfully")

    def find_similar_training_embeddings(self, query: Union[np.ndarray, str], limit: int = 10) -> List[Dict]:
        """
        Find similar training embeddings with labels
        
        Args:
            query: Either a numpy array embedding or text content to embed
            limit: Maximum number of similar cases to return
            
        Returns:
            List of similar cases with their metadata and labels
        """
        if not hasattr(self, 'training_table') or self.training_table is None:
            logger.warning("Training embeddings table not available, attempting to connect")
            # Try to open the table directly
            try:
                import lancedb
                import os
                
                # Use absolute path
                abs_lancedb_dir = os.path.abspath(self.lancedb_dir)
                logger.info(f"Connecting to LanceDB at absolute path: {abs_lancedb_dir}")
                db = lancedb.connect(abs_lancedb_dir)
                
                # List available tables
                tables = db.table_names()
                logger.info(f"Available tables: {tables}")
                
                # Try both table names
                for table_name in ['userposts_embeddings_training', 'userposts_embeddings']:
                    if table_name in tables:
                        try:
                            self.training_table = db.open_table(table_name)
                            logger.info(f"Successfully opened {table_name} for similarity search")
                            break
                        except Exception as e:
                            logger.warning(f"Could not open {table_name}: {e}")
                
                if self.training_table is None:
                    logger.warning("No suitable training table found")
                    return []
            except Exception as e:
                logger.error(f"Error connecting to LanceDB: {e}")
                return []
        
        try:
            # Handle text input by creating embedding
            if isinstance(query, str):
                logger.info(f"Creating embedding for text: {query[:100]}...")
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder="models/sentence_transformers", local_files_only=True)
                query = model.encode(query)
                logger.info(f"Created embedding with shape: {query.shape}")
            
            # Ensure numpy array
            if not isinstance(query, np.ndarray):
                query = np.array(query)
            
            logger.info(f"Searching for similar embeddings with limit: {limit}")
            
            # Search for similar cases
            results = self.training_table.search(query).limit(limit).to_pandas()
            logger.info(f"Found {len(results)} similar cases")
            
            # Check if results contain labels
            if 'outperformance_10d' in results.columns:
                avg_outperformance = results['outperformance_10d'].mean()
                logger.info(f"Average outperformance of similar cases: {avg_outperformance}")
            else:
                logger.warning("Similar cases do not contain outperformance_10d labels")
                
            return results.to_dict('records')
        except Exception as e:
            logger.error(f"Error searching similar embeddings: {e}")
            return []
    
    def compute_similarity_features(self, similar_cases: List[Dict]) -> Dict[str, float]:
        """
        Compute similarity-based features from similar cases
        
        Args:
            similar_cases: List of similar cases with their metadata and labels
            
        Returns:
            Dictionary of similarity-based features
        """
        if not similar_cases:
            return {
                'positive_signal_strength': 0.0,
                'negative_risk_score': 0.0,
                'neutral_probability': 0.0,
                'historical_pattern_confidence': 0.0,
                'similar_cases_count': 0
            }
        
        # Use label converter for classification
        label_converter = LabelConverter()
        performance_values = [c.get('outperformance_10d', 0) for c in similar_cases]
        
        # Calculate performance-based features
        positive_cases = [c for c in similar_cases if label_converter.outperformance_to_class_int(c.get('outperformance_10d', 0)) == 1]
        negative_cases = [c for c in similar_cases if label_converter.outperformance_to_class_int(c.get('outperformance_10d', 0)) == -1]
        neutral_cases = [c for c in similar_cases if label_converter.outperformance_to_class_int(c.get('outperformance_10d', 0)) == 0]
        
        total_cases = len(similar_cases)
        
        # Calculate weighted features based on similarity scores
        similarity_scores = [c.get('_distance', 1.0) for c in similar_cases]  # Lower distance = higher similarity
        max_distance = max(similarity_scores) if similarity_scores else 1.0
        similarity_weights = [1 - (score / max_distance) for score in similarity_scores]
        
        # Calculate weighted ratios (with safety check)
        sum_weights = sum(similarity_weights) if similarity_weights else 1.0
        if sum_weights == 0:
            sum_weights = 1.0
            
        # Use label converter thresholds
        positive_ratio = sum(w for c, w in zip(similar_cases, similarity_weights) 
                           if label_converter.outperformance_to_class_int(c.get('outperformance_10d', 0)) == 1) / sum_weights
        
        negative_ratio = sum(w for c, w in zip(similar_cases, similarity_weights)
                           if label_converter.outperformance_to_class_int(c.get('outperformance_10d', 0)) == -1) / sum_weights
        
        # Calculate pattern confidence based on consistency
        performance_std = np.std(performance_values) if performance_values else 1.0
        pattern_confidence = 1.0 / (1.0 + performance_std)  # Higher std = lower confidence
        
        return {
            'positive_signal_strength': float(positive_ratio),
            'negative_risk_score': float(negative_ratio),
            'neutral_probability': float(len(neutral_cases) / total_cases),
            'historical_pattern_confidence': float(pattern_confidence),
            'similar_cases_count': total_cases
        }
    
    def predict_via_similarity(self, query: Union[np.ndarray, str]) -> Dict[str, Any]:
        """
        Direct prediction using similarity to training embeddings
        
        Args:
            query: Either a numpy array embedding or text content to embed
            
        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Starting similarity-based prediction for query: {query[:100]}...")
        
        # Find similar cases
        similar_cases = self.find_similar_training_embeddings(query, limit=10)
        
        if not similar_cases:
            logger.warning("No similar cases found for prediction")
            return {}
        
        # Use label converter
        label_converter = LabelConverter()
            
        # Extract labels from similar cases
        similar_labels = [case.get('outperformance_10d', 0.0) for case in similar_cases]
        logger.info(f"Similar labels: {similar_labels}")
        
        # Compute weighted average prediction
        similarity_scores = [1 - case.get('_distance', 1.0) for case in similar_cases]
        logger.info(f"Similarity scores: {similarity_scores}")
        
        sum_scores = sum(similarity_scores) if similarity_scores else 1.0
        if sum_scores == 0:
            sum_scores = 1.0
            
        weighted_prediction = sum(label * score for label, score in zip(similar_labels, similarity_scores)) / sum_scores
        logger.info(f"Weighted prediction: {weighted_prediction}")
        
        # Get class distribution using the label converter
        class_distribution = get_class_distribution(similar_labels)
        
        # Compute confidence metrics
        prediction = {
            'predicted_outperformance': float(weighted_prediction),
            'confidence': float(1.0 / (1.0 + np.std(similar_labels))),
            'positive_ratio': class_distribution['positive'],
            'negative_ratio': class_distribution['negative'],
            'neutral_ratio': class_distribution['neutral'],
            'similar_cases_count': len(similar_cases)
        }
        
        logger.info(f"Final prediction: {prediction}")
        return prediction

def main():
    """Main function for testing"""
    import sys
    from pathlib import Path
    
    # Add project root to path for proper imports
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from tools.setup_validator_duckdb import SetupValidatorDuckDB
    
    # Initialize agent
    agent = EnhancedUserPostsAgentComplete()
    
    # Load confirmed setups
    validator = SetupValidatorDuckDB()
    confirmed_setups = validator.get_confirmed_setup_ids()
    
    logger.info(f"Enhanced UserPosts Agent (Complete) initialized successfully")
    logger.info(f"Found {len(confirmed_setups)} confirmed setups")
    
    # Test with a few setups that we know have posts
    test_setups = [setup for setup in confirmed_setups if setup.startswith(('HWDN', 'KZG', 'BLND'))][:3]
    
    logger.info(f"Testing with {len(test_setups)} setups...")
    
    for setup_id in test_setups:
        print(f"\nProcessing {setup_id}...")
        features = agent.process_setup(setup_id)
        if features:
            print(f"  ✅ Success: {features.post_count} posts, sentiment: {features.avg_sentiment:.3f}")
        else:
            print(f"  ❌ Failed")
    
    agent.cleanup()

if __name__ == "__main__":
    main() 