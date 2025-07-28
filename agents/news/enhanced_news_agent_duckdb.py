#!/usr/bin/env python3
"""
Enhanced News Agent with DuckDB Integration

This agent processes news data from DuckDB and extracts features using LLMs.
It also supports similarity-based prediction using embeddings.
"""

# Force offline mode for model loading
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_CACHE'] = os.path.join('models', 'cache')
os.environ['HF_HOME'] = os.path.join('models', 'hub')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join('models', 'sentence_transformers')

import sys
import logging
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from datetime import datetime, timedelta
import json
import re
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
import duckdb
import lancedb
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import difflib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from agents.news.news_categories import (
    FINANCIAL_RESULTS_PATTERNS,
    CORPORATE_ACTIONS_PATTERNS,
    GOVERNANCE_PATTERNS,
    CORPORATE_EVENTS_PATTERNS,
    OTHER_SIGNALS_PATTERNS,
    CATEGORY_TO_GROUP
)

# Import setup validator
try:
    from tools.setup_validator_duckdb import SetupValidatorDuckDB
except ImportError:
    logger.warning("SetupValidatorDuckDB not found, using local implementation")
    
    class SetupValidatorDuckDB:
        """Dummy implementation of SetupValidatorDuckDB"""
        def __init__(self, db_path):
            self.db_path = db_path
            self.conn = duckdb.connect(db_path)
        
        def is_valid_setup(self, setup_id):
            return True
        
        def close(self):
            self.conn.close()

# Constants
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_HEADLINE_LENGTH = 150
MAX_CONTENT_LENGTH = 5000
MIN_CONTENT_LENGTH = 10
MAX_ITEMS_PER_GROUP = 10
SIMILARITY_THRESHOLD = 0.7
DEFAULT_SIMILARITY_LIMIT = 10

class NewsFeatureSchema(BaseModel):
    """Pydantic schema for News feature extraction (matches feature_plan.md exactly)"""
    
    # Metadata
    setup_id: str
    extraction_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    llm_model: str = "gpt-4o-mini"
    
    # Financial Results group (7 features)
    count_financial_results: int = Field(ge=0)
    max_severity_financial_results: float = Field(ge=0.0, le=1.0)
    avg_headline_spin_financial_results: str = Field(pattern="^(positive|negative|neutral|uncertain)$")
    sentiment_score_financial_results: float = Field(ge=-1.0, le=1.0)
    profit_warning_present: bool
    synthetic_summary_financial_results: str = Field(max_length=240)
    cot_explanation_financial_results: str
    
    # Corporate Actions group (7 features)
    count_corporate_actions: int = Field(ge=0)
    max_severity_corporate_actions: float = Field(ge=0.0, le=1.0)
    avg_headline_spin_corporate_actions: str = Field(pattern="^(positive|negative|neutral|uncertain)$")
    sentiment_score_corporate_actions: float = Field(ge=-1.0, le=1.0)
    capital_raise_present: bool
    synthetic_summary_corporate_actions: str = Field(max_length=240)
    cot_explanation_corporate_actions: str
    
    # Governance group (7 features)
    count_governance: int = Field(ge=0)
    max_severity_governance: float = Field(ge=0.0, le=1.0)
    avg_headline_spin_governance: str = Field(pattern="^(positive|negative|neutral|uncertain)$")
    sentiment_score_governance: float = Field(ge=-1.0, le=1.0)
    board_change_present: bool
    synthetic_summary_governance: str = Field(max_length=240)
    cot_explanation_governance: str
    
    # Corporate Events group (8 features)
    count_corporate_events: int = Field(ge=0)
    max_severity_corporate_events: float = Field(ge=0.0, le=1.0)
    avg_headline_spin_corporate_events: str = Field(pattern="^(positive|negative|neutral|uncertain)$")
    sentiment_score_corporate_events: float = Field(ge=-1.0, le=1.0)
    contract_award_present: bool
    merger_or_acquisition_present: bool
    synthetic_summary_corporate_events: str = Field(max_length=240)
    cot_explanation_corporate_events: str
    
    # Other Signals group (8 features)
    count_other_signals: int = Field(ge=0)
    max_severity_other_signals: float = Field(ge=0.0, le=1.0)
    avg_headline_spin_other_signals: str = Field(pattern="^(positive|negative|neutral|uncertain)$")
    sentiment_score_other_signals: float = Field(ge=-1.0, le=1.0)
    broker_recommendation_present: bool
    credit_rating_change_present: bool
    synthetic_summary_other_signals: str = Field(max_length=240)
    cot_explanation_other_signals: str
    
    # Global explanation
    cot_explanation_news_grouped: str

class EnhancedNewsAgentDuckDB:
    """Enhanced News Agent for DuckDB with embedding-based similarity search"""
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        lancedb_dir: str = "lancedb_store", 
        table_name: str = "news_embeddings",
        llm_model: str = "gpt-4o-mini",
        max_group_size: int = 10,  # Max items per group before chunking
        mode: str = "training",     # Either "training" or "prediction"
        use_cached_models: bool = False,
        local_files_only: bool = False
    ):
        """
        Initialize the Enhanced News Agent
        
        Args:
            db_path: Path to DuckDB database
            lancedb_dir: Path to LanceDB directory
            table_name: Name of the embedding table
            llm_model: LLM model to use
            max_group_size: Maximum number of items per group before chunking
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
        
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Initialize category classifier
        self.classifier = CategoryClassifier(self.client, use_cached_models=self.use_cached_models)
        
        # Initialize embedding model
        model_kwargs = {}
        if self.use_cached_models or self.local_files_only:
            model_kwargs['cache_folder'] = os.path.join('models', 'sentence_transformers')
            model_kwargs['local_files_only'] = True
            
        self.embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL, **model_kwargs)
        
        # Initialize setup validator
        self.setup_validator = SetupValidatorDuckDB(db_path)
        
        # Connect to LanceDB
        self._connect_to_lancedb()
        
        # Initialize DuckDB feature storage
        self._init_duckdb_feature_storage()
        
        # Determine table name based on mode
        if mode == "training":
            self.embedding_table_name = f"{table_name}_training"
        else:
            self.embedding_table_name = f"{table_name}_prediction"
            
        logger.info(f"Enhanced News Agent initialized in {mode} mode with table {self.embedding_table_name}")

class CategoryClassifier:
    """News headline classifier for categorization"""
    
    def __init__(self, openai_client: OpenAI, use_cached_models: bool = False):
        """Initialize the category classifier"""
        self.client = openai_client
        self.use_cached_models = use_cached_models
        self.classification_count = 0
        self.category_counts = {
            "financial_results": 0,
            "corporate_actions": 0,
            "governance": 0,
            "corporate_events": 0,
            "other_signals": 0
        }
        
    def classify_headline(self, headline: str) -> str:
        """
        Classify a headline into one of the predefined categories
        
        Args:
            headline: The headline to classify
            
        Returns:
            Category name (one of the predefined categories)
        """
        self.classification_count += 1
        
        # Try fast pattern matching first
        category = self._fast_pattern_match(headline)
        if category:
            self.category_counts[category] += 1
            return category
            
        # Try fuzzy matching next
        category = self._fuzzy_match(headline)
        if category:
            self.category_counts[category] += 1
            return category
            
        # Fall back to LLM classification
        category = self._llm_classify(headline)
        self.category_counts[category] += 1
        return category
    
    def _fast_pattern_match(self, headline: str) -> Optional[str]:
        """Fast pattern matching for common headline patterns"""
        headline_lower = headline.lower()
        
        # Direct keyword mapping
        patterns = {
            "final results": "Final Results",
            "interim results": "Interim Results", 
            "half year results": "Interim Results",
            "annual results": "Final Results",
            "profit warning": "Profit Warning",
            "trading update": "Trading Update",
            "trading statement": "Trading Statement",
            "dividend": "Dividend Declaration",
            "share buyback": "Share Buyback",
            "buyback programme": "Share Buyback",
            "placing": "Placing",
            "board changes": "Board Changes",
            "director dealing": "Director Dealings",
            "agm": "AGM Statement",
            "contract award": "Contract Award",
            "acquisition": "Merger & Acquisition",
            "merger": "Merger & Acquisition",
            "broker": "Broker Recommendation",
            "credit rating": "Credit Rating"
        }
        
        for pattern, category in patterns.items():
            if pattern in headline_lower:
                return category
        
        return None
    
    def _fuzzy_match(self, headline: str) -> Optional[str]:
        """Fuzzy matching against known categories"""
        # Use difflib to find close matches
        matches = difflib.get_close_matches(
            headline, RNS_CATEGORIES, n=1, cutoff=0.6
        )
        
        if matches:
            return matches[0]
        
        # Also try matching against parts of the headline
        words = headline.split()
        for word in words:
            if len(word) > 3:  # Skip short words
                matches = difflib.get_close_matches(
                    word, RNS_CATEGORIES, n=1, cutoff=0.8
                )
                if matches:
                    return matches[0]
        
        return None
    
    def _llm_classify(self, headline: str) -> str:
        """LLM fallback for ambiguous headlines"""
        prompt = f"""You are a financial news classifier.
Given the following headline, assign it to the single most appropriate category from this list:

{', '.join(RNS_CATEGORIES)}

Headline: "{headline}"

Respond with only the category name from the list above."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=50
            )
            
            category = response.choices[0].message.content.strip()
            
            # Validate the response is in our category list
            if category in RNS_CATEGORIES:
                return category
            else:
                # Try fuzzy match on LLM response
                matches = difflib.get_close_matches(category, RNS_CATEGORIES, n=1, cutoff=0.6)
                if matches:
                    return matches[0]
                else:
                    logger.warning(f"LLM returned invalid category '{category}' for headline: {headline}")
                    return "Investment Update"  # Default fallback
                    
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            return "Investment Update"  # Default fallback


# RNS Category to Group Mapping System
RNS_CATEGORIES = [
    "AGM Statement", "Admission of Securities", "Annual Financial Report", "Blocklisting",
    "Board Changes", "Broker Recommendation", "Capital Reorganisation", "Change of Adviser",
    "Change of Name", "Circular", "Company Secretary Change", "Contract Award",
    "Contract Termination", "Conversion of Securities", "Credit Rating", "Debt Issue",
    "Delisting", "Director Dealings", "Director/PDMR Shareholding", "Dividend Declaration",
    "Final Results", "Interim Results", "Investment Update", "Issue of Equity",
    "Launch of New Product", "Major Interest in Shares", "Merger & Acquisition",
    "Notice of AGM", "Placing", "Profit Warning", "Resignation", "Results of AGM",
    "Share Buyback", "Shareholder Meeting", "Trading Statement", "Trading Update"
]

CATEGORY_TO_GROUP = {
    # Financial Results
    "Final Results": "financial_results",
    "Interim Results": "financial_results", 
    "Profit Warning": "financial_results",
    "Trading Statement": "financial_results",
    "Trading Update": "financial_results",
    "Annual Financial Report": "financial_results",
    
    # Corporate Actions
    "Share Buyback": "corporate_actions",
    "Issue of Equity": "corporate_actions",
    "Dividend Declaration": "corporate_actions",
    "Capital Reorganisation": "corporate_actions",
    "Placing": "corporate_actions",
    "Debt Issue": "corporate_actions",
    "Admission of Securities": "corporate_actions",
    "Conversion of Securities": "corporate_actions",
    "Blocklisting": "corporate_actions",
    "Delisting": "corporate_actions",
    
    # Governance
    "Board Changes": "governance",
    "Director Dealings": "governance",
    "Director/PDMR Shareholding": "governance",
    "AGM Statement": "governance",
    "Results of AGM": "governance",
    "Notice of AGM": "governance",
    "Shareholder Meeting": "governance",
    "Company Secretary Change": "governance",
    "Resignation": "governance",
    "Major Interest in Shares": "governance",
    "Change of Name": "governance",
    "Change of Adviser": "governance",
    "Circular": "governance",
    
    # Corporate Events
    "Contract Award": "corporate_events",
    "Merger & Acquisition": "corporate_events",
    "Launch of New Product": "corporate_events",
    "Contract Termination": "corporate_events",
    
    # Other Signals
    "Broker Recommendation": "other_signals",
    "Credit Rating": "other_signals",
    "Investment Update": "other_signals"
}


class EnhancedNewsAgentDuckDB:
    """Enhanced News Agent for DuckDB with embedding-based similarity search"""
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        lancedb_dir: str = "lancedb_store", 
        table_name: str = "news_embeddings",
        llm_model: str = "gpt-4o-mini",
        max_group_size: int = 10,  # Max items per group before chunking
        mode: str = "training",     # Either "training" or "prediction"
        use_cached_models: bool = False,
        local_files_only: bool = False
    ):
        """
        Initialize the Enhanced News Agent
        
        Args:
            db_path: Path to DuckDB database
            lancedb_dir: Path to LanceDB directory
            table_name: Name of the embedding table
            llm_model: LLM model to use
            max_group_size: Maximum number of items per group before chunking
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
        
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Initialize category classifier
        self.classifier = CategoryClassifier(self.client, use_cached_models=self.use_cached_models)
        
        # Initialize embedding model
        model_kwargs = {}
        if self.use_cached_models or self.local_files_only:
            model_kwargs['cache_folder'] = os.path.join('models', 'sentence_transformers')
            model_kwargs['local_files_only'] = True
            
        self.embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL, **model_kwargs)
        
        # Initialize setup validator
        self.setup_validator = SetupValidatorDuckDB(db_path)
        
        # Connect to LanceDB
        self._connect_to_lancedb()
        
        # Initialize DuckDB feature storage
        self._init_duckdb_feature_storage()
        
        # Determine table name based on mode
        if mode == "training":
            self.embedding_table_name = f"{table_name}_training"
        else:
            self.embedding_table_name = f"{table_name}_prediction"
            
        logger.info(f"Enhanced News Agent initialized in {mode} mode with table {self.embedding_table_name}")

    def _connect_to_lancedb(self):
        """Connect to LanceDB"""
        try:
            # Use absolute path for LanceDB connection
            abs_lancedb_dir = os.path.abspath(self.lancedb_dir)
            logger.info(f"Connecting to LanceDB at absolute path: {abs_lancedb_dir}")
            
            lancedb_path = Path(abs_lancedb_dir)
            if not lancedb_path.exists():
                raise FileNotFoundError(f"LanceDB directory not found: {lancedb_path}")
            
            self.db = lancedb.connect(abs_lancedb_dir)
            
            # In prediction mode, we don't need to open any tables
            if hasattr(self, 'mode') and self.mode == 'prediction':
                logger.info("Prediction mode: Skipping table opening")
                self.table = None
                return
                
            # Try to open the table
            try:
                self.table = self.db.open_table(self.table_name)
                logger.info(f"Connected to LanceDB table: {self.table_name}")
            except Exception as e:
                logger.warning(f"Could not open table {self.table_name}: {e}")
                self.table = None
                
            # Initialize training table for similarity search
            self.training_table = None
            try:
                if hasattr(self, 'db') and self.db:
                    # List available tables
                    tables = self.db.table_names()
                    logger.info(f"Available tables: {tables}")
                    
                    # Try to open with new table name first, then fall back to old name
                    if "news_embeddings_training" in tables:
                        self.training_table = self.db.open_table("news_embeddings_training")
                        logger.info("Connected to news_embeddings_training table for similarity search")
                    elif "news_embeddings" in tables:
                        # Fall back to old table name for backward compatibility
                        self.training_table = self.db.open_table("news_embeddings")
                        logger.info("Connected to news_embeddings table for similarity search (legacy name)")
                    else:
                        logger.warning("No news embeddings table found in available tables")
                        # In prediction mode without training data, we can't do similarity search
                        if hasattr(self, 'mode') and self.mode == 'prediction':
                            logger.warning("No training embeddings available for similarity search in prediction mode")
            except Exception as e:
                logger.warning(f"Could not open training embeddings table: {e}")
        except Exception as e:
            logger.error(f"Failed to connect to LanceDB: {e}")
            # In prediction mode, we can continue without LanceDB
            if hasattr(self, 'mode') and self.mode == 'prediction':
                logger.warning("Continuing in prediction mode without LanceDB connection")
                self.db = None
                self.table = None
                self.training_table = None
            else:
                raise
    
    def _init_duckdb_feature_storage(self):
        """Initialize DuckDB feature storage for news"""
        conn = duckdb.connect(self.db_path)
        
        # Create news_features table with all 37 features
        conn.execute('''
            CREATE TABLE IF NOT EXISTS news_features (
                setup_id TEXT PRIMARY KEY,
                -- Financial Results (7 features)
                count_financial_results INTEGER DEFAULT 0,
                max_severity_financial_results REAL DEFAULT 0.0,
                avg_headline_spin_financial_results TEXT DEFAULT 'neutral',
                sentiment_score_financial_results REAL DEFAULT 0.0,
                profit_warning_present BOOLEAN DEFAULT FALSE,
                synthetic_summary_financial_results TEXT DEFAULT '',
                cot_explanation_financial_results TEXT DEFAULT '',
                -- Corporate Actions (7 features)
                count_corporate_actions INTEGER DEFAULT 0,
                max_severity_corporate_actions REAL DEFAULT 0.0,
                avg_headline_spin_corporate_actions TEXT DEFAULT 'neutral',
                sentiment_score_corporate_actions REAL DEFAULT 0.0,
                capital_raise_present BOOLEAN DEFAULT FALSE,
                synthetic_summary_corporate_actions TEXT DEFAULT '',
                cot_explanation_corporate_actions TEXT DEFAULT '',
                -- Governance (7 features)
                count_governance INTEGER DEFAULT 0,
                max_severity_governance REAL DEFAULT 0.0,
                avg_headline_spin_governance TEXT DEFAULT 'neutral',
                sentiment_score_governance REAL DEFAULT 0.0,
                board_change_present BOOLEAN DEFAULT FALSE,
                synthetic_summary_governance TEXT DEFAULT '',
                cot_explanation_governance TEXT DEFAULT '',
                -- Corporate Events (8 features)
                count_corporate_events INTEGER DEFAULT 0,
                max_severity_corporate_events REAL DEFAULT 0.0,
                avg_headline_spin_corporate_events TEXT DEFAULT 'neutral',
                sentiment_score_corporate_events REAL DEFAULT 0.0,
                contract_award_present BOOLEAN DEFAULT FALSE,
                merger_or_acquisition_present BOOLEAN DEFAULT FALSE,
                synthetic_summary_corporate_events TEXT DEFAULT '',
                cot_explanation_corporate_events TEXT DEFAULT '',
                -- Other Signals (8 features)
                count_other_signals INTEGER DEFAULT 0,
                max_severity_other_signals REAL DEFAULT 0.0,
                avg_headline_spin_other_signals TEXT DEFAULT 'neutral',
                sentiment_score_other_signals REAL DEFAULT 0.0,
                broker_recommendation_present BOOLEAN DEFAULT FALSE,
                credit_rating_change_present BOOLEAN DEFAULT FALSE,
                synthetic_summary_other_signals TEXT DEFAULT '',
                cot_explanation_other_signals TEXT DEFAULT '',
                -- Global explanation
                cot_explanation_news_grouped TEXT DEFAULT '',
                -- Metadata
                extraction_timestamp TIMESTAMP,
                llm_model TEXT,
                CONSTRAINT chk_headline_spin_fr CHECK (avg_headline_spin_financial_results IN ('positive', 'negative', 'neutral', 'uncertain')),
                CONSTRAINT chk_headline_spin_ca CHECK (avg_headline_spin_corporate_actions IN ('positive', 'negative', 'neutral', 'uncertain')),
                CONSTRAINT chk_headline_spin_gov CHECK (avg_headline_spin_governance IN ('positive', 'negative', 'neutral', 'uncertain')),
                CONSTRAINT chk_headline_spin_ce CHECK (avg_headline_spin_corporate_events IN ('positive', 'negative', 'neutral', 'uncertain')),
                CONSTRAINT chk_headline_spin_os CHECK (avg_headline_spin_other_signals IN ('positive', 'negative', 'neutral', 'uncertain'))
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("DuckDB news feature storage tables initialized")
    
    def retrieve_news_by_setup_id(self, setup_id: str) -> pd.DataFrame:
        """Retrieve ALL news for a specific setup_id from DuckDB source (not top-K retrieval)"""
        import duckdb
        
        try:
            # Query DuckDB source table directly
            conn = duckdb.connect(self.db_path)
            
            query = """
                SELECT 
                    rns.setup_id,
                    rns.ticker,
                    rns.headline,
                    rns.rns_date,
                    rns.rns_time,
                    rns.text as chunk_text,
                    rns.url,
                    'rns_announcement' as source_type
                FROM rns_announcements rns
                WHERE rns.setup_id = ?
                    AND rns.headline IS NOT NULL
                    AND LENGTH(rns.headline) > 10
                ORDER BY rns.rns_date DESC, rns.rns_time DESC
            """
            
            results = conn.execute(query, [setup_id]).fetchdf()
            conn.close()
            
            if len(results) == 0:
                logger.warning(f"No RNS announcements found for setup_id: {setup_id}")
                return pd.DataFrame()
            
            # Ensure date is datetime
            results['rns_date'] = pd.to_datetime(results['rns_date'], errors='coerce')
            
            logger.info(f"Retrieved {len(results)} RNS announcements for setup_id: {setup_id} from DuckDB source")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving news for setup_id '{setup_id}': {e}")
            return pd.DataFrame()
    
    def _filter_quality_news(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filtering to reduce noise"""
        if len(news_df) == 0:
            return news_df
        
        original_count = len(news_df)
        
        # Filter out very short headlines (likely garbage)
        news_df = news_df[news_df['headline'].str.len() >= 10]
        
        # Filter out obvious duplicates based on headline similarity
        seen_headlines = set()
        filtered_indices = []
        
        for idx, row in news_df.iterrows():
            headline = row['headline'].lower().strip()
            
            # Simple duplicate detection
            if headline not in seen_headlines:
                seen_headlines.add(headline)
                filtered_indices.append(idx)
        
        news_df = news_df.loc[filtered_indices]
        
        filtered_count = len(news_df)
        if original_count != filtered_count:
            logger.info(f"Quality filtering: {original_count} → {filtered_count} news items")
        
        return news_df
    
    def classify_and_group_news(self, news_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Classify news items and group them by major categories"""
        if len(news_df) == 0:
            return {}
        
        # Initialize groups
        groups = {
            'financial_results': [],
            'corporate_actions': [],
            'governance': [],
            'corporate_events': [],
            'other_signals': []
        }
        
        classification_stats = {}
        
        for _, row in news_df.iterrows():
            headline = row['headline']
            
            # Classify headline to RNS category
            category = self.classifier.classify_headline(headline)
            
            # Track classification stats
            classification_stats[category] = classification_stats.get(category, 0) + 1
            
            # Map category to group
            group = CATEGORY_TO_GROUP.get(category, 'other_signals')  # Default to other_signals
            
            # Add to appropriate group
            news_item = {
                'headline': headline,
                'category': category,
                'rns_date': row.get('rns_date', ''),
                'rns_time': row.get('rns_time', ''),
                'content': row.get('chunk_text', ''),
                'url': row.get('url', '')
            }
            
            groups[group].append(news_item)
        
        # Log classification results
        logger.info(f"News classification stats: {classification_stats}")
        group_stats = {group: len(items) for group, items in groups.items() if len(items) > 0}
        logger.info(f"Group distribution: {group_stats}")
        
        return groups
    
    def find_similar_training_embeddings(self, query_embedding: Union[np.ndarray, str], limit: int = 10) -> List[Dict]:
        """
        Find similar training embeddings with labels
        
        Args:
            query_embedding: Either a numpy array embedding or text content to embed
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
                for table_name in ['news_embeddings_training', 'news_embeddings']:
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
            if isinstance(query_embedding, str):
                model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder="models/sentence_transformers", local_files_only=True)
                query_embedding = model.encode(query_embedding)
            
            # Ensure numpy array
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)
                
            # Search for similar cases
            results = self.training_table.search(query_embedding).limit(limit).to_pandas()
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
        # Find similar cases
        similar_cases = self.find_similar_training_embeddings(query, limit=10)
        
        if not similar_cases:
            logger.warning("No similar cases found for prediction")
            return {}
        
        # Use label converter
        label_converter = LabelConverter()
            
        # Extract labels from similar cases
        similar_labels = [case.get('outperformance_10d', 0.0) for case in similar_cases]
        
        # Compute weighted average prediction
        similarity_scores = [1 - case.get('_distance', 1.0) for case in similar_cases]
        sum_scores = sum(similarity_scores) if similarity_scores else 1.0
        if sum_scores == 0:
            sum_scores = 1.0
            
        weighted_prediction = sum(label * score for label, score in zip(similar_labels, similarity_scores)) / sum_scores
        
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
        
        return prediction

    def extract_group_features(self, group_name: str, news_items: List[Dict], mode: str = "training") -> Dict[str, Any]:
        """Extract features for a single news group using LLM and similarity search"""
        if not news_items:
            return self._get_default_group_features(group_name)
        
        # Handle large groups by chunking
        if len(news_items) > self.max_group_size:
            return self._extract_large_group_features(group_name, news_items, mode)
        else:
            return self._extract_single_group_features(group_name, news_items, mode)
    
    def _extract_single_group_features(self, group_name: str, news_items: List[Dict], mode: str = "training") -> Dict[str, Any]:
        """Extract features for a group that fits in one LLM call with optional similarity enhancement"""
        
        # Prepare context for LLM
        context = self._prepare_group_context(news_items)
        
        # Load prompt template
        prompt = self._load_group_prompt_template(group_name, context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            features = json.loads(response_text)
            
            # Ensure features is a dictionary (safety check)
            if not isinstance(features, dict):
                logger.error(f"LLM returned non-dict response for {group_name}: {type(features)}")
                return self._get_default_group_features(group_name)
            
            # Add metadata
            features['count'] = len(news_items)
            
            # In prediction mode, enhance with similarity features
            if mode == "prediction" and self.training_table is not None:
                similarity_features = self._compute_group_similarity_features(group_name, news_items)
                features.update(similarity_features)
            
            return features
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON for {group_name}: {e}")
            logger.error(f"Raw response: {response_text}")
            return self._get_default_group_features(group_name)
        except Exception as e:
            logger.error(f"Error extracting features for {group_name}: {e}")
            return self._get_default_group_features(group_name)
    
    def _extract_large_group_features(self, group_name: str, news_items: List[Dict], mode: str = "training") -> Dict[str, Any]:
        """Extract features for large groups using chunking and aggregation"""
        
        # Split into chunks
        chunks = [news_items[i:i + self.max_group_size] 
                 for i in range(0, len(news_items), self.max_group_size)]
        
        chunk_features = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing {group_name} chunk {i+1}/{len(chunks)} ({len(chunk)} items)")
            features = self._extract_single_group_features(group_name, chunk, mode)
            chunk_features.append(features)
        
        # Aggregate chunk features
        return self._aggregate_chunk_features(group_name, chunk_features, len(news_items))
    
    def _prepare_group_context(self, news_items: List[Dict]) -> str:
        """Prepare news items as context for LLM"""
        context_parts = []
        
        for i, item in enumerate(news_items, 1):
            date_str = item.get('rns_date', 'Unknown date')
            headline = item['headline']
            content = item.get('content', '')[:200]  # Limit content length
            
            context_parts.append(f"{i}. [{date_str}] {headline}")
            if content and content != headline:
                context_parts.append(f"   Content: {content}...")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _load_group_prompt_template(self, group_name: str, context: str) -> str:
        """Load and format the prompt template for group feature extraction"""
        
        # Group-specific feature requirements
        group_features = {
            'financial_results': {
                'max_severity': 'Maximum event severity (0.0-1.0)',
                'avg_headline_spin': 'Average headline sentiment (positive/negative/neutral/uncertain)',
                'sentiment_score': 'Overall sentiment score (-1.0 to +1.0)',
                'profit_warning_present': 'Whether any profit warning is present (true/false)',
                'synthetic_summary': 'Brief summary of financial results news (≤240 chars)',
                'cot_explanation': 'Reasoning for analysis'
            },
            'corporate_actions': {
                'max_severity': 'Maximum event severity (0.0-1.0)',
                'avg_headline_spin': 'Average headline sentiment (positive/negative/neutral/uncertain)',
                'sentiment_score': 'Overall sentiment score (-1.0 to +1.0)',
                'capital_raise_present': 'Whether capital raising activity is present (true/false)',
                'synthetic_summary': 'Brief summary of corporate actions (≤240 chars)',
                'cot_explanation': 'Reasoning for analysis'
            },
            'governance': {
                'max_severity': 'Maximum event severity (0.0-1.0)',
                'avg_headline_spin': 'Average headline sentiment (positive/negative/neutral/uncertain)',
                'sentiment_score': 'Overall sentiment score (-1.0 to +1.0)',
                'board_change_present': 'Whether board changes are present (true/false)',
                'synthetic_summary': 'Brief summary of governance news (≤240 chars)',
                'cot_explanation': 'Reasoning for analysis'
            },
            'corporate_events': {
                'max_severity': 'Maximum event severity (0.0-1.0)',
                'avg_headline_spin': 'Average headline sentiment (positive/negative/neutral/uncertain)',
                'sentiment_score': 'Overall sentiment score (-1.0 to +1.0)',
                'contract_award_present': 'Whether contract awards are present (true/false)',
                'merger_or_acquisition_present': 'Whether M&A activity is present (true/false)',
                'synthetic_summary': 'Brief summary of corporate events (≤240 chars)',
                'cot_explanation': 'Reasoning for analysis'
            },
            'other_signals': {
                'max_severity': 'Maximum event severity (0.0-1.0)',
                'avg_headline_spin': 'Average headline sentiment (positive/negative/neutral/uncertain)',
                'sentiment_score': 'Overall sentiment score (-1.0 to +1.0)',
                'broker_recommendation_present': 'Whether broker recommendations are present (true/false)',
                'credit_rating_change_present': 'Whether credit rating changes are present (true/false)',
                'synthetic_summary': 'Brief summary of other market signals (≤240 chars)',
                'cot_explanation': 'Reasoning for analysis'
            }
        }
        
        features = group_features.get(group_name, group_features['other_signals'])
        
        # Build JSON schema for this group
        json_template = "{\n"
        for feature, description in features.items():
            if feature in ['max_severity', 'sentiment_score']:
                json_template += f'  "{feature}": 0.0,  // {description}\n'
            elif feature == 'avg_headline_spin':
                json_template += f'  "{feature}": "neutral",  // {description}\n'
            elif 'present' in feature:
                json_template += f'  "{feature}": false,  // {description}\n'
            else:
                json_template += f'  "{feature}": "",  // {description}\n'
        json_template += "}"
        
        prompt = f"""You are a financial news analyst. Analyze the {group_name.replace('_', ' ')} news items below and extract features as JSON.

**Instructions:**
- Analyze ALL the provided news items for this group
- Extract features as specified in the JSON template
- For categorical fields, use ONLY the specified allowed values
- Synthetic summary must be ≤240 characters
- Return ONLY valid JSON, no extra text

**JSON Template:**
{json_template}

**News Items to Analyze:**
{context}

Extract the features as JSON:"""

        return prompt
    
    def _get_default_group_features(self, group_name: str) -> Dict[str, Any]:
        """Get default features for empty or failed groups"""
        defaults = {
            'max_severity': 0.0,
            'avg_headline_spin': 'neutral', 
            'sentiment_score': 0.0,
            'synthetic_summary': f'No {group_name.replace("_", " ")} news available',
            'cot_explanation': f'No news items in {group_name} group',
            'count': 0
        }
        
        # Add group-specific boolean features
        if group_name == 'financial_results':
            defaults['profit_warning_present'] = False
        elif group_name == 'corporate_actions':
            defaults['capital_raise_present'] = False
        elif group_name == 'governance':
            defaults['board_change_present'] = False
        elif group_name == 'corporate_events':
            defaults['contract_award_present'] = False
            defaults['merger_or_acquisition_present'] = False
        elif group_name == 'other_signals':
            defaults['broker_recommendation_present'] = False
            defaults['credit_rating_change_present'] = False
        
        return defaults
    
    def _aggregate_chunk_features(self, group_name: str, chunk_features: List[Dict], total_count: int) -> Dict[str, Any]:
        """Aggregate features from multiple chunks"""
        
        if not chunk_features:
            return self._get_default_group_features(group_name)
        
        # Aggregate numeric features
        max_severity = max(f.get('max_severity', 0.0) for f in chunk_features)
        avg_sentiment = sum(f.get('sentiment_score', 0.0) for f in chunk_features) / len(chunk_features)
        
        # Aggregate boolean features (OR logic - true if any chunk has true)
        boolean_features = {}
        for chunk in chunk_features:
            for key, value in chunk.items():
                if 'present' in key and isinstance(value, bool):
                    boolean_features[key] = boolean_features.get(key, False) or value
        
        # Aggregate categorical features (most common)
        spin_values = [f.get('avg_headline_spin', 'neutral') for f in chunk_features]
        most_common_spin = max(set(spin_values), key=spin_values.count)
        
        # Combine summaries
        summaries = [f.get('synthetic_summary', '') for f in chunk_features if f.get('synthetic_summary')]
        combined_summary = '; '.join(summaries)[:240]  # Limit to 240 chars
        
        # Combine explanations
        explanations = [f.get('cot_explanation', '') for f in chunk_features if f.get('cot_explanation')]
        combined_explanation = '; '.join(explanations)
        
        result = {
            'max_severity': max_severity,
            'avg_headline_spin': most_common_spin,
            'sentiment_score': avg_sentiment,
            'synthetic_summary': combined_summary or f'Analyzed {total_count} {group_name.replace("_", " ")} items',
            'cot_explanation': combined_explanation or f'Aggregated from {len(chunk_features)} chunks',
            'count': total_count
        }
        
        # Add boolean features
        result.update(boolean_features)
        
        return result
    
    def store_features(self, features: NewsFeatureSchema):
        """Store extracted news features in DuckDB"""
        conn = duckdb.connect(self.db_path)
        
        try:
            # Store in news_features table
            conn.execute('''
                INSERT OR REPLACE INTO news_features (
                    setup_id,
                    count_financial_results, max_severity_financial_results, avg_headline_spin_financial_results,
                    sentiment_score_financial_results, profit_warning_present, synthetic_summary_financial_results,
                    cot_explanation_financial_results,
                    count_corporate_actions, max_severity_corporate_actions, avg_headline_spin_corporate_actions,
                    sentiment_score_corporate_actions, capital_raise_present, synthetic_summary_corporate_actions,
                    cot_explanation_corporate_actions,
                    count_governance, max_severity_governance, avg_headline_spin_governance,
                    sentiment_score_governance, board_change_present, synthetic_summary_governance,
                    cot_explanation_governance,
                    count_corporate_events, max_severity_corporate_events, avg_headline_spin_corporate_events,
                    sentiment_score_corporate_events, contract_award_present, merger_or_acquisition_present,
                    synthetic_summary_corporate_events, cot_explanation_corporate_events,
                    count_other_signals, max_severity_other_signals, avg_headline_spin_other_signals,
                    sentiment_score_other_signals, broker_recommendation_present, credit_rating_change_present,
                    synthetic_summary_other_signals, cot_explanation_other_signals,
                    cot_explanation_news_grouped, extraction_timestamp, llm_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                features.setup_id,
                # Financial Results
                features.count_financial_results, features.max_severity_financial_results,
                features.avg_headline_spin_financial_results, features.sentiment_score_financial_results,
                features.profit_warning_present, features.synthetic_summary_financial_results,
                features.cot_explanation_financial_results,
                # Corporate Actions
                features.count_corporate_actions, features.max_severity_corporate_actions,
                features.avg_headline_spin_corporate_actions, features.sentiment_score_corporate_actions,
                features.capital_raise_present, features.synthetic_summary_corporate_actions,
                features.cot_explanation_corporate_actions,
                # Governance
                features.count_governance, features.max_severity_governance,
                features.avg_headline_spin_governance, features.sentiment_score_governance,
                features.board_change_present, features.synthetic_summary_governance,
                features.cot_explanation_governance,
                # Corporate Events
                features.count_corporate_events, features.max_severity_corporate_events,
                features.avg_headline_spin_corporate_events, features.sentiment_score_corporate_events,
                features.contract_award_present, features.merger_or_acquisition_present,
                features.synthetic_summary_corporate_events, features.cot_explanation_corporate_events,
                # Other Signals
                features.count_other_signals, features.max_severity_other_signals,
                features.avg_headline_spin_other_signals, features.sentiment_score_other_signals,
                features.broker_recommendation_present, features.credit_rating_change_present,
                features.synthetic_summary_other_signals, features.cot_explanation_other_signals,
                # Global
                features.cot_explanation_news_grouped,
                features.extraction_timestamp, features.llm_model
            ))
            
            conn.commit()
            logger.info(f"Stored news features for setup_id: {features.setup_id} in DuckDB")
            
        except Exception as e:
            logger.error(f"Error storing news features in DuckDB: {e}")
            raise
        finally:
            conn.close()
    
    def get_stored_features(self, setup_id: str) -> Optional[Dict]:
        """Retrieve stored news features for a setup from DuckDB"""
        conn = duckdb.connect(self.db_path)
        
        try:
            result = conn.execute(
                "SELECT * FROM news_features WHERE setup_id = ?",
                (setup_id,)
            ).fetchone()
            
            if result:
                columns = [desc[0] for desc in conn.description]
                feature_dict = dict(zip(columns, result))
                return feature_dict
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving stored news features from DuckDB: {e}")
            return None
        finally:
            conn.close()
    
    def _compute_group_similarity_features(self, group_name: str, news_items: List[Dict]) -> Dict[str, Any]:
        """Compute similarity-based features for a news group"""
        if not news_items or not self.training_table:
            return {}
        
        all_similarity_features = []
        
        for item in news_items:
            # Get content from the item
            content = item.get('content', '')
            if not content:
                continue
                
            # Use the improved predict_via_similarity method directly with content
            similar_cases = self.find_similar_training_embeddings(content)
            if similar_cases:
                similarity_features = self.compute_similarity_features(similar_cases)
                all_similarity_features.append(similarity_features)
        
        # Aggregate similarity features
        if all_similarity_features:
            aggregated = {}
            for key in all_similarity_features[0].keys():
                values = [f[key] for f in all_similarity_features]
                aggregated[f"{group_name}_similarity_{key}"] = float(np.mean(values))
            return aggregated
        
        return {}

    def process_setup(self, setup_id: str, mode: str = None) -> Optional[NewsFeatureSchema]:
        """
        Complete pipeline: retrieve, classify, group, extract features, and store
        
        Args:
            setup_id: Trading setup identifier
            mode: Either 'training' or 'prediction' (overrides self.mode if provided)
            
        Returns:
            Extracted news features or None if processing failed
        """
        # Use provided mode or fall back to instance mode
        current_mode = mode if mode is not None else self.mode
        logger.info(f"Processing news for setup: {setup_id} in {current_mode} mode")
        
        # Step 1: Retrieve ALL news for this setup
        news_df = self.retrieve_news_by_setup_id(setup_id)
        if len(news_df) == 0:
            logger.warning(f"No news found for setup {setup_id}")
            # Still create a features object with all zeros/defaults
            return self._create_empty_features(setup_id)
        
        # Step 2: Apply quality filtering
        filtered_news_df = self._filter_quality_news(news_df)
        
        # Step 3: Classify and group news
        news_groups = self.classify_and_group_news(filtered_news_df)
        
        # Step 4: Extract features for each group
        all_group_features = {}
        
        for group_name in ['financial_results', 'corporate_actions', 'governance', 'corporate_events', 'other_signals']:
            group_items = news_groups.get(group_name, [])
            logger.info(f"Processing {group_name}: {len(group_items)} items")
            
            group_features = self.extract_group_features(group_name, group_items, current_mode)
            all_group_features[group_name] = group_features
        
        # Step 5: Create global explanation
        global_explanation = self._create_global_explanation(all_group_features, len(filtered_news_df))
        
        # Step 6: Build final feature schema
        try:
            features = self._build_feature_schema(setup_id, all_group_features, global_explanation)
            
            # Step 7: Store features
            self.store_features(features)
            
            # Step 8: In prediction mode, generate similarity-based predictions
            if current_mode == 'prediction' and self.training_table is not None:
                # Create content for similarity search
                content = global_explanation
                for group_name, features in all_group_features.items():
                    if features.get('synthetic_summary'):
                        content += f" {features['synthetic_summary']}"
                
                # Generate prediction
                prediction = self.predict_via_similarity(content)
                if prediction:
                    prediction['setup_id'] = setup_id
                    prediction['domain'] = 'news'
                    prediction['prediction_timestamp'] = datetime.now().isoformat()
                    logger.info(f"Generated similarity prediction for {setup_id}: {prediction.get('predicted_outperformance', 'N/A')}")
                    return features, prediction
            
            logger.info(f"Successfully processed news features for setup {setup_id}")
            return features
            
        except Exception as e:
            logger.error(f"Error building feature schema for setup {setup_id}: {e}")
            return None
    
    def _create_empty_features(self, setup_id: str) -> NewsFeatureSchema:
        """Create empty feature set for setups with no news"""
        return NewsFeatureSchema(
            setup_id=setup_id,
            # Financial Results
            count_financial_results=0, max_severity_financial_results=0.0,
            avg_headline_spin_financial_results='neutral', sentiment_score_financial_results=0.0,
            profit_warning_present=False, synthetic_summary_financial_results='No financial results news',
            cot_explanation_financial_results='No news available',
            # Corporate Actions
            count_corporate_actions=0, max_severity_corporate_actions=0.0,
            avg_headline_spin_corporate_actions='neutral', sentiment_score_corporate_actions=0.0,
            capital_raise_present=False, synthetic_summary_corporate_actions='No corporate actions news',
            cot_explanation_corporate_actions='No news available',
            # Governance
            count_governance=0, max_severity_governance=0.0,
            avg_headline_spin_governance='neutral', sentiment_score_governance=0.0,
            board_change_present=False, synthetic_summary_governance='No governance news',
            cot_explanation_governance='No news available',
            # Corporate Events
            count_corporate_events=0, max_severity_corporate_events=0.0,
            avg_headline_spin_corporate_events='neutral', sentiment_score_corporate_events=0.0,
            contract_award_present=False, merger_or_acquisition_present=False,
            synthetic_summary_corporate_events='No corporate events news',
            cot_explanation_corporate_events='No news available',
            # Other Signals
            count_other_signals=0, max_severity_other_signals=0.0,
            avg_headline_spin_other_signals='neutral', sentiment_score_other_signals=0.0,
            broker_recommendation_present=False, credit_rating_change_present=False,
            synthetic_summary_other_signals='No other signals news',
            cot_explanation_other_signals='No news available',
            # Global
            cot_explanation_news_grouped='No news available for this setup'
        )
    
    def _create_global_explanation(self, group_features: Dict[str, Dict], total_news_count: int) -> str:
        """Create a global explanation combining insights from all groups"""
        
        explanations = []
        active_groups = []
        
        for group_name, features in group_features.items():
            if features.get('count', 0) > 0:
                active_groups.append(group_name.replace('_', ' '))
                if features.get('cot_explanation'):
                    explanations.append(f"{group_name.replace('_', ' ')}: {features['cot_explanation'][:100]}")
        
        if not explanations:
            return f'No news items found for analysis'
        
        global_explanation = f"Analyzed {total_news_count} news items across {len(active_groups)} categories. " + \
                           "; ".join(explanations)
        
        return global_explanation
    
    def _build_feature_schema(self, setup_id: str, group_features: Dict[str, Dict], global_explanation: str) -> NewsFeatureSchema:
        """Build the final NewsFeatureSchema from group features"""
        
        # Extract features for each group
        fr = group_features.get('financial_results', {})
        ca = group_features.get('corporate_actions', {})
        gov = group_features.get('governance', {})
        ce = group_features.get('corporate_events', {})
        os = group_features.get('other_signals', {})
        
        return NewsFeatureSchema(
            setup_id=setup_id,
            llm_model=self.llm_model,
            # Financial Results
            count_financial_results=fr.get('count', 0),
            max_severity_financial_results=fr.get('max_severity', 0.0),
            avg_headline_spin_financial_results=fr.get('avg_headline_spin', 'neutral'),
            sentiment_score_financial_results=fr.get('sentiment_score', 0.0),
            profit_warning_present=fr.get('profit_warning_present', False),
            synthetic_summary_financial_results=fr.get('synthetic_summary', 'No financial results')[:240],
            cot_explanation_financial_results=fr.get('cot_explanation', 'No analysis'),
            # Corporate Actions
            count_corporate_actions=ca.get('count', 0),
            max_severity_corporate_actions=ca.get('max_severity', 0.0),
            avg_headline_spin_corporate_actions=ca.get('avg_headline_spin', 'neutral'),
            sentiment_score_corporate_actions=ca.get('sentiment_score', 0.0),
            capital_raise_present=ca.get('capital_raise_present', False),
            synthetic_summary_corporate_actions=ca.get('synthetic_summary', 'No corporate actions')[:240],
            cot_explanation_corporate_actions=ca.get('cot_explanation', 'No analysis'),
            # Governance
            count_governance=gov.get('count', 0),
            max_severity_governance=gov.get('max_severity', 0.0),
            avg_headline_spin_governance=gov.get('avg_headline_spin', 'neutral'),
            sentiment_score_governance=gov.get('sentiment_score', 0.0),
            board_change_present=gov.get('board_change_present', False),
            synthetic_summary_governance=gov.get('synthetic_summary', 'No governance news')[:240],
            cot_explanation_governance=gov.get('cot_explanation', 'No analysis'),
            # Corporate Events
            count_corporate_events=ce.get('count', 0),
            max_severity_corporate_events=ce.get('max_severity', 0.0),
            avg_headline_spin_corporate_events=ce.get('avg_headline_spin', 'neutral'),
            sentiment_score_corporate_events=ce.get('sentiment_score', 0.0),
            contract_award_present=ce.get('contract_award_present', False),
            merger_or_acquisition_present=ce.get('merger_or_acquisition_present', False),
            synthetic_summary_corporate_events=ce.get('synthetic_summary', 'No corporate events')[:240],
            cot_explanation_corporate_events=ce.get('cot_explanation', 'No analysis'),
            # Other Signals
            count_other_signals=os.get('count', 0),
            max_severity_other_signals=os.get('max_severity', 0.0),
            avg_headline_spin_other_signals=os.get('avg_headline_spin', 'neutral'),
            sentiment_score_other_signals=os.get('sentiment_score', 0.0),
            broker_recommendation_present=os.get('broker_recommendation_present', False),
            credit_rating_change_present=os.get('credit_rating_change_present', False),
            synthetic_summary_other_signals=os.get('synthetic_summary', 'No other signals')[:240],
            cot_explanation_other_signals=os.get('cot_explanation', 'No analysis'),
            # Global
            cot_explanation_news_grouped=global_explanation
        )
    
    def batch_process_setups(self, setup_ids: List[str]) -> Dict[str, Optional[NewsFeatureSchema]]:
        """Process multiple setups in batch, skipping already-processed setups"""
        results = {}
        skipped_count = 0
        
        for i, setup_id in enumerate(setup_ids, 1):
            # Check if features already exist
            existing_features = self.get_stored_features(setup_id)
            if existing_features:
                logger.info(f"⏭️ Skipping setup {i}/{len(setup_ids)}: {setup_id} (already processed)")
                results[setup_id] = existing_features  # Return existing features
                skipped_count += 1
                continue
                
            logger.info(f"Processing setup {i}/{len(setup_ids)}: {setup_id}")
            
            try:
                result = self.process_setup(setup_id)
                results[setup_id] = result
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing setup {setup_id}: {e}")
                results[setup_id] = None
        
        if skipped_count > 0:
            logger.info(f"✅ Skipped {skipped_count} already-processed setups")
        
        return results
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get statistics about the classifier cache"""
        stats = {
            'total_classifications': len(self.classifier.cache),
            'category_distribution': {},
            'group_distribution': {}
        }
        
        for headline, category in self.classifier.cache.items():
            stats['category_distribution'][category] = stats['category_distribution'].get(category, 0) + 1
            
            group = CATEGORY_TO_GROUP.get(category, 'other_signals')
            stats['group_distribution'][group] = stats['group_distribution'].get(group, 0) + 1
        
        return stats
    
    def create_embeddings_only(self, setup_id: str, mode: str = None) -> bool:
        """
        Create embeddings for a setup without making predictions
        
        Args:
            setup_id: Setup ID to process
            mode: Override the agent's mode (training or prediction)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use provided mode or default to agent's mode
            mode = mode or self.mode
            
            # Get news for the setup
            news_df = self.retrieve_news_by_setup_id(setup_id)
            
            if news_df.empty:
                logger.warning(f"No news found for setup {setup_id}")
                return False
            
            # Filter out low-quality news
            news_df = self._filter_quality_news(news_df)
            
            if news_df.empty:
                logger.warning(f"No quality news found for setup {setup_id}")
                return False
            
            # Classify and group news
            grouped_news = self.classify_and_group_news(news_df)
            
            # Create embeddings for each group
            for group_name, news_items in grouped_news.items():
                for news_item in news_items:
                    # Create text for embedding
                    text = f"{news_item['headline']} {news_item.get('content', '')}"
                    
                    # Initialize embedding pipeline if needed
                    if not hasattr(self, 'embedding_pipeline'):
                        self.embedding_pipeline = NewsEmbeddingPipelineDuckDB(
                            db_path=self.db_path,
                            lancedb_dir=self.lancedb_dir,
                            mode=mode
                        )
                    
                    # Create embedding
                    self.embedding_pipeline.create_embedding(
                        setup_id=setup_id,
                        text=text,
                        metadata={
                            'headline': news_item['headline'],
                            'category': news_item['category'],
                            'group': group_name,
                            'date': news_item.get('date', ''),
                            'source': news_item.get('source', '')
                        }
                    )
            
            logger.info(f"Created embeddings for {len(news_df)} news items for setup {setup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating embeddings for setup {setup_id}: {str(e)}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.setup_validator.close()
            logger.info("Enhanced News Agent cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class NewsEmbeddingPipelineDuckDB:
    """Pipeline for creating and storing news embeddings in DuckDB"""
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        lancedb_dir: str = "lancedb_store",
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        mode: str = "training"  # Either "training" or "prediction"
    ):
        self.db_path = db_path
        self.lancedb_dir = lancedb_dir
        self.embedding_model_name = embedding_model_name
        self.mode = mode
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Connect to LanceDB
        self.db = lancedb.connect(lancedb_dir)
        
        # Table name depends on mode
        self.table_name = f"news_embeddings_{mode}"
        
        # Create table if it doesn't exist
        self._create_table_if_not_exists()
        
    def _create_table_if_not_exists(self):
        """Create embedding table if it doesn't exist"""
        try:
            # Check if table exists
            tables = self.db.table_names()
            if self.table_name not in tables:
                # Create schema
                schema = {
                    "setup_id": "string",
                    "text": "string",
                    "vector": "float32[384]",  # Adjust dimension based on model
                    "headline": "string",
                    "category": "string",
                    "group": "string",
                    "date": "string",
                    "source": "string"
                }
                
                # Create empty table
                self.db.create_table(
                    self.table_name,
                    schema=schema,
                    mode="overwrite"
                )
                
                logger.info(f"Created new embedding table: {self.table_name}")
            
            # Open the table
            self.table = self.db.open_table(self.table_name)
            logger.info(f"Connected to embedding table: {self.table_name}")
            
        except Exception as e:
            logger.error(f"Error creating/opening embedding table: {e}")
            self.table = None
    
    def create_embedding(
        self,
        setup_id: str,
        text: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Create and store embedding for a text
        
        Args:
            setup_id: Setup ID
            text: Text to embed
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(text)
            
            # Prepare data for insertion
            data = {
                "setup_id": setup_id,
                "text": text[:1000],  # Truncate text to avoid storage issues
                "vector": embedding,
                "headline": metadata.get("headline", "")[:200],
                "category": metadata.get("category", "")[:50],
                "group": metadata.get("group", "")[:50],
                "date": metadata.get("date", "")[:50],
                "source": metadata.get("source", "")[:50]
            }
            
            # Add to table
            self.table.add([data])
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return False


def main():
    """Test the Enhanced News Agent"""
    import os
    
    # Set up environment
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize agent
    agent = EnhancedNewsAgentDuckDB(
        db_path="../data/sentiment_system.duckdb",
        lancedb_dir="../lancedb_store"
    )
    
    try:
        # Test classification
        test_headlines = [
            "Company announces final results for the year",
            "Interim dividend declared",
            "Board changes announced", 
            "Contract awarded for major project",
            "Broker upgrades rating to buy"
        ]
        
        print("\n🔍 Testing headline classification:")
        for headline in test_headlines:
            category = agent.classifier.classify_headline(headline)
            group = CATEGORY_TO_GROUP.get(category, 'other_signals')
            print(f"  '{headline}' → {category} → {group}")
        
        # Test processing (if setup_ids are available)
        print("\n📊 Testing setup processing:")
        # Get available setup_ids
        all_data = agent.table.to_pandas()
        setup_ids = all_data['setup_id'].unique()[:3]  # Test first 3
        
        for setup_id in setup_ids:
            print(f"\nProcessing {setup_id}...")
            features = agent.process_setup(setup_id)
            if features:
                print(f"  ✅ Success: {features.count_financial_results + features.count_corporate_actions + features.count_governance + features.count_corporate_events + features.count_other_signals} total news items")
                print(f"  Groups: FR={features.count_financial_results}, CA={features.count_corporate_actions}, GOV={features.count_governance}, CE={features.count_corporate_events}, OS={features.count_other_signals}")
            else:
                print(f"  ❌ Failed")
        
        print("\n📈 Classification statistics:")
        stats = agent.get_classification_stats()
        print(f"  Total classifications: {stats['total_classifications']}")
        print(f"  Category distribution: {stats['category_distribution']}")
        print(f"  Group distribution: {stats['group_distribution']}")
        
    finally:
        agent.cleanup()


if __name__ == "__main__":
    main() 