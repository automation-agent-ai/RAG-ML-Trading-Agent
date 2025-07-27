#!/usr/bin/env python3
"""
Enhanced News Agent with DuckDB Feature Storage and Grouped Processing
====================================================================

This agent processes RNS announcements for feature extraction with:
- Category classification (dictionary + fuzzy matching + LLM fallback)
- Group-level feature extraction (5 major groups)
- Complete record processing (not top-K retrieval)
- DuckDB storage integration

Author: Enhanced News Agent
Date: 2025-01-06
"""

import json
import hashlib
import time
import logging
import difflib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import duckdb
import lancedb
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np

# Import from existing modules
import sys
from pathlib import Path

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.setup_validator_duckdb import SetupValidatorDuckDB
from embeddings.embed_news_duckdb import NewsEmbeddingPipelineDuckDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class CategoryClassifier:
    """Handles RNS category classification with multiple strategies"""
    
    def __init__(self, openai_client: OpenAI):
        self.openai_client = openai_client
        self.cache = {}  # Simple in-memory cache for this session
    
    def classify_headline(self, headline: str) -> str:
        """
        Classify headline to RNS category using:
        1. Fast dictionary lookup for exact/common patterns
        2. Fuzzy matching for near-matches  
        3. LLM fallback for ambiguous cases
        """
        headline = headline.strip()
        
        # Check cache first
        if headline in self.cache:
            return self.cache[headline]
        
        # Strategy 1: Fast dictionary lookup for common patterns
        category = self._fast_pattern_match(headline)
        if category:
            self.cache[headline] = category
            return category
        
        # Strategy 2: Fuzzy matching for near-matches
        category = self._fuzzy_match(headline)
        if category:
            self.cache[headline] = category
            return category
        
        # Strategy 3: LLM fallback for ambiguous cases
        category = self._llm_classify(headline)
        self.cache[headline] = category
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
            response = self.openai_client.chat.completions.create(
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


class EnhancedNewsAgentDuckDB:
    """
    Enhanced News Agent with DuckDB feature storage and grouped processing
    
    This agent processes ALL RNS news for each setup_id, classifies to groups,
    and extracts group-level features using LLM calls.
    """
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        lancedb_dir: str = "lancedb_store", 
        table_name: str = "news_embeddings",
        llm_model: str = "gpt-4o-mini",
        max_group_size: int = 10  # Max items per group before chunking
    ):
        """
        Initialize Enhanced News Agent
        
        Args:
            db_path: Path to DuckDB database
            lancedb_dir: Directory for LanceDB vector store
            table_name: LanceDB table name for news embeddings
            llm_model: OpenAI model for feature extraction
            max_group_size: Maximum news items per group before chunking
        """
        self.db_path = db_path
        self.lancedb_dir = lancedb_dir
        self.table_name = table_name
        self.llm_model = llm_model
        self.max_group_size = max_group_size
        
        # Initialize components
        self.setup_validator = SetupValidatorDuckDB(db_path=db_path)
        self.openai_client = OpenAI()
        self.classifier = CategoryClassifier(self.openai_client)
        
        # Connect to databases
        self._connect_to_lancedb()
        self._init_duckdb_feature_storage()
        
        # Initialize training table for similarity search
        self.training_table = None
        try:
            if hasattr(self, 'db') and self.db:
                self.training_table = self.db.open_table("news_embeddings_training")
                logger.info("Connected to news_embeddings_training table for similarity search")
        except Exception as e:
            logger.warning(f"Could not open training embeddings table: {e}")
        
        logger.info("Enhanced News Agent (DuckDB) initialized successfully")
    
    def _connect_to_lancedb(self):
        """Connect to LanceDB"""
        try:
            lancedb_path = Path(self.lancedb_dir)
            if not lancedb_path.exists():
                raise FileNotFoundError(f"LanceDB directory not found: {lancedb_path}")
            
            self.db = lancedb.connect(self.lancedb_dir)
            self.table = self.db.open_table(self.table_name)
            logger.info(f"Connected to LanceDB table: {self.table_name}")
        except Exception as e:
            logger.error(f"Failed to connect to LanceDB: {e}")
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
    
    def find_similar_training_embeddings(self, query_embedding: np.ndarray, limit: int = 10) -> List[Dict]:
        """Find similar training embeddings with labels"""
        if self.training_table is None:
            logger.warning("Training embeddings table not available")
            return []
        
        try:
            results = self.training_table.search(query_embedding).limit(limit).to_pandas()
            return results.to_dict('records')
        except Exception as e:
            logger.error(f"Error searching similar embeddings: {e}")
            return []
    
    def compute_similarity_features(self, similar_cases: List[Dict]) -> Dict[str, float]:
        """Compute similarity-based features from similar cases"""
        if not similar_cases:
            return {
                'positive_signal_strength': 0.0,
                'negative_risk_score': 0.0,
                'neutral_probability': 0.0,
                'historical_pattern_confidence': 0.0,
                'similar_cases_count': 0
            }
        
        # Calculate performance-based features
        positive_cases = [c for c in similar_cases if c.get('outperformance_10d', 0) > 0.02]  # 2% threshold
        negative_cases = [c for c in similar_cases if c.get('outperformance_10d', 0) < -0.02]
        neutral_cases = [c for c in similar_cases if abs(c.get('outperformance_10d', 0)) <= 0.02]
        
        total_cases = len(similar_cases)
        
        # Calculate weighted features based on similarity scores
        similarity_scores = [c.get('_distance', 1.0) for c in similar_cases]  # Lower distance = higher similarity
        max_distance = max(similarity_scores)
        similarity_weights = [1 - (score / max_distance) for score in similarity_scores]
        
        # Calculate weighted ratios
        positive_ratio = sum(w for c, w in zip(similar_cases, similarity_weights) 
                           if c.get('outperformance_10d', 0) > 0.02) / sum(similarity_weights)
        
        negative_ratio = sum(w for c, w in zip(similar_cases, similarity_weights)
                           if c.get('outperformance_10d', 0) < -0.02) / sum(similarity_weights)
        
        # Calculate pattern confidence based on consistency
        performance_std = np.std([c.get('outperformance_10d', 0) for c in similar_cases])
        pattern_confidence = 1.0 / (1.0 + performance_std)  # Higher std = lower confidence
        
        return {
            'positive_signal_strength': float(positive_ratio),
            'negative_risk_score': float(negative_ratio),
            'neutral_probability': float(len(neutral_cases) / total_cases),
            'historical_pattern_confidence': float(pattern_confidence),
            'similar_cases_count': total_cases
        }
    
    def predict_via_similarity(self, setup_id: str, content: str) -> Dict[str, Any]:
        """Direct prediction using similarity to training embeddings"""
        # Create embedding for prediction
        embedder = NewsEmbeddingPipelineDuckDB(
            db_path=str(self.db_path),
            lancedb_dir=str(self.lancedb_dir),
            include_labels=False,
            mode="prediction"
        )
        
        # Create temporary record for embedding
        record = {
            'setup_id': setup_id,
            'chunk_text': content,
            'text_length': len(content)
        }
        
        # Create embedding
        records = embedder.create_embeddings([record])
        if not records:
            logger.error("Failed to create embedding for prediction")
            return {}
            
        query_embedding = np.array(records[0]['vector'])
        
        # Find similar cases
        similar_cases = self.find_similar_training_embeddings(query_embedding, limit=10)
        
        if not similar_cases:
            logger.warning("No similar cases found for prediction")
            return {}
            
        # Extract labels from similar cases
        similar_labels = [case.get('outperformance_10d', 0.0) for case in similar_cases]
        
        # Compute weighted average prediction
        similarity_scores = [1 - case.get('_distance', 1.0) for case in similar_cases]
        weighted_prediction = np.average(similar_labels, weights=similarity_scores)
        
        # Compute confidence metrics
        prediction = {
            'predicted_outperformance': float(weighted_prediction),
            'confidence': float(1.0 / (1.0 + np.std(similar_labels))),
            'positive_ratio': sum(1 for l in similar_labels if l > 0.02) / len(similar_labels),
            'negative_ratio': sum(1 for l in similar_labels if l < -0.02) / len(similar_labels),
            'neutral_ratio': sum(1 for l in similar_labels if abs(l) <= 0.02) / len(similar_labels),
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
            response = self.openai_client.chat.completions.create(
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
        
        # Create embeddings for news items
        embedder = NewsEmbeddingPipelineDuckDB(
            db_path=str(self.db_path),
            lancedb_dir=str(self.lancedb_dir),
            include_labels=False,
            mode="prediction"
        )
        
        all_similarity_features = []
        
        for item in news_items:
            # Create temporary record for embedding
            record = {
                'setup_id': item.get('setup_id', ''),
                'chunk_text': item.get('content', ''),
                'text_length': len(item.get('content', ''))
            }
            
            # Create embedding
            embedded_records = embedder.create_embeddings([record])
            if embedded_records:
                query_embedding = np.array(embedded_records[0]['vector'])
                similar_cases = self.find_similar_training_embeddings(query_embedding)
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

    def process_setup(self, setup_id: str, mode: str = "training") -> Optional[NewsFeatureSchema]:
        """
        Complete pipeline: retrieve, classify, group, extract features, and store
        
        Args:
            setup_id: Trading setup identifier
            mode: Either 'training' or 'prediction'
            
        Returns:
            Extracted news features or None if processing failed
        """
        logger.info(f"Processing news for setup: {setup_id}")
        
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
            
            group_features = self.extract_group_features(group_name, group_items, mode)
            all_group_features[group_name] = group_features
        
        # Step 5: Create global explanation
        global_explanation = self._create_global_explanation(all_group_features, len(filtered_news_df))
        
        # Step 6: Build final feature schema
        try:
            features = self._build_feature_schema(setup_id, all_group_features, global_explanation)
            
            # Step 7: Store features
            self.store_features(features)
            
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
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.setup_validator.close()
            logger.info("Enhanced News Agent cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


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