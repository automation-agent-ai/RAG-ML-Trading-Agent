import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import duckdb

# Import feature extraction components
from agents.fundamentals.enhanced_fundamentals_agent_duckdb import EnhancedFundamentalsAgentDuckDB
from agents.news.enhanced_news_agent_duckdb import EnhancedNewsAgentDuckDB
from agents.userposts.enhanced_userposts_agent_complete import EnhancedUserPostsAgentComplete
from agents.analyst_recommendations.enhanced_analyst_recommendations_agent_duckdb import EnhancedAnalystRecommendationsAgentDuckDB
from core.ml_feature_merger import MLFeatureMerger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLPipelineRunner:
    def __init__(self, db_path: str = "data/sentiment_system.duckdb",
                 lancedb_dir: str = "lancedb_store",
                 mode: str = "training"):
        """Initialize ML Pipeline Runner
        
        Args:
            db_path: Path to DuckDB database
            lancedb_dir: Directory for LanceDB storage
            mode: Either 'training' or 'prediction'
        """
        self.db_path = Path(db_path)
        self.lancedb_dir = Path(lancedb_dir)
        self.mode = mode
        
        # Initialize agents with proper mode
        self.news_agent = EnhancedNewsAgentDuckDB(
            db_path=str(self.db_path),
            lancedb_dir=str(self.lancedb_dir)
        )
        
        self.fundamentals_agent = EnhancedFundamentalsAgentDuckDB(
            db_path=str(self.db_path),
            lancedb_dir=str(self.lancedb_dir)
        )
        
        self.analyst_agent = EnhancedAnalystRecommendationsAgentDuckDB(
            db_path=str(self.db_path),
            lancedb_dir=str(self.lancedb_dir)
        )
        
        self.userposts_agent = EnhancedUserPostsAgentComplete(
            db_path=str(self.db_path),
            lancedb_dir=str(self.lancedb_dir)
        )

    def create_embeddings(self, setup_ids: List[str]) -> None:
        """Create embeddings with proper training/prediction separation"""
        logger.info(f"Creating embeddings in {self.mode} mode for {len(setup_ids)} setups")
        
        # Initialize embedders with proper mode
        news_embedder = NewsEmbeddingPipelineDuckDB(
            db_path=str(self.db_path),
            lancedb_dir=str(self.lancedb_dir),
            include_labels=(self.mode == "training"),
            mode=self.mode
        )
        
        fundamentals_embedder = FundamentalsEmbedderDuckDB(
            db_path=str(self.db_path),
            lancedb_dir=str(self.lancedb_dir),
            include_labels=(self.mode == "training"),
            mode=self.mode
        )
        
        analyst_embedder = AnalystRecommendationsEmbedderDuckDB(
            db_path=str(self.db_path),
            lancedb_dir=str(self.lancedb_dir),
            include_labels=(self.mode == "training"),
            mode=self.mode
        )
        
        userposts_embedder = UserPostsEmbedderDuckDB(
            db_path=str(self.db_path),
            lancedb_dir=str(self.lancedb_dir),
            include_labels=(self.mode == "training"),
            mode=self.mode
        )
        
        # Create embeddings for each domain
        for embedder in [news_embedder, fundamentals_embedder, analyst_embedder, userposts_embedder]:
            try:
                embedder.run_pipeline()
            except Exception as e:
                logger.error(f"Error creating embeddings with {embedder.__class__.__name__}: {e}")

    def extract_features(self, setup_ids: List[str]) -> None:
        """Extract features using agents with similarity enhancement in prediction mode"""
        logger.info(f"Extracting features in {self.mode} mode for {len(setup_ids)} setups")
        
        for setup_id in setup_ids:
            try:
                # Extract features from each agent
                news_features = self.news_agent.process_setup(setup_id, mode=self.mode)
                fundamentals_features = self.fundamentals_agent.process_setup(setup_id, mode=self.mode)
                analyst_features = self.analyst_agent.process_setup(setup_id, mode=self.mode)
                userposts_features = self.userposts_agent.process_setup(setup_id, mode=self.mode)
                
                # In prediction mode, also get direct similarity predictions
                if self.mode == "prediction":
                    similarity_predictions = {
                        'news': self.news_agent.predict_via_similarity(setup_id, news_features),
                        'fundamentals': self.fundamentals_agent.predict_via_similarity(setup_id, fundamentals_features),
                        'analyst': self.analyst_agent.predict_via_similarity(setup_id, analyst_features),
                        'userposts': self.userposts_agent.predict_via_similarity(setup_id, userposts_features)
                    }
                    
                    # Store similarity predictions in a separate table
                    self._store_similarity_predictions(setup_id, similarity_predictions)
            
            except Exception as e:
                logger.error(f"Error processing setup {setup_id}: {e}")

    def _store_similarity_predictions(self, setup_id: str, predictions: Dict[str, Dict]) -> None:
        """Store similarity-based predictions in DuckDB"""
        try:
            # Flatten predictions for storage
            flat_predictions = {
                'setup_id': setup_id,
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            for domain, pred in predictions.items():
                for key, value in pred.items():
                    flat_predictions[f"{domain}_{key}"] = value
            
            # Create table if not exists
            with duckdb.connect(str(self.db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS similarity_predictions (
                        setup_id VARCHAR,
                        extraction_timestamp TIMESTAMP,
                        news_predicted_outperformance DOUBLE,
                        news_confidence DOUBLE,
                        news_positive_ratio DOUBLE,
                        news_negative_ratio DOUBLE,
                        news_neutral_ratio DOUBLE,
                        news_similar_cases_count INTEGER,
                        fundamentals_predicted_outperformance DOUBLE,
                        fundamentals_confidence DOUBLE,
                        fundamentals_positive_ratio DOUBLE,
                        fundamentals_negative_ratio DOUBLE,
                        fundamentals_neutral_ratio DOUBLE,
                        fundamentals_similar_cases_count INTEGER,
                        analyst_predicted_outperformance DOUBLE,
                        analyst_confidence DOUBLE,
                        analyst_positive_ratio DOUBLE,
                        analyst_negative_ratio DOUBLE,
                        analyst_neutral_ratio DOUBLE,
                        analyst_similar_cases_count INTEGER,
                        userposts_predicted_outperformance DOUBLE,
                        userposts_confidence DOUBLE,
                        userposts_positive_ratio DOUBLE,
                        userposts_negative_ratio DOUBLE,
                        userposts_neutral_ratio DOUBLE,
                        userposts_similar_cases_count INTEGER
                    )
                """)
                
                # Insert predictions
                placeholders = ', '.join(['?' for _ in flat_predictions])
                columns = ', '.join(flat_predictions.keys())
                values = tuple(flat_predictions.values())
                
                conn.execute(f"""
                    INSERT INTO similarity_predictions ({columns})
                    VALUES ({placeholders})
                """, values)
                
        except Exception as e:
            logger.error(f"Error storing similarity predictions for setup {setup_id}: {e}")

    def run_pipeline(self, setup_ids: List[str]) -> None:
        """Run complete ML pipeline with training/prediction mode separation"""
        logger.info(f"Running ML pipeline in {self.mode} mode")
        logger.info(f"Processing {len(setup_ids)} setups")
        
        # Step 1: Create embeddings (with or without labels based on mode)
        self.create_embeddings(setup_ids)
        
        # Step 2: Extract features (enhanced with similarity in prediction mode)
        self.extract_features(setup_ids)
        
        # Step 3: Merge features into final tables
        table_suffix = "_training" if self.mode == "training" else "_prediction"
        self.merge_features(setup_ids, table_suffix)
        
        logger.info(f"Pipeline completed in {self.mode} mode") 