#!/usr/bin/env python3
"""
Simple Feature Loader for ML Pipeline

Loads text and financial features from DuckDB database for ML training.
"""

import numpy as np
import pandas as pd
import duckdb
from typing import List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleFeatureLoader:
    """Simple feature loader for text and financial data"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    def load_text_features(self, setup_ids: Optional[List[str]] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load text features from the database
        
        Args:
            setup_ids: List of setup IDs to load. If None, loads all available.
            
        Returns:
            Tuple of (features_df, labels_array)
        """
        conn = duckdb.connect(self.db_path)
        
        try:
            # Get setup IDs with labels if not provided
            if setup_ids is None:
                setup_ids_query = """
                SELECT DISTINCT setup_id 
                FROM labels 
                WHERE outperformance_10d IS NOT NULL
                """
                setup_ids_df = conn.execute(setup_ids_query).df()
                setup_ids = setup_ids_df['setup_id'].tolist()
                
            if not setup_ids:
                logger.error("No setup IDs found")
                return None, None
                
            logger.info(f"Loading text features for {len(setup_ids)} setup IDs")
            
            # Convert setup_ids to string for SQL query
            setup_ids_str = "'" + "','".join(str(id) for id in setup_ids) + "'"
            
            # Load features from each text source
            all_features = pd.DataFrame(index=pd.Index(setup_ids, name='setup_id'))
            
            # News features
            try:
                news_query = f"""
                SELECT setup_id,
                       sentiment_score_financial_results,
                       sentiment_score_corporate_actions,
                       sentiment_score_governance,
                       count_financial_results,
                       count_corporate_actions,
                       max_severity_financial_results
                FROM news_features 
                WHERE setup_id IN ({setup_ids_str})
                """
                news_df = conn.execute(news_query).df()
                if not news_df.empty:
                    news_df.set_index('setup_id', inplace=True)
                    # Add prefix
                    news_df.columns = [f'news_{col}' for col in news_df.columns]
                    all_features = all_features.join(news_df, how='left')
                    logger.info(f"Loaded news features: {len(news_df)} rows, {len(news_df.columns)} columns")
            except Exception as e:
                logger.warning(f"Error loading news features: {e}")
                
            # User posts features
            try:
                userposts_query = f"""
                SELECT setup_id,
                       avg_sentiment,
                       post_count,
                       community_sentiment_score,
                       bull_bear_ratio,
                       contrarian_signal,
                       consensus_level
                FROM userposts_features 
                WHERE setup_id IN ({setup_ids_str})
                """
                userposts_df = conn.execute(userposts_query).df()
                if not userposts_df.empty:
                    userposts_df.set_index('setup_id', inplace=True)
                    # Convert contrarian_signal to numeric
                    if 'contrarian_signal' in userposts_df.columns:
                        userposts_df['contrarian_signal'] = pd.to_numeric(userposts_df['contrarian_signal'], errors='coerce')
                    # Add prefix
                    userposts_df.columns = [f'userposts_{col}' for col in userposts_df.columns]
                    all_features = all_features.join(userposts_df, how='left')
                    logger.info(f"Loaded userposts features: {len(userposts_df)} rows, {len(userposts_df.columns)} columns")
            except Exception as e:
                logger.warning(f"Error loading userposts features: {e}")
                
            # Analyst recommendations features
            try:
                analyst_query = f"""
                SELECT setup_id,
                       recommendation_count,
                       buy_recommendations,
                       sell_recommendations,
                       hold_recommendations,
                       consensus_rating,
                       analyst_conviction_score
                FROM analyst_recommendations_features 
                WHERE setup_id IN ({setup_ids_str})
                """
                analyst_df = conn.execute(analyst_query).df()
                if not analyst_df.empty:
                    analyst_df.set_index('setup_id', inplace=True)
                    # Add prefix
                    analyst_df.columns = [f'analyst_{col}' for col in analyst_df.columns]
                    all_features = all_features.join(analyst_df, how='left')
                    logger.info(f"Loaded analyst features: {len(analyst_df)} rows, {len(analyst_df.columns)} columns")
            except Exception as e:
                logger.warning(f"Error loading analyst features: {e}")
                
            # Load labels
            labels_query = f"""
            SELECT setup_id,
                   CASE 
                       WHEN outperformance_10d > 5 THEN 2   -- Positive: >5% outperformance
                       WHEN outperformance_10d < -5 THEN 0  -- Negative: <-5% outperformance  
                       ELSE 1                               -- Neutral: between -5% and 5%
                   END as label
            FROM labels
            WHERE setup_id IN ({setup_ids_str})
            AND outperformance_10d IS NOT NULL
            """
            labels_df = conn.execute(labels_query).df()
            
            if labels_df.empty:
                logger.error("No labels found")
                return None, None
                
            labels_df.set_index('setup_id', inplace=True)
            
            # Keep only setup_ids that have both features and labels
            common_ids = all_features.index.intersection(labels_df.index)
            if len(common_ids) == 0:
                logger.error("No common setup_ids between features and labels")
                return None, None
                
            all_features = all_features.loc[common_ids]
            labels = labels_df.loc[common_ids, 'label'].values
            
                         # Convert non-numeric columns to numeric
             for col in all_features.columns:
                 all_features[col] = pd.to_numeric(all_features[col], errors='coerce')
             
             # Fill NaN values with 0
             all_features = all_features.fillna(0)
            
            logger.info(f"Final text features shape: {all_features.shape}")
            logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
            
            return all_features, labels
            
        except Exception as e:
            logger.error(f"Error loading text features: {e}")
            return None, None
        finally:
            conn.close()
            
    def load_financial_features(self, setup_ids: Optional[List[str]] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load financial features from the database
        
        Args:
            setup_ids: List of setup IDs to load. If None, loads all available.
            
        Returns:
            Tuple of (features_df, labels_array)
        """
        conn = duckdb.connect(self.db_path)
        
        try:
            # Get setup IDs with labels if not provided
            if setup_ids is None:
                setup_ids_query = """
                SELECT DISTINCT setup_id 
                FROM labels 
                WHERE outperformance_10d IS NOT NULL
                """
                setup_ids_df = conn.execute(setup_ids_query).df()
                setup_ids = setup_ids_df['setup_id'].tolist()
                
            if not setup_ids:
                logger.error("No setup IDs found")
                return None, None
                
            logger.info(f"Loading financial features for {len(setup_ids)} setup IDs")
            
            # Convert setup_ids to string for SQL query
            setup_ids_str = "'" + "','".join(str(id) for id in setup_ids) + "'"
            
            # Load financial data with growth calculations
            financial_query = f"""
            WITH latest_ratios AS (
                SELECT s.setup_id,
                       fr.current_ratio, fr.debt_to_equity, fr.roe, fr.roa,
                       fr.gross_margin, fr.operating_margin, fr.net_margin,
                       fr.pe_ratio, fr.pb_ratio, fr.asset_turnover,
                       ROW_NUMBER() OVER (PARTITION BY s.setup_id ORDER BY fr.period_end DESC) as rn
                FROM setups s
                JOIN financial_ratios fr ON CONCAT(s.lse_ticker, '.L') = fr.ticker
                WHERE s.setup_id IN ({setup_ids_str})
                AND fr.period_end <= s.spike_timestamp
            ),
            latest_fundamentals AS (
                SELECT s.setup_id,
                       f.total_revenue, f.total_assets, f.total_debt,
                       f.net_income, f.operating_cash_flow, f.free_cash_flow,
                       -- Calculate growth metrics
                       LAG(f.total_revenue) OVER (PARTITION BY s.lse_ticker ORDER BY f.date) as prev_revenue,
                       LAG(f.net_income) OVER (PARTITION BY s.lse_ticker ORDER BY f.date) as prev_net_income,
                       ROW_NUMBER() OVER (PARTITION BY s.setup_id ORDER BY f.date DESC) as rn
                FROM setups s  
                JOIN fundamentals f ON CONCAT(s.lse_ticker, '.L') = f.ticker
                WHERE s.setup_id IN ({setup_ids_str})
                AND f.date <= s.spike_timestamp
            )
            SELECT r.setup_id,
                   -- Ratios
                   r.current_ratio, r.debt_to_equity, r.roe, r.roa,
                   r.gross_margin, r.operating_margin, r.net_margin,
                   r.pe_ratio, r.pb_ratio, r.asset_turnover,
                   -- Fundamentals
                   f.total_revenue, f.total_assets, f.total_debt,
                   f.net_income, f.operating_cash_flow, f.free_cash_flow,
                   -- Growth metrics
                   CASE WHEN f.prev_revenue > 0 THEN (f.total_revenue - f.prev_revenue) / f.prev_revenue ELSE 0 END as revenue_growth,
                   CASE WHEN f.prev_net_income > 0 THEN (f.net_income - f.prev_net_income) / f.prev_net_income ELSE 0 END as net_income_growth
            FROM latest_ratios r
            JOIN latest_fundamentals f ON r.setup_id = f.setup_id
            WHERE r.rn = 1 AND f.rn = 1
            """
            
            financial_df = conn.execute(financial_query).df()
            
            if financial_df.empty:
                logger.error("No financial features found")
                return None, None
                
            financial_df.set_index('setup_id', inplace=True)
            
            # Load labels
            labels_query = f"""
            SELECT setup_id,
                   CASE 
                       WHEN outperformance_10d > 5 THEN 2   -- Positive: >5% outperformance
                       WHEN outperformance_10d < -5 THEN 0  -- Negative: <-5% outperformance  
                       ELSE 1                               -- Neutral: between -5% and 5%
                   END as label
            FROM labels
            WHERE setup_id IN ({setup_ids_str})
            AND outperformance_10d IS NOT NULL
            """
            labels_df = conn.execute(labels_query).df()
            
            if labels_df.empty:
                logger.error("No labels found")
                return None, None
                
            labels_df.set_index('setup_id', inplace=True)
            
            # Keep only setup_ids that have both features and labels
            common_ids = financial_df.index.intersection(labels_df.index)
            if len(common_ids) == 0:
                logger.error("No common setup_ids between features and labels")
                return None, None
                
            financial_df = financial_df.loc[common_ids]
            labels = labels_df.loc[common_ids, 'label'].values
            
            # Fill NaN values with 0 and clip extreme values
            financial_df = financial_df.fillna(0)
            financial_df = financial_df.clip(-1000, 1000)  # Clip extreme outliers
            
            logger.info(f"Final financial features shape: {financial_df.shape}")
            logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
            
            return financial_df, labels
            
        except Exception as e:
            logger.error(f"Error loading financial features: {e}")
            return None, None
        finally:
            conn.close()
            
    def get_common_setup_ids(self) -> List[str]:
        """Get setup IDs that have both text and financial features"""
        
        # Load both feature sets
        text_features, text_labels = self.load_text_features()
        financial_features, financial_labels = self.load_financial_features()
        
        if text_features is None or financial_features is None:
            return []
            
        # Find common setup IDs
        common_ids = list(set(text_features.index) & set(financial_features.index))
        logger.info(f"Found {len(common_ids)} setup IDs with both text and financial features")
        
        return common_ids 