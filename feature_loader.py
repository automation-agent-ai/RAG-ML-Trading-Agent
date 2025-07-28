"""
Feature loader class for text and financial data.
"""

import numpy as np
import pandas as pd
import duckdb
from typing import List, Optional, Tuple, Union

from .setup_loader import SetupLoader

class FeatureLoader:
    """Handles loading and preprocessing of text and financial features"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.setup_loader = SetupLoader(db_path)
        self.source_tables = {
            'news': 'news_features',
            'userposts': 'userposts_features',
            'analyst': 'analyst_recommendations_features'
        }
    
    def load_text_features(
        self,
        setup_ids: Optional[Union[str, List[str]]] = None,
        setup_ids_file: Optional[str] = None,
        is_training: bool = False
    ) -> Tuple[pd.DataFrame, List[str], Optional[pd.Series]]:
        """
        Load text features for training or specific setup_ids.
        
        Args:
            setup_ids: Single setup ID or list of setup IDs
            setup_ids_file: Path to file containing setup IDs (one per line)
            is_training: If True, get setup IDs marked for training in database
            
        Returns:
            Tuple of (features DataFrame, feature column names, target labels if training)
        """
        # Load and validate setup IDs
        valid_ids = self.setup_loader.load_setup_ids(
            setup_ids=setup_ids,
            setup_ids_file=setup_ids_file,
            is_training=is_training
        )
        if not valid_ids:
            return None, None, None
            
        print(f"   📊 Found {len(valid_ids)} valid setup_ids")
        
        conn = duckdb.connect(self.db_path)
        try:
            # Load data from each source
            setup_ids_str = "'" + "','".join(str(id) for id in valid_ids) + "'"
            source_dfs = {}
            
            # Load target labels if training
            y = None
            if is_training:
                y = self._get_labels(conn, valid_ids)
                if y is None:
                    print("   ⚠️ Warning: No labels found for training")
                    return None, None, None
            
            # Create empty DataFrame with all setup IDs
            merged_df = pd.DataFrame(index=pd.Index(valid_ids, name='setup_id'))
            
            for source, table in self.source_tables.items():
                # Special handling for analyst recommendations to include changes
                if source == 'analyst':
                    query = f"""
                    WITH current_recs AS (
                        SELECT 
                            setup_id,
                            consensus_rating, price_target_vs_current, recommendation_count,
                            buy_recommendations, hold_recommendations, sell_recommendations,
                            analyst_conviction_score, recent_upgrades, recent_downgrades,
                            ROW_NUMBER() OVER (PARTITION BY setup_id ORDER BY extraction_timestamp DESC) as rn
                        FROM {table}
                        WHERE setup_id IN ({setup_ids_str})
                    ),
                    prev_recs AS (
                        SELECT 
                            setup_id,
                            consensus_rating as prev_consensus_rating,
                            price_target_vs_current as prev_price_target,
                            recommendation_count as prev_rec_count,
                            buy_recommendations as prev_buy_recs,
                            hold_recommendations as prev_hold_recs,
                            sell_recommendations as prev_sell_recs,
                            analyst_conviction_score as prev_conviction,
                            ROW_NUMBER() OVER (PARTITION BY setup_id ORDER BY extraction_timestamp DESC) as rn
                        FROM {table}
                        WHERE setup_id IN ({setup_ids_str})
                    )
                    SELECT 
                        c.setup_id,
                        c.consensus_rating, c.price_target_vs_current, c.recommendation_count,
                        c.buy_recommendations, c.hold_recommendations, c.sell_recommendations,
                        c.analyst_conviction_score, c.recent_upgrades, c.recent_downgrades,
                        -- Changes in key metrics
                        COALESCE(c.consensus_rating - p.prev_consensus_rating, 0) as consensus_rating_change,
                        COALESCE(c.price_target_vs_current - p.prev_price_target, 0) as price_target_change,
                        COALESCE(c.recommendation_count - p.prev_rec_count, 0) as rec_count_change,
                        COALESCE(c.buy_recommendations - p.prev_buy_recs, 0) as buy_recs_change,
                        COALESCE(c.hold_recommendations - p.prev_hold_recs, 0) as hold_recs_change,
                        COALESCE(c.sell_recommendations - p.prev_sell_recs, 0) as sell_recs_change,
                        COALESCE(c.analyst_conviction_score - p.prev_conviction, 0) as conviction_change
                    FROM current_recs c
                    LEFT JOIN prev_recs p ON c.setup_id = p.setup_id AND p.rn = 2
                    WHERE c.rn = 1
                    """
                else:
                    query = f"""
                    WITH latest_data AS (
                        SELECT *,
                            ROW_NUMBER() OVER (PARTITION BY setup_id ORDER BY extraction_timestamp DESC) as rn
                        FROM {table}
                        WHERE setup_id IN ({setup_ids_str})
                    )
                    SELECT setup_id, *
                    FROM latest_data
                    WHERE rn = 1
                    """
                
                df = conn.execute(query).df()
                
                if not df.empty:
                    print(f"   ✅ {source.title()} features: {len(df)} rows")
                    # Check for duplicates
                    duplicates = df['setup_id'].duplicated()
                    if duplicates.any():
                        print(f"   ⚠️ Warning: Found {duplicates.sum()} duplicate setup IDs in {source}")
                        print(f"   ⚠️ Duplicate IDs: {df[duplicates]['setup_id'].tolist()}")
                        # Keep only the first occurrence
                        df = df[~df['setup_id'].duplicated()]
                    
                    # Add source suffix to feature names
                    df.columns = [col if col == 'setup_id' else f"{col}_{source}" 
                                for col in df.columns]
                    # Convert to numeric
                    df = self._convert_to_numeric(df)
                    
                    # Add features to merged_df
                    feature_cols = [col for col in df.columns if col != 'setup_id']
                    for col in feature_cols:
                        merged_df[col] = 0.0  # Initialize with 0
                    
                    # Create a mapping of setup_id to values
                    value_map = {}
                    for _, row in df.iterrows():
                        setup_id = row['setup_id']
                        if setup_id in merged_df.index:
                            value_map[setup_id] = {col: row[col] for col in feature_cols}
                    
                    # Update merged_df using the mapping
                    for setup_id, values in value_map.items():
                        for col, val in values.items():
                            merged_df.loc[setup_id, col] = val
            
            # Fill missing values with 0
            merged_df.fillna(0.0, inplace=True)
            
            print(f"   📊 Text features shape: {merged_df.shape}")
            print(f"   📊 Missing values: {merged_df.isna().sum().sum()}")
            
            return merged_df, merged_df.columns.tolist(), y
            
        except Exception as e:
            print(f"Error loading text features: {str(e)}")
            return None, None, None
        finally:
            conn.close()
    
    def load_financial_features(
        self,
        setup_ids: Optional[Union[str, List[str]]] = None,
        setup_ids_file: Optional[str] = None,
        is_training: bool = False
    ) -> Tuple[pd.DataFrame, List[str], Optional[pd.Series]]:
        """
        Load financial features for training or specific setup_ids.
        
        Args:
            setup_ids: Single setup ID or list of setup IDs
            setup_ids_file: Path to file containing setup IDs (one per line)
            is_training: If True, get setup IDs marked for training in database
            
        Returns:
            Tuple of (features DataFrame, feature column names, target labels if training)
        """
        # Load and validate setup IDs
        valid_ids = self.setup_loader.load_setup_ids(
            setup_ids=setup_ids,
            setup_ids_file=setup_ids_file,
            is_training=is_training
        )
        if not valid_ids:
            return None, None, None
            
        print(f"   📊 Found {len(valid_ids)} valid setup_ids")
        
        conn = duckdb.connect(self.db_path)
        try:
            # Load target labels if training
            y = None
            if is_training:
                y = self._get_labels(conn, valid_ids)
                if y is None:
                    print("   ⚠️ Warning: No labels found for training")
                    return None, None, None
            
            # Load financial data with growth ratios
            setup_ids_str = "'" + "','".join(str(id) for id in valid_ids) + "'"
            query = f"""
            WITH latest_ratios AS (
                SELECT 
                    s.setup_id,
                    fr.*,
                    ROW_NUMBER() OVER (PARTITION BY s.setup_id ORDER BY fr.period_end DESC NULLS LAST) as rn
                FROM setups s
                LEFT JOIN financial_ratios fr ON CONCAT(s.lse_ticker, '.L') = fr.ticker
                    AND fr.period_end <= s.spike_timestamp
                WHERE s.setup_id IN ({setup_ids_str})
            ),
            latest_fundamentals AS (
                SELECT 
                    s.setup_id,
                    f.*,
                    ROW_NUMBER() OVER (PARTITION BY s.setup_id ORDER BY f.date DESC NULLS LAST) as rn
                FROM setups s
                LEFT JOIN fundamentals f ON CONCAT(s.lse_ticker, '.L') = f.ticker
                    AND f.date <= s.spike_timestamp
                WHERE s.setup_id IN ({setup_ids_str})
            )
            SELECT 
                r.setup_id,
                -- Raw values for ratio calculations
                CAST(f.total_revenue AS DOUBLE) as total_revenue,
                CAST(f.gross_profit AS DOUBLE) as gross_profit,
                CAST(f.operating_income AS DOUBLE) as operating_income,
                CAST(f.net_income AS DOUBLE) as net_income,
                CAST(f.total_assets AS DOUBLE) as total_assets,
                CAST(f.total_equity AS DOUBLE) as total_equity,
                CAST(f.operating_cash_flow AS DOUBLE) as operating_cash_flow,
                CAST(f.free_cash_flow AS DOUBLE) as free_cash_flow,
                CAST(f.total_debt AS DOUBLE) as total_debt,
                CAST(f.cash_and_equivalents AS DOUBLE) as cash_and_equivalents,
                CAST(f.current_assets AS DOUBLE) as current_assets,
                CAST(f.current_liabilities AS DOUBLE) as current_liabilities,
                CAST(f.working_capital AS DOUBLE) as working_capital,
                CAST(f.property_plant_equipment AS DOUBLE) as property_plant_equipment,
                CAST(f.capital_expenditure AS DOUBLE) as capital_expenditure,
                CAST(f.financing_cash_flow AS DOUBLE) as financing_cash_flow,
                CAST(f.investing_cash_flow AS DOUBLE) as investing_cash_flow,
                -- Previous period values
                LAG(CAST(f.total_revenue AS DOUBLE)) OVER (PARTITION BY s.lse_ticker ORDER BY f.date) as prev_total_revenue,
                LAG(CAST(f.gross_profit AS DOUBLE)) OVER (PARTITION BY s.lse_ticker ORDER BY f.date) as prev_gross_profit,
                LAG(CAST(f.operating_income AS DOUBLE)) OVER (PARTITION BY s.lse_ticker ORDER BY f.date) as prev_operating_income,
                LAG(CAST(f.net_income AS DOUBLE)) OVER (PARTITION BY s.lse_ticker ORDER BY f.date) as prev_net_income,
                LAG(CAST(f.total_assets AS DOUBLE)) OVER (PARTITION BY s.lse_ticker ORDER BY f.date) as prev_total_assets,
                LAG(CAST(f.total_equity AS DOUBLE)) OVER (PARTITION BY s.lse_ticker ORDER BY f.date) as prev_total_equity,
                LAG(CAST(f.operating_cash_flow AS DOUBLE)) OVER (PARTITION BY s.lse_ticker ORDER BY f.date) as prev_operating_cash_flow,
                LAG(CAST(f.free_cash_flow AS DOUBLE)) OVER (PARTITION BY s.lse_ticker ORDER BY f.date) as prev_free_cash_flow
            FROM latest_ratios r
            LEFT JOIN latest_fundamentals f ON r.setup_id = f.setup_id
            LEFT JOIN setups s ON r.setup_id = s.setup_id
            WHERE r.rn = 1 AND f.rn = 1
            """
            
            df = conn.execute(query).df()
            
            if df.empty:
                print("   ❌ No financial features found")
                return None, None, None
            
            print(f"   ✅ Financial features: {len(df)} rows")
            
            # Set setup_id as index
            df.set_index('setup_id', inplace=True)
            
            # Fill missing values with median
            df.fillna(df.median(), inplace=True)
            
            print(f"   📊 Features shape: {df.shape}")
            print(f"   📊 Missing values: {df.isna().sum().sum()}")
            
            return df, df.columns.tolist(), y
            
        except Exception as e:
            print(f"Error loading financial features: {str(e)}")
            return None, None, None
        finally:
            conn.close()

    def _get_labels(self, conn, setup_ids: List[str]) -> Optional[pd.Series]:
        """Get labels for setup IDs"""
        setup_ids_str = "'" + "','".join(str(id) for id in setup_ids) + "'"
        query = f"""
        SELECT setup_id,
               CASE
                   WHEN outperformance_10d > 0.05 THEN 2  -- Positive
                   WHEN outperformance_10d < -0.05 THEN 0  -- Negative
                   ELSE 1  -- Neutral
               END as label
        FROM labels
        WHERE setup_id IN ({setup_ids_str})
        """
        
        labels_df = conn.execute(query).df()
        if not labels_df.empty:
            labels_df.set_index('setup_id', inplace=True)
            return labels_df['label']
        return None

    def _convert_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert string columns to numeric"""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert strings to numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    # If conversion fails, drop the column
                    print(f"   ⚠️ Warning: Dropping non-numeric column {col}")
                    df.drop(columns=[col], inplace=True)
        return df 