import logging
from typing import Dict, List, Optional
import duckdb
from datetime import datetime

logger = logging.getLogger(__name__)

class MLFeatureMerger:
    """Merges individual feature tables into comprehensive ML feature tables"""
    
    def __init__(self, 
                 db_path: str = "data/sentiment_system.duckdb",
                 prediction_db_path: str = "data/prediction_features.duckdb"):
        self.db_path = db_path
        self.prediction_db_path = prediction_db_path
    
    def merge_text_features(self, setup_ids: List[str], mode: str = 'training') -> Dict[str, int]:
        """
        Merge all text-based features (news, userposts, analyst) into a single ML features table
        
        Args:
            setup_ids: List of setup_ids to process
            mode: Either 'training' or 'prediction'
            
        Returns:
            Dict with feature and row counts
        """
        logger.info("🔄 Merging text-based features...")
        
        try:
            # Always use the main database for reading features
            conn = duckdb.connect(self.db_path)
            
            # Check if required tables exist
            required_tables = ['news_features', 'userposts_features', 'analyst_recommendations_features']
            existing_tables = []
            missing_tables = []
            
            for table in required_tables:
                table_exists = conn.execute(f"""
                    SELECT count(*) FROM information_schema.tables 
                    WHERE table_name = '{table}'
                """).fetchone()[0]
                
                if table_exists:
                    existing_tables.append(table)
                else:
                    missing_tables.append(table)
            
            if missing_tables:
                logger.warning(f"Missing tables for text feature merging: {missing_tables}")
                if not existing_tables:
                    logger.error("No text feature tables available, skipping merge")
                    return {
                        "table_name": f"text_ml_features_{mode}",
                        "feature_count": 0,
                        "row_count": 0,
                        "error": f"Missing tables: {missing_tables}"
                    }
            
            # Modify the query based on available tables
            from_clause_parts = []
            join_clauses = []
            
            if 'news_features' in existing_tables:
                from_clause_parts.append("news_features n")
                
                if 'userposts_features' in existing_tables:
                    join_clauses.append("FULL OUTER JOIN userposts_features u ON n.setup_id = u.setup_id")
                    
                if 'analyst_recommendations_features' in existing_tables:
                    if 'userposts_features' in existing_tables:
                        join_clauses.append("FULL OUTER JOIN analyst_recommendations_features a ON COALESCE(n.setup_id, u.setup_id) = a.setup_id")
                    else:
                        join_clauses.append("FULL OUTER JOIN analyst_recommendations_features a ON n.setup_id = a.setup_id")
            elif 'userposts_features' in existing_tables:
                from_clause_parts.append("userposts_features u")
                
                if 'analyst_recommendations_features' in existing_tables:
                    join_clauses.append("FULL OUTER JOIN analyst_recommendations_features a ON u.setup_id = a.setup_id")
            elif 'analyst_recommendations_features' in existing_tables:
                from_clause_parts.append("analyst_recommendations_features a")
            
            # If no tables exist, return early
            if not from_clause_parts:
                logger.error("No text feature tables available, skipping merge")
                return {
                    "table_name": f"text_ml_features_{mode}",
                    "feature_count": 0,
                    "row_count": 0,
                    "error": f"Missing tables: {missing_tables}"
                }
            
            # Build setup_id coalesce based on available tables
            setup_id_parts = []
            if 'news_features' in existing_tables:
                setup_id_parts.append("n.setup_id")
            if 'userposts_features' in existing_tables:
                setup_id_parts.append("u.setup_id")
            if 'analyst_recommendations_features' in existing_tables:
                setup_id_parts.append("a.setup_id")
            
            setup_id_coalesce = "COALESCE(" + ", ".join(setup_id_parts) + ")"
            
            # Build feature columns based on available tables
            feature_columns = [f"{setup_id_coalesce} as setup_id"]
            
            # News features
            if 'news_features' in existing_tables:
                feature_columns.extend([
                    "n.count_financial_results",
                    "n.max_severity_financial_results",
                    "n.sentiment_score_financial_results",
                    "n.profit_warning_present::INTEGER as profit_warning",
                    "n.count_corporate_actions",
                    "n.max_severity_corporate_actions",
                    "n.sentiment_score_corporate_actions",
                    "n.capital_raise_present::INTEGER as capital_raise",
                    "n.count_governance",
                    "n.max_severity_governance",
                    "n.sentiment_score_governance",
                    "n.board_change_present::INTEGER as board_change",
                    "n.count_corporate_events",
                    "n.max_severity_corporate_events",
                    "n.sentiment_score_corporate_events",
                    "n.contract_award_present::INTEGER as contract_award",
                    "n.merger_or_acquisition_present::INTEGER as merger_acquisition",
                    "n.count_other_signals",
                    "n.max_severity_other_signals",
                    "n.sentiment_score_other_signals",
                    "n.broker_recommendation_present::INTEGER as broker_recommendation",
                    "n.credit_rating_change_present::INTEGER as credit_rating_change"
                ])
            
            # User Posts features
            if 'userposts_features' in existing_tables:
                feature_columns.extend([
                    "u.avg_sentiment as posts_avg_sentiment",
                    "u.post_count",
                    "u.community_sentiment_score",
                    "u.bull_bear_ratio",
                    "u.rumor_intensity",
                    "u.trusted_user_sentiment",
                    "u.relevance_score as posts_relevance",
                    "u.engagement_score",
                    "u.unique_users",
                    "CAST(u.contrarian_signal AS INTEGER) as contrarian_signal_numeric"
                ])
            
            # Analyst features
            if 'analyst_recommendations_features' in existing_tables:
                feature_columns.extend([
                    "a.recommendation_count",
                    "a.buy_recommendations",
                    "a.sell_recommendations",
                    "a.hold_recommendations",
                    "a.avg_price_target",
                    "a.price_target_vs_current",
                    "a.price_target_spread",
                    "a.coverage_breadth",
                    "a.consensus_rating",
                    "a.recent_upgrades",
                    "a.recent_downgrades",
                    "a.analyst_conviction_score"
                ])
            
            # Build comprehensive text ML features query
            text_features_query = f"""
            WITH base_features AS (
                SELECT DISTINCT
                    {", ".join(feature_columns)}
                FROM {from_clause_parts[0]}
                {" ".join(join_clauses)}
                WHERE {setup_id_coalesce} = ANY(?)
            )
            SELECT 
                f.*,
                CASE 
                    WHEN '{mode}' = 'training' THEN l.outperformance_10d 
                    ELSE NULL 
                END as label
            FROM base_features f
            LEFT JOIN labels l ON f.setup_id = l.setup_id
            """
            
            # Create or replace the table
            table_name = f"text_ml_features_{mode}"
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"CREATE TABLE {table_name} AS {text_features_query}", [setup_ids])
            
            # Get table info
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            feature_count = len(columns)
            
            logger.info(f"✅ Text ML features merged:")
            logger.info(f"- Table: {table_name}")
            logger.info(f"- Features: {feature_count}")
            logger.info(f"- Rows: {row_count}")
            
            return {
                "table_name": table_name,
                "feature_count": feature_count,
                "row_count": row_count
            }
            
        except Exception as e:
            logger.error(f"❌ Error merging text features: {str(e)}")
            raise
        finally:
            conn.close()
    
    def merge_financial_features(self, setup_ids: List[str], mode: str = 'training') -> Dict[str, int]:
        """
        Merge all financial features into a single ML features table
        
        Args:
            setup_ids: List of setup_ids to process
            mode: Either 'training' or 'prediction'
            
        Returns:
            Dict with feature and row counts
        """
        logger.info("🔄 Merging financial features...")
        
        try:
            # Always use the main database for reading features
            conn = duckdb.connect(self.db_path)
            
            # Check if required tables exist
            required_tables = ['setups', 'fundamentals', 'financial_ratios', 'company_info', 'fundamentals_features']
            existing_tables = []
            missing_tables = []
            
            for table in required_tables:
                table_exists = conn.execute(f"""
                    SELECT count(*) FROM information_schema.tables 
                    WHERE table_name = '{table}'
                """).fetchone()[0]
                
                if table_exists:
                    existing_tables.append(table)
                else:
                    missing_tables.append(table)
            
            if 'setups' not in existing_tables:
                logger.error("Required 'setups' table is missing, cannot merge financial features")
                return {
                    "table_name": f"financial_ml_features_{mode}",
                    "feature_count": 0,
                    "row_count": 0,
                    "error": f"Missing required 'setups' table"
                }
                
            # Create a simplified query if some tables are missing
            if missing_tables:
                logger.warning(f"Missing tables for financial feature merging: {missing_tables}")
                
                # Create a simple query with just the setup_ids
                simple_query = f"""
                SELECT 
                    s.setup_id
                FROM setups s
                WHERE s.setup_id = ANY(?)
                """
                
                # Create or replace the table
                table_name = f"financial_ml_features_{mode}"
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.execute(f"CREATE TABLE {table_name} AS {simple_query}", [setup_ids])
                
                # Get table info
                row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                
                logger.info(f"✅ Created simplified financial ML features table due to missing tables:")
                logger.info(f"- Table: {table_name}")
                logger.info(f"- Features: 1 (setup_id only)")
                logger.info(f"- Rows: {row_count}")
                
                return {
                    "table_name": table_name,
                    "feature_count": 1,
                    "row_count": row_count,
                    "warning": f"Simplified table created due to missing tables: {missing_tables}"
                }
            
            # If all tables exist, proceed with the full query
            financial_features_query = """
            WITH 
            -- Get latest fundamentals data for each setup
            latest_fundamentals AS (
                SELECT 
                    s.setup_id,
                    s.lse_ticker,
                    s.spike_timestamp,
                    f.*,
                    ROW_NUMBER() OVER (PARTITION BY s.setup_id ORDER BY f.date DESC) as row_num
                FROM setups s
                LEFT JOIN fundamentals f 
                    ON s.lse_ticker = f.ticker 
                    AND f.date <= s.spike_timestamp
                WHERE s.setup_id = ANY(?)
            ),
            fundamentals_clean AS (
                SELECT setup_id, lse_ticker, spike_timestamp,
                       total_revenue, gross_profit, operating_income, net_income,
                       ebitda, operating_cash_flow, free_cash_flow, total_assets,
                       total_debt, total_equity, current_assets, current_liabilities,
                       working_capital
                FROM latest_fundamentals
                WHERE row_num = 1
            ),
            -- Get latest financial ratios
            latest_ratios AS (
                SELECT 
                    s.setup_id,
                    fr.*,
                    ROW_NUMBER() OVER (PARTITION BY s.setup_id ORDER BY fr.period_end DESC) as row_num
                FROM setups s
                LEFT JOIN financial_ratios fr 
                    ON s.lse_ticker = fr.ticker
                    AND fr.period_end <= s.spike_timestamp
                WHERE s.setup_id = ANY(?)
            ),
            ratios_clean AS (
                SELECT setup_id,
                       current_ratio, quick_ratio, cash_ratio, debt_to_equity,
                       debt_to_assets, equity_ratio, gross_margin, operating_margin,
                       net_margin, roe, roa, roic, asset_turnover, inventory_turnover,
                       receivables_turnover, pe_ratio as price_to_earnings,
                       pb_ratio as price_to_book, ev_ebitda
                FROM latest_ratios
                WHERE row_num = 1
            ),
            -- Get previous period data for growth calculations
            prev_fundamentals AS (
                SELECT 
                    s.setup_id,
                    f.total_revenue as prev_revenue,
                    f.net_income as prev_net_income,
                    f.operating_cash_flow as prev_operating_cash_flow,
                    f.ebitda as prev_ebitda,
                    ROW_NUMBER() OVER (PARTITION BY s.setup_id ORDER BY f.date DESC) as row_num
                FROM setups s
                LEFT JOIN fundamentals f 
                    ON s.lse_ticker = f.ticker 
                    AND f.date <= s.spike_timestamp - INTERVAL '1 year'
                WHERE s.setup_id = ANY(?)
            ),
            prev_clean AS (
                SELECT setup_id, prev_revenue, prev_net_income,
                       prev_operating_cash_flow, prev_ebitda
                FROM prev_fundamentals
                WHERE row_num = 1
            ),
            -- Calculate growth metrics
            growth_metrics AS (
                SELECT 
                    f.setup_id,
                    CASE 
                        WHEN p.prev_revenue > 0 THEN (f.total_revenue - p.prev_revenue) / p.prev_revenue 
                        ELSE NULL 
                    END as revenue_growth_yoy,
                    CASE 
                        WHEN p.prev_net_income > 0 THEN (f.net_income - p.prev_net_income) / p.prev_net_income 
                        ELSE NULL 
                    END as net_income_growth_yoy,
                    CASE 
                        WHEN p.prev_operating_cash_flow > 0 THEN (f.operating_cash_flow - p.prev_operating_cash_flow) / p.prev_operating_cash_flow 
                        ELSE NULL 
                    END as operating_cash_flow_growth_yoy,
                    CASE 
                        WHEN p.prev_ebitda > 0 THEN (f.ebitda - p.prev_ebitda) / p.prev_ebitda 
                        ELSE NULL 
                    END as ebitda_growth_yoy
                FROM fundamentals_clean f
                LEFT JOIN prev_clean p USING (setup_id)
            ),
            -- Calculate per share metrics
            per_share_metrics AS (
                SELECT 
                    f.setup_id,
                    CASE 
                        WHEN ci.shares_outstanding > 0 THEN f.total_revenue / ci.shares_outstanding 
                        ELSE NULL 
                    END as revenue_per_share,
                    CASE 
                        WHEN ci.shares_outstanding > 0 THEN f.operating_cash_flow / ci.shares_outstanding 
                        ELSE NULL 
                    END as operating_cash_flow_per_share,
                    CASE 
                        WHEN ci.shares_outstanding > 0 THEN f.net_income / ci.shares_outstanding 
                        ELSE NULL 
                    END as earnings_per_share,
                    CASE 
                        WHEN ci.shares_outstanding > 0 THEN f.total_equity / ci.shares_outstanding 
                        ELSE NULL 
                    END as book_value_per_share
                FROM fundamentals_clean f
                LEFT JOIN company_info ci ON f.lse_ticker = ci.ticker
            ),
            -- Combine all features
            base_features AS (
                SELECT 
                    COALESCE(f.setup_id, r.setup_id) as setup_id,
                    
                    -- Raw fundamentals
                    f.total_revenue,
                    f.gross_profit,
                    f.operating_income,
                    f.net_income,
                    f.ebitda,
                    f.operating_cash_flow,
                    f.free_cash_flow,
                    f.total_assets,
                    f.total_debt,
                    f.total_equity,
                    f.current_assets,
                    f.current_liabilities,
                    f.working_capital,
                    
                    -- Financial ratios
                    r.current_ratio,
                    r.quick_ratio,
                    r.cash_ratio,
                    r.debt_to_equity,
                    r.debt_to_assets,
                    r.equity_ratio,
                    r.gross_margin,
                    r.operating_margin,
                    r.net_margin,
                    r.roe,
                    r.roa,
                    r.roic,
                    r.asset_turnover,
                    r.inventory_turnover,
                    r.receivables_turnover,
                    r.price_to_earnings,
                    r.price_to_book,
                    r.ev_ebitda,
                    
                    -- Per share metrics
                    ps.revenue_per_share,
                    ps.operating_cash_flow_per_share,
                    ps.earnings_per_share,
                    ps.book_value_per_share,
                    
                    -- Growth metrics
                    g.revenue_growth_yoy,
                    g.net_income_growth_yoy,
                    g.operating_cash_flow_growth_yoy,
                    g.ebitda_growth_yoy,
                    
                    -- Features from fundamentals_features table
                    ff.*
                    
                FROM fundamentals_clean f
                FULL OUTER JOIN ratios_clean r USING (setup_id)
                LEFT JOIN per_share_metrics ps USING (setup_id)
                LEFT JOIN growth_metrics g USING (setup_id)
                LEFT JOIN fundamentals_features ff USING (setup_id)
                WHERE COALESCE(f.setup_id, r.setup_id) = ANY(?)
            )
            SELECT 
                f.*,
                CASE 
                    WHEN '{mode}' = 'training' THEN l.outperformance_10d 
                    ELSE NULL 
                END as label
            FROM base_features f
            LEFT JOIN labels l ON f.setup_id = l.setup_id
            """
            
            # Create or replace the table
            table_name = f"financial_ml_features_{mode}"
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"CREATE TABLE {table_name} AS {financial_features_query}", [setup_ids, setup_ids, setup_ids, setup_ids])
            
            # Get table info
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            feature_count = len(columns)
            
            logger.info(f"✅ Financial ML features merged:")
            logger.info(f"- Table: {table_name}")
            logger.info(f"- Features: {feature_count}")
            logger.info(f"- Rows: {row_count}")
            
            # Log feature names
            logger.info("\nFinancial feature categories:")
            logger.info("1. Raw Fundamentals (13 features)")
            logger.info("2. Financial Ratios (19 features)")
            logger.info("3. Per Share Metrics (4 features)")
            logger.info("4. Growth Metrics (4 features)")
            logger.info("5. Processed Features (from fundamentals_features table)")
            
            return {
                "table_name": table_name,
                "feature_count": feature_count,
                "row_count": row_count
            }
            
        except Exception as e:
            logger.error(f"❌ Error merging financial features: {str(e)}")
            raise
        finally:
            conn.close()
    
    def merge_all_features(self, setup_ids: List[str], mode: str = 'training') -> Dict[str, Dict[str, int]]:
        """
        Merge all features (text and financial) into ML feature tables
        
        Args:
            setup_ids: List of setup_ids to process
            mode: Either 'training' or 'prediction'
            
        Returns:
            Dict with results for each feature type
        """
        results = {}
        
        # Merge text features
        try:
            results['text'] = self.merge_text_features(setup_ids, mode)
        except Exception as e:
            logger.error(f"Failed to merge text features: {str(e)}")
            results['text'] = {"error": str(e)}
        
        # Merge financial features
        try:
            results['financial'] = self.merge_financial_features(setup_ids, mode)
        except Exception as e:
            logger.error(f"Failed to merge financial features: {str(e)}")
            results['financial'] = {"error": str(e)}
        
        return results 