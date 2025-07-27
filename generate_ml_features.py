#!/usr/bin/env python3
"""
Generate Comprehensive ML Feature Tables

This script:
1. Merges features from different domains (news, fundamentals, userposts, analyst recommendations)
2. Creates comprehensive ML feature tables for both training and prediction
3. Exports the tables to CSV files with timestamps

Usage:
    python generate_ml_features.py --mode training
    python generate_ml_features.py --mode prediction
"""

import os
import sys
import argparse
import logging
import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveMLFeatureGenerator:
    """Generates comprehensive ML feature tables by merging domain-specific features"""
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        output_dir: str = "data/ml_features"
    ):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def find_complete_setups(self) -> List[str]:
        """Find setups with data (less restrictive approach)"""
        logger.info("🔍 Finding setups with sufficient data...")
        
        try:
            conn = duckdb.connect(self.db_path)
            
            # Query to find setups with data in any domain
            query = """
            -- Get all setups with labels
            SELECT DISTINCT s.setup_id
            FROM setups s
            JOIN daily_labels l ON s.setup_id = l.setup_id
            WHERE EXISTS (
                -- Ensure we have fundamentals data for this ticker
                SELECT 1 FROM fundamentals f 
                WHERE s.lse_ticker = f.ticker 
                LIMIT 1
            )
            GROUP BY s.setup_id
            HAVING COUNT(DISTINCT l.day_number) >= 5  -- Relaxed: only need 5 days of labels
            LIMIT 5000  -- Limit to prevent processing too many at once
            """
            
            setup_ids = conn.execute(query).df()['setup_id'].tolist()
            logger.info(f"Found {len(setup_ids)} setups with sufficient data")
            
            return setup_ids
            
        except Exception as e:
            logger.error(f"Error finding setups: {e}")
            return []
        finally:
            conn.close()
    
    def merge_text_features(self, setup_ids: List[str], mode: str = 'training') -> Dict[str, Any]:
        """
        Merge all text-based features (news, userposts, analyst) into a single ML features table
        
        Args:
            setup_ids: List of setup_ids to process
            mode: Either 'training' or 'prediction'
            
        Returns:
            Dict with feature and row counts
        """
        logger.info(f"🔄 Merging text-based features for {mode} mode...")
        
        conn = None
        try:
            conn = duckdb.connect(self.db_path)
            
            # Build comprehensive text ML features query
            text_features_query = """
            WITH base_features AS (
                SELECT DISTINCT
                    s.setup_id,
                    
                    -- News features
                    n.count_financial_results,
                    n.max_severity_financial_results,
                    n.sentiment_score_financial_results,
                    n.profit_warning_present::INTEGER as profit_warning,
                    n.count_corporate_actions,
                    n.max_severity_corporate_actions,
                    n.sentiment_score_corporate_actions,
                    n.capital_raise_present::INTEGER as capital_raise,
                    n.count_governance,
                    n.max_severity_governance,
                    n.sentiment_score_governance,
                    n.board_change_present::INTEGER as board_change,
                    n.count_corporate_events,
                    n.max_severity_corporate_events,
                    n.sentiment_score_corporate_events,
                    n.contract_award_present::INTEGER as contract_award,
                    n.merger_or_acquisition_present::INTEGER as merger_acquisition,
                    n.count_other_signals,
                    n.max_severity_other_signals,
                    n.sentiment_score_other_signals,
                    n.broker_recommendation_present::INTEGER as broker_recommendation,
                    n.credit_rating_change_present::INTEGER as credit_rating_change,
                    
                    -- User Posts features
                    u.avg_sentiment as posts_avg_sentiment,
                    u.post_count,
                    u.community_sentiment_score,
                    u.bull_bear_ratio,
                    u.rumor_intensity,
                    u.trusted_user_sentiment,
                    u.relevance_score as posts_relevance,
                    u.engagement_score,
                    u.unique_users,
                    CAST(u.contrarian_signal AS INTEGER) as contrarian_signal_numeric,
                    
                    -- Analyst features
                    a.recommendation_count,
                    a.buy_recommendations,
                    a.sell_recommendations,
                    a.hold_recommendations,
                    a.coverage_breadth,
                    a.consensus_rating,
                    a.recent_upgrades,
                    a.recent_downgrades,
                    a.analyst_conviction_score
                    
                FROM setups s
                LEFT JOIN news_features n ON s.setup_id = n.setup_id
                LEFT JOIN userposts_features u ON s.setup_id = u.setup_id
                LEFT JOIN analyst_recommendations_features a ON s.setup_id = a.setup_id
                WHERE s.setup_id = ANY(?)
            )
            SELECT 
                f.*,
                CASE 
                    WHEN '{mode}' = 'training' THEN l.outperformance_10d 
                    ELSE NULL 
                END as label
            FROM base_features f
            LEFT JOIN (
                SELECT setup_id, AVG(outperformance_day) as outperformance_10d
                FROM daily_labels
                WHERE day_number <= 10
                GROUP BY setup_id
                HAVING COUNT(*) = 10  -- Ensure we have all 10 days
            ) l ON f.setup_id = l.setup_id
            """
            
            # Create or replace the table
            table_name = f"text_ml_features_{mode}"
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"CREATE TABLE {table_name} AS {text_features_query}", [setup_ids])
            
            # Get table info
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            feature_count = len(columns)
            
            # Export to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"{table_name}_{timestamp}.csv"
            
            df = conn.execute(f"SELECT * FROM {table_name}").df()
            df.to_csv(output_file, index=False)
            
            logger.info(f"✅ Text ML features merged and exported:")
            logger.info(f"- Table: {table_name}")
            logger.info(f"- Features: {feature_count}")
            logger.info(f"- Rows: {row_count}")
            logger.info(f"- Exported to: {output_file}")
            
            return {
                "table_name": table_name,
                "feature_count": feature_count,
                "row_count": row_count,
                "output_file": str(output_file)
            }
            
        except Exception as e:
            logger.error(f"❌ Error merging text features: {str(e)}")
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    def merge_financial_features(self, setup_ids: List[str], mode: str = 'training') -> Dict[str, Any]:
        """
        Merge all financial features into a single ML features table
        
        Args:
            setup_ids: List of setup_ids to process
            mode: Either 'training' or 'prediction'
            
        Returns:
            Dict with feature and row counts
        """
        logger.info(f"🔄 Merging financial features for {mode} mode...")
        
        conn = None
        try:
            conn = duckdb.connect(self.db_path)
            
            # Build comprehensive financial ML features query
            financial_features_query = """
            -- Simplified query that directly joins relevant tables
            SELECT 
                s.setup_id,
                
                -- Fundamentals data
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
                fr.current_ratio,
                fr.quick_ratio,
                fr.cash_ratio,
                fr.debt_to_equity,
                fr.debt_to_assets,
                fr.equity_ratio,
                fr.gross_margin,
                fr.operating_margin,
                fr.net_margin,
                fr.roe,
                fr.roa,
                fr.roic,
                fr.asset_turnover,
                fr.inventory_turnover,
                fr.receivables_turnover,
                fr.pe_ratio as price_to_earnings,
                fr.pb_ratio as price_to_book,
                fr.ev_ebitda,
                
                -- Features from fundamentals_features table
                ff.*,
                
                -- Label for training mode
                CASE 
                    WHEN '{mode}' = 'training' THEN l.outperformance_10d 
                    ELSE NULL 
                END as label
                
            FROM setups s
            LEFT JOIN (
                -- Get latest fundamentals before setup date
                SELECT DISTINCT ON (s_inner.setup_id) 
                    s_inner.setup_id,
                    f_inner.*
                FROM setups s_inner
                JOIN fundamentals f_inner
                    ON s_inner.lse_ticker = f_inner.ticker
                    AND f_inner.date <= s_inner.spike_timestamp
                WHERE s_inner.setup_id = ANY(?)
                ORDER BY s_inner.setup_id, f_inner.date DESC
            ) f ON s.setup_id = f.setup_id
            
            LEFT JOIN (
                -- Get latest financial ratios before setup date
                SELECT DISTINCT ON (s_inner.setup_id) 
                    s_inner.setup_id,
                    fr_inner.*
                FROM setups s_inner
                JOIN financial_ratios fr_inner
                    ON s_inner.lse_ticker = fr_inner.ticker
                    AND fr_inner.period_end <= s_inner.spike_timestamp
                WHERE s_inner.setup_id = ANY(?)
                ORDER BY s_inner.setup_id, fr_inner.period_end DESC
            ) fr ON s.setup_id = fr.setup_id
            
            LEFT JOIN fundamentals_features ff ON s.setup_id = ff.setup_id
            
            LEFT JOIN (
                -- Calculate average outperformance for first 10 days
                SELECT 
                    setup_id, 
                    AVG(outperformance_day) as outperformance_10d
                FROM daily_labels
                WHERE day_number <= 10
                GROUP BY setup_id
                HAVING COUNT(*) >= 5  -- Relaxed: require at least 5 days
            ) l ON s.setup_id = l.setup_id
            
            WHERE s.setup_id = ANY(?)
            """
            
            # Create or replace the table
            table_name = f"financial_ml_features_{mode}"
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"CREATE TABLE {table_name} AS {financial_features_query}", [setup_ids, setup_ids, setup_ids])
            
            # Get table info
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            feature_count = len(columns)
            
            # Export to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"{table_name}_{timestamp}.csv"
            
            df = conn.execute(f"SELECT * FROM {table_name}").df()
            df.to_csv(output_file, index=False)
            
            logger.info(f"✅ Financial ML features merged and exported:")
            logger.info(f"- Table: {table_name}")
            logger.info(f"- Features: {feature_count}")
            logger.info(f"- Rows: {row_count}")
            logger.info(f"- Exported to: {output_file}")
            
            return {
                "table_name": table_name,
                "feature_count": feature_count,
                "row_count": row_count,
                "output_file": str(output_file)
            }
            
        except Exception as e:
            logger.error(f"❌ Error merging financial features: {str(e)}")
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    def generate_features(self, mode: str = 'training', setup_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate all ML feature tables for the specified mode
        
        Args:
            mode: Either 'training' or 'prediction'
            setup_ids: Optional list of setup IDs to process (if not provided, will find complete setups)
            
        Returns:
            Dictionary with results for each feature type
        """
        if setup_ids is None:
            setup_ids = self.find_complete_setups()
            
        if not setup_ids:
            logger.error("No setup IDs to process")
            return {}
            
        logger.info(f"Generating ML features for {len(setup_ids)} setups in {mode} mode")
        
        results = {}
        
        # Generate text features
        try:
            results['text'] = self.merge_text_features(setup_ids, mode)
        except Exception as e:
            logger.error(f"Failed to merge text features: {str(e)}")
            results['text'] = {"error": str(e)}
        
        # Generate financial features
        try:
            results['financial'] = self.merge_financial_features(setup_ids, mode)
        except Exception as e:
            logger.error(f"Failed to merge financial features: {str(e)}")
            results['financial'] = {"error": str(e)}
        
        return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate ML feature tables')
    parser.add_argument('--mode', choices=['training', 'prediction'], default='training',
                       help='Mode: training or prediction')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--output-dir', default='data/ml_features',
                       help='Directory to save output CSV files')
    parser.add_argument('--setup-list', help='File containing setup IDs to process (one per line)')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = ComprehensiveMLFeatureGenerator(
        db_path=args.db_path,
        output_dir=args.output_dir
    )
    
    # Get setup IDs if provided
    setup_ids = None
    if args.setup_list:
        with open(args.setup_list, 'r') as f:
            setup_ids = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(setup_ids)} setup IDs from {args.setup_list}")
    
    # Generate features
    results = generator.generate_features(args.mode, setup_ids)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("📊 ML FEATURES GENERATION SUMMARY")
    logger.info("="*50)
    
    for feature_type, result in results.items():
        if "error" in result:
            logger.info(f"{feature_type.upper()} FEATURES: ❌ Error: {result['error']}")
        else:
            logger.info(f"{feature_type.upper()} FEATURES: ✅ Success")
            logger.info(f"- Table: {result['table_name']}")
            logger.info(f"- Features: {result['feature_count']}")
            logger.info(f"- Rows: {result['row_count']}")
            logger.info(f"- Output: {result['output_file']}")
        logger.info("-"*50)
    
    logger.info("✅ ML Features Generation Complete!")

if __name__ == "__main__":
    main() 