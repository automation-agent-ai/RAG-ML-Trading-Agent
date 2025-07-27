#!/usr/bin/env python3
"""
Extract Financial Features

This script extracts financial features from the fundamentals and financial_ratios tables
and creates a comprehensive financial_features table.

Usage:
    python extract_financial_features.py --mode [training|prediction] --setup-list [file]
"""

import os
import sys
import argparse
import logging
import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinancialFeaturesExtractor:
    """Extracts financial features from fundamentals and financial_ratios tables"""
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        output_dir: str = "data/ml_features"
    ):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def extract_financial_features(self, setup_ids: List[str], mode: str = 'training') -> Dict[str, Any]:
        """
        Extract financial features for the given setup IDs
        
        Args:
            setup_ids: List of setup IDs to process
            mode: Either 'training' or 'prediction'
            
        Returns:
            Dictionary with feature and row counts
        """
        logger.info(f"🔄 Extracting financial features for {len(setup_ids)} setups in {mode} mode...")
        
        conn = None
        try:
            conn = duckdb.connect(self.db_path)
            
            # Create financial features query
            financial_features_query = """
            -- Create a comprehensive financial features table
            WITH setup_data AS (
                SELECT 
                    s.setup_id,
                    s.lse_ticker,
                    s.spike_timestamp
                FROM setups s
                WHERE s.setup_id = ANY(?)
            ),
            -- Get latest fundamentals data before setup date
            latest_fundamentals AS (
                SELECT 
                    sd.setup_id,
                    f.*,
                    ROW_NUMBER() OVER (PARTITION BY sd.setup_id ORDER BY f.date DESC) as row_num
                FROM setup_data sd
                JOIN fundamentals f ON sd.lse_ticker = f.ticker AND f.date <= sd.spike_timestamp
            ),
            clean_fundamentals AS (
                SELECT * FROM latest_fundamentals WHERE row_num = 1
            ),
            -- Get latest financial ratios before setup date
            latest_ratios AS (
                SELECT 
                    sd.setup_id,
                    fr.*,
                    ROW_NUMBER() OVER (PARTITION BY sd.setup_id ORDER BY fr.period_end DESC) as row_num
                FROM setup_data sd
                JOIN financial_ratios fr ON sd.lse_ticker = fr.ticker AND fr.period_end <= sd.spike_timestamp
            ),
            clean_ratios AS (
                SELECT * FROM latest_ratios WHERE row_num = 1
            ),
            -- Get previous year fundamentals for growth calculations
            prev_year_fundamentals AS (
                SELECT 
                    sd.setup_id,
                    f.total_revenue as prev_revenue,
                    f.net_income as prev_net_income,
                    f.operating_cash_flow as prev_operating_cash_flow,
                    f.ebitda as prev_ebitda,
                    ROW_NUMBER() OVER (PARTITION BY sd.setup_id ORDER BY f.date DESC) as row_num
                FROM setup_data sd
                JOIN fundamentals f ON sd.lse_ticker = f.ticker 
                AND f.date <= (sd.spike_timestamp - INTERVAL '1 year')
            ),
            clean_prev_year AS (
                SELECT * FROM prev_year_fundamentals WHERE row_num = 1
            ),
            -- Calculate growth metrics
            growth_metrics AS (
                SELECT 
                    cf.setup_id,
                    -- Revenue growth
                    CASE 
                        WHEN py.prev_revenue IS NOT NULL AND py.prev_revenue > 0 AND cf.total_revenue IS NOT NULL
                        THEN (cf.total_revenue - py.prev_revenue) / py.prev_revenue
                        ELSE NULL
                    END as revenue_growth_yoy,
                    -- Net income growth
                    CASE 
                        WHEN py.prev_net_income IS NOT NULL AND py.prev_net_income > 0 AND cf.net_income IS NOT NULL
                        THEN (cf.net_income - py.prev_net_income) / py.prev_net_income
                        ELSE NULL
                    END as net_income_growth_yoy,
                    -- Operating cash flow growth
                    CASE 
                        WHEN py.prev_operating_cash_flow IS NOT NULL AND py.prev_operating_cash_flow > 0 AND cf.operating_cash_flow IS NOT NULL
                        THEN (cf.operating_cash_flow - py.prev_operating_cash_flow) / py.prev_operating_cash_flow
                        ELSE NULL
                    END as operating_cash_flow_growth_yoy,
                    -- EBITDA growth
                    CASE 
                        WHEN py.prev_ebitda IS NOT NULL AND py.prev_ebitda > 0 AND cf.ebitda IS NOT NULL
                        THEN (cf.ebitda - py.prev_ebitda) / py.prev_ebitda
                        ELSE NULL
                    END as ebitda_growth_yoy
                FROM clean_fundamentals cf
                LEFT JOIN clean_prev_year py ON cf.setup_id = py.setup_id
            ),
            -- Combine all financial data
            combined_financial_data AS (
                SELECT 
                    sd.setup_id,
                    
                    -- Raw fundamentals
                    cf.total_revenue,
                    cf.gross_profit,
                    cf.operating_income,
                    cf.net_income,
                    cf.ebitda,
                    cf.operating_cash_flow,
                    cf.free_cash_flow,
                    cf.total_assets,
                    cf.total_debt,
                    cf.total_equity,
                    cf.current_assets,
                    cf.current_liabilities,
                    cf.working_capital,
                    
                    -- Financial ratios
                    cr.current_ratio,
                    cr.quick_ratio,
                    cr.cash_ratio,
                    cr.debt_to_equity,
                    cr.debt_to_assets,
                    cr.equity_ratio,
                    cr.gross_margin,
                    cr.operating_margin,
                    cr.net_margin,
                    cr.roe,
                    cr.roa,
                    cr.roic,
                    cr.asset_turnover,
                    cr.inventory_turnover,
                    cr.receivables_turnover,
                    cr.pe_ratio as price_to_earnings,
                    cr.pb_ratio as price_to_book,
                    cr.ev_ebitda,
                    
                    -- Growth metrics
                    gm.revenue_growth_yoy,
                    gm.net_income_growth_yoy,
                    gm.operating_cash_flow_growth_yoy,
                    gm.ebitda_growth_yoy,
                    
                    -- Additional metrics
                    CASE 
                        WHEN cf.total_equity IS NOT NULL AND cf.total_equity > 0 AND cf.net_income IS NOT NULL
                        THEN cf.net_income / cf.total_equity
                        ELSE cr.roe
                    END as calculated_roe,
                    
                    CASE 
                        WHEN cf.total_assets IS NOT NULL AND cf.total_assets > 0 AND cf.net_income IS NOT NULL
                        THEN cf.net_income / cf.total_assets
                        ELSE cr.roa
                    END as calculated_roa,
                    
                    CASE 
                        WHEN cf.total_equity IS NOT NULL AND cf.total_debt IS NOT NULL
                        THEN cf.total_debt / NULLIF(cf.total_equity, 0)
                        ELSE cr.debt_to_equity
                    END as calculated_debt_to_equity,
                    
                    -- Label for training mode
                    CASE 
                        WHEN '{mode}' = 'training' THEN l.outperformance_10d 
                        ELSE NULL 
                    END as label
                
                FROM setup_data sd
                LEFT JOIN clean_fundamentals cf ON sd.setup_id = cf.setup_id
                LEFT JOIN clean_ratios cr ON sd.setup_id = cr.setup_id
                LEFT JOIN growth_metrics gm ON sd.setup_id = gm.setup_id
                LEFT JOIN (
                    -- Calculate average outperformance for first 10 days
                    SELECT 
                        setup_id, 
                        AVG(outperformance_day) as outperformance_10d
                    FROM daily_labels
                    WHERE day_number <= 10
                    GROUP BY setup_id
                    HAVING COUNT(*) >= 5  -- Relaxed: require at least 5 days
                ) l ON sd.setup_id = l.setup_id
            )
            
            SELECT * FROM combined_financial_data
            """
            
            # Create or replace the table
            table_name = f"financial_ml_features_{mode}"
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"CREATE TABLE {table_name} AS {financial_features_query}", [setup_ids])
            
            # Get table info
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            feature_count = len(columns)
            
            # Export to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"{table_name}_{timestamp}.csv"
            
            df = conn.execute(f"SELECT * FROM {table_name}").df()
            df.to_csv(output_file, index=False)
            
            logger.info(f"✅ Financial features extracted and exported:")
            logger.info(f"- Table: {table_name}")
            logger.info(f"- Features: {feature_count}")
            logger.info(f"- Rows: {row_count}")
            logger.info(f"- Exported to: {output_file}")
            
            # Check for non-null values in key columns
            total_revenue_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE total_revenue IS NOT NULL").fetchone()[0]
            logger.info(f"- Rows with non-null total_revenue: {total_revenue_count} ({total_revenue_count/row_count:.1%})")
            
            return {
                "table_name": table_name,
                "feature_count": feature_count,
                "row_count": row_count,
                "output_file": str(output_file)
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting financial features: {str(e)}")
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Extract financial features')
    parser.add_argument('--mode', choices=['training', 'prediction'], default='training',
                       help='Mode: training or prediction')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--output-dir', default='data/ml_features',
                       help='Directory to save output CSV files')
    parser.add_argument('--setup-list', help='File containing setup IDs to process (one per line)')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = FinancialFeaturesExtractor(
        db_path=args.db_path,
        output_dir=args.output_dir
    )
    
    # Get setup IDs if provided
    setup_ids = None
    if args.setup_list:
        with open(args.setup_list, 'r') as f:
            setup_ids = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(setup_ids)} setup IDs from {args.setup_list}")
    else:
        # Use all setup IDs
        conn = duckdb.connect(args.db_path)
        setup_ids = conn.execute('SELECT DISTINCT setup_id FROM setups LIMIT 1000').df()['setup_id'].tolist()
        conn.close()
        logger.info(f"Using {len(setup_ids)} setup IDs from database")
    
    # Extract features
    result = extractor.extract_financial_features(setup_ids, args.mode)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("📊 FINANCIAL FEATURES EXTRACTION SUMMARY")
    logger.info("="*50)
    logger.info(f"✅ Success")
    logger.info(f"- Table: {result['table_name']}")
    logger.info(f"- Features: {result['feature_count']}")
    logger.info(f"- Rows: {result['row_count']}")
    logger.info(f"- Output: {result['output_file']}")
    logger.info("="*50)

if __name__ == "__main__":
    main() 