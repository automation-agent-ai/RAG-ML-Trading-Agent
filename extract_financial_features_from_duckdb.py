#!/usr/bin/env python3
"""
Extract Financial Features From DuckDB

This script extracts financial features from the fundamentals and financial_ratios tables
in DuckDB and creates comprehensive ML feature tables for both training and prediction.

Usage:
    python extract_financial_features_from_duckdb.py --mode [training|prediction] --setup-list [file]
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
            
            # Build comprehensive financial ML features query
            # Based on merge_financial_features.py from old repo
            # NOTE: Adding .L suffix to setup tickers when joining with fundamentals and financial_ratios
            financial_features_query = """
            WITH 
            -- Get latest fundamentals data for each setup
            latest_fundamentals AS (
                SELECT 
                    s.setup_id,
                    s.lse_ticker,
                    s.spike_timestamp,
                    f.total_revenue,
                    f.gross_profit,
                    f.operating_income,
                    f.net_income,
                    f.ebitda,
                    f.basic_eps,
                    f.diluted_eps,
                    f.total_assets,
                    f.total_debt,
                    f.total_equity,
                    f.cash_and_equivalents,
                    f.current_assets,
                    f.current_liabilities,
                    f.working_capital,
                    f.property_plant_equipment,
                    f.operating_cash_flow,
                    f.free_cash_flow,
                    f.capital_expenditure,
                    f.financing_cash_flow,
                    f.investing_cash_flow,
                    ROW_NUMBER() OVER (PARTITION BY s.setup_id ORDER BY f.date DESC) as row_num
                FROM setups s
                LEFT JOIN fundamentals f 
                    ON s.lse_ticker || '.L' = f.ticker  -- Add .L suffix to match fundamentals ticker format
                    AND f.date <= s.spike_timestamp
                WHERE s.setup_id = ANY(?)
            ),
            fundamentals_clean AS (
                SELECT setup_id, lse_ticker, spike_timestamp,
                       total_revenue, gross_profit, operating_income,
                       net_income, ebitda, basic_eps, diluted_eps,
                       total_assets, total_debt, total_equity,
                       cash_and_equivalents, current_assets,
                       current_liabilities, working_capital,
                       property_plant_equipment, operating_cash_flow,
                       free_cash_flow, capital_expenditure,
                       financing_cash_flow, investing_cash_flow
                FROM latest_fundamentals
                WHERE row_num = 1
            ),
            -- Get latest financial ratios
            latest_ratios AS (
                SELECT 
                    s.setup_id,
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
                    fr.pe_ratio,
                    fr.pb_ratio,
                    fr.ps_ratio,
                    fr.ev_ebitda,
                    fr.book_value_per_share,
                    fr.revenue_per_share,
                    fr.cash_per_share,
                    ROW_NUMBER() OVER (PARTITION BY s.setup_id ORDER BY fr.period_end DESC) as row_num
                FROM setups s
                LEFT JOIN financial_ratios fr 
                    ON s.lse_ticker || '.L' = fr.ticker  -- Add .L suffix to match financial_ratios ticker format
                    AND fr.period_end <= s.spike_timestamp
                WHERE s.setup_id = ANY(?)
            ),
            ratios_clean AS (
                SELECT setup_id, current_ratio, quick_ratio, cash_ratio,
                       debt_to_equity, debt_to_assets, equity_ratio,
                       gross_margin, operating_margin, net_margin,
                       roe, roa, roic, asset_turnover, inventory_turnover,
                       receivables_turnover, pe_ratio, pb_ratio, ps_ratio,
                       ev_ebitda, book_value_per_share, revenue_per_share,
                       cash_per_share
                FROM latest_ratios
                WHERE row_num = 1
            ),
            -- Get previous year fundamentals for growth calculations
            prev_year_fundamentals AS (
                SELECT 
                    s.setup_id,
                    f.total_revenue as prev_revenue,
                    f.net_income as prev_net_income,
                    f.operating_cash_flow as prev_operating_cash_flow,
                    f.ebitda as prev_ebitda,
                    ROW_NUMBER() OVER (PARTITION BY s.setup_id ORDER BY f.date DESC) as row_num
                FROM setups s
                LEFT JOIN fundamentals f 
                    ON s.lse_ticker || '.L' = f.ticker  -- Add .L suffix to match fundamentals ticker format
                    AND f.date <= (s.spike_timestamp - INTERVAL '1 year')
                WHERE s.setup_id = ANY(?)
            ),
            prev_clean AS (
                SELECT setup_id, prev_revenue, prev_net_income,
                       prev_operating_cash_flow, prev_ebitda
                FROM prev_year_fundamentals
                WHERE row_num = 1
            ),
            -- Calculate growth metrics
            growth_metrics AS (
                SELECT 
                    f.setup_id,
                    -- Revenue growth
                    CASE 
                        WHEN p.prev_revenue IS NOT NULL AND p.prev_revenue > 0 AND f.total_revenue IS NOT NULL
                        THEN (f.total_revenue - p.prev_revenue) / p.prev_revenue
                        ELSE NULL
                    END as revenue_growth_yoy,
                    -- Net income growth
                    CASE 
                        WHEN p.prev_net_income IS NOT NULL AND p.prev_net_income > 0 AND f.net_income IS NOT NULL
                        THEN (f.net_income - p.prev_net_income) / p.prev_net_income
                        ELSE NULL
                    END as net_income_growth_yoy,
                    -- Operating cash flow growth
                    CASE 
                        WHEN p.prev_operating_cash_flow IS NOT NULL AND p.prev_operating_cash_flow > 0 AND f.operating_cash_flow IS NOT NULL
                        THEN (f.operating_cash_flow - p.prev_operating_cash_flow) / p.prev_operating_cash_flow
                        ELSE NULL
                    END as operating_cash_flow_growth_yoy,
                    -- EBITDA growth
                    CASE 
                        WHEN p.prev_ebitda IS NOT NULL AND p.prev_ebitda > 0 AND f.ebitda IS NOT NULL
                        THEN (f.ebitda - p.prev_ebitda) / p.prev_ebitda
                        ELSE NULL
                    END as ebitda_growth_yoy
                FROM fundamentals_clean f
                LEFT JOIN prev_clean p ON f.setup_id = p.setup_id
            ),
            -- Combine all financial data
            combined_financial_data AS (
                SELECT 
                    s.setup_id,
                    
                    -- Raw fundamentals
                    f.total_revenue,
                    f.gross_profit,
                    f.operating_income,
                    f.net_income,
                    f.ebitda,
                    f.basic_eps,
                    f.diluted_eps,
                    f.total_assets,
                    f.total_debt,
                    f.total_equity,
                    f.cash_and_equivalents,
                    f.current_assets,
                    f.current_liabilities,
                    f.working_capital,
                    f.property_plant_equipment,
                    f.operating_cash_flow,
                    f.free_cash_flow,
                    f.capital_expenditure,
                    f.financing_cash_flow,
                    f.investing_cash_flow,
                    
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
                    r.pe_ratio as price_earnings_ratio,
                    r.pb_ratio as price_to_book_ratio,
                    r.ps_ratio as price_to_sales_ratio,
                    r.ev_ebitda as enterprise_value_to_ebitda,
                    r.book_value_per_share,
                    r.revenue_per_share,
                    r.cash_per_share,
                    
                    -- Growth metrics
                    g.revenue_growth_yoy,
                    g.net_income_growth_yoy,
                    g.operating_cash_flow_growth_yoy,
                    g.ebitda_growth_yoy,
                    
                    -- Calculated metrics
                    CASE 
                        WHEN f.total_equity IS NOT NULL AND f.total_equity > 0 AND f.net_income IS NOT NULL
                        THEN f.net_income / f.total_equity
                        ELSE r.roe
                    END as calculated_roe,
                    
                    CASE 
                        WHEN f.total_assets IS NOT NULL AND f.total_assets > 0 AND f.net_income IS NOT NULL
                        THEN f.net_income / f.total_assets
                        ELSE r.roa
                    END as calculated_roa,
                    
                    CASE 
                        WHEN f.total_equity IS NOT NULL AND f.total_debt IS NOT NULL AND f.total_equity > 0
                        THEN f.total_debt / f.total_equity
                        ELSE r.debt_to_equity
                    END as calculated_debt_to_equity,
                    
                    -- Label for training mode
                    CASE 
                        WHEN '{mode}' = 'training' THEN l.outperformance_10d 
                        ELSE NULL 
                    END as label
                
                FROM setups s
                LEFT JOIN fundamentals_clean f ON s.setup_id = f.setup_id
                LEFT JOIN ratios_clean r ON s.setup_id = r.setup_id
                LEFT JOIN growth_metrics g ON s.setup_id = g.setup_id
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
            )
            
            SELECT * FROM combined_financial_data
            """
            
            # Create or replace the table
            table_name = f"financial_ml_features_{mode}"
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"CREATE TABLE {table_name} AS {financial_features_query}", [setup_ids, setup_ids, setup_ids, setup_ids])
            
            # Get table info
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            feature_count = len(columns)
            
            # Export to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"{table_name}_{timestamp}.csv"
            
            df = conn.execute(f"SELECT * FROM {table_name}").df()
            df.to_csv(output_file, index=False)
            
            # Check for non-null values in key columns
            total_revenue_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE total_revenue IS NOT NULL").fetchone()[0]
            
            logger.info(f"✅ Financial features extracted and exported:")
            logger.info(f"- Table: {table_name}")
            logger.info(f"- Features: {feature_count}")
            logger.info(f"- Rows: {row_count}")
            logger.info(f"- Rows with non-null total_revenue: {total_revenue_count} ({total_revenue_count/row_count if row_count > 0 else 0:.1%})")
            logger.info(f"- Exported to: {output_file}")
            
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
    parser = argparse.ArgumentParser(description='Extract financial features from DuckDB')
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