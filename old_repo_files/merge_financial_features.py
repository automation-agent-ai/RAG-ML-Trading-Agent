#!/usr/bin/env python3
"""
Merge financial features from fundamentals and financial_ratios tables
"""

import logging
import duckdb
from typing import List
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def merge_financial_features(db_path: str, setup_ids: List[str], mode: str = 'training'):
    """
    Merge financial features from fundamentals and financial_ratios tables
    
    Args:
        db_path: Path to DuckDB database
        setup_ids: List of setup_ids to process
        mode: Either 'training' or 'prediction'
    """
    logger.info("🔄 Merging financial features...")
    
    try:
        conn = duckdb.connect(db_path)
        
        # Build comprehensive financial ML features query
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
                ON s.lse_ticker = f.ticker 
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
                ON s.lse_ticker = fr.ticker
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
                    WHEN p.prev_revenue > 0 THEN (f.total_revenue - p.prev_revenue) / CAST(p.prev_revenue AS FLOAT)
                    ELSE NULL 
                END as revenue_growth_yoy,
                CASE 
                    WHEN p.prev_net_income > 0 THEN (f.net_income - p.prev_net_income) / CAST(p.prev_net_income AS FLOAT)
                    ELSE NULL 
                END as net_income_growth_yoy,
                CASE 
                    WHEN p.prev_operating_cash_flow > 0 THEN (f.operating_cash_flow - p.prev_operating_cash_flow) / CAST(p.prev_operating_cash_flow AS FLOAT)
                    ELSE NULL 
                END as operating_cash_flow_growth_yoy,
                CASE 
                    WHEN p.prev_ebitda > 0 THEN (f.ebitda - p.prev_ebitda) / CAST(p.prev_ebitda AS FLOAT)
                    ELSE NULL 
                END as ebitda_growth_yoy
            FROM fundamentals_clean f
            LEFT JOIN prev_clean p USING (setup_id)
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
                r.pe_ratio,
                r.pb_ratio,
                r.ps_ratio,
                r.ev_ebitda,
                r.book_value_per_share,
                r.revenue_per_share,
                r.cash_per_share,
                
                -- Growth metrics
                g.revenue_growth_yoy,
                g.net_income_growth_yoy,
                g.operating_cash_flow_growth_yoy,
                g.ebitda_growth_yoy
                
            FROM fundamentals_clean f
            FULL OUTER JOIN ratios_clean r USING (setup_id)
            LEFT JOIN growth_metrics g USING (setup_id)
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
        
        # Log feature categories
        logger.info("\nFinancial feature categories:")
        logger.info("1. Raw Fundamentals (20 features):")
        logger.info("   - Balance sheet: total_assets, total_debt, total_equity, etc.")
        logger.info("   - Income statement: total_revenue, gross_profit, operating_income, etc.")
        logger.info("   - Cash flow: operating_cash_flow, free_cash_flow, etc.")
        logger.info("2. Financial Ratios (23 features):")
        logger.info("   - Liquidity: current_ratio, quick_ratio, cash_ratio")
        logger.info("   - Profitability: gross_margin, operating_margin, net_margin")
        logger.info("   - Efficiency: asset_turnover, inventory_turnover")
        logger.info("   - Valuation: pe_ratio, pb_ratio, ps_ratio, ev_ebitda")
        logger.info("3. Growth Metrics (4 features):")
        logger.info("   - YoY growth: revenue, net_income, operating_cash_flow, ebitda")
        
        # Print sample data
        logger.info("\nSample data:")
        sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 1").fetchdf()
        logger.info(sample)
        
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

def main():
    """Run financial feature merger"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge financial features')
    parser.add_argument('--setup-ids', nargs='+', help='List of setup_ids to process')
    parser.add_argument('--mode', choices=['training', 'prediction'], default='training',
                       help='Pipeline mode: training or prediction')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    
    args = parser.parse_args()
    
    if not args.setup_ids:
        # If no setup_ids provided, get all from setups table
        conn = duckdb.connect(args.db_path)
        setup_ids = [row[0] for row in conn.execute("SELECT setup_id FROM setups").fetchall()]
        conn.close()
    else:
        setup_ids = args.setup_ids
    
    # Merge features
    result = merge_financial_features(args.db_path, setup_ids, args.mode)
    
    logger.info("\n🎉 Financial Feature Merger Complete!")
    logger.info(f"- Features: {result['feature_count']}")
    logger.info(f"- Rows: {result['row_count']}")

if __name__ == "__main__":
    main() 