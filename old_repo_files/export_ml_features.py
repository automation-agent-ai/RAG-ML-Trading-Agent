#!/usr/bin/env python3
"""
Export ML features for training and prediction.

This script:
1. Creates data/ml_training directory if it doesn't exist
2. Exports text ML features with labels for training
3. Exports text ML features for prediction (ensuring no overlap with training)
"""

import logging
import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_to_categorical_labels(raw_labels: pd.Series) -> pd.Series:
    """Convert raw outperformance values to categorical labels"""
    # Calculate percentiles
    p33 = raw_labels.quantile(1/3)
    p67 = raw_labels.quantile(2/3)
    
    # Convert to categories
    labels = pd.Series(index=raw_labels.index, dtype=str)
    labels[raw_labels <= p33] = 'negative'
    labels[(raw_labels > p33) & (raw_labels <= p67)] = 'neutral'
    labels[raw_labels > p67] = 'positive'
    
    return labels

def export_training_features(conn: duckdb.DuckDBPyConnection, output_dir: Path):
    """Export training features with labels"""
    logger.info("📊 Exporting training features...")
    
    # Query to get training data with labels
    text_query = """
    WITH setup_labels AS (
        SELECT setup_id,
               AVG(outperformance_day) as outperformance_10d
        FROM daily_labels
        WHERE day_number <= 10
        GROUP BY setup_id
        HAVING COUNT(*) = 10  -- Ensure we have all 10 days
    ),
    complete_features AS (
        SELECT t.*, l.outperformance_10d as raw_label
        FROM text_ml_features_training t
        INNER JOIN setup_labels l ON t.setup_id = l.setup_id
        WHERE (
            -- Ensure we have at least some non-null features
            t.count_financial_results IS NOT NULL OR
            t.sentiment_score_financial_results IS NOT NULL OR
            t.count_corporate_actions IS NOT NULL OR
            t.posts_avg_sentiment IS NOT NULL OR
            t.post_count IS NOT NULL OR
            t.recommendation_count IS NOT NULL
        )
        AND l.outperformance_10d IS NOT NULL
    )
    SELECT * FROM complete_features
    """
    
    # Execute query and save to CSV
    text_df = conn.execute(text_query).df()
    
    # Convert raw labels to categorical classes
    text_df['label'] = convert_to_categorical_labels(text_df['raw_label'])
    text_df = text_df.drop('raw_label', axis=1)
    
    # Save setup IDs for reference
    training_setup_ids = text_df['setup_id'].tolist()
    
    # Basic data quality checks for text features
    logger.info("\n📊 Text Features Quality Report:")
    logger.info(f"- Total samples: {len(text_df)}")
    logger.info(f"- Features: {len(text_df.columns)-2}")  # -2 for setup_id and label
    
    # Check missing values
    missing = text_df.drop(['setup_id', 'label'], axis=1).isnull().sum()
    logger.info("\nMissing Values:")
    for col in missing[missing > 0].index:
        pct_missing = (missing[col] / len(text_df)) * 100
        logger.info(f"  - {col}: {missing[col]} missing ({pct_missing:.1f}%)")
    
    # Drop analyst features (100% missing)
    analyst_features = [
        'avg_price_target',
        'price_target_vs_current',
        'price_target_spread'
    ]
    text_df = text_df.drop(analyst_features, axis=1)
    logger.info("\n🗑️ Dropped analyst features (100% missing)")
    
    # Save text features to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_output_file = output_dir / f"text_ml_features_training_{timestamp}.csv"
    text_df.to_csv(text_output_file, index=False)
    logger.info(f"\n✅ Text features saved to: {text_output_file}")
    
    # Query to get financial features with labels
    financial_query = """
    WITH setup_labels AS (
        SELECT setup_id,
               AVG(outperformance_day) as outperformance_10d
        FROM daily_labels
        WHERE day_number <= 10
        GROUP BY setup_id
        HAVING COUNT(*) = 10  -- Ensure we have all 10 days
    ),
    historical_fundamentals AS (
        SELECT 
            s.setup_id,
            s.lse_ticker,
            s.spike_timestamp,
            -- Current year (0Y)
            f0.total_revenue as revenue_0y,
            f0.gross_profit as gross_profit_0y,
            f0.operating_income as operating_income_0y,
            f0.net_income as net_income_0y,
            f0.ebitda as ebitda_0y,
            f0.basic_eps as basic_eps_0y,
            f0.diluted_eps as diluted_eps_0y,
            f0.total_assets as total_assets_0y,
            f0.total_debt as total_debt_0y,
            f0.total_equity as total_equity_0y,
            f0.cash_and_equivalents as cash_and_equivalents_0y,
            f0.current_assets as current_assets_0y,
            f0.current_liabilities as current_liabilities_0y,
            f0.working_capital as working_capital_0y,
            f0.property_plant_equipment as property_plant_equipment_0y,
            f0.operating_cash_flow as operating_cash_flow_0y,
            f0.free_cash_flow as free_cash_flow_0y,
            f0.capital_expenditure as capital_expenditure_0y,
            
            -- Previous year (-1Y)
            f1.total_revenue as revenue_1y,
            f1.gross_profit as gross_profit_1y,
            f1.operating_income as operating_income_1y,
            f1.net_income as net_income_1y,
            f1.ebitda as ebitda_1y,
            f1.total_assets as total_assets_1y,
            f1.total_debt as total_debt_1y,
            f1.total_equity as total_equity_1y,
            f1.operating_cash_flow as operating_cash_flow_1y,
            f1.free_cash_flow as free_cash_flow_1y,
            
            -- Two years ago (-2Y)
            f2.total_revenue as revenue_2y,
            f2.gross_profit as gross_profit_2y,
            f2.operating_income as operating_income_2y,
            f2.net_income as net_income_2y,
            f2.ebitda as ebitda_2y,
            f2.total_assets as total_assets_2y,
            f2.total_debt as total_debt_2y,
            f2.total_equity as total_equity_2y,
            f2.operating_cash_flow as operating_cash_flow_2y,
            f2.free_cash_flow as free_cash_flow_2y,
            
            -- Three years ago (-3Y)
            f3.total_revenue as revenue_3y,
            f3.gross_profit as gross_profit_3y,
            f3.operating_income as operating_income_3y,
            f3.net_income as net_income_3y,
            f3.ebitda as ebitda_3y,
            f3.total_assets as total_assets_3y,
            f3.total_debt as total_debt_3y,
            f3.total_equity as total_equity_3y,
            f3.operating_cash_flow as operating_cash_flow_3y,
            f3.free_cash_flow as free_cash_flow_3y,
            
            ROW_NUMBER() OVER (PARTITION BY s.setup_id ORDER BY f0.date DESC) as row_num
        FROM setups s
        -- Current year
        LEFT JOIN fundamentals f0 
            ON s.lse_ticker || '.L' = f0.ticker
            AND f0.date <= s.spike_timestamp
        -- Previous year
        LEFT JOIN fundamentals f1
            ON s.lse_ticker || '.L' = f1.ticker
            AND f1.date <= s.spike_timestamp - INTERVAL '1 year'
            AND f1.date > s.spike_timestamp - INTERVAL '1.5 years'  -- Ensure we get the right year
        -- Two years ago
        LEFT JOIN fundamentals f2
            ON s.lse_ticker || '.L' = f2.ticker
            AND f2.date <= s.spike_timestamp - INTERVAL '2 years'
            AND f2.date > s.spike_timestamp - INTERVAL '2.5 years'
        -- Three years ago
        LEFT JOIN fundamentals f3
            ON s.lse_ticker || '.L' = f3.ticker
            AND f3.date <= s.spike_timestamp - INTERVAL '3 years'
            AND f3.date > s.spike_timestamp - INTERVAL '3.5 years'
        WHERE s.setup_id IN (SELECT setup_id FROM setup_labels)
    ),
    fundamentals_clean AS (
        SELECT * FROM historical_fundamentals WHERE row_num = 1
    ),
    growth_metrics AS (
        SELECT 
            setup_id,
            
            -- Revenue growth rates (YoY)
            CASE WHEN revenue_1y > 0 THEN (revenue_0y - revenue_1y) / CAST(revenue_1y AS FLOAT) ELSE NULL END as revenue_growth_1y,
            CASE WHEN revenue_2y > 0 THEN (revenue_1y - revenue_2y) / CAST(revenue_2y AS FLOAT) ELSE NULL END as revenue_growth_2y,
            CASE WHEN revenue_3y > 0 THEN (revenue_2y - revenue_3y) / CAST(revenue_3y AS FLOAT) ELSE NULL END as revenue_growth_3y,
            
            -- Operating income growth
            CASE WHEN operating_income_1y > 0 THEN (operating_income_0y - operating_income_1y) / CAST(operating_income_1y AS FLOAT) ELSE NULL END as operating_income_growth_1y,
            CASE WHEN operating_income_2y > 0 THEN (operating_income_1y - operating_income_2y) / CAST(operating_income_2y AS FLOAT) ELSE NULL END as operating_income_growth_2y,
            CASE WHEN operating_income_3y > 0 THEN (operating_income_2y - operating_income_3y) / CAST(operating_income_3y AS FLOAT) ELSE NULL END as operating_income_growth_3y,
            
            -- Net income growth (handle negative values)
            CASE WHEN net_income_1y != 0 THEN (net_income_0y - net_income_1y) / CAST(ABS(net_income_1y) AS FLOAT) ELSE NULL END as net_income_growth_1y,
            CASE WHEN net_income_2y != 0 THEN (net_income_1y - net_income_2y) / CAST(ABS(net_income_2y) AS FLOAT) ELSE NULL END as net_income_growth_2y,
            CASE WHEN net_income_3y != 0 THEN (net_income_2y - net_income_3y) / CAST(ABS(net_income_3y) AS FLOAT) ELSE NULL END as net_income_growth_3y,
            
            -- EBITDA growth
            CASE WHEN ebitda_1y > 0 THEN (ebitda_0y - ebitda_1y) / CAST(ebitda_1y AS FLOAT) ELSE NULL END as ebitda_growth_1y,
            CASE WHEN ebitda_2y > 0 THEN (ebitda_1y - ebitda_2y) / CAST(ebitda_2y AS FLOAT) ELSE NULL END as ebitda_growth_2y,
            CASE WHEN ebitda_3y > 0 THEN (ebitda_2y - ebitda_3y) / CAST(ebitda_3y AS FLOAT) ELSE NULL END as ebitda_growth_3y,
            
            -- Operating Cash Flow growth
            CASE WHEN operating_cash_flow_1y != 0 THEN (operating_cash_flow_0y - operating_cash_flow_1y) / CAST(ABS(operating_cash_flow_1y) AS FLOAT) ELSE NULL END as ocf_growth_1y,
            CASE WHEN operating_cash_flow_2y != 0 THEN (operating_cash_flow_1y - operating_cash_flow_2y) / CAST(ABS(operating_cash_flow_2y) AS FLOAT) ELSE NULL END as ocf_growth_2y,
            CASE WHEN operating_cash_flow_3y != 0 THEN (operating_cash_flow_2y - operating_cash_flow_3y) / CAST(ABS(operating_cash_flow_3y) AS FLOAT) ELSE NULL END as ocf_growth_3y
        FROM fundamentals_clean
    ),
    rolling_stats AS (
        SELECT
            setup_id,
            
            -- 3-year moving averages (where data available)
            CASE WHEN revenue_0y IS NOT NULL AND revenue_1y IS NOT NULL AND revenue_2y IS NOT NULL 
                 THEN (revenue_0y + revenue_1y + revenue_2y) / 3.0 ELSE NULL END as revenue_3y_avg,
            CASE WHEN operating_income_0y IS NOT NULL AND operating_income_1y IS NOT NULL AND operating_income_2y IS NOT NULL 
                 THEN (operating_income_0y + operating_income_1y + operating_income_2y) / 3.0 ELSE NULL END as operating_income_3y_avg,
            CASE WHEN net_income_0y IS NOT NULL AND net_income_1y IS NOT NULL AND net_income_2y IS NOT NULL 
                 THEN (net_income_0y + net_income_1y + net_income_2y) / 3.0 ELSE NULL END as net_income_3y_avg,
            CASE WHEN ebitda_0y IS NOT NULL AND ebitda_1y IS NOT NULL AND ebitda_2y IS NOT NULL 
                 THEN (ebitda_0y + ebitda_1y + ebitda_2y) / 3.0 ELSE NULL END as ebitda_3y_avg,
            
            -- Financial health evolution - margin trends
            CASE WHEN revenue_0y > 0 THEN operating_income_0y / CAST(revenue_0y AS FLOAT) ELSE NULL END as operating_margin_0y,
            CASE WHEN revenue_1y > 0 THEN operating_income_1y / CAST(revenue_1y AS FLOAT) ELSE NULL END as operating_margin_1y,
            CASE WHEN revenue_2y > 0 THEN operating_income_2y / CAST(revenue_2y AS FLOAT) ELSE NULL END as operating_margin_2y,
            
            CASE WHEN revenue_0y > 0 THEN net_income_0y / CAST(revenue_0y AS FLOAT) ELSE NULL END as net_margin_0y,
            CASE WHEN revenue_1y > 0 THEN net_income_1y / CAST(revenue_1y AS FLOAT) ELSE NULL END as net_margin_1y,
            CASE WHEN revenue_2y > 0 THEN net_income_2y / CAST(revenue_2y AS FLOAT) ELSE NULL END as net_margin_2y,
            
            -- Leverage evolution
            CASE WHEN total_equity_0y > 0 THEN total_debt_0y / CAST(total_equity_0y AS FLOAT) ELSE NULL END as debt_to_equity_0y,
            CASE WHEN total_equity_1y > 0 THEN total_debt_1y / CAST(total_equity_1y AS FLOAT) ELSE NULL END as debt_to_equity_1y,
            CASE WHEN total_equity_2y > 0 THEN total_debt_2y / CAST(total_equity_2y AS FLOAT) ELSE NULL END as debt_to_equity_2y
        FROM fundamentals_clean
    ),
    trend_indicators AS (
        SELECT
            fc.setup_id,
            rs.revenue_3y_avg,
            rs.operating_income_3y_avg,
            rs.net_income_3y_avg,
            rs.ebitda_3y_avg,
            rs.operating_margin_0y,
            rs.operating_margin_1y,
            rs.operating_margin_2y,
            rs.net_margin_0y,
            rs.net_margin_1y,
            rs.net_margin_2y,
            rs.debt_to_equity_0y,
            rs.debt_to_equity_1y,
            rs.debt_to_equity_2y,
            
            -- Consecutive growth indicators
            CASE 
                WHEN fc.revenue_0y > fc.revenue_1y 
                 AND fc.revenue_1y > fc.revenue_2y 
                 AND fc.revenue_2y > fc.revenue_3y THEN 3
                WHEN fc.revenue_0y > fc.revenue_1y 
                 AND fc.revenue_1y > fc.revenue_2y THEN 2
                WHEN fc.revenue_0y > fc.revenue_1y THEN 1
                ELSE 0 
            END as revenue_consecutive_growth_years,
            
            -- Growth acceleration/deceleration
            CASE
                WHEN gm.revenue_growth_1y IS NOT NULL AND gm.revenue_growth_2y IS NOT NULL AND gm.revenue_growth_3y IS NOT NULL THEN
                    CASE 
                        WHEN gm.revenue_growth_1y > gm.revenue_growth_2y AND gm.revenue_growth_2y > gm.revenue_growth_3y THEN 1
                        WHEN gm.revenue_growth_1y < gm.revenue_growth_2y AND gm.revenue_growth_2y < gm.revenue_growth_3y THEN -1
                        ELSE 0
                    END
                ELSE NULL
            END as revenue_growth_acceleration,
            
            -- Operating margin improvement trend
            CASE
                WHEN rs.operating_margin_0y IS NOT NULL AND rs.operating_margin_1y IS NOT NULL AND rs.operating_margin_2y IS NOT NULL THEN
                    CASE 
                        WHEN rs.operating_margin_0y > rs.operating_margin_1y AND rs.operating_margin_1y > rs.operating_margin_2y THEN 1
                        WHEN rs.operating_margin_0y < rs.operating_margin_1y AND rs.operating_margin_1y < rs.operating_margin_2y THEN -1
                        ELSE 0
                    END
                ELSE NULL
            END as operating_margin_improvement_trend,
            
            -- Net margin improvement trend
            CASE
                WHEN rs.net_margin_0y IS NOT NULL AND rs.net_margin_1y IS NOT NULL AND rs.net_margin_2y IS NOT NULL THEN
                    CASE 
                        WHEN rs.net_margin_0y > rs.net_margin_1y AND rs.net_margin_1y > rs.net_margin_2y THEN 1
                        WHEN rs.net_margin_0y < rs.net_margin_1y AND rs.net_margin_1y < rs.net_margin_2y THEN -1
                        ELSE 0
                    END
                ELSE NULL
            END as net_margin_improvement_trend,
            
            -- Leverage trend (1 = improving/decreasing debt, -1 = worsening/increasing debt)
            CASE
                WHEN rs.debt_to_equity_0y IS NOT NULL AND rs.debt_to_equity_1y IS NOT NULL AND rs.debt_to_equity_2y IS NOT NULL THEN
                    CASE 
                        WHEN rs.debt_to_equity_0y < rs.debt_to_equity_1y AND rs.debt_to_equity_1y < rs.debt_to_equity_2y THEN 1
                        WHEN rs.debt_to_equity_0y > rs.debt_to_equity_1y AND rs.debt_to_equity_1y > rs.debt_to_equity_2y THEN -1
                        ELSE 0
                    END
                ELSE NULL
            END as leverage_improvement_trend,
            
            -- Growth stability (coefficient of variation for revenue growth)
            CASE 
                WHEN gm.revenue_growth_1y IS NOT NULL AND gm.revenue_growth_2y IS NOT NULL AND gm.revenue_growth_3y IS NOT NULL THEN
                    SQRT(
                        POWER(gm.revenue_growth_1y - (gm.revenue_growth_1y + gm.revenue_growth_2y + gm.revenue_growth_3y)/3.0, 2) +
                        POWER(gm.revenue_growth_2y - (gm.revenue_growth_1y + gm.revenue_growth_2y + gm.revenue_growth_3y)/3.0, 2) +
                        POWER(gm.revenue_growth_3y - (gm.revenue_growth_1y + gm.revenue_growth_2y + gm.revenue_growth_3y)/3.0, 2)
                    ) / 3.0
                ELSE NULL
            END as revenue_growth_stability
        FROM fundamentals_clean fc
        LEFT JOIN growth_metrics gm USING (setup_id)
        LEFT JOIN rolling_stats rs USING (setup_id)
    ),
    ratios_data AS (
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
            ROW_NUMBER() OVER (PARTITION BY s.setup_id ORDER BY fr.period_end DESC) as row_num
        FROM setups s
        LEFT JOIN financial_ratios fr 
            ON s.lse_ticker || '.L' = fr.ticker
            AND fr.period_end <= s.spike_timestamp
        WHERE s.setup_id IN (SELECT setup_id FROM setup_labels)
    ),
    ratios_clean AS (
        SELECT 
            setup_id,
            current_ratio,
            quick_ratio,
            cash_ratio,
            debt_to_equity,
            debt_to_assets,
            equity_ratio,
            gross_margin,
            operating_margin,
            net_margin,
            roe,
            roa,
            roic,
            asset_turnover
        FROM ratios_data
        WHERE row_num = 1
    ),
    base_features AS (
        SELECT 
            COALESCE(fc.setup_id, r.setup_id, ti.setup_id) as setup_id,
            
            -- Current year raw fundamentals
            fc.revenue_0y as total_revenue,
            fc.gross_profit_0y as gross_profit,
            fc.operating_income_0y as operating_income,
            fc.net_income_0y as net_income,
            fc.ebitda_0y as ebitda,
            fc.basic_eps_0y as basic_eps,
            fc.diluted_eps_0y as diluted_eps,
            fc.total_assets_0y as total_assets,
            fc.total_debt_0y as total_debt,
            fc.total_equity_0y as total_equity,
            fc.cash_and_equivalents_0y as cash_and_equivalents,
            fc.current_assets_0y as current_assets,
            fc.current_liabilities_0y as current_liabilities,
            fc.working_capital_0y as working_capital,
            fc.property_plant_equipment_0y as property_plant_equipment,
            fc.operating_cash_flow_0y as operating_cash_flow,
            fc.free_cash_flow_0y as free_cash_flow,
            fc.capital_expenditure_0y as capital_expenditure,
            
            -- Historical growth metrics
            gm.revenue_growth_1y,
            gm.revenue_growth_2y,
            gm.revenue_growth_3y,
            gm.operating_income_growth_1y,
            gm.operating_income_growth_2y,
            gm.operating_income_growth_3y,
            gm.net_income_growth_1y,
            gm.net_income_growth_2y,
            gm.net_income_growth_3y,
            gm.ebitda_growth_1y,
            gm.ebitda_growth_2y,
            gm.ebitda_growth_3y,
            gm.ocf_growth_1y,
            gm.ocf_growth_2y,
            gm.ocf_growth_3y,
            
            -- Rolling statistics
            ti.revenue_3y_avg,
            ti.operating_income_3y_avg,
            ti.net_income_3y_avg,
            ti.ebitda_3y_avg,
            
            -- Trend indicators
            ti.revenue_consecutive_growth_years,
            ti.revenue_growth_acceleration,
            ti.operating_margin_improvement_trend,
            ti.net_margin_improvement_trend,
            ti.leverage_improvement_trend,
            ti.revenue_growth_stability,
            
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
            
            -- Label
            l.outperformance_10d as raw_label
            
        FROM fundamentals_clean fc
        FULL OUTER JOIN ratios_clean r USING (setup_id)
        LEFT JOIN growth_metrics gm USING (setup_id)
        LEFT JOIN trend_indicators ti USING (setup_id)
        INNER JOIN setup_labels l ON COALESCE(fc.setup_id, r.setup_id, ti.setup_id) = l.setup_id
        WHERE l.outperformance_10d IS NOT NULL
        AND (
            -- Ensure we have at least some non-null features from any source
            fc.revenue_0y IS NOT NULL OR
            fc.operating_income_0y IS NOT NULL OR
            fc.net_income_0y IS NOT NULL OR
            fc.ebitda_0y IS NOT NULL OR
            fc.total_assets_0y IS NOT NULL OR
            fc.total_equity_0y IS NOT NULL OR
            r.current_ratio IS NOT NULL OR
            r.quick_ratio IS NOT NULL OR
            r.debt_to_equity IS NOT NULL OR
            r.gross_margin IS NOT NULL OR
            r.operating_margin IS NOT NULL OR
            r.net_margin IS NOT NULL
        )
    )
    SELECT * FROM base_features
    """
    
    # Execute query and save to CSV
    financial_df = conn.execute(financial_query).df()
    
    # Convert raw labels to categorical classes (using same bins as text features)
    financial_df['label'] = convert_to_categorical_labels(financial_df['raw_label'])
    financial_df = financial_df.drop('raw_label', axis=1)
    
    # Basic data quality checks for financial features
    logger.info("\n📊 Financial Features Quality Report:")
    logger.info(f"- Total samples: {len(financial_df)}")
    logger.info(f"- Features: {len(financial_df.columns)-2}")  # -2 for setup_id and label
    
    # Check missing values
    missing = financial_df.drop(['setup_id', 'label'], axis=1).isnull().sum()
    logger.info("\nMissing Values:")
    for col in missing[missing > 0].index:
        pct_missing = (missing[col] / len(financial_df)) * 100
        logger.info(f"  - {col}: {missing[col]} missing ({pct_missing:.1f}%)")
    
    # Check label distribution
    logger.info("\nLabel Distribution:")
    label_dist = financial_df['label'].value_counts()
    for label, count in label_dist.items():
        logger.info(f"  - {label}: {count} samples ({count/len(financial_df)*100:.1f}%)")
    
    # Save financial features to CSV
    financial_output_file = output_dir / f"financial_ml_features_training_{timestamp}.csv"
    financial_df.to_csv(financial_output_file, index=False)
    logger.info(f"\n✅ Financial features saved to: {financial_output_file}")
    
    return training_setup_ids

def export_prediction_features(conn: duckdb.DuckDBPyConnection, output_dir: Path, training_setup_ids: list):
    """Export prediction features (excluding training setup IDs)"""
    logger.info("\n🔮 Exporting prediction features...")
    
    # Query to get prediction data (excluding training setup IDs)
    text_query = """
    WITH setup_labels AS (
        SELECT setup_id,
               AVG(outperformance_day) as outperformance_10d
        FROM daily_labels
        WHERE day_number <= 10
        GROUP BY setup_id
        HAVING COUNT(*) = 10  -- Ensure we have all 10 days
    ),
    prediction_features AS (
        SELECT t.*
        FROM text_ml_features_training t
        WHERE t.setup_id NOT IN (SELECT setup_id FROM setup_labels)
        AND (
            -- Ensure we have at least some non-null features
            t.count_financial_results IS NOT NULL OR
            t.sentiment_score_financial_results IS NOT NULL OR
            t.count_corporate_actions IS NOT NULL OR
            t.posts_avg_sentiment IS NOT NULL OR
            t.post_count IS NOT NULL OR
            t.recommendation_count IS NOT NULL
        )
        LIMIT 50  -- Get 50 setup IDs for prediction
    )
    SELECT * FROM prediction_features
    """
    
    # Execute query and save to CSV
    text_df = conn.execute(text_query).df()
    
    # Drop label column if it exists
    if 'label' in text_df.columns:
        text_df = text_df.drop('label', axis=1)
    
    # Drop analyst features (100% missing)
    analyst_features = [
        'avg_price_target',
        'price_target_vs_current',
        'price_target_spread'
    ]
    text_df = text_df.drop(analyst_features, axis=1)
    logger.info("\n🗑️ Dropped analyst features (100% missing)")
    
    # Basic data quality checks
    logger.info("\n📊 Text Features Prediction Quality Report:")
    logger.info(f"- Total samples: {len(text_df)}")
    logger.info(f"- Features: {len(text_df.columns)-1}")  # -1 for setup_id
    
    # Save prediction setup IDs for reference
    prediction_setup_ids = set(text_df['setup_id'])
    
    # Verify no overlap with training
    training_setup_ids_set = set(training_setup_ids)
    overlap = prediction_setup_ids & training_setup_ids_set
    if overlap:
        raise ValueError(f"Found {len(overlap)} setup IDs that overlap with training data!")
    
    # Save text features to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_output_file = output_dir / f"text_ml_features_prediction_{timestamp}.csv"
    text_df.to_csv(text_output_file, index=False)
    logger.info(f"\n✅ Text features saved to: {text_output_file}")
    
    # Query to get financial features for prediction
    financial_query = """
    WITH historical_fundamentals AS (
        SELECT 
            s.setup_id,
            s.lse_ticker,
            s.spike_timestamp,
            -- Current year (0Y)
            f0.total_revenue as revenue_0y,
            f0.gross_profit as gross_profit_0y,
            f0.operating_income as operating_income_0y,
            f0.net_income as net_income_0y,
            f0.ebitda as ebitda_0y,
            f0.basic_eps as basic_eps_0y,
            f0.diluted_eps as diluted_eps_0y,
            f0.total_assets as total_assets_0y,
            f0.total_debt as total_debt_0y,
            f0.total_equity as total_equity_0y,
            f0.cash_and_equivalents as cash_and_equivalents_0y,
            f0.current_assets as current_assets_0y,
            f0.current_liabilities as current_liabilities_0y,
            f0.working_capital as working_capital_0y,
            f0.property_plant_equipment as property_plant_equipment_0y,
            f0.operating_cash_flow as operating_cash_flow_0y,
            f0.free_cash_flow as free_cash_flow_0y,
            f0.capital_expenditure as capital_expenditure_0y,
            f0.financing_cash_flow as financing_cash_flow_0y,
            f0.investing_cash_flow as investing_cash_flow_0y,
            
            -- Previous year (-1Y)
            f1.total_revenue as revenue_1y,
            f1.gross_profit as gross_profit_1y,
            f1.operating_income as operating_income_1y,
            f1.net_income as net_income_1y,
            f1.ebitda as ebitda_1y,
            f1.total_assets as total_assets_1y,
            f1.total_debt as total_debt_1y,
            f1.total_equity as total_equity_1y,
            f1.operating_cash_flow as operating_cash_flow_1y,
            f1.free_cash_flow as free_cash_flow_1y,
            
            -- Two years ago (-2Y)
            f2.total_revenue as revenue_2y,
            f2.gross_profit as gross_profit_2y,
            f2.operating_income as operating_income_2y,
            f2.net_income as net_income_2y,
            f2.ebitda as ebitda_2y,
            f2.total_assets as total_assets_2y,
            f2.total_debt as total_debt_2y,
            f2.total_equity as total_equity_2y,
            f2.operating_cash_flow as operating_cash_flow_2y,
            f2.free_cash_flow as free_cash_flow_2y,
            
            -- Three years ago (-3Y)
            f3.total_revenue as revenue_3y,
            f3.gross_profit as gross_profit_3y,
            f3.operating_income as operating_income_3y,
            f3.net_income as net_income_3y,
            f3.ebitda as ebitda_3y,
            f3.total_assets as total_assets_3y,
            f3.total_debt as total_debt_3y,
            f3.total_equity as total_equity_3y,
            f3.operating_cash_flow as operating_cash_flow_3y,
            f3.free_cash_flow as free_cash_flow_3y,
            
            ROW_NUMBER() OVER (PARTITION BY s.setup_id ORDER BY f0.date DESC) as row_num
        FROM setups s
        -- Current year
        LEFT JOIN fundamentals f0 
            ON s.lse_ticker || '.L' = f0.ticker
            AND f0.date <= s.spike_timestamp
        -- Previous year
        LEFT JOIN fundamentals f1
            ON s.lse_ticker || '.L' = f1.ticker
            AND f1.date <= s.spike_timestamp - INTERVAL '1 year'
            AND f1.date > s.spike_timestamp - INTERVAL '1.5 years'
        -- Two years ago
        LEFT JOIN fundamentals f2
            ON s.lse_ticker || '.L' = f2.ticker
            AND f2.date <= s.spike_timestamp - INTERVAL '2 years'
            AND f2.date > s.spike_timestamp - INTERVAL '2.5 years'
        -- Three years ago
        LEFT JOIN fundamentals f3
            ON s.lse_ticker || '.L' = f3.ticker
            AND f3.date <= s.spike_timestamp - INTERVAL '3 years'
            AND f3.date > s.spike_timestamp - INTERVAL '3.5 years'
        WHERE s.setup_id = ANY(?)
    ),
    fundamentals_clean AS (
        SELECT * FROM historical_fundamentals WHERE row_num = 1
    ),
    growth_metrics AS (
        SELECT 
            setup_id,
            
            -- Revenue growth rates (YoY)
            CASE WHEN revenue_1y > 0 THEN (revenue_0y - revenue_1y) / CAST(revenue_1y AS FLOAT) ELSE NULL END as revenue_growth_1y,
            CASE WHEN revenue_2y > 0 THEN (revenue_1y - revenue_2y) / CAST(revenue_2y AS FLOAT) ELSE NULL END as revenue_growth_2y,
            CASE WHEN revenue_3y > 0 THEN (revenue_2y - revenue_3y) / CAST(revenue_3y AS FLOAT) ELSE NULL END as revenue_growth_3y,
            
            -- Operating income growth
            CASE WHEN operating_income_1y > 0 THEN (operating_income_0y - operating_income_1y) / CAST(operating_income_1y AS FLOAT) ELSE NULL END as operating_income_growth_1y,
            CASE WHEN operating_income_2y > 0 THEN (operating_income_1y - operating_income_2y) / CAST(operating_income_2y AS FLOAT) ELSE NULL END as operating_income_growth_2y,
            CASE WHEN operating_income_3y > 0 THEN (operating_income_2y - operating_income_3y) / CAST(operating_income_3y AS FLOAT) ELSE NULL END as operating_income_growth_3y,
            
            -- Net income growth (handle negative values)
            CASE WHEN net_income_1y != 0 THEN (net_income_0y - net_income_1y) / CAST(ABS(net_income_1y) AS FLOAT) ELSE NULL END as net_income_growth_1y,
            CASE WHEN net_income_2y != 0 THEN (net_income_1y - net_income_2y) / CAST(ABS(net_income_2y) AS FLOAT) ELSE NULL END as net_income_growth_2y,
            CASE WHEN net_income_3y != 0 THEN (net_income_2y - net_income_3y) / CAST(ABS(net_income_3y) AS FLOAT) ELSE NULL END as net_income_growth_3y,
            
            -- EBITDA growth
            CASE WHEN ebitda_1y > 0 THEN (ebitda_0y - ebitda_1y) / CAST(ebitda_1y AS FLOAT) ELSE NULL END as ebitda_growth_1y,
            CASE WHEN ebitda_2y > 0 THEN (ebitda_1y - ebitda_2y) / CAST(ebitda_2y AS FLOAT) ELSE NULL END as ebitda_growth_2y,
            CASE WHEN ebitda_3y > 0 THEN (ebitda_2y - ebitda_3y) / CAST(ebitda_3y AS FLOAT) ELSE NULL END as ebitda_growth_3y,
            
            -- Operating Cash Flow growth
            CASE WHEN operating_cash_flow_1y != 0 THEN (operating_cash_flow_0y - operating_cash_flow_1y) / CAST(ABS(operating_cash_flow_1y) AS FLOAT) ELSE NULL END as ocf_growth_1y,
            CASE WHEN operating_cash_flow_2y != 0 THEN (operating_cash_flow_1y - operating_cash_flow_2y) / CAST(ABS(operating_cash_flow_2y) AS FLOAT) ELSE NULL END as ocf_growth_2y,
            CASE WHEN operating_cash_flow_3y != 0 THEN (operating_cash_flow_2y - operating_cash_flow_3y) / CAST(ABS(operating_cash_flow_3y) AS FLOAT) ELSE NULL END as ocf_growth_3y
        FROM fundamentals_clean
    ),
    rolling_stats AS (
        SELECT
            setup_id,
            
            -- 3-year moving averages (where data available)
            CASE WHEN revenue_0y IS NOT NULL AND revenue_1y IS NOT NULL AND revenue_2y IS NOT NULL 
                 THEN (revenue_0y + revenue_1y + revenue_2y) / 3.0 ELSE NULL END as revenue_3y_avg,
            CASE WHEN operating_income_0y IS NOT NULL AND operating_income_1y IS NOT NULL AND operating_income_2y IS NOT NULL 
                 THEN (operating_income_0y + operating_income_1y + operating_income_2y) / 3.0 ELSE NULL END as operating_income_3y_avg,
            CASE WHEN net_income_0y IS NOT NULL AND net_income_1y IS NOT NULL AND net_income_2y IS NOT NULL 
                 THEN (net_income_0y + net_income_1y + net_income_2y) / 3.0 ELSE NULL END as net_income_3y_avg,
            CASE WHEN ebitda_0y IS NOT NULL AND ebitda_1y IS NOT NULL AND ebitda_2y IS NOT NULL 
                 THEN (ebitda_0y + ebitda_1y + ebitda_2y) / 3.0 ELSE NULL END as ebitda_3y_avg,
            
            -- Financial health evolution - margin trends
            CASE WHEN revenue_0y > 0 THEN operating_income_0y / CAST(revenue_0y AS FLOAT) ELSE NULL END as operating_margin_0y,
            CASE WHEN revenue_1y > 0 THEN operating_income_1y / CAST(revenue_1y AS FLOAT) ELSE NULL END as operating_margin_1y,
            CASE WHEN revenue_2y > 0 THEN operating_income_2y / CAST(revenue_2y AS FLOAT) ELSE NULL END as operating_margin_2y,
            
            CASE WHEN revenue_0y > 0 THEN net_income_0y / CAST(revenue_0y AS FLOAT) ELSE NULL END as net_margin_0y,
            CASE WHEN revenue_1y > 0 THEN net_income_1y / CAST(revenue_1y AS FLOAT) ELSE NULL END as net_margin_1y,
            CASE WHEN revenue_2y > 0 THEN net_income_2y / CAST(revenue_2y AS FLOAT) ELSE NULL END as net_margin_2y,
            
            -- Leverage evolution
            CASE WHEN total_equity_0y > 0 THEN total_debt_0y / CAST(total_equity_0y AS FLOAT) ELSE NULL END as debt_to_equity_0y,
            CASE WHEN total_equity_1y > 0 THEN total_debt_1y / CAST(total_equity_1y AS FLOAT) ELSE NULL END as debt_to_equity_1y,
            CASE WHEN total_equity_2y > 0 THEN total_debt_2y / CAST(total_equity_2y AS FLOAT) ELSE NULL END as debt_to_equity_2y
        FROM fundamentals_clean
    ),
    trend_indicators AS (
        SELECT
            fc.setup_id,
            rs.revenue_3y_avg,
            rs.operating_income_3y_avg,
            rs.net_income_3y_avg,
            rs.ebitda_3y_avg,
            rs.operating_margin_0y,
            rs.operating_margin_1y,
            rs.operating_margin_2y,
            rs.net_margin_0y,
            rs.net_margin_1y,
            rs.net_margin_2y,
            rs.debt_to_equity_0y,
            rs.debt_to_equity_1y,
            rs.debt_to_equity_2y,
            
            -- Consecutive growth indicators
            CASE 
                WHEN fc.revenue_0y > fc.revenue_1y 
                 AND fc.revenue_1y > fc.revenue_2y 
                 AND fc.revenue_2y > fc.revenue_3y THEN 3
                WHEN fc.revenue_0y > fc.revenue_1y 
                 AND fc.revenue_1y > fc.revenue_2y THEN 2
                WHEN fc.revenue_0y > fc.revenue_1y THEN 1
                ELSE 0 
            END as revenue_consecutive_growth_years,
            
            -- Growth acceleration/deceleration
            CASE
                WHEN gm.revenue_growth_1y IS NOT NULL AND gm.revenue_growth_2y IS NOT NULL AND gm.revenue_growth_3y IS NOT NULL THEN
                    CASE 
                        WHEN gm.revenue_growth_1y > gm.revenue_growth_2y AND gm.revenue_growth_2y > gm.revenue_growth_3y THEN 1
                        WHEN gm.revenue_growth_1y < gm.revenue_growth_2y AND gm.revenue_growth_2y < gm.revenue_growth_3y THEN -1
                        ELSE 0
                    END
                ELSE NULL
            END as revenue_growth_acceleration,
            
            -- Operating margin improvement trend
            CASE
                WHEN rs.operating_margin_0y IS NOT NULL AND rs.operating_margin_1y IS NOT NULL AND rs.operating_margin_2y IS NOT NULL THEN
                    CASE 
                        WHEN rs.operating_margin_0y > rs.operating_margin_1y AND rs.operating_margin_1y > rs.operating_margin_2y THEN 1
                        WHEN rs.operating_margin_0y < rs.operating_margin_1y AND rs.operating_margin_1y < rs.operating_margin_2y THEN -1
                        ELSE 0
                    END
                ELSE NULL
            END as operating_margin_improvement_trend,
            
            -- Net margin improvement trend
            CASE
                WHEN rs.net_margin_0y IS NOT NULL AND rs.net_margin_1y IS NOT NULL AND rs.net_margin_2y IS NOT NULL THEN
                    CASE 
                        WHEN rs.net_margin_0y > rs.net_margin_1y AND rs.net_margin_1y > rs.net_margin_2y THEN 1
                        WHEN rs.net_margin_0y < rs.net_margin_1y AND rs.net_margin_1y < rs.net_margin_2y THEN -1
                        ELSE 0
                    END
                ELSE NULL
            END as net_margin_improvement_trend,
            
            -- Leverage trend (1 = improving/decreasing debt, -1 = worsening/increasing debt)
            CASE
                WHEN rs.debt_to_equity_0y IS NOT NULL AND rs.debt_to_equity_1y IS NOT NULL AND rs.debt_to_equity_2y IS NOT NULL THEN
                    CASE 
                        WHEN rs.debt_to_equity_0y < rs.debt_to_equity_1y AND rs.debt_to_equity_1y < rs.debt_to_equity_2y THEN 1
                        WHEN rs.debt_to_equity_0y > rs.debt_to_equity_1y AND rs.debt_to_equity_1y > rs.debt_to_equity_2y THEN -1
                        ELSE 0
                    END
                ELSE NULL
            END as leverage_improvement_trend,
            
            -- Growth stability (coefficient of variation for revenue growth)
            CASE 
                WHEN gm.revenue_growth_1y IS NOT NULL AND gm.revenue_growth_2y IS NOT NULL AND gm.revenue_growth_3y IS NOT NULL THEN
                    SQRT(
                        POWER(gm.revenue_growth_1y - (gm.revenue_growth_1y + gm.revenue_growth_2y + gm.revenue_growth_3y)/3.0, 2) +
                        POWER(gm.revenue_growth_2y - (gm.revenue_growth_1y + gm.revenue_growth_2y + gm.revenue_growth_3y)/3.0, 2) +
                        POWER(gm.revenue_growth_3y - (gm.revenue_growth_1y + gm.revenue_growth_2y + gm.revenue_growth_3y)/3.0, 2)
                    ) / 3.0
                ELSE NULL
            END as revenue_growth_stability
        FROM fundamentals_clean fc
        LEFT JOIN growth_metrics gm USING (setup_id)
        LEFT JOIN rolling_stats rs USING (setup_id)
    ),
    latest_ratios AS (
        SELECT 
            s.setup_id,
            fr.*,
            ROW_NUMBER() OVER (PARTITION BY s.setup_id ORDER BY fr.period_end DESC) as row_num
        FROM setups s
        LEFT JOIN financial_ratios fr 
            ON s.lse_ticker || '.L' = fr.ticker
            AND fr.period_end <= s.spike_timestamp
        WHERE s.setup_id = ANY(?)
    ),
    ratios_clean AS (
        SELECT setup_id,
               current_ratio, quick_ratio, cash_ratio, debt_to_equity,
               debt_to_assets, equity_ratio, gross_margin, operating_margin,
               net_margin, roe, roa, roic, asset_turnover, inventory_turnover,
               receivables_turnover, pe_ratio as price_to_earnings,
               pb_ratio as price_to_book, ps_ratio as price_to_sales,
               ev_ebitda, book_value_per_share, revenue_per_share,
               cash_per_share
        FROM latest_ratios
        WHERE row_num = 1
    ),
    base_features AS (
        SELECT 
            COALESCE(fc.setup_id, r.setup_id, ti.setup_id) as setup_id,
            
            -- Current year raw fundamentals
            fc.revenue_0y as total_revenue,
            fc.gross_profit_0y as gross_profit,
            fc.operating_income_0y as operating_income,
            fc.net_income_0y as net_income,
            fc.ebitda_0y as ebitda,
            fc.basic_eps_0y as basic_eps,
            fc.diluted_eps_0y as diluted_eps,
            fc.total_assets_0y as total_assets,
            fc.total_debt_0y as total_debt,
            fc.total_equity_0y as total_equity,
            fc.cash_and_equivalents_0y as cash_and_equivalents,
            fc.current_assets_0y as current_assets,
            fc.current_liabilities_0y as current_liabilities,
            fc.working_capital_0y as working_capital,
            fc.property_plant_equipment_0y as property_plant_equipment,
            fc.operating_cash_flow_0y as operating_cash_flow,
            fc.free_cash_flow_0y as free_cash_flow,
            fc.capital_expenditure_0y as capital_expenditure,
            fc.financing_cash_flow_0y as financing_cash_flow,
            fc.investing_cash_flow_0y as investing_cash_flow,
            
            -- Historical growth metrics
            gm.revenue_growth_1y,
            gm.revenue_growth_2y,
            gm.revenue_growth_3y,
            gm.operating_income_growth_1y,
            gm.operating_income_growth_2y,
            gm.operating_income_growth_3y,
            gm.net_income_growth_1y,
            gm.net_income_growth_2y,
            gm.net_income_growth_3y,
            gm.ebitda_growth_1y,
            gm.ebitda_growth_2y,
            gm.ebitda_growth_3y,
            gm.ocf_growth_1y,
            gm.ocf_growth_2y,
            gm.ocf_growth_3y,
            
            -- Rolling statistics
            ti.revenue_3y_avg,
            ti.operating_income_3y_avg,
            ti.net_income_3y_avg,
            ti.ebitda_3y_avg,
            
            -- Trend indicators
            ti.revenue_consecutive_growth_years,
            ti.revenue_growth_acceleration,
            ti.operating_margin_improvement_trend,
            ti.net_margin_improvement_trend,
            ti.leverage_improvement_trend,
            ti.revenue_growth_stability,
            
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
            r.price_to_sales,
            r.ev_ebitda,
            r.book_value_per_share,
            r.revenue_per_share,
            r.cash_per_share
            
        FROM fundamentals_clean fc
        FULL OUTER JOIN ratios_clean r USING (setup_id)
        LEFT JOIN growth_metrics gm USING (setup_id)
        LEFT JOIN trend_indicators ti USING (setup_id)
        WHERE (
            -- Ensure we have at least some non-null features
            fc.revenue_0y IS NOT NULL OR
            fc.operating_income_0y IS NOT NULL OR
            fc.net_income_0y IS NOT NULL OR
            r.current_ratio IS NOT NULL OR
            r.quick_ratio IS NOT NULL OR
            r.debt_to_equity IS NOT NULL
        )
    )
    SELECT * FROM base_features
    """
    
    # Execute query and save to CSV
    financial_df = conn.execute(financial_query, [list(prediction_setup_ids), list(prediction_setup_ids), list(prediction_setup_ids)]).df()
    
    # Basic data quality checks
    logger.info("\n📊 Financial Features Prediction Quality Report:")
    logger.info(f"- Total samples: {len(financial_df)}")
    logger.info(f"- Features: {len(financial_df.columns)-1}")  # -1 for setup_id
    
    # Save financial features to CSV
    financial_output_file = output_dir / f"financial_ml_features_prediction_{timestamp}.csv"
    financial_df.to_csv(financial_output_file, index=False)
    logger.info(f"\n✅ Financial features saved to: {financial_output_file}")
    
    # Save setup IDs for reference
    setup_ids_file = output_dir / f"prediction_setup_ids_{timestamp}.txt"
    with open(setup_ids_file, 'w') as f:
        for setup_id in sorted(prediction_setup_ids):
            f.write(f"{setup_id}\n")
    logger.info(f"✅ Prediction setup IDs saved to: {setup_ids_file}")

def main():
    """Export ML features for training and prediction"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export ML features')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--output-dir', default='data/ml_training',
                       help='Directory to save exported features')
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        conn = duckdb.connect(args.db_path)
        
        # Export training features
        training_setup_ids = export_training_features(conn, output_dir)
        
        # Export prediction features
        export_prediction_features(conn, output_dir, training_setup_ids)
        
        logger.info("\n🎉 Feature Export Complete!")
        
    except Exception as e:
        logger.error(f"❌ Feature export failed: {str(e)}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main() 