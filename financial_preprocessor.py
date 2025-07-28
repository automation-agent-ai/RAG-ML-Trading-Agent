#!/usr/bin/env python3
"""
Financial Data Preprocessor

This module provides comprehensive financial data preprocessing for ML pipelines:
1. Extracts raw financial data from DuckDB
2. Calculates comprehensive financial ratios (P&L/revenue, balance sheet/total assets)  
3. Handles imputation and scaling consistently
4. Saves preprocessing parameters for prediction
5. Ensures data quality through validation checks

Features extracted:
- Profitability ratios (ROE, ROA, profit margins)
- Liquidity ratios (current ratio, quick ratio)
- Leverage ratios (debt-to-equity, debt-to-assets)
- Efficiency ratios (asset turnover, working capital ratios)
- Growth metrics (revenue growth, earnings growth, 1-3 years)
- Cash flow metrics (FCF yield, OCF ratios)
- Per-share metrics (EPS, cash flow per share, book value per share)
- Valuation ratios (PE ratio, PB ratio, EV/EBITDA)
- Trend indicators and data quality scores
"""

import os
import logging
import pickle
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FinancialMetrics:
    """Container for financial metrics and ratios"""
    setup_id: str
    ticker: str
    
    # Raw fundamentals (current year)
    total_revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    ebitda: Optional[float] = None
    basic_eps: Optional[float] = None
    diluted_eps: Optional[float] = None
    total_assets: Optional[float] = None
    total_debt: Optional[float] = None
    total_equity: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    current_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    working_capital: Optional[float] = None
    property_plant_equipment: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    free_cash_flow: Optional[float] = None
    capital_expenditure: Optional[float] = None
    financing_cash_flow: Optional[float] = None
    investing_cash_flow: Optional[float] = None
    
    # Profitability ratios (calculated from P&L / revenue)
    gross_profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_profit_margin: Optional[float] = None
    ebitda_margin: Optional[float] = None
    
    # Return ratios (calculated from balance sheet / total assets)
    return_on_assets: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_invested_capital: Optional[float] = None
    
    # Asset efficiency ratios (normalized by total assets)
    asset_turnover: Optional[float] = None
    working_capital_to_assets: Optional[float] = None
    cash_to_assets: Optional[float] = None
    debt_to_assets: Optional[float] = None
    
    # Liquidity ratios
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    cash_ratio: Optional[float] = None
    
    # Leverage ratios
    debt_to_equity: Optional[float] = None
    equity_ratio: Optional[float] = None
    debt_to_ebitda: Optional[float] = None
    
    # Cash flow ratios
    operating_cash_flow_ratio: Optional[float] = None
    free_cash_flow_yield: Optional[float] = None
    cash_flow_to_debt: Optional[float] = None
    capex_to_revenue: Optional[float] = None
    
    # Per-share metrics
    revenue_per_share: Optional[float] = None
    operating_cash_flow_per_share: Optional[float] = None
    free_cash_flow_per_share: Optional[float] = None
    book_value_per_share: Optional[float] = None
    
    # Valuation ratios (from financial_ratios table)
    price_earnings_ratio: Optional[float] = None
    price_to_book_ratio: Optional[float] = None
    price_to_sales_ratio: Optional[float] = None
    enterprise_value_to_ebitda: Optional[float] = None
    
    # Growth metrics (1-3 years)
    revenue_growth_1y: Optional[float] = None
    revenue_growth_2y: Optional[float] = None
    revenue_growth_3y: Optional[float] = None
    operating_income_growth_1y: Optional[float] = None
    operating_income_growth_2y: Optional[float] = None
    operating_income_growth_3y: Optional[float] = None
    net_income_growth_1y: Optional[float] = None
    net_income_growth_2y: Optional[float] = None
    net_income_growth_3y: Optional[float] = None
    ebitda_growth_1y: Optional[float] = None
    ebitda_growth_2y: Optional[float] = None
    ebitda_growth_3y: Optional[float] = None
    operating_cash_flow_growth_1y: Optional[float] = None
    operating_cash_flow_growth_2y: Optional[float] = None
    operating_cash_flow_growth_3y: Optional[float] = None
    
    # Average growth metrics
    revenue_3y_avg: Optional[float] = None
    operating_income_3y_avg: Optional[float] = None
    net_income_3y_avg: Optional[float] = None
    ebitda_3y_avg: Optional[float] = None
    
    # Trend indicators
    revenue_consecutive_growth_years: Optional[int] = None
    revenue_growth_acceleration: Optional[int] = None
    operating_margin_improvement_trend: Optional[int] = None
    net_margin_improvement_trend: Optional[int] = None
    leverage_improvement_trend: Optional[int] = None
    revenue_growth_stability: Optional[float] = None
    
    # Data quality indicators
    data_completeness_score: Optional[float] = None
    financial_data_age_days: Optional[int] = None
    fundamental_data_quality: Optional[float] = None
    ratios_data_quality: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation"""
        return asdict(self)

class FinancialDataValidator:
    """Validates financial data quality and consistency"""
    
    @staticmethod
    def validate_ticker_format(ticker: str, table_name: str) -> str:
        """
        Validate and normalize ticker format for different tables
        
        Args:
            ticker: Original ticker (e.g., 'ABC' from setups.lse_ticker)
            table_name: Target table name ('fundamentals' or 'financial_ratios')
            
        Returns:
            Normalized ticker for the table
        """
        if not ticker:
            return None
        
        # fundamentals and financial_ratios tables use .L suffix for LSE tickers
        if table_name in ['fundamentals', 'financial_ratios']:
            if not ticker.endswith('.L'):
                return f"{ticker}.L"
        
        return ticker
    
    @staticmethod
    def validate_financial_data(data: Dict) -> float:
        """
        Calculate data quality score based on completeness and consistency
        
        Args:
            data: Dictionary of financial data
            
        Returns:
            Quality score between 0 and 1
        """
        if not data:
            return 0.0
        
        # Key fundamental metrics for quality assessment
        key_metrics = [
            'total_revenue', 'net_income', 'total_assets', 'total_equity',
            'basic_eps', 'operating_cash_flow'
        ]
        
        # Count non-null key metrics
        non_null_count = sum(1 for metric in key_metrics if data.get(metric) is not None)
        completeness_score = non_null_count / len(key_metrics)
        
        # Check for reasonable value ranges (basic sanity checks)
        consistency_score = 1.0
        
        # Revenue should generally be positive
        if data.get('total_revenue') is not None and data.get('total_revenue') < 0:
            consistency_score -= 0.1
        
        # Assets should be positive
        if data.get('total_assets') is not None and data.get('total_assets') <= 0:
            consistency_score -= 0.2
        
        # Equity can be negative but warn if extreme
        if data.get('total_equity') is not None and abs(data.get('total_equity')) < 1e-6:
            consistency_score -= 0.1
        
        return min(1.0, max(0.0, completeness_score * consistency_score))

class FinancialPreprocessor:
    """
    Comprehensive financial data preprocessor with consistent scaling and imputation
    """
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        preprocessing_params_path: str = "models/financial/preprocessing_params.pkl"
    ):
        """
        Initialize the financial preprocessor
        
        Args:
            db_path: Path to DuckDB database
            preprocessing_params_path: Path to save/load preprocessing parameters
        """
        self.db_path = db_path
        self.preprocessing_params_path = Path(preprocessing_params_path)
        self.preprocessing_params_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Preprocessing components
        self.imputer = None
        self.scaler = None
        self.feature_columns = None
        self.preprocessing_fitted = False
        
        # Data validator
        self.validator = FinancialDataValidator()
    
    def extract_comprehensive_financial_data(
        self,
        setup_ids: List[str],
        spike_timestamps: Optional[Dict[str, datetime]] = None
    ) -> List[FinancialMetrics]:
        """
        Extract comprehensive financial data for multiple setup IDs
        
        Args:
            setup_ids: List of setup IDs to process
            spike_timestamps: Optional dictionary mapping setup_id to spike_timestamp
            
        Returns:
            List of FinancialMetrics objects
        """
        logger.info(f"💰 Extracting comprehensive financial data for {len(setup_ids)} setups...")
        
        conn = None
        try:
            conn = duckdb.connect(self.db_path)
            
            # Get setup information if spike_timestamps not provided
            if spike_timestamps is None:
                spike_timestamps = self._get_spike_timestamps(conn, setup_ids)
            
            results = []
            
            for i, setup_id in enumerate(setup_ids):
                if i % 100 == 0:
                    logger.info(f"Processing setup {i+1}/{len(setup_ids)}: {setup_id}")
                
                spike_timestamp = spike_timestamps.get(setup_id)
                if not spike_timestamp:
                    logger.warning(f"No spike timestamp found for setup {setup_id}")
                    continue
                
                # Extract financial metrics
                metrics = self._extract_single_setup_metrics(conn, setup_id, spike_timestamp)
                if metrics:
                    results.append(metrics)
            
            logger.info(f"✅ Extracted financial data for {len(results)}/{len(setup_ids)} setups successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error extracting financial data: {str(e)}")
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    def _get_spike_timestamps(self, conn, setup_ids: List[str]) -> Dict[str, datetime]:
        """Get spike timestamps for setup IDs"""
        placeholders = ','.join(['?' for _ in setup_ids])
        query = f"""
        SELECT setup_id, lse_ticker, spike_timestamp 
        FROM setups 
        WHERE setup_id IN ({placeholders})
        """
        
        result = conn.execute(query, setup_ids).fetchall()
        return {row[0]: row[2] for row in result if row[2]}
    
    def _extract_single_setup_metrics(
        self,
        conn,
        setup_id: str,
        spike_timestamp: datetime
    ) -> Optional[FinancialMetrics]:
        """Extract financial metrics for a single setup"""
        try:
            # Get setup info
            setup_info = conn.execute(
                "SELECT lse_ticker, yahoo_ticker FROM setups WHERE setup_id = ?",
                [setup_id]
            ).fetchone()
            
            if not setup_info:
                return None
            
            lse_ticker, yahoo_ticker = setup_info
            ticker = lse_ticker or yahoo_ticker
            
            if not ticker:
                return None
            
            # Initialize metrics object
            metrics = FinancialMetrics(setup_id=setup_id, ticker=ticker)
            
            # Extract historical fundamentals (0-3 years)
            historical_data = self._extract_historical_fundamentals(conn, ticker, spike_timestamp)
            
            # Extract current ratios
            ratios_data = self._extract_financial_ratios(conn, ticker, spike_timestamp)
            
            # Calculate all metrics
            self._calculate_comprehensive_metrics(metrics, historical_data, ratios_data, spike_timestamp)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting metrics for setup {setup_id}: {str(e)}")
            return None
    
    def _extract_historical_fundamentals(
        self,
        conn,
        ticker: str,
        spike_timestamp: datetime
    ) -> Dict[str, Dict]:
        """Extract 4 years of historical fundamentals data"""
        normalized_ticker = self.validator.validate_ticker_format(ticker, 'fundamentals')
        
        query = """
        WITH historical_fundamentals AS (
            SELECT 
                *,
                CASE 
                    WHEN date <= ? THEN 0  -- Current year
                    WHEN date <= ? - INTERVAL '1 year' AND date > ? - INTERVAL '1.5 years' THEN 1  -- 1 year ago
                    WHEN date <= ? - INTERVAL '2 years' AND date > ? - INTERVAL '2.5 years' THEN 2  -- 2 years ago
                    WHEN date <= ? - INTERVAL '3 years' AND date > ? - INTERVAL '3.5 years' THEN 3  -- 3 years ago
                    ELSE -1
                END as year_offset,
                ROW_NUMBER() OVER (PARTITION BY 
                    CASE 
                        WHEN date <= ? THEN 0
                        WHEN date <= ? - INTERVAL '1 year' AND date > ? - INTERVAL '1.5 years' THEN 1
                        WHEN date <= ? - INTERVAL '2 years' AND date > ? - INTERVAL '2.5 years' THEN 2
                        WHEN date <= ? - INTERVAL '3 years' AND date > ? - INTERVAL '3.5 years' THEN 3
                        ELSE -1
                    END 
                    ORDER BY date DESC
                ) as rn
            FROM fundamentals 
            WHERE ticker = ?
            AND date <= ?
        )
        SELECT * FROM historical_fundamentals 
        WHERE year_offset >= 0 AND rn = 1
        ORDER BY year_offset
        """
        
        # Create parameter list with correct count (14 spike_timestamp + 1 ticker + 1 spike_timestamp = 16 total)
        params = [spike_timestamp] * 14 + [normalized_ticker, spike_timestamp]
        
        result = conn.execute(query, params).fetchall()
        
        # Organize by year offset
        historical = {}
        if result:
            cols = [desc[0] for desc in conn.description]
            for row in result:
                row_dict = dict(zip(cols, row))
                year_offset = row_dict['year_offset']
                historical[year_offset] = row_dict
        
        return historical
    
    def _extract_financial_ratios(
        self,
        conn,
        ticker: str,
        spike_timestamp: datetime
    ) -> Dict:
        """Extract most recent financial ratios"""
        normalized_ticker = self.validator.validate_ticker_format(ticker, 'financial_ratios')
        
        query = """
        SELECT *
        FROM financial_ratios 
        WHERE ticker = ?
        AND period_end <= ?
        ORDER BY period_end DESC
        LIMIT 1
        """
        
        result = conn.execute(query, [normalized_ticker, spike_timestamp]).fetchone()
        
        if result:
            cols = [desc[0] for desc in conn.description]
            return dict(zip(cols, result))
        
        return {}
    
    def _calculate_comprehensive_metrics(
        self,
        metrics: FinancialMetrics,
        historical_data: Dict[str, Dict],
        ratios_data: Dict,
        spike_timestamp: datetime
    ):
        """Calculate all financial metrics and ratios"""
        # Get current year data (year_offset = 0)
        current_data = historical_data.get(0, {})
        
        # Extract raw fundamentals (current year)
        self._extract_raw_fundamentals(metrics, current_data)
        
        # Calculate profitability ratios (P&L / revenue)
        self._calculate_profitability_ratios(metrics, current_data)
        
        # Calculate asset efficiency ratios (balance sheet / total assets)
        self._calculate_asset_efficiency_ratios(metrics, current_data)
        
        # Calculate liquidity and leverage ratios
        self._calculate_liquidity_leverage_ratios(metrics, current_data, ratios_data)
        
        # Calculate cash flow ratios
        self._calculate_cash_flow_ratios(metrics, current_data)
        
        # Calculate per-share metrics
        self._calculate_per_share_metrics(metrics, current_data, ratios_data)
        
        # Extract valuation ratios from ratios table
        self._extract_valuation_ratios(metrics, ratios_data)
        
        # Calculate growth metrics (1-3 years)
        self._calculate_growth_metrics(metrics, historical_data)
        
        # Calculate average metrics and trend indicators
        self._calculate_trend_indicators(metrics, historical_data)
        
        # Calculate data quality scores
        self._calculate_data_quality_scores(metrics, current_data, ratios_data, spike_timestamp)
    
    def _extract_raw_fundamentals(self, metrics: FinancialMetrics, current_data: Dict):
        """Extract raw fundamental data"""
        if not current_data:
            return
        
        # Income statement
        metrics.total_revenue = self._safe_float(current_data.get('total_revenue'))
        metrics.gross_profit = self._safe_float(current_data.get('gross_profit'))
        metrics.operating_income = self._safe_float(current_data.get('operating_income'))
        metrics.net_income = self._safe_float(current_data.get('net_income'))
        metrics.ebitda = self._safe_float(current_data.get('ebitda'))
        metrics.basic_eps = self._safe_float(current_data.get('basic_eps'))
        metrics.diluted_eps = self._safe_float(current_data.get('diluted_eps'))
        
        # Balance sheet
        metrics.total_assets = self._safe_float(current_data.get('total_assets'))
        metrics.total_debt = self._safe_float(current_data.get('total_debt'))
        metrics.total_equity = self._safe_float(current_data.get('total_equity'))
        metrics.cash_and_equivalents = self._safe_float(current_data.get('cash_and_equivalents'))
        metrics.current_assets = self._safe_float(current_data.get('current_assets'))
        metrics.current_liabilities = self._safe_float(current_data.get('current_liabilities'))
        metrics.working_capital = self._safe_float(current_data.get('working_capital'))
        metrics.property_plant_equipment = self._safe_float(current_data.get('property_plant_equipment'))
        
        # Cash flow statement
        metrics.operating_cash_flow = self._safe_float(current_data.get('operating_cash_flow'))
        metrics.free_cash_flow = self._safe_float(current_data.get('free_cash_flow'))
        metrics.capital_expenditure = self._safe_float(current_data.get('capital_expenditure'))
        metrics.financing_cash_flow = self._safe_float(current_data.get('financing_cash_flow'))
        metrics.investing_cash_flow = self._safe_float(current_data.get('investing_cash_flow'))
    
    def _calculate_profitability_ratios(self, metrics: FinancialMetrics, current_data: Dict):
        """Calculate profitability ratios (P&L metrics / revenue)"""
        revenue = metrics.total_revenue
        
        if revenue and revenue > 0:
            # Margin ratios
            if metrics.gross_profit is not None:
                metrics.gross_profit_margin = metrics.gross_profit / revenue
            
            if metrics.operating_income is not None:
                metrics.operating_margin = metrics.operating_income / revenue
            
            if metrics.net_income is not None:
                metrics.net_profit_margin = metrics.net_income / revenue
            
            if metrics.ebitda is not None:
                metrics.ebitda_margin = metrics.ebitda / revenue
    
    def _calculate_asset_efficiency_ratios(self, metrics: FinancialMetrics, current_data: Dict):
        """Calculate asset efficiency ratios (balance sheet / total assets)"""
        assets = metrics.total_assets
        
        if assets and assets > 0:
            # Return ratios
            if metrics.net_income is not None:
                metrics.return_on_assets = metrics.net_income / assets
            
            # Asset utilization
            if metrics.total_revenue is not None:
                metrics.asset_turnover = metrics.total_revenue / assets
            
            if metrics.working_capital is not None:
                metrics.working_capital_to_assets = metrics.working_capital / assets
            
            if metrics.cash_and_equivalents is not None:
                metrics.cash_to_assets = metrics.cash_and_equivalents / assets
            
            if metrics.total_debt is not None:
                metrics.debt_to_assets = metrics.total_debt / assets
        
        # Return on equity
        if metrics.total_equity and metrics.total_equity > 0 and metrics.net_income is not None:
            metrics.return_on_equity = metrics.net_income / metrics.total_equity
        
        # Return on invested capital (approximation)
        if (metrics.total_equity and metrics.total_debt and 
            metrics.net_income is not None and metrics.total_equity > 0):
            invested_capital = metrics.total_equity + metrics.total_debt
            if invested_capital > 0:
                metrics.return_on_invested_capital = metrics.net_income / invested_capital
    
    def _calculate_liquidity_leverage_ratios(self, metrics: FinancialMetrics, current_data: Dict, ratios_data: Dict):
        """Calculate liquidity and leverage ratios"""
        # Liquidity ratios
        if metrics.current_liabilities and metrics.current_liabilities > 0:
            if metrics.current_assets is not None:
                metrics.current_ratio = metrics.current_assets / metrics.current_liabilities
            
            if metrics.cash_and_equivalents is not None:
                metrics.cash_ratio = metrics.cash_and_equivalents / metrics.current_liabilities
            
            # Quick ratio (current assets - inventory) / current liabilities
            # Use quick_ratio from ratios table if available, otherwise approximate
            if ratios_data.get('quick_ratio') is not None:
                metrics.quick_ratio = self._safe_float(ratios_data.get('quick_ratio'))
            elif metrics.current_assets is not None:
                # Approximate as current assets / current liabilities (conservative)
                metrics.quick_ratio = metrics.current_assets / metrics.current_liabilities
        
        # Leverage ratios
        if metrics.total_equity and metrics.total_equity > 0:
            if metrics.total_debt is not None:
                metrics.debt_to_equity = metrics.total_debt / metrics.total_equity
            
            if metrics.total_assets and metrics.total_assets > 0:
                metrics.equity_ratio = metrics.total_equity / metrics.total_assets
        
        # Debt to EBITDA
        if metrics.ebitda and metrics.ebitda > 0 and metrics.total_debt is not None:
            metrics.debt_to_ebitda = metrics.total_debt / metrics.ebitda
    
    def _calculate_cash_flow_ratios(self, metrics: FinancialMetrics, current_data: Dict):
        """Calculate cash flow ratios"""
        # Operating cash flow ratio
        if metrics.operating_cash_flow and metrics.total_revenue and metrics.total_revenue > 0:
            metrics.operating_cash_flow_ratio = metrics.operating_cash_flow / metrics.total_revenue
        
        # Free cash flow yield
        if metrics.free_cash_flow and metrics.total_revenue and metrics.total_revenue > 0:
            metrics.free_cash_flow_yield = metrics.free_cash_flow / metrics.total_revenue
        
        # Cash flow to debt
        if metrics.operating_cash_flow and metrics.total_debt and metrics.total_debt > 0:
            metrics.cash_flow_to_debt = metrics.operating_cash_flow / metrics.total_debt
        
        # CapEx to revenue
        if metrics.capital_expenditure and metrics.total_revenue and metrics.total_revenue > 0:
            metrics.capex_to_revenue = abs(metrics.capital_expenditure) / metrics.total_revenue
    
    def _calculate_per_share_metrics(self, metrics: FinancialMetrics, current_data: Dict, ratios_data: Dict):
        """Calculate per-share metrics"""
        # Use from ratios table if available
        metrics.revenue_per_share = self._safe_float(ratios_data.get('revenue_per_share'))
        metrics.book_value_per_share = self._safe_float(ratios_data.get('book_value_per_share'))
        
        # Calculate OCF and FCF per share if shares outstanding is available
        shares_outstanding = self._safe_float(current_data.get('shares_outstanding'))
        
        if shares_outstanding and shares_outstanding > 0:
            if metrics.operating_cash_flow is not None:
                metrics.operating_cash_flow_per_share = metrics.operating_cash_flow / shares_outstanding
            
            if metrics.free_cash_flow is not None:
                metrics.free_cash_flow_per_share = metrics.free_cash_flow / shares_outstanding
    
    def _extract_valuation_ratios(self, metrics: FinancialMetrics, ratios_data: Dict):
        """Extract valuation ratios from financial_ratios table"""
        if not ratios_data:
            return
        
        metrics.price_earnings_ratio = self._safe_float(ratios_data.get('pe_ratio'))
        metrics.price_to_book_ratio = self._safe_float(ratios_data.get('pb_ratio'))
        metrics.price_to_sales_ratio = self._safe_float(ratios_data.get('ps_ratio'))
        metrics.enterprise_value_to_ebitda = self._safe_float(ratios_data.get('ev_ebitda'))
    
    def _calculate_growth_metrics(self, metrics: FinancialMetrics, historical_data: Dict[str, Dict]):
        """Calculate comprehensive growth metrics (1-3 years)"""
        current = historical_data.get(0, {})
        one_year_ago = historical_data.get(1, {})
        two_years_ago = historical_data.get(2, {})
        three_years_ago = historical_data.get(3, {})
        
        # 1-year growth rates
        if current and one_year_ago:
            metrics.revenue_growth_1y = self._calculate_growth_rate(
                current.get('total_revenue'), one_year_ago.get('total_revenue'))
            metrics.operating_income_growth_1y = self._calculate_growth_rate(
                current.get('operating_income'), one_year_ago.get('operating_income'))
            metrics.net_income_growth_1y = self._calculate_growth_rate(
                current.get('net_income'), one_year_ago.get('net_income'), allow_negative=True)
            metrics.ebitda_growth_1y = self._calculate_growth_rate(
                current.get('ebitda'), one_year_ago.get('ebitda'))
            metrics.operating_cash_flow_growth_1y = self._calculate_growth_rate(
                current.get('operating_cash_flow'), one_year_ago.get('operating_cash_flow'), allow_negative=True)
        
        # 2-year growth rates
        if one_year_ago and two_years_ago:
            metrics.revenue_growth_2y = self._calculate_growth_rate(
                one_year_ago.get('total_revenue'), two_years_ago.get('total_revenue'))
            metrics.operating_income_growth_2y = self._calculate_growth_rate(
                one_year_ago.get('operating_income'), two_years_ago.get('operating_income'))
            metrics.net_income_growth_2y = self._calculate_growth_rate(
                one_year_ago.get('net_income'), two_years_ago.get('net_income'), allow_negative=True)
            metrics.ebitda_growth_2y = self._calculate_growth_rate(
                one_year_ago.get('ebitda'), two_years_ago.get('ebitda'))
            metrics.operating_cash_flow_growth_2y = self._calculate_growth_rate(
                one_year_ago.get('operating_cash_flow'), two_years_ago.get('operating_cash_flow'), allow_negative=True)
        
        # 3-year growth rates
        if two_years_ago and three_years_ago:
            metrics.revenue_growth_3y = self._calculate_growth_rate(
                two_years_ago.get('total_revenue'), three_years_ago.get('total_revenue'))
            metrics.operating_income_growth_3y = self._calculate_growth_rate(
                two_years_ago.get('operating_income'), three_years_ago.get('operating_income'))
            metrics.net_income_growth_3y = self._calculate_growth_rate(
                two_years_ago.get('net_income'), three_years_ago.get('net_income'), allow_negative=True)
            metrics.ebitda_growth_3y = self._calculate_growth_rate(
                two_years_ago.get('ebitda'), three_years_ago.get('ebitda'))
            metrics.operating_cash_flow_growth_3y = self._calculate_growth_rate(
                two_years_ago.get('operating_cash_flow'), three_years_ago.get('operating_cash_flow'), allow_negative=True)
    
    def _calculate_trend_indicators(self, metrics: FinancialMetrics, historical_data: Dict[str, Dict]):
        """Calculate trend indicators and rolling statistics"""
        current = historical_data.get(0, {})
        one_year_ago = historical_data.get(1, {})
        two_years_ago = historical_data.get(2, {})
        three_years_ago = historical_data.get(3, {})
        
        # 3-year averages
        revenues = [self._safe_float(data.get('total_revenue')) for data in [current, one_year_ago, two_years_ago] 
                   if data and self._safe_float(data.get('total_revenue')) is not None]
        if len(revenues) >= 2:
            metrics.revenue_3y_avg = sum(revenues) / len(revenues)
        
        operating_incomes = [self._safe_float(data.get('operating_income')) for data in [current, one_year_ago, two_years_ago] 
                           if data and self._safe_float(data.get('operating_income')) is not None]
        if len(operating_incomes) >= 2:
            metrics.operating_income_3y_avg = sum(operating_incomes) / len(operating_incomes)
        
        net_incomes = [self._safe_float(data.get('net_income')) for data in [current, one_year_ago, two_years_ago] 
                      if data and self._safe_float(data.get('net_income')) is not None]
        if len(net_incomes) >= 2:
            metrics.net_income_3y_avg = sum(net_incomes) / len(net_incomes)
        
        ebitdas = [self._safe_float(data.get('ebitda')) for data in [current, one_year_ago, two_years_ago] 
                  if data and self._safe_float(data.get('ebitda')) is not None]
        if len(ebitdas) >= 2:
            metrics.ebitda_3y_avg = sum(ebitdas) / len(ebitdas)
        
        # Consecutive growth years
        revenue_values = [self._safe_float(data.get('total_revenue')) for data in [current, one_year_ago, two_years_ago, three_years_ago] 
                         if data and self._safe_float(data.get('total_revenue')) is not None]
        
        if len(revenue_values) >= 2:
            consecutive_years = 0
            for i in range(len(revenue_values) - 1):
                if revenue_values[i] > revenue_values[i + 1]:
                    consecutive_years += 1
                else:
                    break
            metrics.revenue_consecutive_growth_years = consecutive_years
        
        # Growth acceleration (simplified)
        growth_rates = [metrics.revenue_growth_1y, metrics.revenue_growth_2y, metrics.revenue_growth_3y]
        valid_growth_rates = [rate for rate in growth_rates if rate is not None]
        
        if len(valid_growth_rates) >= 3:
            if valid_growth_rates[0] > valid_growth_rates[1] > valid_growth_rates[2]:
                metrics.revenue_growth_acceleration = 1  # Accelerating
            elif valid_growth_rates[0] < valid_growth_rates[1] < valid_growth_rates[2]:
                metrics.revenue_growth_acceleration = -1  # Decelerating
            else:
                metrics.revenue_growth_acceleration = 0  # Mixed
        
        # Growth stability (coefficient of variation)
        if len(valid_growth_rates) >= 3:
            mean_growth = sum(valid_growth_rates) / len(valid_growth_rates)
            variance = sum((rate - mean_growth) ** 2 for rate in valid_growth_rates) / len(valid_growth_rates)
            if mean_growth != 0:
                metrics.revenue_growth_stability = (variance ** 0.5) / abs(mean_growth)
    
    def _calculate_data_quality_scores(
        self,
        metrics: FinancialMetrics,
        current_data: Dict,
        ratios_data: Dict,
        spike_timestamp: datetime
    ):
        """Calculate comprehensive data quality scores"""
        # Fundamental data quality
        metrics.fundamental_data_quality = self.validator.validate_financial_data(current_data)
        
        # Ratios data quality
        metrics.ratios_data_quality = self.validator.validate_financial_data(ratios_data)
        
        # Overall completeness score
        all_metrics = metrics.to_dict()
        financial_fields = [k for k, v in all_metrics.items() 
                          if k not in ['setup_id', 'ticker'] and v is not None]
        total_possible_fields = len([k for k in all_metrics.keys() 
                                   if k not in ['setup_id', 'ticker']])
        
        metrics.data_completeness_score = len(financial_fields) / total_possible_fields if total_possible_fields > 0 else 0.0
        
        # Data age
        if current_data.get('date'):
            data_date = current_data.get('date')
            if isinstance(data_date, str):
                data_date = datetime.strptime(data_date, '%Y-%m-%d').date()
            spike_date = spike_timestamp.date() if isinstance(spike_timestamp, datetime) else spike_timestamp
            metrics.financial_data_age_days = (spike_date - data_date).days
    
    def _calculate_growth_rate(
        self,
        current_value: Any,
        previous_value: Any,
        allow_negative: bool = False
    ) -> Optional[float]:
        """Calculate growth rate between two periods"""
        current = self._safe_float(current_value)
        previous = self._safe_float(previous_value)
        
        if current is None or previous is None:
            return None
        
        if previous == 0:
            return None
        
        if not allow_negative and previous < 0:
            return None
        
        # For negative denominators, use absolute value to get meaningful growth rate
        if previous < 0:
            return (current - previous) / abs(previous)
        
        return (current - previous) / previous
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def prepare_ml_features(
        self,
        metrics_list: List[FinancialMetrics],
        mode: str = 'training'
    ) -> pd.DataFrame:
        """
        Prepare ML features DataFrame with consistent preprocessing
        
        Args:
            metrics_list: List of FinancialMetrics objects
            mode: 'training' or 'prediction'
            
        Returns:
            Preprocessed DataFrame ready for ML
        """
        logger.info(f"🔄 Preparing ML features for {len(metrics_list)} setups in {mode} mode...")
        
        # Convert to DataFrame
        df = pd.DataFrame([metrics.to_dict() for metrics in metrics_list])
        
        if df.empty:
            logger.warning("Empty DataFrame created from metrics")
            return df
        
        # Define feature columns (exclude metadata)
        exclude_cols = ['setup_id', 'ticker']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle infinite values
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        
        if mode == 'training':
            # Fit preprocessing on training data
            self._fit_preprocessing(df[feature_cols])
            
            # Save preprocessing parameters
            self._save_preprocessing_params()
            
        else:
            # Load preprocessing parameters for prediction
            if not self.preprocessing_fitted:
                self._load_preprocessing_params()
        
        # Apply preprocessing
        df_processed = self._apply_preprocessing(df, feature_cols)
        
        logger.info(f"✅ ML features prepared: {df_processed.shape[0]} samples, {len(feature_cols)} features")
        return df_processed
    
    def _fit_preprocessing(self, X: pd.DataFrame):
        """Fit imputer and scaler on training data"""
        logger.info("Fitting preprocessing pipeline on training data...")
        
        # Filter out features with no valid values
        valid_features = []
        for col in X.columns:
            if X[col].notna().any():
                valid_features.append(col)
            else:
                logger.warning(f"Dropping feature '{col}' - no valid values found")
        
        if not valid_features:
            raise ValueError("No features with valid values found")
        
        # Work only with valid features
        X_valid = X[valid_features]
        
        # Initialize preprocessing components
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        
        # Store feature column order (only valid features)
        self.feature_columns = valid_features
        
        # Fit imputer
        X_imputed = self.imputer.fit_transform(X_valid)
        
        # Fit scaler
        self.scaler.fit(X_imputed)
        
        self.preprocessing_fitted = True
        logger.info(f"✅ Preprocessing pipeline fitted with {len(valid_features)} valid features")
    
    def _apply_preprocessing(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Apply preprocessing to features"""
        if not self.preprocessing_fitted:
            raise ValueError("Preprocessing not fitted. Call _fit_preprocessing or _load_preprocessing_params first.")
        
        # Create copy
        df_processed = df.copy()
        
        # Only use features that were included in training
        available_features = [col for col in self.feature_columns if col in df_processed.columns]
        missing_features = [col for col in self.feature_columns if col not in df_processed.columns]
        
        if missing_features:
            logger.warning(f"Missing features in data: {missing_features[:5]}...")
            # Add missing features with NaN
            for col in missing_features:
                df_processed[col] = np.nan
        
        # Ensure feature columns are in the same order as training
        X = df_processed[self.feature_columns]
        
        # Apply imputation
        X_imputed = self.imputer.transform(X)
        
        # Apply scaling
        X_scaled = self.scaler.transform(X_imputed)
        
        # Update DataFrame with processed features
        df_processed[self.feature_columns] = X_scaled
        
        return df_processed
    
    def _save_preprocessing_params(self):
        """Save preprocessing parameters to disk"""
        params = {
            'imputer': self.imputer,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'preprocessing_fitted': self.preprocessing_fitted
        }
        
        with open(self.preprocessing_params_path, 'wb') as f:
            pickle.dump(params, f)
        
        logger.info(f"✅ Preprocessing parameters saved to {self.preprocessing_params_path}")
    
    def _load_preprocessing_params(self):
        """Load preprocessing parameters from disk"""
        if not self.preprocessing_params_path.exists():
            raise FileNotFoundError(f"Preprocessing parameters not found at {self.preprocessing_params_path}")
        
        with open(self.preprocessing_params_path, 'rb') as f:
            params = pickle.load(f)
        
        self.imputer = params['imputer']
        self.scaler = params['scaler']
        self.feature_columns = params['feature_columns']
        self.preprocessing_fitted = params['preprocessing_fitted']
        
        logger.info(f"✅ Preprocessing parameters loaded from {self.preprocessing_params_path}")

# Convenience functions
def extract_and_preprocess_financial_features(
    setup_ids: List[str],
    mode: str = 'training',
    db_path: str = "data/sentiment_system.duckdb",
    preprocessing_params_path: str = "models/financial/preprocessing_params.pkl"
) -> pd.DataFrame:
    """
    Convenience function to extract and preprocess financial features
    
    Args:
        setup_ids: List of setup IDs to process
        mode: 'training' or 'prediction'
        db_path: Path to DuckDB database
        preprocessing_params_path: Path to preprocessing parameters
        
    Returns:
        Preprocessed DataFrame ready for ML
    """
    # Initialize preprocessor
    preprocessor = FinancialPreprocessor(
        db_path=db_path,
        preprocessing_params_path=preprocessing_params_path
    )
    
    # Extract comprehensive financial data
    metrics_list = preprocessor.extract_comprehensive_financial_data(setup_ids)
    
    # Prepare ML features
    df = preprocessor.prepare_ml_features(metrics_list, mode=mode)
    
    return df

if __name__ == "__main__":
    # Test the preprocessor
    import duckdb
    
    # Get sample setup IDs
    conn = duckdb.connect("data/sentiment_system.duckdb")
    sample_setup_ids = conn.execute("SELECT setup_id FROM setups LIMIT 10").fetchall()
    setup_ids = [row[0] for row in sample_setup_ids]
    conn.close()
    
    if setup_ids:
        logger.info(f"Testing financial preprocessor with {len(setup_ids)} sample setups...")
        
        # Test preprocessing
        df = extract_and_preprocess_financial_features(
            setup_ids=setup_ids,
            mode='training'
        )
        
        logger.info(f"✅ Test successful!")
        logger.info(f"  - Shape: {df.shape}")
        logger.info(f"  - Features: {df.shape[1] - 2}")  # -2 for setup_id and ticker
        logger.info(f"  - Data completeness: {df.notna().sum().sum() / (df.shape[0] * df.shape[1]):.2%}")
    else:
        logger.error("No setup IDs found for testing") 