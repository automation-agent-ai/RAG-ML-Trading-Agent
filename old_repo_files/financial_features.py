#!/usr/bin/env python3
"""
Financial Features Extractor
============================

Extracts financial features directly from DuckDB fundamentals table.
Computes financial ratios, growth metrics, and financial health indicators.

Features extracted:
- Profitability ratios (ROE, ROA, profit margins)
- Liquidity ratios (current ratio, quick ratio) 
- Leverage ratios (debt-to-equity, debt-to-assets)
- Efficiency ratios (asset turnover, working capital ratios)
- Growth metrics (revenue growth, earnings growth)
- Cash flow metrics (FCF yield, OCF ratios)
- Per-share metrics (EPS, cash flow per share, book value per share)
- Valuation ratios (PE ratio, PB ratio, EV/EBITDA)

This module works for both training and prediction pipelines.
"""

import logging
import duckdb
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FinancialFeatures:
    """Financial features extracted from fundamentals data"""
    setup_id: str
    ticker: str
    
    # Basic financial data
    total_revenue: Optional[float] = None
    net_income: Optional[float] = None
    total_assets: Optional[float] = None
    total_equity: Optional[float] = None
    total_debt: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    free_cash_flow: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    ebitda: Optional[float] = None
    
    # Per-share metrics
    basic_eps: Optional[float] = None
    diluted_eps: Optional[float] = None
    book_value_per_share: Optional[float] = None
    cash_flow_per_share: Optional[float] = None
    revenue_per_share: Optional[float] = None
    operating_cash_flow_per_share: Optional[float] = None
    
    # Profitability ratios
    net_profit_margin: Optional[float] = None
    gross_profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    return_on_invested_capital: Optional[float] = None
    
    # Liquidity ratios
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    cash_ratio: Optional[float] = None
    working_capital_ratio: Optional[float] = None
    
    # Leverage ratios  
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    equity_ratio: Optional[float] = None
    debt_to_ebitda: Optional[float] = None
    
    # Efficiency ratios
    asset_turnover: Optional[float] = None
    inventory_turnover: Optional[float] = None
    receivables_turnover: Optional[float] = None
    
    # Valuation ratios
    price_earnings_ratio: Optional[float] = None
    price_to_book_ratio: Optional[float] = None
    price_to_sales_ratio: Optional[float] = None
    enterprise_value_to_ebitda: Optional[float] = None
    
    # Cash flow ratios
    operating_cash_flow_ratio: Optional[float] = None
    free_cash_flow_yield: Optional[float] = None
    cash_flow_to_debt: Optional[float] = None
    
    # Growth metrics (YoY if available)
    revenue_growth_yoy: Optional[float] = None
    net_income_growth_yoy: Optional[float] = None
    earnings_growth_current: Optional[float] = None
    earnings_growth_previous: Optional[float] = None
    operating_cash_flow_growth: Optional[float] = None
    
    # Data quality indicators
    data_completeness_score: Optional[float] = None
    financial_data_age_days: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'setup_id': self.setup_id,
            'ticker': self.ticker,
            
            # Basic data
            'total_revenue': self.total_revenue,
            'net_income': self.net_income,
            'total_assets': self.total_assets,
            'total_equity': self.total_equity,
            'total_debt': self.total_debt,
            'cash_and_equivalents': self.cash_and_equivalents,
            'operating_cash_flow': self.operating_cash_flow,
            'free_cash_flow': self.free_cash_flow,
            'gross_profit': self.gross_profit,
            'operating_income': self.operating_income,
            'ebitda': self.ebitda,
            
            # Per-share metrics
            'basic_eps': self.basic_eps,
            'diluted_eps': self.diluted_eps,
            'book_value_per_share': self.book_value_per_share,
            'cash_flow_per_share': self.cash_flow_per_share,
            'revenue_per_share': self.revenue_per_share,
            'operating_cash_flow_per_share': self.operating_cash_flow_per_share,
            
            # Ratios
            'net_profit_margin': self.net_profit_margin,
            'gross_profit_margin': self.gross_profit_margin,
            'operating_margin': self.operating_margin,
            'return_on_equity': self.return_on_equity,
            'return_on_assets': self.return_on_assets,
            'return_on_invested_capital': self.return_on_invested_capital,
            'current_ratio': self.current_ratio,
            'quick_ratio': self.quick_ratio,
            'cash_ratio': self.cash_ratio,
            'working_capital_ratio': self.working_capital_ratio,
            'debt_to_equity': self.debt_to_equity,
            'debt_to_assets': self.debt_to_assets,
            'equity_ratio': self.equity_ratio,
            'debt_to_ebitda': self.debt_to_ebitda,
            'asset_turnover': self.asset_turnover,
            'inventory_turnover': self.inventory_turnover,
            'receivables_turnover': self.receivables_turnover,
            'price_earnings_ratio': self.price_earnings_ratio,
            'price_to_book_ratio': self.price_to_book_ratio,
            'price_to_sales_ratio': self.price_to_sales_ratio,
            'enterprise_value_to_ebitda': self.enterprise_value_to_ebitda,
            'operating_cash_flow_ratio': self.operating_cash_flow_ratio,
            'free_cash_flow_yield': self.free_cash_flow_yield,
            'cash_flow_to_debt': self.cash_flow_to_debt,
            
            # Growth
            'revenue_growth_yoy': self.revenue_growth_yoy,
            'net_income_growth_yoy': self.net_income_growth_yoy,
            'earnings_growth_current': self.earnings_growth_current,
            'earnings_growth_previous': self.earnings_growth_previous,
            'operating_cash_flow_growth': self.operating_cash_flow_growth,
            
            # Quality
            'data_completeness_score': self.data_completeness_score,
            'financial_data_age_days': self.financial_data_age_days
        }

class FinancialFeaturesExtractor:
    """Extracts financial features from DuckDB fundamentals table"""
    
    def __init__(self, db_path: str = "data/sentiment_system.duckdb"):
        self.db_path = db_path
        
    def extract_features(self, setup_id: str) -> Optional[FinancialFeatures]:
        """
        Extract financial features for a setup_id
        
        Args:
            setup_id: The setup identifier
            
        Returns:
            FinancialFeatures object or None if no data found
        """
        try:
            with duckdb.connect(self.db_path) as conn:
                # Get setup info (ticker and spike date)
                setup_query = """
                SELECT lse_ticker, yahoo_ticker, spike_timestamp 
                FROM setups 
                WHERE setup_id = ?
                """
                setup_result = conn.execute(setup_query, [setup_id]).fetchone()
                
                if not setup_result:
                    logger.warning(f"Setup {setup_id} not found")
                    return None
                
                lse_ticker, yahoo_ticker, spike_timestamp = setup_result
                ticker = lse_ticker or yahoo_ticker
                
                logger.info(f"💰 Extracting financial features for {setup_id} ({ticker})")
                
                # Get the most recent financial data before spike date
                financial_data = self._get_comprehensive_financial_data(conn, ticker, spike_timestamp)
                
                if not financial_data:
                    logger.warning(f"No financial data found for {ticker}")
                    return self._get_default_features(setup_id, ticker)
                
                # Calculate all financial features
                features = self._calculate_comprehensive_features(setup_id, ticker, financial_data, spike_timestamp)
                
                # Store features in database
                self._store_features(conn, features)
                
                logger.info(f"✓ Financial features extracted for {setup_id}")
                return features
                
        except Exception as e:
            logger.error(f"Error extracting financial features for {setup_id}: {e}")
            return self._get_default_features(setup_id, "UNKNOWN")
    
    def _get_comprehensive_financial_data(self, conn, ticker: str, spike_timestamp: datetime) -> Optional[Dict]:
        """Get comprehensive financial data combining fundamentals and ratios tables"""
        
        spike_date = spike_timestamp.date() if isinstance(spike_timestamp, datetime) else spike_timestamp
        
        # Get recent fundamentals data
        fundamentals_query = """
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY statement_type, period_type 
                   ORDER BY 
                       CASE WHEN date IS NOT NULL THEN date 
                            ELSE DATE(fiscal_year || '-12-31') 
                       END DESC
               ) as rn
        FROM fundamentals 
        WHERE ticker = ?
        AND (
            date <= ? OR 
            (date IS NULL AND DATE(fiscal_year || '-12-31') <= ?)
        )
        AND fiscal_year >= EXTRACT(year FROM ?) - 3  -- Last 3 years for growth calculations
        """
        
        fundamentals_data = conn.execute(fundamentals_query, [ticker, spike_date, spike_date, spike_date]).fetchall()
        
        # Get recent financial ratios
        ratios_query = """
        SELECT *,
               ROW_NUMBER() OVER (ORDER BY period_end DESC) as rn
        FROM financial_ratios 
        WHERE ticker = ?
        AND period_end <= ?
        """
        
        ratios_data = conn.execute(ratios_query, [ticker, spike_date]).fetchall()
        
        if not fundamentals_data and not ratios_data:
            return None
        
        # Organize data
        result = {
            'fundamentals': {},
            'ratios': {},
            'fundamentals_raw': fundamentals_data,
            'ratios_raw': ratios_data
        }
        
        # Process fundamentals data
        if fundamentals_data:
            fundamentals_cols = [desc[0] for desc in conn.description]
            for row in fundamentals_data:
                row_dict = dict(zip(fundamentals_cols, row))
                statement_type = row_dict['statement_type']
                period_type = row_dict['period_type']
                key = f"{statement_type}_{period_type}"
                
                if row_dict['rn'] == 1:  # Most recent for each type
                    result['fundamentals'][key] = row_dict
        
        # Process ratios data  
        if ratios_data:
            ratios_cols = [desc[0] for desc in conn.description] 
            most_recent_ratios = ratios_data[0]  # First row is most recent
            result['ratios']['current'] = dict(zip(ratios_cols, most_recent_ratios))
        
        return result
    
    def _calculate_comprehensive_features(self, setup_id: str, ticker: str, financial_data: Dict, spike_timestamp: datetime) -> FinancialFeatures:
        """Calculate comprehensive financial ratios and features"""
        
        features = FinancialFeatures(setup_id=setup_id, ticker=ticker)
        
        # Get the most comprehensive dataset (prefer annual income statement)
        primary_data = None
        for key in ['income_annual', 'income_quarterly', 'balance_annual', 'balance_quarterly', 'cash_annual', 'cash_quarterly']:
            if key in financial_data['fundamentals']:
                primary_data = financial_data['fundamentals'][key]
                break
        
        # Get ratios data
        ratios_data = financial_data['ratios'].get('current', {})
        
        if not primary_data and not ratios_data:
            logger.warning(f"No primary financial data found for {ticker}")
            return features
        
        # Extract basic financial data from fundamentals
        if primary_data:
            features.total_revenue = self._safe_float(primary_data.get('total_revenue'))
            features.net_income = self._safe_float(primary_data.get('net_income'))
            features.total_assets = self._safe_float(primary_data.get('total_assets'))
            features.total_equity = self._safe_float(primary_data.get('total_equity'))
            features.total_debt = self._safe_float(primary_data.get('total_debt'))
            features.cash_and_equivalents = self._safe_float(primary_data.get('cash_and_equivalents'))
            features.operating_cash_flow = self._safe_float(primary_data.get('operating_cash_flow'))
            features.free_cash_flow = self._safe_float(primary_data.get('free_cash_flow'))
            features.gross_profit = self._safe_float(primary_data.get('gross_profit'))
            features.operating_income = self._safe_float(primary_data.get('operating_income'))
            features.ebitda = self._safe_float(primary_data.get('ebitda'))
            features.basic_eps = self._safe_float(primary_data.get('basic_eps'))
            features.diluted_eps = self._safe_float(primary_data.get('diluted_eps'))
        
        # Extract data from ratios table (pre-calculated ratios)
        if ratios_data:
            features.current_ratio = self._safe_float(ratios_data.get('current_ratio'))
            features.quick_ratio = self._safe_float(ratios_data.get('quick_ratio'))
            features.cash_ratio = self._safe_float(ratios_data.get('cash_ratio'))
            features.debt_to_equity = self._safe_float(ratios_data.get('debt_to_equity'))
            features.debt_to_assets = self._safe_float(ratios_data.get('debt_to_assets'))
            features.equity_ratio = self._safe_float(ratios_data.get('equity_ratio'))
            features.gross_profit_margin = self._safe_float(ratios_data.get('gross_margin'))
            features.operating_margin = self._safe_float(ratios_data.get('operating_margin'))
            features.net_profit_margin = self._safe_float(ratios_data.get('net_margin'))
            features.return_on_equity = self._safe_float(ratios_data.get('roe'))
            features.return_on_assets = self._safe_float(ratios_data.get('roa'))
            features.return_on_invested_capital = self._safe_float(ratios_data.get('roic'))
            features.asset_turnover = self._safe_float(ratios_data.get('asset_turnover'))
            features.inventory_turnover = self._safe_float(ratios_data.get('inventory_turnover'))
            features.receivables_turnover = self._safe_float(ratios_data.get('receivables_turnover'))
            features.price_earnings_ratio = self._safe_float(ratios_data.get('pe_ratio'))
            features.price_to_book_ratio = self._safe_float(ratios_data.get('pb_ratio'))
            features.price_to_sales_ratio = self._safe_float(ratios_data.get('ps_ratio'))
            features.enterprise_value_to_ebitda = self._safe_float(ratios_data.get('ev_ebitda'))
            features.book_value_per_share = self._safe_float(ratios_data.get('book_value_per_share'))
            features.revenue_per_share = self._safe_float(ratios_data.get('revenue_per_share'))
            features.cash_flow_per_share = self._safe_float(ratios_data.get('cash_per_share'))
        
        # Calculate derived metrics not in ratios table
        self._calculate_derived_metrics(features, primary_data)
        
        # Calculate growth metrics
        self._calculate_growth_metrics(features, financial_data, ticker)
        
        # Calculate per-share metrics if missing
        self._calculate_per_share_metrics(features, primary_data)
        
        # Calculate cash flow ratios
        self._calculate_cash_flow_ratios(features)
        
        # Calculate data quality score
        features.data_completeness_score = self._calculate_completeness_score(features)
        
        # Calculate data age
        if primary_data and primary_data.get('date'):
            data_date = primary_data.get('date')
            spike_date = spike_timestamp.date() if isinstance(spike_timestamp, datetime) else spike_timestamp
            features.financial_data_age_days = (spike_date - data_date).days
        
        return features
    
    def _calculate_derived_metrics(self, features: FinancialFeatures, primary_data: Optional[Dict]):
        """Calculate metrics not directly available in ratios table"""
        
        if not primary_data:
            return
        
        # Calculate margins if not available from ratios
        if features.total_revenue and features.total_revenue > 0:
            if features.net_profit_margin is None and features.net_income is not None:
                features.net_profit_margin = features.net_income / features.total_revenue
            
            if features.gross_profit_margin is None and features.gross_profit is not None:
                features.gross_profit_margin = features.gross_profit / features.total_revenue
            
            if features.operating_margin is None and features.operating_income is not None:
                features.operating_margin = features.operating_income / features.total_revenue
        
        # Calculate return ratios if not available
        if features.return_on_equity is None and features.total_equity and features.total_equity > 0 and features.net_income is not None:
            features.return_on_equity = features.net_income / features.total_equity
        
        if features.return_on_assets is None and features.total_assets and features.total_assets > 0 and features.net_income is not None:
            features.return_on_assets = features.net_income / features.total_assets
        
        # Calculate liquidity ratios if not available
        current_assets = self._safe_float(primary_data.get('current_assets'))
        current_liabilities = self._safe_float(primary_data.get('current_liabilities'))
        
        if features.current_ratio is None and current_assets and current_liabilities and current_liabilities > 0:
            features.current_ratio = current_assets / current_liabilities
        
        if features.cash_ratio is None and features.cash_and_equivalents and current_liabilities and current_liabilities > 0:
            features.cash_ratio = features.cash_and_equivalents / current_liabilities
        
        # Calculate working capital ratio
        working_capital = self._safe_float(primary_data.get('working_capital'))
        if working_capital and features.total_revenue and features.total_revenue > 0:
            features.working_capital_ratio = working_capital / features.total_revenue
        
        # Calculate debt ratios if not available
        if features.debt_to_equity is None and features.total_debt and features.total_equity and features.total_equity > 0:
            features.debt_to_equity = features.total_debt / features.total_equity
        
        if features.debt_to_assets is None and features.total_debt and features.total_assets and features.total_assets > 0:
            features.debt_to_assets = features.total_debt / features.total_assets
        
        if features.equity_ratio is None and features.total_equity and features.total_assets and features.total_assets > 0:
            features.equity_ratio = features.total_equity / features.total_assets
        
        # Calculate debt to EBITDA
        if features.ebitda and features.ebitda > 0 and features.total_debt:
            features.debt_to_ebitda = features.total_debt / features.ebitda
        
        # Calculate efficiency ratios if not available
        if features.asset_turnover is None and features.total_revenue and features.total_assets and features.total_assets > 0:
            features.asset_turnover = features.total_revenue / features.total_assets
    
    def _calculate_per_share_metrics(self, features: FinancialFeatures, primary_data: Optional[Dict]):
        """Calculate per-share metrics if not available from ratios table"""
        
        if not primary_data:
            return
        
        shares_outstanding = self._safe_float(primary_data.get('shares_outstanding'))
        
        if shares_outstanding and shares_outstanding > 0:
            # Operating cash flow per share
            if features.operating_cash_flow_per_share is None and features.operating_cash_flow:
                features.operating_cash_flow_per_share = features.operating_cash_flow / shares_outstanding
            
            # Revenue per share if not available
            if features.revenue_per_share is None and features.total_revenue:
                features.revenue_per_share = features.total_revenue / shares_outstanding
            
            # Cash flow per share (using free cash flow)
            if features.cash_flow_per_share is None and features.free_cash_flow:
                features.cash_flow_per_share = features.free_cash_flow / shares_outstanding
    
    def _calculate_cash_flow_ratios(self, features: FinancialFeatures):
        """Calculate cash flow related ratios"""
        
        # Operating cash flow ratio
        if features.operating_cash_flow and features.total_revenue and features.total_revenue > 0:
            features.operating_cash_flow_ratio = features.operating_cash_flow / features.total_revenue
        
        # Free cash flow yield
        if features.free_cash_flow and features.total_revenue and features.total_revenue > 0:
            features.free_cash_flow_yield = features.free_cash_flow / features.total_revenue
        
        # Cash flow to debt
        if features.operating_cash_flow and features.total_debt and features.total_debt > 0:
            features.cash_flow_to_debt = features.operating_cash_flow / features.total_debt
    
    def _calculate_growth_metrics(self, features: FinancialFeatures, financial_data: Dict, ticker: str):
        """Calculate growth metrics using historical data"""
        
        fundamentals_raw = financial_data.get('fundamentals_raw', [])
        
        if len(fundamentals_raw) < 2:
            return  # Need at least 2 periods for growth calculation
        
        # Organize data by year for growth calculations
        annual_data = {}
        for row_data in fundamentals_raw:
            if isinstance(row_data, (list, tuple)):
                # Convert to dict using column names
                continue  # Skip if we can't process the format
            
            fiscal_year = row_data.get('fiscal_year')
            statement_type = row_data.get('statement_type', 'income')
            
            if fiscal_year and statement_type == 'income':
                annual_data[fiscal_year] = row_data
        
        years = sorted(annual_data.keys(), reverse=True)  # Most recent first
        
        if len(years) >= 2:
            current_year = years[0]
            previous_year = years[1]
            
            current_data = annual_data[current_year]
            previous_data = annual_data[previous_year]
            
            # Revenue growth
            current_revenue = self._safe_float(current_data.get('total_revenue'))
            previous_revenue = self._safe_float(previous_data.get('total_revenue'))
            
            if current_revenue and previous_revenue and previous_revenue > 0:
                features.revenue_growth_yoy = (current_revenue - previous_revenue) / previous_revenue
            
            # Net income growth (current period)
            current_ni = self._safe_float(current_data.get('net_income'))
            previous_ni = self._safe_float(previous_data.get('net_income'))
            
            if current_ni and previous_ni and previous_ni != 0:
                features.net_income_growth_yoy = (current_ni - previous_ni) / abs(previous_ni)
                features.earnings_growth_current = features.net_income_growth_yoy
            
            # Operating cash flow growth
            current_ocf = self._safe_float(current_data.get('operating_cash_flow'))
            previous_ocf = self._safe_float(previous_data.get('operating_cash_flow'))
            
            if current_ocf and previous_ocf and previous_ocf != 0:
                features.operating_cash_flow_growth = (current_ocf - previous_ocf) / abs(previous_ocf)
        
        # Previous period growth (if we have 3+ years)
        if len(years) >= 3:
            previous_year = years[1]
            two_years_ago = years[2]
            
            previous_data = annual_data[previous_year]
            older_data = annual_data[two_years_ago]
            
            previous_ni = self._safe_float(previous_data.get('net_income'))
            older_ni = self._safe_float(older_data.get('net_income'))
            
            if previous_ni and older_ni and older_ni != 0:
                features.earnings_growth_previous = (previous_ni - older_ni) / abs(older_ni)
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _calculate_completeness_score(self, features: FinancialFeatures) -> float:
        """Calculate data completeness score (0-1)"""
        key_fields = [
            features.total_revenue, features.net_income, features.total_assets,
            features.total_equity, features.basic_eps, features.net_profit_margin, 
            features.return_on_equity, features.current_ratio, features.debt_to_equity,
            features.price_earnings_ratio, features.operating_cash_flow_per_share
        ]
        
        non_null_count = sum(1 for field in key_fields if field is not None)
        return non_null_count / len(key_fields)
    
    def _get_default_features(self, setup_id: str, ticker: str) -> FinancialFeatures:
        """Return default features when no financial data available"""
        return FinancialFeatures(
            setup_id=setup_id,
            ticker=ticker,
            data_completeness_score=0.0
        )
    
    def _store_features(self, conn, features: FinancialFeatures):
        """Store financial features in database"""
        try:
            # Create enhanced table with all new fields
            conn.execute("""
                CREATE TABLE IF NOT EXISTS financial_features (
                    setup_id VARCHAR PRIMARY KEY,
                    ticker VARCHAR,
                    
                    -- Basic financial data
                    total_revenue DOUBLE,
                    net_income DOUBLE,
                    total_assets DOUBLE,
                    total_equity DOUBLE,
                    total_debt DOUBLE,
                    cash_and_equivalents DOUBLE,
                    operating_cash_flow DOUBLE,
                    free_cash_flow DOUBLE,
                    gross_profit DOUBLE,
                    operating_income DOUBLE,
                    ebitda DOUBLE,
                    
                    -- Per-share metrics
                    basic_eps DOUBLE,
                    diluted_eps DOUBLE,
                    book_value_per_share DOUBLE,
                    cash_flow_per_share DOUBLE,
                    revenue_per_share DOUBLE,
                    operating_cash_flow_per_share DOUBLE,
                    
                    -- Profitability ratios
                    net_profit_margin DOUBLE,
                    gross_profit_margin DOUBLE,
                    operating_margin DOUBLE,
                    return_on_equity DOUBLE,
                    return_on_assets DOUBLE,
                    return_on_invested_capital DOUBLE,
                    
                    -- Liquidity ratios
                    current_ratio DOUBLE,
                    quick_ratio DOUBLE,
                    cash_ratio DOUBLE,
                    working_capital_ratio DOUBLE,
                    
                    -- Leverage ratios
                    debt_to_equity DOUBLE,
                    debt_to_assets DOUBLE,
                    equity_ratio DOUBLE,
                    debt_to_ebitda DOUBLE,
                    
                    -- Efficiency ratios
                    asset_turnover DOUBLE,
                    inventory_turnover DOUBLE,
                    receivables_turnover DOUBLE,
                    
                    -- Valuation ratios
                    price_earnings_ratio DOUBLE,
                    price_to_book_ratio DOUBLE,
                    price_to_sales_ratio DOUBLE,
                    enterprise_value_to_ebitda DOUBLE,
                    
                    -- Cash flow ratios
                    operating_cash_flow_ratio DOUBLE,
                    free_cash_flow_yield DOUBLE,
                    cash_flow_to_debt DOUBLE,
                    
                    -- Growth metrics
                    revenue_growth_yoy DOUBLE,
                    net_income_growth_yoy DOUBLE,
                    earnings_growth_current DOUBLE,
                    earnings_growth_previous DOUBLE,
                    operating_cash_flow_growth DOUBLE,
                    
                    -- Quality indicators
                    data_completeness_score DOUBLE,
                    financial_data_age_days INTEGER,
                    
                    extraction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert or replace features
            feature_dict = features.to_dict()
            columns = list(feature_dict.keys())
            placeholders = ['?' for _ in columns]
            values = list(feature_dict.values())
            
            query = f"""
                INSERT OR REPLACE INTO financial_features ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
            """
            
            conn.execute(query, values)
            logger.debug(f"Stored comprehensive financial features for {features.setup_id}")
            
        except Exception as e:
            logger.error(f"Failed to store financial features for {features.setup_id}: {e}")
    
    def extract_batch(self, setup_ids: List[str]) -> List[FinancialFeatures]:
        """Extract financial features for multiple setup_ids"""
        results = []
        
        logger.info(f"💰 Extracting comprehensive financial features for {len(setup_ids)} setups")
        
        for i, setup_id in enumerate(setup_ids):
            logger.info(f"Processing {i+1}/{len(setup_ids)}: {setup_id}")
            
            features = self.extract_features(setup_id)
            if features:
                results.append(features)
        
        logger.info(f"✅ Financial feature extraction complete: {len(results)}/{len(setup_ids)} successful")
        return results

def extract_financial_features(setup_ids: List[str], db_path: str = "data/sentiment_system.duckdb") -> List[FinancialFeatures]:
    """Convenience function to extract financial features"""
    extractor = FinancialFeaturesExtractor(db_path)
    return extractor.extract_batch(setup_ids)

def integrate_with_fundamentals_features(setup_ids: List[str], db_path: str = "data/sentiment_system.duckdb"):
    """
    Integrate structured financial features with the fundamentals_features table
    This addresses the issue of fundamentals model not accessing all financial data
    """
    logger.info("🔗 Integrating structured financial data with fundamentals_features table")
    
    extractor = FinancialFeaturesExtractor(db_path)
    
    with duckdb.connect(db_path) as conn:
        # First, add new columns to fundamentals_features table to accommodate structured data
        logger.info("📊 Enhancing fundamentals_features table schema...")
        
        try:
            # Add structured financial columns to fundamentals_features
            alter_queries = [
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS basic_eps REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS diluted_eps REAL DEFAULT NULL", 
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS price_earnings_ratio REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS operating_cash_flow_per_share REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS cash_flow_per_share REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS book_value_per_share REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS revenue_per_share REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS total_revenue REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS quick_ratio REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS cash_ratio REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS working_capital_ratio REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS asset_turnover REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS inventory_turnover REAL DEFAULT NULL", 
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS receivables_turnover REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS price_to_book_ratio REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS price_to_sales_ratio REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS enterprise_value_to_ebitda REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS operating_cash_flow_ratio REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS free_cash_flow_yield REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS cash_flow_to_debt REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS earnings_growth_current REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS earnings_growth_previous REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS operating_cash_flow_growth REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS return_on_invested_capital REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS debt_to_ebitda REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS equity_ratio REAL DEFAULT NULL",
                "ALTER TABLE fundamentals_features ADD COLUMN IF NOT EXISTS debt_to_assets REAL DEFAULT NULL"
            ]
            
            for query in alter_queries:
                try:
                    conn.execute(query)
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Failed to add column: {e}")
            
            logger.info("✅ Enhanced fundamentals_features table schema")
            
        except Exception as e:
            logger.error(f"Error enhancing table schema: {e}")
        
        # Now extract and update features for each setup
        successful_updates = 0
        for i, setup_id in enumerate(setup_ids):
            logger.info(f"🔄 Processing {i+1}/{len(setup_ids)}: {setup_id}")
            
            try:
                # Extract comprehensive financial features  
                financial_features = extractor.extract_features(setup_id)
                
                if financial_features:
                    # Update fundamentals_features table with the extracted data
                    update_query = """
                    UPDATE fundamentals_features SET
                        basic_eps = ?,
                        diluted_eps = ?,
                        price_earnings_ratio = ?,
                        operating_cash_flow_per_share = ?,
                        cash_flow_per_share = ?,
                        book_value_per_share = ?,
                        revenue_per_share = ?,
                        total_revenue = ?,
                        quick_ratio = ?,
                        cash_ratio = ?,
                        working_capital_ratio = ?,
                        asset_turnover = ?,
                        inventory_turnover = ?,
                        receivables_turnover = ?,
                        price_to_book_ratio = ?,
                        price_to_sales_ratio = ?,
                        enterprise_value_to_ebitda = ?,
                        operating_cash_flow_ratio = ?,
                        free_cash_flow_yield = ?,
                        cash_flow_to_debt = ?,
                        earnings_growth_current = ?,
                        earnings_growth_previous = ?,
                        operating_cash_flow_growth = ?,
                        return_on_invested_capital = ?,
                        debt_to_ebitda = ?,
                        equity_ratio = ?,
                        debt_to_assets = ?
                    WHERE setup_id = ?
                    """
                    
                    values = [
                        financial_features.basic_eps,
                        financial_features.diluted_eps,
                        financial_features.price_earnings_ratio,
                        financial_features.operating_cash_flow_per_share,
                        financial_features.cash_flow_per_share,
                        financial_features.book_value_per_share,
                        financial_features.revenue_per_share,
                        financial_features.total_revenue,
                        financial_features.quick_ratio,
                        financial_features.cash_ratio,
                        financial_features.working_capital_ratio,
                        financial_features.asset_turnover,
                        financial_features.inventory_turnover,
                        financial_features.receivables_turnover,
                        financial_features.price_to_book_ratio,
                        financial_features.price_to_sales_ratio,
                        financial_features.enterprise_value_to_ebitda,
                        financial_features.operating_cash_flow_ratio,
                        financial_features.free_cash_flow_yield,
                        financial_features.cash_flow_to_debt,
                        financial_features.earnings_growth_current,
                        financial_features.earnings_growth_previous,
                        financial_features.operating_cash_flow_growth,
                        financial_features.return_on_invested_capital,
                        financial_features.debt_to_ebitda,
                        financial_features.equity_ratio,
                        financial_features.debt_to_assets,
                        setup_id
                    ]
                    
                    result = conn.execute(update_query, values)
                    if result.rowcount > 0:
                        successful_updates += 1
                        logger.debug(f"✅ Updated fundamentals_features for {setup_id}")
                    else:
                        logger.warning(f"⚠️ No existing fundamentals_features row found for {setup_id}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {setup_id}: {e}")
        
        logger.info(f"🎉 Integration complete: {successful_updates}/{len(setup_ids)} setups updated with structured financial data")
        
        # Show summary of enhanced features
        feature_count_query = """
        SELECT COUNT(*) as total_setups,
               COUNT(basic_eps) as has_eps,
               COUNT(price_earnings_ratio) as has_pe_ratio,
               COUNT(operating_cash_flow_per_share) as has_ocf_per_share,
               COUNT(earnings_growth_current) as has_earnings_growth
        FROM fundamentals_features 
        WHERE setup_id IN ({})
        """.format(','.join(['?' for _ in setup_ids]))
        
        summary = conn.execute(feature_count_query, setup_ids).fetchone()
        if summary:
            logger.info(f"📈 Feature coverage summary:")
            logger.info(f"   Total setups: {summary[0]}")
            logger.info(f"   Have EPS: {summary[1]} ({100*summary[1]/summary[0]:.1f}%)")
            logger.info(f"   Have PE ratio: {summary[2]} ({100*summary[2]/summary[0]:.1f}%)")
            logger.info(f"   Have OCF per share: {summary[3]} ({100*summary[3]/summary[0]:.1f}%)")
            logger.info(f"   Have earnings growth: {summary[4]} ({100*summary[4]/summary[0]:.1f}%)")

if __name__ == "__main__":
    # Test with a single setup
    extractor = FinancialFeaturesExtractor()
    
    # Get a test setup_id
    with duckdb.connect("data/sentiment_system.duckdb") as conn:
        test_setup = conn.execute("SELECT setup_id FROM setups LIMIT 1").fetchone()
        if test_setup:
            features = extractor.extract_features(test_setup[0])
            if features:
                print(f"✅ Test successful for {features.setup_id}")
                print(f"   Revenue: {features.total_revenue}")
                print(f"   EPS: {features.basic_eps}")
                print(f"   PE Ratio: {features.price_earnings_ratio}")
                print(f"   OCF per Share: {features.operating_cash_flow_per_share}")
                print(f"   Net Profit Margin: {features.net_profit_margin}")
                print(f"   ROE: {features.return_on_equity}")
                print(f"   Current Ratio: {features.current_ratio}")
                print(f"   Data Completeness: {features.data_completeness_score}")
            else:
                print("❌ No features extracted")
        else:
            print("❌ No setups found for testing") 