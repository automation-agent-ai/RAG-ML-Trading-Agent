#!/usr/bin/env python3
"""
Enhance Financial Features

This script enhances financial features by:
1. Calculating additional growth ratios from fundamentals data
2. Scaling P&L positions by revenue
3. Scaling balance sheet positions by total assets
4. Scaling cash flow positions by operating cash flow
5. Imputing missing values

Usage:
    python enhance_financial_features.py --input data/ml_features/financial_ml_features_training_*.csv --output data/ml_features/enhanced_financial_ml_features_training.csv
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinancialFeatureEnhancer:
    """Class for enhancing financial features"""
    
    def __init__(
        self,
        impute_strategy: str = "median",
        output_dir: str = "data/ml_features"
    ):
        """
        Initialize the enhancer
        
        Args:
            impute_strategy: Strategy for imputing missing values
            output_dir: Directory to save enhanced features
        """
        self.impute_strategy = impute_strategy
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load financial features from CSV
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with financial features
        """
        logger.info(f"Loading financial features from {file_path}")
        df = pd.read_csv(file_path)
        
        # Basic info
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Check for non-null values in key columns
        for col in ['total_revenue', 'net_income', 'total_assets']:
            if col in df.columns:
                non_null = df[col].notna().sum()
                logger.info(f"- Rows with non-null {col}: {non_null} ({non_null/len(df):.1%})")
        
        return df
    
    def calculate_growth_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional growth ratios
        
        Args:
            df: DataFrame with financial features
            
        Returns:
            DataFrame with additional growth ratios
        """
        logger.info("Calculating additional growth ratios")
        
        # Check if we already have growth metrics
        existing_growth_cols = [col for col in df.columns if 'growth' in col.lower()]
        logger.info(f"Found {len(existing_growth_cols)} existing growth metrics: {existing_growth_cols}")
        
        # Additional growth metrics to calculate
        # We'll only calculate these if we have the required data
        growth_metrics = []
        
        # EPS growth
        if 'basic_eps' in df.columns and 'diluted_eps' in df.columns:
            # Use diluted EPS if available, otherwise basic EPS
            df['eps'] = df['diluted_eps'].fillna(df['basic_eps'])
            growth_metrics.append('eps')
        
        # Total assets growth
        if 'total_assets' in df.columns:
            growth_metrics.append('total_assets')
        
        # Total equity growth
        if 'total_equity' in df.columns:
            growth_metrics.append('total_equity')
        
        # Calculate growth for each metric
        # This is a simplistic approach - in a real scenario, we would need
        # to compare with previous year's data for the same company
        for metric in growth_metrics:
            # Calculate YoY growth if not already present
            growth_col = f"{metric}_growth_yoy"
            if growth_col not in df.columns:
                # We'll use the median value as a proxy for previous year
                # In a real scenario, we would use actual previous year data
                median_value = df[metric].median()
                df[growth_col] = (df[metric] - median_value) / median_value
                logger.info(f"- Added {growth_col}")
        
        return df
    
    def scale_pl_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale P&L positions by revenue
        
        Args:
            df: DataFrame with financial features
            
        Returns:
            DataFrame with scaled P&L positions
        """
        logger.info("Scaling P&L positions by revenue")
        
        # Check if we have revenue
        if 'total_revenue' not in df.columns:
            logger.warning("total_revenue not found, skipping P&L scaling")
            return df
        
        # P&L positions to scale
        pl_positions = [
            'gross_profit',
            'operating_income',
            'net_income',
            'ebitda'
        ]
        
        # Scale each position
        for position in pl_positions:
            if position in df.columns:
                # Create scaled column
                scaled_col = f"{position}_to_revenue"
                # Only scale where revenue is positive
                mask = (df['total_revenue'] > 0) & df[position].notna()
                df.loc[mask, scaled_col] = df.loc[mask, position] / df.loc[mask, 'total_revenue']
                logger.info(f"- Added {scaled_col}")
        
        return df
    
    def scale_bs_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale balance sheet positions by total assets
        
        Args:
            df: DataFrame with financial features
            
        Returns:
            DataFrame with scaled balance sheet positions
        """
        logger.info("Scaling balance sheet positions by total assets")
        
        # Check if we have total assets
        if 'total_assets' not in df.columns:
            logger.warning("total_assets not found, skipping balance sheet scaling")
            return df
        
        # Balance sheet positions to scale
        bs_positions = [
            'total_debt',
            'total_equity',
            'cash_and_equivalents',
            'current_assets',
            'current_liabilities',
            'working_capital',
            'property_plant_equipment'
        ]
        
        # Scale each position
        for position in bs_positions:
            if position in df.columns:
                # Create scaled column
                scaled_col = f"{position}_to_assets"
                # Only scale where total assets is positive
                mask = (df['total_assets'] > 0) & df[position].notna()
                df.loc[mask, scaled_col] = df.loc[mask, position] / df.loc[mask, 'total_assets']
                logger.info(f"- Added {scaled_col}")
        
        return df
    
    def scale_cf_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale cash flow positions by operating cash flow
        
        Args:
            df: DataFrame with financial features
            
        Returns:
            DataFrame with scaled cash flow positions
        """
        logger.info("Scaling cash flow positions by operating cash flow")
        
        # Check if we have operating cash flow
        if 'operating_cash_flow' not in df.columns:
            logger.warning("operating_cash_flow not found, skipping cash flow scaling")
            return df
        
        # Cash flow positions to scale
        cf_positions = [
            'free_cash_flow',
            'capital_expenditure',
            'financing_cash_flow',
            'investing_cash_flow'
        ]
        
        # Scale each position
        for position in cf_positions:
            if position in df.columns:
                # Create scaled column
                scaled_col = f"{position}_to_ocf"
                # Only scale where operating cash flow is positive
                mask = (df['operating_cash_flow'] > 0) & df[position].notna()
                df.loc[mask, scaled_col] = df.loc[mask, position] / df.loc[mask, 'operating_cash_flow']
                logger.info(f"- Added {scaled_col}")
        
        return df
    
    def calculate_efficiency_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate efficiency ratios
        
        Args:
            df: DataFrame with financial features
            
        Returns:
            DataFrame with additional efficiency ratios
        """
        logger.info("Calculating efficiency ratios")
        
        # Calculate Asset Turnover if not already present
        if 'asset_turnover' not in df.columns and 'total_revenue' in df.columns and 'total_assets' in df.columns:
            mask = (df['total_assets'] > 0) & df['total_revenue'].notna()
            df.loc[mask, 'asset_turnover'] = df.loc[mask, 'total_revenue'] / df.loc[mask, 'total_assets']
            logger.info("- Added asset_turnover")
        
        # Calculate Return on Assets (ROA) if not already present
        if 'roa' not in df.columns and 'net_income' in df.columns and 'total_assets' in df.columns:
            mask = (df['total_assets'] > 0) & df['net_income'].notna()
            df.loc[mask, 'roa'] = df.loc[mask, 'net_income'] / df.loc[mask, 'total_assets']
            logger.info("- Added roa")
        
        # Calculate Return on Equity (ROE) if not already present
        if 'roe' not in df.columns and 'net_income' in df.columns and 'total_equity' in df.columns:
            mask = (df['total_equity'] > 0) & df['net_income'].notna()
            df.loc[mask, 'roe'] = df.loc[mask, 'net_income'] / df.loc[mask, 'total_equity']
            logger.info("- Added roe")
        
        # Calculate Debt to Equity if not already present
        if 'debt_to_equity' not in df.columns and 'total_debt' in df.columns and 'total_equity' in df.columns:
            mask = (df['total_equity'] > 0) & df['total_debt'].notna()
            df.loc[mask, 'debt_to_equity'] = df.loc[mask, 'total_debt'] / df.loc[mask, 'total_equity']
            logger.info("- Added debt_to_equity")
        
        # Calculate Current Ratio if not already present
        if 'current_ratio' not in df.columns and 'current_assets' in df.columns and 'current_liabilities' in df.columns:
            mask = (df['current_liabilities'] > 0) & df['current_assets'].notna()
            df.loc[mask, 'current_ratio'] = df.loc[mask, 'current_assets'] / df.loc[mask, 'current_liabilities']
            logger.info("- Added current_ratio")
        
        # Calculate Gross Margin if not already present
        if 'gross_margin' not in df.columns and 'gross_profit' in df.columns and 'total_revenue' in df.columns:
            mask = (df['total_revenue'] > 0) & df['gross_profit'].notna()
            df.loc[mask, 'gross_margin'] = df.loc[mask, 'gross_profit'] / df.loc[mask, 'total_revenue']
            logger.info("- Added gross_margin")
        
        # Calculate Net Margin if not already present
        if 'net_margin' not in df.columns and 'net_income' in df.columns and 'total_revenue' in df.columns:
            mask = (df['total_revenue'] > 0) & df['net_income'].notna()
            df.loc[mask, 'net_margin'] = df.loc[mask, 'net_income'] / df.loc[mask, 'total_revenue']
            logger.info("- Added net_margin")
        
        return df
    
    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values
        
        Args:
            df: DataFrame with financial features
            
        Returns:
            DataFrame with imputed values
        """
        logger.info(f"Imputing missing values using {self.impute_strategy} strategy")
        
        # Get columns to impute (exclude setup_id and label)
        cols_to_impute = [col for col in df.columns if col not in ['setup_id', 'label']]
        
        # Count missing values before imputation
        missing_before = df[cols_to_impute].isna().sum().sum()
        logger.info(f"- Missing values before imputation: {missing_before}")
        
        # Create a copy of the DataFrame for imputation
        df_imputed = df.copy()
        
        # Identify columns with all NaN values
        all_nan_cols = [col for col in cols_to_impute if df[col].isna().all()]
        if all_nan_cols:
            logger.warning(f"Columns with all NaN values (will be filled with 0): {all_nan_cols}")
            # Fill these columns with 0
            for col in all_nan_cols:
                df_imputed[col] = 0
        
        # Identify columns with at least one non-NaN value
        imputable_cols = [col for col in cols_to_impute if col not in all_nan_cols]
        
        if imputable_cols:
            # Create imputer
            imputer = SimpleImputer(strategy=self.impute_strategy)
            
            # Impute missing values
            df_imputed[imputable_cols] = imputer.fit_transform(df[imputable_cols])
        
        # Count missing values after imputation
        missing_after = df_imputed[cols_to_impute].isna().sum().sum()
        logger.info(f"- Missing values after imputation: {missing_after}")
        
        return df_imputed
    
    def enhance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance financial features
        
        Args:
            df: DataFrame with financial features
            
        Returns:
            DataFrame with enhanced features
        """
        # Calculate additional growth ratios
        df = self.calculate_growth_ratios(df)
        
        # Scale P&L positions by revenue
        df = self.scale_pl_positions(df)
        
        # Scale balance sheet positions by total assets
        df = self.scale_bs_positions(df)
        
        # Scale cash flow positions by operating cash flow
        df = self.scale_cf_positions(df)
        
        # Calculate efficiency ratios
        df = self.calculate_efficiency_ratios(df)
        
        # Count features added
        logger.info(f"Enhanced features: {len(df.columns)} columns")
        
        return df
    
    def process_file(self, input_file: str, output_file: str = None, impute: bool = True) -> str:
        """
        Process a financial features file
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            impute: Whether to impute missing values
            
        Returns:
            Path to output CSV file
        """
        # Load data
        df = self.load_data(input_file)
        
        # Enhance features
        df_enhanced = self.enhance_features(df)
        
        # Impute missing values if requested
        if impute:
            df_enhanced = self.impute_missing_values(df_enhanced)
        
        # Generate output file name if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_path = Path(input_file)
            output_file = self.output_dir / f"enhanced_{input_path.name.replace('.csv', '')}_{timestamp}.csv"
        
        # Save enhanced features
        df_enhanced.to_csv(output_file, index=False)
        logger.info(f"Saved enhanced features to {output_file}")
        
        # Print summary
        logger.info("\nEnhanced Features Summary:")
        logger.info(f"- Input rows: {len(df)}")
        logger.info(f"- Input columns: {len(df.columns)}")
        logger.info(f"- Output rows: {len(df_enhanced)}")
        logger.info(f"- Output columns: {len(df_enhanced.columns)}")
        logger.info(f"- New features added: {len(df_enhanced.columns) - len(df.columns)}")
        
        return str(output_file)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Enhance financial features')
    parser.add_argument('--input', required=True,
                       help='Path to input financial features CSV')
    parser.add_argument('--output',
                       help='Path to output enhanced features CSV')
    parser.add_argument('--output-dir', default='data/ml_features',
                       help='Directory to save enhanced features')
    parser.add_argument('--impute-strategy', choices=['mean', 'median', 'most_frequent'], default='median',
                       help='Strategy for imputing missing values')
    parser.add_argument('--no-impute', action='store_true',
                       help='Do not impute missing values')
    
    args = parser.parse_args()
    
    # Initialize enhancer
    enhancer = FinancialFeatureEnhancer(
        impute_strategy=args.impute_strategy,
        output_dir=args.output_dir
    )
    
    # Process file
    enhancer.process_file(
        input_file=args.input,
        output_file=args.output,
        impute=not args.no_impute
    )

if __name__ == '__main__':
    main() 