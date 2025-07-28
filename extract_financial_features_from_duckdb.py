#!/usr/bin/env python3
"""
Extract Financial Features From DuckDB (Enhanced)

This script extracts comprehensive financial features using the new FinancialPreprocessor
module, which provides:
- Comprehensive financial ratio calculations (P&L/revenue, balance sheet/total assets)
- Consistent imputation and scaling that saves parameters for prediction
- Data quality validation and checks
- Proper ticker format handling
- 1-3 year growth metrics and trend indicators

Usage:
    python extract_financial_features_from_duckdb.py --mode [training|prediction] --setup-list [file]
"""

import os
import sys
import argparse
import logging
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Import the new financial preprocessor
from financial_preprocessor import FinancialPreprocessor, extract_and_preprocess_financial_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedFinancialFeaturesExtractor:
    """Enhanced financial features extractor using the new preprocessor"""
    
    def __init__(
        self,
        db_path: str = "data/sentiment_system.duckdb",
        output_dir: str = "data/ml_features",
        preprocessing_params_path: str = "models/financial/preprocessing_params.pkl"
    ):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.preprocessing_params_path = preprocessing_params_path
        
        # Initialize the financial preprocessor
        self.preprocessor = FinancialPreprocessor(
            db_path=db_path,
            preprocessing_params_path=preprocessing_params_path
        )
    
    def extract_financial_features(
        self,
        setup_ids: List[str],
        mode: str = 'training',
        add_labels: bool = True
    ) -> Dict[str, Any]:
        """
        Extract comprehensive financial features using the new preprocessor
        
        Args:
            setup_ids: List of setup IDs to process
            mode: Either 'training' or 'prediction'
            add_labels: Whether to add labels (only applies to training mode)
            
        Returns:
            Dictionary with extraction results and metadata
        """
        logger.info(f"🔄 Enhanced financial features extraction for {len(setup_ids)} setups in {mode} mode...")
        
        try:
            # Extract comprehensive financial data using the new preprocessor
            logger.info("📊 Extracting comprehensive financial metrics...")
            metrics_list = self.preprocessor.extract_comprehensive_financial_data(setup_ids)
            
            if not metrics_list:
                logger.warning("No financial metrics extracted")
                return {"error": "No financial metrics extracted"}
            
            logger.info(f"✅ Extracted metrics for {len(metrics_list)} setups")
            
            # Prepare ML features with consistent preprocessing
            logger.info("🔧 Applying preprocessing pipeline...")
            df = self.preprocessor.prepare_ml_features(metrics_list, mode=mode)
            
            if df.empty:
                logger.warning("Empty DataFrame after preprocessing")
                return {"error": "Empty DataFrame after preprocessing"}
            
            # Add labels for training mode
            if mode == 'training' and add_labels:
                df = self._add_labels_to_features(df)
            
            # Generate data quality report
            quality_report = self._generate_data_quality_report(df, metrics_list)
            
            # Save to CSV with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            table_name = f"financial_ml_features_{mode}"
            output_file = self.output_dir / f"{table_name}_{timestamp}.csv"
            
            df.to_csv(output_file, index=False)
            
            # Save to DuckDB table as well
            self._save_to_duckdb(df, table_name)
            
            # Log results
            feature_count = len([col for col in df.columns if col not in ['setup_id', 'ticker', 'label']])
            
            logger.info(f"✅ Enhanced financial features extracted and saved:")
            logger.info(f"- Table: {table_name}")
            logger.info(f"- Features: {feature_count}")
            logger.info(f"- Rows: {len(df)}")
            logger.info(f"- CSV file: {output_file}")
            logger.info(f"- Mode: {mode}")
            if mode == 'training':
                logger.info(f"- Preprocessing parameters saved for prediction consistency")
            else:
                logger.info(f"- Used saved preprocessing parameters for consistency")
            
            return {
                "table_name": table_name,
                "feature_count": feature_count,
                "row_count": len(df),
                "output_file": str(output_file),
                "quality_report": quality_report,
                "mode": mode,
                "preprocessing_applied": True
            }
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced financial features extraction: {str(e)}")
            raise
    
    def _add_labels_to_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add labels to training features"""
        logger.info("🏷️ Adding labels to training features...")
        
        conn = None
        try:
            conn = duckdb.connect(self.db_path)
            
            # Get labels for the setup IDs in the DataFrame
            setup_ids = df['setup_id'].tolist()
            placeholders = ','.join(['?' for _ in setup_ids])
            
            labels_query = f"""
            SELECT 
                setup_id,
                AVG(outperformance_day) as outperformance_10d
            FROM daily_labels
            WHERE setup_id IN ({placeholders})
            AND day_number <= 10
            GROUP BY setup_id
            HAVING COUNT(*) >= 5  -- Require at least 5 days
            """
            
            labels_df = conn.execute(labels_query, setup_ids).df()
            
            if not labels_df.empty:
                # Merge labels with features
                df = df.merge(labels_df, on='setup_id', how='left')
                
                # Convert to categorical labels (optional - depends on your preference)
                # You can uncomment this if you want categorical labels
                # df['label'] = self._convert_to_categorical_labels(df['outperformance_10d'])
                
                # Use continuous labels
                df['label'] = df['outperformance_10d']
                df = df.drop('outperformance_10d', axis=1)
                
                labeled_count = df['label'].notna().sum()
                logger.info(f"✅ Added labels to {labeled_count}/{len(df)} setups")
            else:
                logger.warning("No labels found for the provided setup IDs")
                df['label'] = None
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding labels: {str(e)}")
            df['label'] = None
            return df
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    def _convert_to_categorical_labels(self, values: pd.Series) -> pd.Series:
        """Convert continuous outperformance to categorical labels"""
        # Calculate percentiles for balanced classes
        p33 = values.quantile(1/3)
        p67 = values.quantile(2/3)
        
        # Convert to categories (-1, 0, 1)
        labels = pd.Series(index=values.index, dtype=int)
        labels[values <= p33] = -1  # Underperform
        labels[(values > p33) & (values <= p67)] = 0  # Neutral
        labels[values > p67] = 1  # Outperform
        
        return labels
    
    def _generate_data_quality_report(
        self,
        df: pd.DataFrame,
        metrics_list: List
    ) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        logger.info("📋 Generating data quality report...")
        
        # Basic DataFrame statistics
        total_setups = len(df)
        total_features = len([col for col in df.columns if col not in ['setup_id', 'ticker', 'label']])
        
        # Missing value analysis
        feature_cols = [col for col in df.columns if col not in ['setup_id', 'ticker', 'label']]
        missing_analysis = df[feature_cols].isnull().sum().sort_values(ascending=False)
        missing_pct = (missing_analysis / total_setups * 100).round(2)
        
        # Features with high missing rates
        high_missing_features = missing_pct[missing_pct > 50].to_dict()
        
        # Data completeness by setup
        setup_completeness = df[feature_cols].notna().sum(axis=1) / total_features
        avg_completeness = setup_completeness.mean()
        min_completeness = setup_completeness.min()
        
        # Financial data quality scores (from preprocessor)
        quality_scores = []
        data_ages = []
        
        for metrics in metrics_list:
            if hasattr(metrics, 'data_completeness_score') and metrics.data_completeness_score is not None:
                quality_scores.append(metrics.data_completeness_score)
            if hasattr(metrics, 'financial_data_age_days') and metrics.financial_data_age_days is not None:
                data_ages.append(metrics.financial_data_age_days)
        
        avg_quality_score = np.mean(quality_scores) if quality_scores else 0.0
        avg_data_age = np.mean(data_ages) if data_ages else None
        
        # Feature categories coverage
        profitability_features = [col for col in feature_cols if 'margin' in col or 'return_on' in col]
        liquidity_features = [col for col in feature_cols if 'ratio' in col and any(term in col for term in ['current', 'quick', 'cash'])]
        growth_features = [col for col in feature_cols if 'growth' in col]
        
        profitability_coverage = (df[profitability_features].notna().sum().sum() / 
                                (len(profitability_features) * total_setups) * 100) if profitability_features else 0
        liquidity_coverage = (df[liquidity_features].notna().sum().sum() / 
                            (len(liquidity_features) * total_setups) * 100) if liquidity_features else 0
        growth_coverage = (df[growth_features].notna().sum().sum() / 
                         (len(growth_features) * total_setups) * 100) if growth_features else 0
        
        report = {
            "total_setups": total_setups,
            "total_features": total_features,
            "average_completeness_pct": round(avg_completeness * 100, 2),
            "minimum_completeness_pct": round(min_completeness * 100, 2),
            "average_quality_score": round(avg_quality_score, 3),
            "average_data_age_days": round(avg_data_age, 1) if avg_data_age else None,
            "high_missing_features": high_missing_features,
            "feature_category_coverage": {
                "profitability_pct": round(profitability_coverage, 2),
                "liquidity_pct": round(liquidity_coverage, 2),
                "growth_pct": round(growth_coverage, 2)
            },
            "preprocessing_status": "Applied consistently" if hasattr(self.preprocessor, 'preprocessing_fitted') else "Not applied"
        }
        
        # Log key quality metrics
        logger.info(f"📊 Data Quality Summary:")
        logger.info(f"  - Average completeness: {report['average_completeness_pct']:.1f}%")
        logger.info(f"  - Average quality score: {report['average_quality_score']:.3f}")
        logger.info(f"  - Profitability feature coverage: {report['feature_category_coverage']['profitability_pct']:.1f}%")
        logger.info(f"  - Liquidity feature coverage: {report['feature_category_coverage']['liquidity_pct']:.1f}%")
        logger.info(f"  - Growth feature coverage: {report['feature_category_coverage']['growth_pct']:.1f}%")
        
        if high_missing_features:
            logger.warning(f"  - Features with >50% missing data: {len(high_missing_features)}")
            for feature, pct in list(high_missing_features.items())[:5]:  # Show top 5
                logger.warning(f"    * {feature}: {pct:.1f}% missing")
        
        return report
    
    def _save_to_duckdb(self, df: pd.DataFrame, table_name: str):
        """Save DataFrame to DuckDB table"""
        try:
            conn = duckdb.connect(self.db_path)
            
            # Drop existing table if it exists
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Create table from DataFrame
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            
            logger.info(f"✅ Saved to DuckDB table: {table_name}")
            
        except Exception as e:
            logger.error(f"Error saving to DuckDB: {str(e)}")
        finally:
            try:
                conn.close()
            except:
                pass
    
    def validate_ticker_coverage(self, setup_ids: List[str]) -> Dict[str, Any]:
        """
        Validate ticker format coverage and data availability
        
        Args:
            setup_ids: List of setup IDs to validate
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"🔍 Validating ticker coverage for {len(setup_ids)} setups...")
        
        conn = None
        try:
            conn = duckdb.connect(self.db_path)
            
            # Get setup ticker information
            placeholders = ','.join(['?' for _ in setup_ids])
            setup_query = f"""
            SELECT setup_id, lse_ticker, yahoo_ticker, spike_timestamp
            FROM setups 
            WHERE setup_id IN ({placeholders})
            """
            
            setups_df = conn.execute(setup_query, setup_ids).df()
            
            validation_results = {
                "total_setups": len(setups_df),
                "lse_ticker_coverage": 0,
                "yahoo_ticker_coverage": 0,
                "fundamentals_coverage": 0,
                "financial_ratios_coverage": 0,
                "both_tables_coverage": 0,
                "no_data_setups": []
            }
            
            fundamentals_count = 0
            ratios_count = 0
            both_count = 0
            
            for _, row in setups_df.iterrows():
                setup_id = row['setup_id']
                lse_ticker = row['lse_ticker']
                yahoo_ticker = row['yahoo_ticker']
                spike_timestamp = row['spike_timestamp']
                
                # Count ticker availability
                if lse_ticker:
                    validation_results["lse_ticker_coverage"] += 1
                if yahoo_ticker:
                    validation_results["yahoo_ticker_coverage"] += 1
                
                # Check data availability in fundamentals
                ticker = lse_ticker or yahoo_ticker
                if ticker:
                    # Check fundamentals with .L suffix
                    fundamentals_ticker = f"{ticker}.L" if not ticker.endswith('.L') else ticker
                    fundamentals_check = conn.execute(
                        "SELECT COUNT(*) FROM fundamentals WHERE ticker = ? AND date <= ?",
                        [fundamentals_ticker, spike_timestamp]
                    ).fetchone()[0]
                    
                    # Check financial_ratios with .L suffix
                    ratios_check = conn.execute(
                        "SELECT COUNT(*) FROM financial_ratios WHERE ticker = ? AND period_end <= ?",
                        [fundamentals_ticker, spike_timestamp]
                    ).fetchone()[0]
                    
                    if fundamentals_check > 0:
                        fundamentals_count += 1
                    if ratios_check > 0:
                        ratios_count += 1
                    if fundamentals_check > 0 and ratios_check > 0:
                        both_count += 1
                    
                    if fundamentals_check == 0 and ratios_check == 0:
                        validation_results["no_data_setups"].append(setup_id)
            
            # Calculate percentages
            total = validation_results["total_setups"]
            validation_results["lse_ticker_coverage"] = round(validation_results["lse_ticker_coverage"] / total * 100, 2)
            validation_results["yahoo_ticker_coverage"] = round(validation_results["yahoo_ticker_coverage"] / total * 100, 2)
            validation_results["fundamentals_coverage"] = round(fundamentals_count / total * 100, 2)
            validation_results["financial_ratios_coverage"] = round(ratios_count / total * 100, 2)
            validation_results["both_tables_coverage"] = round(both_count / total * 100, 2)
            
            # Log validation results
            logger.info(f"📈 Ticker Coverage Validation:")
            logger.info(f"  - LSE ticker availability: {validation_results['lse_ticker_coverage']:.1f}%")
            logger.info(f"  - Yahoo ticker availability: {validation_results['yahoo_ticker_coverage']:.1f}%")
            logger.info(f"  - Fundamentals data coverage: {validation_results['fundamentals_coverage']:.1f}%")
            logger.info(f"  - Financial ratios data coverage: {validation_results['financial_ratios_coverage']:.1f}%")
            logger.info(f"  - Both tables coverage: {validation_results['both_tables_coverage']:.1f}%")
            logger.info(f"  - Setups with no financial data: {len(validation_results['no_data_setups'])}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in ticker validation: {str(e)}")
            return {"error": str(e)}
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Extract enhanced financial features from DuckDB')
    parser.add_argument('--mode', choices=['training', 'prediction'], default='training',
                       help='Mode: training or prediction')
    parser.add_argument('--db-path', default='data/sentiment_system.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--output-dir', default='data/ml_features',
                       help='Directory to save output CSV files')
    parser.add_argument('--preprocessing-params-path', default='models/financial/preprocessing_params.pkl',
                       help='Path to save/load preprocessing parameters')
    parser.add_argument('--setup-list', help='File containing setup IDs to process (one per line)')
    parser.add_argument('--validate-coverage', action='store_true',
                       help='Run ticker coverage validation before extraction')
    parser.add_argument('--skip-labels', action='store_true',
                       help='Skip adding labels (useful for testing)')
    
    args = parser.parse_args()
    
    # Initialize enhanced extractor
    extractor = EnhancedFinancialFeaturesExtractor(
        db_path=args.db_path,
        output_dir=args.output_dir,
        preprocessing_params_path=args.preprocessing_params_path
    )
    
    # Get setup IDs
    setup_ids = None
    if args.setup_list:
        with open(args.setup_list, 'r') as f:
            setup_ids = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(setup_ids)} setup IDs from {args.setup_list}")
    else:
        # Use sample setup IDs for testing
        conn = duckdb.connect(args.db_path)
        setup_ids = conn.execute('SELECT DISTINCT setup_id FROM setups LIMIT 100').df()['setup_id'].tolist()
        conn.close()
        logger.info(f"Using {len(setup_ids)} sample setup IDs from database")
    
    if not setup_ids:
        logger.error("No setup IDs found to process")
        return
    
    # Validate ticker coverage if requested
    if args.validate_coverage:
        validation_results = extractor.validate_ticker_coverage(setup_ids)
        if "error" not in validation_results:
            # Filter out setups with no data if coverage is low
            if validation_results["both_tables_coverage"] < 50:
                logger.warning(f"Low data coverage ({validation_results['both_tables_coverage']:.1f}%), consider reviewing ticker formats")
    
    # Extract enhanced financial features
    result = extractor.extract_financial_features(
        setup_ids=setup_ids,
        mode=args.mode,
        add_labels=(args.mode == 'training' and not args.skip_labels)
    )
    
    # Print comprehensive summary
    logger.info("\n" + "="*70)
    logger.info("📊 ENHANCED FINANCIAL FEATURES EXTRACTION SUMMARY")
    logger.info("="*70)
    
    if "error" in result:
        logger.error(f"❌ Extraction failed: {result['error']}")
    else:
        logger.info(f"✅ Extraction successful!")
        logger.info(f"  - Mode: {result['mode']}")
        logger.info(f"  - Table: {result['table_name']}")
        logger.info(f"  - Features: {result['feature_count']}")
        logger.info(f"  - Rows: {result['row_count']}")
        logger.info(f"  - Output file: {result['output_file']}")
        logger.info(f"  - Preprocessing applied: {result['preprocessing_applied']}")
        
        # Print quality report summary
        quality = result.get('quality_report', {})
        if quality:
            logger.info(f"\n📋 Data Quality Report:")
            logger.info(f"  - Average completeness: {quality.get('average_completeness_pct', 'N/A')}%")
            logger.info(f"  - Quality score: {quality.get('average_quality_score', 'N/A')}")
            logger.info(f"  - Data age: {quality.get('average_data_age_days', 'N/A')} days")
            
            coverage = quality.get('feature_category_coverage', {})
            logger.info(f"  - Profitability coverage: {coverage.get('profitability_pct', 'N/A')}%")
            logger.info(f"  - Liquidity coverage: {coverage.get('liquidity_pct', 'N/A')}%")
            logger.info(f"  - Growth coverage: {coverage.get('growth_pct', 'N/A')}%")
    
    logger.info("\n" + "="*70)
    logger.info("🎯 Key Improvements:")
    logger.info("  ✅ Comprehensive financial ratios (P&L/revenue, balance sheet/total assets)")
    logger.info("  ✅ Consistent preprocessing with saved parameters")
    logger.info("  ✅ 1-3 year growth metrics and trend indicators")
    logger.info("  ✅ Data quality validation and reporting")
    logger.info("  ✅ Proper ticker format handling (.L suffix)")
    logger.info("  ✅ Enhanced feature engineering pipeline")
    logger.info("="*70)

if __name__ == "__main__":
    main() 