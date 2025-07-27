#!/usr/bin/env python3
"""
embed_fundamentals_duckdb.py - DuckDB-based Fundamentals Domain Embedding Pipeline

Processes fundamentals data from DuckDB database, creates embeddings,
and stores in LanceDB with performance labels for RAG retrieval.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import sys

import lancedb
from sentence_transformers import SentenceTransformer

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.setup_validator_duckdb import SetupValidatorDuckDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FundamentalsEmbedderDuckDB:
    """DuckDB-based Fundamentals Domain Embedding Pipeline"""
    
    def __init__(self, db_path="data/sentiment_system.duckdb", lancedb_dir="lancedb_store", 
                 embedding_model="all-MiniLM-L6-v2", specific_setup_ids=None):
        self.db_path = Path(db_path)
        self.lancedb_dir = Path(lancedb_dir)
        self.specific_setup_ids = specific_setup_ids
        
        # Initialize setup validator
        self.setup_validator = SetupValidatorDuckDB(db_path=str(self.db_path))
        
        # Use specific setup_ids if provided, otherwise use all confirmed setups
        if self.specific_setup_ids:
            self.target_setup_ids = set(self.specific_setup_ids)
            logger.info(f"Using specific setup_ids: {self.specific_setup_ids}")
        else:
            self.target_setup_ids = self.setup_validator.confirmed_setup_ids
            logger.info(f"Setup validator initialized with {len(self.setup_validator.confirmed_setup_ids)} confirmed setups")
        
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.db = lancedb.connect(str(self.lancedb_dir))
        
        # Data containers
        self.fundamentals_data = None
        self.ratios_data = None
        self.labels_data = None
        
    def load_data(self):
        """Load financial data from DuckDB with setup validation"""
        logger.info("Loading financial data from DuckDB...")
        
        # Load labels first (filtered by confirmed setups)
        self.labels_data = self.setup_validator.get_labels_for_confirmed_setups()
        logger.info(f"Loaded {len(self.labels_data)} confirmed setup labels")
        
        # Load fundamentals and ratios for all confirmed setups
        all_fundamentals = []
        all_ratios = []
        
        for setup_id in self.target_setup_ids:
            # Get fundamentals for this setup (already filtered by date)
            fundamentals = self.setup_validator.get_fundamentals_for_setup(setup_id)
            if not fundamentals.empty:
                fundamentals['setup_id'] = setup_id
                all_fundamentals.append(fundamentals)
            
            # Get financial ratios for this setup
            ratios = self.setup_validator.get_financial_ratios_for_setup(setup_id)
            if not ratios.empty:
                ratios['setup_id'] = setup_id
                all_ratios.append(ratios)
        
        # Combine all data
        if all_fundamentals:
            self.fundamentals_data = pd.concat(all_fundamentals, ignore_index=True)
            logger.info(f"Loaded {len(self.fundamentals_data)} fundamentals records")
        else:
            self.fundamentals_data = pd.DataFrame()
            logger.warning("No fundamentals data found")
        
        if all_ratios:
            self.ratios_data = pd.concat(all_ratios, ignore_index=True)
            logger.info(f"Loaded {len(self.ratios_data)} ratios records")
        else:
            self.ratios_data = pd.DataFrame()
            logger.warning("No ratios data found")
    
    def merge_financial_data(self):
        """Merge fundamentals and ratios data"""
        if self.fundamentals_data.empty and self.ratios_data.empty:
            return pd.DataFrame()
        
        # If no ratios data, return fundamentals only
        if self.ratios_data.empty:
            return self.fundamentals_data
        
        # If no fundamentals data, return ratios only
        if self.fundamentals_data.empty:
            return self.ratios_data
        
        # Create merge keys (using available columns)
        self.fundamentals_data['merge_key'] = (
            self.fundamentals_data['ticker'].astype(str) + "_" +
            self.fundamentals_data['date'].astype(str)
        )
        
        self.ratios_data['merge_key'] = (
            self.ratios_data['ticker'].astype(str) + "_" +
            self.ratios_data['period_end'].astype(str)
        )
        
        # Merge dataframes
        merged_df = pd.merge(
            self.fundamentals_data, self.ratios_data,
            on=['merge_key', 'setup_id'], how='outer', suffixes=('_fund', '_ratio')
        )
        
        # Clean up columns - handle different schemas
        # Fundamentals uses 'date', ratios use 'period_end'
        if 'date_fund' in merged_df.columns and 'period_end_ratio' in merged_df.columns:
            merged_df['date'] = merged_df['date_fund'].fillna(merged_df['period_end_ratio'])
            merged_df = merged_df.drop(columns=['date_fund', 'period_end_ratio'])
        
        # Ticker should be the same
        if 'ticker_fund' in merged_df.columns and 'ticker_ratio' in merged_df.columns:
            merged_df['ticker'] = merged_df['ticker_fund'].fillna(merged_df['ticker_ratio'])
            merged_df = merged_df.drop(columns=['ticker_fund', 'ticker_ratio'])
        
        merged_df = merged_df.drop(columns=['merge_key'])
        logger.info(f"Merged data contains {len(merged_df)} records")
        return merged_df
    
    def create_financial_summary(self, row):
        """Convert financial data into text summary"""
        ticker = row.get('ticker', 'Unknown')
        period_end = row.get('period_end', 'Unknown')
        report_type = row.get('report_type', 'Unknown')
        setup_id = row.get('setup_id', 'Unknown')
        
        summary_parts = []
        summary_parts.append(f"Financial Analysis for {ticker} (Setup: {setup_id}) - {report_type.title()} Report ending {period_end}")
        
        # Revenue and Profitability
        revenue = row.get('revenue')
        net_income = row.get('net_income')
        
        if pd.notna(revenue) and revenue != 0:
            revenue_mil = revenue / 1_000_000
            summary_parts.append(f"Revenue of ${revenue_mil:.1f} million")
            
            gross_margin = row.get('gross_margin', 0) * 100 if pd.notna(row.get('gross_margin')) else 0
            operating_margin = row.get('operating_margin', 0) * 100 if pd.notna(row.get('operating_margin')) else 0
            net_margin = row.get('net_margin', 0) * 100 if pd.notna(row.get('net_margin')) else 0
            
            summary_parts.append(f"with gross margin of {gross_margin:.1f}%, operating margin of {operating_margin:.1f}%")
            
            if pd.notna(net_income):
                net_income_mil = net_income / 1_000_000
                if net_income > 0:
                    summary_parts.append(f"delivering net profit of ${net_income_mil:.1f} million ({net_margin:.1f}% margin)")
                else:
                    summary_parts.append(f"reporting net loss of ${abs(net_income_mil):.1f} million")
        
        # Balance Sheet
        total_assets = row.get('total_assets')
        debt_to_equity = row.get('debt_to_equity')
        
        if pd.notna(total_assets) and total_assets != 0:
            assets_bil = total_assets / 1_000_000_000
            summary_parts.append(f"Total assets of ${assets_bil:.1f} billion")
            
            if pd.notna(debt_to_equity):
                if debt_to_equity < 0.3:
                    debt_desc = "conservative debt levels"
                elif debt_to_equity < 0.7:
                    debt_desc = "moderate debt levels"
                else:
                    debt_desc = "high debt levels"
                summary_parts.append(f"with {debt_desc} (D/E ratio: {debt_to_equity:.2f})")
        
        # Profitability Ratios
        roe = row.get('roe')
        roa = row.get('roa')
        
        profitability_parts = []
        if pd.notna(roe) and roe != 0:
            roe_pct = roe * 100
            if roe_pct > 15:
                profitability_parts.append(f"excellent return on equity of {roe_pct:.1f}%")
            elif roe_pct > 10:
                profitability_parts.append(f"good return on equity of {roe_pct:.1f}%")
            elif roe_pct > 0:
                profitability_parts.append(f"modest return on equity of {roe_pct:.1f}%")
            else:
                profitability_parts.append(f"negative return on equity of {roe_pct:.1f}%")
        
        if pd.notna(roa) and roa != 0:
            roa_pct = roa * 100
            if roa_pct > 5:
                profitability_parts.append(f"strong asset efficiency (ROA: {roa_pct:.1f}%)")
            elif roa_pct > 0:
                profitability_parts.append(f"moderate asset efficiency (ROA: {roa_pct:.1f}%)")
            else:
                profitability_parts.append(f"poor asset efficiency (ROA: {roa_pct:.1f}%)")
        
        if profitability_parts:
            summary_parts.append("Profitability metrics show " + " and ".join(profitability_parts))
        
        # Liquidity
        current_ratio = row.get('current_ratio')
        if pd.notna(current_ratio):
            if current_ratio > 2:
                summary_parts.append(f"Strong liquidity position (current ratio: {current_ratio:.2f})")
            elif current_ratio > 1:
                summary_parts.append(f"Adequate liquidity (current ratio: {current_ratio:.2f})")
            else:
                summary_parts.append(f"Tight liquidity (current ratio: {current_ratio:.2f})")
        
        # Valuation
        pe_ratio = row.get('pe_ratio')
        pb_ratio = row.get('pb_ratio')
        
        valuation_parts = []
        if pd.notna(pe_ratio) and pe_ratio > 0:
            if pe_ratio < 10:
                valuation_parts.append(f"attractive valuation (P/E: {pe_ratio:.1f})")
            elif pe_ratio < 20:
                valuation_parts.append(f"reasonable valuation (P/E: {pe_ratio:.1f})")
            else:
                valuation_parts.append(f"premium valuation (P/E: {pe_ratio:.1f})")
        
        if pd.notna(pb_ratio):
            if pb_ratio < 1:
                valuation_parts.append(f"trading below book value (P/B: {pb_ratio:.2f})")
            elif pb_ratio < 3:
                valuation_parts.append(f"reasonable price-to-book ratio ({pb_ratio:.2f})")
            else:
                valuation_parts.append(f"high price-to-book ratio ({pb_ratio:.2f})")
        
        if valuation_parts:
            summary_parts.append("Valuation indicates " + " and ".join(valuation_parts))
        
        return ". ".join(summary_parts) + "."
    
    def create_records_with_labels(self, merged_df):
        """Create embedding records with performance labels"""
        if merged_df.empty:
            return []
        
        records = []
        
        for idx, row in merged_df.iterrows():
            if pd.isna(row.get('ticker')) or pd.isna(row.get('period_end')):
                continue
            
            setup_id = row.get('setup_id', '')
            if not setup_id:
                continue
            
            financial_summary = self.create_financial_summary(row)
            
            record = {
                'id': f"{row.get('ticker', '')}_{row.get('report_type', '')}_{row.get('period_end', '')}_{setup_id}",
                'setup_id': setup_id,
                'ticker': str(row.get('ticker', '')),
                'report_type': str(row.get('report_type', '')),
                'period_end': str(row.get('period_end', '')),
                'financial_summary': financial_summary,
                'text_length': len(financial_summary),
                'summary_type': 'comprehensive_financial_analysis',
                
                # Core Metrics
                'revenue': float(row.get('revenue', 0)) if pd.notna(row.get('revenue')) else 0,
                'net_income': float(row.get('net_income', 0)) if pd.notna(row.get('net_income')) else 0,
                'total_assets': float(row.get('total_assets', 0)) if pd.notna(row.get('total_assets')) else 0,
                
                # Key Ratios
                'roe': float(row.get('roe', 0)) if pd.notna(row.get('roe')) else 0,
                'roa': float(row.get('roa', 0)) if pd.notna(row.get('roa')) else 0,
                'debt_to_equity': float(row.get('debt_to_equity', 0)) if pd.notna(row.get('debt_to_equity')) else 0,
                'current_ratio': float(row.get('current_ratio', 0)) if pd.notna(row.get('current_ratio')) else 0,
                'pe_ratio': float(row.get('pe_ratio', 0)) if pd.notna(row.get('pe_ratio')) else 0,
                
                'embedded_at': datetime.now().isoformat()
            }
            
            # Add performance labels
            setup_labels = self.labels_data[self.labels_data['setup_id'] == setup_id]
            if not setup_labels.empty:
                latest_label = setup_labels.iloc[-1]
                record.update({
                    'stock_return_10d': float(latest_label.get('stock_return_10d', 0)),
                    'outperformance_10d': float(latest_label.get('outperformance_10d', 0)),
                    'days_outperformed_10d': int(latest_label.get('days_outperformed_10d', 0)),
                    'has_performance_labels': True
                })
            else:
                record.update({
                    'stock_return_10d': 0.0,
                    'outperformance_10d': 0.0,
                    'days_outperformed_10d': 0,
                    'has_performance_labels': False
                })
            
            records.append(record)
        
        logger.info(f"Created {len(records)} fundamentals records for confirmed setups")
        return records
    
    def create_embeddings(self, records):
        """Create embeddings for financial summaries"""
        if not records:
            return
        
        texts = [record['financial_summary'] for record in records]
        logger.info(f"Generating embeddings for {len(texts)} financial summaries")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        for record, embedding in zip(records, embeddings):
            record['vector'] = embedding.tolist()
        
        logger.info("Embeddings created successfully")
    
    def store_in_lancedb(self, records, table_name="fundamentals_embeddings"):
        """Store records in LanceDB"""
        if not records:
            logger.warning("No records to store")
            return
            
        logger.info(f"Storing {len(records)} records in LanceDB table: {table_name}")
        
        df = pd.DataFrame(records)
        df['vector'] = df['vector'].apply(lambda x: np.array(x, dtype=np.float32))
        
        # Handle data types
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'vector':
                df[col] = df[col].astype(str).replace('nan', '')
        
        try:
            if table_name in self.db.table_names():
                self.db.drop_table(table_name)
                
            table = self.db.create_table(table_name, df)
            logger.info(f"Successfully created table '{table_name}' with {len(df)} records")
            
        except Exception as e:
            logger.error(f"Error storing data in LanceDB: {e}")
            raise
    
    def run_pipeline(self):
        """Execute the complete fundamentals embedding pipeline"""
        logger.info("Starting DuckDB-based Fundamentals Domain Embedding Pipeline")
        
        self.load_data()
        merged_df = self.merge_financial_data()
        records = self.create_records_with_labels(merged_df)
        self.create_embeddings(records)
        self.store_in_lancedb(records)
        
        logger.info("Pipeline Summary:")
        logger.info(f"  Total records: {len(records)}")
        logger.info(f"  With performance labels: {len([r for r in records if r.get('has_performance_labels')])}")
        logger.info(f"  Unique setups: {len(set(r['setup_id'] for r in records))}")
        
        # Close DuckDB connection
        self.setup_validator.close()


def run_fundamentals_pipeline():
    """Run the DuckDB-based fundamentals embedding pipeline"""
    embedder = FundamentalsEmbedderDuckDB()
    embedder.run_pipeline()


if __name__ == "__main__":
    run_fundamentals_pipeline() 