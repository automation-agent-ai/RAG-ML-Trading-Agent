#!/usr/bin/env python3
"""
embed_fundamentals_duckdb.py - DuckDB-based Fundamentals Domain Embedding Pipeline

Processes financial data from DuckDB database, creates embeddings,
and stores in LanceDB with performance labels for RAG retrieval.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import sys

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from embeddings.base_embedder import BaseEmbedder
from tools.setup_validator_duckdb import SetupValidatorDuckDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FundamentalsEmbedderDuckDB(BaseEmbedder):
    """DuckDB-based Fundamentals Domain Embedding Pipeline"""
    
    def __init__(self, db_path: str = "data/sentiment_system.duckdb", 
                 lancedb_dir: str = "lancedb_store",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 include_labels: bool = True,
                 mode: str = "training"):
        """
        Initialize Fundamentals Embedder with DuckDB backend
        
        Args:
            db_path: Path to DuckDB database
            lancedb_dir: Directory for LanceDB storage
            embedding_model: Model name for sentence transformer
            include_labels: Whether to include performance labels in embeddings
            mode: Either 'training' or 'prediction'
        """
        # Initialize base class
        super().__init__(
            db_path=db_path,
            lancedb_dir=lancedb_dir,
            embedding_model=embedding_model,
            include_labels=include_labels,
            mode=mode
        )
        
        # Data containers
        self.fundamentals_data = None
        
    def get_text_field_name(self) -> str:
        """Get the name of the text field to embed"""
        return "financial_summary"
        
    def load_data(self):
        """Load fundamentals data and labels from DuckDB"""
        logger.info("Loading fundamentals data from DuckDB...")
        
        # Load labels for confirmed setups
        super().load_labels()
        
        # Load fundamentals data for confirmed setups
        fundamentals_query = '''
            SELECT 
                f.*,
                s.spike_timestamp as setup_date
            FROM fundamentals f
            JOIN setups s ON f.ticker = s.yahoo_ticker AND f.setup_id = s.setup_id
            WHERE s.setup_id IN ({})
            ORDER BY s.setup_id
        '''.format(','.join([f"'{sid}'" for sid in self.setup_validator.confirmed_setup_ids]))
        
        self.fundamentals_data = self.setup_validator.conn.execute(fundamentals_query).df()
        logger.info(f"Loaded {len(self.fundamentals_data)} fundamentals records")
        
    def merge_financial_data(self):
        """Merge financial data from different tables"""
        if self.fundamentals_data is None or self.fundamentals_data.empty:
            logger.warning("No fundamentals data to merge")
            return pd.DataFrame()
            
        # For now, just return the fundamentals data
        # In a real implementation, you might merge with other financial tables
        return self.fundamentals_data
        
    def create_financial_summary(self, row: pd.Series) -> str:
        """Create a comprehensive text summary of financial data"""
        parts = []
        
        # Basic company info
        ticker = row.get('ticker', 'Unknown')
        company_name = row.get('company_name', ticker)
        parts.append(f"Financial summary for {company_name} ({ticker})")
        
        # Financial metrics
        market_cap = row.get('market_cap', None)
        if market_cap is not None and pd.notna(market_cap):
            market_cap_str = f"${market_cap/1e6:.1f}M" if market_cap < 1e9 else f"${market_cap/1e9:.1f}B"
            parts.append(f"Market Cap: {market_cap_str}")
        
        # Key ratios
        for ratio_name, ratio_key in [
            ("Debt to Equity", "debt_to_equity"),
            ("Current Ratio", "current_ratio"),
            ("Return on Assets", "roa"),
            ("Return on Equity", "roe"),
            ("Gross Margin", "gross_margin_pct"),
            ("Net Margin", "net_margin_pct"),
            ("Revenue Growth", "revenue_growth_pct")
        ]:
            value = row.get(ratio_key)
            if value is not None and pd.notna(value):
                if ratio_key.endswith('_pct'):
                    parts.append(f"{ratio_name}: {value:.1f}%")
                else:
                    parts.append(f"{ratio_name}: {value:.2f}")
        
        # Income statement highlights
        revenue = row.get('revenue')
        if revenue is not None and pd.notna(revenue):
            revenue_str = f"${revenue/1e6:.1f}M" if revenue < 1e9 else f"${revenue/1e9:.1f}B"
            parts.append(f"Revenue: {revenue_str}")
        
        net_income = row.get('net_income')
        if net_income is not None and pd.notna(net_income):
            net_income_str = f"${net_income/1e6:.1f}M" if abs(net_income) < 1e9 else f"${net_income/1e9:.1f}B"
            parts.append(f"Net Income: {net_income_str}")
        
        # Balance sheet highlights
        total_assets = row.get('total_assets')
        if total_assets is not None and pd.notna(total_assets):
            assets_str = f"${total_assets/1e6:.1f}M" if total_assets < 1e9 else f"${total_assets/1e9:.1f}B"
            parts.append(f"Total Assets: {assets_str}")
        
        # Period information
        report_type = row.get('report_type')
        period_end = row.get('period_end')
        if report_type is not None and period_end is not None:
            parts.append(f"Report Type: {report_type}, Period End: {period_end}")
        
        # Join all parts with newlines
        return "\n".join(parts)
        
    def create_records_with_labels(self, merged_df):
        """Create embedding records with optional labels based on mode"""
        records = []
        
        for setup_id in self.setup_validator.confirmed_setup_ids:
            if setup_id not in merged_df.index:
                continue
            
            row = merged_df.loc[setup_id]
            
            # Create financial summary
            financial_summary = self.create_financial_summary(row)
            
            record = {
                'setup_id': setup_id,
                'ticker': row.get('ticker', ''),
                'financial_summary': financial_summary,
                'market_cap': float(row.get('market_cap', 0)),
                'debt_to_equity': float(row.get('debt_to_equity', 0)),
                'current_ratio': float(row.get('current_ratio', 0)),
                'roa': float(row.get('roa', 0)),
                'roe': float(row.get('roe', 0)),
                'gross_margin_pct': float(row.get('gross_margin_pct', 0)),
                'net_margin_pct': float(row.get('net_margin_pct', 0)),
                'revenue_growth_pct': float(row.get('revenue_growth_pct', 0)),
                'embedded_at': datetime.now().isoformat()
            }
            
            records.append(record)
        
        logger.info(f"Created {len(records)} fundamentals records for confirmed setups")
        return records
    
    def run_pipeline(self):
        """Execute the complete fundamentals embedding pipeline"""
        logger.info("Starting DuckDB-based Fundamentals Domain Embedding Pipeline")
        
        self.load_data()
        merged_df = self.merge_financial_data()
        records = self.create_records_with_labels(merged_df)
        
        # Enrich with labels
        enriched_records = self.enrich_with_labels(records)
        
        # Create embeddings
        final_records = self.create_embeddings(enriched_records)
        
        # Store in LanceDB (only in training mode)
        self.store_in_lancedb(final_records, "fundamentals_embeddings")
        
        # Display summary
        self.display_summary(final_records)
        
        # Close DuckDB connection
        self.cleanup()
    
    def process_setups(self, setup_ids: List[str]) -> bool:
        """Process specific setup IDs"""
        logger.info(f"Processing fundamentals embeddings for {len(setup_ids)} setups")
        
        # Override confirmed setup IDs
        self.setup_validator.confirmed_setup_ids = set(setup_ids)
        
        try:
            self.run_pipeline()
            return True
        except Exception as e:
            logger.error(f"Error processing setups: {e}")
            return False
    
    def embed_fundamentals(self, setup_ids: List[str] = None) -> bool:
        """Backward compatibility method"""
        if setup_ids:
            return self.process_setups(setup_ids)
        else:
            return self.process_setups(list(self.setup_validator.confirmed_setup_ids))
    
    def display_summary(self, records: List[Dict[str, Any]]) -> None:
        """Display pipeline summary statistics"""
        if not records:
            logger.info("No records processed")
            return
        
        logger.info("\n" + "="*50)
        logger.info("FUNDAMENTALS EMBEDDING PIPELINE SUMMARY")
        logger.info("="*50)
        
        # Basic stats
        logger.info(f"Total records processed: {len(records)}")
        logger.info(f"Unique setups: {len(set(r['setup_id'] for r in records))}")
        logger.info(f"Unique tickers: {len(set(r['ticker'] for r in records))}")
        
        # Label stats
        labeled_records = [r for r in records if r.get('has_performance_labels', False)]
        logger.info(f"Records with performance labels: {len(labeled_records)}")
        
        logger.info("="*50)

def main():
    """Main function to run the fundamentals embedding pipeline"""
    embedder = FundamentalsEmbedderDuckDB()
    embedder.run_pipeline()

if __name__ == "__main__":
    main() 