"""
Fundamentals Agent Package
=========================

Enhanced Fundamentals Agent with DuckDB integration that handles both:
1. Structured fundamental metrics (ROE, D/E, margins, etc.) - no LLM needed
2. LLM-extracted features moved from news agent:
   - Financial Results (7 features): earnings, profit warnings, etc.
   - Corporate Actions (2 features): capital_raise_present, synthetic_summary

This follows the smart feature split strategy where quarterly/annual fundamental 
data is processed by the fundamentals agent instead of the news agent.
"""

from .enhanced_fundamentals_agent_duckdb import EnhancedFundamentalsAgentDuckDB

__all__ = ['EnhancedFundamentalsAgentDuckDB'] 