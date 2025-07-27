"""
News Agent Package
==================

Enhanced News Agent with DuckDB Feature Storage and Grouped Processing that:
- Category classification (dictionary + fuzzy matching + LLM fallback)
- Group-level feature extraction (5 major groups)
- Complete record processing (not top-K retrieval)
- DuckDB storage integration

Post Phase 1 optimization: Removed earnings-related features (moved to fundamentals agent)
for better domain separation and feature organization.
"""

from .enhanced_news_agent_duckdb import EnhancedNewsAgentDuckDB

__all__ = ['EnhancedNewsAgentDuckDB'] 