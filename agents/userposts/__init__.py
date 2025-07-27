"""
UserPosts Agent Package
=======================

Enhanced UserPosts Agent (Complete Retrieval Pattern) that removes:
- Top-K retrieval limiting
- Cross-encoder reranking
- Vector search limitations

Instead uses:
- Complete record retrieval per setup_id
- All available posts for feature extraction
- Direct LLM processing with chunking for large datasets

Similar to enhanced_news_agent_duckdb.py but for UserPosts domain.
"""

from .enhanced_userposts_agent_complete import EnhancedUserPostsAgentComplete

__all__ = ['EnhancedUserPostsAgentComplete'] 