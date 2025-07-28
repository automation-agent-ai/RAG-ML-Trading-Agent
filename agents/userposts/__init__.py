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
# Force offline mode for model loading
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_CACHE'] = os.path.join('models', 'cache')
os.environ['HF_HOME'] = os.path.join('models', 'hub')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join('models', 'sentence_transformers')


__all__ = ['EnhancedUserPostsAgentComplete'] 