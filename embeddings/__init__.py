"""
Embeddings Package
==================

Embedding pipelines for the RAG system:
- embed_fundamentals_duckdb: Fundamentals data embedding
- embed_userposts_duckdb: User posts embedding
- embed_news_duckdb: News and RNS announcements embedding

All embeddings are stored in LanceDB vector database for efficient retrieval.
"""

# Import available embedding modules
try:
    from .embed_fundamentals_duckdb import FundamentalsEmbedderDuckDB
except ImportError:
    FundamentalsEmbedderDuckDB = None

try:
    from .embed_userposts_duckdb import UserPostsEmbedderDuckDB
except ImportError:
    UserPostsEmbedderDuckDB = None

try:
    from .embed_news_duckdb import NewsEmbeddingPipelineDuckDB
except ImportError:
    NewsEmbeddingPipelineDuckDB = None

__all__ = [
    'FundamentalsEmbedderDuckDB',
    'UserPostsEmbedderDuckDB', 
    'NewsEmbeddingPipelineDuckDB'
] 