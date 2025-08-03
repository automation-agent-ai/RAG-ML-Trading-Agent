# Enhanced RAG Pipeline Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV USE_CACHED_MODELS=true
ENV PORT=8001
ENV HOST=0.0.0.0

# Create directories for models and data
RUN mkdir -p /app/models /app/data

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install lightgbm xgboost lancedb openai

# Copy the application code
COPY . .

# Download and cache models during build
RUN python download_models.py && \
    python use_cached_model.py

# Create production environment file
RUN echo "# Production Environment Configuration" > .env && \
    echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env && \
    echo "DEFAULT_LLM_MODEL=gpt-4o-mini" >> .env && \
    echo "LOG_LEVEL=INFO" >> .env && \
    echo "DUCKDB_PATH=data/sentiment_system.duckdb" >> .env && \
    echo "LANCEDB_PATH=data/storage_lancedb_store" >> .env && \
    echo "HOST=0.0.0.0" >> .env && \
    echo "PORT=8001" >> .env && \
    echo "MODEL_CACHE_DIR=models/sentence_transformers" >> .env && \
    echo "USE_CACHED_MODELS=true" >> .env && \
    echo "DUCKDB_MEMORY_LIMIT=8GB" >> .env && \
    echo "DUCKDB_THREADS=4" >> .env && \
    echo "ENVIRONMENT=production" >> .env && \
    echo "DEBUG=false" >> .env

# Expose the application port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/api/health || exit 1

# Note: Data and models will be mounted at runtime
# Start the Enhanced RAG Pipeline dashboard
CMD ["python", "start_dashboard.py", "--no-browser"]