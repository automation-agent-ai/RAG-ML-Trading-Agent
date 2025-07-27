# 🚀 Production Pipeline - Complete ML System

## 📁 Directory Structure

```
production_pipeline/
├── 🎯 run_complete_ml_pipeline.py          # MAIN ORCHESTRATOR
├── 🔧 export_ml_features.py               # Enhanced SQL with historical features
├── 🔧 merge_financial_features.py         # Financial preprocessing
├── 🎯 run_ml_training.py                  # ML training
├── 📦 requirements.txt                    # Dependencies
├── 📖 README.md                           # This documentation
│
├── 📊 data/                               # ALL DATA HERE
│   ├── sentiment_system.duckdb           # Main database (225MB)
│   ├── lancedb_store/                    # Vector embeddings
│   └── storage_lancedb_store/            # Additional vector storage
│
├── 🤖 agents/                             # LLM AGENTS
│   ├── fundamentals/enhanced_fundamentals_agent_duckdb.py
│   ├── news/enhanced_news_agent_duckdb.py
│   ├── userposts/enhanced_userposts_agent_complete.py
│   └── analyst_recommendations/enhanced_analyst_recommendations_agent_duckdb.py
│
├── 🎰 embeddings/                         # VECTOR GENERATION
│   ├── embed_fundamentals_duckdb.py
│   ├── embed_news_duckdb.py
│   ├── embed_userposts_duckdb.py
│   └── embed_analyst_recommendations_duckdb.py
│
├── 🔧 core/                               # FEATURE MERGING
│   ├── ml_feature_merger.py
│   └── financial_features.py
│
└── 🛠️ tools/                              # UTILITIES
    ├── setup_validator_duckdb.py
    └── cli_extract_*.py
```

## 🚀 Quick Start

### **New Data Processing Workflow**

```bash
# Navigate to production pipeline
cd production_pipeline/

# Option 1: Complete Pipeline (Recommended)
python run_complete_ml_pipeline.py --mode training

# Option 2: Step-by-Step Control
# 1. Create embeddings for new data
python embeddings/embed_news_duckdb.py --db-path data/sentiment_system.duckdb --lancedb-dir data/lancedb_store

# 2. Run complete pipeline
python run_complete_ml_pipeline.py --mode training
```

### **Programmatic Usage**

```python
from run_complete_ml_pipeline import CompletePipeline

# Initialize pipeline
pipeline = CompletePipeline()

# Run training with setup IDs
setup_ids = ['SETUP_001', 'SETUP_002', 'SETUP_003']
results = pipeline.run_complete_pipeline(setup_ids, mode='training')

# Run prediction
results = pipeline.run_complete_pipeline(setup_ids, mode='prediction')
```

### **Enhanced Features Usage**

```python
from export_ml_features import export_training_features, export_prediction_features
from merge_financial_features import merge_financial_features

# Extract enhanced features with historical analysis
export_training_features(
    db_path="data/sentiment_system.duckdb", 
    setup_ids=setup_ids,
    output_dir="output/"
)

# Merge financial features
merge_financial_features(
    db_path="data/sentiment_system.duckdb",
    setup_ids=setup_ids,
    mode='training'
)
```

## 🎯 Enhanced Historical Features

This production pipeline includes comprehensive historical financial analysis:

### **Multi-Year Growth Features**
- YoY growth rates for revenue, operating income, net income, EBITDA (1-3 years)
- Growth consistency and volatility metrics
- Growth acceleration/deceleration indicators

### **Rolling Statistics**
- 3-year moving averages for key financial metrics
- Historical margin analysis (operating margin, net margin)
- Leverage evolution tracking

### **Trend Indicators**
- Consecutive growth years tracking
- Financial health evolution
- Trend strength scoring

## 🔄 Data Flow

```
📊 NEW DATA (DuckDB) 
    ↓
🎰 EMBEDDINGS (LanceDB)
    ↓  
🤖 AGENTS (Feature Extraction)
    ↓
🔧 FEATURE MERGING (Enhanced Historical)
    ↓
🎯 ML TRAINING
```

## 📋 System Requirements

- Python 3.8+
- DuckDB with financial data
- LanceDB for vector storage
- Dependencies in requirements.txt

## 🎉 Benefits

1. **🏠 Self-Contained**: Everything in one directory
2. **📊 Enhanced Features**: Historical financial analysis included
3. **🚀 Simple Workflow**: One command handles everything
4. **💾 Co-located Data**: DuckDB and LanceDB together
5. **🔧 Production Ready**: Clean, organized, tested structure

## 🚨 Usage Examples

### For New Financial Data
```bash
cd production_pipeline/
python run_complete_ml_pipeline.py --mode training --setup-ids SETUP_001 SETUP_002
```

### For Prediction
```bash
cd production_pipeline/
python run_complete_ml_pipeline.py --mode prediction --setup-ids SETUP_003 SETUP_004
```

### Custom Database Path
```bash
cd production_pipeline/
python run_complete_ml_pipeline.py --mode training --db-path data/sentiment_system.duckdb --lancedb-dir data/lancedb_store
```

This production pipeline combines the best of both worlds: your proven original pipeline architecture with enhanced historical financial features, all organized in a clean, self-contained structure! 🎯 