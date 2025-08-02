# GitHub Migration Plan - Enhanced RAG Pipeline

## 📋 Migration Overview

This document outlines the complete migration plan to move the Enhanced RAG Pipeline to GitHub while excluding large data files and ensuring all 17 workflow steps can be executed on the Hetzner server.

## 🗂️ Files to Include in GitHub Repository

### Core Pipeline Files (Essential for 17 Steps)
```
✅ INCLUDE THESE FILES:

Root Level Scripts:
- run_complete_ml_pipeline.py           # Complete pipeline orchestration
- run_enhanced_ml_pipeline.py           # Enhanced ML pipeline
- train_3stage_ml_pipeline.py           # 3-stage training
- predict_3stage_ml_pipeline.py         # 3-stage predictions
- create_prediction_list.py             # Step 1: Select setups
- create_training_list.py               # Step 1: Create training list
- preserve_restore_embeddings.py        # Steps 2,3,16,17: Data management
- extract_all_ml_features_from_duckdb.py # Step 7: Extract ML features
- extract_financial_features_from_duckdb.py # Step 7: Financial features
- add_labels_to_features.py             # Step 8: Add labels
- balance_ml_datasets.py                # Step 9: Balance datasets
- make_agent_predictions.py             # Step 10: Agent predictions
- generate_results_table.py             # Step 13: Results table
- evaluate_predictions.py               # Step 15: Evaluate predictions
- visualize_ensemble_results.py         # Step 14: Visualizations
- simple_visualize.py                   # Alternative visualization

Agent System:
- agents/                               # All agent directories and files
  - fundamentals/enhanced_fundamentals_agent_duckdb.py
  - news/enhanced_news_agent_duckdb.py
  - analyst_recommendations/enhanced_analyst_recommendations_agent_duckdb.py
  - userposts/enhanced_userposts_agent_complete.py
  - news/news_categories.py

Embeddings:
- embeddings/                           # All embedding scripts
  - base_embedder.py
  - embed_*.py (all embedding scripts)

Core Modules:
- core/                                 # Core functionality
  - financial_features.py
  - label_converter.py
  - ml_feature_merger.py

Frontend/Backend:
- backend.py                           # Main backend
- backend_fast.py                      # Fast backend (mock agents)
- index.html                           # Frontend HTML
- static/                              # CSS, JS files
  - app.js
  - app_new.js
  - style.css
- start_dashboard.py                   # Dashboard startup script

Utilities:
- tools/                               # All utility scripts
- tests/                               # All test files
- financial_preprocessor.py           # Enhanced financial preprocessing
- threshold_manager.py                 # Threshold management
- download_models.py                   # Model caching
- use_cached_model.py                  # Cached model usage

Configuration:
- requirements.txt                     # Python dependencies
- README.md                           # Project documentation
- .gitignore                          # Git ignore rules
- data/label_thresholds.json          # Threshold configuration
- data/all_setups.txt                 # Setup lists
- data/fundamentals_setups.txt
- data/training_setups.txt
- data/prediction_setups.txt
- data/test_setups.txt

Documentation:
- docs/                               # All documentation
  - OPTIMIZED_WORKFLOW.md             # 17-step workflow guide
  - COMPLETE_PIPELINE_GUIDE.md
  - ENHANCED_RAG_GUIDE.md
  - (all other .md files)
```

## 🚫 Files to Exclude from GitHub

### Large Data Files (Upload Separately)
```
❌ EXCLUDE THESE FILES (Upload to Hetzner separately):

Database Files:
- data/*.duckdb                       # All DuckDB files (~250MB each)
- data/*.duckdb.wal                   # WAL files
- data/*.duckdb.tmp/                  # Temp directories

Vector Database:
- data/storage_lancedb_store/         # LanceDB vector store
- data/lancedb_store/                 # Alternative LanceDB location
- lancedb_store/                      # Any LanceDB directories
- storage/                            # Storage directories

ML Models:
- models/sentence_transformers/       # Cached transformer models (~100MB)
- models/ensemble/                    # Trained ensemble models
- models/financial/                   # Trained financial models
- models/text/                        # Trained text models
- models/3stage*/                     # 3-stage model directories
- models_clean/                       # Clean model directories

Generated Data:
- data/ml_features/                   # Generated ML features
- data/ml_features_clean/             # Cleaned ML features
- data/ml_pipeline_output/            # Pipeline outputs
- data/predictions/                   # Prediction results
- data/predictions_corrected/         # Corrected predictions
- data/leakage_analysis/              # Analysis outputs
- evaluation_results/                 # Evaluation outputs
- visualizations/                     # Generated visualizations
- ml/                                 # ML analysis outputs
- test_output/                        # Test outputs

Temporary Files:
- *.pkl                              # Pickle files
- *.pickle                           # Pickle files
- *.tmp                              # Temporary files
- *.log                              # Log files
- __pycache__/                       # Python cache
```

## 🧹 Git History Cleanup

### Large Files to Remove from Git History
```bash
# Files currently tracked in git that need removal:
- models/sentence_transformers/ (large model files)
- data/*.duckdb (if committed before .gitignore)
- Any large .pkl files in git history
- Large model files in models/ directories
```

### Cleanup Commands (Run on Local Machine)
```bash
# 1. Remove large files from git history
git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch data/*.duckdb* || true' \
--prune-empty --tag-name-filter cat -- --all

git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch -r models/sentence_transformers/ || true' \
--prune-empty --tag-name-filter cat -- --all

git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch -r data/storage_lancedb_store/ || true' \
--prune-empty --tag-name-filter cat -- --all

git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch *.pkl || true' \
--prune-empty --tag-name-filter cat -- --all

# 2. Clean up refs
git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 3. Force push to clean remote
git push origin --force --all
```

## 📤 Data Upload Plan for Hetzner

### Required Data Files for Hetzner Server
```
Upload these large files separately to Hetzner:

1. DuckDB Database:
   - data/sentiment_system.duckdb (~250MB)
   - data/sentiment_system.duckdb.wal (if exists)

2. LanceDB Vector Store:
   - data/storage_lancedb_store/ (entire directory)

3. Pre-trained Models (Optional, can be downloaded):
   - models/sentence_transformers/ (for offline operation)

Upload Methods:
- scp/sftp for large files
- rsync for directory structures
- Direct upload via hosting panel
- Git LFS (alternative, but more complex)
```

## 🔄 Migration Steps

### Step 1: Prepare Local Repository
```bash
# 1. Create new clean branch
git checkout -b github-migration

# 2. Ensure .gitignore is properly configured
# (Already includes necessary exclusions)

# 3. Clean git history of large files
# (Run cleanup commands above)

# 4. Commit final clean state
git add .
git commit -m "Clean repository for GitHub migration"
```

### Step 2: Create GitHub Repository
```bash
# 1. Create new repository on GitHub
# Repository name: enhanced-rag-pipeline

# 2. Add GitHub remote
git remote add github https://github.com/YOUR_USERNAME/enhanced-rag-pipeline.git

# 3. Push to GitHub
git push github github-migration:main
```

### Step 3: Upload Data to Hetzner
```bash
# Upload DuckDB files
scp data/sentiment_system.duckdb user@hetzner-server:/home/user/production_pipeline/data/

# Upload LanceDB store
rsync -avz data/storage_lancedb_store/ user@hetzner-server:/home/user/production_pipeline/data/storage_lancedb_store/
```

## ✅ Verification Checklist

### Essential Files Present
- [ ] All 17 workflow steps can be executed
- [ ] All agent files included
- [ ] Core ML pipeline scripts present
- [ ] Frontend/backend files included
- [ ] Documentation complete
- [ ] Configuration files present

### Large Files Excluded
- [ ] No .duckdb files in repository
- [ ] No model files > 10MB
- [ ] No vector database files
- [ ] Git history cleaned
- [ ] Repository size < 100MB

### Deployment Ready
- [ ] requirements.txt complete
- [ ] Setup instructions provided
- [ ] Environment configuration documented
- [ ] Data upload plan specified

## 🎯 Expected Repository Structure on GitHub
```
enhanced-rag-pipeline/
├── agents/                     # AI agents
├── backend.py                  # Main backend
├── backend_fast.py            # Fast backend
├── core/                      # Core modules
├── docs/                      # Documentation
├── embeddings/                # Embedding scripts
├── index.html                 # Frontend
├── static/                    # CSS/JS
├── tests/                     # Test suite
├── tools/                     # Utilities
├── requirements.txt           # Dependencies
├── README.md                  # Main documentation
├── DEPLOYMENT_GUIDE.md        # Deployment instructions
├── GITHUB_MIGRATION_PLAN.md   # This file
└── .gitignore                 # Git exclusions
```

Repository size should be < 100MB after cleanup.