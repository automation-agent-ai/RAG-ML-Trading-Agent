# 🤖 Cursor AI Deployment Prompt for Hetzner Server

## Copy and Paste this Prompt into Cursor on your Hetzner Server

```
I need to deploy the Enhanced RAG Pipeline project on this Ubuntu 22.04 Hetzner CX41 server. This is a comprehensive ML pipeline with 17 workflow steps for stock trading analysis that includes:

🎯 PROJECT OVERVIEW:
- 17-step ML pipeline for stock outperformance prediction
- 4 AI agents (Fundamentals, News, Analyst, Community sentiment)
- 3-stage ML training (Text → Financial → Ensemble)
- FastAPI backend with WebSocket support
- Interactive dashboard frontend
- DuckDB database with LanceDB vector store
- SentenceTransformers for embeddings
- OpenAI GPT-4o-mini for agent predictions

📁 EXPECTED PROJECT STRUCTURE:
The repository should contain these key components:
- agents/ (4 domain-specific prediction agents)
- backend.py & backend_fast.py (API servers)
- core/ (financial features, label conversion, ML merger)
- embeddings/ (embedding generation scripts) 
- docs/ (including OPTIMIZED_WORKFLOW.md with 17 steps)
- tools/ (utility scripts)
- static/ & index.html (frontend dashboard)
- requirements.txt (Python dependencies)
- All ML pipeline scripts (train_*, predict_*, extract_*, etc.)

🚀 DEPLOYMENT TASKS:

1. **INITIAL SETUP**:
   - Update Ubuntu system packages
   - Install Python 3.11, git, build tools
   - Create application user 'ragpipeline' 
   - Clone repository from GitHub: https://github.com/USERNAME/enhanced-rag-pipeline

2. **CONDA ENVIRONMENT**:
   - Download and install Miniconda3
   - Create conda environment named 'sts' with Python 3.11
   - Install all dependencies from requirements.txt
   - Add lightgbm, xgboost, lancedb, openai packages

3. **MODEL CACHING**:
   - Run download_models.py to cache sentence-transformers models
   - Run use_cached_model.py to configure offline model usage
   - Verify models cached in models/sentence_transformers/

4. **DATA SETUP**:
   - Create data/ directories for DuckDB and LanceDB files
   - Set up placeholder files until data is uploaded
   - Create .env file with configuration variables
   - Ensure proper file permissions

5. **TESTING**:
   - Test database connectivity with DuckDB
   - Verify agent imports work correctly
   - Test API health endpoints
   - Run start_dashboard.py to test frontend

6. **PRODUCTION SETUP**:
   - Configure UFW firewall (ports 22, 8000)
   - Create systemd service for auto-startup
   - Setup monitoring and logging
   - Configure proper security permissions

🔧 REQUIREMENTS:
- Use conda environment named 'sts' for all operations
- Install from requirements.txt plus additional ML packages
- Cache sentence-transformer models for offline operation
- Configure for both fast backend (mock) and full backend (real agents)
- Ensure all 17 pipeline steps can execute (see docs/OPTIMIZED_WORKFLOW.md)

💾 DATA FILES (Upload Separately):
- data/sentiment_system.duckdb (~250MB)
- data/storage_lancedb_store/ (vector database)
- Any .pkl preserved data files

🎯 SUCCESS CRITERIA:
- Repository cloned and dependencies installed
- Conda environment 'sts' created and working
- Models cached for offline operation
- API server starts successfully on port 8000
- Frontend dashboard accessible
- All imports work without errors
- Ready for large data file upload

⚡ OPTIMIZATION FOR HETZNER CX41:
- Configure for 16GB RAM usage
- Set threading for 4 CPU cores
- Optimize DuckDB memory settings
- Use efficient model caching

Please help me deploy this step by step, ensuring each component works before proceeding to the next. Start with system setup and work through to a fully functional deployment.
```

## Alternative Short Prompt (if the above is too long):

```
Deploy Enhanced RAG Pipeline on Ubuntu 22.04 Hetzner server:

1. Setup: Update system, install Python 3.11, git, create 'ragpipeline' user
2. Clone: git clone enhanced-rag-pipeline repository 
3. Environment: Install Miniconda, create 'sts' conda env, install requirements.txt + lightgbm/xgboost/lancedb/openai
4. Models: Run download_models.py and use_cached_model.py for offline operation
5. Config: Create .env file, setup data/ directories, configure permissions
6. Test: Verify imports, test start_dashboard.py on port 8000
7. Production: Setup systemd service, configure firewall (ports 22,8000)

This is a 17-step ML pipeline with 4 AI agents, 3-stage training, FastAPI backend, and dashboard frontend. Optimize for 16GB RAM/4 CPU. Need to upload DuckDB/LanceDB files separately (~250MB).
```

## What to Expect After Running This Prompt:

The AI assistant should help you:

1. **Execute system setup commands** to prepare Ubuntu server
2. **Install and configure conda** with the 'sts' environment  
3. **Clone the repository** and install all dependencies
4. **Download and cache models** for offline operation
5. **Create configuration files** and directory structure
6. **Test all components** to ensure they work
7. **Setup production services** for automatic startup
8. **Provide troubleshooting help** if issues arise

## Commands the AI Should Run:

### System Setup:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget htop build-essential python3.11 python3.11-dev python3.11-venv
```

### Conda Installation:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc
conda create -n sts python=3.11 -y
conda activate sts
```

### Dependencies:
```bash
pip install -r requirements.txt
pip install lightgbm xgboost lancedb openai
```

### Model Caching:
```bash
python download_models.py
python use_cached_model.py
```

### Testing:
```bash
python start_dashboard.py
curl http://localhost:8000/api/health
```

## 📋 Verification Checklist:

After deployment, verify:
- [ ] Conda environment 'sts' exists and works
- [ ] All Python imports succeed
- [ ] Models cached in models/sentence_transformers/
- [ ] API server starts on port 8000
- [ ] Frontend loads in browser
- [ ] Database connection ready (placeholder until data upload)
- [ ] All 17 pipeline steps can be executed
- [ ] System service configured for production
- [ ] Firewall configured properly

## 🚨 Common Issues to Watch For:

1. **Memory Issues**: CX41 has 16GB RAM - configure appropriately
2. **Permission Issues**: Ensure proper file ownership and permissions
3. **Model Download Issues**: May need manual download if automated fails
4. **Conda Issues**: Environment activation problems
5. **Port Conflicts**: Ensure port 8000 is available

The deployment should result in a fully functional Enhanced RAG Pipeline ready for data upload and production use.