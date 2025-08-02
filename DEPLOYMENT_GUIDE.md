# 🚀 Hetzner Server Deployment Guide - Enhanced RAG Pipeline

## 📋 Server Specifications

**Recommended Hetzner CX41 Configuration:**
- CPU: 4 vCPUs (AMD)
- RAM: 16 GB
- Storage: 160 GB SSD
- OS: Ubuntu 22.04 LTS
- Network: 1 Gbit/s

## 🏗️ Complete Deployment Instructions

### Step 1: Initial Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y git curl wget htop nano vim unzip build-essential

# Install Python 3.11 and pip
sudo apt install -y python3.11 python3.11-dev python3.11-venv python3-pip

# Create application user (optional, for security)
sudo useradd -m -s /bin/bash ragpipeline
sudo usermod -aG sudo ragpipeline
```

### Step 2: Clone Repository from GitHub

```bash
# Navigate to home directory
cd /home/ragpipeline  # or your user directory

# Clone the repository
git clone https://github.com/YOUR_USERNAME/enhanced-rag-pipeline.git
cd enhanced-rag-pipeline

# Set proper permissions
chmod +x *.py
```

### Step 3: Setup Conda Environment

```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Initialize conda
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc

# Create conda environment named 'sts'
conda create -n sts python=3.11 -y
conda activate sts

# Verify conda environment
echo "Conda environment created: $(conda info --envs | grep sts)"
```

### Step 4: Install Python Dependencies

```bash
# Ensure sts environment is active
conda activate sts

# Install core dependencies
pip install --upgrade pip

# Install from requirements.txt
pip install -r requirements.txt

# Install additional ML dependencies
pip install lightgbm xgboost lancedb openai

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import sentence_transformers; print('SentenceTransformers: OK')"
python -c "import duckdb; print(f'DuckDB: {duckdb.__version__}')"
```

### Step 5: Upload Data Files

#### Option A: Upload via SCP (from local machine)
```bash
# Upload DuckDB database (run from local machine)
scp data/sentiment_system.duckdb user@YOUR_HETZNER_IP:/home/ragpipeline/enhanced-rag-pipeline/data/

# Upload LanceDB vector store
rsync -avz data/storage_lancedb_store/ user@YOUR_HETZNER_IP:/home/ragpipeline/enhanced-rag-pipeline/data/storage_lancedb_store/

# Upload any preserved data files
scp data/*.pkl user@YOUR_HETZNER_IP:/home/ragpipeline/enhanced-rag-pipeline/data/
```

#### Option B: Upload via Web Panel/Direct Transfer
```bash
# Create data directories
mkdir -p data/storage_lancedb_store
mkdir -p data/ml_features
mkdir -p data/predictions
mkdir -p models

# Upload files through Hetzner console or file manager
# Place sentiment_system.duckdb in data/ directory
# Place LanceDB store in data/storage_lancedb_store/
```

### Step 6: Download and Cache Models

```bash
# Activate conda environment
conda activate sts

# Download sentence transformer models for offline use
python download_models.py

# Patch pipeline to use cached models
python use_cached_model.py

# Verify model caching
ls -la models/sentence_transformers/
```

### Step 7: Configure Environment Variables

```bash
# Create environment file
cat > .env << EOF
# OpenAI API Configuration (for agent predictions)
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
DUCKDB_PATH=data/sentiment_system.duckdb
LANCEDB_PATH=data/storage_lancedb_store

# Server Configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info

# Model Configuration
MODEL_CACHE_DIR=models/sentence_transformers
USE_CACHED_MODELS=true
EOF

# Set proper permissions
chmod 600 .env
```

### Step 8: Verify Database and Setup

```bash
# Activate environment
conda activate sts

# Test database connection
python -c "
import duckdb
conn = duckdb.connect('data/sentiment_system.duckdb')
tables = conn.execute('SHOW TABLES').fetchall()
print(f'Found {len(tables)} tables in database')
for table in tables[:5]:
    print(f'- {table[0]}')
conn.close()
"

# Check setup files
ls -la data/*setups.txt
ls -la data/label_thresholds.json

# Verify LanceDB store
ls -la data/storage_lancedb_store/
```

### Step 9: Test Core Pipeline Components

```bash
# Activate environment
conda activate sts

# Test 1: Basic functionality
python -c "
from agents.fundamentals.enhanced_fundamentals_agent_duckdb import EnhancedFundamentalsAgent
print('✅ Fundamentals agent import successful')
"

# Test 2: Database connectivity
python check_tables.py

# Test 3: Embedding functionality
python -c "
from embeddings.base_embedder import BaseEmbedder
print('✅ Embeddings module import successful')
"

# Test 4: ML pipeline components
python -c "
from core.financial_features import FinancialFeaturesExtractor
print('✅ Financial features extractor import successful')
"
```

### Step 10: Configure Firewall and Network

```bash
# Configure UFW firewall
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8000/tcp  # API server
sudo ufw allow 80/tcp    # HTTP (optional)
sudo ufw allow 443/tcp   # HTTPS (optional)
sudo ufw --force enable

# Check firewall status
sudo ufw status verbose
```

### Step 11: Test Frontend/Backend

```bash
# Activate environment
conda activate sts

# Test fast backend (mock agents)
python start_dashboard.py &

# Wait a few seconds then test
sleep 5
curl http://localhost:8000/api/health

# Stop test server
pkill -f start_dashboard.py

# Test full backend (real agents) - requires OpenAI key
# python backend.py
```

### Step 12: Setup as System Service (Production)

```bash
# Create systemd service file
sudo tee /etc/systemd/system/rag-pipeline.service > /dev/null <<EOF
[Unit]
Description=Enhanced RAG Pipeline API Server
After=network.target

[Service]
Type=simple
User=ragpipeline
WorkingDirectory=/home/ragpipeline/enhanced-rag-pipeline
Environment=PATH=/home/ragpipeline/miniconda3/envs/sts/bin
ExecStart=/home/ragpipeline/miniconda3/envs/sts/bin/python start_dashboard.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable rag-pipeline
sudo systemctl start rag-pipeline

# Check service status
sudo systemctl status rag-pipeline
```

## 🧪 Testing the Complete 17-Step Pipeline

### Quick Pipeline Test
```bash
# Activate environment
conda activate sts

# Test with small dataset (Steps 1-17)
python run_complete_ml_pipeline.py --mode all --count 10

# Monitor progress
tail -f *.log
```

### Manual Step-by-Step Testing
```bash
# Activate environment
conda activate sts

# Step 1: Create prediction list
python create_prediction_list.py --count 10 --output data/test_prediction_setups.txt

# Step 2-3: Preserve and remove data
python preserve_restore_embeddings.py preserve --prediction-setup-file data/test_prediction_setups.txt

# Continue with remaining steps...
# (See OPTIMIZED_WORKFLOW.md for complete step sequence)
```

## 🔍 Monitoring and Logs

### Log Files Location
```bash
# Application logs
tail -f /home/ragpipeline/enhanced-rag-pipeline/*.log

# System service logs
sudo journalctl -u rag-pipeline -f

# System resource monitoring
htop
df -h
```

### Health Checks
```bash
# API health check
curl http://localhost:8000/api/health

# Database check
python -c "import duckdb; conn = duckdb.connect('data/sentiment_system.duckdb'); print('DB OK')"

# Model cache check
ls -la models/sentence_transformers/
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. Memory Issues
```bash
# Check memory usage
free -h

# Increase swap if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 2. Permission Issues
```bash
# Fix file permissions
sudo chown -R ragpipeline:ragpipeline /home/ragpipeline/enhanced-rag-pipeline
chmod -R 755 /home/ragpipeline/enhanced-rag-pipeline
chmod 600 .env
```

#### 3. Conda Environment Issues
```bash
# Recreate environment
conda remove -n sts --all -y
conda create -n sts python=3.11 -y
conda activate sts
pip install -r requirements.txt
```

#### 4. Model Download Issues
```bash
# Manual model download
mkdir -p models/sentence_transformers
cd models/sentence_transformers
wget -r --no-parent https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/
```

#### 5. Database Connection Issues
```bash
# Check DuckDB file
file data/sentiment_system.duckdb
ls -la data/sentiment_system.duckdb

# Test connection
python -c "import duckdb; print(duckdb.connect('data/sentiment_system.duckdb').execute('SELECT 1').fetchone())"
```

## 📊 Performance Optimization

### For CX41 Server (16GB RAM)
```bash
# Configure Python memory settings
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# DuckDB memory settings (in your Python scripts)
# conn.execute("SET memory_limit='8GB'")
# conn.execute("SET threads=4")
```

### Model Optimization
```bash
# Use quantized models for better performance
pip install torch-audio torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# Enable model caching
echo "USE_CACHED_MODELS=true" >> .env
```

## 🔒 Security Recommendations

### Basic Security Setup
```bash
# Create non-root user (if not done)
sudo useradd -m -G sudo ragpipeline

# SSH key authentication (recommended)
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Disable root login
sudo sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo systemctl restart ssh

# Install fail2ban
sudo apt install -y fail2ban
```

### API Security
```bash
# Add API key authentication (optional)
echo "API_KEY=your_secure_api_key_here" >> .env

# Use reverse proxy (optional - nginx)
sudo apt install -y nginx
# Configure nginx proxy to port 8000
```

## 📋 Deployment Checklist

### ✅ Pre-Deployment
- [ ] Server provisioned (CX41 or better)
- [ ] Ubuntu 22.04 installed
- [ ] SSH access configured
- [ ] Domain name configured (optional)

### ✅ Environment Setup
- [ ] Repository cloned from GitHub
- [ ] Conda environment created (sts)
- [ ] Dependencies installed
- [ ] Data files uploaded
- [ ] Models cached
- [ ] Environment variables configured

### ✅ Testing
- [ ] Database connectivity verified
- [ ] API health check passes
- [ ] Frontend accessible
- [ ] Sample pipeline execution successful
- [ ] All 17 steps can be executed

### ✅ Production Setup
- [ ] System service configured
- [ ] Firewall configured
- [ ] Monitoring setup
- [ ] Backups configured
- [ ] Security hardened

## 🎯 Expected Performance

### On Hetzner CX41:
- **Pipeline Execution**: 5-10 minutes for 50 setups
- **API Response Time**: < 2 seconds for predictions
- **Memory Usage**: 8-12GB during ML training
- **Storage Usage**: ~20GB including models and data

### Resource Monitoring Commands:
```bash
# Monitor during pipeline execution
watch -n 1 'free -h && df -h | head -5'
top -p $(pgrep -f python)
```

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review log files for error messages
3. Ensure all dependencies are properly installed
4. Verify data files are uploaded correctly
5. Test individual components before running full pipeline

The pipeline should run out-of-the-box on a fresh Ubuntu 22.04 Hetzner CX41 server following these instructions.