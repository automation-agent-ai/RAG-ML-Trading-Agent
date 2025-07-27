# 🚀 Production Pipeline Setup Guide

## 📋 Prerequisites

### **1. Python Environment**
```bash
# Recommended: Python 3.9+
python --version
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Environment Variables**
Create a `.env` file in the root directory:

```bash
# .env file content:
OPENAI_API_KEY=sk-your-openai-api-key-here
DEFAULT_LLM_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
```

**Or set environment variables directly:**
```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY="sk-your-api-key-here"
```

## 🎯 Quick Start

### **1. Verify Setup**
```bash
python tools/verify_setup.py
```

### **2. Run Complete Pipeline**
```bash
# Training mode
python run_complete_ml_pipeline.py --mode training

# Prediction mode  
python run_complete_ml_pipeline.py --mode prediction --setup-ids YOUR_SETUP_ID
```

### **3. Quick Demo**
```bash
python complete_workflow.py --demo
```

## 📂 Directory Structure

```
production_pipeline/
├── run_complete_ml_pipeline.py    # Main pipeline orchestrator
├── agents/                         # LLM feature extraction agents
├── embeddings/                     # Vector embedding creation
├── core/                          # Feature merging & ML utilities
├── tools/                         # CLI utilities & validation
├── data/                          # Databases & storage
└── docs/                          # Guides & documentation
```

## 🔍 Troubleshooting

### **Common Issues:**

1. **Missing OpenAI API Key:**
   ```
   ERROR: OPENAI_API_KEY environment variable not set
   ```
   **Solution:** Set the environment variable or create `.env` file

2. **Missing Dependencies:**
   ```
   ImportError: No module named 'lancedb'
   ```
   **Solution:** `pip install -r requirements.txt`

3. **Database Not Found:**
   ```
   ERROR: Table 'setups' does not exist
   ```
   **Solution:** Ensure `data/sentiment_system.duckdb` exists with your data

## 📚 Documentation

- **[Complete Pipeline Guide](COMPLETE_PIPELINE_GUIDE.md)** - Comprehensive workflow documentation
- **[Quick Reference](QUICK_REFERENCE.md)** - Common commands and usage
- **[Data Leakage Fix Guide](DATA_LEAKAGE_FIX_GUIDE.md)** - Critical implementation notes

---

**🎉 Your standalone production pipeline is ready!** 