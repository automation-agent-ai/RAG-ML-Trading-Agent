# RAG Pipeline Frontend Dashboard

A comprehensive web interface for the RAG Pipeline ML analysis system, providing real-time predictions, data visualization, and AI-powered explanations.

## ✨ Features

### 🎯 **Core Functionality**
- **Setup Analysis**: Detailed analysis of stock setups with fundamentals, news, and user posts data
- **Real-time Predictions**: ML ensemble model predictions with confidence scores
- **Knowledge Explainer**: AI-powered explanations of prediction reasoning
- **Setup Scanner**: Discover high-probability investment opportunities
- **Perfect Setup Guide**: Optimal feature ranges for successful setups

### 📊 **Data Ribbons**
- **Fundamentals Data**: Financial ratios, margins, revenue metrics with filtering
- **RNS News Data**: Corporate announcements with date-based filtering  
- **User Posts Data**: Community sentiment with sentiment-based filtering

### 📈 **Visualizations**
- **25+ PNG Charts**: Performance comparisons, feature importance, precision-recall curves
- **Interactive Reports**: Parsed analysis reports with actionable insights
- **Multi-Modal Analysis**: Fundamentals, text, and ensemble model visualizations

## 🚀 Quick Start

### Prerequisites
```bash
# Install frontend dependencies
pip install -r frontend/requirements.txt

# Or install from main requirements.txt
pip install -r requirements.txt
```

### Launch Application
```bash
# From project root directory
python start_frontend.py
```

The startup script will:
1. ✅ Check dependencies and database
2. 🚀 Start FastAPI backend on `localhost:8000`
3. 🌐 Automatically open browser to frontend
4. 📊 Load analysis visualizations and data

### Manual Launch (Alternative)
```bash
cd frontend
uvicorn backend:app --host 127.0.0.1 --port 8000 --reload
```

## 📱 Interface Overview

### 🔧 **Control Panel**
- **Setup Selector**: Choose from 100+ confirmed stock setups
- **Date Range Filters**: Customize analysis timeframes  
- **Probability Threshold**: Adjust scanner sensitivity

### 📋 **Information Ribbons**
1. **Prediction Analysis**: ML predictions with confidence metrics
2. **Raw Data Tables**: Filtered fundamentals, news, and posts data
3. **Scanner Results**: High-probability setup discovery
4. **Perfect Setup Guide**: Optimal feature benchmarks
5. **Knowledge Graph**: AI explanations with key factors
6. **Evaluation Metrics**: Model performance across components
7. **Visualizations**: Interactive charts and reports

## 🎛️ **API Endpoints**

### Data Access
- `GET /api/setups` - List all confirmed setups
- `GET /api/setup/{id}/fundamentals` - Setup fundamentals data
- `GET /api/setup/{id}/news` - Setup news data  
- `GET /api/setup/{id}/userposts` - Setup user posts data

### Predictions
- `POST /api/predict` - Generate setup prediction
- `GET /api/batch_predict` - Batch predictions with filtering

### Analysis
- `POST /api/scan` - Scan for high-probability setups
- `GET /api/perfect_setup` - Get optimal feature ranges
- `GET /api/visualizations` - List available charts
- `GET /api/reports/{category}` - Get analysis reports

### Utilities
- `GET /api/health` - Backend health check
- `GET /api/visualization/{category}/{file}` - Serve PNG/TXT files

## 💡 Usage Tips

### 📊 **Effective Analysis Workflow**
1. **Start with Scanner**: Find high-probability setups (`> 70%`)
2. **Select Setup**: Choose a setup from dropdown or scanner results
3. **Review Data**: Check fundamentals, news, and user sentiment
4. **Analyze Prediction**: Understand ML model reasoning
5. **Compare to Perfect**: See how setup compares to optimal ranges
6. **Check Visualizations**: Dive into detailed performance metrics

### 🔍 **Advanced Features**
- **Filter Data Tables**: Use search boxes to find specific metrics
- **Compare Models**: Switch between fundamentals, text, and ensemble tabs
- **Export Reports**: Click "View Full Report" for detailed TXT analysis
- **Scanner Optimization**: Adjust probability thresholds for different risk tolerance

## 🏗️ **Architecture**

### Backend (FastAPI)
- **Database Layer**: DuckDB integration for data access
- **ML Integration**: Ensemble model predictions with 3 components
- **Knowledge Explainer**: LLM-based explanation generation
- **File Serving**: PNG visualizations and TXT reports
- **Error Handling**: Comprehensive error responses

### Frontend (HTML/CSS/JS)
- **Responsive Design**: Mobile-friendly ribbon layout
- **Real-time Updates**: Dynamic content loading via AXIOS
- **Interactive Charts**: Chart.js integration for metrics
- **Modern UI**: CSS Grid/Flexbox with smooth animations

## 🔧 **Configuration**

### Database Path
```python
# In backend.py
DB_PATH = "data/sentiment_system.duckdb"
```

### API Base URL  
```javascript
// In app.js
this.apiBase = 'http://localhost:8000/api';
```

### Port Configuration
```python
# In start_frontend.py
uvicorn.run(host="127.0.0.1", port=8000)
```

## 🛠️ **Troubleshooting**

### Common Issues

**Backend Connection Failed**
- Ensure database file exists: `data/sentiment_system.duckdb`
- Check port 8000 is available
- Verify all dependencies installed

**Missing Visualizations**  
- Run ML pipeline: `python orchestration/run_complete_pipeline_duckdb.py --skip-features`
- Check analysis directories exist: `analysis_ml_fundamentals/`, `analysis_ml_text/`, `analysis_ml_ensemble/`

**No Setup Data**
- Verify database has confirmed setups: `SELECT COUNT(*) FROM setups WHERE confirmed = true`
- Check features tables exist: `news_features`, `userposts_features`

**Prediction Errors**
- Ensure ML models loaded successfully (check startup logs)
- Verify setup has required feature data
- Check ensemble analyzer initialization

### Debug Mode
```bash
# Enable debug logging
cd frontend
uvicorn backend:app --log-level debug --reload
```

## 📚 **Related Documentation**

- [ML Pipeline README](../docs/ML_PIPELINE_ENHANCEMENTS.md)
- [Complete Analysis Summary](../docs/COMPLETE_ML_ANALYSIS_SUMMARY.md)  
- [Ensemble Model Documentation](../docs/HYBRID_MODEL_README.md)
- [Database Schema](../docs/ENHANCED_USERPOSTS_SCHEMA_SYNC.md)

## 🎉 **Success Metrics**

Once running successfully, you should see:
- ✅ Backend health check passes
- 📊 Setup selector populated with 100+ setups
- 🎯 Predictions generate with 60-80% accuracy
- 📈 25+ visualizations load correctly
- 🧠 Knowledge explanations provide actionable insights

---

**Happy Analyzing! 🚀📊** 