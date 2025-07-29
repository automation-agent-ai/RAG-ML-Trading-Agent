# 🚀 Enhanced RAG Pipeline Frontend Dashboard

A sophisticated, modern web dashboard for your RAG (Retrieval-Augmented Generation) pipeline that provides live AI agent predictions, comprehensive data analysis, and interactive visualizations for stock trading setup analysis.

## ✨ Key Features

### 🎭 Live Agent Prediction Theater
- **Real-time WebSocket streaming** of 4 AI agents (Fundamentals, News, Analyst, Community)
- **Step-by-step visualization** of the complete prediction workflow
- **Progress tracking** with agent status cards and live logging
- **Ensemble prediction** combining all agent outputs
- **Ground truth comparison** with actual market outcomes

### 🔍 Enhanced Prediction Analysis
- **Similar embeddings discovery** to find historical cases
- **Smart setup selection** with random sampling capabilities
- **Confidence scoring** and accuracy indicators
- **AI reasoning explanations** for transparent decision-making

### 📊 Interactive Data Explorer
- **Multi-domain data analysis** (Fundamentals, News, Community, Analyst Coverage)
- **Comprehensive feature visualization** from your ML pipeline
- **Advanced filtering and search** capabilities
- **Real-time data loading** from DuckDB

### 🤖 Model Performance Dashboard
- **ML model comparison** with precision, recall, F1-score, and AUC metrics
- **Performance tracking** and evaluation history
- **Best model identification** with crown indicators

### 🎨 Visualization Gallery
- **Interactive analysis charts** from your ML pipeline
- **Category-based organization** (Fundamentals, Text, Ensemble)
- **PNG and TXT report support** with metadata display

### 🧠 Knowledge Graph Explorer
- **Interactive graph visualization** using Cytoscape.js
- **Relationship mapping** between setups, features, and predictions
- **Similar node discovery** and reasoning path analysis

### 🎨 Professional Themes
- **6 beautiful themes**: Modern Blue, Dark, Professional Slate, Modern Teal, Executive Indigo, Financial Rose
- **Smooth transitions** and responsive design
- **Persistent theme preferences** with localStorage

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Your existing RAG pipeline with DuckDB database
- Node.js (for development, optional)

### Installation

1. **Clone and navigate to your pipeline directory**
```bash
   cd your-rag-pipeline
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the enhanced backend**
   ```bash
   python backend.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8000`

### Configuration

The backend automatically connects to your DuckDB database at `data/sentiment_system.duckdb`. Ensure your database contains the required tables:
- `labels` - Ground truth labels
- `fundamentals_features` - Financial analysis features
- `news_features` - News sentiment features
- `userposts_features` - Community sentiment features
- `analyst_recommendations_features` - Analyst coverage features

## 📱 User Interface Guide

### Live Prediction Theater 🎭
1. **Select number of setups** (1-20) to analyze
2. **Click "Start Live Prediction Theater"** to begin
3. **Watch real-time progress** as 4 AI agents analyze each setup:
   - 🏛️ **Fundamentals Agent**: Financial metrics and performance
   - 📰 **News Agent**: Corporate news and sentiment analysis
   - 📈 **Analyst Agent**: Professional analyst recommendations
   - 💬 **Community Agent**: Social media and forum sentiment
4. **View ensemble predictions** combined from all agents
5. **Compare with actual outcomes** for accuracy assessment

### Setup Analysis 📊
1. **Select a setup** from the dropdown or use "Random Setup"
2. **Generate Enhanced Prediction** to see AI analysis
3. **Find Similar Setups** to discover historical patterns
4. **View detailed metrics** including confidence scores and accuracy

### Portfolio Scanner 🔍
1. **Set date range** for analysis period
2. **Adjust minimum probability** threshold
3. **Scan for high-probability opportunities** with risk assessment

### Data Explorer 📋
- **Browse comprehensive features** extracted by your pipeline
- **Switch between data domains** using tabs
- **Search and filter** specific setups or metrics

## 🛠️ Technical Architecture

### Backend (FastAPI)
- **Async WebSocket support** for real-time streaming
- **RESTful API endpoints** for data access
- **DuckDB integration** with connection pooling
- **Agent prediction integration** with your existing pipeline
- **Error handling and validation** with Pydantic models

### Frontend (Vanilla JS)
- **Modern ES6+ JavaScript** with modular architecture
- **WebSocket client** for real-time communication
- **Responsive CSS Grid/Flexbox** layout
- **CSS Custom Properties** for advanced theming
- **Toast notifications** for user feedback

### Key Technologies
- **FastAPI** - High-performance async web framework
- **WebSockets** - Real-time bidirectional communication
- **DuckDB** - High-performance analytical database
- **Cytoscape.js** - Graph visualization engine
- **CSS Grid/Flexbox** - Modern responsive layouts

## 📊 API Endpoints

### Core Endpoints
- `GET /api/health` - System health and component status
- `GET /api/setups` - Available trading setups with filtering
- `POST /api/predict` - Enhanced prediction with similarity analysis
- `GET /api/setup/{id}/similar` - Find similar setups by embeddings

### Data Endpoints
- `GET /api/setup/{id}/fundamentals` - Financial analysis data
- `GET /api/setup/{id}/news` - News sentiment analysis
- `GET /api/setup/{id}/userposts` - Community sentiment data
- `GET /api/model-performance` - ML model comparison metrics
- `GET /api/visualizations` - Available analysis charts

### Live Features
- `WS /ws/live-prediction` - Real-time agent prediction theater
- Progressive agent status updates
- Ensemble prediction streaming
- Error handling and recovery

## 🎨 Customization

### Adding New Themes
1. Add theme variables to `static/style.css`:
   ```css
   [data-theme="custom"] {
       --primary-color: #your-color;
       --secondary-color: #your-color;
       /* ... other variables */
   }
   ```

2. Add theme option to `index.html`:
   ```html
   <option value="custom">🎨 Custom Theme</option>
   ```

### Extending API Endpoints
1. Add new endpoints to `backend.py`
2. Update JavaScript API client in `static/app.js`
3. Add UI components as needed

### Custom Visualizations
1. Place PNG/TXT files in `visualizations/` or `ml/analysis/` directory
2. Files will automatically appear in the gallery
3. Organize by subdirectories for categories

## 🔧 Development

### Running in Development Mode
```bash
# Backend with auto-reload
uvicorn backend:app --reload --host 0.0.0.0 --port 8000

# For frontend development, serve static files separately if needed
python -m http.server 3000 --directory static
```

### Code Structure
```
production_pipeline/
├── backend.py              # Enhanced FastAPI backend
├── index.html              # Main frontend application
├── static/
│   ├── style.css          # Enhanced CSS with themes
│   └── app.js             # Comprehensive JavaScript app
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚨 Troubleshooting

### Common Issues

**WebSocket Connection Fails**
- Ensure backend is running on port 8000
- Check firewall settings
- Verify WebSocket support in browser

**Database Connection Errors**
- Confirm `data/sentiment_system.duckdb` exists
- Check database table structure
- Verify file permissions

**Agent Predictions Not Working**
- Ensure agent classes are properly imported
- Check OpenAI API key configuration
- Verify agent model initialization

**Themes Not Loading**
- Clear browser cache and localStorage
- Check CSS custom property support
- Verify theme selector JavaScript

## 📈 Performance

### Optimization Features
- **Async request handling** for non-blocking operations
- **WebSocket connection pooling** for efficient real-time updates
- **Progressive data loading** to improve perceived performance
- **CSS animations with GPU acceleration**
- **Efficient DOM manipulation** with minimal reflows

### Recommended Limits
- **Live Theater**: Maximum 20 setups per session
- **Similar Setups**: Default 10 results for optimal performance
- **Health Monitoring**: 30-second intervals to avoid API overload

## 🔒 Security

### Security Features
- **Input validation** with Pydantic models
- **CORS configuration** for controlled access
- **Rate limiting** on WebSocket connections
- **SQL injection protection** through parameterized queries

### Production Deployment
- Use HTTPS for WebSocket connections (WSS)
- Configure proper CORS origins
- Set up authentication if needed
- Monitor API usage and implement rate limiting

## 🤝 Contributing

We welcome contributions! The frontend is designed to be:
- **Modular and extensible**
- **Well-documented with inline comments**
- **Following modern web development practices**
- **Responsive and accessible**

## 📄 License

This enhanced frontend is part of your RAG pipeline project. Please refer to your main project license.

---

**Built with ❤️ for sophisticated ML pipeline analysis and visualization**

🎭 **Enjoy the Live Agent Prediction Theater!** 🎭 