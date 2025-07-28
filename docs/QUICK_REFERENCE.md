# 🚀 Production Pipeline - Quick Reference

## 🔥 One-Line Commands

### **Complete Pipeline (Recommended)**
```bash
cd production_pipeline/
python run_complete_ml_pipeline.py --mode training
```

### **Quick Demo**
```bash
python complete_workflow.py --demo
```

### **Custom Setup IDs**
```bash
python run_complete_ml_pipeline.py --mode training --setup-ids AFN_2023-09-11 SETUP_002
```

---

## 📊 Step-by-Step Commands

### **1. Create Embeddings** (if needed)
```bash
python embeddings/embed_news_duckdb.py --db-path data/sentiment_system.duckdb --lancedb-dir data/lancedb_store
```

### **2. Train & Predict** (Python)
```python
from run_complete_ml_pipeline import CompletePipeline
from ml.multiclass_predictor import MultiClassPredictor

# Complete pipeline
pipeline = CompletePipeline()
results = pipeline.run_complete_pipeline(['SETUP_001'], mode='training')

# ML training & prediction
predictor = MultiClassPredictor('data/sentiment_system.duckdb')
model = predictor.train_quick_model()
prediction = predictor.predict_setup('SETUP_001')
```

---

## 📈 What You Get

### **Features**
- **Text Features**: 46 (news sentiment, user posts, analyst ratings)
- **Financial Features**: 60 (fundamentals + ratios + YoY growth)
- **Preprocessing**: Automated scaling, imputation, outlier handling

### **ML Model**
- **Algorithm**: RandomForestClassifier (balanced)
- **Classes**: 3 (negative, neutral, positive)
- **Output**: Predictions with confidence scores

### **Performance**
- **Speed**: ~3.7s for 613 setups
- **Training**: 723 samples, 9 selected features
- **Real-time**: Instant predictions

---

## 🔧 Common Workflows

### **New Data Processing**
```bash
# 1. Add new data to DuckDB
# 2. Run pipeline
python run_complete_ml_pipeline.py --mode training

# 3. Make predictions
python complete_workflow.py --setup-ids NEW_SETUP_001
```

### **Production Batch Processing**
```python
# Process all available data
pipeline = CompletePipeline()
results = pipeline.run_complete_pipeline(mode='training')

# Train model
predictor = MultiClassPredictor('data/sentiment_system.duckdb')
model = predictor.train_quick_model()

# Batch predictions
new_setups = ['NEW_001', 'NEW_002', 'NEW_003']
predictions = predictor.batch_predict(new_setups)
```

---

## 🚨 Troubleshooting

### **Embeddings Not Found**
```bash
# Solution: Create embeddings first
python embeddings/embed_news_duckdb.py --db-path data/sentiment_system.duckdb --lancedb-dir data/lancedb_store
```

### **Model Not Trained**
```python
# Solution: Train and predict in same session
model = predictor.train_quick_model()
prediction = predictor.predict_setup('SETUP_ID')
```

### **Check Data Availability**
```python
import duckdb
conn = duckdb.connect('data/sentiment_system.duckdb')
print("Setups:", conn.execute('SELECT COUNT(*) FROM setups').fetchone()[0])
print("Features:", conn.execute('SELECT COUNT(*) FROM news_features').fetchone()[0])
```

---

## 📚 File Locations

- **Main Pipeline**: `run_complete_ml_pipeline.py`
- **ML Training**: `../ml/multiclass_predictor.py` 
- **Features Export**: `export_ml_features.py`
- **Financial Merging**: `merge_financial_features.py`
- **Complete Demo**: `complete_workflow.py`
- **Full Documentation**: `COMPLETE_PIPELINE_GUIDE.md`

---

## 🎯 Expected Output

```
🚀 Training quick multi-class model...
✅ Model trained on 723 samples
   Features: 9
   Classes: [0 1 2]

Prediction for SETUP_001: (2, 0.425, {
    'setup_id': 'SETUP_001',
    'prediction': 2,
    'confidence': 0.425,
    'probabilities': {
        'negative': 0.236,
        'neutral': 0.339,
        'positive': 0.425
    }
})
```

**🎉 Your production pipeline is ready!** 🚀 