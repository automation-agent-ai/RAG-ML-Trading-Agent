# 🎯 Cursor AI Prompt: Enhanced RAG-Based ML Pipeline with Similarity Search

## 📖 **Project Context**

You are working on a **sophisticated ML pipeline for stock performance prediction** that uses LLM agents to extract features from financial data. The pipeline is documented in:

- **[COMPLETE_PIPELINE_GUIDE.md](COMPLETE_PIPELINE_GUIDE.md)** - Complete workflow documentation
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Common commands and usage patterns
- **[DATA_LEAKAGE_FIX_GUIDE.md](DATA_LEAKAGE_FIX_GUIDE.md)** - Critical data leakage issue we're solving

### **Current Pipeline Architecture:**
```
1. 📊 Raw Data (DuckDB): news, userposts, fundamentals, analyst_recommendations
2. 🎰 Create Embeddings → Store in LanceDB (WITH labels for training)
3. 🤖 LLM Agents → Extract features from raw data → Store in DuckDB
4. 🔄 Merge Features → Create ML feature tables
5. 🎯 Train/Predict → ML models
```

## 🚨 **Critical Problem: Data Leakage in Prediction Mode**

### **The Issue:**
Currently, embeddings include `outperformance_10d` labels in metadata for **BOTH training AND prediction modes**. This creates data leakage when predicting new setups.

**Evidence of the problem:**
```python
# ❌ embeddings/embed_news_duckdb.py:343
'outperformance_10d': float(label_row.get('outperformance_10d', 0.0))  # Always included!

# ❌ embeddings/embed_fundamentals_duckdb.py:305
'outperformance_10d': float(latest_label.get('outperformance_10d', 0))  # Always included!
```

### **✅ Correct Pattern (already implemented for features):**
We **correctly** separate training and prediction data in ML feature tables:
- `text_ml_features_training` vs `text_ml_features_prediction`
- `financial_ml_features_training` vs `financial_ml_features_prediction`

**We need the same separation for embeddings!**

## 🎯 **Goal: Implement Enhanced RAG-Based Agent System**

### **Training Mode:**
```
Content → Embedding + Label Metadata → Store in LanceDB_Training
Agents → Extract features + Create enhanced similarity features
```

### **Prediction Mode:**
```
Content → Embedding (NO label) → Use for similarity search (DON'T store)
Agents → Find similar training embeddings → Extract features + Predict
```

## 💡 **Two-Phase Implementation Plan**

### **Phase 1: Option 1 - Enhanced Features via Similarity**

Modify agents to create **inference-aware features** by finding similar training cases:

```python
def extract_enhanced_features(self, setup_id, content, mode='training'):
    if mode == 'training':
        # Current feature extraction + store embedding with label
        features = self.extract_basic_features(content)
        embedding = self.create_embedding(content)
        self.store_embedding_with_label(embedding, label)  # Store in LanceDB
        
    else:  # prediction mode
        # Create embedding but DON'T store, use for similarity search
        embedding = self.create_embedding(content)  # NO label
        similar_cases = self.find_similar_training_embeddings(embedding)
        
        # Extract basic features
        features = self.extract_basic_features(content)
        
        # NEW: Add similarity-based inference features
        enhanced_features = {
            'positive_signal_strength': self.compute_positive_similarity(similar_cases),
            'negative_risk_score': self.compute_negative_similarity(similar_cases),
            'neutral_probability': self.compute_neutral_similarity(similar_cases),
            'historical_pattern_confidence': self.compute_pattern_confidence(similar_cases),
            'similar_cases_count': len(similar_cases)
        }
        
        features.update(enhanced_features)
        # DON'T store the prediction embedding!
        
    return features
```

### **Phase 2: Option 2 - Direct Similarity Prediction**

Add direct similarity-based prediction capability:

```python
def predict_via_similarity(self, setup_id, content):
    """Direct prediction using similarity to training embeddings"""
    embedding = self.create_embedding(content)  # NO label
    similar_cases = self.find_similar_training_embeddings(embedding, limit=10)
    
    # Extract labels from similar cases
    similar_labels = [case['outperformance_10d'] for case in similar_cases]
    
    # Compute prediction
    prediction = {
        'predicted_label': self.aggregate_similar_labels(similar_labels),
        'confidence': self.compute_confidence(similar_labels),
        'positive_ratio': sum(l > 0 for l in similar_labels) / len(similar_labels),
        'negative_ratio': sum(l < 0 for l in similar_labels) / len(similar_labels),
        'neutral_ratio': sum(abs(l) < 0.01 for l in similar_labels) / len(similar_labels),
        'similar_cases': similar_cases  # For interpretability
    }
    
    return prediction
```

## 🔧 **Implementation Requirements**

### **1. Modify Embedding Scripts**
Update embedding classes to accept `include_labels` parameter:

```python
# embeddings/embed_news_duckdb.py
class NewsEmbeddingPipelineDuckDB:
    def __init__(self, db_path, lancedb_dir, include_labels=True):
        self.include_labels = include_labels
        
    def enrich_records_with_labels(self, records):
        if not self.include_labels:  # Prediction mode
            return records  # Skip label enrichment
        # ... existing label enrichment code
```

### **2. Add Similarity Search Methods to Agents**

```python
# Add to each agent class:
def find_similar_training_embeddings(self, query_embedding, limit=10):
    """Find similar training embeddings with labels"""
    if self.table is None:
        raise ValueError("LanceDB table not available")
    
    results = self.table.search(query_embedding).limit(limit).to_pandas()
    return results.to_dict('records')
def compute_positive_similarity(self, similar_cases):
    """Compute positive signal strength from similar cases"""
    if not similar_cases:
        return 0.0
    
    positive_cases = [c for c in similar_cases if c.get('outperformance_10d', 0) > 0]
    return len(positive_cases) / len(similar_cases)
def compute_negative_similarity(self, similar_cases):
    """Compute negative risk score from similar cases"""
    if not similar_cases:
        return 0.0
        
    negative_cases = [c for c in similar_cases if c.get('outperformance_10d', 0) < 0]
    return len(negative_cases) / len(similar_cases)
```

### **3. Update Pipeline Integration**

Modify `run_complete_ml_pipeline.py` to handle the new modes:

```python
def create_embeddings(self, setup_ids, mode='training'):
    """Create embeddings with proper training/prediction separation"""
    include_labels = (mode == 'training')
    
    # Pass include_labels to embedding scripts
    news_embedder = NewsEmbeddingPipelineDuckDB(
        db_path=self.db_path,
        lancedb_dir=self.lancedb_dir,
        include_labels=include_labels
    )
```

## 🎯 **Success Criteria**

1. **✅ No Data Leakage**: Prediction embeddings contain no labels
2. **✅ Enhanced Features**: Agents create similarity-based inference features
3. **✅ Direct Prediction**: Agents can predict directly via similarity
4. **✅ Ensemble Capability**: Combine similarity + ML predictions
5. **✅ Interpretability**: Can see which similar cases influenced predictions

## 📋 **Key Files to Modify**

- `agents/*/enhanced_*_agent_duckdb.py` - Add similarity search methods
- `embeddings/embed_*_duckdb.py` - Add include_labels parameter
- `run_complete_ml_pipeline.py` - Update pipeline orchestration
- `core/ml_feature_merger.py` - Handle enhanced features

## 🚀 **Testing Strategy**

1. Test embedding creation with/without labels
2. Verify similarity search returns relevant cases
3. Test enhanced feature extraction in prediction mode
4. Compare Option 1 vs Option 2 vs Ensemble predictions
5. Verify no prediction embeddings are stored in LanceDB

---

**🎯 Implement both Option 1 (enhanced features) and Option 2 (direct similarity prediction) to create a robust, interpretable, and leak-free prediction system.** 