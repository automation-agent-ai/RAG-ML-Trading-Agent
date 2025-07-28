# Feature Differences Between Training and Prediction Modes

This guide explains the key differences between features generated in training mode versus prediction mode, and how to manage these differences effectively.

## Understanding Feature Types

Our ML pipeline generates two main types of features:

### 1. Raw Features (Domain-Specific)

These features are extracted directly from source data without using similarity search or labels:

- **News features**: Sentiment scores, entity counts, topic classifications
- **Fundamentals features**: Financial ratios, growth rates, balance sheet metrics
- **UserPosts features**: Sentiment analysis, engagement metrics, topic distribution
- **Analyst Recommendations features**: Rating counts, consensus metrics

### 2. Similarity-Based Features

These features are derived from comparing a setup with other similar setups:

- **Positive signal strength**: Weighted ratio of positive outcomes in similar cases
- **Negative risk score**: Weighted ratio of negative outcomes in similar cases
- **Neutral probability**: Weighted ratio of neutral outcomes in similar cases
- **Historical pattern confidence**: Confidence score based on similarity distribution
- **Similar cases count**: Number of similar cases found

## Key Differences Between Training and Prediction Features

| Feature Aspect | Training Mode | Prediction Mode | Difference |
|----------------|---------------|----------------|------------|
| **Raw Features** | Extracted from source data | Extracted from source data | **No difference** - Raw features are identical in both modes |
| **Similarity Search** | Uses all training embeddings | Uses all training embeddings (excluding prediction setups) | **Different pool** - Prediction mode has fewer reference embeddings |
| **Label Access** | Has access to true labels | No access to true labels | **Different inputs** - Prediction mode cannot use label information |
| **Similarity Features** | Based on all training data | Based on reduced training data | **Different values** - Due to different reference embeddings |
| **Feature Storage** | Stored in main feature tables | Stored in main feature tables with domain identification | **Same storage** - But can be distinguished by metadata |

## Why Features Differ

1. **Different Embedding Pools**: In prediction mode, the embeddings for prediction setups are removed from the training tables, so similarity searches have a slightly different pool of reference embeddings.

2. **No Label Access**: Prediction mode cannot use true labels (`outperformance_10d`) when generating features, so any feature that depends on labels will be different.

3. **Different Similar Cases**: Because of the different embedding pools, the "most similar cases" found for a setup might be different between training and prediction modes.

## Managing Feature Differences

### Option 1: Complete Re-extraction

Re-extract all features in prediction mode. This ensures complete separation but is computationally expensive.

**Pros:**
- Complete separation between training and prediction
- Ensures no data leakage

**Cons:**
- Computationally expensive
- Raw features will be identical to original extraction in most cases

### Option 2: Selective Re-extraction (Recommended)

Keep raw features but re-extract only similarity-based features. This is more efficient while maintaining separation where it matters.

**Pros:**
- More efficient - avoids redundant computation
- Maintains data separation where it matters
- Raw features remain consistent

**Cons:**
- Requires careful tracking of which features are similarity-based

## Implementation

### Using the `reset-similarity` Command

The `reset-similarity` command in `preserve_restore_embeddings.py` resets only the similarity-based features:

```bash
python preserve_restore_embeddings.py reset-similarity --prediction-setup-file data/prediction_setups.txt
```

This command:
1. Identifies the similarity-based features for each domain
2. Sets these features to NULL for the prediction setups
3. Leaves the raw features intact

### Similarity-Based Features by Domain

Each domain has specific similarity-based features that need to be reset:

```python
similarity_columns = {
    "news": [
        "positive_signal_strength", "negative_risk_score", 
        "neutral_probability", "historical_pattern_confidence",
        "similar_cases_count"
    ],
    "fundamentals": [
        "positive_signal_strength", "negative_risk_score", 
        "neutral_probability", "historical_pattern_confidence",
        "similar_cases_count"
    ],
    "analyst_recommendations": [
        "positive_signal_strength", "negative_risk_score", 
        "neutral_probability", "historical_pattern_confidence",
        "similar_cases_count"
    ],
    "userposts": [
        "positive_signal_strength", "negative_risk_score", 
        "neutral_probability", "historical_pattern_confidence",
        "similar_cases_count"
    ]
}
```

## Ensemble Prediction and Feature Differences

The ensemble prediction system combines predictions from different domains, which helps mitigate the impact of feature differences:

1. **Majority Voting**: By taking a vote across domains, outliers due to feature differences are less likely to affect the final prediction.

2. **Weighted Average**: Domains with more reliable features can be given higher weights, reducing the impact of domains with less reliable features.

3. **Confidence Scores**: The confidence score provides a measure of how reliable the prediction is, which can help identify cases where feature differences might be causing issues.

## Best Practices

1. **Always reset similarity features** before re-extracting in prediction mode to ensure clean separation.

2. **Monitor feature distributions** between training and prediction modes to identify any unexpected differences.

3. **Use ensemble prediction** to combine results from multiple domains, which helps reduce the impact of feature differences in any single domain.

4. **For critical applications**, consider using the complete re-extraction approach to ensure maximum separation between training and prediction.

## Conclusion

While raw features remain consistent between training and prediction modes, similarity-based features will differ due to the different embedding pools and lack of label access. The selective re-extraction approach provides an efficient way to manage these differences while maintaining proper separation between training and prediction data. 