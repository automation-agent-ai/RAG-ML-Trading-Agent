# Enhanced Agent Prediction System

## Overview

The Enhanced Agent Prediction System is designed to leverage domain-specific expertise to make high-quality predictions about stock outperformance. The system uses a confidence-weighted ensemble approach to combine predictions from four specialized domain agents:

1. **News Agent**: Analyzes financial news and press releases
2. **Fundamentals Agent**: Evaluates financial metrics and ratios
3. **Analyst Recommendations Agent**: Processes analyst ratings and target prices
4. **User Posts Agent**: Analyzes social media sentiment and discussions

Each agent makes independent predictions, which are then combined using confidence-weighted voting to produce a final ensemble prediction.

## Domain-Specific Prediction Methods

### News Agent

The News Agent predicts outperformance based on:
- **Sentiment Analysis**: Evaluates the tone and content of financial news
- **Event Detection**: Identifies significant events like earnings releases, M&A, etc.
- **Risk Assessment**: Detects profit warnings and negative signals

**Outperformance Formula**:
```
IF sentiment_score > 0.6:
    outperformance = sentiment_score * 10
ELSE IF sentiment_score < -0.3:
    outperformance = sentiment_score * 8
ELSE:
    outperformance = sentiment_score * 5
```

### Fundamentals Agent

The Fundamentals Agent predicts outperformance based on:
- **Profitability Metrics**: ROA, ROE
- **Growth Metrics**: Revenue growth, earnings growth
- **Financial Health**: Debt-to-equity ratio, current ratio
- **Margin Analysis**: Gross margin, net margin

**Outperformance Formula**:
```
outperformance = (ROA * 2) + (ROE * 2) + (revenue_growth * 3) - 
                 (debt_to_equity * 0.5) + (current_ratio * 0.5) + 
                 (gross_margin * 2)
```

### Analyst Recommendations Agent

The Analyst Recommendations Agent predicts outperformance based on:
- **Consensus Ratings**: Buy/Hold/Sell recommendations
- **Price Targets**: Analyst price targets relative to current price
- **Rating Changes**: Recent upgrades or downgrades
- **Conviction Scores**: Strength of analyst opinions

**Outperformance Formula**:
```
outperformance = (analyst_conviction_score * 7.5) - 
                 (recent_downgrades * 2.5) + 
                 (recent_upgrades * 2.5)
```

### User Posts Agent

The User Posts Agent predicts outperformance based on:
- **Community Sentiment**: Overall sentiment from social media
- **Sentiment Trend**: Changes in sentiment over time
- **Contrarian Signals**: Excessive optimism or pessimism
- **Topic Analysis**: Discussion topics and their relevance

**Outperformance Formula**:
```
outperformance = (avg_sentiment * 4) + 
                 (community_sentiment_score * 3) - 
                 (contrarian_signal * 2)
```

## Confidence-Weighted Ensemble

The system uses a sophisticated confidence-weighted approach to combine predictions from all domains:

### 1. Confidence Score Calculation

Each domain agent calculates a confidence score based on:
- Signal strength (positive and negative)
- Data quality and completeness
- Historical accuracy in similar situations

```python
confidence_score = min(0.95, max(0.33, 
    (abs(positive_signal_strength) + abs(negative_risk_score)) / 2 + 0.3))
```

### 2. Weighted Label Prediction

The ensemble prediction uses confidence scores as weights:
```python
weighted_sum = sum(label * confidence for label, confidence in zip(domain_labels, domain_confidences))
weighted_label_value = weighted_sum / sum(domain_confidences)
```

The final label is determined using the same thresholds as the ML models:
```python
if weighted_label_value >= pos_threshold:
    ensemble_label = 1  # Positive
elif weighted_label_value <= neg_threshold:
    ensemble_label = -1  # Negative
else:
    ensemble_label = 0  # Neutral
```

### 3. Weighted Outperformance Prediction

Similarly, outperformance predictions are weighted by confidence:
```python
weighted_outperformance = sum(outperf * conf for outperf, conf in zip(domain_outperformances, domain_confidences))
weighted_outperformance = weighted_outperformance / sum(domain_confidences)
```

### 4. Ensemble Confidence Calculation

The ensemble confidence considers both:
- Average confidence across domains
- Agreement between domains (lower when domains disagree)

```python
# Calculate domain agreement factor
if len(domain_labels) > 1:
    label_std = np.std(domain_labels) / 2
    domain_agreement = max(0.5, 1.0 - label_std)
else:
    domain_agreement = 1.0

# Final confidence
ensemble_confidence = (avg_domain_confidence) * domain_agreement
```

## Consistency with ML Models

The agent prediction system ensures consistency with ML models through:

1. **Shared Thresholds**: Both systems use the same thresholds to convert continuous predictions to discrete labels (-1, 0, 1)
2. **Consistent Label Format**: The same label format (-1, 0, 1) is used throughout
3. **Comparable Confidence Scores**: Both systems produce confidence scores in a similar range
4. **Normalized Outperformance**: Predictions are clipped to a realistic range (-15% to +15%)

## Benefits of the Enhanced Approach

The enhanced agent prediction system offers several advantages:

1. **Domain Expertise**: Leverages specialized knowledge from different domains
2. **Confidence-Weighted Decisions**: More reliable predictions have greater influence
3. **Explicit Outperformance Prediction**: Provides numeric predictions, not just labels
4. **Transparency**: Detailed logging of prediction distribution and confidence
5. **Complementary to ML**: Provides an independent prediction approach that can be compared with ML models
6. **Adaptability**: Each domain can be tuned or improved independently
7. **Robustness**: The ensemble approach reduces the impact of errors from individual domains

## Implementation Details

The agent prediction system is implemented in `make_agent_predictions.py` with the following key components:

1. **Domain-specific SQL queries**: Extract relevant features for each domain
2. **Prediction formulas**: Calculate outperformance predictions for each domain
3. **Confidence calculation**: Determine confidence scores for each prediction
4. **Ensemble algorithm**: Combine predictions using confidence-weighted voting
5. **Threshold management**: Ensure consistent thresholds with ML models
6. **Comprehensive logging**: Detailed statistics about predictions and confidence

## Performance Metrics

The system tracks several key metrics:

1. **Class Distribution**: Percentage of positive, neutral, and negative predictions
2. **Outperformance Statistics**: Mean, median, min, and max predicted outperformance
3. **Confidence Statistics**: Mean, median, min, and max confidence scores
4. **Domain Coverage**: Number of domains contributing to each prediction
5. **Agreement Rate**: How often domains agree on the prediction

These metrics help evaluate the system's performance and identify areas for improvement. 