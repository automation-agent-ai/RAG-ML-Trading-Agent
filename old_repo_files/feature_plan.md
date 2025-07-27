# Feature Engineering Plan

| Domain        | Feature Name            | Type        | LLM Needed | Categories/Allowed Values                                                                 | Notes/How Extracted                      |
|---------------|------------------------|-------------|------------|------------------------------------------------------------------------------------------|------------------------------------------|
| UserPosts     | outperformance_pred    | numeric     | no         | (float)                                                                                  | Retrieval + label mean                   |
| UserPosts     | avg_sentiment          | numeric     | no         | -1.0 to +1.0                                                                             | Statistical sentiment (VADER)            |
| UserPosts     | consensus_level        | categorical | yes        | high, medium, low                                                                        | LLM feature extraction                   |
| UserPosts     | consensus_topics       | category[]  | yes        | earnings, dividends, guidance, macro, product, rumor, other                              | LLM feature extraction                   |
| UserPosts     | controversial_topics   | category[]  | yes        | earnings, dividends, guidance, macro, product, rumor, other                              | LLM feature extraction                   |
| UserPosts     | rumor_intensity        | numeric     | yes        | 0.0–1.0                                                                                  | LLM feature extraction                   |
| UserPosts     | trusted_user_sentiment | numeric     | yes        | -1.0 to +1.0                                                                             | LLM - trusted users focus                |
| UserPosts     | post_count             | numeric     | no         | (int ≥ 0)                                                                                | Record count                             |
| UserPosts     | unique_users           | numeric     | yes        | (int ≥ 0)                                                                                | LLM or record count                      |
| UserPosts     | community_sentiment_score | numeric  | yes        | -1.0 to +1.0                                                                             | LLM aggregate score                      |
| UserPosts     | bull_bear_ratio        | numeric     | yes        | float ≥ 0                                                                                | LLM ratio (bullish/bearish posts)        |
| UserPosts     | contrarian_signal      | boolean     | yes        | true, false                                                                              | LLM flag                                 |
| UserPosts     | relevance_score        | numeric     | yes        | 0.0–1.0                                                                                  | LLM confidence                           |
| UserPosts     | recent_sentiment_shift | categorical | yes        | up, down, stable, unknown                                                                | LLM enum                                 |
| UserPosts     | sentiment_distribution | dict        | yes        | {bullish:int, bearish:int, neutral:int}                                                  | LLM or counted                           |
| UserPosts     | engagement_score       | numeric     | yes        | 0.0–1.0                                                                                  | LLM or computed                          |
| UserPosts     | coherence              | categorical | yes        | high, medium, low                                                                        | LLM enum                                 |
| UserPosts     | synthetic_post (summary) | text      | yes        | (string, ≤240 chars, for dashboard only)                                                 | LLM synthetic summarization              |
| UserPosts     | cot_explanation        | text        | yes        | (string, short explanation, optional, for explainability only)                           | LLM rationale (not for ML)               |

| Domain     | Feature Name                               | Type        | LLM Needed | Categories/Allowed Values                                        | Notes/How Extracted                      |
|------------|--------------------------------------------|-------------|------------|-------------------------------------------------------------------|------------------------------------------|
| News (RNS) | count_financial_results                    | int         | no         | 0+                                                                | Count of news in this group              |
| News (RNS) | max_severity_financial_results             | float       | yes        | 0.0–1.0                                                           | Max event severity in group              |
| News (RNS) | avg_headline_spin_financial_results        | categorical | yes        | positive, negative, neutral, uncertain                            | LLM (categorical)                        |
| News (RNS) | sentiment_score_financial_results          | float       | yes        | -1.0 to +1.0                                                      | LLM (numeric sentiment)                  |
| News (RNS) | profit_warning_present                     | boolean     | yes        | true, false                                                       | LLM (any profit warning in group)        |
| News (RNS) | synthetic_summary_financial_results        | text        | yes        | (≤240 chars)                                                      | LLM group summary                        |
| News (RNS) | cot_explanation_financial_results          | text        | yes        | short text                                                        | LLM rationale                            |
| News (RNS) | count_corporate_actions                    | int         | no         | 0+                                                                |                                           |
| News (RNS) | max_severity_corporate_actions             | float       | yes        | 0.0–1.0                                                           |                                           |
| News (RNS) | avg_headline_spin_corporate_actions        | categorical | yes        | positive, negative, neutral, uncertain                            |                                           |
| News (RNS) | sentiment_score_corporate_actions          | float       | yes        | -1.0 to +1.0                                                      |                                           |
| News (RNS) | capital_raise_present                      | boolean     | yes        | true, false                                                       |                                           |
| News (RNS) | synthetic_summary_corporate_actions        | text        | yes        | (≤240 chars)                                                      |                                           |
| News (RNS) | cot_explanation_corporate_actions          | text        | yes        | short text                                                        |                                           |
| News (RNS) | count_governance                           | int         | no         | 0+                                                                |                                           |
| News (RNS) | max_severity_governance                    | float       | yes        | 0.0–1.0                                                           |                                           |
| News (RNS) | avg_headline_spin_governance               | categorical | yes        | positive, negative, neutral, uncertain                            |                                           |
| News (RNS) | sentiment_score_governance                 | float       | yes        | -1.0 to +1.0                                                      |                                           |
| News (RNS) | board_change_present                       | boolean     | yes        | true, false                                                       |                                           |
| News (RNS) | synthetic_summary_governance               | text        | yes        | (≤240 chars)                                                      |                                           |
| News (RNS) | cot_explanation_governance                 | text        | yes        | short text                                                        |                                           |
| News (RNS) | count_corporate_events                     | int         | no         | 0+                                                                |                                           |
| News (RNS) | max_severity_corporate_events              | float       | yes        | 0.0–1.0                                                           |                                           |
| News (RNS) | avg_headline_spin_corporate_events         | categorical | yes        | positive, negative, neutral, uncertain                            |                                           |
| News (RNS) | sentiment_score_corporate_events           | float       | yes        | -1.0 to +1.0                                                      |                                           |
| News (RNS) | contract_award_present                     | boolean     | yes        | true, false                                                       |                                           |
| News (RNS) | merger_or_acquisition_present              | boolean     | yes        | true, false                                                       |                                           |
| News (RNS) | synthetic_summary_corporate_events         | text        | yes        | (≤240 chars)                                                      |                                           |
| News (RNS) | cot_explanation_corporate_events           | text        | yes        | short text                                                        |                                           |
| News (RNS) | count_other_signals                        | int         | no         | 0+                                                                |                                           |
| News (RNS) | max_severity_other_signals                 | float       | yes        | 0.0–1.0                                                           |                                           |
| News (RNS) | avg_headline_spin_other_signals            | categorical | yes        | positive, negative, neutral, uncertain                            |                                           |
| News (RNS) | sentiment_score_other_signals              | float       | yes        | -1.0 to +1.0                                                      |                                           |
| News (RNS) | broker_recommendation_present              | boolean     | yes        | true, false                                                       |                                           |
| News (RNS) | credit_rating_change_present               | boolean     | yes        | true, false                                                       |                                           |
| News (RNS) | synthetic_summary_other_signals            | text        | yes        | (≤240 chars)                                                      |                                           |
| News (RNS) | cot_explanation_other_signals              | text        | yes        | short text                                                        |                                           |
| News (RNS) | cot_explanation_news_grouped               | text        | yes        | short text                                                        | overall news reasoning                   |


| Domain        | Feature Name            | Type        | LLM Needed | Categories/Allowed Values                                                                 | Notes/How Extracted                      |
|---------------|------------------------|-------------|------------|------------------------------------------------------------------------------------------|------------------------------------------|
| Fundamentals  | fundamental_metrics    | numeric[]   | no         | ROE, D/E, margin, etc. (list of floats)                                                  | Raw numeric                              |
| Fundamentals  | llm_health_summary     | categorical | yes        | strong, neutral, weak                                                                    | LLM summary                              |
| Fundamentals  | uncertainty_score      | numeric     | yes        | 0–1                                                                                      | LLM feature extraction                   |
| Fundamentals  | synthetic_fundamentals | text       | yes        | (string, ≤240 chars, for dashboard only)                                                 | LLM synthetic summarization              |

| Domain                | Feature Name                    | Type        | LLM Needed | Categories/Allowed Values                                         | Notes/How Extracted                      |
|-----------------------|---------------------------------|-------------|------------|-------------------------------------------------------------------|------------------------------------------|
| AnalystRecommendations| recommendation_count            | numeric     | no         | (int ≥ 0)                                                         | Count of recommendations                 |
| AnalystRecommendations| buy_recommendations             | numeric     | no         | (int ≥ 0)                                                         | Count of buy/strong buy ratings          |
| AnalystRecommendations| sell_recommendations            | numeric     | no         | (int ≥ 0)                                                         | Count of sell/strong sell ratings        |
| AnalystRecommendations| hold_recommendations            | numeric     | no         | (int ≥ 0)                                                         | Count of hold ratings                    |
| AnalystRecommendations| avg_price_target                | numeric     | no         | (float > 0)                                                       | Average price target                     |
| AnalystRecommendations| price_target_vs_current         | numeric     | no         | (float, can be negative)                                          | (avg_target - current_price) / current   |
| AnalystRecommendations| consensus_rating                | numeric     | yes        | 1.0–5.0 (1=strong buy, 5=strong sell)                            | LLM-weighted consensus                   |
| AnalystRecommendations| recent_upgrades                 | numeric     | yes        | (int ≥ 0)                                                         | LLM count of recent upgrades             |
| AnalystRecommendations| recent_downgrades               | numeric     | yes        | (int ≥ 0)                                                         | LLM count of recent downgrades           |
| AnalystRecommendations| analyst_conviction_score        | numeric     | yes        | 0.0–1.0                                                           | LLM assessment of analyst confidence     |
| AnalystRecommendations| recommendation_momentum         | categorical | yes        | improving, stable, deteriorating                                  | LLM trend analysis                       |
| AnalystRecommendations| price_target_spread             | numeric     | no         | (float ≥ 0)                                                       | max_target - min_target                  |
| AnalystRecommendations| coverage_breadth                | numeric     | no         | (int ≥ 0)                                                         | Number of unique analysts covering       |
| AnalystRecommendations| synthetic_analyst_summary       | text        | yes        | (≤240 chars)                                                      | LLM summary of analyst sentiment         |
| AnalystRecommendations| cot_explanation_analyst         | text        | yes        | short text                                                        | LLM rationale                            |

| ALL           | llm_red_flag           | text[]      | yes        | (list of strings; free text, for warning/explainability only)                            | LLM extraction from all context          |
| ALL           | llm_green_flag         | text[]      | yes        | (list of strings; free text, for positive explainability only)                           | LLM extraction from all context          |
| ALL           | risk_mentions          | int         | yes        | (int ≥ 0)                                                                                | LLM extraction (risk keyword count)      |

---

**For all categorical and enum fields, values must match those in the 'Categories/Allowed Values' column.  
Numeric fields must use the ranges specified.  
For details, see deep research notes and RNS categories mapping.**
