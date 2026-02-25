# sentiment_module.py (Core Sentiment Logic)
## Purpose: Core sentiment analysis library with reusable functions

### Key Features:

RoBERTa transformer model: cardiffnlp/twitter-roberta-base-sentiment for nuanced sentiment
Clause-aware analysis: Splits text on contrast markers ("but", "however") and weights later clauses more heavily
Coping detection: 30+ patterns (exercise, social support, mindfulness, etc.) with negation handling
Lazy model loading: Loads RoBERTa model only on first use
Core functions:
roberta_compound() - Batch sentiment scoring
clause_aware_compound() - Contrast-aware sentiment
detect_coping() - Coping mechanism detection
label_from_compound() - Convert scores to labels
aggregate_sentiment_by_id() - Column-weighted aggregation
Processing: Single-threaded, works with pandas DataFrames

### Use case: Standalone sentiment analysis or called by other modules

pyspark_sentiment.py (PySpark Wrapper)
Purpose: Parallel distributed sentiment processing for large datasets

### Key Features:

PySpark Pandas UDF: Distributes sentiment analysis across Spark cluster
10-100x faster than pandas .iterrows() for large datasets
Batch processing: Processes rows in batches across executors
Imports sentiment_module.py: Calls weighted_sentiment_for_row() function
Null text integration: Respects _is_meaningful flags from null_text_detector
Column-aware weighting: 50%/35%/15% weights for Wellbeing_Details/Areas_Improve/Support_Provided
Processing: Distributed parallel processing on Spark cluster

### Use case: Production pipeline for processing 625+ survey responses efficiently

### Relationship
pyspark_sentiment.py (Spark wrapper)
    └── calls → sentiment_module.py (core logic)
                    └── uses → RoBERTa transformer model


### Summary:
sentiment_module.py = The engine (core sentiment logic)
pyspark_sentiment.py = The turbocharger (parallel distributed processing)