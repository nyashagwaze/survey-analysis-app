# Pipeline Dependencies

Quick reference for what libraries the pipeline uses, where, and why.

---

## Installation

```python
%pip install sentence-transformers transformers torch pandas numpy scikit-learn spacy joblib openpyxl
dbutils.library.restartPython()

# After restart:
import subprocess, sys
subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
```

Local (pip):
```
pip install -e .
# or: pip install -r requirements.txt
```

---

## Core Dependencies

### sentence-transformers
**What**: Semantic embeddings (all-MiniLM-L6-v2, 384-dim)  
**Why**: Converts text to vectors for similarity matching  
**Where**:
* `semantic_taxonomy.py` - Encodes 8,186 phrases + survey responses
* `embedding_classifier.py` - Encodes synthetic training data
* **Cells 16-17**: Current survey semantic matching
* **Cell 22**: Historical survey semantic matching

---

### transformers
**What**: RoBERTa sentiment model (cardiffnlp/twitter-roberta-base-sentiment-latest)  
**Why**: Analyzes sentiment (positive/negative/neutral)  
**Where**:
* `sentiment_module.py` - Batch sentiment scoring
* **Cell 23**: Historical survey sentiment analysis (2,003 responses)
* **Cell 30**: Privacy-preserving sentiment profiling

---

### torch
**What**: PyTorch backend  
**Why**: Required by sentence-transformers and transformers  
**Where**:
* Backend for all neural network operations
* Enables GPU acceleration (if available)
* Used implicitly by sentence-transformers and transformers

---

### pandas
**What**: DataFrame operations  
**Why**: Primary data structure for all pipeline operations  
**Where**:
* **All modules**: semantic_taxonomy.py, null_text_detector.py, sentiment_module.py
* **All analysis cells**: Loading CSV/Excel, data manipulation, saving results
* **Cells 15, 20, 26**: Loading survey data
* **Cells 17, 22**: Creating assignments DataFrames

---

### numpy
**What**: Numerical operations and arrays  
**Why**: Fast vectorized calculations, embedding normalization  
**Where**:
* `semantic_taxonomy.py` - Cosine similarity calculations, embedding normalization
* `sentiment_module.py` - Array operations for sentiment scores
* **Privacy-preserving profiler** (Cells 28-34): Statistical calculations (mean, median, std)

---

### scikit-learn
**What**: ML utilities (clustering, TF-IDF, classification)  
**Why**: Topic clustering, embedding classifier training  
**Where**:
* `embedding_classifier.py` - Multi-label classifier training
* **Cell 33**: Topic surface analysis (KMeans clustering, TF-IDF vectorization)

---

### spacy
**What**: NLP processing (POS tagging)  
**Why**: Grammatical profiling without exposing text  
**Where**:
* **Cell 29**: POS distribution analysis (nouns, verbs, adjectives, etc.)
* Privacy-preserving profiler - Lexical/grammatical statistics

**Model**: `en_core_web_sm` (English language model, 13MB)

---

### joblib
**What**: Model serialization  
**Why**: Save/load trained embedding classifier  
**Where**:
* `embedding_classifier.py` - Saves trained model
* **Cell 4**: Loads trained classifier from `classifier.joblib`

---

### openpyxl
**What**: Excel file support  
**Why**: Read .xlsx files with pandas  
**Where**:
* **Cell 20/26**: Loading `historic_survey.xlsx` (3,600 responses)
* Required by `pd.read_excel()`

---

## Optional Dependencies

### matplotlib
**What**: Plotting library  
**Why**: Create visualizations  
**Where**:
* **Cell 12**: Comparison visualization (4-panel charts)
* `column_wordclouds.py` - Display wordclouds

**Status**: Only needed for visualizations

---

### seaborn
**What**: Statistical visualization styling  
**Why**: Enhanced plot aesthetics  
**Where**:
* **Cell 12**: Comparison visualization styling

**Status**: Only needed for visualizations

---

### wordcloud
**What**: Word cloud generation  
**Why**: Exploratory text analysis  
**Where**:
* `column_wordclouds.py` - Generate TF-IDF wordclouds

**Status**: Only for exploratory analysis

---

### pyyaml
**What**: YAML file parsing  
**Why**: Load YAML configuration files  
**Where**:
* `config/pipeline_settings.yaml`
* `config/profiles/{profile}/dictionary.yaml`
* `config/profiles/{profile}/themes.yaml`

**Status**: Not currently used (pipeline uses JSON configs)

---

## Dependency Map by Component

### Component 0: Synthetic Data Generation
* **sentence-transformers** - Encode synthetic phrases
* **scikit-learn** - Train classifier
* **joblib** - Save trained model
* **pandas** - Data handling

### Component 1: Enriched Dictionary Generation
* **pandas** - Load CSV, generate JSON
* **numpy** - Array operations

### Component 2: Null Text Detection
* **pandas** - DataFrame operations
* **numpy** - NaN handling

### Component 3: Semantic Taxonomy Matching
* **sentence-transformers** - Encode phrases and responses
* **numpy** - Cosine similarity calculations
* **pandas** - Data handling

### Component 4: Sentiment Analysis
* **transformers** - RoBERTa model
* **torch** - PyTorch backend
* **pandas** - Data handling
* **numpy** - Score calculations

### Privacy-Preserving Profiler
* **pandas** - Data loading and manipulation
* **numpy** - Statistical calculations
* **spacy** - POS tagging
* **scikit-learn** - Clustering and TF-IDF
* **transformers** - Sentiment analysis

---

## Model Downloads (Automatic on First Use)

### all-MiniLM-L6-v2 (~90MB)
* Downloaded by sentence-transformers
* Cached in `/tmp/huggingface_cache/`
* Used for semantic embeddings

### cardiffnlp/twitter-roberta-base-sentiment-latest (~500MB)
* Downloaded by transformers
* Cached in `/tmp/huggingface_cache/`
* Used for sentiment analysis

### en_core_web_sm (~13MB)
* Downloaded by spacy
* Cached in spacy data directory
* Used for POS tagging

**Total download size**: ~600MB (first time only)

---

## Quick Reference

| Library | Size | Purpose | Critical? |
|---------|------|---------|-----------|
| sentence-transformers | ~90MB | Semantic embeddings | ✅ Yes |
| transformers | ~500MB | Sentiment analysis | ✅ Yes |
| torch | ~800MB | Neural network backend | ✅ Yes |
| pandas | ~20MB | Data processing | ✅ Yes |
| numpy | ~15MB | Numerical operations | ✅ Yes |
| scikit-learn | ~30MB | ML utilities | ✅ Yes |
| spacy | ~13MB | POS tagging | 🔧 Optional* |
| joblib | ~1MB | Model serialization | ✅ Yes |
| openpyxl | ~1MB | Excel support | ✅ Yes |
| matplotlib | ~10MB | Visualizations | 🔧 Optional |
| seaborn | ~1MB | Plot styling | 🔧 Optional |
| wordcloud | ~1MB | Word clouds | 🔧 Optional |

*Required only for privacy-preserving profiler

**Total required**: ~1.5GB (including models)

---

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'sentence_transformers'`  
**Fix**: Run installation cell and restart Python

**Issue**: `OSError: [E050] Can't find model 'en_core_web_sm'`  
**Fix**: Download spaCy model: `python -m spacy download en_core_web_sm`

**Issue**: `ImportError: cannot import name 'SentenceTransformer'`  
**Fix**: Restart Python after installation: `dbutils.library.restartPython()`

**Issue**: Models downloading slowly  
**Note**: First-time download is ~600MB, takes 2-5 minutes depending on connection
