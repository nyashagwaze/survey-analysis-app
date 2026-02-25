# Taxonomy Matching Module

Three complementary approaches for classifying survey responses into themes:

1. **`keyword_taxonomy.py`** - Rule-based keyword/fuzzy matching
2. **`semantic_taxonomy.py`** - ML-based semantic similarity matching
3. **`cross_encoder_taxonomy.py`** - Semantic retrieval + cross-encoder re-ranking (highest precision)

All can be used independently or for cross-validation.

---

## Directory Structure

```
src/wellbeing_pipeline/taxonomy/
  keyword_taxonomy.py
  semantic_taxonomy.py
  cross_encoder_taxonomy.py
  synthetic_generation/
    plan.py
    enhanced_dictionary.py
    transfer.py
```

**Assets (data):**
```
assets/taxonomy/
  <profile>/
    theme_phrase_library.csv
    theme_subtheme_dictionary_v3_enriched.json
```

---

## Approach 1: Keyword Taxonomy (`keyword_taxonomy.py`)

**Purpose**: Fast, explainable rule-based matching using curated phrase libraries.

**Input**: `assets/taxonomy/<profile>/theme_phrase_library.csv` (2,844 expert-curated phrases)

**Method**:
- Token matching for short text (<= 30 chars)
- Phrase matching for long text
- Fuzzy matching for typos
- Returns top-k matches above confidence threshold

---

## Approach 2: Semantic Taxonomy (`semantic_taxonomy.py`)

**Purpose**: Robust ML-based matching for varied natural language.

**Input**: `assets/taxonomy/<profile>/theme_subtheme_dictionary_v3_enriched.json`

**Method**:
- Pre-encodes all phrases with `all-MiniLM-L6-v2`
- Cosine similarity between response and phrases
- Sentiment boost when polarity aligns
- Returns top-k matches above threshold

---

## Approach 3: Semantic + Cross-Encoder (`cross_encoder_taxonomy.py`)

**Purpose**: Highest-accuracy theme matching by re-ranking semantic candidates with a cross-encoder.

**Method**:
1. Bi-encoder retrieves top-N candidate themes
2. Cross-encoder scores response + candidate together
3. Keeps top-k themes/subthemes for final output

---

## Usage

### Keyword Taxonomy
```python
from wellbeing_pipeline.taxonomy.keyword_taxonomy import TaxonomyMatcher

matcher = TaxonomyMatcher(themes_json, fuzzy_threshold=0.78, top_k=3)
matches = matcher.match("I'm overwhelmed with workload", "Wellbeing_Details")
```

### Semantic Taxonomy
```python
from wellbeing_pipeline.taxonomy.semantic_taxonomy import SemanticTaxonomyMatcher

matcher = SemanticTaxonomyMatcher(enriched_json_path, similarity_threshold=0.35, top_k=3)
results = matcher.match_batch(texts, columns, sentiment_labels)
```

### Cross-Encoder (pipeline)
Enable in `config/pipeline_settings.yaml` and run the pipeline:
```yaml
semantic:
  use_cross_encoder: true
```

### Transfer CSV -> Enriched JSON
```bash
python -m wellbeing_pipeline.taxonomy.synthetic_generation.transfer \
  --input-csv assets/taxonomy/<profile>/theme_phrase_library.csv \
  --output-json assets/taxonomy/<profile>/theme_subtheme_dictionary_v3_enriched.json
```
