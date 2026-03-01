# Research Dissertation - Survey Analysis App

Author: Nyasha Gwaze  
Project: Survey Analysis App  
Date: 2026-02-28  

## Abstract
This document describes the research and engineering decisions behind the Survey Analysis App. It explains what was built, why each approach was chosen, and the evidence used to validate those choices. The goal is to provide a transparent, reproducible, and defensible methodology for analyzing open-text survey responses using taxonomy matching and sentiment analysis.

## 1. Introduction
Free-text survey responses contain rich qualitative insights but are difficult to analyze at scale. This project transforms free-text answers into structured themes, subthemes, and sentiment signals using a combination of taxonomy-based matching and semantic similarity. The solution is designed to be practical, privacy-preserving, and easy to use for non-technical users.

## 2. Problem Statement
Organizations need to analyze open-ended survey responses consistently and at scale. Manual coding is slow and inconsistent, while keyword rules alone are brittle and fail on natural language variation. This project addresses those gaps with a general-purpose, configurable pipeline and a Streamlit interface.

## 3. Objectives
- Provide a repeatable workflow for categorizing free-text survey responses.
- Support both keyword and semantic taxonomy matching.
- Provide sentiment analysis to enrich insights.
- Allow non-technical users to upload data and run analysis.
- Maintain privacy by avoiding exposure of raw text where possible.

## 4. Research Questions
- RQ1: Can taxonomy-based semantic matching outperform keyword-only matching on real survey text?
- RQ2: What balance of transparency and performance is acceptable for theme assignment?
- RQ3: Can the system be generalized for different survey types without code changes?

## 5. Scope and Constraints
- Data types: English-language survey responses in CSV format.
- Compute: Local workstation or Databricks compatible workflow.
- Privacy: Avoid exposing raw text in outputs unless explicitly requested.
- Interpretability: Provide matched themes and supporting phrases to enable review.

## 6. Methodology Overview
The methodology is composed of five stages:
1. Input ingestion and column mapping
2. Text normalization and null-response detection
3. Taxonomy matching (keyword or semantic)
4. Sentiment analysis (optional)
5. Output generation and reporting

## 7. Taxonomy Design
Taxonomies are defined in `theme_phrase_library.csv` using a simple schema:
`column, parent, theme, subtheme, polarity, phrase`.

From this CSV we generate:
- `theme_subtheme_dictionary_v3_enriched.json` for semantic matching
- `themes.yaml` for keyword matching

Rationale:
- CSV is accessible to non-developers.
- Enriched JSON enables efficient semantic lookups.
- YAML keeps keyword matching transparent and easy to edit.

Evidence to include:
- Validation logs from `scripts/build_taxonomy.py`
- Example taxonomy files used in real runs

## 8. Matching Approaches

### 8.1 Keyword Matching
Keyword matching is used when transparency and deterministic matching are preferred. It supports fuzzy matching and column-aware boosts.

Why this choice:
- Easy to audit
- Fast to run
- Works well when phrase libraries are complete

Evidence to include:
- Matched and unmatched reports from keyword runs
- Manual review of false positives and false negatives

### 8.2 Semantic Matching
Semantic matching uses sentence embeddings to compare survey responses with the taxonomy phrases.

Why this choice:
- Handles varied phrasing and typos
- Captures semantic similarity beyond exact words
- Produces confidence scores

Evidence to include:
- Match rates and score distributions from semantic runs
- Examples where semantic succeeds and keyword fails

## 9. Sentiment Analysis
Sentiment analysis is optional and provides a numeric score and label per response. It can be weighted by column importance.

Why this choice:
- Adds emotional signal to thematic categorization
- Useful for prioritizing themes by impact

Evidence to include:
- Sentiment distribution plots
- Comparison of sentiment across themes

## 10. Evaluation Strategy
Evaluation is based on a combination of automated metrics and manual review:
- Match rate: % of responses assigned a theme above threshold
- Coverage: proportion of unique themes used
- Confidence distribution: similarity scores
- Manual verification: sampled responses reviewed by a human

Evidence sources:
- `assignments_*.csv`
- `taxonomy_matched.csv` / `taxonomy_unmatched.csv`
- Review notes or annotated samples

## 11. Results Summary (Fill With Latest Run)
Add results from the latest runs here.

Example format:
- Dataset: [name]
- Rows: [count]
- Matching mode: keyword / semantic
- Match rate: [percent]
- Average similarity: [value]
- No match rate: [percent]

## 12. Discussion
Interpret results relative to the research questions.
- Does semantic matching improve coverage and accuracy?
- Are there themes that remain underrepresented?
- What types of responses are still missed?

## 13. Limitations
- Taxonomy quality depends on phrase library coverage.
- Semantic similarity can produce false positives at low thresholds.
- Sentiment models are trained on general data and may not capture domain nuance.

## 14. Future Work
- Add domain-specific sentiment calibration
- Improve taxonomy authoring UI
- Add automated evaluation against labeled gold sets

## 15. Evidence Register
Use this table to list evidence artifacts and where they live.

| Claim | Evidence Artifact | Location |
| --- | --- | --- |
| Semantic matching improves coverage | Match rate summary | outputs/... |
| Keyword matching is transparent | Matched/unmatched reports | outputs/... |
| Sentiment adds signal | Sentiment distribution charts | outputs/... |

## 16. Reproducibility
Use these commands to reproduce the main workflow.

Build taxonomy:
```bash
python scripts/build_taxonomy.py \
  --input-csv assets/taxonomy/<profile>/theme_phrase_library.csv \
  --output-json assets/taxonomy/<profile>/theme_subtheme_dictionary_v3_enriched.json \
  --output-themes config/profiles/<profile>/themes.yaml
```

Run pipeline (semantic):
```bash
survey-app --input Data/survey.csv --taxonomy-mode semantic
```

Run pipeline (keyword):
```bash
survey-app --input Data/survey.csv --taxonomy-mode keyword
```

## 17. Related Documentation
- `Methedology/Comprehensive_description`
- `docs/pipeline.md`
- `docs/dependencies.md`
