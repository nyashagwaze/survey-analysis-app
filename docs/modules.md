# Module README

This project is split into small modules that you import into a Databricks notebook or a local script. Below is a quick reference for what each module provides and how it is used by the pipeline.

## clean_normalise.py
Purpose: text preprocessing and normalization used before taxonomy and sentiment.

Key functions:
- basic_normalise(text) -> str
  Normalizes text, handles nulls, lowercases, removes punctuation/extra whitespace.
- apply_business_map(text, business_map) -> str
  Replaces business-specific terms (from config/profiles/{profile}/dictionary.yaml).
- merge_to_canonical(text, merge_map) -> str
  Maps synonyms to canonical forms.
- force_phrases(text, forced) -> str
  Keeps multi-word phrases together with underscores.
- keep_unigrams(text, unigram_whitelist, always_drop, must_be_phrase) -> str
  Token filtering rules.
- load_dictionary_config(path) -> dict
  Loads config/profiles/{profile}/dictionary.yaml into usable maps and sets.
- preprocess_text(text, assets, settings) -> str
  End-to-end preprocessing for one text field.
- preprocess_dataframe(df, text_columns, assets, settings) -> DataFrame
  Adds <col>_processed columns for all text fields.

Used by:
- run_pipeline() to create processed text columns before null detection and taxonomy.

## null_text_detector.py
Purpose: gate non-meaningful responses before taxonomy and sentiment.

Key functions:
- is_null_text(text, min_meaningful_length=3, max_dismissive_length=50) -> bool
- classify_response_detail(text, ...) -> str
  Returns "Detailed response" or a dismissive label ("Yes", "No", "No sentiment").
- add_response_quality_flags(df, text_columns, ...) -> DataFrame
  Adds <col>_is_meaningful and <col>_response_detail.
- get_response_quality_report(df, text_columns, ...) -> DataFrame

Used by:
- run_pipeline() before taxonomy/sentiment.
- taxonomy assigners for dismissed-row handling.

Null text outputs:
- null_text_report.csv: counts per response_detail
- null_text_details.csv: row-level dismissed text + reason

## semantic_taxonomy.py
Purpose: semantic matching against an enriched phrase library using sentence transformers.

Key classes/functions:
- SemanticTaxonomyMatcher(enriched_json_path, model_name, similarity_threshold, top_k, sentiment_weight)
  Pre-encodes phrases and returns top-k semantic matches.
- assign_taxonomy_semantic(df_or_spark_df, text_columns, enriched_json_path, ...) -> DataFrame
  Batch semantic assignment using the matcher.

Used by:
- run_pipeline() when taxonomy_mode = "semantic".

Outputs:
- taxonomy_matched.csv (matched rows + evidence)
- taxonomy_unmatched.csv (unmatched/dismissed rows + reason)

## keyword_taxonomy.py
Purpose: keyword/fuzzy matching against config/profiles/{profile}/themes.yaml (classic taxonomy).

Key classes/functions:
- MatchResult dataclass
- TaxonomyMatcher(themes_json, use_fuzzy, min_accept_score, ...)
  Keyword/fuzzy matcher with column-aware scoring.
- assign_taxonomy_keyword(df, text_columns, themes, taxonomy_cfg, output_cfg) -> DataFrame
  High-level keyword taxonomy assignment.

Used by:
- run_pipeline() when taxonomy_mode = "keyword".

Outputs:
- taxonomy_matched.csv (matched rows + evidence)
- taxonomy_unmatched.csv (unmatched/dismissed rows + reason)

## sentiment_module.py
Purpose: RoBERTa sentiment scoring and coping detection.

Key functions:
- roberta_probs(texts) -> ndarray
- roberta_compound(texts) -> ndarray
- split_on_contrast(text) -> list[str]
- clause_aware_compound(text) -> float
- label_from_compound(score, pos_thresh=0.05, neg_thresh=-0.05) -> str
- detect_coping(text) -> bool
- weighted_sentiment_for_row(row, text_columns, ...) -> dict
- add_sentiment_columns_pandas(df, text_columns, ...) -> DataFrame
- aggregate_sentiment_by_id(df, text_columns, weights=None) -> DataFrame

Used by:
- run_pipeline() for pandas sentiment.
- pyspark_sentiment.py UDFs (if Spark is used).

## pyspark_sentiment.py (optional)
Purpose: Spark UDF sentiment scoring for large datasets.

Key functions:
- add_sentiment_columns(df_spark, text_columns=...) -> Spark DataFrame
  Adds compound, sentiment_label, coping_flag, and per-column compounds.

Notes:
- Only available if pyspark is installed. Import guarded.

## pipeline runner (pipeline integration)
Purpose: top-level orchestration.

Key functions:
- load_settings(settings_path="config/pipeline_settings.yaml", validate=True, apply_defaults=True)
  Loads config with defaults merged and validates required keys.
- run_pipeline(settings_path, input_csv, themes_path, dictionary_path, enriched_json_path, taxonomy_mode, output_dir, deliverables_dir)
  End-to-end flow: preprocess -> null detect -> sentiment -> taxonomy -> outputs.

CLI:
- wellbeing-pipeline --help
- wellbeing-pipeline --input <csv> --taxonomy-mode keyword
- wellbeing-pipeline --taxonomy-mode semantic --enriched-json <json>

## scripts/compare_taxonomy_outputs.py
Purpose: compare keyword vs semantic assignments and produce a mismatch report.

Usage:
- python scripts/compare_taxonomy_outputs.py
- python scripts/compare_taxonomy_outputs.py --use-deliverables
- python scripts/compare_taxonomy_outputs.py --keyword <path> --semantic <path> --output <csv>

## Common import pattern (Databricks)
Use the bootstrap helper:
- import_bootstrap.py
  Edit PROJECT_ROOT, then import modules from there.

## Config files used by modules
- config/pipeline_settings.yaml
  Central settings for null detection, taxonomy, sentiment, preprocessing, outputs.
- config/profiles/{profile}/dictionary.yaml
  Business maps, whitelists, forced phrases, and canonical merges.
- config/profiles/{profile}/themes.yaml
  Keyword taxonomy definitions and columns/polarity.
