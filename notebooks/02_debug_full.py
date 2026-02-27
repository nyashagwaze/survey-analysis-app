# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline Debug Notebook (Full)
# MAGIC
# MAGIC Debug for each module
# MAGIC

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install sentence-transformers transformers torch pandas numpy scikit-learn spacy joblib openpyxl pyyaml

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

import sys
from pathlib import Path
import pandas as pd

# EDIT THIS: Databricks repo/workspace path to the project root
PROJECT_ROOT = "/Workspace/Users/ngwaze@anglianwater.co.uk/survey-analysis-app"

# Add src to module path
src_dir = Path(PROJECT_ROOT) / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from survey_app.config_runtime import (
    load_settings,
    resolve_path,
    load_dictionary_config,
    preprocess_dataframe,
    data_audit_summary,
    build_null_text_details,
    generate_taxonomy_reports,
)
from survey_app.clean_normalise.null_text_detector import add_response_quality_flags, get_response_quality_report
from survey_app.sentiment.sentiment_module import add_sentiment_columns_pandas, DEFAULT_COLUMN_WEIGHTS
from survey_app.taxonomy.semantic_taxonomy import assign_taxonomy_semantic
from survey_app.taxonomy.keyword_taxonomy import assign_taxonomy_keyword

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC ### Full settings tweaks at
# MAGIC  (/Workspace/Users/ngwaze@anglianwater.co.uk/survey-analysis-app/config/pipeline_settings.yaml) 

# COMMAND ----------

settings_path = f"{PROJECT_ROOT}/config/pipeline_settings.yaml"
settings = load_settings(settings_path)
paths_cfg = settings.get("paths", {})
base_dir = resolve_path(Path(PROJECT_ROOT), paths_cfg.get("base_dir", "."))
text_columns = settings.get("text_columns", [])

input_csv = paths_cfg.get("input_csv")
input_path = resolve_path(base_dir, input_csv)

dictionary_path = resolve_path(base_dir, paths_cfg.get("dictionary"))
themes_path = resolve_path(base_dir, paths_cfg.get("themes"))

output_tables = resolve_path(base_dir, paths_cfg.get("output_tables", "outputs/tables"))
output_tables.mkdir(parents=True, exist_ok=True)

print("input:", input_path)
print("dictionary:", dictionary_path)
print("themes:", themes_path)
print("output:", output_tables)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

df = pd.read_csv(input_path)
print(df.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Audit (basic)

# COMMAND ----------

audit = data_audit_summary(df, text_columns)
print(audit)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocess Text

# COMMAND ----------

dict_assets = load_dictionary_config(dictionary_path)
df = preprocess_dataframe(df, text_columns, dict_assets, settings)

df[[c for c in df.columns if c.endswith("_processed")]].head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Null Text Detection

# COMMAND ----------

null_cfg = settings.get("null_detection", {})
df = add_response_quality_flags(
    df,
    text_columns,
    min_meaningful_length=null_cfg.get("min_meaningful_length", 3),
    max_dismissive_length=null_cfg.get("max_dismissive_length", 50)
)

null_report = get_response_quality_report(df, text_columns)
null_report

# Optional: detailed nulls
null_details = build_null_text_details(df, text_columns)
null_details.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sentiment (pandas)

# COMMAND ----------

sent_cfg = settings.get("sentiment", {})

df = add_sentiment_columns_pandas(
    df,
    text_columns=text_columns,
    weights=sent_cfg.get("column_weights", DEFAULT_COLUMN_WEIGHTS),
    pos=sent_cfg.get("positive_threshold", 0.05),
    neg=sent_cfg.get("negative_threshold", -0.05),
    skip_dismissed=sent_cfg.get("skip_dismissed", True),
    dismissed_sentiment_value=sent_cfg.get("dismissed_sentiment_value", None)
)

df[["compound", "sentiment_label", "coping_flag"]].head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Keyword Taxonomy

# COMMAND ----------

themes = load_settings(settings_path).get("themes", None)
if themes is None:
    # fallback: load directly from YAML
    import yaml
    with open(themes_path, "r", encoding="utf-8") as f:
        themes = yaml.safe_load(f)

assignments_keyword = assign_taxonomy_keyword(
    df,
    text_columns=text_columns,
    themes=themes,
    taxonomy_cfg=settings.get("taxonomy", {}),
    output_cfg=settings.get("output", {})
)

assignments_keyword.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Semantic Taxonomy (optional)

# COMMAND ----------

# DBTITLE 1,Semantic Taxonomy
import os

# Use user-specific cache directory (not shared /tmp)
cache_dir = "/Workspace/Users/ngwaze@anglianwater.co.uk/.cache/huggingface"
os.makedirs(cache_dir, exist_ok=True)

os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir

profile = "general"
enriched_json_path = f"{PROJECT_ROOT}/assets/taxonomy/{profile}/theme_subtheme_dictionary_v3_enriched.json"

assignments_semantic = assign_taxonomy_semantic(
    df,
    text_columns=text_columns,
    enriched_json_path=Path(enriched_json_path),
    model_name="all-MiniLM-L6-v2",
    similarity_threshold=0.35,
    top_k=3
)
assignments_semantic.head(10)

# COMMAND ----------

# DBTITLE 1,Compare Taxonomy Outputs
# Compare keyword vs semantic taxonomy outputs

# Ensure both DataFrames have ID column
if 'ID' not in assignments_keyword.columns:
    assignments_keyword['ID'] = assignments_keyword.index
if 'ID' not in assignments_semantic.columns:
    assignments_semantic['ID'] = assignments_semantic.index

# Ensure required columns exist
for col in ["theme", "subtheme", "parent_theme", "match_method", "evidence", "reason"]:
    if col not in assignments_keyword.columns:
        assignments_keyword[col] = ""
    if col not in assignments_semantic.columns:
        assignments_semantic[col] = ""

# Merge on ID and TextColumn
key_cols = ["ID", "TextColumn"]
merged = assignments_keyword.merge(
    assignments_semantic, 
    on=key_cols, 
    how="outer", 
    suffixes=("_keyword", "_semantic")
)

# Fill NaN values
merged["theme_keyword"] = merged["theme_keyword"].fillna("")
merged["theme_semantic"] = merged["theme_semantic"].fillna("")
merged["subtheme_keyword"] = merged["subtheme_keyword"].fillna("")
merged["subtheme_semantic"] = merged["subtheme_semantic"].fillna("")

# Check if theme and subtheme match
merged["match"] = (
    (merged["theme_keyword"] == merged["theme_semantic"]) & 
    (merged["subtheme_keyword"] == merged["subtheme_semantic"])
)

# Summary statistics
total = len(merged)
matches = int(merged["match"].sum())
mismatches = total - matches

print("="*60)
print("TAXONOMY COMPARISON: Keyword vs Semantic")
print("="*60)
print(f"Total rows:    {total}")
print(f"Matches:       {matches} ({100*matches/total:.1f}%)")
print(f"Mismatches:    {mismatches} ({100*mismatches/total:.1f}%)")
print("="*60)

# Show mismatches
mismatched_rows = merged[~merged["match"]]
if len(mismatched_rows) > 0:
    print(f"\nShowing first 10 mismatches:")
    display(mismatched_rows[["ID", "TextColumn", "theme_keyword", "theme_semantic", 
                             "subtheme_keyword", "subtheme_semantic"]].head(10))
    
    # Save mismatches
    mismatch_path = output_tables / "taxonomy_comparison_mismatches.csv"
    mismatched_rows.to_csv(mismatch_path, index=False)
    print(f"\nFull mismatch report saved to: {mismatch_path}")
else:
    print("\n✅ Perfect match! All assignments agree.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Reports

# COMMAND ----------

output_cfg = settings.get("output", {})

# Keyword reports
if output_cfg.get("generate_taxonomy_reports", True):
    generate_taxonomy_reports(
        assignments=assignments_keyword,
        output_path=output_tables,
        deliverables_path=None,
        taxonomy_mode="keyword",
        matched_filename=output_cfg.get("taxonomy_matched_filename"),
        unmatched_filename=output_cfg.get("taxonomy_unmatched_filename")
    )

# Optional: save outputs
assignments_keyword.to_csv(output_tables / "assignments_keyword.csv", index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick Checks

# COMMAND ----------

print("Keyword assignments:", len(assignments_keyword))
print("Matched:", (assignments_keyword["reason"] == "matched").sum())
print("Unmatched:", (assignments_keyword["reason"] != "matched").sum())
