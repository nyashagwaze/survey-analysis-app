# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline Runner (Minimal)
# MAGIC
# MAGIC Minimal run test
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies
# MAGIC
# MAGIC /Workspace/Users/ngwaze@anglianwater.co.uk/survey-analysis-app/docs/dependencies.md for details

# COMMAND ----------

# DBTITLE 1,Cell 2
import sys
from pathlib import Path
import pandas as pd

# EDIT THIS: Databricks repo/workspace path to the project root
PROJECT_ROOT = "/Workspace/Users/ngwaze@anglianwater.co.uk/survey-analysis-app"

# Add src to module path
src_dir = Path(PROJECT_ROOT) / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from survey_app.pipeline import run_pipeline
from survey_app.config_runtime import load_settings

# COMMAND ----------

# DBTITLE 1,Import Pipeline Modules


# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC ### Full settings tweaks at
# MAGIC  (/Workspace/Users/ngwaze@anglianwater.co.uk/survey-analysis-app/config/pipeline_settings.yaml) 

# COMMAND ----------

# DBTITLE 1,Configuration
settings_path = f"{PROJECT_ROOT}/config/pipeline_settings.yaml"
settings = load_settings(settings_path)

# Optional overrides
input_csv = "/Workspace/Users/ngwaze@anglianwater.co.uk/survey-analysis-app/Data/survey.csv"
output_dir =  f"{PROJECT_ROOT}/outputs/tables"
deliverables_dir = f"{PROJECT_ROOT}/Deliverables"

# Taxonomy options
# Use "keyword" or "semantic". Semantic requires enriched_json_path.
taxonomy_mode = "keyword"
profile = "general"
enriched_json_path =  f"{PROJECT_ROOT}/assets/taxonomy/{profile}/theme_subtheme_dictionary_v3_enriched.json"

# Analytics on/off
analytics = True

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Pipeline

# COMMAND ----------

results = run_pipeline(
    settings=settings,
    input_csv=input_csv,
    taxonomy_mode=taxonomy_mode,
    enriched_json_path=enriched_json_path,
    output_dir=output_dir,
    deliverables_dir=deliverables_dir,
    analytics=analytics
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick sanity checks

# COMMAND ----------

df = results["data"]
assignments = results["assignments"]

print("Rows:", len(df))
print("Assignments:", len(assignments))

# Show sample
assignments.head(10)
