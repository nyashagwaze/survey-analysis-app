# Databricks notebook source
# MAGIC %md
# MAGIC # Minimal Pipeline Runner
# MAGIC Adjust PROJECT_ROOT if your repo lives elsewhere.

# COMMAND ----------

import sys
from pathlib import Path

PROJECT_ROOT = "/Workspace/Users/ngwaze@anglianwater.co.uk/survey-analysis-app"
src_dir = Path(PROJECT_ROOT) / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from survey_app.pipeline import run_pipeline

settings_path = str(Path(PROJECT_ROOT) / "config/pipeline_settings.yaml")
profile = "general"
enriched_json_path = str(Path(PROJECT_ROOT) / f"assets/taxonomy/{profile}/theme_subtheme_dictionary_v3_enriched.json")

run_pipeline(
    settings_path=settings_path,
    enriched_json_path=enriched_json_path
)
