# Databricks notebook source
# MAGIC %md
# MAGIC # Run Pipeline by Profile
# MAGIC
# MAGIC Minimal, profile-driven runner for Databricks.
# MAGIC Edit `PROJECT_ROOT` and optionally set `PROFILE` and `INPUT_CSV`.

# COMMAND ----------

import sys
from pathlib import Path

# EDIT THIS: Databricks repo/workspace path to the project root
PROJECT_ROOT = "/Workspace/Users/<you>/survey-analysis-app"

# Optional overrides (leave empty to use config/pipeline_settings.yaml)
PROFILE = ""  # e.g. "pbt"
INPUT_CSV = ""  # e.g. "/Workspace/Users/<you>/survey-analysis-app/Data/pbt.csv"

# Add src to module path
src_dir = Path(PROJECT_ROOT) / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Settings

# COMMAND ----------

from survey_app.config_runtime import load_settings, apply_profile
from survey_app.pipeline import run_pipeline

settings_path = str(Path(PROJECT_ROOT) / "config/pipeline_settings.yaml")
settings = load_settings(settings_path)

if PROFILE and PROFILE != settings.get("profile"):
    settings = apply_profile(settings, PROFILE)

if INPUT_CSV:
    settings.setdefault("paths", {})["input_csv"] = INPUT_CSV

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Pipeline

# COMMAND ----------

results = run_pipeline(settings=settings, settings_path=settings_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick sanity checks

# COMMAND ----------

print("Rows:", len(results["data"]))
print("Assignments:", len(results["assignments"]))
results["assignments"].head(10)
