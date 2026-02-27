# Databricks notebook source
# MAGIC %md
# MAGIC # Unpack Bundle to DBFS
# MAGIC
# MAGIC Upload the zip to DBFS first (for example to `dbfs:/FileStore/survey_bundle.zip`).
# MAGIC Then run this notebook to unpack and use the repo structure.

# COMMAND ----------

import os
import zipfile

ZIP_PATH = "/dbfs/FileStore/survey_bundle.zip"
TARGET_DIR = "/dbfs/FileStore/survey-analysis-app"

if not os.path.exists(ZIP_PATH):
    raise FileNotFoundError(f"Zip not found: {ZIP_PATH}")

os.makedirs(TARGET_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall(TARGET_DIR)

print(f"Unpacked to: {TARGET_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next: Run the pipeline from DBFS
# MAGIC
# MAGIC Use any of your run notebooks, but set:
# MAGIC
# MAGIC `PROJECT_ROOT = "/dbfs/FileStore/survey-analysis-app"`

# COMMAND ----------

import sys
from pathlib import Path

PROJECT_ROOT = "/dbfs/FileStore/survey-analysis-app"
src_dir = Path(PROJECT_ROOT) / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from survey_app.pipeline import run_pipeline

run_pipeline(settings_path="config/pipeline_settings.yaml")
