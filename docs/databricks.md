# Databricks Guide

Consolidated guidance for running the Survey Analysis App pipeline on Databricks.

## Quick Start (Notebook)
If you are not installing the package on the cluster, add `src/` to `sys.path`:
```python
import sys
from pathlib import Path

PROJECT_ROOT = "/Workspace/Users/<you>/survey-analysis-app"
src_dir = Path(PROJECT_ROOT) / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from survey_app.pipeline import run_pipeline
run_pipeline(settings_path="config/pipeline_settings.yaml")
```

Alternative: use the bootstrap helper:
```python
from survey_app.import_bootstrap import bootstrap
bootstrap("/Workspace/Users/<you>/survey-analysis-app")

from survey_app.pipeline import run_pipeline
run_pipeline(settings_path="config/pipeline_settings.yaml")
```

## Paths and Filesystems
- `/Workspace/...` is driver-only and is not directly readable by Spark.
- Spark can read:
  - DBFS paths like `dbfs:/...` or `/dbfs/...`.
  - Unity Catalog Volumes like `/Volumes/<catalog>/<schema>/<volume>/...`.
  - Cloud storage (ADLS/S3/GCS).
  - Delta tables via `spark.read.table(...)`.
- `/Workspace` is for notebooks, code, and configs. Data should live in DBFS or Volumes.

## YAML Toggles (Single Source of Truth)
Set these in `config/pipeline_settings.yaml`:
```yaml
databricks:
  enabled: auto
  project_root: "/Workspace/Users/{user}/survey-analysis-app"
  hf_cache: "/Workspace/Users/{user}/.cache/huggingface"
  set_hf_cache_env: true
  override_env: true
  spark_copy_workspace_to_dbfs: false
  spark_dbfs_dir: "dbfs:/tmp/survey_app"

performance:
  use_pyspark: false
```

## Workspace to DBFS Helper (Spark)
If `performance.use_pyspark: true` and the input path starts with `/Workspace/`, enable auto-copy:
```yaml
databricks:
  spark_copy_workspace_to_dbfs: true
  spark_dbfs_dir: "dbfs:/tmp/survey_app"
```
The helper lives in `src/survey_app/databricks_helpers.py` and is called by `python -m survey_app.run`.

## Cache and Model Downloads
- Cluster shared cache locations are not writable (for example `/root/.cache`).
- Use a user workspace cache like `/Workspace/Users/<you>/.cache/huggingface`.
- The pipeline can set `HF_HOME`, `TRANSFORMERS_CACHE`, and `SENTENCE_TRANSFORMERS_HOME` via YAML.

## Notebook Caveats
- `__file__` does not exist in notebooks; scripts can use it.
- Notebooks are not importable modules. Use `%run` for notebooks.
- Relative imports often fail; prefer absolute imports or `sys.path` injection.
- Notebooks run in a shared namespace; use widgets for parameters.
- `%pip` installs are notebook-scoped unless attached as cluster libraries.
- Use `dbutils.secrets.get()` for secrets instead of environment variables.
- Use `display(df)` for rich output and `print()` for logs.

## Spark Handling (Essentials)
- Spark is lazy; only actions trigger execution (`count`, `show`, `collect`, `write`).
- Prefer built-in Spark functions over Python UDFs.
- Broadcast small lookup tables to reduce shuffles.
- Partitioning matters; use `repartition` or `coalesce` strategically.
- Cache carefully and `unpersist()` when done.
- Avoid `.collect()` or `.toPandas()` on large datasets (OOM risk).
- Enforce schemas and check `printSchema()`.
- Be explicit about null handling (`na.fill`, `na.drop`).
- Use `pyspark.sql.functions` for string/date ops.
- Shuffles are expensive; minimize joins and groupBy where possible.

## Spark and Pandas Conversions
- Spark to Pandas: `limit()` or `sample()` before `.toPandas()`.
- Pandas to Spark: pass an explicit schema to `spark.createDataFrame()`.
- Pandas index is dropped; call `reset_index()` if needed.
- Sanitize column names (spaces and special chars).
- Watch datetime, timezone, and nullable integer types.

## Storage Best Practices
- Prefer Unity Catalog Volumes for durable storage.
- Prefer Delta tables for large or frequently queried datasets.
- Avoid local file I/O on executors (ephemeral storage).

## Base Directory Options
1. Use notebook path discovery (recommended):
   `dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath()`
2. Use environment variables to hold the project root.
3. Use Databricks widgets to pass the project root.
4. Hardcode the project root in config (simple and reliable).
5. Remove base_dir entirely and use absolute paths in config.

## Common Pitfalls
- Column name mismatches (extra spaces or variants).
- Wrong CSV delimiter.
- Using Spark with `/Workspace` input paths.
- Missing `enriched_json` path for semantic mode.
- Not setting cache env vars for Transformers.
