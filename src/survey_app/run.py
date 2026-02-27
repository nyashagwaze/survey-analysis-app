"""
Main Pipeline Orchestrator (Databricks/Spark optional)
Runs the complete NLP pipeline: Clean -> Taxonomy -> Sentiment
"""

from pathlib import Path

from .config_runtime import load_settings, resolve_base_dir, resolve_path, resolve_processing_columns
from .databricks_helpers import copy_workspace_to_dbfs
from .clean_normalise.null_text_detector import add_response_quality_flags
from .taxonomy.semantic_taxonomy import assign_taxonomy_semantic
from .taxonomy.cross_encoder_taxonomy import assign_taxonomy_semantic_cross_encoder

try:
    from pyspark.sql import SparkSession
    from .sentiment.pyspark_sentiment import add_sentiment_columns
    _HAS_SPARK = True
except Exception:
    _HAS_SPARK = False


def _looks_like_uri(path: str) -> bool:
    if not path:
        return False
    p = str(path)
    return "://" in p or p.startswith("dbfs:")


def _resolve_input_path(base_dir: Path, input_csv: str):
    if input_csv is None:
        return None
    if _looks_like_uri(input_csv):
        return input_csv
    p = resolve_path(base_dir, input_csv)
    return str(p)


def _apply_column_map_spark(df, column_map: dict):
    if not column_map:
        return df
    for src, dest in column_map.items():
        if src in df.columns and dest and src != dest:
            df = df.withColumnRenamed(src, dest)
    return df


def run_pipeline():
    package_dir = Path(__file__).resolve().parent
    settings_path = "config/pipeline_settings.yaml"
    settings = load_settings(settings_path)
    paths_cfg = settings.get("paths", {})
    base_dir = resolve_base_dir(settings, settings_path=settings_path, pipeline_dir=package_dir)
    input_cfg = settings.get("input", {}) or {}
    columns_cfg = resolve_processing_columns(settings)
    text_columns = columns_cfg.get("text_columns") or []
    taxonomy_columns = columns_cfg.get("taxonomy_columns") or text_columns
    sentiment_columns = columns_cfg.get("sentiment_columns") or text_columns

    if not text_columns:
        raise ValueError("No text columns configured. Set processing.text_columns or text_columns in settings.")

    input_csv = paths_cfg.get("input_csv")
    if not input_csv:
        raise ValueError("paths.input_csv is required in pipeline_settings.yaml")

    input_path = _resolve_input_path(base_dir, input_csv)

    output_path = paths_cfg.get("output_tables", "outputs/tables")
    output_path = resolve_path(base_dir, output_path)

    enriched_json = paths_cfg.get("enriched_json")
    enriched_json = resolve_path(base_dir, enriched_json) if enriched_json else None

    semantic_cfg = settings.get("semantic", {})
    semantic_model = semantic_cfg.get("model_name", "all-MiniLM-L6-v2")
    try:
        model_path = resolve_path(base_dir, semantic_model)
        if model_path and model_path.exists():
            semantic_model = str(model_path)
    except Exception:
        pass

    similarity_threshold = semantic_cfg.get("similarity_threshold", settings["taxonomy"]["min_accept_score"])
    top_k = semantic_cfg.get("top_k", settings["taxonomy"]["top_k"])
    use_cross_encoder = bool(semantic_cfg.get("use_cross_encoder", False))

    use_pyspark = settings.get("performance", {}).get("use_pyspark", False)
    db_cfg = settings.get("databricks", {}) or {}
    if use_pyspark and not _HAS_SPARK:
        raise ImportError("pyspark not available. Set performance.use_pyspark=false or install pyspark.")

    if use_pyspark:
        if str(input_path).startswith("/Workspace/"):
            if db_cfg.get("spark_copy_workspace_to_dbfs", False):
                dbfs_dir = db_cfg.get("spark_dbfs_dir", "dbfs:/tmp/survey_app")
                input_path = copy_workspace_to_dbfs(str(input_path), dbfs_dir, overwrite=True)
            else:
                raise ValueError(
                    "Spark cannot read /Workspace paths. "
                    "Enable databricks.spark_copy_workspace_to_dbfs or set performance.use_pyspark=false."
                )

        spark = SparkSession.builder.getOrCreate()

        print("=" * 80)
        print("SURVEY NLP PIPELINE (SPARK)")
        print("=" * 80)

        print("\nLoading data...")
        df = spark.read.csv(input_path, header=True, inferSchema=True)
        df = _apply_column_map_spark(df, input_cfg.get("column_map", {}))
        print(f"   Loaded {df.count()} responses")

        print("\nDetecting null/dismissive text...")
        df = add_response_quality_flags(df, text_columns)

        print("\nAssigning themes...")
        if not enriched_json:
            raise ValueError("paths.enriched_json is required for semantic taxonomy.")
        if not taxonomy_columns:
            raise ValueError("No taxonomy columns configured. Set processing.taxonomy_columns or text_columns.")
        if use_cross_encoder:
            scores_output = semantic_cfg.get("scores_output")
            scores_output = resolve_path(base_dir, scores_output) if scores_output else None
            if semantic_cfg.get("assignments_filename"):
                settings.setdefault("output", {})["assignments_filename"] = semantic_cfg.get("assignments_filename")
            df = assign_taxonomy_semantic_cross_encoder(
                df_spark=df,
                text_columns=taxonomy_columns,
                enriched_json_path=enriched_json,
                model_name=semantic_model,
                cross_encoder_model=semantic_cfg.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                bi_top_k=int(semantic_cfg.get("bi_top_k", 15)),
                bi_threshold=float(semantic_cfg.get("bi_threshold", 0.20)),
                top_k=int(top_k),
                pair_template=semantic_cfg.get("pair_template", "{phrase}"),
                scores_output=scores_output,
                candidates_output=None,
                delimiter=settings.get("taxonomy", {}).get("multi_label", {}).get("delimiter", " | ")
            )
        else:
            df = assign_taxonomy_semantic(
                df,
                taxonomy_columns,
                enriched_json,
                model_name=semantic_model,
                similarity_threshold=similarity_threshold,
                top_k=top_k
            )

        print("\nAnalyzing sentiment...")
        if sentiment_columns:
            df = add_sentiment_columns(df, sentiment_columns)

        print(f"\nSaving results to {output_path}...")
        df.write.mode("overwrite").parquet(str(output_path))

        print("\nPipeline complete!")
        return df

    # Fallback: run pandas pipeline for Databricks or local
    from .pipeline import run_pipeline as run_pipeline_pandas

    print("=" * 80)
    print("SURVEY NLP PIPELINE (PANDAS)")
    print("=" * 80)

    return run_pipeline_pandas(
        settings_path=settings_path,
        enriched_json_path=str(enriched_json) if enriched_json else None
    )


if __name__ == "__main__":
    run_pipeline()
