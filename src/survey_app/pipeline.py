from pathlib import Path
import pandas as pd

from .config_runtime import (
    _get_project_root,
    resolve_base_dir,
    resolve_path,
    load_settings,
    load_yaml,
    load_dictionary_config,
    preprocess_dataframe,
    data_audit_summary,
    build_null_text_details,
    generate_taxonomy_reports,
    resolve_processing_columns,
)

from .clean_normalise.null_text_detector import add_response_quality_flags, get_response_quality_report
from .sentiment.sentiment_module import add_sentiment_columns_pandas, DEFAULT_COLUMN_WEIGHTS
from .taxonomy.semantic_taxonomy import assign_taxonomy_semantic
from .taxonomy.cross_encoder_taxonomy import assign_taxonomy_semantic_cross_encoder
from .taxonomy.keyword_taxonomy import assign_taxonomy_keyword
from .grouping.segmentation import build_segments, analyze_segments
from .grouping.segment_summary import segment_sentiment_summary
from .grouping.absa_module import absa_on_segments

# ---------------------------------------------------
# Pipeline runner (pandas-first)
# ---------------------------------------------------

def _apply_column_map(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    if not column_map:
        return df
    rename_map = {src: dest for src, dest in column_map.items() if src in df.columns and dest}
    if not rename_map:
        return df
    return df.rename(columns=rename_map)


def _build_sentiment_weights(sentiment_columns: list, sentiment_cfg: dict) -> dict:
    weights_cfg = (sentiment_cfg or {}).get("column_weights") or {}
    if weights_cfg:
        active = {col: float(weights_cfg[col]) for col in sentiment_columns if col in weights_cfg}
        if active:
            return active
    if not sentiment_columns:
        return {}
    w = 1.0 / len(sentiment_columns)
    return {col: w for col in sentiment_columns}

def run_pipeline(
    settings_path: str = "config/pipeline_settings.yaml",
    settings: dict = None,
    input_csv: str = None,
    themes_path: str = None,
    dictionary_path: str = None,
    enriched_json_path: str = None,
    taxonomy_mode: str = None,
    output_dir: str = None,
    deliverables_dir: str = None,
    analytics: bool = None
):
    """
    End-to-end pipeline runner for local/pandas usage.
    Reads settings, preprocesses, runs null detection, sentiment, and taxonomy.
    """
    from pathlib import Path

    pipeline_dir = Path(__file__).resolve().parent
    project_root = _get_project_root(start=pipeline_dir)
    if settings is None:
        settings = load_settings(settings_path)

    paths_cfg = settings.get("paths", {})
    base_dir = resolve_base_dir(settings, settings_path=settings_path, project_root=project_root, pipeline_dir=pipeline_dir)
    input_cfg = settings.get("input", {}) or {}
    columns_cfg = resolve_processing_columns(settings)
    text_columns = columns_cfg.get("text_columns") or []
    taxonomy_columns = columns_cfg.get("taxonomy_columns") or text_columns
    sentiment_columns = columns_cfg.get("sentiment_columns") or text_columns
    segmentation_columns = columns_cfg.get("segmentation_columns") or text_columns

    if not text_columns:
        raise ValueError("No text columns configured. Set processing.text_columns or text_columns in settings.")

    input_csv = input_csv or paths_cfg.get("input_csv")
    if not input_csv:
        raise ValueError("input_csv is required (pass --input or set paths.input_csv in pipeline_settings.yaml).")
    input_path = resolve_path(base_dir, input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    dictionary_path = dictionary_path or paths_cfg.get("dictionary")
    dictionary_path = resolve_path(base_dir, dictionary_path) if dictionary_path else None
    dict_assets = load_dictionary_config(dictionary_path) if dictionary_path and dictionary_path.exists() else {}

    themes_path = themes_path or paths_cfg.get("themes")
    themes_path = resolve_path(base_dir, themes_path) if themes_path else None
    themes = load_yaml(themes_path) if themes_path and themes_path.exists() else {}

    enriched_json_path = enriched_json_path or paths_cfg.get("enriched_json")
    enriched_json_path = resolve_path(base_dir, enriched_json_path) if enriched_json_path else None

    if taxonomy_mode is None:
        if enriched_json_path:
            taxonomy_mode = "semantic"
        elif themes_path and themes_path.suffix.lower() in {".json"}:
            taxonomy_mode = "semantic"
        else:
            taxonomy_mode = "keyword"

    print("Loading input...")
    df = pd.read_csv(input_path)
    df = _apply_column_map(df, input_cfg.get("column_map", {}))

    print("Preprocessing text...")
    df = preprocess_dataframe(df, text_columns, dict_assets, settings)

    null_cfg = settings.get("null_detection", {})
    min_len = null_cfg.get("min_meaningful_length", 3)
    max_len = null_cfg.get("max_dismissive_length", 50)

    print("Running null text detection...")
    df = add_response_quality_flags(
        df,
        text_columns,
        min_meaningful_length=min_len,
        max_dismissive_length=max_len
    )

    output_cfg = settings.get("output", {})
    semantic_cfg = settings.get("semantic", {})
    semantic_model = semantic_cfg.get("model_name", "all-MiniLM-L6-v2")
    try:
        model_path = resolve_path(base_dir, semantic_model)
        if model_path and model_path.exists():
            semantic_model = str(model_path)
    except Exception:
        pass
    sentiment_cfg = settings.get("sentiment", {})
    analytics_cfg = settings.get("analytics", {})

    output_dir = output_dir or paths_cfg.get("output_tables") or "outputs/tables"
    output_path = resolve_path(base_dir, output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    deliverables_dir = deliverables_dir or paths_cfg.get("deliverables")
    deliverables_path = None
    if deliverables_dir:
        deliverables_path = resolve_path(base_dir, deliverables_dir)
        deliverables_path.mkdir(parents=True, exist_ok=True)

    analytics_enabled = analytics_cfg.get("enabled", False)
    if analytics is not None:
        analytics_enabled = bool(analytics)

    if analytics_enabled:
        audit = data_audit_summary(df, text_columns)
        print("Data audit summary:")
        print(audit.to_string(index=False))

        if analytics_cfg.get("export_csv", True):
            audit_name = analytics_cfg.get("audit_filename", "data_audit.csv")
            audit.to_csv(output_path / audit_name, index=False)
            if deliverables_path is not None:
                audit.to_csv(deliverables_path / audit_name, index=False)

    if output_cfg.get("generate_sentiment", True) and sentiment_columns:
        print("Running sentiment analysis...")
        sentiment_weights = _build_sentiment_weights(sentiment_columns, sentiment_cfg)
        df = add_sentiment_columns_pandas(
            df,
            text_columns=sentiment_columns,
            weights=sentiment_weights,
            pos=sentiment_cfg.get("positive_threshold", 0.05),
            neg=sentiment_cfg.get("negative_threshold", -0.05),
            skip_dismissed=sentiment_cfg.get("skip_dismissed", True),
            dismissed_sentiment_value=sentiment_cfg.get("dismissed_sentiment_value", None)
        )

    if not taxonomy_columns:
        raise ValueError("No taxonomy columns configured. Set processing.taxonomy_columns or text_columns.")

    print(f"Running taxonomy ({taxonomy_mode})...")
    assignments = None
    if taxonomy_mode == "semantic":
        if not enriched_json_path:
            raise ValueError("enriched_json_path is required for semantic taxonomy mode.")
        similarity_threshold = semantic_cfg.get("similarity_threshold", 0.35)
        top_k = semantic_cfg.get("top_k", 3)
        use_cross_encoder = bool(semantic_cfg.get("use_cross_encoder", False))
        if use_cross_encoder:
            scores_output = semantic_cfg.get("scores_output")
            scores_output = resolve_path(base_dir, scores_output) if scores_output else None
            if semantic_cfg.get("assignments_filename"):
                output_cfg["assignments_filename"] = semantic_cfg.get("assignments_filename")
            assignments = assign_taxonomy_semantic_cross_encoder(
                df_spark=df,
                text_columns=taxonomy_columns,
                enriched_json_path=Path(enriched_json_path),
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
            assignments = assign_taxonomy_semantic(
                df,
                text_columns=taxonomy_columns,
                enriched_json_path=Path(enriched_json_path),
                model_name=semantic_model,
                similarity_threshold=similarity_threshold,
                top_k=top_k
            )
    else:
        assignments = assign_taxonomy_keyword(
            df,
            text_columns=taxonomy_columns,
            themes=themes,
            taxonomy_cfg=settings.get("taxonomy", {}),
            output_cfg=output_cfg
        )

    if output_cfg.get("generate_taxonomy_reports", False):
        generate_taxonomy_reports(
            assignments=assignments,
            output_path=output_path,
            deliverables_path=deliverables_path,
            taxonomy_mode=taxonomy_mode,
            matched_filename=output_cfg.get("taxonomy_matched_filename"),
            unmatched_filename=output_cfg.get("taxonomy_unmatched_filename")
        )

    # ---------------------------------------------------
    # Segmentation (optional)
    # ---------------------------------------------------
    seg_cfg = settings.get("segmentation", {}) or {}
    if seg_cfg.get("enabled") and (output_cfg.get("generate_segmented") or output_cfg.get("generate_detailed_segments")):
        seg_semantic_cfg = dict(semantic_cfg)
        seg_semantic_cfg["model_name"] = semantic_model
        if not segmentation_columns:
            raise ValueError("segmentation_columns is empty but segmentation is enabled.")
        segments_df = build_segments(df, segmentation_columns, seg_cfg)
        segment_results = analyze_segments(
            segments_df=segments_df,
            taxonomy_mode=taxonomy_mode,
            themes=themes,
            enriched_json_path=Path(enriched_json_path) if enriched_json_path else None,
            semantic_cfg=seg_semantic_cfg,
            taxonomy_cfg=settings.get("taxonomy", {}),
            sentiment_cfg=settings.get("sentiment", {})
        )

        if output_cfg.get("generate_detailed_segments", False):
            detailed_name = output_cfg.get("segments_detailed_filename", "segments_detailed.csv")
            segment_results.to_csv(output_path / detailed_name, index=False)
            if deliverables_path is not None:
                segment_results.to_csv(deliverables_path / detailed_name, index=False)

        if output_cfg.get("generate_segmented", False):
            segmented_name = output_cfg.get("segmented_analysis_filename", "segmented_analysis.csv")
            cols = [
                "ID", "TextColumn", "segment_index", "segment_text",
                "theme", "subtheme", "parent_theme", "sentiment_label", "compound"
            ]
            available_cols = [c for c in cols if c in segment_results.columns]
            segment_results[available_cols].to_csv(output_path / segmented_name, index=False)
            if deliverables_path is not None:
                segment_results[available_cols].to_csv(deliverables_path / segmented_name, index=False)

        if output_cfg.get("generate_segment_sentiment_summary", False):
            summary_cfg = settings.get("segment_summary", {}) or {}
            summary = segment_sentiment_summary(segment_results, summary_cfg)
            summary_name = output_cfg.get("segment_sentiment_summary_filename", "segment_sentiment_summary.csv")
            summary.to_csv(output_path / summary_name, index=False)
            if deliverables_path is not None:
                summary.to_csv(deliverables_path / summary_name, index=False)

        absa_cfg = settings.get("absa", {}) or {}
        if output_cfg.get("generate_absa", False) and absa_cfg.get("enabled", False):
            absa_results = absa_on_segments(segment_results, absa_cfg)
            absa_name = output_cfg.get("absa_filename", "absa_segment_sentiment.csv")
            absa_results.to_csv(output_path / absa_name, index=False)
            if deliverables_path is not None:
                absa_results.to_csv(deliverables_path / absa_name, index=False)

    if output_cfg.get("generate_assignments", True):
        out_name = output_cfg.get("assignments_filename")
        if not out_name:
            out_name = f"assignments_{taxonomy_mode}.csv"
        out_file = output_path / out_name
        assignments.to_csv(out_file, index=False)
        if deliverables_path is not None:
            deliverables_out_name = output_cfg.get("deliverables_assignments_filename") or out_name
            assignments.to_csv(deliverables_path / deliverables_out_name, index=False)

    if output_cfg.get("generate_quality_report", True):
        report = get_response_quality_report(
            df,
            text_columns,
            min_meaningful_length=min_len,
            max_dismissive_length=max_len
        )
        report.to_csv(output_path / "null_text_report.csv", index=False)

    if output_cfg.get("generate_null_text_details", False):
        details_name = output_cfg.get("null_text_details_filename", "null_text_details.csv")
        details = build_null_text_details(df, text_columns)
        details.to_csv(output_path / details_name, index=False)
        if deliverables_path is not None:
            details.to_csv(deliverables_path / details_name, index=False)

    if output_cfg.get("generate_sentiment", True) and sentiment_columns:
        cols = []
        if "ID" in df.columns:
            cols.append("ID")
        cols += ["compound", "sentiment_label", "coping_flag"]
        cols += [f"compound_{c}" for c in sentiment_columns if f"compound_{c}" in df.columns]
        df[cols].to_csv(output_path / "sentiment_by_id.csv", index=False)
        if deliverables_path is not None:
            df[cols].to_csv(deliverables_path / "sentiment_by_id.csv", index=False)

    print("Pipeline complete.")
    return {
        "data": df,
        "assignments": assignments
    }


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the survey NLP pipeline (pandas mode).")
    parser.add_argument("--settings", default="config/pipeline_settings.yaml", help="Path to pipeline_settings.yaml")
    parser.add_argument("--input", dest="input_csv", default=None, help="Input CSV path")
    parser.add_argument("--themes", dest="themes_path", default=None, help="Themes YAML path")
    parser.add_argument("--dictionary", dest="dictionary_path", default=None, help="Dictionary YAML path")
    parser.add_argument("--enriched-json", dest="enriched_json_path", default=None, help="Enriched JSON path for semantic mode")
    parser.add_argument("--taxonomy-mode", dest="taxonomy_mode", default=None, choices=["keyword", "semantic"], help="Taxonomy mode")
    parser.add_argument("--output-dir", dest="output_dir", default=None, help="Output directory")
    parser.add_argument("--deliverables-dir", dest="deliverables_dir", default=None, help="Deliverables directory")
    parser.add_argument("--analytics", dest="analytics", action="store_true", help="Enable basic data audit output")
    parser.add_argument("--no-analytics", dest="analytics", action="store_false", help="Disable basic data audit output")
    parser.set_defaults(analytics=None)
    args = parser.parse_args()

    run_pipeline(
        settings_path=args.settings,
        input_csv=args.input_csv,
        themes_path=args.themes_path,
        dictionary_path=args.dictionary_path,
        enriched_json_path=args.enriched_json_path,
        taxonomy_mode=args.taxonomy_mode,
        output_dir=args.output_dir,
        deliverables_dir=args.deliverables_dir,
        analytics=args.analytics
    )


if __name__ == "__main__":
    _main()
