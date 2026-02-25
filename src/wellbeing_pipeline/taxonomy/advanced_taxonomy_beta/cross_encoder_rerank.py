"""
Cross-Encoder Re-Ranking for Taxonomy Assignment

Workflow:
1) Use semantic bi-encoder to retrieve top-N candidate themes
2) Re-rank candidates with a cross-encoder for better precision
3) Emit final theme/subtheme assignments
"""

import argparse
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


from ...config_runtime import load_settings, resolve_base_dir, resolve_path
from ..cross_encoder_taxonomy import assign_taxonomy_semantic_cross_encoder


def _default_output_name() -> str:
    return "assignments_semantic_cross_encoder.csv"


def main():
    parser = argparse.ArgumentParser(description="Cross-encoder reranking for taxonomy assignments.")
    project_root = Path(__file__).resolve().parents[4]
    parser.add_argument("--settings", default=str(project_root / "config/pipeline_settings.yaml"))
    parser.add_argument("--input", dest="input_csv", default=None)
    parser.add_argument("--enriched-json", dest="enriched_json", default=None)
    parser.add_argument("--bi-model", dest="bi_model", default=None)
    parser.add_argument("--ce-model", dest="ce_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--bi-top-k", dest="bi_top_k", type=int, default=15)
    parser.add_argument("--bi-threshold", dest="bi_threshold", type=float, default=0.20)
    parser.add_argument("--top-k", dest="top_k", type=int, default=3)
    parser.add_argument("--pair-template", dest="pair_template", default="{phrase}")
    parser.add_argument("--output", dest="output_csv", default=None)
    parser.add_argument("--candidates-output", dest="candidates_output", default=None)
    parser.add_argument("--scores-output", dest="scores_output", default=None)
    args = parser.parse_args()

    settings = load_settings(args.settings)
    paths_cfg = settings.get("paths", {})
    package_dir = Path(__file__).resolve().parents[2]
    base_dir = resolve_base_dir(settings, settings_path=args.settings, pipeline_dir=package_dir)

    input_csv = args.input_csv or paths_cfg.get("input_csv")
    if not input_csv:
        raise ValueError("input_csv is required (pass --input or set paths.input_csv)")
    input_path = resolve_path(base_dir, input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    enriched_json = args.enriched_json or paths_cfg.get("enriched_json")
    if not enriched_json:
        raise ValueError("enriched_json is required (pass --enriched-json or set paths.enriched_json)")
    enriched_path = resolve_path(base_dir, enriched_json)
    if not enriched_path.exists():
        raise FileNotFoundError(f"Enriched JSON not found: {enriched_path}")

    semantic_cfg = settings.get("semantic", {})
    bi_model = args.bi_model or semantic_cfg.get("model_name", "all-MiniLM-L6-v2")
    try:
        model_path = resolve_path(base_dir, bi_model)
        if model_path and model_path.exists():
            bi_model = str(model_path)
    except Exception:
        pass

    text_columns = settings.get("text_columns", [])
    if not text_columns:
        raise ValueError("text_columns must be set in settings")

    df = pd.read_csv(input_path)

    delimiter = settings.get("taxonomy", {}).get("multi_label", {}).get("delimiter", " | ")
    out_df = assign_taxonomy_semantic_cross_encoder(
        df_spark=df,
        text_columns=text_columns,
        enriched_json_path=enriched_path,
        model_name=bi_model,
        cross_encoder_model=args.ce_model,
        bi_top_k=int(args.bi_top_k),
        bi_threshold=float(args.bi_threshold),
        top_k=int(args.top_k),
        pair_template=args.pair_template,
        scores_output=Path(args.scores_output) if args.scores_output else None,
        candidates_output=Path(args.candidates_output) if args.candidates_output else None,
        delimiter=delimiter
    )

    output_csv = args.output_csv
    if not output_csv:
        output_dir = paths_cfg.get("output_tables", "outputs/tables")
        output_path = resolve_path(base_dir, output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_csv = output_path / _default_output_name()

    out_df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
