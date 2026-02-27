"""Lightweight end-to-end smoke test for the pipeline.

Run after: `pip install -e .` (or set `PYTHONPATH=src`).
"""

from pathlib import Path
import pandas as pd

from survey_app.config_runtime import load_settings
from survey_app.pipeline import run_pipeline


def main():
    project_root = Path(__file__).resolve().parents[1]
    input_csv = project_root / "Data" / "survey.csv"
    smoke_csv = project_root / "Data" / "survey_smoke.csv"

    if input_csv.exists():
        df = pd.read_csv(input_csv).head(25)
    elif smoke_csv.exists():
        df = pd.read_csv(smoke_csv)
    else:
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv} or {smoke_csv}. "
            "Create a small sample CSV first."
        )

    df.to_csv(smoke_csv, index=False)

    settings = load_settings("config/pipeline_settings.yaml")

    settings.setdefault("performance", {})["use_pyspark"] = False
    output_cfg = settings.setdefault("output", {})
    output_cfg["generate_sentiment"] = False
    output_cfg["generate_wordclouds"] = False
    output_cfg["generate_segmented"] = False
    output_cfg["generate_detailed_segments"] = False
    output_cfg["generate_absa"] = False

    run_pipeline(
        settings=settings,
        input_csv=str(smoke_csv),
        taxonomy_mode="keyword",
        output_dir="outputs/smoke",
        deliverables_dir=None,
        analytics=False,
    )

    print("Smoke test complete. Outputs in outputs/smoke")


if __name__ == "__main__":
    main()
