import argparse
from pathlib import Path

import pandas as pd

from ..config_runtime import load_settings, resolve_base_dir, resolve_path


def _summarize_empty(df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
    rows = []
    total_rows = len(df)

    for col in text_columns:
        if col not in df.columns:
            rows.append({
                "column": col,
                "total_rows": total_rows,
                "missing_column": True,
                "null_count": None,
                "empty_or_whitespace": None,
                "empty_only": None,
                "no_entry_count": None,
            })
            continue

        s = df[col]
        null_count = int(s.isna().sum())
        s_clean = s.fillna("")
        empty_or_whitespace = int(s_clean.astype(str).str.strip().eq("").sum())
        empty_only = max(0, empty_or_whitespace - null_count)
        no_entry_count = int(s_clean.astype(str).str.strip().str.lower().eq("no entry").sum())

        rows.append({
            "column": col,
            "total_rows": total_rows,
            "missing_column": False,
            "null_count": null_count,
            "empty_or_whitespace": empty_or_whitespace,
            "empty_only": empty_only,
            "no_entry_count": no_entry_count,
        })

    return pd.DataFrame(rows)


def main():
    package_dir = Path(__file__).resolve().parents[1]
    default_settings = package_dir.parent.parent / "config/pipeline_settings.yaml"

    parser = argparse.ArgumentParser(description="Basic data shape and empty-row audit.")
    parser.add_argument("--settings", default=str(default_settings), help="Path to pipeline_settings.yaml")
    parser.add_argument("--input", dest="input_csv", default=None, help="Override input CSV path")
    args = parser.parse_args()

    settings_path = Path(args.settings)
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings not found: {settings_path}")

    settings = load_settings(str(settings_path))
    paths_cfg = settings.get("paths", {})
    text_columns = settings.get("text_columns", [])

    base_dir = resolve_base_dir(settings, settings_path=settings_path, pipeline_dir=package_dir)

    input_csv = args.input_csv or paths_cfg.get("input_csv")
    if not input_csv:
        raise ValueError("input_csv is required (pass --input or set paths.input_csv in settings).")

    input_path = resolve_path(base_dir, input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    print("Data shape:")
    print(f"  rows: {df.shape[0]}")
    print(f"  cols: {df.shape[1]}")
    print("")

    if not text_columns:
        print("No text_columns defined in settings. Add a text_columns list to pipeline_settings.yaml")
        return

    summary = _summarize_empty(df, text_columns)
    print("Empty / null summary (per text column):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
