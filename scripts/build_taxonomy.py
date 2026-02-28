import argparse
import sys
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from survey_app.taxonomy.synthetic_generation.transfer import build_enriched_structure


REQUIRED_COLUMNS = ["column", "parent", "theme", "subtheme", "polarity", "phrase"]
ALLOWED_POLARITY = {"positive", "negative", "neutral", "either"}


def validate_taxonomy(df: pd.DataFrame, min_phrases: int = 2):
    errors = []
    warnings = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
        return errors, warnings

    for col in REQUIRED_COLUMNS:
        empty = df[col].isna() | df[col].astype(str).str.strip().eq("")
        empty_count = int(empty.sum())
        if empty_count:
            errors.append(f"Column '{col}' has {empty_count} empty values.")

    pol = df["polarity"].astype(str).str.strip().str.lower()
    bad_pol = df[~pol.isin(ALLOWED_POLARITY)]
    if not bad_pol.empty:
        warnings.append(
            f"{len(bad_pol)} rows have polarity outside {sorted(ALLOWED_POLARITY)}."
        )

    dupes = df.duplicated(subset=REQUIRED_COLUMNS).sum()
    if dupes:
        warnings.append(
            f"{dupes} duplicate rows found (same column/parent/theme/subtheme/polarity/phrase)."
        )

    counts = df.groupby(["column", "parent", "theme", "subtheme"]).size()
    low = counts[counts < min_phrases]
    if not low.empty:
        warnings.append(
            f"{len(low)} subthemes have fewer than {min_phrases} phrases."
        )

    return errors, warnings


def build_themes_yaml(df: pd.DataFrame) -> dict:
    themes = {}
    grouped = df.groupby("theme")
    for theme_name, theme_df in grouped:
        parent_vals = theme_df["parent"].dropna().astype(str)
        parent_theme = parent_vals.mode().iloc[0] if not parent_vals.empty else ""

        subthemes = []
        for sub_name, sub_df in theme_df.groupby("subtheme"):
            phrases = sorted(set(sub_df["phrase"].dropna().astype(str).str.strip()))
            col_vals = sorted(set(sub_df["column"].dropna().astype(str)))
            polarity_vals = sub_df["polarity"].dropna().astype(str)
            default_polarity = (
                polarity_vals.mode().iloc[0] if not polarity_vals.empty else "Either"
            )
            subthemes.append(
                {
                    "name": str(sub_name),
                    "keywords_phrases": phrases,
                    "likely_columns": col_vals,
                    "default_polarity": default_polarity,
                }
            )

        themes[str(theme_name)] = {
            "parent_theme": parent_theme,
            "subthemes": subthemes,
        }

    return {"themes": themes}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and build taxonomy files from CSV.")
    parser.add_argument("--input-csv", required=True, help="Path to theme_phrase_library.csv")
    parser.add_argument("--output-json", default=None, help="Path for enriched JSON output")
    parser.add_argument("--output-themes", default=None, help="Path for themes.yaml output")
    parser.add_argument("--min-phrases", type=int, default=2, help="Min phrases per subtheme")
    parser.add_argument("--validate-only", action="store_true", help="Only validate CSV")
    parser.add_argument("--allow-invalid", action="store_true", help="Write outputs even if warnings/errors")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Input CSV not found: {input_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(input_path)
    errors, warnings = validate_taxonomy(df, min_phrases=args.min_phrases)

    if errors:
        print("Validation errors:")
        for msg in errors:
            print(f"- {msg}")
    if warnings:
        print("Validation warnings:")
        for msg in warnings:
            print(f"- {msg}")

    if errors and not args.allow_invalid:
        return 2

    if args.validate_only:
        return 0

    output_json = Path(args.output_json) if args.output_json else input_path.with_name(
        "theme_subtheme_dictionary_v3_enriched.json"
    )
    enriched = build_enriched_structure(df)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    print(f"Wrote enriched JSON: {output_json}")

    if args.output_themes:
        output_themes = Path(args.output_themes)
        themes_yaml = build_themes_yaml(df)
        try:
            import yaml
            output_themes.write_text(yaml.safe_dump(themes_yaml, sort_keys=False), encoding="utf-8")
        except Exception:
            output_themes.write_text(json.dumps(themes_yaml, indent=2), encoding="utf-8")
        print(f"Wrote themes YAML: {output_themes}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
