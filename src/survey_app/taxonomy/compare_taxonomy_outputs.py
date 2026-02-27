import argparse
from pathlib import Path

import pandas as pd

from ..config_runtime import load_settings, resolve_base_dir, resolve_path


def _ensure_id(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if id_col in df.columns:
        return df
    df = df.copy()
    df[id_col] = df.index
    return df


def main():
    here = Path(__file__).resolve()
    default_settings = here.parents[2] / "config/pipeline_settings.yaml"

    parser = argparse.ArgumentParser(description="Compare keyword vs semantic taxonomy outputs.")
    parser.add_argument("--settings", default=str(default_settings), help="Path to pipeline_settings.yaml")
    parser.add_argument("--keyword", dest="keyword_path", default=None, help="Path to assignments_keyword.csv")
    parser.add_argument("--semantic", dest="semantic_path", default=None, help="Path to assignments_semantic.csv")
    parser.add_argument("--output", dest="output_path", default=None, help="Output CSV path for mismatches")
    parser.add_argument("--use-deliverables", action="store_true", help="Read outputs from Deliverables dir")
    parser.add_argument("--id-col", default="ID", help="ID column name")
    args = parser.parse_args()

    settings_path = Path(args.settings)
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings not found: {settings_path}")

    settings = load_settings(str(settings_path))
    paths_cfg = settings.get("paths", {})

    base_dir = resolve_base_dir(settings, settings_path=settings_path, pipeline_dir=here.parents[1])

    if args.use_deliverables:
        output_base = resolve_path(base_dir, paths_cfg.get("deliverables", "Deliverables"))
    else:
        output_base = resolve_path(base_dir, paths_cfg.get("output_tables", "outputs/tables"))

    keyword_path = args.keyword_path or (output_base / "assignments_keyword.csv")
    semantic_path = args.semantic_path or (output_base / "assignments_semantic.csv")

    if not Path(keyword_path).exists():
        raise FileNotFoundError(f"Keyword assignments not found: {keyword_path}")
    if not Path(semantic_path).exists():
        raise FileNotFoundError(f"Semantic assignments not found: {semantic_path}")

    df_k = pd.read_csv(keyword_path)
    df_s = pd.read_csv(semantic_path)

    id_col = args.id_col
    df_k = _ensure_id(df_k, id_col)
    df_s = _ensure_id(df_s, id_col)

    for col in ["theme", "subtheme", "parent_theme", "match_method", "evidence", "reason"]:
        if col not in df_k.columns:
            df_k[col] = ""
        if col not in df_s.columns:
            df_s[col] = ""

    key_cols = [id_col, "TextColumn"]
    if "TextColumn" not in df_k.columns or "TextColumn" not in df_s.columns:
        raise ValueError("Both files must include TextColumn column.")

    merged = df_k.merge(df_s, on=key_cols, how="outer", suffixes=("_keyword", "_semantic"))

    merged["theme_keyword"] = merged["theme_keyword"].fillna("")
    merged["theme_semantic"] = merged["theme_semantic"].fillna("")
    merged["subtheme_keyword"] = merged["subtheme_keyword"].fillna("")
    merged["subtheme_semantic"] = merged["subtheme_semantic"].fillna("")

    merged["match"] = (merged["theme_keyword"] == merged["theme_semantic"]) & (
        merged["subtheme_keyword"] == merged["subtheme_semantic"]
    )

    total = len(merged)
    matches = int(merged["match"].sum())
    mismatches = total - matches

    print("Taxonomy comparison summary:")
    print(f"  total rows: {total}")
    print(f"  matches:    {matches}")
    print(f"  mismatches: {mismatches}")

    mismatched_rows = merged[~merged["match"]]

    output_path = args.output_path or (output_base / "taxonomy_comparison_mismatches.csv")
    mismatched_rows.to_csv(output_path, index=False)
    print(f"  mismatch report: {output_path}")


if __name__ == "__main__":
    main()
