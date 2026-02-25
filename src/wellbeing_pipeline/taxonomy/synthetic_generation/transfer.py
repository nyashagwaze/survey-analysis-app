"""
Transfer Script: CSV Theme Library -> Enriched JSON Dictionary
Converts theme_phrase_library.csv to theme_subtheme_dictionary_v3_enriched.json
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ...config_runtime import _get_project_root


def load_csv(path: Path) -> pd.DataFrame:
    """Load the CSV theme phrase library"""
    print(f"Loading CSV from: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    return df


def build_enriched_structure(df):
    """
    Build the enriched JSON structure from CSV data

    Expected CSV columns: column, parent, theme, subtheme, polarity, phrase
    """

    # Get unique columns from the CSV
    columns = sorted(df["column"].dropna().unique())
    print(f"\nFound {len(columns)} columns: {columns}")

    # Initialize the structure
    enriched = {
        "metadata": {
            "schema_version": "3.0",
            "description": "Column-specific theme libraries with strict enforcement",
            "generated_from": "theme_phrase_library.csv",
            "columns": columns,
        },
        "column_libraries": {},
    }

    # Process each column
    for col in columns:
        print(f"\nProcessing column: {col}")
        col_data = df[df["column"] == col].copy()

        # Build nested structure: parent -> theme -> subtheme -> phrases
        parent_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        polarity_dict = {}

        for _, row in col_data.iterrows():
            parent = row["parent"]
            theme = row["theme"]
            subtheme = row["subtheme"]
            phrase = row["phrase"]
            polarity = row["polarity"]

            # Skip if any required field is missing
            if pd.isna(parent) or pd.isna(theme) or pd.isna(subtheme) or pd.isna(phrase):
                continue

            # Add phrase to the appropriate subtheme
            parent_dict[parent][theme][subtheme].append(phrase)

            # Store polarity (use the first one encountered for each subtheme)
            key = (parent, theme, subtheme)
            if key not in polarity_dict:
                polarity_dict[key] = polarity if not pd.isna(polarity) else "Either"

        # Convert to the required JSON structure
        parents_list = []
        for parent_name in sorted(parent_dict.keys()):
            themes_list = []

            for theme_name in sorted(parent_dict[parent_name].keys()):
                subthemes_list = []

                for subtheme_name in sorted(parent_dict[parent_name][theme_name].keys()):
                    phrases = parent_dict[parent_name][theme_name][subtheme_name]
                    polarity = polarity_dict.get((parent_name, theme_name, subtheme_name), "Either")

                    subthemes_list.append(
                        {
                            "name": subtheme_name,
                            "keywords_phrases": sorted(set(phrases)),
                            "default_polarity": polarity,
                        }
                    )

                    print(
                        f"   OK {parent_name} > {theme_name} > {subtheme_name}: {len(phrases)} phrases"
                    )

                themes_list.append({"theme_name": theme_name, "subthemes": subthemes_list})

            parents_list.append({"parent_name": parent_name, "themes": themes_list})

        # Add column description based on column name
        descriptions = {
            "Wellbeing_Details": "Problems, issues, stressors affecting wellbeing",
            "Areas_Improve": "Suggestions for workplace improvements",
            "Support_Provided": "Support mechanisms and resources available",
        }

        enriched["column_libraries"][col] = {
            "description": descriptions.get(col, f"Themes for {col}"),
            "parents": parents_list,
        }

    return enriched


def save_json(enriched, output_path: Path):
    """Save the enriched structure to JSON"""
    print(f"\nSaving to: {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    print("Saved enriched dictionary.")


def main():
    project_root = _get_project_root(start=Path(__file__).resolve())
    profile = os.getenv("PROFILE", "wellbeing")
    default_csv = project_root / "assets" / "taxonomy" / profile / "theme_phrase_library.csv"
    default_json = project_root / "assets" / "taxonomy" / profile / "theme_subtheme_dictionary_v3_enriched.json"

    parser = argparse.ArgumentParser(description="Convert theme phrase CSV to enriched JSON.")
    parser.add_argument("--input-csv", default=str(default_csv), help="Path to theme_phrase_library.csv")
    parser.add_argument("--output-json", default=str(default_json), help="Path to output JSON")
    args = parser.parse_args()

    csv_path = Path(args.input_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_csv(csv_path)
    enriched = build_enriched_structure(df)
    save_json(enriched, output_path)


if __name__ == "__main__":
    main()
