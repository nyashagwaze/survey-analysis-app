# Survey Analysis App

NLP pipeline for survey free-text responses. Uses semantic matching, keyword taxonomy, and optional sentiment analysis to produce theme assignments and reporting outputs without exposing raw text.

```
survey-analysis-app/
  src/
    survey_app/
      clean_normalise/
      grouping/
      taxonomy/
      sentiment/
      wordcloud/
      pipeline.py
      run.py
      config_runtime.py
  config/
    pipeline_settings.yaml
    profiles/
      general/
        dictionary.yaml
        themes.yaml
        profile.yaml
      hearing/
        dictionary.yaml
        themes.yaml
        profile.yaml
  assets/
    taxonomy/
      general/
        theme_phrase_library.csv
        theme_subtheme_dictionary_v3_enriched.json
      hearing/
        theme_phrase_library.csv
        theme_subtheme_dictionary_v3_enriched.json
  notebooks/
    01_run_pipeline_minimal.py
    02_debug_full.py
    10_run_profile.py
    11_unpack_bundle.py
  Data/
  outputs/
  Deliverables/
```

## Quick Start (Databricks)

1) Import the minimal notebook:
- `notebooks/01_run_pipeline_minimal.py`

2) Edit `PROJECT_ROOT` at the top of the notebook:
```
PROJECT_ROOT = "/Workspace/Repos/<your_user>/<your_repo>/survey-analysis-app"
```

3) Run the notebook. It loads `config/pipeline_settings.yaml` and executes the full pipeline.

## Debug Notebook

Use the full debug notebook to step through each module:
- `notebooks/02_debug_full.py`

It runs:
- data audit
- preprocessing
- null text detection
- sentiment
- keyword taxonomy (semantic optional)
- report generation

## Configuration

All key settings are centralized in:
- `config/pipeline_settings.yaml`

Key path settings (relative to `base_dir`, resolved against the project root):
```
profile: "general"
input:
  column_map: {}
processing:
  # text_columns, taxonomy_columns, sentiment_columns, segmentation_columns
  # (see config/pipeline_settings.yaml for examples)
paths:
  base_dir: "."
  input_csv: "Data/survey.csv"
  dictionary: "config/profiles/{profile}/dictionary.yaml"
  themes: "config/profiles/{profile}/themes.yaml"
  enriched_json: "assets/taxonomy/{profile}/theme_subtheme_dictionary_v3_enriched.json"
  output_tables: "outputs/tables"
  deliverables: "Deliverables"
```

Paths support `{profile}` token replacement based on the `profile` setting.

Profile-specific overrides can live in:
- `config/profiles/<profile>/profile.yaml`
This file can define `text_columns`, `processing` overrides, and `input.column_map`.

To move workspaces, either set `PROJECT_ROOT`/`PIPELINE_PROJECT_ROOT` or update `base_dir` to an absolute path.

## Outputs

Generated in `outputs/tables` by default:
- `assignments_keyword.csv` or `assignments_semantic.csv`
- `taxonomy_matched_<mode>.csv`
- `taxonomy_unmatched_<mode>.csv`
- `sentiment_by_id.csv`
- `null_text_report.csv`
- `null_text_details.csv`
- `data_audit.csv` (if analytics enabled)

Final deliverables (if configured) are also copied to `Deliverables/`.

## Wordclouds

Generate cleaner wordclouds per column:
```
python -m survey_app.wordcloud.column_wordclouds --settings config/pipeline_settings.yaml
```

Tuning options are under `wordcloud` in `pipeline_settings.yaml`.

## Notes

- Keyword taxonomy uses `config/profiles/{profile}/themes.yaml`.
- Semantic taxonomy requires an enriched JSON phrase library in `assets/taxonomy/{profile}/`.
- Sentiment uses RoBERTa; thresholds configurable in settings.
