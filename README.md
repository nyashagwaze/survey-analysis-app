# Survey Analysis App

> **General-purpose survey analysis app and NLP pipeline**
>
> Upload a CSV, configure columns and taxonomy settings, and generate theme assignments and reports without exposing raw text.

## What It Does

- Semantic or keyword taxonomy matching for free-text survey responses
- Optional sentiment analysis
- Output tables for assignments, matched/unmatched reports, and audits
- Streamlit app for non-technical users

## Quick Start (Streamlit)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run streamlit_app.py
```

## Quick Start (CLI)

1. Install the package:

```bash
pip install -e .
```

2. Run the pipeline:

```bash
survey-app --input Data/survey.csv
```

## Configuration

Main settings live in:
- `config/pipeline_settings.yaml`

Profile overrides live in:
- `config/profiles/<profile>/profile.yaml`

Example config snippet:

```yaml
profile: "general"
paths:
  base_dir: "."
  input_csv: "Data/survey.csv"
  dictionary: "config/profiles/{profile}/dictionary.yaml"
  themes: "config/profiles/{profile}/themes.yaml"
  enriched_json: "assets/taxonomy/{profile}/theme_subtheme_dictionary_v3_enriched.json"
```

## Profiles And Taxonomies

- Profiles live under `config/profiles/`
- Taxonomy assets live under `assets/taxonomy/<profile>/`
- Choose a profile in `config/pipeline_settings.yaml` or in the Streamlit sidebar
- Starter templates live under `assets/taxonomy/templates/`.
- Templates are optional and can be deleted if you don't want sample taxonomies.

### Build Taxonomies In The App

The Streamlit app includes a Taxonomy Builder that can:
1. Validate a `theme_phrase_library.csv`
2. Generate `theme_subtheme_dictionary_v3_enriched.json`
3. Generate `themes.yaml` for keyword matching
4. Save files into `assets/taxonomy/<profile>/`

### Create A New Profile

Use the template helper to create a new profile folder:

```bash
python scripts/create_profile.py --name my_profile
```

Optionally copy an existing taxonomy as a starting point:

```bash
python scripts/create_profile.py --name my_profile --with-taxonomy --from-taxonomy general
```

## Outputs

Default outputs go to `outputs/tables` and include:
- `assignments_<mode>.csv`
- `taxonomy_matched_<mode>.csv`
- `taxonomy_unmatched_<mode>.csv`
- `sentiment_by_id.csv`
- `null_text_report.csv`
- `data_audit.csv` (if analytics enabled)

## Development

Install dev tooling:

```bash
pip install ".[dev]"
```

### Developer Scripts

Windows (PowerShell):

```powershell
.\scripts\dev_setup.ps1
.\scripts\run_app.ps1 -InstallDeps
```

Cross-platform (Make):

```bash
make setup
make run-app
```

## License

MIT
