import contextlib
import hashlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
TEMPLATES_DIR = ROOT / "assets" / "taxonomy" / "templates"

from survey_app.pipeline import run_pipeline
from survey_app import config_runtime as cfg
from survey_app.taxonomy.synthetic_generation.transfer import build_enriched_structure


st.set_page_config(page_title="Survey Analysis App", layout="wide")
st.title("Survey Analysis App")
st.caption("Upload a CSV, tune the pipeline settings, and run the analysis.")


def _discover_profiles():
    profiles_dir = ROOT / "config" / "profiles"
    if not profiles_dir.exists():
        return []
    return sorted([p.name for p in profiles_dir.iterdir() if p.is_dir()])


def _load_uploaded_df(uploaded_file) -> pd.DataFrame:
    data = uploaded_file.getvalue()
    return pd.read_csv(io.BytesIO(data))


def _ensure_run_dirs():
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    run_root = ROOT / "outputs" / "app_runs" / run_id
    upload_dir = run_root / "uploads"
    tables_dir = run_root / "tables"
    deliverables_dir = run_root / "deliverables"
    upload_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    deliverables_dir.mkdir(parents=True, exist_ok=True)
    return run_root, upload_dir, tables_dir, deliverables_dir


def _build_sample_df(rows: int = 12) -> pd.DataFrame:
    columns = ["Free_Text_1", "Free_Text_2", "Free_Text_3"]
    data = {}
    for col in columns:
        data[col] = [f"Sample response {i + 1} for {col}" for i in range(rows)]
    return pd.DataFrame(data)


def _load_settings_for_profile(profile: str) -> dict:
    # Rebuild from raw YAML so {profile} tokens stay replaceable.
    settings_path = ROOT / "config" / "pipeline_settings.yaml"
    raw = cfg.load_yaml(settings_path)
    merged = cfg._deep_merge(cfg._default_settings(), raw)
    merged["profile"] = profile

    overrides = cfg._load_profile_overrides(
        profile,
        project_root=ROOT,
        pipeline_dir=ROOT / "src" / "survey_app",
    )
    if overrides:
        merged = cfg._deep_merge(merged, overrides)

    merged = cfg.apply_runtime_overrides(merged)
    merged = cfg._apply_profile_paths(merged)
    return merged


def _resolve_asset(settings: dict, key: str):
    paths_cfg = settings.get("paths", {}) or {}
    rel = paths_cfg.get(key)
    if not rel:
        return None
    base_dir = cfg.resolve_base_dir(
        settings,
        settings_path="config/pipeline_settings.yaml",
        project_root=ROOT,
        pipeline_dir=ROOT / "src" / "survey_app",
    )
    return cfg.resolve_path(base_dir, rel)


def _validate_taxonomy_df(df: pd.DataFrame, min_phrases: int = 2):
    errors = []
    warnings = []
    required = ["column", "parent", "theme", "subtheme", "polarity", "phrase"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
        return errors, warnings

    for col in required:
        empty = df[col].isna() | df[col].astype(str).str.strip().eq("")
        empty_count = int(empty.sum())
        if empty_count:
            errors.append(f"Column '{col}' has {empty_count} empty values.")

    allowed = {"positive", "negative", "neutral", "either"}
    pol = df["polarity"].astype(str).str.strip().str.lower()
    bad_pol = df[~pol.isin(allowed)]
    if not bad_pol.empty:
        warnings.append(
            f"{len(bad_pol)} rows have polarity outside {sorted(allowed)}."
        )

    dupes = df.duplicated(subset=required).sum()
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


def _build_themes_yaml(df: pd.DataFrame) -> dict:
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


def _settings_fingerprint(settings: dict) -> str:
    payload = json.dumps(settings, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_run_key(settings: dict, taxonomy_mode: str, data_bytes: bytes) -> str:
    data_hash = hashlib.sha256(data_bytes or b"").hexdigest()
    settings_hash = _settings_fingerprint(settings)
    return f"{taxonomy_mode}:{data_hash}:{settings_hash}"


def _dump_settings(settings: dict) -> bytes:
    try:
        import yaml
        return yaml.safe_dump(settings, sort_keys=False).encode("utf-8")
    except Exception:
        return json.dumps(settings, indent=2, default=str).encode("utf-8")


profiles = _discover_profiles()
default_profile = "general" if "general" in profiles else (profiles[0] if profiles else "general")
if not profiles:
    profiles = [default_profile]

df_raw = None
df_for_run = None
selected_text_cols = []

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        file_changed = (
            st.session_state.get("uploaded_name") != uploaded_file.name
            or st.session_state.get("uploaded_size") != uploaded_file.size
        )
        if file_changed:
            try:
                data_bytes = uploaded_file.getvalue()
                st.session_state["uploaded_df"] = pd.read_csv(io.BytesIO(data_bytes))
                st.session_state["uploaded_name"] = uploaded_file.name
                st.session_state["uploaded_size"] = uploaded_file.size
                st.session_state["uploaded_bytes"] = data_bytes
            except Exception as exc:
                st.session_state["uploaded_df"] = None
                st.error(f"Failed to read CSV: {exc}")
        df_raw = st.session_state.get("uploaded_df")
    else:
        sample_df = _build_sample_df()
        sample_bytes = sample_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download sample CSV",
            data=sample_bytes,
            file_name="sample_survey.csv",
            mime="text/csv",
        )
        if st.button("Use sample data"):
            st.session_state["uploaded_df"] = sample_df
            st.session_state["uploaded_name"] = "sample_survey.csv"
            st.session_state["uploaded_size"] = len(sample_bytes)
            st.session_state["uploaded_bytes"] = sample_bytes
            st.rerun()

    st.header("Profile")
    profile = st.selectbox("Taxonomy profile", options=profiles, index=profiles.index(default_profile))
    settings = _load_settings_for_profile(profile)

    enriched_path = _resolve_asset(settings, "enriched_json")
    themes_path = _resolve_asset(settings, "themes")
    dictionary_path = _resolve_asset(settings, "dictionary")

    if enriched_path and not enriched_path.exists():
        st.warning(f"Missing enriched taxonomy: {enriched_path}")
    if themes_path and not themes_path.exists():
        st.warning(f"Missing themes file: {themes_path}")
    if dictionary_path and not dictionary_path.exists():
        st.warning(f"Missing dictionary file: {dictionary_path}")

    st.header("Columns")
    id_column = "(none)"
    if df_raw is not None:
        id_column = st.selectbox("ID column (optional)", options=["(none)"] + list(df_raw.columns))
        df_for_run = df_raw.copy()
        if id_column != "(none)" and id_column != "ID":
            df_for_run = df_for_run.rename(columns={id_column: "ID"})

        text_candidates = [
            c for c in df_for_run.select_dtypes(include=["object"]).columns if c != "ID"
        ]
        if not text_candidates:
            text_candidates = [c for c in df_for_run.columns if c != "ID"]
        selected_text_cols = st.multiselect(
            "Text columns",
            options=text_candidates,
            default=text_candidates,
        )

    st.header("Taxonomy")
    taxonomy_mode = st.radio("Mode", ["semantic", "keyword"], horizontal=True)
    semantic_model = st.text_input(
        "Embedding model",
        value=settings.get("semantic", {}).get("model_name", "all-MiniLM-L6-v2"),
    )
    semantic_threshold = st.slider(
        "Similarity threshold",
        0.0,
        1.0,
        value=float(settings.get("semantic", {}).get("similarity_threshold", 0.35)),
        step=0.01,
    )
    semantic_top_k = st.slider(
        "Top-k themes (semantic)",
        1,
        10,
        value=int(settings.get("semantic", {}).get("top_k", 3)),
    )
    keyword_min_score = st.slider(
        "Min accept score (keyword)",
        0.0,
        1.0,
        value=float(settings.get("taxonomy", {}).get("min_accept_score", 0.7)),
        step=0.01,
    )
    keyword_top_k = st.slider(
        "Top-k themes (keyword)",
        1,
        10,
        value=int(settings.get("taxonomy", {}).get("top_k", 5)),
    )

    use_cross_encoder = st.checkbox(
        "Use cross-encoder re-rank (slower)",
        value=bool(settings.get("semantic", {}).get("use_cross_encoder", False)),
    )
    cross_encoder_model = settings.get("semantic", {}).get(
        "cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    bi_top_k = int(settings.get("semantic", {}).get("bi_top_k", 15))
    bi_threshold = float(settings.get("semantic", {}).get("bi_threshold", 0.20))
    if use_cross_encoder:
        cross_encoder_model = st.text_input("Cross-encoder model", value=cross_encoder_model)
        bi_top_k = st.slider("Bi-encoder candidates", 5, 50, value=bi_top_k)
        bi_threshold = st.slider("Bi-encoder threshold", 0.0, 1.0, value=bi_threshold, step=0.01)

    st.header("Sentiment")
    sentiment_enabled = st.checkbox(
        "Enable sentiment analysis",
        value=bool(settings.get("output", {}).get("generate_sentiment", True)),
    )
    pos_threshold = st.slider(
        "Positive threshold",
        -1.0,
        1.0,
        value=float(settings.get("sentiment", {}).get("positive_threshold", 0.10)),
        step=0.01,
    )
    neg_threshold = st.slider(
        "Negative threshold",
        -1.0,
        1.0,
        value=float(settings.get("sentiment", {}).get("negative_threshold", -0.10)),
        step=0.01,
    )
    skip_dismissed = st.checkbox(
        "Skip dismissed responses",
        value=bool(settings.get("sentiment", {}).get("skip_dismissed", True)),
    )

    weight_mode = st.radio("Column weights", ["Auto (equal)", "Custom"], horizontal=True)
    column_weights = {}
    if weight_mode == "Custom" and selected_text_cols:
        default_weight = round(1.0 / max(len(selected_text_cols), 1), 2)
        weight_inputs = {}
        for col in selected_text_cols:
            weight_inputs[col] = st.number_input(
                f"{col} weight",
                min_value=0.0,
                max_value=1.0,
                value=default_weight,
                step=0.05,
            )
        total_weight = sum(weight_inputs.values())
        if total_weight > 0:
            column_weights = {k: v / total_weight for k, v in weight_inputs.items()}
        else:
            st.warning("Weights must sum to more than 0. Using equal weights.")
            column_weights = {}

    st.header("Null Detection")
    min_len = st.number_input(
        "Min meaningful length",
        min_value=1,
        max_value=100,
        value=int(settings.get("null_detection", {}).get("min_meaningful_length", 3)),
    )
    max_len = st.number_input(
        "Max dismissive length",
        min_value=5,
        max_value=500,
        value=int(settings.get("null_detection", {}).get("max_dismissive_length", 50)),
    )

    st.header("Outputs")
    generate_assignments = st.checkbox("Assignments", value=True)
    generate_taxonomy_reports = st.checkbox(
        "Taxonomy matched/unmatched reports",
        value=bool(settings.get("output", {}).get("generate_taxonomy_reports", True)),
    )
    generate_quality_report = st.checkbox(
        "Null text report",
        value=bool(settings.get("output", {}).get("generate_quality_report", True)),
    )
    generate_null_details = st.checkbox(
        "Null text details",
        value=bool(settings.get("output", {}).get("generate_null_text_details", True)),
    )
    analytics_enabled = st.checkbox(
        "Data audit",
        value=bool(settings.get("analytics", {}).get("enabled", True)),
    )

    st.header("Performance")
    use_cache = st.checkbox("Use cached results", value=True)
    if st.button("Clear cached results"):
        st.session_state["run_cache"] = {}
        st.info("Cache cleared.")

    run_btn = st.button("Run pipeline", type="primary", use_container_width=True)


if df_raw is not None:
    st.subheader("Preview")
    st.dataframe(df_raw.head(50), use_container_width=True)

can_run = df_for_run is not None and bool(selected_text_cols)
if taxonomy_mode == "semantic" and (not enriched_path or not enriched_path.exists()):
    can_run = False

if run_btn:
    if not can_run:
        st.error("Upload a CSV and select at least one text column.")
    else:
        run_root, upload_dir, tables_dir, deliverables_dir = _ensure_run_dirs()
        input_path = upload_dir / "input.csv"
        df_for_run.to_csv(input_path, index=False)

        settings = _load_settings_for_profile(profile)
        settings["text_columns"] = selected_text_cols
        settings["null_detection"]["min_meaningful_length"] = int(min_len)
        settings["null_detection"]["max_dismissive_length"] = int(max_len)

        settings["taxonomy"]["min_accept_score"] = float(keyword_min_score)
        settings["taxonomy"]["top_k"] = int(keyword_top_k)

        settings["semantic"]["model_name"] = semantic_model
        settings["semantic"]["similarity_threshold"] = float(semantic_threshold)
        settings["semantic"]["top_k"] = int(semantic_top_k)
        settings["semantic"]["use_cross_encoder"] = bool(use_cross_encoder)
        settings["semantic"]["cross_encoder_model"] = cross_encoder_model
        settings["semantic"]["bi_top_k"] = int(bi_top_k)
        settings["semantic"]["bi_threshold"] = float(bi_threshold)

        settings["sentiment"]["positive_threshold"] = float(pos_threshold)
        settings["sentiment"]["negative_threshold"] = float(neg_threshold)
        settings["sentiment"]["skip_dismissed"] = bool(skip_dismissed)
        if column_weights:
            settings["sentiment"]["column_weights"] = column_weights
        else:
            settings["sentiment"]["column_weights"] = {}

        settings["output"]["generate_assignments"] = bool(generate_assignments)
        settings["output"]["generate_sentiment"] = bool(sentiment_enabled)
        settings["output"]["generate_taxonomy_reports"] = bool(generate_taxonomy_reports)
        settings["output"]["generate_quality_report"] = bool(generate_quality_report)
        settings["output"]["generate_null_text_details"] = bool(generate_null_details)
        settings["analytics"]["enabled"] = bool(analytics_enabled)

        run_cache = st.session_state.setdefault("run_cache", {})
        data_bytes = st.session_state.get("uploaded_bytes") or b""
        run_key = _build_run_key(settings, taxonomy_mode, data_bytes)

        if use_cache and run_key in run_cache:
            cached = run_cache[run_key]
            st.session_state["run_result"] = cached["result"]
            st.session_state["run_logs"] = cached["logs"]
            st.session_state["run_dirs"] = cached["dirs"]
            st.session_state["run_settings"] = cached["settings"]
            st.info("Loaded cached results for this configuration.")
        else:
            log_buffer = io.StringIO()
            with st.spinner("Running pipeline..."):
                with contextlib.redirect_stdout(log_buffer):
                    result = run_pipeline(
                        settings=settings,
                        input_csv=str(input_path),
                        taxonomy_mode=taxonomy_mode,
                        output_dir=str(tables_dir),
                        deliverables_dir=str(deliverables_dir),
                        analytics=analytics_enabled,
                    )

            run_payload = {
                "result": result,
                "logs": log_buffer.getvalue(),
                "dirs": {
                    "root": run_root,
                    "tables": tables_dir,
                    "deliverables": deliverables_dir,
                },
                "settings": settings,
            }
            st.session_state["run_result"] = result
            st.session_state["run_logs"] = log_buffer.getvalue()
            st.session_state["run_dirs"] = run_payload["dirs"]
            st.session_state["run_settings"] = settings
            run_cache[run_key] = run_payload


result = st.session_state.get("run_result")
if result:
    data_df = result.get("data")
    assignments_df = result.get("assignments")
    run_settings = st.session_state.get("run_settings")

    st.header("Results")
    if data_df is not None:
        st.write(f"Rows processed: {len(data_df)}")
    if assignments_df is not None:
        st.write(f"Assignments created: {len(assignments_df)}")

    if assignments_df is not None and not assignments_df.empty:
        st.subheader("Assignments Preview")
        st.dataframe(assignments_df.head(200), use_container_width=True)

    if data_df is not None and "sentiment_label" in data_df.columns:
        st.subheader("Sentiment Distribution")
        sentiment_counts = data_df["sentiment_label"].value_counts()
        st.bar_chart(sentiment_counts)

    run_dirs = st.session_state.get("run_dirs", {})
    output_root = run_dirs.get("tables")
    deliverables_root = run_dirs.get("deliverables")

    st.subheader("Downloads")
    if run_settings:
        st.download_button(
            "Download run config (YAML)",
            data=_dump_settings(run_settings),
            file_name="run_config.yaml",
            mime="text/yaml",
        )
    if output_root:
        for csv_file in sorted(Path(output_root).glob("*.csv")):
            st.download_button(
                label=f"Download {csv_file.name}",
                data=csv_file.read_bytes(),
                file_name=csv_file.name,
                mime="text/csv",
            )
    if deliverables_root:
        for csv_file in sorted(Path(deliverables_root).glob("*.csv")):
            st.download_button(
                label=f"Download deliverable: {csv_file.name}",
                data=csv_file.read_bytes(),
                file_name=csv_file.name,
                mime="text/csv",
            )

    with st.expander("Pipeline Logs"):
        st.text(st.session_state.get("run_logs", ""))


st.header("Taxonomy Builder")
st.caption("Create a theme phrase library CSV and generate enriched taxonomy files.")

template_files = sorted(TEMPLATES_DIR.glob("*.csv")) if TEMPLATES_DIR.exists() else []
template_names = [p.name for p in template_files]
selected_template = st.selectbox("Starter template", options=["(none)"] + template_names)
if selected_template != "(none)":
    template_path = TEMPLATES_DIR / selected_template
    template_bytes = template_path.read_bytes()
    st.download_button(
        "Download template CSV",
        data=template_bytes,
        file_name=selected_template,
        mime="text/csv",
    )
    if st.button("Use template in builder"):
        st.session_state["taxonomy_df"] = pd.read_csv(io.BytesIO(template_bytes))
        st.session_state["taxonomy_name"] = selected_template
        st.session_state["taxonomy_bytes"] = template_bytes
        st.session_state.pop("taxonomy_outputs", None)
        st.rerun()

taxonomy_upload = st.file_uploader("Upload theme_phrase_library.csv", type=["csv"], key="taxonomy_upload")
if taxonomy_upload is not None:
    tax_bytes = taxonomy_upload.getvalue()
    st.session_state["taxonomy_df"] = pd.read_csv(io.BytesIO(tax_bytes))
    st.session_state["taxonomy_name"] = taxonomy_upload.name
    st.session_state["taxonomy_bytes"] = tax_bytes
    st.session_state.pop("taxonomy_outputs", None)

tax_df = st.session_state.get("taxonomy_df")
if tax_df is not None:
    st.subheader("Taxonomy Preview")
    st.dataframe(tax_df.head(50), use_container_width=True)

    min_phrases = st.number_input("Min phrases per subtheme", min_value=1, max_value=10, value=2)
    errors, warnings = _validate_taxonomy_df(tax_df, min_phrases=int(min_phrases))
    if errors:
        for msg in errors:
            st.error(msg)
    if warnings:
        for msg in warnings:
            st.warning(msg)
    if not errors:
        st.success("Validation passed.")

    target_profile = st.text_input("Target profile name", value=profile if "profile" in locals() else "general")
    save_to_project = st.checkbox("Save files to project taxonomy folder", value=False)
    make_enriched = st.checkbox("Generate enriched JSON (semantic)", value=True)
    make_themes = st.checkbox("Generate themes.yaml (keyword)", value=True)
    make_phrase_csv = st.checkbox("Save phrase library CSV", value=True)
    if save_to_project and target_profile not in profiles:
        st.warning("Target profile does not exist yet. Create it in config/profiles/ before running.")

    if st.button("Build taxonomy files", disabled=bool(errors)):
        outputs = {}
        if make_phrase_csv:
            outputs["theme_phrase_library.csv"] = tax_df.to_csv(index=False).encode("utf-8")
        if make_enriched:
            enriched = build_enriched_structure(tax_df)
            outputs["theme_subtheme_dictionary_v3_enriched.json"] = (
                json.dumps(enriched, indent=2).encode("utf-8")
            )
        if make_themes:
            themes_yaml = _build_themes_yaml(tax_df)
            try:
                import yaml
                outputs["themes.yaml"] = yaml.safe_dump(themes_yaml, sort_keys=False).encode("utf-8")
            except Exception:
                outputs["themes.yaml"] = json.dumps(themes_yaml, indent=2).encode("utf-8")

        st.session_state["taxonomy_outputs"] = outputs

        if save_to_project:
            target_dir = ROOT / "assets" / "taxonomy" / target_profile
            target_dir.mkdir(parents=True, exist_ok=True)
            for name, content in outputs.items():
                (target_dir / name).write_bytes(content)
            st.info(f"Saved files to {target_dir}")

    outputs = st.session_state.get("taxonomy_outputs")
    if outputs:
        st.subheader("Download Generated Files")
        for name, content in outputs.items():
            st.download_button(
                f"Download {name}",
                data=content,
                file_name=name,
                mime="application/octet-stream",
            )
