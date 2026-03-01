import contextlib
import io

import pandas as pd
import streamlit as st

from app_shared import (
    apply_global_styles,
    build_run_key,
    build_sample_df,
    discover_profiles,
    dump_settings,
    ensure_run_dirs,
    load_settings_for_profile,
    resolve_asset,
)
from survey_app.pipeline import run_pipeline


st.set_page_config(page_title="Data & Settings", layout="wide")
apply_global_styles()

st.header("Data & Settings")
st.caption("Upload survey data, configure the pipeline, and run the analysis.")

profiles = discover_profiles()
default_profile = "general" if "general" in profiles else (profiles[0] if profiles else "general")
if not profiles:
    profiles = [default_profile]

df_raw = None
df_for_run = None
selected_text_cols = []

col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("Upload Data")
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
        sample_df = build_sample_df()
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

    if df_raw is not None:
        st.subheader("Preview")
        st.dataframe(df_raw.head(50), use_container_width=True)

with col_right:
    st.subheader("Profile")
    profile = st.selectbox("Taxonomy profile", options=profiles, index=profiles.index(default_profile))
    settings = load_settings_for_profile(profile)

    enriched_path = resolve_asset(settings, "enriched_json")
    themes_path = resolve_asset(settings, "themes")
    dictionary_path = resolve_asset(settings, "dictionary")

    if enriched_path and not enriched_path.exists():
        st.warning(f"Missing enriched taxonomy: {enriched_path}")
    if themes_path and not themes_path.exists():
        st.warning(f"Missing themes file: {themes_path}")
    if dictionary_path and not dictionary_path.exists():
        st.warning(f"Missing dictionary file: {dictionary_path}")

    st.subheader("Columns")
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

st.divider()

with st.expander("Taxonomy Settings", expanded=True):
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

with st.expander("Sentiment Settings"):
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

with st.expander("Null Detection"):
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

with st.expander("Outputs"):
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

with st.expander("Performance"):
    use_cache = st.checkbox("Use cached results", value=True)
    if st.button("Clear cached results"):
        st.session_state["run_cache"] = {}
        st.info("Cache cleared.")

def _execute_run(
    df_input: pd.DataFrame,
    text_cols: list,
    run_mode: str,
    fast_demo: bool = False,
):
    run_root, upload_dir, tables_dir, deliverables_dir = ensure_run_dirs()
    input_path = upload_dir / "input.csv"
    df_input.to_csv(input_path, index=False)

    settings = load_settings_for_profile(profile)
    settings["text_columns"] = text_cols
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
    settings.setdefault("run_metadata", {})["taxonomy_mode"] = run_mode

    if fast_demo:
        settings["output"]["generate_sentiment"] = True
        settings["output"]["generate_quality_report"] = False
        settings["output"]["generate_null_text_details"] = False
        settings["output"]["generate_taxonomy_reports"] = True
        settings["analytics"]["enabled"] = False

    run_cache = st.session_state.setdefault("run_cache", {})
    data_bytes = df_input.to_csv(index=False).encode("utf-8")
    run_key = build_run_key(settings, run_mode, data_bytes)

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
                    taxonomy_mode=run_mode,
                    output_dir=str(tables_dir),
                    deliverables_dir=str(deliverables_dir),
                    analytics=settings["analytics"]["enabled"],
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
        st.success("Run complete. Open the Results page to inspect outputs.")

    st.download_button(
        "Download run config (YAML)",
        data=dump_settings(settings),
        file_name="run_config.yaml",
        mime="text/yaml",
    )


can_run = df_for_run is not None and bool(selected_text_cols)
if taxonomy_mode == "semantic" and (not enriched_path or not enriched_path.exists()):
    can_run = False

btn_col1, btn_col2 = st.columns(2)
demo_btn = btn_col1.button("Demo run (sample data)", use_container_width=True)
run_btn = btn_col2.button("Run pipeline", type="primary", use_container_width=True)

if demo_btn:
    demo_df = build_sample_df()
    demo_cols = list(demo_df.columns)
    if not themes_path or not themes_path.exists():
        st.error("Keyword demo requires themes.yaml. Select a profile with keyword taxonomy.")
    else:
        _execute_run(demo_df, demo_cols, run_mode="keyword", fast_demo=True)

if run_btn:
    if not can_run:
        st.error("Upload a CSV and select at least one text column.")
    else:
        _execute_run(df_for_run, selected_text_cols, run_mode=taxonomy_mode, fast_demo=False)
