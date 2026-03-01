import streamlit as st

from app_shared import (
    apply_global_styles,
    build_null_summary,
    build_response_detail_counts,
    build_theme_counts,
    dump_settings,
)


st.set_page_config(page_title="Results", layout="wide")
apply_global_styles()

st.header("Results")
st.caption("Review the outputs, null-text metrics, and sentiment distribution.")

result = st.session_state.get("run_result")
if not result:
    st.info("Run the pipeline from the Data & Settings page to see results.")
    st.stop()

data_df = result.get("data")
assignments_df = result.get("assignments")
run_settings = st.session_state.get("run_settings") or {}

text_columns = run_settings.get("text_columns") or []
delimiter = run_settings.get("taxonomy", {}).get("multi_label", {}).get("delimiter", " | ")

metric_cols = st.columns(4)
metric_cols[0].metric("Rows processed", len(data_df) if data_df is not None else 0)
metric_cols[1].metric("Assignments", len(assignments_df) if assignments_df is not None else 0)

if assignments_df is not None and "match_method" in assignments_df.columns:
    matched = assignments_df[assignments_df["match_method"] != "none"]
    dismissed = assignments_df[assignments_df["match_method"] == "dismissed"]
    match_rate = round(len(matched) / max(len(assignments_df), 1) * 100, 2)
    metric_cols[2].metric("Match rate", f"{match_rate}%")
    metric_cols[3].metric("Dismissed", len(dismissed))

st.divider()

st.subheader("Null Text Metrics")
if data_df is not None and text_columns:
    null_summary = build_null_summary(data_df, text_columns)
    st.dataframe(null_summary, use_container_width=True)

    detail_counts = build_response_detail_counts(data_df, text_columns)
    if not detail_counts.empty:
        pivot = detail_counts.pivot(index="response_detail", columns="TextColumn", values="count").fillna(0)
        st.bar_chart(pivot)
else:
    st.info("No null-text data available for this run.")

st.divider()

st.subheader("Theme Coverage")
if assignments_df is not None and not assignments_df.empty:
    theme_counts = build_theme_counts(assignments_df, delimiter)
    if not theme_counts.empty:
        st.dataframe(theme_counts.head(20), use_container_width=True)
        st.bar_chart(theme_counts.set_index("theme").head(12))
    else:
        st.info("No theme counts available.")

st.divider()

if data_df is not None and "sentiment_label" in data_df.columns:
    st.subheader("Sentiment Distribution")
    sentiment_counts = data_df["sentiment_label"].value_counts()
    st.bar_chart(sentiment_counts)

st.divider()

st.subheader("Assignments Preview")
if assignments_df is not None and not assignments_df.empty:
    st.dataframe(assignments_df.head(200), use_container_width=True)

st.divider()

st.subheader("Downloads")
if run_settings:
    st.download_button(
        "Download run config (YAML)",
        data=dump_settings(run_settings),
        file_name="run_config.yaml",
        mime="text/yaml",
    )

run_dirs = st.session_state.get("run_dirs", {})
output_root = run_dirs.get("tables")
deliverables_root = run_dirs.get("deliverables")

if output_root:
    for csv_file in sorted(output_root.glob("*.csv")):
        st.download_button(
            label=f"Download {csv_file.name}",
            data=csv_file.read_bytes(),
            file_name=csv_file.name,
            mime="text/csv",
        )

if deliverables_root:
    for csv_file in sorted(deliverables_root.glob("*.csv")):
        st.download_button(
            label=f"Download deliverable: {csv_file.name}",
            data=csv_file.read_bytes(),
            file_name=csv_file.name,
            mime="text/csv",
        )

with st.expander("Pipeline Logs"):
    st.text(st.session_state.get("run_logs", ""))
