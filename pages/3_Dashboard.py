import streamlit as st

from app_shared import (
    apply_global_styles,
    build_response_long,
    merge_assignments_with_text,
    split_multi,
)


st.set_page_config(page_title="Dashboard", layout="wide")
apply_global_styles()

st.header("Dashboard")
st.caption("Filter and explore responses across themes, sentiment, and match methods.")

result = st.session_state.get("run_result")
if not result:
    st.info("Run the pipeline from the Data & Settings page to explore results.")
    st.stop()

data_df = result.get("data")
assignments_df = result.get("assignments")
run_settings = st.session_state.get("run_settings") or {}
text_columns = run_settings.get("text_columns") or []
delimiter = run_settings.get("taxonomy", {}).get("multi_label", {}).get("delimiter", " | ")

assigned_text_columns = []
if assignments_df is not None and not assignments_df.empty and "TextColumn" in assignments_df.columns:
    assigned_text_columns = sorted(assignments_df["TextColumn"].dropna().astype(str).unique().tolist())
columns_for_responses = assigned_text_columns if assigned_text_columns else text_columns

responses = build_response_long(data_df, columns_for_responses)
view_df = merge_assignments_with_text(assignments_df, responses)

theme_values = view_df["theme"] if "theme" in view_df.columns else []
subtheme_values = view_df["subtheme"] if "subtheme" in view_df.columns else []
themes = sorted({t for val in theme_values for t in split_multi(val, delimiter)})
subthemes = sorted({t for val in subtheme_values for t in split_multi(val, delimiter)})
sentiments = (
    sorted(view_df["sentiment_label"].dropna().astype(str).unique().tolist())
    if "sentiment_label" in view_df.columns
    else []
)
match_methods = (
    sorted(view_df["match_method"].dropna().astype(str).unique().tolist())
    if "match_method" in view_df.columns
    else []
)
available_text_columns = (
    sorted(view_df["TextColumn"].dropna().astype(str).unique().tolist())
    if "TextColumn" in view_df.columns
    else []
)

filter_cols = st.columns(5)
with filter_cols[0]:
    theme_filter = st.multiselect("Theme", options=themes)
with filter_cols[1]:
    subtheme_filter = st.multiselect("Subtheme", options=subthemes)
with filter_cols[2]:
    sentiment_filter = st.multiselect("Sentiment", options=sentiments)
with filter_cols[3]:
    match_filter = st.multiselect("Match method", options=match_methods)
with filter_cols[4]:
    text_col_filter = st.multiselect(
        "Text column",
        options=available_text_columns,
        default=available_text_columns,
    )

only_meaningful = st.checkbox("Only meaningful responses", value=False)
search_text = st.text_input("Search responses")

filtered = view_df.copy()

if text_col_filter and "TextColumn" in filtered.columns:
    filtered = filtered[filtered["TextColumn"].isin(text_col_filter)]

if theme_filter and "theme" in filtered.columns:
    filtered = filtered[filtered["theme"].apply(lambda x: any(t in split_multi(x, delimiter) for t in theme_filter))]

if subtheme_filter and "subtheme" in filtered.columns:
    filtered = filtered[
        filtered["subtheme"].apply(lambda x: any(t in split_multi(x, delimiter) for t in subtheme_filter))
    ]

if sentiment_filter and "sentiment_label" in filtered.columns:
    filtered = filtered[filtered["sentiment_label"].isin(sentiment_filter)]

if match_filter and "match_method" in filtered.columns:
    filtered = filtered[filtered["match_method"].isin(match_filter)]

meaningful_col = next(
    (c for c in ["is_meaningful", "is_meaningful_x", "is_meaningful_y"] if c in filtered.columns),
    None,
)
if only_meaningful and meaningful_col:
    filtered = filtered[filtered[meaningful_col] == True]

if search_text and "text" in filtered.columns:
    filtered = filtered[filtered["text"].astype(str).str.contains(search_text, case=False, na=False)]

metric_cols = st.columns(3)
metric_cols[0].metric("Rows", len(filtered))
metric_cols[1].metric("Unique IDs", filtered["ID"].nunique() if "ID" in filtered.columns else 0)
metric_cols[2].metric("Themes", filtered["theme"].nunique() if "theme" in filtered.columns else 0)

st.subheader("Filtered Responses")
display_cols = [
    col
    for col in [
        "ID",
        "TextColumn",
        "text",
        "theme",
        "subtheme",
        "parent_theme",
        "match_method",
        "rule_score",
        "sentiment_label",
        "compound",
        "response_detail",
    ]
    if col in filtered.columns
]
st.dataframe(filtered[display_cols], use_container_width=True)

st.download_button(
    "Download filtered CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_responses.csv",
    mime="text/csv",
)
