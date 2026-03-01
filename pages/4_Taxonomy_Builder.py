import io
import json

import pandas as pd
import streamlit as st

from app_shared import (
    TEMPLATES_DIR,
    apply_global_styles,
    build_themes_yaml,
    build_enriched_structure,
    discover_profiles,
    validate_taxonomy_df,
)


st.set_page_config(page_title="Taxonomy Builder", layout="wide")
apply_global_styles()

st.header("Taxonomy Builder")
st.caption("Create a theme phrase library CSV and generate enriched taxonomy files.")

profiles = discover_profiles()
default_profile = "general" if "general" in profiles else (profiles[0] if profiles else "general")
if not profiles:
    profiles = [default_profile]

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
    errors, warnings = validate_taxonomy_df(tax_df, min_phrases=int(min_phrases))
    if errors:
        for msg in errors:
            st.error(msg)
    if warnings:
        for msg in warnings:
            st.warning(msg)
    if not errors:
        st.success("Validation passed.")

    target_profile = st.text_input("Target profile name", value=default_profile)
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
            themes_yaml = build_themes_yaml(tax_df)
            try:
                import yaml
                outputs["themes.yaml"] = yaml.safe_dump(themes_yaml, sort_keys=False).encode("utf-8")
            except Exception:
                outputs["themes.yaml"] = json.dumps(themes_yaml, indent=2).encode("utf-8")

        st.session_state["taxonomy_outputs"] = outputs

        if save_to_project:
            target_dir = TEMPLATES_DIR.parent / target_profile
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
