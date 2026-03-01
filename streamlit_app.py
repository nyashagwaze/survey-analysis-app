import streamlit as st

from app_shared import apply_global_styles, render_hero


st.set_page_config(page_title="Survey Analysis App", layout="wide")
apply_global_styles()
render_hero()

st.markdown("### Choose a workspace")
st.markdown(
    """
Use the left sidebar to navigate:

- **Data & Settings**: upload CSVs, configure taxonomy and sentiment, and run.
- **Results**: inspect summaries, null-text metrics, and download outputs.
- **Dashboard**: filter and explore responses interactively.
- **Taxonomy Builder**: validate and generate taxonomy assets.
"""
)
