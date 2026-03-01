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

from survey_app import config_runtime as cfg
from survey_app.taxonomy.synthetic_generation.transfer import build_enriched_structure


def apply_global_styles():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

:root {
  --bg: #0B1220;
  --panel: #101B2F;
  --ink: #F8FAFC;
  --muted: #E2E8F0;
  --accent: #38BDF8;
  --accent-2: #22D3EE;
  --stroke: #243B5A;
  --shadow: rgba(4, 10, 22, 0.7);
  --button-bg: #CBD5E1;
  --button-hover: #94A3B8;
  --button-text: #0B1220;
}

html, body, [class*="css"]  {
  font-family: 'Space Grotesk', sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(1200px 600px at 6% -10%, #162742 0%, rgba(22, 39, 66, 0) 60%),
    radial-gradient(1000px 600px at 108% 0%, #0E2E40 0%, rgba(14, 46, 64, 0) 58%),
    var(--bg);
}

h1, h2, h3, .hero-title {
  font-family: 'Fraunces', serif;
  letter-spacing: -0.5px;
}

.block-container {
  padding-top: 2.4rem;
}

.hero {
  background: linear-gradient(135deg, rgba(17, 27, 46, 0.95), rgba(12, 22, 39, 0.98));
  border: 1px solid var(--stroke);
  border-radius: 24px;
  padding: 28px 32px;
  box-shadow: 0 18px 40px var(--shadow);
  animation: fadeUp 700ms ease;
}

.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(45, 212, 191, 0.15);
  color: #7CE7DB;
  font-weight: 600;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  border: 1px solid rgba(45, 212, 191, 0.35);
}

.hero-subtitle {
  color: var(--muted);
  font-size: 1.02rem;
  margin-top: 0.4rem;
}

.step-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  margin-top: 18px;
}

.step-card {
  background: #0F1A2B;
  border: 1px solid #1B2B45;
  border-radius: 16px;
  padding: 12px 14px;
  font-size: 0.9rem;
  color: var(--ink);
}

.step-card strong {
  color: var(--ink);
}

[data-testid="stSidebar"] {
  background: #0F172A;
  border-right: 1px solid var(--stroke);
}

[data-testid="stSidebar"] .block-container {
  padding-top: 1.6rem;
}

.stButton > button,
.stDownloadButton > button,
button[kind="primary"],
button[kind="secondary"] {
  background: var(--button-bg) !important;
  color: var(--button-text) !important;
  border: 1px solid #94A3B8 !important;
  border-radius: 12px;
  padding: 0.6rem 1rem;
  font-weight: 600;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.35);
}

.stButton > button *,
.stDownloadButton > button *,
button[kind="primary"] *,
button[kind="secondary"] * {
  color: var(--button-text) !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
button[kind="primary"]:hover,
button[kind="secondary"]:hover {
  background: var(--button-hover) !important;
  color: var(--button-text) !important;
}

.stTabs [data-baseweb="tab"] {
  font-weight: 600;
}

div[data-testid="stMetric"] {
  background: #0F1A2B;
  border: 1px solid var(--stroke);
  padding: 12px;
  border-radius: 16px;
}

label, .stMarkdown, .stText, .stCaption, .stAlert, .stMetric {
  color: var(--ink);
}

[data-testid="stCaption"] {
  color: var(--muted);
}

input, textarea, select {
  color: var(--ink) !important;
}

/* Sidebar text and section headers */
[data-testid="stSidebar"] * {
  color: var(--ink) !important;
}

[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stDownloadButton > button,
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] button[kind="secondary"] {
  color: var(--button-text) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
  background: #0F1A2B;
  border: 1px dashed #2B3C5A;
  border-radius: 12px;
  padding: 12px;
}

[data-testid="stFileUploader"] button {
  background: var(--button-bg) !important;
  color: var(--button-text) !important;
  border: 1px solid #94A3B8 !important;
}

[data-testid="stFileUploader"] * {
  color: var(--ink) !important;
}

.fade-in {
  animation: fadeUp 700ms ease;
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
<div class="hero">
  <div class="hero-badge">Survey Analysis Studio</div>
  <h1 class="hero-title">Survey Analysis App</h1>
  <div class="hero-subtitle">
    Upload a CSV, tune taxonomy and sentiment, and export deliverables without exposing raw text.
  </div>
  <div class="step-grid">
    <div class="step-card"><strong>1. Upload</strong><br/>Drop your survey CSV.</div>
    <div class="step-card"><strong>2. Configure</strong><br/>Pick profiles, columns, and thresholds.</div>
    <div class="step-card"><strong>3. Run</strong><br/>Generate assignments and reports.</div>
    <div class="step-card"><strong>4. Export</strong><br/>Download tables for reporting.</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def discover_profiles():
    profiles_dir = ROOT / "config" / "profiles"
    if not profiles_dir.exists():
        return []
    return sorted([p.name for p in profiles_dir.iterdir() if p.is_dir()])


def build_sample_df(rows: int = 12) -> pd.DataFrame:
    columns = ["Free_Text_1", "Free_Text_2", "Free_Text_3"]
    data = {}
    for col in columns:
        data[col] = [f"Sample response {i + 1} for {col}" for i in range(rows)]
    return pd.DataFrame(data)


def ensure_run_dirs():
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    run_root = ROOT / "outputs" / "app_runs" / run_id
    upload_dir = run_root / "uploads"
    tables_dir = run_root / "tables"
    deliverables_dir = run_root / "deliverables"
    upload_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    deliverables_dir.mkdir(parents=True, exist_ok=True)
    return run_root, upload_dir, tables_dir, deliverables_dir


def load_settings_for_profile(profile: str) -> dict:
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


def resolve_asset(settings: dict, key: str):
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


def validate_taxonomy_df(df: pd.DataFrame, min_phrases: int = 2):
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


def build_themes_yaml(df: pd.DataFrame) -> dict:
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


def settings_fingerprint(settings: dict) -> str:
    payload = json.dumps(settings, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_run_key(settings: dict, taxonomy_mode: str, data_bytes: bytes) -> str:
    data_hash = hashlib.sha256(data_bytes or b"").hexdigest()
    settings_hash = settings_fingerprint(settings)
    return f"{taxonomy_mode}:{data_hash}:{settings_hash}"


def dump_settings(settings: dict) -> bytes:
    try:
        import yaml
        return yaml.safe_dump(settings, sort_keys=False).encode("utf-8")
    except Exception:
        return json.dumps(settings, indent=2, default=str).encode("utf-8")


def split_multi(value, delimiter: str):
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(delimiter) if part.strip()]


def build_theme_counts(assignments: pd.DataFrame, delimiter: str) -> pd.DataFrame:
    if assignments is None or assignments.empty or "theme" not in assignments.columns:
        return pd.DataFrame(columns=["theme", "count"])
    themes = []
    for val in assignments["theme"].fillna(""):
        themes.extend(split_multi(val, delimiter))
    cleaned = [t for t in themes if t and t.lower() != "brief response"]
    if not cleaned:
        return pd.DataFrame(columns=["theme", "count"])
    counts = pd.Series(cleaned).value_counts().reset_index()
    counts.columns = ["theme", "count"]
    return counts


def build_null_summary(df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
    rows = []
    total_rows = len(df)
    for col in text_columns:
        meaningful_col = f"{col}_is_meaningful"
        detail_col = f"{col}_response_detail"
        if col not in df.columns:
            rows.append(
                {
                    "TextColumn": col,
                    "total_rows": total_rows,
                    "meaningful": None,
                    "dismissed": None,
                    "dismissed_pct": None,
                    "detail_column": detail_col,
                }
            )
            continue
        meaningful = int(df[meaningful_col].sum()) if meaningful_col in df.columns else None
        dismissed = total_rows - meaningful if meaningful is not None else None
        dismissed_pct = (dismissed / total_rows * 100) if dismissed is not None and total_rows else None
        rows.append(
            {
                "TextColumn": col,
                "total_rows": total_rows,
                "meaningful": meaningful,
                "dismissed": dismissed,
                "dismissed_pct": round(dismissed_pct, 2) if dismissed_pct is not None else None,
                "detail_column": detail_col,
            }
        )
    return pd.DataFrame(rows)


def build_response_detail_counts(df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
    rows = []
    for col in text_columns:
        detail_col = f"{col}_response_detail"
        if detail_col not in df.columns:
            continue
        counts = df[detail_col].fillna("Unknown").value_counts()
        for detail, count in counts.items():
            rows.append(
                {
                    "TextColumn": col,
                    "response_detail": str(detail),
                    "count": int(count),
                }
            )
    return pd.DataFrame(rows)


def build_response_long(df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        row_id = row["ID"] if "ID" in df.columns else idx
        for col in text_columns:
            if col not in df.columns:
                continue
            rows.append(
                {
                    "ID": row_id,
                    "TextColumn": col,
                    "text": row.get(col, ""),
                    "is_meaningful": row.get(f"{col}_is_meaningful"),
                    "response_detail": row.get(f"{col}_response_detail"),
                    "sentiment_label": row.get("sentiment_label"),
                    "compound": row.get("compound"),
                }
            )
    return pd.DataFrame(rows)


def merge_assignments_with_text(assignments: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
    if assignments is None or assignments.empty:
        return responses
    if responses is None or responses.empty:
        return assignments
    return assignments.merge(responses, on=["ID", "TextColumn"], how="left")
