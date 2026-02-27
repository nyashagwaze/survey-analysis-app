import os
import copy
from pathlib import Path
import pandas as pd

from .clean_normalise.clean_normalise import (
    basic_normalise,
    apply_business_map,
    merge_to_canonical,
    force_phrases,
    keep_unigrams,
)

# ---------------------------------------------------
# Config + preprocessing helpers
# ---------------------------------------------------

def _find_project_root(start: Path, max_depth: int = 8):
    candidate = start if start.is_dir() else start.parent
    for _ in range(max_depth):
        if (candidate / "pyproject.toml").exists():
            return candidate
        if (candidate / "src" / "survey_app").exists():
            return candidate
        if (candidate / "config" / "pipeline_settings.yaml").exists():
            return candidate
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return None


def _expand_path_tokens(value: str) -> str:
    if value is None:
        return value
    value = _expand_user_tokens(value)
    value = os.path.expandvars(value)
    return str(Path(value).expanduser())


def _get_project_root(start: Path = None):
    env_root = (
        os.environ.get("PROJECT_ROOT")
        or os.environ.get("PIPELINE_PROJECT_ROOT")
        or os.environ.get("DATABRICKS_PROJECT_ROOT")
    )
    if env_root:
        return Path(_expand_path_tokens(env_root))

    base = start or Path(__file__).resolve()
    root = _find_project_root(base)
    if root:
        return root

    return base if base.is_dir() else base.parent

def _is_databricks_env() -> bool:
    return bool(os.environ.get("DATABRICKS_RUNTIME_VERSION") or os.environ.get("DB_IS_DRIVER"))

def _expand_user_tokens(path: str) -> str:
    if "{user}" not in path:
        return path
    user = (
        os.environ.get("DATABRICKS_USER")
        or os.environ.get("DATABRICKS_USERNAME")
        or os.environ.get("USER")
        or os.environ.get("USERNAME")
    )
    if not user:
        return path
    return path.replace("{user}", user)

def resolve_settings_path(settings_path, project_root: Path = None, pipeline_dir: Path = None) -> Path:
    p = Path(settings_path)
    if p.is_absolute():
        return p
    pipeline_dir = pipeline_dir or Path(__file__).resolve().parent
    project_root = project_root or _get_project_root(start=pipeline_dir)

    candidate = project_root / p
    if candidate.exists():
        return candidate

    candidate = pipeline_dir / p
    if candidate.exists():
        return candidate

    return candidate


def resolve_base_dir(settings: dict, settings_path: str = None, project_root: Path = None, pipeline_dir: Path = None) -> Path:
    pipeline_dir = pipeline_dir or Path(__file__).resolve().parent
    project_root = project_root or _get_project_root(start=pipeline_dir)
    if settings_path:
        resolved_settings = resolve_settings_path(settings_path, project_root=project_root, pipeline_dir=pipeline_dir)
        project_root = _find_project_root(resolved_settings) or project_root

    paths_cfg = (settings or {}).get("paths", {}) or {}
    base_dir_cfg = paths_cfg.get("base_dir")

    if not base_dir_cfg:
        return project_root

    base_dir_cfg = _expand_path_tokens(str(base_dir_cfg))
    base_dir = Path(base_dir_cfg)
    if base_dir.is_absolute():
        return base_dir

    return (project_root / base_dir).resolve()

def apply_runtime_overrides(settings: dict) -> dict:
    """
    Apply environment-specific overrides (e.g., Databricks) based on settings.
    """
    if not settings:
        return settings

    db_cfg = settings.get("databricks", {}) or {}
    enabled = db_cfg.get("enabled", False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower()
        if enabled == "auto":
            enabled = _is_databricks_env()
        else:
            enabled = enabled in {"1", "true", "yes", "on"}

    if not enabled:
        return settings

    paths_cfg = settings.get("paths", {}) or {}
    project_root = db_cfg.get("project_root")
    if project_root:
        project_root = _expand_user_tokens(project_root)
        paths_cfg = dict(paths_cfg)
        paths_cfg["base_dir"] = project_root
        settings["paths"] = paths_cfg

    cache_dir = db_cfg.get("hf_cache")
    if cache_dir and db_cfg.get("set_hf_cache_env", True):
        cache_dir = _expand_user_tokens(cache_dir)
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            pass

        override_env = db_cfg.get("override_env", True)
        def _set_env(key: str, value: str):
            if override_env or key not in os.environ:
                os.environ[key] = value

        _set_env("HF_HOME", cache_dir)
        _set_env("TRANSFORMERS_CACHE", cache_dir)
        _set_env("SENTENCE_TRANSFORMERS_HOME", cache_dir)

    return settings

def _apply_profile_paths(settings: dict) -> dict:
    if not settings:
        return settings
    profile = settings.get("profile")
    if not profile:
        return settings
    paths_cfg = settings.get("paths", {}) or {}
    if not isinstance(paths_cfg, dict):
        return settings
    def _replace(value):
        if isinstance(value, str):
            return value.replace("{profile}", str(profile))
        return value
    settings["paths"] = {k: _replace(v) for k, v in paths_cfg.items()}
    return settings

def _load_profile_overrides(profile: str, project_root: Path = None, pipeline_dir: Path = None) -> dict:
    if not profile:
        return {}
    pipeline_dir = pipeline_dir or Path(__file__).resolve().parent
    project_root = project_root or _get_project_root(start=pipeline_dir)
    profile_path = project_root / "config" / "profiles" / str(profile) / "profile.yaml"
    return load_yaml(profile_path)

def _dedupe_keep_order(items):
    seen = set()
    out = []
    for item in items:
        if item is None:
            continue
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out

def resolve_processing_columns(settings: dict) -> dict:
    """
    Resolve column lists for each processing stage with sensible fallbacks.
    """
    processing_cfg = (settings or {}).get("processing", {}) or {}

    base_cols = processing_cfg.get("text_columns") or (settings or {}).get("text_columns") or []
    tax_cols = processing_cfg.get("taxonomy_columns") or base_cols
    sent_cols = processing_cfg.get("sentiment_columns") or base_cols
    seg_cols = processing_cfg.get("segmentation_columns") or base_cols

    if not base_cols:
        base_cols = _dedupe_keep_order(list(tax_cols or []) + list(sent_cols or []) + list(seg_cols or []))

    return {
        "text_columns": list(base_cols) if base_cols else [],
        "taxonomy_columns": list(tax_cols) if tax_cols else [],
        "sentiment_columns": list(sent_cols) if sent_cols else [],
        "segmentation_columns": list(seg_cols) if seg_cols else [],
    }

def apply_profile(settings: dict, profile: str, project_root: Path = None, pipeline_dir: Path = None) -> dict:
    """
    Apply a profile override on top of existing settings.
    """
    if not settings or not profile:
        return settings
    pipeline_dir = pipeline_dir or Path(__file__).resolve().parent
    project_root = project_root or _get_project_root(start=pipeline_dir)

    merged = copy.deepcopy(settings)
    merged["profile"] = profile

    profile_overrides = _load_profile_overrides(profile, project_root=project_root, pipeline_dir=pipeline_dir)
    if profile_overrides:
        merged = _deep_merge(merged, profile_overrides)

    merged = _apply_profile_paths(merged)
    return merged

def resolve_path(base_dir, relative_path):
    from pathlib import Path
    if relative_path is None:
        return None
    base = Path(base_dir)
    # Contingency for where base_dir is not absolute (can't handle __file__)
    if not base.is_absolute():
        base = base.resolve()
    p = Path(relative_path)
    if p.is_absolute():
        return p
    candidate = base / p
    if candidate.exists():
        return candidate
    alt = base / p.name
    if alt.exists():
        return alt
    return candidate

def load_yaml(path):
    from pathlib import Path
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml
    except Exception as exc:
        raise ImportError("Missing dependency: pyyaml. Install with `pip install pyyaml`.") from exc
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

def _default_settings():
    return {
        "profile": "general",
        "null_detection": {
            "min_meaningful_length": 3,
            "max_dismissive_length": 50,
        },
        "taxonomy": {
            "use_fuzzy": True,
            "fuzzy_threshold": 0.70,
            "min_accept_score": 0.70,
            "top_k": 5,
            "column_bonus": 0.25,
            "column_penalty": 0.10,
            "prefer_exact": True,
            "phrase_first": True,
            "allow_token_fallback": False,
            "multi_label": {
                "enabled": True,
                "delimiter": " | ",
                "order_by": "score",
            },
        },
        "semantic": {
            "model_name": "all-MiniLM-L6-v2",
            "similarity_threshold": 0.35,
            "top_k": 3,
            "use_cross_encoder": False,
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "bi_top_k": 15,
            "bi_threshold": 0.20,
            "pair_template": "{phrase}",
            "scores_output": None,
        },
        "sentiment": {
            "positive_threshold": 0.05,
            "negative_threshold": -0.05,
            "column_weights": {},
            "skip_dismissed": True,
            "dismissed_sentiment_value": None,
        },
        "preprocessing": {
            "force_phrases_enabled": True,
            "unigram_whitelist_enabled": True,
            "always_drop_unigrams": True,
            "must_be_phrase_enabled": True,
        },
        "output": {
            "generate_assignments": True,
            "generate_sentiment": True,
            "generate_quality_report": True,
            "generate_null_text_details": True,
            "null_text_details_filename": "null_text_details.csv",
            "generate_taxonomy_reports": True,
            "taxonomy_matched_filename": "taxonomy_matched.csv",
            "taxonomy_unmatched_filename": "taxonomy_unmatched.csv",
            "include_dismissed_in_assignments": True,
            "dismissed_theme_label": "Brief response",
            "dismissed_subtheme_from": "response_detail",
        },
        "analytics": {
            "enabled": False,
            "export_csv": True,
            "audit_filename": "data_audit.csv",
        },
        "databricks": {
            "enabled": False,
            "project_root": None,
            "hf_cache": None,
            "set_hf_cache_env": True,
            "override_env": True,
            "spark_copy_workspace_to_dbfs": False,
            "spark_dbfs_dir": "dbfs:/tmp/survey_app",
        },
        "text_columns": ["Free_Text_1", "Free_Text_2", "Free_Text_3", "Free_Text_4", "Free_Text_5"],
        "paths": {
            # base_dir is resolved relative to the project root (or absolute)
            "base_dir": ".",
            "input_csv": None,
            "dictionary": "config/profiles/{profile}/dictionary.yaml",
            "themes": "config/profiles/{profile}/themes.yaml",
            "pipeline_settings": "config/pipeline_settings.yaml",
            "enriched_json": "assets/taxonomy/{profile}/theme_subtheme_dictionary_v3_enriched.json",
            "output_tables": "outputs/tables",
            "deliverables": "Deliverables",
        },
    }

def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(base[k], v)
        else:
            base[k] = v
    return base

def load_settings(settings_path: str = "config/pipeline_settings.yaml", validate: bool = True, apply_defaults: bool = True):
    pipeline_dir = Path(__file__).resolve().parent
    project_root = _get_project_root(start=pipeline_dir)
    resolved = resolve_settings_path(settings_path, project_root=project_root, pipeline_dir=pipeline_dir)
    data = load_yaml(resolved)

    if apply_defaults:
        merged = _deep_merge(_default_settings(), data)
    else:
        merged = data

    profile_overrides = _load_profile_overrides(merged.get("profile"), project_root=project_root, pipeline_dir=pipeline_dir)
    if profile_overrides:
        merged = _deep_merge(merged, profile_overrides)

    settings = apply_runtime_overrides(merged)
    settings = _apply_profile_paths(settings)

    if validate:
        text_columns = settings.get("text_columns")
        if not isinstance(text_columns, list) or not text_columns:
            raise ValueError("Invalid settings: text_columns must be a non-empty list.")
        if "paths" not in settings or not isinstance(settings["paths"], dict):
            raise ValueError("Invalid settings: paths must be a mapping.")

    return settings

def load_dictionary_config(path):
    cfg = load_yaml(path)
    business_map = {str(k).lower(): str(v).lower() for k, v in (cfg.get("BUSINESS_MAP") or {}).items()}
    merge_map = {str(k).lower(): str(v).lower() for k, v in (cfg.get("MERGE_TO_CANONICAL") or {}).items()}

    force_phrases = [str(x).lower() for x in (cfg.get("FORCE_PHRASES") or [])]
    unigram_whitelist = set(str(x).lower() for x in (cfg.get("UNIGRAM_WHITELIST") or []))
    unigram_whitelist |= set(str(x).lower() for x in (cfg.get("DOMAIN_WHITELIST") or []))
    unigram_whitelist |= set(str(x).lower() for x in (cfg.get("ACCRONYM_WHITELIST") or []))

    always_drop = set(str(x).lower() for x in (cfg.get("ALWAYS_DROP_UNIGRAMS") or []))
    must_be_phrase = set(str(x).lower() for x in (cfg.get("MUST_BE_PHRASE") or []))

    return {
        "business_map": business_map,
        "merge_map": merge_map,
        "force_phrases": force_phrases,
        "unigram_whitelist": unigram_whitelist,
        "always_drop": always_drop,
        "must_be_phrase": must_be_phrase,
    }

def preprocess_text(text, assets, settings):
    t = basic_normalise(text)
    if t == "no entry":
        return t

    if assets.get("business_map"):
        t = apply_business_map(t, assets["business_map"])
    if assets.get("merge_map"):
        t = merge_to_canonical(t, assets["merge_map"])

    prep_cfg = (settings or {}).get("preprocessing", {})
    if prep_cfg.get("force_phrases_enabled", True):
        t = force_phrases(t, assets.get("force_phrases", []))

    use_unigrams = prep_cfg.get("unigram_whitelist_enabled", True)
    use_drop = prep_cfg.get("always_drop_unigrams", True)
    use_must_phrase = prep_cfg.get("must_be_phrase_enabled", True)

    if use_unigrams or use_drop or use_must_phrase:
        t = keep_unigrams(
            t,
            assets.get("unigram_whitelist", set()) if use_unigrams else set(),
            assets.get("always_drop", set()) if use_drop else set(),
            assets.get("must_be_phrase", set()) if use_must_phrase else set()
        )

    return t

def preprocess_dataframe(df, text_columns, assets, settings):
    df_out = df.copy()
    for col in text_columns:
        if col not in df_out.columns:
            continue
        df_out[f"{col}_processed"] = df_out[col].apply(lambda x: preprocess_text(x, assets, settings))
    return df_out

def data_audit_summary(df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
    rows = []
    total_rows = len(df)

    for col in text_columns:
        if col not in df.columns:
            rows.append({
                "column": col,
                "total_rows": total_rows,
                "missing_column": True,
                "null_count": None,
                "empty_or_whitespace": None,
                "empty_only": None,
                "no_entry_count": None,
            })
            continue

        s = df[col]
        null_count = int(s.isna().sum())
        s_clean = s.fillna("")
        empty_or_whitespace = int(s_clean.astype(str).str.strip().eq("").sum())
        empty_only = max(0, empty_or_whitespace - null_count)
        no_entry_count = int(s_clean.astype(str).str.strip().str.lower().eq("no entry").sum())

        rows.append({
            "column": col,
            "total_rows": total_rows,
            "missing_column": False,
            "null_count": null_count,
            "empty_or_whitespace": empty_or_whitespace,
            "empty_only": empty_only,
            "no_entry_count": no_entry_count,
        })

    return pd.DataFrame(rows)

def build_null_text_details(df: pd.DataFrame, text_columns: list, id_col: str = "ID") -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        row_id = row[id_col] if id_col in df.columns else idx
        for col in text_columns:
            meaningful_col = f"{col}_is_meaningful"
            detail_col = f"{col}_response_detail"
            if meaningful_col not in df.columns:
                continue
            is_meaningful = row.get(meaningful_col, True)
            if is_meaningful:
                continue
            rows.append({
                "ID": row_id,
                "TextColumn": col,
                "text": row.get(col, ""),
                "response_detail": row.get(detail_col, None),
                "is_meaningful": False
            })
    return pd.DataFrame(rows)

def generate_taxonomy_reports(
    assignments: pd.DataFrame,
    output_path,
    deliverables_path,
    taxonomy_mode: str,
    matched_filename: str = None,
    unmatched_filename: str = None
):
    from pathlib import Path
    if assignments is None or assignments.empty:
        return

    matched_filename = matched_filename or f"taxonomy_matched_{taxonomy_mode}.csv"
    unmatched_filename = unmatched_filename or f"taxonomy_unmatched_{taxonomy_mode}.csv"

    reason_col = "reason" if "reason" in assignments.columns else None
    if reason_col:
        matched = assignments[assignments[reason_col] == "matched"]
        unmatched = assignments[assignments[reason_col] != "matched"]
    else:
        matched = assignments[assignments.get("match_method", "") != "none"]
        unmatched = assignments[assignments.get("match_method", "") == "none"]

    matched.to_csv(Path(output_path) / matched_filename, index=False)
    unmatched.to_csv(Path(output_path) / unmatched_filename, index=False)

    if deliverables_path is not None:
        matched.to_csv(Path(deliverables_path) / matched_filename, index=False)
        unmatched.to_csv(Path(deliverables_path) / unmatched_filename, index=False)
