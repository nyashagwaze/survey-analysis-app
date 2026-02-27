"""
Cross-Encoder Re-Ranking for Taxonomy Assignment

Workflow:
1) Use semantic bi-encoder to retrieve top-N candidate themes
2) Re-rank candidates with a cross-encoder for better precision
3) Emit final theme/subtheme assignments
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from .semantic_taxonomy import SemanticTaxonomyMatcher

try:
    from sentence_transformers import CrossEncoder
except Exception as exc:
    raise ImportError("Missing dependency: sentence-transformers (CrossEncoder)") from exc

_CROSS_ENCODER_CACHE = {}


def _get_cross_encoder(model_name: str):
    if model_name in _CROSS_ENCODER_CACHE:
        return _CROSS_ENCODER_CACHE[model_name]
    model = CrossEncoder(model_name)
    _CROSS_ENCODER_CACHE[model_name] = model
    return model


def _build_pairs(text: str, candidates: List[Dict[str, Any]], template: str) -> List[str]:
    pairs = []
    for c in candidates:
        phrase = c.get("matched_phrase") or c.get("phrase") or ""
        theme = c.get("theme", "")
        subtheme = c.get("subtheme", "")
        parent = c.get("parent_theme", "")
        if template:
            rhs = template.format(
                phrase=phrase,
                theme=theme,
                subtheme=subtheme,
                parent_theme=parent
            )
        else:
            rhs = phrase or f"{theme} | {subtheme}"
        pairs.append((text, rhs))
    return pairs


def assign_taxonomy_semantic_cross_encoder(
    df_spark,
    text_columns: List[str],
    enriched_json_path: Path,
    model_name: str = "all-MiniLM-L6-v2",
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    bi_top_k: int = 15,
    bi_threshold: float = 0.20,
    top_k: int = 3,
    pair_template: str = "{phrase}",
    scores_output: Optional[Path] = None,
    candidates_output: Optional[Path] = None,
    delimiter: str = " | "
) -> pd.DataFrame:
    """
    Assign taxonomy using semantic retrieval + cross-encoder reranking.

    Args:
        df_spark: Spark or Pandas DataFrame.
        text_columns: List of text column names.
        enriched_json_path: Enriched theme dictionary JSON.
        model_name: Bi-encoder model name or path.
        cross_encoder_model: Cross-encoder model name.
        bi_top_k: Number of candidates retrieved by bi-encoder.
        bi_threshold: Minimum bi-encoder similarity.
        top_k: Final top-k after reranking.
        pair_template: Template for candidate text (uses {phrase}, {theme}, {subtheme}, {parent_theme}).
        scores_output: Optional CSV path to write candidate scores.
        candidates_output: Optional CSV path to write bi-encoder candidates.
        delimiter: Multi-label delimiter.
    """
    print("\n" + "=" * 80)
    print("SEMANTIC TAXONOMY (CROSS-ENCODER RERANK)")
    print("=" * 80)

    matcher = SemanticTaxonomyMatcher(
        enriched_json_path=enriched_json_path,
        model_name=model_name,
        similarity_threshold=bi_threshold,
        top_k=bi_top_k
    )

    if hasattr(df_spark, "toPandas"):
        df_pd = df_spark.toPandas()
    elif isinstance(df_spark, pd.DataFrame):
        df_pd = df_spark.copy()
    else:
        df_pd = pd.DataFrame(df_spark)

    all_texts = []
    all_columns = []
    all_ids = []
    all_sentiments = []
    all_is_meaningful = []
    all_response_detail = []

    for col in text_columns:
        processed_col = f"{col}_processed"
        meaningful_col = f"{col}_is_meaningful"

        for _, row in df_pd.iterrows():
            text = row.get(processed_col, None)
            if text is None or (isinstance(text, float) and pd.isna(text)):
                text = row.get(col, "")

            meaningful_val = row.get(meaningful_col, None)
            if meaningful_val is None or (isinstance(meaningful_val, float) and pd.isna(meaningful_val)):
                is_meaningful = None
            else:
                is_meaningful = bool(meaningful_val)

            response_detail = row.get(f"{col}_response_detail", None)

            all_texts.append(text if pd.notna(text) else "")
            all_columns.append(col)
            row_id = row["ID"] if "ID" in df_pd.columns else row.name
            all_ids.append(row_id)
            all_sentiments.append(row.get("sentiment_label", None))
            all_is_meaningful.append(is_meaningful)
            all_response_detail.append(response_detail)

    all_candidates = matcher.match_batch(
        texts=all_texts,
        columns=all_columns,
        sentiment_labels=all_sentiments
    )

    if candidates_output:
        cand_rows = []
        for i, cand_list in enumerate(all_candidates):
            for c in cand_list:
                cand_rows.append({
                    "ID": all_ids[i],
                    "TextColumn": all_columns[i],
                    "text": all_texts[i],
                    **c
                })
        pd.DataFrame(cand_rows).to_csv(candidates_output, index=False)

    ce = _get_cross_encoder(cross_encoder_model)
    score_rows = []
    assignments = []

    for i, cand_list in enumerate(all_candidates):
        text_id = all_ids[i]
        text_col = all_columns[i]
        is_meaningful = all_is_meaningful[i]
        response_detail = all_response_detail[i]
        text = all_texts[i]

        if is_meaningful is False:
            assignments.append({
                "ID": text_id,
                "TextColumn": text_col,
                "is_meaningful": False,
                "theme": "Brief response",
                "subtheme": response_detail or "No",
                "parent_theme": "Brief response",
                "rule_score": 0.0,
                "expected_polarity": "",
                "match_method": "dismissed",
                "evidence": "",
                "reason": f"dismissed:{response_detail}" if response_detail else "dismissed"
            })
            continue

        if not cand_list:
            assignments.append({
                "ID": text_id,
                "TextColumn": text_col,
                "is_meaningful": True,
                "theme": "",
                "subtheme": "",
                "parent_theme": "",
                "rule_score": 0.0,
                "expected_polarity": "",
                "match_method": "none",
                "evidence": "",
                "reason": "below_threshold"
            })
            continue

        pairs = _build_pairs(text, cand_list, pair_template)
        scores = ce.predict(pairs)

        for c, s in zip(cand_list, scores):
            c["ce_score"] = float(s)
            if scores_output:
                score_rows.append({
                    "ID": text_id,
                    "TextColumn": text_col,
                    "text": text,
                    "theme": c.get("theme", ""),
                    "subtheme": c.get("subtheme", ""),
                    "parent_theme": c.get("parent_theme", ""),
                    "matched_phrase": c.get("matched_phrase", ""),
                    "bi_score": c.get("score", None),
                    "ce_score": float(s),
                })

        ranked = sorted(cand_list, key=lambda x: x.get("ce_score", 0.0), reverse=True)[: int(top_k)]

        themes = delimiter.join([c.get("theme", "") for c in ranked])
        subthemes = delimiter.join([c.get("subtheme", "") for c in ranked])
        parents = delimiter.join([c.get("parent_theme", "") for c in ranked])
        evidence = delimiter.join([c.get("matched_phrase", "") for c in ranked if c.get("matched_phrase")])
        avg_score = sum(c.get("ce_score", 0.0) for c in ranked) / max(len(ranked), 1)

        assignments.append({
            "ID": text_id,
            "TextColumn": text_col,
            "is_meaningful": True,
            "theme": themes,
            "subtheme": subthemes,
            "parent_theme": parents,
            "rule_score": avg_score,
            "expected_polarity": ranked[0].get("polarity", "") if ranked else "",
            "match_method": "semantic_cross_encoder",
            "evidence": evidence,
            "reason": "matched"
        })

    out_df = pd.DataFrame(assignments)

    if scores_output:
        pd.DataFrame(score_rows).to_csv(scores_output, index=False)

    return out_df
