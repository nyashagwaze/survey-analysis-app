"""
Contrast-aware segmentation and per-segment taxonomy/sentiment analysis.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from ..sentiment.sentiment_module import split_on_contrast, roberta_compound, label_from_compound
from ..taxonomy.semantic_taxonomy import SemanticTaxonomyMatcher
from ..taxonomy.keyword_taxonomy import TaxonomyMatcher


def _token_count(text: str) -> int:
    if not text or not isinstance(text, str):
        return 0
    return len(text.strip().split())


def build_segments(
    df: pd.DataFrame,
    text_columns: List[str],
    segmentation_cfg: dict
) -> pd.DataFrame:
    """
    Build a segment-level DataFrame from contrast-aware splitting.
    """
    rows = []
    min_tokens = int(segmentation_cfg.get("min_tokens", 3))
    detect_contrast = bool(segmentation_cfg.get("detect_contrast", True))
    fallback_to_whole_text = bool(segmentation_cfg.get("fallback_to_whole_text", True))

    for idx, row in df.iterrows():
        row_id = row["ID"] if "ID" in df.columns else idx
        for col in text_columns:
            if col not in df.columns and f"{col}_processed" not in df.columns:
                continue

            text = row.get(f"{col}_processed", None)
            if text is None or (isinstance(text, float) and pd.isna(text)):
                text = row.get(col, "")

            is_meaningful = row.get(f"{col}_is_meaningful", None)
            response_detail = row.get(f"{col}_response_detail", None)

            # Dismissed text: keep a single segment if fallback enabled
            if is_meaningful is False:
                if fallback_to_whole_text:
                    segments = [text]
                else:
                    segments = []
            else:
                if detect_contrast:
                    segments = split_on_contrast(text)
                else:
                    segments = [text]

                segments = [s for s in segments if _token_count(s) >= min_tokens]
                if not segments and fallback_to_whole_text:
                    segments = [text] if text else []

            for seg_idx, seg in enumerate(segments):
                rows.append({
                    "ID": row_id,
                    "TextColumn": col,
                    "segment_index": seg_idx,
                    "segment_text": seg,
                    "source_text": text,
                    "segment_tokens": _token_count(seg),
                    "is_meaningful": is_meaningful,
                    "response_detail": response_detail,
                })

    return pd.DataFrame(rows)


def _add_segment_sentiment(
    segments_df: pd.DataFrame,
    pos_thresh: float,
    neg_thresh: float
) -> pd.DataFrame:
    if segments_df.empty:
        return segments_df

    texts = segments_df["segment_text"].fillna("").astype(str).tolist()
    compounds = roberta_compound(texts)
    segments_df = segments_df.copy()
    segments_df["compound"] = compounds
    segments_df["sentiment_label"] = [
        label_from_compound(c, pos_thresh=pos_thresh, neg_thresh=neg_thresh)
        for c in compounds
    ]

    # Respect dismissed rows
    dismissed_mask = segments_df["is_meaningful"] is False
    if "is_meaningful" in segments_df.columns:
        dismissed_mask = segments_df["is_meaningful"] == False  # noqa: E712
        segments_df.loc[dismissed_mask, "compound"] = None
        segments_df.loc[dismissed_mask, "sentiment_label"] = "No sentiment"

    return segments_df


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


def analyze_segments(
    segments_df: pd.DataFrame,
    taxonomy_mode: str,
    themes: dict,
    enriched_json_path: Optional[Path],
    semantic_cfg: dict,
    taxonomy_cfg: dict,
    sentiment_cfg: dict
) -> pd.DataFrame:
    """
    Run taxonomy + sentiment on segments and return a detailed segment-level output.
    """
    if segments_df.empty:
        return segments_df

    segments_df = _add_segment_sentiment(
        segments_df,
        pos_thresh=sentiment_cfg.get("positive_threshold", 0.05),
        neg_thresh=sentiment_cfg.get("negative_threshold", -0.05)
    )

    delimiter = taxonomy_cfg.get("multi_label", {}).get("delimiter", " | ")

    if taxonomy_mode == "semantic":
        if not enriched_json_path:
            raise ValueError("enriched_json_path is required for semantic segmentation.")

        model_name = semantic_cfg.get("model_name", "all-MiniLM-L6-v2")
        bi_threshold = float(semantic_cfg.get("bi_threshold", 0.20))
        bi_top_k = int(semantic_cfg.get("bi_top_k", 15))
        top_k = int(semantic_cfg.get("top_k", 3))
        use_cross_encoder = bool(semantic_cfg.get("use_cross_encoder", False))

        matcher = SemanticTaxonomyMatcher(
            enriched_json_path=enriched_json_path,
            model_name=model_name,
            similarity_threshold=bi_threshold if use_cross_encoder else float(semantic_cfg.get("similarity_threshold", 0.35)),
            top_k=bi_top_k if use_cross_encoder else top_k
        )

        texts = segments_df["segment_text"].fillna("").astype(str).tolist()
        cols = segments_df["TextColumn"].tolist()
        sentiments = segments_df["sentiment_label"].tolist()
        all_candidates = matcher.match_batch(texts=texts, columns=cols, sentiment_labels=sentiments)

        ce = None
        if use_cross_encoder:
            from sentence_transformers import CrossEncoder
            ce = CrossEncoder(semantic_cfg.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
            pair_template = semantic_cfg.get("pair_template", "{phrase}")

        out_rows = []
        for i, cand_list in enumerate(all_candidates):
            row = segments_df.iloc[i]
            is_meaningful = row.get("is_meaningful", None)

            if is_meaningful is False:
                out_rows.append({
                    **row.to_dict(),
                    "theme": "Brief response",
                    "subtheme": row.get("response_detail") or "No",
                    "parent_theme": "Brief response",
                    "rule_score": 0.0,
                    "expected_polarity": "",
                    "match_method": "dismissed",
                    "evidence": "",
                    "reason": "dismissed"
                })
                continue

            if not cand_list:
                out_rows.append({
                    **row.to_dict(),
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

            if use_cross_encoder and ce is not None:
                pairs = _build_pairs(row["segment_text"], cand_list, pair_template)
                scores = ce.predict(pairs)
                for c, s in zip(cand_list, scores):
                    c["ce_score"] = float(s)
                ranked = sorted(cand_list, key=lambda x: x.get("ce_score", 0.0), reverse=True)[:top_k]
                avg_score = sum(c.get("ce_score", 0.0) for c in ranked) / max(len(ranked), 1)
                match_method = "semantic_cross_encoder"
            else:
                ranked = cand_list[:top_k]
                avg_score = sum(c.get("score", 0.0) for c in ranked) / max(len(ranked), 1)
                match_method = "semantic_similarity"

            out_rows.append({
                **row.to_dict(),
                "theme": delimiter.join([c.get("theme", "") for c in ranked]),
                "subtheme": delimiter.join([c.get("subtheme", "") for c in ranked]),
                "parent_theme": delimiter.join([c.get("parent_theme", "") for c in ranked]),
                "rule_score": avg_score,
                "expected_polarity": ranked[0].get("polarity", "") if ranked else "",
                "match_method": match_method,
                "evidence": delimiter.join([c.get("matched_phrase", "") for c in ranked if c.get("matched_phrase")]),
                "reason": "matched"
            })

        return pd.DataFrame(out_rows)

    # Keyword taxonomy for segments
    matcher = TaxonomyMatcher(
        themes_json=themes,
        use_fuzzy=taxonomy_cfg.get("use_fuzzy", True),
        fuzzy_threshold=taxonomy_cfg.get("fuzzy_threshold", 0.78),
        min_accept_score=taxonomy_cfg.get("min_accept_score", 0.85),
        top_k=taxonomy_cfg.get("top_k", 3),
        column_bonus=taxonomy_cfg.get("column_bonus", 0.25),
        column_penalty=taxonomy_cfg.get("column_penalty", 0.20),
        prefer_exact=taxonomy_cfg.get("prefer_exact", True),
        phrase_first=taxonomy_cfg.get("phrase_first", True),
        allow_token_fallback=taxonomy_cfg.get("allow_token_fallback", True),
        max_token_match_length=taxonomy_cfg.get("max_token_match_length", 30),
        strict_columns=taxonomy_cfg.get("strict_columns", False),
    )

    out_rows = []
    for _, row in segments_df.iterrows():
        is_meaningful = row.get("is_meaningful", None)
        if is_meaningful is False:
            out_rows.append({
                **row.to_dict(),
                "theme": "Brief response",
                "subtheme": row.get("response_detail") or "No",
                "parent_theme": "Brief response",
                "rule_score": 0.0,
                "expected_polarity": "",
                "match_method": "dismissed",
                "evidence": "",
                "reason": "dismissed"
            })
            continue

        matches = matcher.assign_many(
            raw_text=row.get("segment_text", ""),
            text_col=row.get("TextColumn", ""),
            is_meaningful=is_meaningful,
            response_detail=row.get("response_detail", None)
        )

        if not matches:
            out_rows.append({
                **row.to_dict(),
                "theme": "",
                "subtheme": "",
                "parent_theme": "",
                "rule_score": 0.0,
                "expected_polarity": "",
                "match_method": "none",
                "evidence": "",
                "reason": "no_match"
            })
            continue

        if taxonomy_cfg.get("multi_label", {}).get("enabled", True) and len(matches) > 1:
            theme = delimiter.join([m.theme for m in matches])
            subtheme = delimiter.join([m.subtheme for m in matches])
            parent = delimiter.join([m.parent_theme for m in matches])
            rule_score = sum(m.score for m in matches) / max(len(matches), 1)
            match_method = delimiter.join([m.method for m in matches])
            evidence = delimiter.join([m.evidence for m in matches if m.evidence])
            expected_polarity = delimiter.join([m.expected_polarity for m in matches if m.expected_polarity])
        else:
            m = matches[0]
            theme = m.theme
            subtheme = m.subtheme
            parent = m.parent_theme
            rule_score = m.score
            match_method = m.method
            evidence = m.evidence
            expected_polarity = m.expected_polarity

        out_rows.append({
            **row.to_dict(),
            "theme": theme,
            "subtheme": subtheme,
            "parent_theme": parent,
            "rule_score": rule_score,
            "expected_polarity": expected_polarity,
            "match_method": match_method,
            "evidence": evidence,
            "reason": "matched"
        })

    return pd.DataFrame(out_rows)
