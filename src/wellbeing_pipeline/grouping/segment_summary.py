"""
Segment-level sentiment summary and insight labeling.
"""

from typing import Dict
import pandas as pd


def _insight_label(net_score: float, dominant_pct: float, cfg: Dict, label_base: str) -> str:
    neutral_low = float(cfg.get("net_neutral_low", -0.1))
    neutral_high = float(cfg.get("net_neutral_high", 0.1))
    net_slight = float(cfg.get("net_slight", 0.2))
    net_strong = float(cfg.get("net_strong", 0.5))
    min_majority_pct = float(cfg.get("min_majority_pct", 0.5))

    if dominant_pct < min_majority_pct:
        return f"Users are mixed/neutral on {label_base}"

    if neutral_low <= net_score <= neutral_high:
        return f"Users are mixed/neutral on {label_base}"

    if net_score > neutral_high:
        if net_score < net_slight:
            return f"Users are slightly happy with {label_base}"
        if net_score < net_strong:
            return f"Users are happy with {label_base}"
        return f"Users are very happy with {label_base}"

    # net_score < neutral_low
    if net_score > -net_slight:
        return f"Users are slightly unhappy with {label_base}"
    if net_score > -net_strong:
        return f"Users are unhappy with {label_base}"
    return f"Users are very unhappy with {label_base}"


def segment_sentiment_summary(
    segment_results: pd.DataFrame,
    cfg: Dict
) -> pd.DataFrame:
    """
    Aggregate segment-level sentiment into theme/subtheme summaries.
    """
    if segment_results.empty:
        return segment_results

    df = segment_results.copy()

    # Filter to meaningful, matched segments
    if "is_meaningful" in df.columns:
        df = df[df["is_meaningful"] != False]  # noqa: E712

    if "theme" in df.columns:
        df = df[df["theme"].fillna("").astype(str).str.strip() != ""]

    if df.empty:
        return df

    group_cols = ["TextColumn", "theme", "subtheme"]
    for c in group_cols:
        if c not in df.columns:
            df[c] = ""

    def _count_sentiments(g: pd.DataFrame) -> pd.Series:
        counts = g["sentiment_label"].value_counts(dropna=False)
        pos = int(counts.get("Positive", 0))
        neg = int(counts.get("Negative", 0))
        neu = int(counts.get("Neutral", 0))
        total = pos + neg + neu
        pos_pct = pos / total if total else 0.0
        neg_pct = neg / total if total else 0.0
        neu_pct = neu / total if total else 0.0
        net = (pos - neg) / total if total else 0.0
        dominant_pct = max(pos_pct, neg_pct) if total else 0.0

        label_base = g["subtheme"].iloc[0] or g["theme"].iloc[0]
        insight = _insight_label(net, dominant_pct, cfg, label_base)

        return pd.Series({
            "total_segments": total,
            "pos_count": pos,
            "neg_count": neg,
            "neutral_count": neu,
            "pos_pct": round(pos_pct, 3),
            "neg_pct": round(neg_pct, 3),
            "neutral_pct": round(neu_pct, 3),
            "net_score": round(net, 3),
            "dominant_pct": round(dominant_pct, 3),
            "insight": insight
        })

    summary = df.groupby(group_cols, dropna=False).apply(_count_sentiments).reset_index()
    return summary
