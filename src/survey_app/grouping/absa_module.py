"""
Aspect-Based Sentiment Analysis (ABSA) on segmented text.
"""

from typing import Dict, List, Optional
import pandas as pd
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def _normalize_label(label: str, idx: int) -> str:
    l = (label or "").lower()
    if "neg" in l:
        return "Negative"
    if "pos" in l:
        return "Positive"
    if "neu" in l:
        return "Neutral"
    # Fallback: common ordering
    if idx == 0:
        return "Negative"
    if idx == 2:
        return "Positive"
    return "Neutral"


def absa_on_segments(
    segments_df: pd.DataFrame,
    cfg: Dict
) -> pd.DataFrame:
    """
    Run ABSA for each segment using theme/subtheme as the aspect.
    Returns a DataFrame with ABSA sentiment per segment.
    """
    if segments_df.empty:
        return segments_df

    model_name = cfg.get("model_name", "yangheng/deberta-v3-base-absa-v1")
    input_mode = cfg.get("input_mode", "pair")  # "pair" or "prompt"
    prompt_template = cfg.get("prompt_template", "aspect: {aspect} text: {text}")
    batch_size = int(cfg.get("batch_size", 16))
    max_len = int(cfg.get("max_len", 128))
    use_subtheme = bool(cfg.get("use_subtheme", True))

    df = segments_df.copy()
    df["aspect"] = df["subtheme"] if use_subtheme else df["theme"]
    df["aspect"] = df["aspect"].fillna("")
    df.loc[df["aspect"].astype(str).str.strip() == "", "aspect"] = df["theme"].fillna("")

    # Filter to rows with an aspect
    df = df[df["aspect"].astype(str).str.strip() != ""].copy()
    if df.empty:
        return df

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    texts = df["segment_text"].fillna("").astype(str).tolist()
    aspects = df["aspect"].fillna("").astype(str).tolist()

    labels = []
    scores = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_aspects = aspects[i:i + batch_size]

        if input_mode == "prompt":
            prompts = [
                prompt_template.format(text=t, aspect=a)
                for t, a in zip(batch_texts, batch_aspects)
            ]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len
            )
        else:
            enc = tokenizer(
                batch_texts,
                batch_aspects,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len
            )

        with torch.no_grad():
            outputs = model(**enc)
            probs = softmax(outputs.logits, dim=1).cpu().numpy()

        for j, row_probs in enumerate(probs):
            best_idx = int(row_probs.argmax())
            score = float(row_probs[best_idx])
            id2label = getattr(model.config, "id2label", None) or {}
            raw_label = id2label.get(best_idx, "")
            label = _normalize_label(raw_label, best_idx)
            labels.append(label)
            scores.append(score)

    df["absa_label"] = labels
    df["absa_score"] = scores
    return df
