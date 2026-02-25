# src/sentiment_module.py
"""
Hybrid Sentiment Module - Best of Both Worlds:
- RoBERTa transformer (better nuance than VADER)
- Clause-aware sentiment (handles "but" contrasts)
- Column-aware weighting (50%/35%/15%)
- Improved coping detection (30+ patterns with negation)
- Null text filtering integration (respects meaningful flags)
- Lazy model loading (loads on first use, respects cache env vars)
"""

import re
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set HuggingFace cache (if not already set)
if 'TRANSFORMERS_CACHE' not in os.environ:
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'
if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = '/tmp/huggingface_cache'

# Transformer model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

# Global variables for lazy loading
_tokenizer = None
_model = None


def _get_model():
    """Lazy load the RoBERTa model and tokenizer on first use"""
    global _tokenizer, _model
    
    if _tokenizer is None or _model is None:
        # Set cache directories before loading
        os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'
        os.environ['HF_HOME'] = '/tmp/huggingface_cache'
        
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    
    return _tokenizer, _model


# ---------------------------------------------------
# 1) ROBERTA SENTIMENT SCORING
# ---------------------------------------------------

def roberta_probs(texts, batch_size=64, max_len=128):
    """
    Returns Nx3 probabilities for [neg, neutral, pos]
    """
    tokenizer, model = _get_model()  # Lazy load
    
    clean_texts = [(t if isinstance(t, str) else "") for t in texts]
    all_probs = []

    for i in range(0, len(clean_texts), batch_size):
        batch = clean_texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len
        )

        with torch.no_grad():
            outputs = model(**enc)
            probs = softmax(outputs.logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    return np.vstack(all_probs)


def roberta_compound(texts):
    """
    Compute compound sentiment score from RoBERTa probabilities.
    Returns values in [-1, 1] range similar to VADER.
    """
    probs = roberta_probs(texts)
    # probs shape: (N, 3) where columns are [neg, neutral, pos]
    # Compound = pos - neg
    compound = probs[:, 2] - probs[:, 0]
    return compound


# ---------------------------------------------------
# 2) CONTRAST MARKERS & CLAUSE SPLITTING
# ---------------------------------------------------

CONTRAST_MARKERS = [
    r"\bbut\b", r"\bhowever\b", r"\balthough\b", r"\bthough\b",
    r"\byet\b", r"\bstill\b", r"\bnevertheless\b", r"\bnonetheless\b",
    r"\bwhile\b", r"\bwhereas\b", r"\bdespite\b", r"\bin spite of\b",
    r"\bon the other hand\b", r"\bthat said\b", r"\beven so\b"
]

CONTRAST_PATTERN = re.compile("|".join(CONTRAST_MARKERS), re.IGNORECASE)


def split_on_contrast(text: str):
    """
    Split text on contrast markers.
    Returns list of clauses.
    """
    if not text or not isinstance(text, str):
        return [text]
    
    # Split on contrast markers
    parts = CONTRAST_PATTERN.split(text)
    
    # Clean up parts
    clauses = [p.strip() for p in parts if p and p.strip()]
    
    return clauses if clauses else [text]


def clause_aware_compound(text: str) -> float:
    """
    Compute sentiment with clause-aware contrast handling.
    
    If text contains contrast markers (but, however, etc.):
    - Split into clauses
    - Weight later clauses more heavily (they're usually the main point)
    - Return weighted average
    
    Otherwise:
    - Return single sentiment score
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    # Check for contrast markers
    if CONTRAST_PATTERN.search(text):
        clauses = split_on_contrast(text)
        
        if len(clauses) > 1:
            # Calculate sentiment for each clause
            scores = roberta_compound(clauses)
            
            # Weight later clauses more (they're usually the main point)
            # Weights: [0.3, 0.7] for 2 clauses, [0.2, 0.3, 0.5] for 3 clauses, etc.
            weights = np.linspace(0.2, 0.8, len(scores))
            weights = weights / weights.sum()  # Normalize to sum to 1
            
            # Weighted average
            compound = np.average(scores, weights=weights)
            return float(compound)
    
    # No contrast markers - single sentiment
    scores = roberta_compound([text])
    return float(scores[0])


def label_from_compound(compound: float, pos_thresh: float = 0.05, neg_thresh: float = -0.05) -> str:
    """
    Convert compound score to sentiment label.
    
    Args:
        compound: Sentiment score in [-1, 1] range
        pos_thresh: Threshold for positive sentiment (default: 0.05)
        neg_thresh: Threshold for negative sentiment (default: -0.05)
    
    Returns:
        "Positive", "Negative", or "Neutral"
    """
    if compound is None:
        return "Neutral"
    
    if compound >= pos_thresh:
        return "Positive"
    elif compound <= neg_thresh:
        return "Negative"
    else:
        return "Neutral"


# ---------------------------------------------------
# 3) COPING MECHANISM DETECTION
# ---------------------------------------------------

COPING_PATTERNS = [
    # Exercise & Physical Activity
    r"\bexercise\b", r"\bexercising\b", r"\bgym\b", r"\bfitness\b",
    r"\brunning\b", r"\bwalking\b", r"\bswimming\b", r"\byoga\b",
    r"\bworking out\b", r"\bphysical activity\b",
    
    # Social Support
    r"\btalking to\b", r"\bspeak to\b", r"\bspoke to\b", r"\bsharing with\b",
    r"\bfamily support\b", r"\bfriends\b", r"\bcolleagues\b", r"\bteam\b",
    r"\bmy manager\b", r"\bline manager\b", r"\bsupport from\b",
    
    # Relaxation & Mindfulness
    r"\bmeditat\w*\b", r"\bmindful\w*\b", r"\brelax\w*\b", r"\bbreath\w*\b",
    r"\bcalm\w*\b", r"\bdeep breath\b",
    
    # Time Off & Breaks
    r"\btime off\b", r"\bholiday\b", r"\bbreak\b", r"\bleave\b",
    r"\bvacation\b", r"\btime away\b",
    
    # Hobbies & Interests
    r"\bhobb\w*\b", r"\binterests\b", r"\bpursuing\b", r"\benjoying\b",
    
    # Professional Help
    r"\bcounseling\b", r"\bcounselling\b", r"\btherapy\b", r"\btherapist\b",
    r"\bpsychologist\b", r"\bmental health support\b",
    
    # Positive Reframing
    r"\bfocusing on\b", r"\blooking forward\b", r"\bstaying positive\b",
    r"\bkeeping perspective\b", r"\breminding myself\b"
]

COPING_REGEX = re.compile("|".join(COPING_PATTERNS), re.IGNORECASE)

# Negation patterns (to avoid false positives)
NEGATION_PATTERNS = [
    r"\bno\b", r"\bnot\b", r"\bnever\b", r"\bcan't\b", r"\bcannot\b",
    r"\bwon't\b", r"\bwouldn't\b", r"\bdon't\b", r"\bdoesn't\b",
    r"\bhadn't\b", r"\bhasn't\b", r"\bhaven't\b", r"\black of\b",
    r"\bwithout\b", r"\bunable to\b", r"\bfailed to\b"
]

NEGATION_REGEX = re.compile("|".join(NEGATION_PATTERNS), re.IGNORECASE)


def detect_coping(text: str) -> bool:
    """
    Detect if text mentions coping mechanisms.
    Returns True if coping patterns found AND not negated.
    """
    if not text or not isinstance(text, str):
        return False
    
    # Check for coping patterns
    if not COPING_REGEX.search(text):
        return False
    
    # Check for negation near coping words
    # Simple heuristic: if negation appears, likely not coping
    if NEGATION_REGEX.search(text):
        # More sophisticated: check if negation is near coping word
        # For now, simple approach: if negation exists, be cautious
        # Could improve with dependency parsing
        return False
    
    return True


# Default weights aligned with pipeline_settings.yaml
DEFAULT_COLUMN_WEIGHTS = {
    "Wellbeing_Details": 0.50,
    "Areas_Improve": 0.30,
    "Support_Provided": 0.20,
}

def weighted_sentiment_for_row(
    row: dict,
    text_columns: list,
    weights: dict = None,
    pos: float = 0.05,
    neg: float = -0.05,
    skip_dismissed: bool = True,
    dismissed_sentiment_value = None
) -> dict:
    """
    Compute weighted sentiment for a single row.
    Returns per-column compounds, overall compound, label, and coping flag.
    """
    if weights is None:
        weights = DEFAULT_COLUMN_WEIGHTS.copy()

    compound_by_column = {}
    active_weights = {}

    for col in text_columns:
        text = row.get(col, "")
        is_meaningful = row.get(f"{col}_is_meaningful", True)

        if skip_dismissed and not is_meaningful:
            compound_by_column[col] = None
            continue

        if not isinstance(text, str) or not text.strip():
            compound_by_column[col] = None
            continue

        compound_by_column[col] = float(roberta_compound([text])[0])
        active_weights[col] = weights.get(col, 0.0)

    # Normalize weights only across non-null columns
    total_weight = sum(active_weights.values())
    if total_weight <= 0:
        compound_weighted = dismissed_sentiment_value
        sentiment_label = "No sentiment" if skip_dismissed else "Neutral"
    else:
        compound_weighted = sum(
            compound_by_column[col] * w for col, w in active_weights.items()
            if compound_by_column.get(col) is not None
        ) / total_weight
        sentiment_label = label_from_compound(compound_weighted, pos_thresh=pos, neg_thresh=neg)

    coping_flag = None
    for col in text_columns:
        text = row.get(col, "")
        if detect_coping(text):
            coping_flag = "Coping"
            break

    return {
        "compound_weighted": compound_weighted,
        "sentiment_label": sentiment_label,
        "coping_flag": coping_flag,
        "compound_by_column": compound_by_column,
    }


# ---------------------------------------------------
# 4) COLUMN-AWARE SENTIMENT AGGREGATION
# ---------------------------------------------------

def aggregate_sentiment_by_id(
    df: pd.DataFrame,
    text_columns: list,
    weights: dict = None
) -> pd.DataFrame:
    """
    Aggregate sentiment across multiple text columns with weighting.
    
    Args:
        df: DataFrame with ID and sentiment columns
        text_columns: List of text column names
        weights: Dict mapping column names to weights (default: equal weighting)
    
    Returns:
        DataFrame with ID and aggregated sentiment
    """
    if weights is None:
        weights = {col: 1.0 / len(text_columns) for col in text_columns}
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # Calculate weighted compound
    df['compound'] = 0.0
    
    for col in text_columns:
        compound_col = f"compound_{col}"
        if compound_col in df.columns:
            weight = weights.get(col, 0.0)
            df['compound'] += df[compound_col].fillna(0.0) * weight
    
    # Add sentiment label
    df['sentiment_label'] = df['compound'].apply(label_from_compound)
    
    # Add coping flag (check all text columns)
    df['coping_flag'] = None
    for col in text_columns:
        if col in df.columns:
            df.loc[df[col].apply(detect_coping), 'coping_flag'] = 'Coping'
    
    return df


def add_sentiment_columns_pandas(
    df: pd.DataFrame,
    text_columns: list,
    weights: dict = None,
    pos: float = 0.05,
    neg: float = -0.05,
    skip_dismissed: bool = True,
    dismissed_sentiment_value = None
) -> pd.DataFrame:
    """
    Add sentiment columns to a pandas DataFrame using weighted_sentiment_for_row.
    """
    df_out = df.copy()

    compound_weighted = []
    sentiment_labels = []
    coping_flags = []
    compound_by_col = {col: [] for col in text_columns}

    for _, row in df_out.iterrows():
        result = weighted_sentiment_for_row(
            row=row.to_dict(),
            text_columns=text_columns,
            weights=weights,
            pos=pos,
            neg=neg,
            skip_dismissed=skip_dismissed,
            dismissed_sentiment_value=dismissed_sentiment_value
        )

        compound_weighted.append(result["compound_weighted"])
        sentiment_labels.append(result["sentiment_label"])
        coping_flags.append(result["coping_flag"])

        for col in text_columns:
            compound_by_col[col].append(result["compound_by_column"].get(col))

    df_out["compound"] = compound_weighted
    df_out["sentiment_label"] = sentiment_labels
    df_out["coping_flag"] = coping_flags

    for col in text_columns:
        df_out[f"compound_{col}"] = compound_by_col[col]

    return df_out
