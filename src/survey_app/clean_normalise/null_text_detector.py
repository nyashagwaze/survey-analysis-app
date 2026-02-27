"""
Response Quality & Detail Classification Module
 
Purpose:
- Detect non-meaningful (null / dismissive) survey text
- Classify response detail type for reporting
- Provide authoritative flags to gate taxonomy & sentiment
 
This module MUST run before dictionary / taxonomy logic.
"""
 
import re
from typing import Union
import pandas as pd
import numpy as np
 
 
# ============================================================
# NULL / NON-INFORMATIVE TEXT DETECTION
# ============================================================
 
NULL_EXACT_MATCHES = {
    # Negative/null responses
    "no", "nope", "none", "null", "nil", "nothing", "nah",
    "no thanks", "no comment", "not really", "nothing extra",
    "nothing to add", "nothing else", "nothing more",
    
    # N/A responses
    "n/a", "na", "n.a", "n.a.", "nan",
    
    # Affirmative short responses
    "yes", "yep", "yeah", "yup", "sure", "yes thanks",
    
    # Irrelevant/reference responses
    "ok", "okay", "fine", "good", "all good", "all is well",
    "everything is fine", "everythings fine",
    "as above", "see above", "same as above", "ditto",
    "as before", "see previous", "same",
    
    # Uncertainty/don't know
    "i dont know", "i don't know", "dont know", "don't know",
    "unsure", "not sure", "no idea",
    
    # Privacy/declined
    "prefer not to say", "rather not say",
    "too private", "private", "personal",
    "dont want to share", "do not want to share"
}
 
NULL_PATTERNS = [
    r"^no+$",
    r"^yes+$",
    r"^n/?a+$",
    r"^none+$",
    r"^nothing\s*(else|more|extra|to\s+add)?$",
    r"^not?\s+really$",
    r"^(as|see)\s+(above|before|previous)$",
    r"^all\s+(good|fine|ok|is\s+well)$",
    r"^everything.*\s+(fine|good|ok)$",
    r"^i.*m\s+(fine|good|ok)$",
    r"^(i\s*)?(dont|don't|do\s+not)\s+know$",
    r"^unsure$",
    r"^not\s+sure$",
    r"^no\s+idea$",
    r"^(too\s+)?private$",
]
 
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in NULL_PATTERNS]
 
 
def is_null_text(
    text: Union[str, None],
    min_meaningful_length: int = 3,
    max_dismissive_length: int = 50
) -> bool:
    """
    Returns True if text is non-informative / null.
    
    Args:
        text: Text to check
        min_meaningful_length: Minimum length for meaningful text (default: 3)
        max_dismissive_length: Maximum character length for dismissive responses (default: 50)
                               Prevents long sentences from being dismissed even if they contain
                               dismissive words like "no" or "yes"
    
    Returns:
        True if text should be dismissed as non-meaningful
    """
    if text is None:
        return True
 
    try:
        if pd.isna(text):
            return True
    except (TypeError, ValueError):
        pass
 
    try:
        if isinstance(text, float) and np.isnan(text):
            return True
    except (TypeError, ValueError):
        pass
 
    text_str = str(text).strip()
    if not text_str or text_str.isspace():
        return True
 
    # Too short to be meaningful
    if len(text_str) < min_meaningful_length:
        return True
    
    # TOO LONG to be a brief dismissive response
    # If text is longer than max_dismissive_length, treat as meaningful
    # This prevents sentences like "I have no issues" from being dismissed
    if len(text_str) > max_dismissive_length:
        return False
 
    text_lower = text_str.lower()
    if text_lower in ("nan", "none", "<na>", "nat"):
        return True
 
    text_normalized = re.sub(r"[^a-z0-9\s/]", "", text_lower).strip()
 
    # Check exact matches (only if text is short enough)
    if text_normalized in NULL_EXACT_MATCHES:
        return True
 
    # Check regex patterns (only if text is short enough)
    for pattern in COMPILED_PATTERNS:
        if pattern.match(text_normalized):
            return True
 
    return False
 
 
# ============================================================
# RESPONSE DETAIL CLASSIFICATION - 3 CATEGORIES
# ============================================================
 
# Category 1: YES - Affirmative responses
YES_RESPONSES = {
    "yes", "yep", "yeah", "yup", "sure", 
    "yes thanks", "yes please", "yea"
}

# Category 2: NO - Negative/null responses  
NO_RESPONSES = {
    "no", "nope", "nah",
    "n/a", "na", "n.a", "n.a.", "nan", 
    "none", "null", "nil", "nothing",
    "nothing to add", "nothing else", "nothing more", "nothing extra",
    "no thanks", "no comment", "not really"
}

# Category 3: NO SENTIMENT - Irrelevant/reference/privacy/uncertainty responses
NO_SENTIMENT_RESPONSES = {
    # Irrelevant acknowledgements
    "ok", "okay", "fine", "good", "all good", "all is well",
    "everything is fine", "everythings fine",
    
    # Reference to other responses
    "as above", "see above", "same as above", "ditto",
    "as before", "see previous", "same",
    
    # Uncertainty/don't know
    "i dont know", "i don't know", "dont know", "don't know",
    "unsure", "not sure", "no idea",
    
    # Privacy/declined
    "prefer not to say", "rather not say",
    "too private", "private", "personal",
    "dont want to share", "do not want to share"
}
 
 
def classify_response_detail(
    text: Union[str, None],
    min_meaningful_length: int = 3,
    max_dismissive_length: int = 50
) -> str:
    """
    Classify response detail type for reporting.
    Returns one of 3 dismissive categories or "Detailed response".
 
    Returns:
    - "Detailed response" - meaningful text (will be processed)
    - "Yes" - affirmative short response
    - "No" - negative/null/n/a response
    - "No sentiment" - irrelevant/reference/privacy/uncertainty response
    """
    # Check if text is meaningful first
    if not is_null_text(
        text,
        min_meaningful_length=min_meaningful_length,
        max_dismissive_length=max_dismissive_length
    ):
        return "Detailed response"
 
    # Handle null/empty
    if text is None:
        return "No"
 
    # Normalize text for classification
    text_norm = re.sub(r"[^a-z0-9\s]", "", str(text).strip().lower())
    
    # Classify into 3 categories
    if text_norm in YES_RESPONSES:
        return "Yes"
    
    if text_norm in NO_RESPONSES:
        return "No"
    
    if text_norm in NO_SENTIMENT_RESPONSES:
        return "No sentiment"
 
    # Default for other dismissed text (empty, very short, etc.)
    return "No"
 
 
# ============================================================
# DATAFRAME HELPERS (PIPELINE INTEGRATION)
# ============================================================
 
def add_response_quality_flags(
    df: pd.DataFrame,
    text_columns: list,
    meaningful_suffix: str = "_is_meaningful",
    detail_suffix: str = "_response_detail",
    min_meaningful_length: int = 3,
    max_dismissive_length: int = 50
) -> pd.DataFrame:
    """
    Add response quality flags to a DataFrame.
 
    Creates:
    - <column>_is_meaningful      (authoritative taxonomy/sentiment gate)
    - <column>_response_detail   (BI / reporting label - used as subtheme for dismissed text)
    """
    df_out = df.copy()
 
    for col in text_columns:
        if col not in df_out.columns:
            continue
 
        df_out[f"{col}{meaningful_suffix}"] = ~df_out[col].apply(
            lambda x: is_null_text(
                x,
                min_meaningful_length=min_meaningful_length,
                max_dismissive_length=max_dismissive_length
            )
        )
        df_out[f"{col}{detail_suffix}"] = df_out[col].apply(
            lambda x: classify_response_detail(
                x,
                min_meaningful_length=min_meaningful_length,
                max_dismissive_length=max_dismissive_length
            )
        )
 
    return df_out
 
 
def get_response_quality_report(
    df: pd.DataFrame,
    text_columns: list,
    min_meaningful_length: int = 3,
    max_dismissive_length: int = 50
) -> pd.DataFrame:
    """
    Summary report of response detail types per column.
    """
    rows = []
 
    for col in text_columns:
        if col not in df.columns:
            continue
 
        counts = df[col].apply(
            lambda x: classify_response_detail(
                x,
                min_meaningful_length=min_meaningful_length,
                max_dismissive_length=max_dismissive_length
            )
        ).value_counts()
 
        for label, count in counts.items():
            rows.append({
                "column": col,
                "response_detail": label,
                "count": count
            })
 
    return pd.DataFrame(rows)
 
 
# ============================================================
# USAGE NOTE (IMPORTANT)
# ============================================================
"""
PIPELINE RULE (NON-NEGOTIABLE):
 
If <column>_is_meaningful == False:
    - Do NOT run dictionary/taxonomy
    - Do NOT assign sentiment
    - USE response_detail as subtheme
    - Theme will be: "Brief response"
    - Subtheme will be: "Yes", "No", or "No sentiment"
 
Only "Detailed response" rows reach taxonomy & sentiment analysis.
"""
