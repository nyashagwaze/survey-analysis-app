# src/clean_normalise.py
import re
from typing import Iterable
import pandas as pd

WS = re.compile(r"\s+")
PUNCT = re.compile(r"[^\w\s'-]+")

def basic_normalise(text: str) -> str:
    """
    Basic text normalization with null handling.
    Converts null/empty/non-informative text to 'no entry' for downstream processing.
    """
    # Handle None, NaN, empty strings
    if text is None or pd.isna(text):
        return "no entry"
    
    if not isinstance(text, str):
        text = str(text)
    
    # Strip and check if empty
    text = text.strip()
    if not text or text.isspace():
        return "no entry"
    
    # Check for string "nan" (pandas converts NaN to "nan" string)
    if text.lower() == "nan":
        return "no entry"
    
    # Check for common null patterns
    text_lower = text.lower()
    null_patterns = ["n/a", "na", "n.a", "n.a.", "null", "none", "nil"]
    if text_lower in null_patterns:
        return "no entry"
    
    # Normal text processing
    t = text.lower()
    # Normalize common curly quotes to ASCII equivalents.
    t = t.translate(str.maketrans({
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": "\"",
        "\u201d": "\"",
    }))
    t = PUNCT.sub(" ", t)
    t = WS.sub(" ", t).strip()
    
    # Final check - if normalization resulted in empty string
    if not t:
        return "no entry"
    
    return t

def apply_business_map(text: str, business_map: dict) -> str:
    """
    Apply business-specific term replacements.
    Preserves 'no entry' marker.
    """
    # Don't process 'no entry' marker
    if text == "no entry":
        return text
    
    # simple token replacement; do multiword first
    out = text
    # multiword keys first
    for k, v in sorted(business_map.items(), key=lambda kv: (-len(kv[0]), kv[0])): 
        out = re.sub(rf"\b{re.escape(k)}\b", v, out, flags=re.IGNORECASE)
    return out

def merge_to_canonical(text: str, merge_map: dict) -> str:
    """
    Merge synonyms to canonical forms.
    Preserves 'no entry' marker.
    """
    # Don't process 'no entry' marker
    if text == "no entry":
        return text
    
    toks = text.split()
    new = [merge_map.get(tok, tok) for tok in toks]
    return " ".join(new)

def force_phrases(text: str, forced: Iterable[str]) -> str:
    """
    Force multi-word phrases to be preserved with underscores.
    Preserves 'no entry' marker.
    """
    # Don't process 'no entry' marker
    if text == "no entry":
        return text
    
    out = text
    for ph in forced:
        if "_" in ph:
            plain = ph.replace("_", " ")
            out = out.replace(plain, ph)
    return out

def keep_unigrams(text: str, unigram_whitelist: set, always_drop: set, must_be_phrase: set) -> str:
    """
    Filter tokens based on whitelists and rules.
    Preserves 'no entry' marker.
    """
    # Don't process 'no entry' marker
    if text == "no entry":
        return text
    
    out = []
    for tok in text.split():
        if "_" in tok: 
            out.append(tok); continue
        if tok in always_drop: 
            continue
        if tok in unigram_whitelist:
            out.append(tok)
        elif tok in must_be_phrase:
            # keep only if used as part of a phrase (handled elsewhere)
            continue
        else:
            # heuristic: drop tiny tokens
            if tok.isalpha() and len(tok) >= 3:
                out.append(tok)
    return " ".join(out)
