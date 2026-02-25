# src/taxonomy.py
# Previously taxonomy.py

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
import pandas as pd


@dataclass
class MatchResult:
    """Result from taxonomy matching"""
    theme: str
    subtheme: str
    parent_theme: str
    score: float
    method: str
    evidence: str
    column: str
    expected_polarity: str
    components: Dict[str, Any]


class TaxonomyMatcher:
    """
    Taxonomy matcher with smart token matching and column enforcement.
    
    Features:
    - Multi-label detection (returns top-k matches)
    - Smart token matching (only for short text ≤30 chars)
    - Fuzzy matching with configurable threshold
    - Column-aware scoring (bonus/penalty or strict enforcement)
    - Token-first for short text, phrase-first for long text
    
    v3 Schema Support:
    - Uses 'allowed_columns' for strict column enforcement
    - Rejects matches if column not in allowed_columns (when strict_columns=True)
    - Falls back to 'likely_columns' for v2 compatibility
    """
    
    def __init__(
        self,
        themes_json: dict,
        use_fuzzy: bool = True,
        fuzzy_threshold: float = 0.78,
        min_accept_score: float = 0.85,
        top_k: int = 3,
        column_bonus: float = 0.25,
        column_penalty: float = 0.20,
        prefer_exact: bool = True,
        phrase_first: bool = True,
        allow_token_fallback: bool = True,
        max_token_match_length: int = 30,
        strict_columns: bool = False,  # NEW: Strict column enforcement for v3
    ):
        self.themes = themes_json.get("themes", {})
        self.use_fuzzy = use_fuzzy
        self.fuzzy_threshold = float(fuzzy_threshold)
        self.min_accept_score = float(min_accept_score)
        self.top_k = int(top_k)
        self.column_bonus = float(column_bonus)
        self.column_penalty = float(column_penalty)
        self.prefer_exact = bool(prefer_exact)
        self.phrase_first = bool(phrase_first)
        self.allow_token_fallback = bool(allow_token_fallback)
        self.max_token_match_length = int(max_token_match_length)
        self.strict_columns = bool(strict_columns)  # NEW

        self.idx = self._build_index(self.themes)

    # -----------------------------
    # Index building
    # -----------------------------

    @staticmethod
    def _compile_terms(terms: List[str]) -> List[re.Pattern]:
        """Compile list of terms into regex patterns"""
        pats = []
        for t in terms:
            try:
                pat = re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)
                pats.append(pat)
            except Exception:
                pass
        return pats

    @staticmethod
    def _is_phrase(term: str) -> bool:
        """Return True if term should be treated as a phrase-level indicator.

        Phrases include: multi-word strings, underscore-joined phrases (from preprocessing),
        and hyphenated forms.
        """
        if term is None:
            return False
        t = str(term).strip()
        if not t:
            return False
        return (' ' in t) or ('_' in t) or ('-' in t)

    def _build_index(self, themes: dict) -> List[Dict[str, Any]]:
        index = []
        for theme, block in themes.items():
            parent = block.get("parent_theme", theme)

            for st in block.get("subthemes", []):
                keywords = st.get("keywords_phrases", []) or []
                keywords = [str(k).strip() for k in keywords if str(k).strip()]
                phrase_terms = [k for k in keywords if self._is_phrase(k)]
                token_terms  = [k for k in keywords if not self._is_phrase(k)]
                phrase_pats = self._compile_terms(phrase_terms)
                token_pats  = self._compile_terms(token_terms)

                # Support both v2 (likely_columns) and v3 (allowed_columns)
                allowed_cols = set(st.get("allowed_columns", []) or [])
                likely_cols = set(st.get("likely_columns", []) or [])
                
                # Use allowed_columns if present (v3), otherwise likely_columns (v2)
                column_set = allowed_cols if allowed_cols else likely_cols
                
                default_polarity = st.get("default_polarity", "Either")

                index.append({
                    "theme": theme,
                    "parent_theme": parent,
                    "subtheme": st.get("name", ""),
                    "phrase_patterns": phrase_pats,
                    "token_patterns": token_pats,
                    "phrase_keywords_raw": phrase_terms,
                    "token_keywords_raw": token_terms,
                    "allowed_cols": column_set,  # NEW: Unified column field
                    "likely_cols": column_set,    # Keep for backward compatibility
                    "default_polarity": default_polarity
                })
        return index

    # -----------------------------
    # Text normalization
    # -----------------------------

    @staticmethod
    def _normalise_for_match(text: str) -> str:
        """Minimal normalization for matching"""
        if not text:
            return ""
        t = text.lower().strip()
        t = re.sub(r"\s+", " ", t)
        return t

    # -----------------------------
    # Dismissive text detection
    # -----------------------------

    def _is_dismissive_text(self, text: str) -> bool:
        """Check if text is dismissive/non-informative"""
        if not text or not isinstance(text, str):
            return True
        
        text_clean = text.strip().lower()
        
        # Very short
        if len(text_clean) <= 2:
            return True
        
        # Common dismissive patterns
        dismissive = [
            "no", "n/a", "na", "none", "nothing", "nil", "nope",
            "yes", "yep", "yeah", "ok", "fine", "good",
            "as above", "see above", "same as above", "ditto",
            "no entry", "no comment", "no response"
        ]
        
        if text_clean in dismissive:
            return True
        
        return False

    # -----------------------------
    # Fuzzy matching
    # -----------------------------

    def _fuzzy_match(self, text: str, keyword: str) -> Tuple[bool, float, str]:
        """
        Fuzzy match keyword against text.
        Returns (matched, confidence, evidence).
        """
        norm_text = self._normalise_for_match(text)
        norm_kw = self._normalise_for_match(keyword)

        if not norm_text or not norm_kw:
            return False, 0.0, ""

        text_words = norm_text.split()
        kw_words = norm_kw.split()

        # Strategy 1: exact substring
        if norm_kw in norm_text:
            return True, 1.0, keyword

        # Strategy 2: sliding window for multi-word keywords
        if len(kw_words) > 1:
            window = len(kw_words)
            max_sim = 0.0
            best_window = ""

            for i in range(len(text_words) - window + 1):
                seg = " ".join(text_words[i:i + window])
                sim = SequenceMatcher(None, norm_kw, seg).ratio()
                if sim > max_sim:
                    max_sim = sim
                    best_window = seg
                if sim >= self.fuzzy_threshold:
                    return True, sim, seg

            # Strategy 3: scattered words (non-consecutive)
            if all(any(kw in tw or tw in kw for tw in text_words) for kw in kw_words):
                return True, 0.80, "scattered:" + " ".join(kw_words)

            # if close but not above threshold, return best candidate
            if max_sim >= (self.fuzzy_threshold - 0.05):
                return True, max_sim, best_window

        return False, 0.0, ""

    # -----------------------------
    # Scoring logic
    # -----------------------------

    def _column_adjustment(self, text_col: str, allowed_cols: set, strict: bool = None) -> Tuple[float, float]:
        """
        Returns (bonus, penalty) depending on column relevance.
        
        Args:
            text_col: Current text column
            allowed_cols: Set of allowed columns for this subtheme
            strict: If True, REJECT match if column not in allowed_cols (returns -999 penalty)
                   If False, apply bonus/penalty scoring (default behavior)
                   If None, uses self.strict_columns
        
        Returns:
            (bonus, penalty) tuple
            
        v3 Strict Mode:
            - If column in allowed_cols: bonus, no penalty
            - If column NOT in allowed_cols: -999 penalty (effectively rejects match)
        
        v2 Soft Mode (default):
            - If allowed_cols empty: no bonus/penalty
            - If column in allowed_cols: bonus, no penalty
            - If column NOT in allowed_cols: no bonus, penalty
        """
        if strict is None:
            strict = self.strict_columns
        
        if not allowed_cols:
            return 0.0, 0.0
        
        if text_col in allowed_cols:
            return self.column_bonus, 0.0
        
        # Column mismatch
        if strict:
            # v3 strict mode: REJECT match (huge penalty)
            return 0.0, 999.0
        else:
            # v2 soft mode: Apply penalty but don't reject
            return 0.0, self.column_penalty

    # -----------------------------
    # Public methods
    # -----------------------------

    def assign_many(
        self, 
        raw_text: str, 
        text_col: str,
        is_meaningful: Optional[bool] = None,
        response_detail: Optional[str] = None
    ) -> List[MatchResult]:
        """
        Assign multiple themes to text (multi-label).
        Returns top-k matches sorted by score.
        
        Args:
            raw_text: Text to classify
            text_col: Column name (for column-aware scoring)
            is_meaningful: Optional flag from null detector
            response_detail: Optional detail from null detector ("Yes", "No", "No sentiment")
        
        Returns:
            List of MatchResult objects (up to top_k matches)
        """
        text = raw_text or ""
        col = text_col or ""

        # Check meaningful flag first (if provided)
        if is_meaningful is not None and not is_meaningful:
            # Text flagged as dismissive - theme="Brief response", subtheme=reason
            if response_detail:
                return [MatchResult(
                    theme="Brief response",
                    subtheme=response_detail,  # Reason: "Yes", "No", "No sentiment"
                    parent_theme="Brief response",
                    score=0.0,
                    method="dismissed",
                    evidence="",
                    column=col,
                    expected_polarity="",
                    components={}
                )]
            # Fallback if no response_detail provided
            return []
        
        # Fallback: check if text is dismissive (if flag not provided)
        if is_meaningful is None and self._is_dismissive_text(text):
            return []

        # Check text length for smart token matching
        text_length = len(text.strip())
        allow_tokens_for_this_text = self.allow_token_fallback and (text_length <= self.max_token_match_length)

        candidates: List[MatchResult] = []

        for row in self.idx:
            base_score = 0.0
            method = "none"
            evidence = ""

            # STRATEGY: For short text, try tokens FIRST (before phrases)
            # This prevents weak fuzzy phrase matches from blocking strong token matches
            
            if allow_tokens_for_this_text and row["token_keywords_raw"]:
                # SHORT TEXT: Try exact token match first
                exact_hit = None
                for pat, kw in zip(row["token_patterns"], row["token_keywords_raw"]):
                    if pat.search(text):
                        exact_hit = kw
                        break
                if exact_hit:
                    base_score = 0.95
                    method = "token_short_text"
                    evidence = exact_hit
                # If no exact token, try fuzzy token
                elif self.use_fuzzy:
                    best_fuzzy = (0.0, "")
                    for kw in row["token_keywords_raw"]:
                        ok, conf, ev = self._fuzzy_match(text, kw)
                        if ok and conf > best_fuzzy[0]:
                            best_fuzzy = (conf, ev or kw)
                    if best_fuzzy[0] > 0.0:
                        base_score = best_fuzzy[0]
                        method = "fuzzy_token"
                        evidence = best_fuzzy[1]
            
            # If no token match found (or text is long), try phrase matching
            if base_score <= 0.0 and self.phrase_first:
                # 1) Exact phrase matching
                exact_hit = None
                for pat, kw in zip(row["phrase_patterns"], row["phrase_keywords_raw"]):
                    if pat.search(text):
                        exact_hit = kw
                        break
                if exact_hit:
                    base_score = 1.0
                    method = "exact"
                    evidence = exact_hit
                # 2) Fuzzy phrase matching
                elif self.use_fuzzy and row["phrase_keywords_raw"]:
                    best_fuzzy = (0.0, "")
                    for kw in row["phrase_keywords_raw"]:
                        ok, conf, ev = self._fuzzy_match(text, kw)
                        if ok and conf > best_fuzzy[0]:
                            best_fuzzy = (conf, ev or kw)
                    if best_fuzzy[0] > 0.0:
                        base_score = best_fuzzy[0]
                        method = "fuzzy"
                        evidence = best_fuzzy[1]
            
            # Legacy behaviour (if phrase_first is False)
            elif not self.phrase_first:
                exact_hit = None
                for pat, kw in zip(row["phrase_patterns"] + row["token_patterns"],
                                   row["phrase_keywords_raw"] + row["token_keywords_raw"]):
                    if pat.search(text):
                        exact_hit = kw
                        break
                if exact_hit:
                    base_score = 1.0
                    method = "exact"
                    evidence = exact_hit
                elif self.use_fuzzy and (row["phrase_keywords_raw"] or row["token_keywords_raw"]):
                    best_fuzzy = (0.0, "")
                    for kw in (row["phrase_keywords_raw"] + row["token_keywords_raw"]):
                        ok, conf, ev = self._fuzzy_match(text, kw)
                        if ok and conf > best_fuzzy[0]:
                            best_fuzzy = (conf, ev or kw)
                    if best_fuzzy[0] > 0.0:
                        base_score = best_fuzzy[0]
                        method = "fuzzy"
                        evidence = best_fuzzy[1]

            if base_score <= 0.0:
                continue

            # Column adjustment (with strict enforcement for v3)
            bonus, penalty = self._column_adjustment(col, row["allowed_cols"], strict=self.strict_columns)

            final_score = base_score + bonus - penalty

            # Prefer exact: small bump to break ties
            if self.prefer_exact and method == "exact":
                final_score += 0.02

            components = {
                "base_score": round(base_score, 3),
                "column_bonus": round(bonus, 3),
                "column_penalty": round(penalty, 3),
                "exact_preference_bump": 0.02 if (self.prefer_exact and method == "exact") else 0.0,
                "text_length": text_length,
                "token_matching_allowed": allow_tokens_for_this_text
            }

            # Accept if above threshold
            if final_score >= self.min_accept_score:
                candidates.append(MatchResult(
                    theme=row["theme"],
                    subtheme=row["subtheme"],
                    parent_theme=row["parent_theme"],
                    score=final_score,
                    method=method,
                    evidence=evidence,
                    column=col,
                    expected_polarity=row["default_polarity"],
                    components=components
                ))

        # Sort by score descending
        candidates.sort(key=lambda x: x.score, reverse=True)

        # Return top-k
        return candidates[:self.top_k]

    def assign_best(
        self, 
        raw_text: str, 
        text_col: str,
        is_meaningful: Optional[bool] = None,
        response_detail: Optional[str] = None
    ) -> Optional[MatchResult]:
        """
        Assign single best theme to text.
        Returns top match or None.
        """
        matches = self.assign_many(raw_text, text_col, is_meaningful, response_detail)
        return matches[0] if matches else None


def assign_taxonomy_keyword(
    df: pd.DataFrame,
    text_columns: List[str],
    themes: dict,
    taxonomy_cfg: dict = None,
    output_cfg: dict = None
) -> pd.DataFrame:
    """
    Assign taxonomy using keyword/fuzzy matching (TaxonomyMatcher).
    Returns a pandas DataFrame of assignments.
    """
    import numpy as np

    taxonomy_cfg = taxonomy_cfg or {}
    output_cfg = output_cfg or {}
    multi_cfg = taxonomy_cfg.get("multi_label", {})

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

    delimiter = multi_cfg.get("delimiter", " | ")
    multi_enabled = multi_cfg.get("enabled", True)
    include_dismissed = output_cfg.get("include_dismissed_in_assignments", True)

    assignments = []

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

            matches = matcher.assign_many(
                raw_text=text,
                text_col=col,
                is_meaningful=is_meaningful,
                response_detail=response_detail
            )

            if not matches:
                if is_meaningful is False and not include_dismissed:
                    continue
                assignments.append({
                    "ID": row_id,
                    "TextColumn": col,
                    "is_meaningful": bool(is_meaningful) if is_meaningful is not None else None,
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

            if matches[0].method == "dismissed" and not include_dismissed:
                continue

            if multi_enabled and len(matches) > 1:
                theme = delimiter.join([m.theme for m in matches])
                subtheme = delimiter.join([m.subtheme for m in matches])
                parent = delimiter.join([m.parent_theme for m in matches])
                rule_score = float(np.mean([m.score for m in matches]))
                match_method = delimiter.join([m.method for m in matches])
                evidence = delimiter.join([m.evidence for m in matches if m.evidence])
                expected_polarity = delimiter.join([m.expected_polarity for m in matches if m.expected_polarity])
                reason = "matched"
            else:
                m = matches[0]
                theme = m.theme
                subtheme = m.subtheme
                parent = m.parent_theme
                rule_score = float(m.score)
                match_method = m.method
                evidence = m.evidence
                expected_polarity = m.expected_polarity
                if m.method == "dismissed":
                    reason = f"dismissed:{m.subtheme}" if m.subtheme else "dismissed"
                else:
                    reason = "matched"

            assignments.append({
                "ID": row_id,
                "TextColumn": col,
                "is_meaningful": bool(is_meaningful) if is_meaningful is not None else None,
                "theme": theme,
                "subtheme": subtheme,
                "parent_theme": parent,
                "rule_score": rule_score,
                "expected_polarity": expected_polarity,
                "match_method": match_method,
                "evidence": evidence,
                "reason": reason
            })

    return pd.DataFrame(assignments)
