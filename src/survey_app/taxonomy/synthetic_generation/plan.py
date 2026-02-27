# build_phrase_library.py
# Pipeline to generate comprehensive, realistic phrase library for v3 theme dictionary
# NO REAL DATA REQUIRED - generates synthetic but realistic phrases for testing
# USES theme_subtheme_dictionary_v3.json for context-aware generation
# THEN adds Enhanced_Dictionary_v3.json for business-specific terms (SAP, projects, etc.)

import os
import json
import time
import csv
import re
import unicodedata
from copy import deepcopy
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from ...config_runtime import _get_project_root

PROMPT_VERSION = "v1.3-enhanced-dict"
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.6
MAX_PHRASES = 50   # hard cap per (column × lens)
 
# ============================================================
# SETUP
# ============================================================
 
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROJECT_ROOT = _get_project_root(start=Path(__file__).resolve())
CONFIG_DIR = PROJECT_ROOT / "config"
PROFILE = os.getenv("PROFILE", "general")
ASSETS_DIR = PROJECT_ROOT / "assets" / "taxonomy" / PROFILE
OUTPUT_DIR = ASSETS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "generated_phrases.json"
TARGET_PHRASES = 30

# =============================
# CONFIG
# =============================
TARGET_PHRASES_PER_PASS = int(os.getenv("TARGET_PHRASES_PER_PASS", "20"))
NUM_GENERATION_PASSES = int(os.getenv("NUM_GENERATION_PASSES", "5"))  # 5 passes now
INPUT_JSON = os.getenv("THEME_JSON", str(ASSETS_DIR / "theme_subtheme_dictionary_v3_enriched.json"))
ENHANCED_DICT_JSON = os.getenv("ENHANCED_DICT_JSON", str(ASSETS_DIR / "Enhanced_Dictionary_v3.json"))
OUT_JSON = os.getenv("OUT_JSON", str(OUTPUT_DIR / "theme_subtheme_dictionary_v3_enriched.json"))
OUT_CSV = os.getenv("OUT_CSV", str(OUTPUT_DIR / "theme_phrase_library.csv"))
METRICS_CSV = os.getenv("METRICS_CSV", str(OUTPUT_DIR / "phrase_generation_metrics.csv"))

REQUEST_TEMPERATURE = float(os.getenv("REQUEST_TEMPERATURE", "0.8"))  # Higher for diversity
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))
RATE_DELAY_SEC = float(os.getenv("RATE_DELAY_SEC", "0.5"))
RETRIES = int(os.getenv("RETRIES", "3"))

# =============================
# LOAD V3 THEME DICTIONARY FOR CONTEXT
# =============================
def load_v3_existing_keywords() -> Dict[tuple, List[str]]:
    """Load existing keywords from theme_subtheme_dictionary_v3.json for context-aware generation"""
    v3_path = Path(INPUT_JSON)
    if not v3_path.exists():
        print(f"⚠️  theme_subtheme_dictionary_v3.json not found at {v3_path}")
        return {}
    
    with open(v3_path, 'r', encoding='utf-8') as f:
        v3_data = json.load(f)
    
    keyword_map = {}
    
    # V3 structure: column_libraries -> column -> parents -> themes -> subthemes
    for col_name, col_lib in v3_data.get('column_libraries', {}).items():
        for parent in col_lib.get('parents', []):
            for theme in parent.get('themes', []):
                theme_name = theme.get('theme_name', '')
                for subtheme in theme.get('subthemes', []):
                    subtheme_name = subtheme.get('name', '')
                    keywords = subtheme.get('keywords_phrases', [])
                    
                    # Store with column context
                    key = (col_name, theme_name, subtheme_name)
                    if keywords:
                        keyword_map[key] = keywords
    
    print(f"✓ Loaded {len(keyword_map)} (column, theme, subtheme) keyword sets from v3 dictionary")
    return keyword_map

V3_KEYWORDS = load_v3_existing_keywords()

# =============================
# LOAD ENHANCED DICTIONARY FOR BUSINESS CONTEXT
# =============================
def load_enhanced_dictionary() -> Dict[str, Any]:
    """Load Enhanced_Dictionary_v3.json with business context examples"""
    dict_path = Path(ENHANCED_DICT_JSON)
    if not dict_path.exists():
        print(f"⚠️  Enhanced_Dictionary_v3.json not found at {dict_path}, skipping business context")
        return {}
    
    with open(dict_path, 'r', encoding='utf-8') as f:
        enhanced_dict = json.load(f)
    
    print(f"✓ Loaded Enhanced Dictionary v3.0")
    
    # Extract business context
    business_context = enhanced_dict.get('business_context', {})
    print(f"   • Business categories: {len(business_context)}")
    
    # Count example phrases
    total_examples = 0
    for category, data in business_context.items():
        if 'example_phrases' in data:
            for col, phrases in data['example_phrases'].items():
                total_examples += len(phrases)
    print(f"   • Total example phrases: {total_examples}")
    
    return enhanced_dict

ENHANCED_DICT = load_enhanced_dictionary()

# Extract business context for easy access
BUSINESS_CONTEXT = ENHANCED_DICT.get('business_context', {})
COLUMN_GUIDANCE = ENHANCED_DICT.get('column_guidance', {})
PHRASE_BUILDING_BLOCKS = ENHANCED_DICT.get('phrase_building_blocks', {})

# =============================
# UK SPELLINGS & TERMS
# =============================
UK_SPELLINGS = {
    "organize": "organise", "organizing": "organising", "organized": "organised",
    "recognize": "recognise", "recognizing": "recognising", "recognized": "recognised",
    "prioritize": "prioritise", "prioritizing": "prioritising", "prioritized": "prioritised",
    "formalize": "formalise", "utilize": "utilise", "utilizing": "utilising", "utilized": "utilised",
    "behavior": "behaviour", "behavioral": "behavioural",
    "center": "centre", "meter": "metre", "odor": "odour", "labor": "labour",
    "color": "colour", "favor": "favour", "honor": "honour", "rumor": "rumour",
    "counseling": "counselling", "counselor": "counsellor",
    "traveling": "travelling", "canceled": "cancelled", "canceling": "cancelling",
    "modeling": "modelling", "program": "programme", "programs": "programmes"
}

US_TO_UK_TERMS = {
    r"\bpto\b": "annual leave",
    r"\bpaid time off\b": "annual leave",
    r"\bvacation(s)?\b": "annual leave",
    r"\bsick day(s)?\b": "sickness absence",
    r"\bseverance\b": "redundancy pay",
    r"\blayoff(s)?\b": "redundancy",
    r"\bhealth insurance\b": "private medical cover",
    r"\bopen enrollment\b": "benefits enrolment",
    r"\bperformance review\b": "appraisal",
    r"\bresume\b": "CV",
    r"\bgas\b": "petrol",
    r"\bcell phone\b": "mobile phone",
    r"\bpaid holiday(s)?\b": "bank holidays",
    r"\bdoctor\b": "GP",
    r"\bsick note\b": "fit note"
}

def replace_manager(text: str) -> str:
    return re.sub(r"(?<!line )\bmanager\b", "line manager", text, flags=re.IGNORECASE)

UK_WORKPLACE_VOCAB = [
    "annual leave", "bank holiday", "TOIL", "line manager", "rota", "shift pattern",
    "overtime", "on‑call", "handover", "stand‑up", "probation", "induction",
    "pay band", "pay grade", "cost of living", "redundancy", "redundancy pay",
    "TUPE", "appraisal", "capability process", "sickness absence", "fit note",
    "GP appointment", "occupational health", "union rep", "ACAS", "pension scheme",
    "mileage claim", "petrol expenses", "hot‑desking", "canteen", "annualised hours",
    "compressed hours", "flexitime", "carer's leave", "bereavement leave"
]

# =============================
# NORMALISATION + DEDUPE
# =============================
def enforce_uk_spelling(text: str) -> str:
    for us, uk in UK_SPELLINGS.items():
        text = re.sub(rf"\b{re.escape(us)}\b", uk, text, flags=re.IGNORECASE)
    return text

def enforce_uk_terms(text: str) -> str:
    for pattern, repl in US_TO_UK_TERMS.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = replace_manager(text)
    return text

# Expanded hard ban list for generic/vague phrases
HARD_BAN = {
    "work is hard", "it is difficult", "things are bad", "please help",
    "help me", "very stressful", "so stressful", "need support", "need help",
    "things are tough", "struggling", "not good", "could be better",
    "more support needed", "better communication", "improve things"
}

def normalise(p: str) -> str:
    p = (p or "").strip()
    if not p:
        return ""
    p = unicodedata.normalize("NFKC", p)
    p = re.sub(r"\s+", " ", p)
    p = enforce_uk_spelling(p)
    p = enforce_uk_terms(p)
    p = p.lower().strip(" .;:!?,")
    return p

def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0

def dedupe(phrases: List[str], threshold: float = 0.65) -> List[str]:
    """Deduplicate with stricter threshold for better diversity"""
    out: List[str] = []
    for p in phrases:
        n = normalise(p)
        if not n or len(n.split()) < 2:
            continue
        if n in HARD_BAN:
            continue
        # Check for generic patterns
        if re.match(r"^(need|want|lack|no|more|better|improve)\s+\w+$", n):
            continue  # Too generic (e.g., "need training", "more support")
        if any(jaccard(n, q) >= threshold for q in out):
            continue
        out.append(n)
    return out

# =============================
# QUALITY SCORING (NO REAL DATA)
# =============================
def score_phrase_quality(phrase: str, subtheme: str) -> Dict[str, float]:
    """Score phrase quality without real data"""
    scores = {}
    
    # Length check (5-15 words is ideal)
    word_count = len(phrase.split())
    scores['length'] = 1.0 if 5 <= word_count <= 15 else 0.5
    
    # Specificity (contains subtheme-related words)
    subtheme_words = set(subtheme.lower().split())
    phrase_words = set(phrase.lower().split())
    overlap = len(subtheme_words & phrase_words)
    scores['specificity'] = min(overlap / len(subtheme_words), 1.0) if subtheme_words else 0.5
    
    # UK vocabulary presence
    uk_vocab_present = any(term in phrase.lower() for term in UK_WORKPLACE_VOCAB)
    scores['uk_vocab'] = 1.0 if uk_vocab_present else 0.7
    
    # Not too generic
    generic_words = {'need', 'want', 'more', 'better', 'improve', 'help', 'support'}
    generic_ratio = len(generic_words & phrase_words) / len(phrase_words) if phrase_words else 0
    scores['not_generic'] = 1.0 - min(generic_ratio * 2, 1.0)
    
    # Overall quality
    scores['overall'] = sum(scores.values()) / len(scores)
    
    return scores

# =============================
# HELPER: GET RELEVANT BUSINESS EXAMPLES
# =============================
def get_relevant_business_examples(column: str, theme: str, subtheme: str) -> List[str]:
    """Extract relevant business examples from Enhanced Dictionary based on theme/subtheme"""
    relevant_examples = []
    
    # Search through business context categories
    for category_name, category_data in BUSINESS_CONTEXT.items():
        # Check if this category is related to the theme
        related_themes = category_data.get('related_themes', [])
        if any(theme.lower() in rt.lower() or rt.lower() in theme.lower() for rt in related_themes):
            # Get example phrases for this column
            example_phrases = category_data.get('example_phrases', {}).get(column, [])
            relevant_examples.extend(example_phrases)
    
    return relevant_examples[:10]  # Limit to 10 most relevant

# =============================
# LLM CALL + PARSING
# =============================
VOCAB_HINT = " | ".join(UK_WORKPLACE_VOCAB[:20])  # First 20 for brevity

SYSTEM_PROMPT = (
    "You are a UK employee survey response generator.\n"
    "Generate REALISTIC phrases that UK employees would actually write.\n"
    "Use British English spelling and workplace vocabulary.\n"
    "Be SPECIFIC to the subtheme - avoid generic phrases.\n"
    "Use first-person perspective where natural.\n"
    "Vary sentence structure and vocabulary.\n"
    "5-15 words per phrase.\n"
    "Return ONLY JSON object with key 'data' containing array of strings."
)

def column_context_prompt(column: str) -> str:
    """Generate column-specific context from Enhanced Dictionary"""
    guidance = COLUMN_GUIDANCE.get(column, {})
    
    if guidance:
        description = guidance.get('description', '')
        tone = guidance.get('tone', '')
        patterns = guidance.get('typical_patterns', [])
        starters = guidance.get('example_starters', [])
        
        return f"""
This column asks: '{description}'
Tone: {tone}
Typical patterns: {', '.join(patterns[:3])}
Example starters: {', '.join(starters[:5])}
""".strip()
    
    # Fallback to original
    contexts = {
        "Wellbeing_Details": "Problem-focused, describing stressors, concerns, difficulties.",
        "Areas_Improve": "Solution-focused, describing needs, requests, suggestions.",
        "Support_Provided": "Positive, describing buffers, coping mechanisms, helpful resources."
    }
    return contexts.get(column, "")

def user_prompt_pass1(column: str, parent: str, theme: str, subtheme: str, 
                      polarity: str, target: int) -> str:
    """First pass: broad generation with high diversity"""
    context = column_context_prompt(column)
    
    return f"""
{context}

Parent Category: {parent}
Theme: {theme}
Subtheme: {subtheme}
Expected Polarity: {polarity}

Task: Generate {target} REALISTIC, SPECIFIC phrases a UK employee would write for this exact subtheme.

Requirements:
- Be SPECIFIC to "{subtheme}" - not generic
- Use British English spelling and workplace terms where natural: {VOCAB_HINT}
- Vary vocabulary and sentence structure
- 5-15 words per phrase
- First-person perspective where natural
- Match the column context above

Output ONLY JSON: {{"data": ["phrase1", "phrase2", ...]}}
""".strip()

def user_prompt_pass2(column: str, subtheme: str, existing: List[str], target: int) -> str:
    """Second pass: fill gaps and add variety"""
    context = column_context_prompt(column)
    
    return f"""
{context}

Subtheme: {subtheme}

Existing phrases:
{json.dumps(existing[:10], ensure_ascii=False)}

Task: Generate {target} MORE phrases that:
- Cover DIFFERENT aspects of "{subtheme}" not in existing phrases
- Use DIFFERENT vocabulary and phrasing
- Are SPECIFIC (not generic)
- Match the column context above

Output ONLY JSON: {{"data": ["phrase1", "phrase2", ...]}}
""".strip()

def user_prompt_pass3(column: str, subtheme: str, existing: List[str], target: int) -> str:
    """Third pass: edge cases and rare expressions"""
    context = column_context_prompt(column)
    
    return f"""
{context}

Subtheme: {subtheme}

Existing phrases cover common cases. Now generate {target} phrases for:
- Edge cases and less common situations
- Specific workplace scenarios
- Nuanced expressions
- Different severity levels

Still SPECIFIC to "{subtheme}", British English, 5-15 words.

Output ONLY JSON: {{"data": ["phrase1", "phrase2", ...]}}
""".strip()

def user_prompt_pass4_v3_context(column: str, theme: str, subtheme: str, 
                                 existing: List[str], v3_keywords: List[str], 
                                 target: int) -> str:
    """Fourth pass: variations using v3 existing keywords for context-aware generation"""
    context = column_context_prompt(column)
    
    # Sample keywords for context (max 15 to avoid token bloat)
    keyword_sample = v3_keywords[:15] if len(v3_keywords) > 15 else v3_keywords
    
    return f"""
{context}

Theme: {theme}
Subtheme: {subtheme}

Existing context-aware keywords from v3 dictionary for this column:
{json.dumps(keyword_sample, ensure_ascii=False)}

Existing generated phrases:
{json.dumps(existing[:5], ensure_ascii=False)}

Task: Generate {target} NEW phrase VARIATIONS that:
- Build on the existing v3 keywords above (expand, rephrase, combine)
- Stay context-aware to the column question type
- Are SPECIFIC to "{subtheme}"
- Use different phrasing than existing phrases
- Match the column context (problem/solution/positive framing)

Output ONLY JSON: {{"data": ["phrase1", "phrase2", ...]}}
""".strip()

def user_prompt_pass5_enhanced_business(column: str, theme: str, subtheme: str, 
                                        existing: List[str], business_examples: List[str], 
                                        target: int) -> str:
    """Fifth pass: use Enhanced Dictionary business context examples"""
    context = column_context_prompt(column)
    
    if not business_examples:
        business_examples = ["No specific business examples found - use general workplace terms"]
    
    return f"""
{context}

Theme: {theme}
Subtheme: {subtheme}

Business context examples from Enhanced Dictionary (real organizational phrases):
{json.dumps(business_examples[:10], ensure_ascii=False)}

These examples show how employees actually talk about business-specific terms like:
- Projects (amp8, pmo, alliance)
- Systems (SAP, ECC, HRS)
- Management (line manager, IPL, senior leadership)
- Operations (site work, field work, reactive work)

Existing generated phrases:
{json.dumps(existing[:5], ensure_ascii=False)}

Task: Generate {target} NEW phrases that:
- Follow the style and patterns of the business examples above
- Incorporate business-specific terms naturally
- Create realistic workplace scenarios
- Stay SPECIFIC to "{subtheme}"
- Match the column context (problem/solution/positive framing)
- Sound like real employee feedback

Output ONLY JSON: {{"data": ["phrase1", "phrase2", ...]}}
""".strip()

def call_llm(messages: List[Dict[str, str]], retries: int = RETRIES) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=REQUEST_TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"}
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            time.sleep(0.8 * attempt)
    if last_err:
        raise last_err
    return ""

def parse_phrases(content: str) -> List[str]:
    if not content:
        return []
    try:
        obj = json.loads(content)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, dict):
            for key in ("data", "phrases", "items", "list"):
                if key in obj and isinstance(obj[key], list):
                    return [str(x) for x in obj[key]]
    except Exception:
        pass
    # Fallback: find first JSON array
    m = re.search(r"\[[\s\S]*\]", content)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(x) for x in arr]
        except Exception:
            return []
    return []

def llm_list(messages: List[Dict[str, str]]) -> List[str]:
    content = call_llm(messages)
    return parse_phrases(content)

# =============================
# MULTI-PASS GENERATION WITH ENHANCED DICTIONARY
# =============================
def generate_phrases_multipass(col: str, parent: str, theme: str,
                               subtheme: str, polarity: str) -> tuple[List[str], Dict[str, Any]]:
    """
    Multi-pass generation:
    1-3: Standard generation (broad, gaps, edge cases)
    4: Context-aware variations using v3 existing keywords
    5: Business-specific using Enhanced Dictionary examples
    """
    all_phrases = []
    metrics = {
        'passes': [],
        'total_generated': 0,
        'total_after_dedupe': 0,
        'avg_quality': 0.0,
        'used_v3_context': False,
        'used_enhanced_business': False
    }
    
    # Pass 1: Broad generation
    print(f"   Pass 1/5: Broad generation...")
    msgs1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_pass1(col, parent, theme, subtheme, polarity, TARGET_PHRASES_PER_PASS)}
    ]
    pass1 = llm_list(msgs1)
    pass1_deduped = dedupe(pass1)
    all_phrases.extend(pass1_deduped)
    metrics['passes'].append({'pass': 1, 'generated': len(pass1), 'after_dedupe': len(pass1_deduped)})
    time.sleep(RATE_DELAY_SEC)
    
    # Pass 2: Fill gaps
    print(f"   Pass 2/5: Filling gaps...")
    msgs2 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_pass2(col, subtheme, all_phrases, TARGET_PHRASES_PER_PASS)}
    ]
    pass2 = llm_list(msgs2)
    pass2_deduped = dedupe(pass2)
    pass2_new = [p for p in pass2_deduped if not any(jaccard(p, existing) >= 0.65 for existing in all_phrases)]
    all_phrases.extend(pass2_new)
    metrics['passes'].append({'pass': 2, 'generated': len(pass2), 'after_dedupe': len(pass2_new)})
    time.sleep(RATE_DELAY_SEC)
    
    # Pass 3: Edge cases
    print(f"   Pass 3/5: Edge cases...")
    msgs3 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_pass3(col, subtheme, all_phrases, TARGET_PHRASES_PER_PASS)}
    ]
    pass3 = llm_list(msgs3)
    pass3_deduped = dedupe(pass3)
    pass3_new = [p for p in pass3_deduped if not any(jaccard(p, existing) >= 0.65 for existing in all_phrases)]
    all_phrases.extend(pass3_new)
    metrics['passes'].append({'pass': 3, 'generated': len(pass3), 'after_dedupe': len(pass3_new)})
    time.sleep(RATE_DELAY_SEC)
    
    # Pass 4: Context-aware variations using v3 existing keywords
    v3_key = (col, theme, subtheme)
    v3_keywords = V3_KEYWORDS.get(v3_key, [])
    if v3_keywords:
        print(f"   Pass 4/5: Context-aware variations (using {len(v3_keywords)} v3 keywords)...")
        msgs4 = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt_pass4_v3_context(col, theme, subtheme, all_phrases, v3_keywords, TARGET_PHRASES_PER_PASS)}
        ]
        pass4 = llm_list(msgs4)
        pass4_deduped = dedupe(pass4)
        pass4_new = [p for p in pass4_deduped if not any(jaccard(p, existing) >= 0.65 for existing in all_phrases)]
        all_phrases.extend(pass4_new)
        metrics['passes'].append({'pass': 4, 'generated': len(pass4), 'after_dedupe': len(pass4_new)})
        metrics['used_v3_context'] = True
    else:
        print(f"   Pass 4/5: Skipped (no v3 keywords found for {col} → {theme} → {subtheme})")
        metrics['passes'].append({'pass': 4, 'generated': 0, 'after_dedupe': 0})
    time.sleep(RATE_DELAY_SEC)
    
    # Pass 5: Business-specific using Enhanced Dictionary
    business_examples = get_relevant_business_examples(col, theme, subtheme)
    if business_examples or BUSINESS_CONTEXT:
        example_count = len(business_examples) if business_examples else 0
        print(f"   Pass 5/5: Enhanced business context ({example_count} relevant examples)...")
        msgs5 = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt_pass5_enhanced_business(col, theme, subtheme, all_phrases, business_examples, TARGET_PHRASES_PER_PASS)}
        ]
        pass5 = llm_list(msgs5)
        pass5_deduped = dedupe(pass5)
        pass5_new = [p for p in pass5_deduped if not any(jaccard(p, existing) >= 0.65 for existing in all_phrases)]
        all_phrases.extend(pass5_new)
        metrics['passes'].append({'pass': 5, 'generated': len(pass5), 'after_dedupe': len(pass5_new)})
        metrics['used_enhanced_business'] = True
    else:
        print(f"   Pass 5/5: Skipped (no Enhanced Dictionary found)")
        metrics['passes'].append({'pass': 5, 'generated': 0, 'after_dedupe': 0})
    
    # Calculate quality scores
    quality_scores = [score_phrase_quality(p, subtheme)['overall'] for p in all_phrases]
    metrics['total_generated'] = sum(p['generated'] for p in metrics['passes'])
    metrics['total_after_dedupe'] = len(all_phrases)
    metrics['avg_quality'] = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    print(f"   ✓ Generated {len(all_phrases)} unique phrases (avg quality: {metrics['avg_quality']:.2f})")
    
    return all_phrases, metrics

# =============================
# FLATTEN → CSV
# =============================
def flatten_rows(col_name: str, parent: str, theme: str, subtheme: str,
                 polarity: str, phrases: List[str]) -> List[List[str]]:
    return [[col_name, parent, theme, subtheme, polarity, p] for p in phrases]

def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

# =============================
# MAIN PIPELINE
# =============================
def main():
    print("="*80)
    print("COMPREHENSIVE PHRASE LIBRARY GENERATOR")
    print("ENHANCED: V3 (context-aware) + Enhanced Dictionary (business examples)")
    print("="*80)
    
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        schema = json.load(f)

    enriched = deepcopy(schema)
    rows_out: List[List[str]] = []
    metrics_out: List[Dict[str, Any]] = []

    columns_meta = safe_get(schema, "metadata", "columns", default=[])
    libraries = safe_get(schema, "column_libraries", default={})

    total_subthemes = sum(
        len(st_list)
        for col_lib in libraries.values()
        for parent in col_lib.get("parents", [])
        for theme in parent.get("themes", [])
        for st_list in [theme.get("subthemes", [])]
    )
    
    current = 0

    for col_name in columns_meta:
        col_lib = libraries.get(col_name, {})
        parents = col_lib.get("parents", [])

        for parent_block in parents:
            parent_name = parent_block.get("parent_name", "")
            for theme_block in parent_block.get("themes", []):
                theme_name = theme_block.get("theme_name", "")
                for st in theme_block.get("subthemes", []):
                    current += 1
                    sub_name = st.get("name", "")
                    polarity = st.get("default_polarity", "Either")

                    print(f"\n[{current}/{total_subthemes}] {col_name} | {parent_name} | {theme_name} | {sub_name}")

                    try:
                        phrases, metrics = generate_phrases_multipass(
                            col_name, parent_name, theme_name, sub_name, polarity
                        )
                    except Exception as e:
                        print(f"   ✗ Error: {e}")
                        phrases, metrics = [], {}

                    st["keywords_phrases"] = phrases
                    rows_out.extend(flatten_rows(col_name, parent_name, theme_name, sub_name, polarity, phrases))
                    
                    metrics_out.append({
                        'column': col_name,
                        'parent': parent_name,
                        'theme': theme_name,
                        'subtheme': sub_name,
                        'polarity': polarity,
                        'phrase_count': len(phrases),
                        'avg_quality': metrics.get('avg_quality', 0.0),
                        'total_generated': metrics.get('total_generated', 0),
                        'total_after_dedupe': metrics.get('total_after_dedupe', 0),
                        'used_v3_context': metrics.get('used_v3_context', False),
                        'used_enhanced_business': metrics.get('used_enhanced_business', False)
                    })

    # Save enriched JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    # Save phrase library CSV
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["column", "parent", "theme", "subtheme", "polarity", "phrase"])
        writer.writerows(rows_out)

    # Save metrics CSV
    with open(METRICS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['column', 'parent', 'theme', 'subtheme', 'polarity', 
                                                'phrase_count', 'avg_quality', 'total_generated', 
                                                'total_after_dedupe', 'used_v3_context', 'used_enhanced_business'])
        writer.writeheader()
        writer.writerows(metrics_out)

    print("\n" + "="*80)
    print("✓ GENERATION COMPLETE")
    print("="*80)
    print(f"Enriched JSON: {OUT_JSON}")
    print(f"Phrase library: {OUT_CSV}")
    print(f"Metrics: {METRICS_CSV}")
    print(f"Total phrases: {len(rows_out)}")
    print(f"Avg phrases per subtheme: {len(rows_out) / len(metrics_out):.1f}")
    
    # Summary of context usage
    v3_used_count = sum(1 for m in metrics_out if m.get('used_v3_context', False))
    business_used_count = sum(1 for m in metrics_out if m.get('used_enhanced_business', False))
    print(f"Subthemes with v3 context: {v3_used_count}/{len(metrics_out)}")
    print(f"Subthemes with Enhanced business context: {business_used_count}/{len(metrics_out)}")

if __name__ == "__main__":
    main()
