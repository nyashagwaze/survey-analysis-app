"""
Column Wordcloud Generator

Creates clean wordclouds per text column with phrase-preservation
and aggressive junk-word filtering using config/profiles/{profile}/dictionary.yaml.
"""
from collections import Counter
from pathlib import Path
import argparse
import re

import pandas as pd
from wordcloud import WordCloud, STOPWORDS

from ..config_runtime import resolve_base_dir
WS = re.compile(r"\s+")
PUNCT = re.compile(r"[^a-z0-9_\s']+")


# --------------------------
# Config helpers
# --------------------------

def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except Exception as exc:
        raise ImportError("Missing dependency: pyyaml. Install with `pip install pyyaml`.") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def resolve_path(base_dir: Path, rel_or_abs: str) -> Path:
    if rel_or_abs is None:
        return None
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_dictionary_config(path: Path) -> dict:
    if path is None or not path.exists():
        return {
            "business_map": {},
            "merge_map": {},
            "force_phrases": [],
            "unigram_whitelist": set(),
            "always_drop": set(),
            "must_be_phrase": set(),
        }

    cfg = load_yaml(path)

    business_map = {str(k).lower(): str(v).lower() for k, v in (cfg.get("BUSINESS_MAP") or {}).items()}
    merge_map = {str(k).lower(): str(v).lower() for k, v in (cfg.get("MERGE_TO_CANONICAL") or {}).items()}

    force_phrases = [str(x).lower() for x in (cfg.get("FORCE_PHRASES") or [])]
    unigram_whitelist = set(str(x).lower() for x in (cfg.get("UNIGRAM_WHITELIST") or []))
    unigram_whitelist |= set(str(x).lower() for x in (cfg.get("DOMAIN_WHITELIST") or []))
    unigram_whitelist |= set(str(x).lower() for x in (cfg.get("ACCRONYM_WHITELIST") or []))

    always_drop = set(str(x).lower() for x in (cfg.get("ALWAYS_DROP_UNIGRAMS") or []))
    always_drop |= set(str(x).lower() for x in (cfg.get("ALWAYS_BLOCK") or []))

    must_be_phrase = set(str(x).lower() for x in (cfg.get("MUST_BE_PHRASE") or []))

    return {
        "business_map": business_map,
        "merge_map": merge_map,
        "force_phrases": force_phrases,
        "unigram_whitelist": unigram_whitelist,
        "always_drop": always_drop,
        "must_be_phrase": must_be_phrase,
    }


# --------------------------
# Text normalization
# --------------------------

def basic_normalise(text: str) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    if not text or text == "nan":
        return ""

    # normalize curly quotes
    text = text.translate(str.maketrans({
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": "\"",
        "\u201d": "\"",
    }))

    text = PUNCT.sub(" ", text)
    text = WS.sub(" ", text).strip()
    return text


def apply_business_map(text: str, business_map: dict) -> str:
    out = text
    for k, v in sorted(business_map.items(), key=lambda kv: (-len(kv[0]), kv[0])):
        out = re.sub(rf"\b{re.escape(k)}\b", v, out, flags=re.IGNORECASE)
    return out


def merge_to_canonical(text: str, merge_map: dict) -> str:
    toks = text.split()
    new = [merge_map.get(tok, tok) for tok in toks]
    return " ".join(new)


def force_phrases(text: str, forced: list) -> str:
    out = text
    for ph in forced:
        if "_" in ph:
            plain = ph.replace("_", " ")
            out = out.replace(plain, ph)
    return out


def preprocess_for_wordcloud(text: str, assets: dict) -> str:
    t = basic_normalise(text)
    if not t:
        return ""
    if assets.get("business_map"):
        t = apply_business_map(t, assets["business_map"])
    if assets.get("merge_map"):
        t = merge_to_canonical(t, assets["merge_map"])
    if assets.get("force_phrases"):
        t = force_phrases(t, assets["force_phrases"])
    return t


# --------------------------
# Wordcloud logic
# --------------------------

def build_stopwords(extra: set = None) -> set:
    base = set(STOPWORDS)
    extra = extra or set()
    base |= extra
    base |= {
        "amp",
        "etc",
        "also",
        "still",
        "really",
        "maybe",
        "much",
        "many",
        "lot",
        "lots",
        "thing",
        "things",
        "everyone",
        "everything",
        "anything",
        "nothing",
        "could",
        "would",
        "should",
        "im",
        "ive",
        "dont",
        "didnt",
        "cant",
        "couldnt",
        "wouldnt",
        "thats",
        "wasnt",
        "isnt",
        "arent",
        "doesnt",
        "hadnt",
        "having",
        "get",
        "gets",
        "got",
        "going",
        "go",
    }
    return base


def tokenize_text(
    text: str,
    stopwords: set,
    must_be_phrase: set,
    unigram_whitelist: set,
    keep_phrases_only: bool = True,
    min_token_len: int = 3
) -> list:
    tokens = []
    for tok in text.split():
        if "_" in tok:
            tokens.append(tok)
            continue

        if keep_phrases_only:
            continue

        if tok in stopwords:
            continue
        if tok in must_be_phrase:
            continue
        if tok.isdigit():
            continue
        if len(tok) < min_token_len and tok not in unigram_whitelist:
            continue
        tokens.append(tok)

    return tokens


def build_frequencies(
    texts,
    assets: dict,
    keep_phrases_only: bool = True,
    min_token_len: int = 3,
    min_freq: int = 2,
    display_phrases_with_spaces: bool = True
) -> Counter:
    stopwords = build_stopwords(extra=assets.get("always_drop", set()))
    must_be_phrase = assets.get("must_be_phrase", set())
    unigram_whitelist = assets.get("unigram_whitelist", set())

    freq = Counter()
    for text in texts:
        t = preprocess_for_wordcloud(text, assets)
        if not t:
            continue
        tokens = tokenize_text(
            t,
            stopwords=stopwords,
            must_be_phrase=must_be_phrase,
            unigram_whitelist=unigram_whitelist,
            keep_phrases_only=keep_phrases_only,
            min_token_len=min_token_len
        )
        if display_phrases_with_spaces:
            tokens = [t.replace("_", " ") for t in tokens]
        freq.update(tokens)

    if min_freq > 1:
        freq = Counter({k: v for k, v in freq.items() if v >= min_freq})

    return freq


def render_wordcloud(
    freq: Counter,
    output_path: Path,
    width: int = 2400,
    height: int = 1400,
    background_color: str = "white",
    colormap: str = "tab20c",
    max_words: int = 140,
    min_font_size: int = 14,
    max_font_size: int = 220,
    prefer_horizontal: float = 1.0,
    relative_scaling: float = 0.3,
    font_path: str = None
) -> None:
    if not freq:
        return

    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        max_words=max_words,
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        prefer_horizontal=prefer_horizontal,
        relative_scaling=relative_scaling,
        normalize_plurals=True,
        font_path=font_path
    ).generate_from_frequencies(freq)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wc.to_file(str(output_path))


def generate_column_wordclouds(
    input_csv: Path,
    text_columns: list,
    output_dir: Path,
    dictionary_path: Path = None,
    keep_phrases_only: bool = True,
    min_token_len: int = 3,
    min_freq: int = 2,
    display_phrases_with_spaces: bool = True,
    use_processed: bool = True,
    max_words: int = 140,
    min_font_size: int = 14,
    max_font_size: int = 220,
    relative_scaling: float = 0.3,
    prefer_horizontal: float = 1.0,
    export_frequencies: bool = False
) -> None:
    df = pd.read_csv(input_csv)
    assets = load_dictionary_config(dictionary_path)

    for col in text_columns:
        if col not in df.columns:
            continue

        source_col = col
        if use_processed and f"{col}_processed" in df.columns:
            source_col = f"{col}_processed"

        texts = df[source_col].dropna().astype(str).tolist()
        freq = build_frequencies(
            texts,
            assets=assets,
            keep_phrases_only=keep_phrases_only,
            min_token_len=min_token_len,
            min_freq=min_freq,
            display_phrases_with_spaces=display_phrases_with_spaces
        )

        if not freq and keep_phrases_only:
            # fallback to unigrams if no phrases were detected
            freq = build_frequencies(
                texts,
                assets=assets,
                keep_phrases_only=False,
                min_token_len=min_token_len,
                min_freq=min_freq,
                display_phrases_with_spaces=display_phrases_with_spaces
            )

        out_file = output_dir / f"wordcloud_{col}.png"
        render_wordcloud(
            freq,
            out_file,
            max_words=max_words,
            min_font_size=min_font_size,
            max_font_size=max_font_size,
            relative_scaling=relative_scaling,
            prefer_horizontal=prefer_horizontal
        )

        if export_frequencies:
            freq_df = pd.DataFrame(freq.most_common(), columns=["token", "count"])
            freq_df.to_csv(output_dir / f"wordcloud_{col}_freq.csv", index=False)


# --------------------------
# CLI
# --------------------------

def main():
    here = Path(__file__).resolve()
    project_root = here.parents[3]
    default_settings = project_root / "config/pipeline_settings.yaml"

    parser = argparse.ArgumentParser(description="Generate clean wordclouds per column.")
    parser.add_argument("--settings", default=str(default_settings), help="Path to pipeline_settings.yaml")
    parser.add_argument("--input", dest="input_csv", default=None, help="Input CSV path")
    parser.add_argument("--outdir", dest="outdir", default=None, help="Output directory")
    parser.add_argument("--dictionary", dest="dictionary_path", default=None, help="dictionary.yaml path")
    parser.add_argument("--columns", dest="columns", default=None, help="Comma-separated list of columns")
    parser.add_argument("--min-freq", dest="min_freq", type=int, default=None, help="Minimum frequency to keep token")
    parser.add_argument("--min-token-len", dest="min_token_len", type=int, default=None, help="Minimum token length")
    parser.add_argument("--max-words", dest="max_words", type=int, default=None, help="Max words in wordcloud")
    parser.add_argument("--min-font-size", dest="min_font_size", type=int, default=None, help="Minimum font size")
    parser.add_argument("--max-font-size", dest="max_font_size", type=int, default=None, help="Maximum font size")
    parser.add_argument("--relative-scaling", dest="relative_scaling", type=float, default=None, help="Relative scaling (0-1)")
    parser.add_argument("--prefer-horizontal", dest="prefer_horizontal", type=float, default=None, help="Prefer horizontal (0-1)")
    parser.add_argument("--keep-phrases-only", dest="keep_phrases_only", action="store_true")
    parser.add_argument("--allow-unigrams", dest="keep_phrases_only", action="store_false")
    parser.add_argument("--no-processed", dest="use_processed", action="store_false", help="Do not use *_processed columns")
    parser.add_argument("--use-processed", dest="use_processed", action="store_true", help="Use *_processed columns if present")
    parser.add_argument("--export-frequencies", dest="export_frequencies", action="store_true", help="Save token frequencies CSV")
    parser.add_argument("--display-phrases-with-spaces", dest="display_phrases_with_spaces", action="store_true")
    parser.add_argument("--keep-underscores", dest="display_phrases_with_spaces", action="store_false")
    parser.set_defaults(keep_phrases_only=True, use_processed=True, display_phrases_with_spaces=True, export_frequencies=False)
    args = parser.parse_args()

    settings_path = Path(args.settings)
    settings = load_yaml(settings_path) if settings_path.exists() else {}
    paths_cfg = settings.get("paths", {})

    package_dir = here.parents[1]
    base_dir = resolve_base_dir(settings, settings_path=settings_path, pipeline_dir=package_dir)

    input_csv = args.input_csv or paths_cfg.get("input_csv")
    if not input_csv:
        raise ValueError("input_csv is required (pass --input or set paths.input_csv).")

    input_path = resolve_path(base_dir, input_csv)
    dictionary_path = args.dictionary_path or paths_cfg.get("dictionary")
    dictionary_path = resolve_path(base_dir, dictionary_path) if dictionary_path else None

    outdir = args.outdir or paths_cfg.get("output_wordclouds") or "outputs/wordclouds"
    outdir_path = resolve_path(base_dir, outdir)

    wc_cfg = settings.get("wordcloud", {})
    text_columns = settings.get("text_columns", [])
    if args.columns:
        text_columns = [c.strip() for c in args.columns.split(",") if c.strip()]

    if not text_columns:
        raise ValueError("No text_columns found (pass --columns or set text_columns in settings).")

    min_freq = args.min_freq if args.min_freq is not None else wc_cfg.get("min_freq", 2)
    min_token_len = args.min_token_len if args.min_token_len is not None else wc_cfg.get("min_token_len", 3)
    max_words = args.max_words if args.max_words is not None else wc_cfg.get("max_words", 140)
    min_font_size = args.min_font_size if args.min_font_size is not None else wc_cfg.get("min_font_size", 14)
    max_font_size = args.max_font_size if args.max_font_size is not None else wc_cfg.get("max_font_size", 220)
    relative_scaling = args.relative_scaling if args.relative_scaling is not None else wc_cfg.get("relative_scaling", 0.3)
    prefer_horizontal = args.prefer_horizontal if args.prefer_horizontal is not None else wc_cfg.get("prefer_horizontal", 1.0)

    generate_column_wordclouds(
        input_csv=input_path,
        text_columns=text_columns,
        output_dir=outdir_path,
        dictionary_path=dictionary_path,
        keep_phrases_only=args.keep_phrases_only,
        min_token_len=min_token_len,
        min_freq=min_freq,
        display_phrases_with_spaces=args.display_phrases_with_spaces,
        use_processed=args.use_processed,
        max_words=max_words,
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        relative_scaling=relative_scaling,
        prefer_horizontal=prefer_horizontal,
        export_frequencies=args.export_frequencies
    )


if __name__ == "__main__":
    main()
