# Segmentation & Clause-Level Analysis

This module provides contrast-aware segmentation and per-clause analysis to avoid
mixing positive and negative sentiment in a single response.

## Why it exists
Survey responses often contain mixed sentiment:
`"Workload is high, but my manager is supportive."`

Segmentation splits on contrast markers and analyzes each clause separately so
the final output can say "unhappy with workload" and "happy with manager support".

## What it does
1. Split text on contrast markers (but/however/although/etc.).
2. Drop tiny segments (min_tokens).
3. Run taxonomy per segment (keyword or semantic, including cross-encoder if enabled).
4. Run sentiment per segment.
5. Write segment-level outputs.

## Files
- `segmentation.py`:
  - `build_segments()` creates segment rows
  - `analyze_segments()` assigns themes + sentiment per segment
- `segment_summary.py`:
  - `segment_sentiment_summary()` aggregates to a deliverable summary

## YAML Toggles
Enable segmentation and outputs in `config/pipeline_settings.yaml`:

```yaml
segmentation:
  enabled: true
  min_tokens: 3
  detect_contrast: true
  fallback_to_whole_text: true

output:
  generate_segmented: true
  generate_detailed_segments: true
  segmented_analysis_filename: "segmented_analysis.csv"
  segments_detailed_filename: "segments_detailed.csv"
  generate_segment_sentiment_summary: true
  segment_sentiment_summary_filename: "segment_sentiment_summary.csv"
```

## Outputs
- `segmented_analysis.csv`:
  - Compact: ID, TextColumn, segment_text, theme, sentiment
- `segments_detailed.csv`:
  - Full segment-level details (scores, evidence, polarity)
- `segment_sentiment_summary.csv`:
  - Aggregated per TextColumn + theme + subtheme
  - Outputs "Users are happy/unhappy with X"
- `absa_segment_sentiment.csv`:
  - Aspect-based sentiment per segment (theme/subtheme as aspect)

## Notes
- Segmentation is opt-in (defaults to off).
- Works with keyword or semantic taxonomy.
- If cross-encoder is enabled in semantic settings, segments are re-ranked.
- ABSA is opt-in (set `absa.enabled: true` and `output.generate_absa: true`).
