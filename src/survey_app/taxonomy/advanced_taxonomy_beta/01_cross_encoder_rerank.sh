#!/usr/bin/env bash
set -euo pipefail

# Cross-encoder reranking pipeline
# 1) Use bi-encoder (semantic_taxonomy) to retrieve top-N candidate themes
# 2) Re-rank candidates with a cross-encoder for final precision
# 3) Emit final theme/subtheme + score

python "$(dirname "$0")/cross_encoder_rerank.py" \
  --settings "$(dirname "$0")/../../config/pipeline_settings.yaml" \
  --bi-top-k 15 \
  --bi-threshold 0.20 \
  --top-k 3 \
  --ce-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --scores-output "$(dirname "$0")/../../outputs/tables/semantic_ce_scores.csv"
