#!/usr/bin/env bash
set -euo pipefail

# Clause segmentation + per-clause sentiment
# Split on contrast markers (but/however/although) and classify each clause.

# TODO: implement python module that accepts:
#   --input <csv>
#   --markers <list>
#   --model <sentiment model>
#
# Example (placeholder):
# python -m advanced_taxonomy.clause_sentiment \
#   --input Data/wellbeing.csv \
#   --markers "but,however,although" \
#   --model cardiffnlp/twitter-roberta-base-sentiment-latest

echo "Clause sentiment stub. Implement advanced_taxonomy.clause_sentiment"
