"""
Semantic Taxonomy Matcher - Sentiment-Based Classification

Replaces keyword matching with semantic similarity using sentence transformers.
Matches survey responses to themes based on:
1. Semantic similarity (cosine similarity of embeddings)
2. Sentiment polarity alignment
3. Column-specific enforcement

Performance optimizations:
- Batch encoding of all phrases at startup (one-time cost)
- Batch encoding of survey responses
- Vectorized similarity calculations
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import pandas as pd
from pathlib import Path
import json


class SemanticTaxonomyMatcher:
    """
    Semantic similarity-based taxonomy matcher using sentence transformers.
    
    Key features:
    - Pre-encodes all theme phrases once at initialization
    - Batch encodes survey responses for efficiency
    - Uses cosine similarity for semantic matching
    - Filters by sentiment polarity alignment
    - Respects column-specific theme libraries
    """
    
    _MODEL_CACHE = {}

    def __init__(
        self,
        enriched_json_path: Path,
        model_name: str = 'all-MiniLM-L6-v2',
        similarity_threshold: float = 0.35,
        top_k: int = 3,
        sentiment_weight: float = 0.15
    ):
        """
        Initialize the semantic matcher.
        
        Args:
            enriched_json_path: Path to enriched theme dictionary JSON
            model_name: Sentence transformer model (default: all-MiniLM-L6-v2, 384 dims, fast)
            similarity_threshold: Minimum cosine similarity to consider a match (0-1)
            top_k: Maximum number of themes to return per response
            sentiment_weight: Boost for polarity alignment (0-1)
        """
        print(f"\n Initializing Semantic Taxonomy Matcher...")
        print(f"   Model: {model_name}")
        print(f"   Similarity threshold: {similarity_threshold}")
        print(f"   Top-k matches: {top_k}")
        print(f"   Sentiment weight: {sentiment_weight}")
        
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.sentiment_weight = sentiment_weight
        
        # Load sentence transformer model (cached across runs)
        if model_name in self._MODEL_CACHE:
            self.model = self._MODEL_CACHE[model_name]
            print(f"\n Using cached sentence transformer model...")
        else:
            print(f"\n Loading sentence transformer model...")
            self.model = SentenceTransformer(model_name)
            self._MODEL_CACHE[model_name] = self.model
        print(f"    Model ready: {self.model.get_sentence_embedding_dimension()} dimensions")
        
        # Load and encode theme library
        print(f"\n Loading enriched theme library...")
        self._load_and_encode_themes(enriched_json_path)
        
    def _load_and_encode_themes(self, json_path: Path):
        """
        Load enriched JSON and pre-encode all phrases.
        
        Creates lookup structures:
        - phrase_embeddings: numpy array of all phrase embeddings
        - phrase_metadata: list of dicts with theme/subtheme/polarity/column info
        - column_phrase_indices: dict mapping column -> list of phrase indices
        """
        with open(json_path, 'r') as f:
            enriched = json.load(f)
        
        # Collect all phrases with metadata
        all_phrases = []
        phrase_metadata = []
        column_phrase_indices = {}
        
        for col, col_data in enriched['column_libraries'].items():
            col_indices = []
            
            for parent in col_data['parents']:
                parent_name = parent['parent_name']
                
                for theme in parent['themes']:
                    theme_name = theme['theme_name']
                    
                    for subtheme in theme['subthemes']:
                        subtheme_name = subtheme['name']
                        polarity = subtheme['default_polarity']
                        phrases = subtheme['keywords_phrases']
                        
                        for phrase in phrases:
                            idx = len(all_phrases)
                            all_phrases.append(phrase)
                            phrase_metadata.append({
                                'column': col,
                                'parent': parent_name,
                                'theme': theme_name,
                                'subtheme': subtheme_name,
                                'polarity': polarity,
                                'phrase': phrase
                            })
                            col_indices.append(idx)
            
            column_phrase_indices[col] = col_indices
        
        print(f"   Total phrases to encode: {len(all_phrases)}")
        print(f"   Columns: {list(column_phrase_indices.keys())}")
        
        # Batch encode all phrases (this is the one-time cost)
        print(f"\n Encoding all phrases (one-time operation)...")
        print(f"   This may take 30-60 seconds for ~8000 phrases...")
        
        self.phrase_embeddings = self.model.encode(
            all_phrases,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        self.phrase_metadata = phrase_metadata
        self.column_phrase_indices = column_phrase_indices
        
        print(f"    Encoded {len(all_phrases)} phrases")
        print(f"   Embedding shape: {self.phrase_embeddings.shape}")
        
        # Normalize embeddings for faster cosine similarity
        self.phrase_embeddings = self.phrase_embeddings / np.linalg.norm(
            self.phrase_embeddings, axis=1, keepdims=True
        )
        print(f"    Embeddings normalized for cosine similarity")
    
    def match_batch(
        self,
        texts: List[str],
        columns: List[str],
        sentiment_labels: Optional[List[str]] = None
    ) -> List[List[Dict]]:
        """
        Match a batch of texts to themes using semantic similarity.
        
        Args:
            texts: List of survey response texts
            columns: List of column names (same length as texts)
            sentiment_labels: Optional list of sentiment labels ('Positive', 'Negative', 'Neutral')
        
        Returns:
            List of lists of match dicts, one list per input text
        """
        if len(texts) == 0:
            return []
        
        # Batch encode all input texts
        text_embeddings = self.model.encode(
            texts,
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Normalize for cosine similarity
        text_embeddings = text_embeddings / np.linalg.norm(
            text_embeddings, axis=1, keepdims=True
        )
        
        # Match each text
        results = []
        for i, (text, col) in enumerate(zip(texts, columns)):
            text_emb = text_embeddings[i]
            sentiment = sentiment_labels[i] if sentiment_labels else None
            
            matches = self._match_single(
                text_emb=text_emb,
                column=col,
                sentiment_label=sentiment,
                original_text=text
            )
            results.append(matches)
        
        return results
    
    def _match_single(
        self,
        text_emb: np.ndarray,
        column: str,
        sentiment_label: Optional[str],
        original_text: str
    ) -> List[Dict]:
        """
        Match a single text embedding to themes.
        
        Returns top-k matches with scores, filtered by column and threshold.
        """
        # Get phrase indices for this column
        col_indices = self.column_phrase_indices.get(column, [])
        if not col_indices:
            return []
        
        # Calculate cosine similarities (dot product of normalized vectors)
        col_embeddings = self.phrase_embeddings[col_indices]
        similarities = np.dot(col_embeddings, text_emb)
        
        # Apply sentiment boost if available
        if sentiment_label and self.sentiment_weight > 0:
            for idx, phrase_idx in enumerate(col_indices):
                phrase_polarity = self.phrase_metadata[phrase_idx]['polarity']
                
                # Boost score if sentiment aligns
                if self._sentiment_aligns(sentiment_label, phrase_polarity):
                    similarities[idx] += self.sentiment_weight
        
        # Get top-k matches above threshold
        top_indices = np.argsort(similarities)[::-1][:self.top_k * 2]  # Get extra for filtering
        
        matches = []
        for idx in top_indices:
            score = similarities[idx]
            if score < self.similarity_threshold:
                break
            
            phrase_idx = col_indices[idx]
            metadata = self.phrase_metadata[phrase_idx]
            
            matches.append({
                'theme': metadata['theme'],
                'subtheme': metadata['subtheme'],
                'parent_theme': metadata['parent'],
                'score': float(score),
                'polarity': metadata['polarity'],
                'matched_phrase': metadata['phrase'],
                'match_method': 'semantic_similarity'
            })
            
            if len(matches) >= self.top_k:
                break
        
        return matches
    
    def _sentiment_aligns(self, sentiment_label: str, polarity: str) -> bool:
        """
        Check if sentiment label aligns with expected polarity.
        """
        if polarity == 'Either':
            return True
        
        if sentiment_label == 'Positive' and polarity == 'Positive':
            return True
        
        if sentiment_label == 'Negative' and polarity == 'Negative':
            return True
        
        return False


def assign_taxonomy_semantic(
    df_spark,
    text_columns: List[str],
    enriched_json_path: Path,
    model_name: str = 'all-MiniLM-L6-v2',
    similarity_threshold: float = 0.35,
    top_k: int = 3
) -> pd.DataFrame:
    """
    Assign taxonomy to survey responses using semantic similarity.
    
    Args:
        df_spark: Spark DataFrame with text columns and sentiment
        text_columns: List of text column names
        enriched_json_path: Path to enriched theme dictionary
        model_name: Sentence transformer model name
        similarity_threshold: Minimum similarity for matches
        top_k: Maximum themes per response
    
    Returns:
        Pandas DataFrame with assignments (ID, TextColumn, theme, subtheme, parent_theme, score, match_method)
    """
    print("\n" + "="*80)
    print("SEMANTIC TAXONOMY MATCHING")
    print("="*80)
    
    # Initialize matcher (encodes all phrases once)
    matcher = SemanticTaxonomyMatcher(
        enriched_json_path=enriched_json_path,
        model_name=model_name,
        similarity_threshold=similarity_threshold,
        top_k=top_k
    )
    
    # Convert to Pandas for processing
    print("\n Converting DataFrame to Pandas (if needed)...")
    if hasattr(df_spark, "toPandas"):
        df_pd = df_spark.toPandas()
    elif isinstance(df_spark, pd.DataFrame):
        df_pd = df_spark.copy()
    else:
        df_pd = pd.DataFrame(df_spark)
    
    # Collect all texts to match
    print("\n Preparing texts for matching...")
    all_texts = []
    all_columns = []
    all_ids = []
    all_sentiments = []
    all_is_meaningful = []
    all_response_detail = []
    
    for col in text_columns:
        processed_col = f"{col}_processed"
        meaningful_col = f"{col}_is_meaningful"
        
        for _, row in df_pd.iterrows():
            text = row.get(processed_col, None)
            if text is None or (isinstance(text, float) and pd.isna(text)):
                text = row.get(col, '')
            meaningful_val = row.get(meaningful_col, None)
            if meaningful_val is None or (isinstance(meaningful_val, float) and pd.isna(meaningful_val)):
                is_meaningful = None
            else:
                is_meaningful = bool(meaningful_val)
            response_detail = row.get(f"{col}_response_detail", None)
            
            all_texts.append(text if pd.notna(text) else '')
            all_columns.append(col)
            row_id = row["ID"] if "ID" in df_pd.columns else row.name
            all_ids.append(row_id)
            all_sentiments.append(row.get('sentiment_label', None))
            all_is_meaningful.append(is_meaningful)
            all_response_detail.append(response_detail)
    
    print(f"   Total texts to match: {len(all_texts)}")
    meaningful_count = sum(1 for v in all_is_meaningful if v is True)
    print(f"   Meaningful texts: {meaningful_count}")
    
    # Batch match all texts
    print("\n Matching texts to themes (batch processing)...")
    all_matches = matcher.match_batch(
        texts=all_texts,
        columns=all_columns,
        sentiment_labels=all_sentiments
    )
    
    # Build assignments DataFrame
    print("\n Building assignments DataFrame...")
    assignments = []
    
    for i, matches in enumerate(all_matches):
        text_id = all_ids[i]
        text_col = all_columns[i]
        is_meaningful = all_is_meaningful[i]
        response_detail = all_response_detail[i]
        col_has_phrases = bool(matcher.column_phrase_indices.get(text_col, []))
        
        if is_meaningful is False:
            # Dismissed text
            assignments.append({
                'ID': text_id,
                'TextColumn': text_col,
                'is_meaningful': False,
                'theme': 'Brief response',
                'subtheme': response_detail or 'No',
                'parent_theme': 'Brief response',
                'rule_score': 0.0,
                'expected_polarity': '',
                'match_method': 'dismissed',
                'evidence': '',
                'reason': f"dismissed:{response_detail}" if response_detail else "dismissed"
            })
        elif not matches:
            # No matches found
            reason = "no_phrases_for_column" if not col_has_phrases else "below_threshold"
            assignments.append({
                'ID': text_id,
                'TextColumn': text_col,
                'is_meaningful': True,
                'theme': '',
                'subtheme': '',
                'parent_theme': '',
                'rule_score': 0.0,
                'expected_polarity': '',
                'match_method': 'none',
                'evidence': '',
                'reason': reason
            })
        else:
            # Concatenate multiple matches with ' | '
            themes = ' | '.join([m['theme'] for m in matches])
            subthemes = ' | '.join([m['subtheme'] for m in matches])
            parents = ' | '.join([m['parent_theme'] for m in matches])
            avg_score = np.mean([m['score'] for m in matches])
            evidence = ' | '.join([m.get('matched_phrase', '') for m in matches if m.get('matched_phrase')])
            
            assignments.append({
                'ID': text_id,
                'TextColumn': text_col,
                'is_meaningful': True,
                'theme': themes,
                'subtheme': subthemes,
                'parent_theme': parents,
                'rule_score': avg_score,
                'expected_polarity': matches[0]['polarity'],
                'match_method': 'semantic_similarity',
                'evidence': evidence,
                'reason': 'matched'
            })
    
    df_assignments = pd.DataFrame(assignments)
    
    # Print statistics
    print(f"\n Matching complete:")
    print(f"   Total assignments: {len(df_assignments)}")
    print(f"   Dismissed: {(df_assignments['match_method'] == 'dismissed').sum()}")
    print(f"   Matched: {(df_assignments['match_method'] == 'semantic_similarity').sum()}")
    print(f"   No match: {(df_assignments['match_method'] == 'none').sum()}")
    print(f"   Avg score (matched): {df_assignments[df_assignments['match_method'] == 'semantic_similarity']['rule_score'].mean():.3f}")
    
    return df_assignments
