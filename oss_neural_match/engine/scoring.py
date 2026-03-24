"""
Open Source Network — Hybrid Scoring Engine
=========================================
Replaces the TF-IDF cosine scorer in server.py with a production-grade
hybrid pipeline:

  final_score = 0.4 * BM25 + 0.4 * dense_cosine + 0.2 * exact_match
             (or 0.6 * BM25 + 0.4 * exact_match when embeddings are absent)

Components
----------
1. BM25Scorer        — Robertson BM25 with per-field weighting
2. ExactMatchScorer  — Tag/domain/category exact and compound matching
3. SynonymExpander   — Unigram and bigram synonym expansion at query time
4. DenseScorer       — Pre-computed numpy embedding cosine similarity (optional)
5. ScoringEngine     — Public API that composes all four components
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# numpy is optional — only needed for dense scoring
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


# ===========================================================================
# CONSTANTS
# ===========================================================================

# BM25 hyper-parameters (standard Okapi BM25)
BM25_K1: float = 1.5
BM25_B: float = 0.75

# Field repetition weights used when building the document term vector.
# Higher weight = term appears more times in the virtual document, raising TF.
FIELD_WEIGHTS: Dict[str, int] = {
    "tags": 3,
    "problem_domains": 2,
    "tagline": 1,
    "use_cases_detailed": 1,
    "replaces": 1,
    "category": 1,
    "sub_category": 1,
}

# Exact match bonus constants (preserved from original server.py logic)
EXACT_TAG_DOMAIN_BONUS: float = 0.35
EXACT_HYPHEN_PARTIAL_BONUS: float = 0.15
EXACT_COMPOUND_BONUS: float = 0.50
EXACT_CATEGORY_BONUS: float = 0.10

# Hybrid weight sets
WEIGHTS_WITH_DENSE = {"bm25": 0.4, "dense": 0.4, "exact": 0.2}
WEIGHTS_NO_DENSE = {"bm25": 0.6, "dense": 0.0, "exact": 0.4}


# ===========================================================================
# HELPERS
# ===========================================================================

def _clean_token(token: str) -> str:
    """Normalise a raw word to lowercase alphanumeric (keeps + and #)."""
    return re.sub(r"[^a-z0-9+#]", "", token.lower())


def _tokenise(text: str) -> List[str]:
    """
    Lowercase, strip punctuation (except + and #), split on whitespace.
    Returns only tokens with length > 1.
    """
    cleaned = re.sub(r"[^a-z0-9+# -]", "", text.lower())
    return [t for t in cleaned.split() if len(t) > 1]


def _extract_weighted_terms(tool: Dict) -> Counter:
    """
    Build a Counter of terms from a tool record.

    Field weights are implemented by repeating tokens:
      tags * 3, problem_domains * 2, tagline * 1, use_cases_detailed * 1,
      replaces * 1, category * 1, sub_category * 1.

    Hyphenated tokens are split and their parts also added (once each).
    """
    parts: List[str] = []

    for tag in tool.get("tags", []):
        parts.extend(tag.replace("-", " ").split() * FIELD_WEIGHTS["tags"])

    for domain in tool.get("problem_domains", []):
        parts.extend(domain.replace("-", " ").split() * FIELD_WEIGHTS["problem_domains"])

    parts.extend(tool.get("tagline", "").lower().split() * FIELD_WEIGHTS["tagline"])

    for uc in tool.get("use_cases_detailed", []):
        parts.extend(uc.lower().split() * FIELD_WEIGHTS["use_cases_detailed"])

    for r in tool.get("replaces", []):
        parts.extend(r.lower().split() * FIELD_WEIGHTS["replaces"])

    parts.extend(tool.get("category", "").lower().split() * FIELD_WEIGHTS["category"])
    parts.extend(tool.get("sub_category", "").lower().split() * FIELD_WEIGHTS["sub_category"])

    terms: Counter = Counter()
    for raw in parts:
        tok = _clean_token(raw)
        if len(tok) > 1:
            terms[tok] += 1

    return terms


# ===========================================================================
# SYNONYM EXPANDER
# ===========================================================================

class SynonymExpander:
    """
    Loads synonym mappings from a JSON file and expands query tokens.

    Expected JSON format (simple flat map):
        {
            "k8s":        ["kubernetes"],
            "kubernetes": ["k8s"],
            "db":         ["database"],
            ...
        }

    Also supports bigram lookup: "machine learning" -> ["ml"]
    """

    def __init__(self, synonyms: Optional[Dict[str, List[str]]] = None) -> None:
        # Normalise all keys and values to lowercase stripped strings
        self._map: Dict[str, List[str]] = {}
        if synonyms:
            for key, values in synonyms.items():
                norm_key = key.strip().lower()
                norm_vals = [v.strip().lower() for v in values if v.strip()]
                if norm_key and norm_vals:
                    self._map[norm_key] = norm_vals

    @classmethod
    def from_file(cls, path: Path) -> "SynonymExpander":
        """Load synonyms from a JSON file. Returns empty expander on failure."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return cls(data)
        except (OSError, json.JSONDecodeError):
            pass
        return cls()

    def expand(self, tokens: List[str]) -> List[str]:
        """
        Return original tokens plus any synonyms found.
        Checks both unigrams and adjacent-pair bigrams.
        Preserves order: originals first, then expansions (deduplicated).
        """
        if not self._map:
            return tokens

        seen: set = set(tokens)
        expanded: List[str] = list(tokens)

        # Unigram expansion
        for tok in tokens:
            for syn in self._map.get(tok, []):
                if syn not in seen:
                    seen.add(syn)
                    expanded.append(syn)

        # Bigram expansion (adjacent pairs in the original token list)
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            for syn in self._map.get(bigram, []):
                if syn not in seen:
                    seen.add(syn)
                    expanded.append(syn)

        return expanded


# ===========================================================================
# BM25 SCORER
# ===========================================================================

class BM25Scorer:
    """
    Robertson BM25 scorer with per-field weighting applied via term repetition.

    BM25 score for document d given query Q:

        score(d, Q) = sum_{t in Q} IDF(t) * (tf_d(t) * (k1 + 1)) /
                      (tf_d(t) + k1 * (1 - b + b * dl_d / avgdl))

    IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
    """

    def __init__(self, tools: List[Dict], k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1 = k1
        self.b = b
        self._slug_index: Dict[str, Dict] = {t["slug"]: t for t in tools}

        # Build per-document term vectors
        self._doc_terms: Dict[str, Counter] = {}
        total_length = 0
        n = len(tools)

        for tool in tools:
            slug = tool["slug"]
            terms = _extract_weighted_terms(tool)
            self._doc_terms[slug] = terms
            total_length += sum(terms.values())

        self._avgdl: float = total_length / n if n > 0 else 1.0

        # Document frequency: how many docs contain each term
        df: Counter = Counter()
        for terms in self._doc_terms.values():
            df.update(terms.keys())

        # IDF per term (Robertson formula, smoothed to avoid negatives)
        self._idf: Dict[str, float] = {
            term: math.log((n - count + 0.5) / (count + 0.5) + 1.0)
            for term, count in df.items()
        }

    def score(self, query_tokens: List[str], slug: str) -> float:
        """
        Compute BM25 score for `slug` given a list of query tokens.
        `query_tokens` should already be normalised and synonym-expanded.
        """
        doc_terms = self._doc_terms.get(slug)
        if not doc_terms:
            return 0.0

        dl = sum(doc_terms.values())
        score = 0.0

        for token in query_tokens:
            idf = self._idf.get(token, 0.0)
            if idf == 0.0:
                continue
            tf = doc_terms.get(token, 0)
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
            score += idf * (numerator / denominator)

        return score

    @property
    def max_possible_score(self) -> float:
        """
        Approximate upper bound: sum of all IDF values.
        Used for normalisation so the BM25 component is in a comparable range.
        """
        return sum(self._idf.values()) or 1.0


# ===========================================================================
# EXACT MATCH SCORER
# ===========================================================================

class ExactMatchScorer:
    """
    Fast exact / compound / partial match bonuses against tool metadata.

    Bonuses (additive, not capped internally):
      +0.35  direct word match against tags or problem_domains
      +0.15  word appears inside a hyphenated tag, or a tag appears inside the word
      +0.50  entire query (space-joined, converted to hyphens) matches a tag
      +0.10  word appears inside category or sub_category string
    """

    def __init__(self, tools: List[Dict]) -> None:
        self._slug_index: Dict[str, Dict] = {t["slug"]: t for t in tools}

    def score(self, query: str, query_tokens: List[str], slug: str) -> float:
        """
        Return the cumulative exact-match bonus for `slug`.

        `query`        — original normalised query string (lowercased, stripped)
        `query_tokens` — tokenised, synonym-expanded query words
        """
        tool = self._slug_index.get(slug)
        if not tool:
            return 0.0

        tool_tags: set = set(tool.get("tags", []))
        tool_domains: set = set(tool.get("problem_domains", []))
        tool_all: set = tool_tags | tool_domains

        bonus = 0.0

        for word in query_tokens:
            # Direct exact hit on a tag or domain token
            if word in tool_all:
                bonus += EXACT_TAG_DOMAIN_BONUS

            # Partial containment (handles "email" matching "email-marketing")
            for tag in tool_all:
                if word in tag or tag in word:
                    bonus += EXACT_HYPHEN_PARTIAL_BONUS
                    break

        # Compound query match: "email marketing" -> check against "email-marketing"
        q_hyphenated = query.replace(" ", "-")
        for tag in tool_all:
            if q_hyphenated == tag or q_hyphenated in tag:
                bonus += EXACT_COMPOUND_BONUS
                break  # one compound bonus per query

        # Category / sub-category word match
        cat_text = (
            tool.get("category", "").lower()
            + " "
            + tool.get("sub_category", "").lower()
        )
        for word in query_tokens:
            if word in cat_text:
                bonus += EXACT_CATEGORY_BONUS

        return bonus


# ===========================================================================
# DENSE (EMBEDDING) SCORER
# ===========================================================================

class DenseScorer:
    """
    Cosine similarity against pre-computed tool embeddings.

    Expects two files:
      embeddings_dir/embeddings.npy        — shape (N, D) float32 array
      embeddings_dir/embedding_index.json  — {"slug": row_index, ...}

    If either file is missing or numpy is unavailable, this scorer silently
    disables itself (available == False).  No warnings are emitted.

    Query embeddings are NOT computed here — sentence_transformers is not
    imported.  This class is designed for scenarios where query embeddings
    are pre-computed and passed in, or for future extension.  For now it
    computes a simple term-overlap proxy when a query vector is not supplied,
    but fully activates when `query_vector` is provided to `score()`.
    """

    def __init__(self, embeddings_dir: Optional[Path] = None) -> None:
        self.available = False
        self._embeddings = None   # numpy array (N, D)
        self._index: Dict[str, int] = {}

        if not _NUMPY_AVAILABLE or embeddings_dir is None:
            return

        emb_path = embeddings_dir / "embeddings.npy"
        idx_path = embeddings_dir / "embedding_index.json"

        if not emb_path.exists() or not idx_path.exists():
            return

        try:
            self._embeddings = np.load(str(emb_path))
            with open(idx_path, "r", encoding="utf-8") as fh:
                raw_index = json.load(fh)
            # Accept both {slug: int} and {int: slug} orientations
            if raw_index and isinstance(next(iter(raw_index.values())), int):
                self._index = raw_index
            else:
                # Invert {row: slug} -> {slug: row}
                self._index = {v: int(k) for k, v in raw_index.items()}
            self.available = True
        except Exception:
            # Any failure: silently degrade
            self._embeddings = None
            self._index = {}
            self.available = False

    def score(self, slug: str, query_vector: Optional["np.ndarray"] = None) -> float:
        """
        Return cosine similarity in [0, 1].

        If `query_vector` is None or dense scoring is unavailable, returns 0.0.
        """
        if not self.available or query_vector is None:
            return 0.0

        row = self._index.get(slug)
        if row is None:
            return 0.0

        tool_vec = self._embeddings[row]
        norm_q = float(np.linalg.norm(query_vector))
        norm_t = float(np.linalg.norm(tool_vec))
        if norm_q == 0.0 or norm_t == 0.0:
            return 0.0

        cosine = float(np.dot(query_vector, tool_vec) / (norm_q * norm_t))
        # Clamp to [0, 1] — negative cosine means dissimilar, treat as 0
        return max(0.0, cosine)


# ===========================================================================
# SCORING ENGINE  (public API)
# ===========================================================================

class ScoringEngine:
    """
    Hybrid scoring engine combining BM25, exact-match bonuses, and optional
    dense embedding similarity.

    Parameters
    ----------
    tools : List[Dict]
        Full tool records loaded from db.json.
    synonyms : Optional[Dict]
        Pre-loaded synonym mapping {term: [synonym, ...]}.
        If None, the engine will attempt to load from
        ``embeddings_dir.parent / "synonyms.json"`` or
        ``<module_dir>/../data/synonyms.json``.
    embeddings_dir : Optional[Path]
        Directory containing ``embeddings.npy`` and ``embedding_index.json``.
        If None, the engine attempts ``<module_dir>/../data/``.
        Dense scoring is silently skipped when files are not found.
    """

    def __init__(
        self,
        tools: List[Dict],
        synonyms: Optional[Dict] = None,
        embeddings_dir: Optional[Path] = None,
    ) -> None:
        if not tools:
            raise ValueError("ScoringEngine requires at least one tool record.")

        self._tools = tools
        self._slug_index: Dict[str, Dict] = {t["slug"]: t for t in tools}

        # --- resolve data directory ---
        _module_dir = Path(__file__).parent
        _data_dir = _module_dir.parent / "data"

        # --- synonym expander ---
        if synonyms is not None:
            self._expander = SynonymExpander(synonyms)
        else:
            syn_path = _data_dir / "synonyms.json"
            self._expander = SynonymExpander.from_file(syn_path)

        # --- BM25 ---
        self._bm25 = BM25Scorer(tools)

        # --- exact match ---
        self._exact = ExactMatchScorer(tools)

        # --- dense (optional) ---
        if embeddings_dir is None:
            embeddings_dir = _data_dir
        self._dense = DenseScorer(embeddings_dir)

        # Choose weight set based on dense availability
        self._weights = WEIGHTS_WITH_DENSE if self._dense.available else WEIGHTS_NO_DENSE

        # Pre-compute BM25 max score for normalisation
        self._bm25_max = self._bm25.max_possible_score

    # -----------------------------------------------------------------------
    # INTERNAL: query preprocessing
    # -----------------------------------------------------------------------

    def _preprocess_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Returns (normalised_query_string, expanded_token_list).

        Steps:
          1. Lowercase, strip leading/trailing whitespace
          2. Tokenise (strip punctuation except + and #)
          3. Also split any hyphenated tokens into their parts
          4. Synonym-expand the token list
        """
        q_lower = query.lower().strip()
        base_tokens = _tokenise(q_lower)

        # Split hyphenated tokens: "email-marketing" -> ["email", "marketing"]
        dehyphenated: List[str] = []
        for tok in base_tokens:
            dehyphenated.append(tok)
            if "-" in tok:
                for part in tok.split("-"):
                    p = _clean_token(part)
                    if len(p) > 1 and p not in dehyphenated:
                        dehyphenated.append(p)

        expanded = self._expander.expand(dehyphenated)
        return q_lower, expanded

    # -----------------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------------

    def score(self, query: str, slug: str) -> float:
        """
        Compute hybrid score for a single tool identified by `slug`.

        Returns a non-negative float.  Higher is more relevant.
        """
        if slug not in self._slug_index:
            return 0.0

        q_norm, q_tokens = self._preprocess_query(query)

        # BM25 — normalised to [0, ~1] by dividing by theoretical max
        bm25_raw = self._bm25.score(q_tokens, slug)
        bm25_norm = bm25_raw / self._bm25_max

        # Exact match
        exact = self._exact.score(q_norm, q_tokens, slug)

        # Dense (0.0 when unavailable)
        dense = self._dense.score(slug)

        w = self._weights
        return w["bm25"] * bm25_norm + w["dense"] * dense + w["exact"] * exact

    def search(
        self,
        query: str,
        candidates: List[Dict],
        limit: int = 10,
    ) -> List[Tuple[Dict, float]]:
        """
        Score all candidates and return the top `limit` results sorted by
        descending score.

        Parameters
        ----------
        query : str
            Natural language or keyword query.
        candidates : List[Dict]
            Subset of tool records to score (e.g. pre-filtered by category).
            Each dict must have a ``"slug"`` key.
        limit : int
            Maximum number of results to return.

        Returns
        -------
        List[Tuple[Dict, float]]
            Sorted list of (tool_record, score) pairs, highest score first.
            Results with score == 0.0 are excluded.
        """
        q_norm, q_tokens = self._preprocess_query(query)
        scored: List[Tuple[Dict, float]] = []

        bm25_max = self._bm25_max
        w = self._weights

        for tool in candidates:
            slug = tool.get("slug", "")
            if not slug or slug not in self._slug_index:
                continue

            bm25_raw = self._bm25.score(q_tokens, slug)
            bm25_norm = bm25_raw / bm25_max
            exact = self._exact.score(q_norm, q_tokens, slug)
            dense = self._dense.score(slug)

            final = w["bm25"] * bm25_norm + w["dense"] * dense + w["exact"] * exact

            if final > 0.0:
                scored.append((tool, final))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def explain_score(self, query: str, slug: str) -> Dict:
        """
        Return a detailed breakdown of every scoring component for a tool.

        Useful for debugging relevance and tuning weights.

        Returns
        -------
        Dict with keys:
          query           — normalised query string
          slug            — tool slug
          tokens          — expanded token list
          bm25_raw        — raw BM25 score (unnormalised)
          bm25_normalised — BM25 / theoretical_max
          exact_bonus     — exact match bonus
          dense_score     — cosine similarity (0.0 if unavailable)
          dense_available — bool
          weights         — weight set applied
          final_score     — weighted combination
          top_bm25_terms  — top 10 terms by BM25 contribution
          exact_hits      — which query tokens triggered exact matches
        """
        tool = self._slug_index.get(slug)
        if not tool:
            return {
                "query": query,
                "slug": slug,
                "error": f"slug '{slug}' not found in index",
            }

        q_norm, q_tokens = self._preprocess_query(query)

        # BM25 component breakdown
        bm25_raw = self._bm25.score(q_tokens, slug)
        bm25_norm = bm25_raw / self._bm25_max

        # Per-term BM25 contributions
        doc_terms = self._bm25._doc_terms.get(slug, Counter())
        dl = sum(doc_terms.values())
        term_contributions: List[Tuple[str, float]] = []
        for token in q_tokens:
            idf = self._bm25._idf.get(token, 0.0)
            if idf == 0.0:
                continue
            tf = doc_terms.get(token, 0)
            num = tf * (self._bm25.k1 + 1.0)
            den = tf + self._bm25.k1 * (
                1.0 - self._bm25.b + self._bm25.b * dl / self._bm25._avgdl
            )
            term_contributions.append((token, idf * num / den))
        term_contributions.sort(key=lambda x: x[1], reverse=True)

        # Exact match — identify which tokens hit
        tool_tags = set(tool.get("tags", []))
        tool_domains = set(tool.get("problem_domains", []))
        tool_all = tool_tags | tool_domains
        cat_text = (
            tool.get("category", "").lower()
            + " "
            + tool.get("sub_category", "").lower()
        )

        exact_hits: List[Dict] = []
        for word in q_tokens:
            hit = {}
            if word in tool_all:
                hit["direct"] = True
            for tag in tool_all:
                if word in tag or tag in word:
                    hit["partial"] = tag
                    break
            if word in cat_text:
                hit["category"] = True
            if hit:
                hit["token"] = word
                exact_hits.append(hit)

        q_hyphenated = q_norm.replace(" ", "-")
        compound_match = any(
            q_hyphenated == tag or q_hyphenated in tag for tag in tool_all
        )
        if compound_match:
            exact_hits.append({"token": q_hyphenated, "compound": True})

        exact_bonus = self._exact.score(q_norm, q_tokens, slug)
        dense_score = self._dense.score(slug)

        w = self._weights
        final = w["bm25"] * bm25_norm + w["dense"] * dense_score + w["exact"] * exact_bonus

        return {
            "query": query,
            "slug": slug,
            "tokens": q_tokens,
            "bm25_raw": round(bm25_raw, 6),
            "bm25_normalised": round(bm25_norm, 6),
            "exact_bonus": round(exact_bonus, 6),
            "dense_score": round(dense_score, 6),
            "dense_available": self._dense.available,
            "weights": w,
            "final_score": round(final, 6),
            "top_bm25_terms": [(t, round(s, 4)) for t, s in term_contributions[:10]],
            "exact_hits": exact_hits,
        }
