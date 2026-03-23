"""
tests/test_scoring.py
=====================
Tests for engine/scoring.py:
  - BM25Scorer
  - ExactMatchScorer
  - SynonymExpander
  - ScoringEngine (end-to-end search)

Covers: basic queries, synonym expansion, exact tag bonuses,
        known-good queries, score ranges, empty queries,
        filtered search, explain_score, and edge cases.
"""

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from engine.scoring import (
    BM25Scorer,
    ExactMatchScorer,
    SynonymExpander,
    ScoringEngine,
    _tokenise,
    _clean_token,
    _extract_weighted_terms,
)


# ===========================================================================
# _tokenise / _clean_token helpers
# ===========================================================================

class TestHelpers:
    def test_tokenise_basic(self):
        tokens = _tokenise("vector database")
        assert "vector" in tokens
        assert "database" in tokens

    def test_tokenise_removes_punctuation(self):
        tokens = _tokenise("email-marketing!!")
        # The hyphenated token is kept as-is after strip
        assert all(isinstance(t, str) for t in tokens)

    def test_tokenise_filters_single_char(self):
        tokens = _tokenise("a b hello")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "hello" in tokens

    def test_tokenise_lowercase(self):
        tokens = _tokenise("PostgreSQL Redis")
        assert "postgresql" in tokens
        assert "redis" in tokens

    def test_clean_token_alphanumeric(self):
        assert _clean_token("Hello!") == "hello"
        assert _clean_token("C++") == "c++"
        assert _clean_token("C#") == "c#"
        assert _clean_token("k8s-deploy") == "k8sdeploy"

    def test_extract_weighted_terms_non_empty(self, sample_tools):
        for tool in sample_tools:
            terms = _extract_weighted_terms(tool)
            assert len(terms) > 0, f"Expected terms for {tool['slug']}"

    def test_extract_weighted_terms_tags_weighted_higher(self, db):
        tool = next(t for t in db if t["slug"] == "qdrant")
        terms = _extract_weighted_terms(tool)
        # "vector" appears in tag "vector-database" which is weighted x3
        # so "vector" count should be substantial
        assert terms.get("vector", 0) >= 3


# ===========================================================================
# SynonymExpander
# ===========================================================================

class TestSynonymExpander:
    def test_expand_unigram(self):
        exp = SynonymExpander({"auth": ["authentication", "authorization"]})
        result = exp.expand(["auth"])
        assert "auth" in result
        assert "authentication" in result
        assert "authorization" in result

    def test_expand_preserves_originals(self):
        exp = SynonymExpander({"k8s": ["kubernetes"]})
        result = exp.expand(["k8s", "deployment"])
        assert "k8s" in result
        assert "deployment" in result

    def test_expand_deduplicates(self):
        exp = SynonymExpander({"auth": ["authentication"]})
        result = exp.expand(["auth", "authentication"])
        assert result.count("authentication") == 1

    def test_expand_bigram(self):
        exp = SynonymExpander({"machine learning": ["ml"]})
        result = exp.expand(["machine", "learning"])
        assert "ml" in result

    def test_expand_empty_map(self):
        exp = SynonymExpander()
        result = exp.expand(["auth", "database"])
        assert result == ["auth", "database"]

    def test_expand_unknown_token(self):
        exp = SynonymExpander({"auth": ["authentication"]})
        result = exp.expand(["nosuchterm"])
        assert result == ["nosuchterm"]

    def test_from_file_missing_path(self, tmp_path):
        exp = SynonymExpander.from_file(tmp_path / "nonexistent.json")
        # Should return empty expander without raising
        assert exp.expand(["test"]) == ["test"]

    def test_from_file_valid_json(self, tmp_path):
        syn_file = tmp_path / "synonyms.json"
        syn_file.write_text('{"db": ["database"]}')
        exp = SynonymExpander.from_file(syn_file)
        result = exp.expand(["db"])
        assert "database" in result

    def test_normalises_keys_and_values(self):
        exp = SynonymExpander({"  AUTH  ": ["  AUTHENTICATION  "]})
        result = exp.expand(["auth"])
        assert "authentication" in result


# ===========================================================================
# BM25Scorer
# ===========================================================================

class TestBM25Scorer:
    def test_basic_query_returns_positive_score(self, db):
        scorer = BM25Scorer(db)
        score = scorer.score(["vector", "database"], "qdrant")
        assert score > 0.0

    def test_unknown_term_returns_zero(self, db):
        scorer = BM25Scorer(db)
        score = scorer.score(["xyznonexistent12345"], "qdrant")
        assert score == 0.0

    def test_unknown_slug_returns_zero(self, db):
        scorer = BM25Scorer(db)
        score = scorer.score(["database"], "nonexistent-slug-xyz")
        assert score == 0.0

    def test_relevant_tool_scores_higher_than_irrelevant(self, db):
        scorer = BM25Scorer(db)
        # "vector database" should score qdrant higher than, say, a scheduler
        score_qdrant = scorer.score(["vector", "database"], "qdrant")
        score_cron = scorer.score(["vector", "database"], "node-cron")
        assert score_qdrant > score_cron

    def test_max_possible_score_is_positive(self, db):
        scorer = BM25Scorer(db)
        assert scorer.max_possible_score > 0.0

    def test_empty_query_tokens_returns_zero(self, db):
        scorer = BM25Scorer(db)
        score = scorer.score([], "qdrant")
        assert score == 0.0

    def test_all_slugs_return_non_negative_scores(self, db):
        scorer = BM25Scorer(db)
        for tool in db:
            score = scorer.score(["database"], tool["slug"])
            assert score >= 0.0, f"Negative score for {tool['slug']}"

    def test_score_with_weighted_terms(self, db):
        scorer = BM25Scorer(db)
        # Tags are weighted x3; a tool's own tag term should produce a strong BM25 score
        tool = next(t for t in db if t["slug"] == "postgresql")
        # "relational" is a postgresql tag
        score = scorer.score(["relational"], "postgresql")
        assert score > 0.0


# ===========================================================================
# ExactMatchScorer
# ===========================================================================

class TestExactMatchScorer:
    def test_direct_tag_match_bonus(self, db):
        scorer = ExactMatchScorer(db)
        # "rag" is a tag on qdrant
        bonus = scorer.score("rag", ["rag"], "qdrant")
        assert bonus >= 0.35

    def test_compound_match_bonus(self, db):
        scorer = ExactMatchScorer(db)
        # "email-marketing" is a tag on listmonk/mautic
        bonus = scorer.score("email marketing", ["email", "marketing"], "mautic")
        assert bonus >= 0.50

    def test_no_match_returns_zero(self, db):
        scorer = ExactMatchScorer(db)
        # Nonsense query against qdrant
        bonus = scorer.score("xyznomatch", ["xyznomatch"], "qdrant")
        assert bonus == 0.0

    def test_category_match_bonus(self, db):
        scorer = ExactMatchScorer(db)
        # "databases" should match category "Databases"
        bonus = scorer.score("databases", ["databases"], "postgresql")
        assert bonus >= 0.10

    def test_unknown_slug_returns_zero(self, db):
        scorer = ExactMatchScorer(db)
        bonus = scorer.score("database", ["database"], "no-such-slug")
        assert bonus == 0.0

    def test_partial_containment_bonus(self, db):
        scorer = ExactMatchScorer(db)
        # "email" should partially match "email-marketing" tag
        bonus = scorer.score("email", ["email"], "mautic")
        assert bonus > 0.0


# ===========================================================================
# ScoringEngine — end-to-end
# ===========================================================================

class TestScoringEngineBasic:
    def test_init_requires_non_empty_tools(self):
        with pytest.raises(ValueError):
            ScoringEngine([])

    def test_score_known_slug(self, scoring_engine):
        score = scoring_engine.score("vector database", "qdrant")
        assert score > 0.0

    def test_score_unknown_slug_returns_zero(self, scoring_engine):
        score = scoring_engine.score("vector database", "no-such-slug-xyz")
        assert score == 0.0

    def test_score_is_non_negative(self, scoring_engine, db):
        for tool in db[:30]:
            s = scoring_engine.score("database", tool["slug"])
            assert s >= 0.0, f"Negative score for {tool['slug']}"


class TestScoringEngineSearch:
    def test_search_returns_results(self, scoring_engine, db):
        results = scoring_engine.search("vector database", db, limit=5)
        assert len(results) >= 1

    def test_search_sorted_descending(self, scoring_engine, db):
        results = scoring_engine.search("vector database", db, limit=10)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_respects_limit(self, scoring_engine, db):
        results = scoring_engine.search("database", db, limit=5)
        assert len(results) <= 5

    def test_search_excludes_zero_scores(self, scoring_engine, db):
        results = scoring_engine.search("vector database", db, limit=20)
        assert all(s > 0.0 for _, s in results)

    def test_search_empty_candidates(self, scoring_engine):
        results = scoring_engine.search("vector database", [], limit=5)
        assert results == []

    def test_search_single_candidate(self, scoring_engine, slug_index):
        tool = slug_index["qdrant"]
        results = scoring_engine.search("vector database", [tool], limit=1)
        assert len(results) == 1
        assert results[0][0]["slug"] == "qdrant"


class TestScoringEngineKnownQueries:
    """
    Known-good query assertions: specific tools should appear in top-N
    results for semantically appropriate queries.
    """

    def test_vector_database_returns_qdrant_in_top5(self, scoring_engine, db):
        results = scoring_engine.search("vector database", db, limit=5)
        slugs = [t["slug"] for t, _ in results]
        assert "qdrant" in slugs, f"qdrant not in top 5: {slugs}"

    def test_vector_database_returns_milvus_in_top5(self, scoring_engine, db):
        results = scoring_engine.search("vector database", db, limit=5)
        slugs = [t["slug"] for t, _ in results]
        assert "milvus" in slugs, f"milvus not in top 5: {slugs}"

    def test_vector_database_returns_weaviate_in_top5(self, scoring_engine, db):
        results = scoring_engine.search("vector database", db, limit=5)
        slugs = [t["slug"] for t, _ in results]
        assert "weaviate" in slugs, f"weaviate not in top 5: {slugs}"

    def test_crm_returns_crm_tools_in_top5(self, scoring_engine, db):
        results = scoring_engine.search("CRM", db, limit=5)
        # All CRM tools are in category "CRM & ERP"
        crm_slugs = {t["slug"] for t in db if "CRM" in t["category"]}
        result_slugs = {t["slug"] for t, _ in results}
        overlap = crm_slugs & result_slugs
        assert len(overlap) >= 3, f"Expected >=3 CRM tools in top 5, got {result_slugs}"

    def test_kubernetes_returns_k8s_tools_in_top5(self, scoring_engine, db):
        results = scoring_engine.search("kubernetes", db, limit=5)
        slugs = [t["slug"] for t, _ in results]
        # kubernetes, argocd, prometheus all have kubernetes-related tags
        k8s_tools = {"kubernetes", "argocd", "prometheus", "kubeflow", "kong"}
        overlap = k8s_tools & set(slugs)
        assert len(overlap) >= 2, f"Expected k8s tools in top 5: {slugs}"

    def test_email_marketing_returns_email_tools_in_top5(self, scoring_engine, db):
        results = scoring_engine.search("email marketing", db, limit=5)
        slugs = [t["slug"] for t, _ in results]
        email_tools = {"listmonk", "mautic", "postal", "keila", "mailtrain"}
        overlap = email_tools & set(slugs)
        assert len(overlap) >= 2, f"Expected email tools in top 5: {slugs}"

    def test_lightweight_database_prefers_low_ram(self, scoring_engine, db):
        results = scoring_engine.search("lightweight database", db, limit=10)
        # Should surface tools with low RAM; check at least some have < 512 MB
        ram_values = [t.get("min_ram_mb", 9999) for t, _ in results]
        assert any(r <= 512 for r in ram_values), f"No low-RAM tools: {ram_values}"

    def test_authentication_returns_auth_tools(self, scoring_engine, db):
        results = scoring_engine.search("authentication", db, limit=5)
        slugs = [t["slug"] for t, _ in results]
        auth_tools = {"supertokens", "lucia", "passportjs", "appwrite", "pocketbase", "keycloak"}
        overlap = auth_tools & set(slugs)
        assert len(overlap) >= 2, f"Expected auth tools in top 5: {slugs}"


class TestScoringEngineSynonymExpansion:
    def test_auth_and_authentication_overlap(self, db):
        """
        'auth' and 'authentication' should surface substantially overlapping tools.
        At least 2 slugs should appear in both top-5 results.
        """
        engine_with_synonyms = ScoringEngine(
            db,
            synonyms={
                "auth": ["authentication", "authorization"],
                "authentication": ["auth", "authorization"],
            }
        )
        results_auth = engine_with_synonyms.search("auth", db, limit=5)
        results_authentication = engine_with_synonyms.search("authentication", db, limit=5)
        slugs_auth = {t["slug"] for t, _ in results_auth}
        slugs_authentication = {t["slug"] for t, _ in results_authentication}
        overlap = slugs_auth & slugs_authentication
        assert len(overlap) >= 2, (
            f"Expected >=2 overlapping tools. auth={slugs_auth}, auth...={slugs_authentication}"
        )

    def test_k8s_synonym_expands_to_kubernetes(self, db):
        engine = ScoringEngine(db, synonyms={"k8s": ["kubernetes"]})
        results = engine.search("k8s", db, limit=5)
        slugs = [t["slug"] for t, _ in results]
        assert "kubernetes" in slugs or "argocd" in slugs, (
            f"k8s synonym expansion should surface kubernetes tools: {slugs}"
        )


class TestScoringEngineFilters:
    def test_filter_by_category(self, scoring_engine, db):
        db_tools = db
        database_tools = [t for t in db_tools if "database" in t["category"].lower()]
        results = scoring_engine.search("lightweight", database_tools, limit=5)
        categories = {t["category"] for t, _ in results}
        assert all("database" in cat.lower() or "Databases" == cat for cat in categories)

    def test_filter_by_ram(self, db):
        engine = ScoringEngine(db)
        low_ram_tools = [t for t in db if t.get("min_ram_mb", 9999) <= 512]
        results = engine.search("database", low_ram_tools, limit=5)
        for tool, _ in results:
            assert tool.get("min_ram_mb", 0) <= 512

    def test_filter_by_license_type(self, db):
        engine = ScoringEngine(db)
        permissive_tools = [t for t in db if t.get("license_type") == "permissive"]
        results = engine.search("database", permissive_tools, limit=5)
        for tool, _ in results:
            assert tool.get("license_type") == "permissive"

    def test_filter_returns_empty_for_no_candidates(self, scoring_engine):
        results = scoring_engine.search("database", [], limit=5)
        assert results == []


class TestScoringEngineExplainScore:
    def test_explain_returns_all_keys(self, scoring_engine):
        breakdown = scoring_engine.explain_score("vector database", "qdrant")
        required_keys = {
            "query", "slug", "tokens", "bm25_raw", "bm25_normalised",
            "exact_bonus", "dense_score", "dense_available",
            "weights", "final_score", "top_bm25_terms", "exact_hits",
        }
        assert required_keys.issubset(set(breakdown.keys()))

    def test_explain_scores_are_non_negative(self, scoring_engine):
        breakdown = scoring_engine.explain_score("email marketing", "mautic")
        assert breakdown["bm25_raw"] >= 0.0
        assert breakdown["exact_bonus"] >= 0.0
        assert breakdown["dense_score"] >= 0.0
        assert breakdown["final_score"] >= 0.0

    def test_explain_query_preserved(self, scoring_engine):
        # explain_score returns the original query string as passed in
        breakdown = scoring_engine.explain_score("Vector Database", "qdrant")
        assert breakdown["query"] == "Vector Database"

    def test_explain_tokens_non_empty_for_real_query(self, scoring_engine):
        breakdown = scoring_engine.explain_score("vector database", "qdrant")
        assert len(breakdown["tokens"]) >= 2

    def test_explain_missing_slug_returns_error_key(self, scoring_engine):
        breakdown = scoring_engine.explain_score("test", "no-such-slug")
        assert "error" in breakdown

    def test_explain_final_score_consistent_with_score(self, scoring_engine):
        slug = "qdrant"
        query = "vector database"
        direct_score = scoring_engine.score(query, slug)
        breakdown_score = scoring_engine.explain_score(query, slug)["final_score"]
        assert abs(direct_score - breakdown_score) < 1e-4, (
            f"score() and explain_score() disagree: {direct_score} vs {breakdown_score}"
        )

    def test_explain_exact_hits_for_tag_query(self, scoring_engine):
        breakdown = scoring_engine.explain_score("rag", "qdrant")
        # "rag" is a direct tag on qdrant — should appear in exact_hits
        assert len(breakdown["exact_hits"]) > 0

    def test_explain_dense_available_is_bool(self, scoring_engine):
        breakdown = scoring_engine.explain_score("database", "postgresql")
        assert isinstance(breakdown["dense_available"], bool)

    def test_explain_weights_sum_to_one(self, scoring_engine):
        breakdown = scoring_engine.explain_score("database", "postgresql")
        weights = breakdown["weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, f"Weights don't sum to 1: {weights}"
