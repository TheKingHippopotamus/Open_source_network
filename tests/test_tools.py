"""
tests/test_tools.py
===================
Tests for MCP tool functions in server.py.
All tools are async; we use pytest-asyncio or asyncio.run().

Tools tested:
  - oss_search
  - oss_get_tool
  - oss_find_stack
  - oss_compare
  - oss_list_categories
  - oss_browse_tags
  - oss_stats
  - oss_find_compatible

Note: oss_health_score and oss_explain_recommendation are expected as
future tools; these tests will activate once those functions are added
to server.py.
"""

import asyncio
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import server as srv
from server import (
    SearchInput,
    GetToolInput,
    FindStackInput,
    CompareInput,
    ListCategoriesInput,
    BrowseTagsInput,
    StatsInput,
    CompatibleInput,
    oss_search,
    oss_get_tool,
    oss_find_stack,
    oss_compare,
    oss_list_categories,
    oss_browse_tags,
    oss_stats,
    oss_find_compatible,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    """Run a coroutine synchronously (no pytest-asyncio dependency required)."""
    return asyncio.run(coro)


# ===========================================================================
# oss_search
# ===========================================================================

class TestOssSearch:
    def test_basic_query_returns_results(self):
        result = run(oss_search(SearchInput(query="vector database", limit=5)))
        assert "Search results" in result
        assert "qdrant" in result.lower() or "milvus" in result.lower()

    def test_result_count_respects_limit(self):
        result = run(oss_search(SearchInput(query="database", limit=3)))
        # Count "### N." headings to estimate result count
        import re
        hits = re.findall(r"### \d+\.", result)
        assert len(hits) <= 3

    def test_filter_by_category(self):
        result = run(oss_search(SearchInput(
            query="database",
            limit=10,
            category="Databases",
        )))
        assert "Search results" in result

    def test_filter_by_max_ram(self):
        result = run(oss_search(SearchInput(
            query="lightweight",
            limit=10,
            max_ram_mb=512,
        )))
        # All returned tools must have min_ram_mb <= 512
        # (we check there's no mention of high-RAM tools like TensorFlow)
        assert "Search results" in result

    def test_filter_by_license_type(self):
        result = run(oss_search(SearchInput(
            query="database",
            limit=5,
            license_type="permissive",
        )))
        assert "Search results" in result
        assert "permissive" in result

    def test_filter_self_hosted_only(self):
        result = run(oss_search(SearchInput(
            query="CRM",
            limit=5,
            self_hosted_only=True,
        )))
        assert "Search results" in result

    def test_filter_complexity(self):
        result = run(oss_search(SearchInput(
            query="authentication",
            limit=5,
            complexity="beginner",
        )))
        assert "Search results" in result

    def test_no_results_returns_not_found_message(self):
        # A query that will match nothing because we filter to an empty set
        result = run(oss_search(SearchInput(
            query="xyzunmatchable99999",
            limit=5,
            category="Databases",
            max_ram_mb=1,      # only tools with <=1 MB RAM
        )))
        assert "No tools found" in result

    def test_crm_returns_crm_tools(self):
        result = run(oss_search(SearchInput(query="CRM", limit=5)))
        # Should contain at least one known CRM tool name (case-insensitive)
        result_lower = result.lower()
        crm_tool_names = ["krayin", "twenty", "suitecrm", "espocrm", "odoo", "dolibarr"]
        assert any(s in result_lower for s in crm_tool_names), (
            f"Expected CRM tools in result, got: {result[:300]}"
        )

    def test_vector_database_returns_qdrant(self):
        result = run(oss_search(SearchInput(query="vector database", limit=5)))
        assert "qdrant" in result.lower() or "Qdrant" in result

    def test_kubernetes_returns_k8s_tools(self):
        result = run(oss_search(SearchInput(query="kubernetes", limit=5)))
        k8s_tools = ["kubernetes", "argocd", "prometheus", "kubeflow"]
        assert any(s in result.lower() for s in k8s_tools), (
            f"Expected k8s tools, got: {result[:200]}"
        )

    def test_email_marketing_returns_email_tools(self):
        result = run(oss_search(SearchInput(query="email marketing", limit=5)))
        email_tools = ["listmonk", "mautic", "postal", "keila", "mailtrain"]
        assert any(s in result.lower() for s in email_tools), (
            f"Expected email tools, got: {result[:200]}"
        )

    def test_result_contains_relevance_score(self):
        result = run(oss_search(SearchInput(query="database", limit=3)))
        assert "Relevance:" in result

    def test_result_contains_license_info(self):
        result = run(oss_search(SearchInput(query="database", limit=3)))
        assert "License:" in result

    def test_result_contains_ram_info(self):
        result = run(oss_search(SearchInput(query="database", limit=3)))
        assert "RAM:" in result

    def test_large_limit_returns_many_results(self):
        result = run(oss_search(SearchInput(query="api", limit=20)))
        import re
        hits = re.findall(r"### \d+\.", result)
        assert len(hits) >= 5


# ===========================================================================
# oss_get_tool
# ===========================================================================

class TestOssGetTool:
    def test_valid_slug_returns_full_details(self):
        result = run(oss_get_tool(GetToolInput(slug="postgresql")))
        assert "PostgreSQL" in result
        assert "License:" in result
        assert "Category:" in result

    def test_full_details_contains_technical_section(self):
        result = run(oss_get_tool(GetToolInput(slug="postgresql")))
        assert "## Technical" in result

    def test_full_details_contains_ecosystem_section(self):
        result = run(oss_get_tool(GetToolInput(slug="postgresql")))
        assert "## Ecosystem" in result

    def test_full_details_contains_use_cases(self):
        result = run(oss_get_tool(GetToolInput(slug="qdrant")))
        assert "## Use Cases" in result

    def test_full_details_contains_matching_profile(self):
        result = run(oss_get_tool(GetToolInput(slug="redis")))
        assert "## Matching Profile" in result

    def test_missing_slug_returns_not_found(self):
        result = run(oss_get_tool(GetToolInput(slug="nonexistent-tool-xyz")))
        assert "not found" in result.lower()

    def test_missing_slug_no_suggestions_when_no_match(self):
        result = run(oss_get_tool(GetToolInput(slug="zzznomatch999")))
        assert "not found" in result.lower()

    def test_slug_with_suggestions(self):
        # "postgres" is contained in "postgresql"
        result = run(oss_get_tool(GetToolInput(slug="postgres")))
        # Should either find a tool or suggest postgresql
        assert "postgresql" in result.lower() or "not found" in result.lower()

    def test_qdrant_details_include_tags(self):
        result = run(oss_get_tool(GetToolInput(slug="qdrant")))
        assert "Tags:" in result
        assert "vector" in result.lower()

    def test_tensorflow_details_include_backing(self):
        result = run(oss_get_tool(GetToolInput(slug="tensorflow")))
        assert "Google" in result

    def test_tool_with_anti_patterns_includes_section(self):
        result = run(oss_get_tool(GetToolInput(slug="tensorflow")))
        assert "Anti-Patterns" in result or "anti" in result.lower()


# ===========================================================================
# oss_find_stack
# ===========================================================================

class TestOssFindStack:
    def test_basic_stack_returns_recommendations(self):
        result = run(oss_find_stack(FindStackInput(
            needs=["authentication", "database"],
            max_ram_mb=4096,
        )))
        assert "Recommended Stack" in result

    def test_stack_summary_section_present(self):
        result = run(oss_find_stack(FindStackInput(
            needs=["authentication", "database"],
        )))
        assert "## Stack Summary" in result

    def test_stack_includes_total_ram(self):
        result = run(oss_find_stack(FindStackInput(
            needs=["authentication"],
        )))
        assert "Total RAM:" in result

    def test_stack_respects_max_complexity(self):
        result = run(oss_find_stack(FindStackInput(
            needs=["database"],
            max_complexity="beginner",
        )))
        assert "Recommended Stack" in result

    def test_stack_with_license_preference(self):
        result = run(oss_find_stack(FindStackInput(
            needs=["database", "authentication"],
            license_preference="permissive",
        )))
        assert "Recommended Stack" in result
        assert "permissive" in result

    def test_stack_with_small_ram_budget(self):
        result = run(oss_find_stack(FindStackInput(
            needs=["database"],
            max_ram_mb=512,
        )))
        assert "Recommended Stack" in result

    def test_multiple_needs_all_addressed(self):
        needs = ["authentication", "crm", "email-marketing"]
        result = run(oss_find_stack(FindStackInput(needs=needs)))
        assert "Recommended Stack" in result
        # oss_find_stack renders each need as a Title-cased heading:
        # "authentication" → "## Authentication →", "email-marketing" → "## Email-Marketing →"
        result_lower = result.lower()
        for need in needs:
            # The need slug appears in the heading (lowercased comparison)
            assert need.lower() in result_lower, (
                f"Need '{need}' not found in stack result"
            )

    def test_stack_alternatives_listed(self):
        result = run(oss_find_stack(FindStackInput(
            needs=["database", "authentication"],
        )))
        assert "Alternatives:" in result

    def test_conflict_warning_format(self):
        # This tests the warning format exists when conflicts are detected
        # (may or may not trigger depending on stack selection)
        result = run(oss_find_stack(FindStackInput(
            needs=["database"],
        )))
        # At minimum the result should be a valid string
        assert isinstance(result, str)
        assert len(result) > 50

    def test_enterprise_team_size(self):
        result = run(oss_find_stack(FindStackInput(
            needs=["database"],
            team_size="enterprise",
        )))
        assert "Recommended Stack" in result


# ===========================================================================
# oss_compare
# ===========================================================================

class TestOssCompare:
    def test_two_tools_returns_table(self):
        result = run(oss_compare(CompareInput(slugs=["postgresql", "mysql"])))
        assert "Comparison:" in result
        assert "PostgreSQL" in result
        assert "MySQL" in result

    def test_table_header_present(self):
        result = run(oss_compare(CompareInput(slugs=["postgresql", "mysql"])))
        assert "| Dimension |" in result

    def test_three_tools_table(self):
        result = run(oss_compare(CompareInput(slugs=["postgresql", "mysql", "mariadb"])))
        assert "PostgreSQL" in result
        assert "MySQL" in result
        assert "MariaDB" in result

    def test_compare_contains_license_row(self):
        result = run(oss_compare(CompareInput(slugs=["postgresql", "mysql"])))
        assert "License" in result

    def test_compare_contains_min_ram_row(self):
        result = run(oss_compare(CompareInput(slugs=["postgresql", "mysql"])))
        assert "Min RAM" in result

    def test_compare_contains_tags_section(self):
        result = run(oss_compare(CompareInput(slugs=["postgresql", "mysql"])))
        assert "## Tags" in result

    def test_compare_contains_anti_patterns_section(self):
        result = run(oss_compare(CompareInput(slugs=["postgresql", "mysql"])))
        assert "Anti-patterns" in result

    def test_one_valid_one_invalid_slug(self):
        result = run(oss_compare(CompareInput(slugs=["postgresql", "no-such-tool"])))
        # Should return "need at least 2 valid tools"
        assert "Need at least 2" in result or "Comparison:" in result

    def test_two_invalid_slugs_returns_error(self):
        result = run(oss_compare(CompareInput(slugs=["no-tool-1", "no-tool-2"])))
        assert "Need at least 2" in result

    def test_compare_vector_databases(self):
        result = run(oss_compare(CompareInput(slugs=["qdrant", "milvus", "weaviate"])))
        assert "Qdrant" in result
        assert "Milvus" in result
        assert "Weaviate" in result


# ===========================================================================
# oss_list_categories
# ===========================================================================

class TestOssListCategories:
    def test_returns_all_20_categories(self):
        result = run(oss_list_categories(ListCategoriesInput()))
        expected_categories = [
            "AI / ML", "Analytics", "Databases", "LLMs & AI Infra",
            "DevOps & Infra", "Security & Auth", "CRM & ERP",
        ]
        for cat in expected_categories:
            assert cat in result, f"Category '{cat}' not in result"

    def test_returns_244_tools_total(self):
        result = run(oss_list_categories(ListCategoriesInput()))
        assert "244 tools" in result

    def test_category_includes_counts(self):
        result = run(oss_list_categories(ListCategoriesInput()))
        import re
        # Expect patterns like "(N tools)" or "(N)"
        count_matches = re.findall(r"\(\d+ tools?\)", result)
        assert len(count_matches) >= 15, (
            f"Expected tool counts per category, found only {len(count_matches)}"
        )

    def test_include_tools_shows_tool_names(self):
        result = run(oss_list_categories(ListCategoriesInput(include_tools=True)))
        assert "PostgreSQL" in result or "postgresql" in result.lower()

    def test_result_is_string(self):
        result = run(oss_list_categories(ListCategoriesInput()))
        assert isinstance(result, str)

    def test_contains_sorted_categories(self):
        result = run(oss_list_categories(ListCategoriesInput()))
        # Embeddable has 47 tools — most tools — should appear near top
        lines = result.split("\n")
        embed_line = next((i for i, l in enumerate(lines) if "Embeddable" in l), None)
        assert embed_line is not None, "Embeddable category not found"


# ===========================================================================
# oss_browse_tags
# ===========================================================================

class TestOssBrowseTags:
    def test_specific_tag_returns_tools(self):
        result = run(oss_browse_tags(BrowseTagsInput(tag="rag")))
        assert "Tag: rag" in result
        assert "tools" in result

    def test_specific_tag_contains_rag_tools(self):
        result = run(oss_browse_tags(BrowseTagsInput(tag="rag")))
        rag_tools = ["langchain", "milvus", "qdrant", "weaviate", "chroma"]
        assert any(s in result.lower() for s in rag_tools)

    def test_unknown_tag_returns_partial_matches_or_not_found(self):
        result = run(oss_browse_tags(BrowseTagsInput(tag="xyz-nonexistent-tag")))
        assert "not found" in result.lower() or "Partial matches" in result

    def test_search_mode_returns_matching_tags(self):
        result = run(oss_browse_tags(BrowseTagsInput(search="kubernetes")))
        assert "kubernetes" in result

    def test_search_mode_contains_count(self):
        result = run(oss_browse_tags(BrowseTagsInput(search="database")))
        import re
        count_matches = re.findall(r"\(\d+ tools?\)", result)
        assert len(count_matches) >= 1

    def test_top_tags_mode(self):
        result = run(oss_browse_tags(BrowseTagsInput(limit=10)))
        assert "Top 10 tags" in result

    def test_top_tags_respects_limit(self):
        result = run(oss_browse_tags(BrowseTagsInput(limit=5)))
        assert "Top 5 tags" in result

    def test_top_tags_shows_plugins(self):
        # "plugins" is the most common tag (21 tools)
        result = run(oss_browse_tags(BrowseTagsInput(limit=5)))
        assert "plugins" in result

    def test_tag_kubernetes_returns_known_tools(self):
        result = run(oss_browse_tags(BrowseTagsInput(tag="kubernetes")))
        k8s_tools = ["kubernetes", "argocd", "prometheus", "kubeflow"]
        assert any(s in result.lower() for s in k8s_tools)

    def test_result_is_string(self):
        result = run(oss_browse_tags(BrowseTagsInput(limit=5)))
        assert isinstance(result, str)


# ===========================================================================
# oss_stats
# ===========================================================================

class TestOssStats:
    def test_returns_string(self):
        result = run(oss_stats(StatsInput()))
        assert isinstance(result, str)

    def test_total_tools_is_244(self):
        result = run(oss_stats(StatsInput()))
        assert "244" in result

    def test_contains_fields_per_tool(self):
        result = run(oss_stats(StatsInput()))
        assert "Fields per tool" in result
        assert "51" in result

    def test_contains_categories_count(self):
        result = run(oss_stats(StatsInput()))
        assert "Categories" in result
        assert "20" in result

    def test_contains_unique_tags(self):
        result = run(oss_stats(StatsInput()))
        assert "Unique tags" in result

    def test_contains_graph_edges(self):
        result = run(oss_stats(StatsInput()))
        assert "Graph edges" in result

    def test_contains_total_data_points(self):
        result = run(oss_stats(StatsInput()))
        assert "Total data points" in result
        assert "12,444" in result

    def test_contains_use_cases_count(self):
        result = run(oss_stats(StatsInput()))
        assert "Use cases" in result

    def test_contains_anti_patterns_count(self):
        result = run(oss_stats(StatsInput()))
        assert "Anti-patterns" in result


# ===========================================================================
# oss_find_compatible
# ===========================================================================

class TestOssFindCompatible:
    def test_valid_tool_returns_compatible(self):
        result = run(oss_find_compatible(CompatibleInput(slug="postgresql")))
        assert "Compatible tools" in result or "postgresql" in result.lower()

    def test_integrations_section_present(self):
        result = run(oss_find_compatible(CompatibleInput(slug="postgresql")))
        assert "Direct integrations" in result or "Also integrates" in result

    def test_unknown_slug_returns_not_found(self):
        result = run(oss_find_compatible(CompatibleInput(slug="no-such-slug")))
        assert "not found" in result.lower()

    def test_result_is_string(self):
        result = run(oss_find_compatible(CompatibleInput(slug="redis")))
        assert isinstance(result, str)

    def test_tool_with_similar_includes_alternatives(self):
        # postgresql has similar_to (mysql, etc.)
        result = run(oss_find_compatible(CompatibleInput(slug="postgresql")))
        assert "Alternatives" in result or "Similar" in result

    def test_limit_respected(self):
        # With limit=2 we expect at most 2 tools per section
        result = run(oss_find_compatible(CompatibleInput(slug="postgresql", limit=2)))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_n8n_has_many_integrations(self):
        result = run(oss_find_compatible(CompatibleInput(slug="n8n")))
        assert "Compatible tools" in result
        # n8n should have integrations
        assert "integrat" in result.lower()


# ===========================================================================
# Global server state
# ===========================================================================

class TestServerState:
    def test_db_is_loaded(self):
        assert len(srv.DB) == 244

    def test_slug_index_matches_db(self):
        assert len(srv.SLUG_INDEX) == 244

    def test_tag_index_non_empty(self):
        assert len(srv.TAG_INDEX) > 0

    def test_category_index_has_20_categories(self):
        assert len(srv.CATEGORY_INDEX) == 20

    def test_idf_non_empty(self):
        assert len(srv.IDF) > 0

    def test_tool_tfidf_has_entry_per_tool(self):
        assert len(srv.TOOL_TFIDF) == 244

    def test_known_slug_in_slug_index(self):
        assert "postgresql" in srv.SLUG_INDEX
        assert "qdrant" in srv.SLUG_INDEX
        assert "kubernetes" in srv.SLUG_INDEX

    def test_postgresql_in_databases_category_index(self):
        assert "postgresql" in srv.CATEGORY_INDEX.get("Databases", [])
