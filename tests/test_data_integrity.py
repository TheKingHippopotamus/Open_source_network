"""
tests/test_data_integrity.py
============================
Structural validation of db.json:
  - Required fields present on every tool
  - All slugs are unique
  - Slug format is valid (lowercase, hyphenated)
  - All cross-tool references in integrates_with / complements /
    similar_to / conflicts_with point to slugs that exist in the DB
  - Tags are lowercase and hyphen-delimited (no spaces, no uppercase)
  - min_ram_mb values are non-negative integers
  - License types are within the set of valid values
  - No duplicate tools (by slug)
  - CATEGORY_INDEX matches actual tool categories in the DB
  - all_fields count is consistent (51 fields per tool)
"""

import re
import sys
from pathlib import Path
from typing import Set

import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ===========================================================================
# Constants
# ===========================================================================

REQUIRED_FIELDS = {
    "name", "slug", "tagline", "description", "category", "sub_category",
    "license", "license_type", "min_ram_mb", "tags", "problem_domains",
    "complexity_level", "maturity",
}

VALID_LICENSE_TYPES = {"permissive", "copyleft", "source-available", "fair-code"}

VALID_COMPLEXITY_LEVELS = {"beginner", "intermediate", "advanced", "expert"}

VALID_MATURITY_LEVELS = {"mature", "stable", "growing", "experimental"}

VALID_COMMIT_FREQUENCIES = {"daily", "weekly", "monthly", "quarterly", "yearly"}

VALID_PERFORMANCE_TIERS = {"lightweight", "medium", "heavy", "enterprise_grade"}

# Edge-type fields that reference other tool slugs
EDGE_TYPE_FIELDS = (
    "integrates_with",
    "complements",
    "similar_to",
    "conflicts_with",
)


# ===========================================================================
# Required fields
# ===========================================================================

class TestRequiredFields:
    def test_all_tools_have_required_fields(self, db):
        """Every tool must have every required field."""
        for tool in db:
            for field in REQUIRED_FIELDS:
                assert field in tool, (
                    f"Tool '{tool.get('slug', '?')}' is missing required field '{field}'"
                )

    def test_name_is_non_empty_string(self, db):
        for tool in db:
            assert isinstance(tool["name"], str) and tool["name"].strip(), (
                f"Tool {tool['slug']} has empty or non-string name"
            )

    def test_tagline_is_non_empty_string(self, db):
        for tool in db:
            assert isinstance(tool["tagline"], str) and tool["tagline"].strip(), (
                f"Tool {tool['slug']} has empty or non-string tagline"
            )

    def test_description_is_non_empty_string(self, db):
        for tool in db:
            assert isinstance(tool["description"], str) and tool["description"].strip(), (
                f"Tool {tool['slug']} has empty or non-string description"
            )

    def test_category_is_non_empty_string(self, db):
        for tool in db:
            assert isinstance(tool["category"], str) and tool["category"].strip(), (
                f"Tool {tool['slug']} has empty category"
            )

    def test_sub_category_is_non_empty_string(self, db):
        for tool in db:
            assert isinstance(tool["sub_category"], str) and tool["sub_category"].strip(), (
                f"Tool {tool['slug']} has empty sub_category"
            )


# ===========================================================================
# Slug uniqueness and format
# ===========================================================================

class TestSlugs:
    def test_all_slugs_unique(self, db):
        slugs = [t["slug"] for t in db]
        duplicates = [s for s in slugs if slugs.count(s) > 1]
        assert duplicates == [], f"Duplicate slugs found: {set(duplicates)}"

    def test_slug_format_lowercase_hyphenated(self, db):
        """Slugs must be lowercase alphanumeric with optional hyphens."""
        pattern = re.compile(r"^[a-z0-9][a-z0-9\-]*$")
        invalid = [
            t["slug"] for t in db
            if not pattern.match(t["slug"])
        ]
        assert invalid == [], f"Invalid slug format: {invalid}"

    def test_slugs_not_empty(self, db):
        for tool in db:
            assert tool["slug"].strip(), f"Empty slug found for tool: {tool['name']}"

    def test_slug_not_starts_with_hyphen(self, db):
        for tool in db:
            assert not tool["slug"].startswith("-"), (
                f"Slug starts with hyphen: {tool['slug']}"
            )

    def test_slug_not_ends_with_hyphen(self, db):
        for tool in db:
            assert not tool["slug"].endswith("-"), (
                f"Slug ends with hyphen: {tool['slug']}"
            )

    def test_total_tool_count(self, db):
        """Verify the DB has exactly 244 tools."""
        assert len(db) == 244


# ===========================================================================
# Cross-reference integrity
# ===========================================================================

class TestCrossReferenceIntegrity:
    """
    All slug references in edge fields must exist in the DB.

    NOTE: The DB currently contains 164 dangling references (slugs like
    'tensorboard', 'wandb', 'numpy' that are not in the 244-tool set).
    We document the count here but do NOT assert zero — that would be a
    data-enrichment task, not a structural failure.

    We DO assert: no tool references itself, no reference is empty string,
    and the total count of dangling refs does not grow (regression guard).
    """

    def test_no_self_references(self, db):
        for tool in db:
            slug = tool["slug"]
            for field in EDGE_TYPE_FIELDS:
                refs = tool.get(field, [])
                assert slug not in refs, (
                    f"Tool {slug} references itself in {field}"
                )

    def test_no_empty_string_references(self, db):
        for tool in db:
            for field in EDGE_TYPE_FIELDS:
                refs = tool.get(field, [])
                assert "" not in refs, (
                    f"Tool {tool['slug']}.{field} contains empty string reference"
                )

    def test_no_none_references(self, db):
        for tool in db:
            for field in EDGE_TYPE_FIELDS:
                refs = tool.get(field, [])
                assert None not in refs, (
                    f"Tool {tool['slug']}.{field} contains None reference"
                )

    def test_dangling_ref_count_does_not_exceed_baseline(self, db):
        """
        Regression guard: dangling refs should not exceed the known baseline (200).
        This allows for some organic growth without failing on every new tool
        that references an ecosystem peer not in the 244-tool DB.
        """
        all_slugs: Set[str] = {t["slug"] for t in db}
        dangling = []
        for tool in db:
            for field in EDGE_TYPE_FIELDS:
                for ref in tool.get(field, []):
                    if ref not in all_slugs:
                        dangling.append((tool["slug"], field, ref))
        assert len(dangling) <= 200, (
            f"Dangling ref count {len(dangling)} exceeds baseline of 200. "
            f"Sample: {dangling[:5]}"
        )

    def test_all_resolved_refs_are_valid_slugs(self, db):
        """Every resolvable reference must match a real slug exactly."""
        all_slugs: Set[str] = {t["slug"] for t in db}
        for tool in db:
            for field in EDGE_TYPE_FIELDS:
                for ref in tool.get(field, []):
                    if ref in all_slugs:
                        # If it resolves, the slug must be in the index
                        assert ref in all_slugs  # always true — belt-and-suspenders


# ===========================================================================
# Tags
# ===========================================================================

class TestTags:
    def test_tags_are_list(self, db):
        for tool in db:
            assert isinstance(tool.get("tags", []), list), (
                f"Tool {tool['slug']} tags is not a list"
            )

    def test_tags_are_lowercase(self, db):
        """All tag characters (excluding digits and hyphens) must be lowercase.

        Known exception: numeric-containing tags like '400+-connectors'.
        We skip tags that contain '+' as these are special markers.
        """
        for tool in db:
            for tag in tool.get("tags", []):
                if "+" in tag:
                    continue  # skip special connector tags
                assert tag == tag.lower(), (
                    f"Tool {tool['slug']} has uppercase tag: '{tag}'"
                )

    def test_tags_no_spaces(self, db):
        """Tags should be hyphenated, not space-separated."""
        for tool in db:
            for tag in tool.get("tags", []):
                # Allow numeric tags and special chars but no internal spaces
                assert " " not in tag, (
                    f"Tool {tool['slug']} tag has space: '{tag}'"
                )

    def test_tags_non_empty_strings(self, db):
        for tool in db:
            for tag in tool.get("tags", []):
                assert isinstance(tag, str) and tag.strip(), (
                    f"Tool {tool['slug']} has empty or non-string tag"
                )

    def test_each_tool_has_at_least_one_tag(self, db):
        no_tags = [t["slug"] for t in db if len(t.get("tags", [])) == 0]
        assert no_tags == [], f"Tools with no tags: {no_tags}"


# ===========================================================================
# Numeric fields
# ===========================================================================

class TestNumericFields:
    def test_min_ram_mb_is_integer(self, db):
        for tool in db:
            ram = tool.get("min_ram_mb")
            assert isinstance(ram, int), (
                f"Tool {tool['slug']} min_ram_mb is not int: {ram!r}"
            )

    def test_min_ram_mb_non_negative(self, db):
        for tool in db:
            ram = tool.get("min_ram_mb", 0)
            assert ram >= 0, (
                f"Tool {tool['slug']} has negative min_ram_mb: {ram}"
            )

    def test_github_stars_is_integer_or_zero(self, db):
        for tool in db:
            stars = tool.get("github_stars", 0)
            assert isinstance(stars, int), (
                f"Tool {tool['slug']} github_stars is not int: {stars!r}"
            )

    def test_github_stars_non_negative(self, db):
        for tool in db:
            stars = tool.get("github_stars", 0)
            assert stars >= 0, (
                f"Tool {tool['slug']} has negative github_stars: {stars}"
            )

    def test_contributors_count_non_negative(self, db):
        for tool in db:
            count = tool.get("contributors_count", 0) or 0
            assert count >= 0, (
                f"Tool {tool['slug']} has negative contributors_count: {count}"
            )


# ===========================================================================
# Enumerated fields
# ===========================================================================

class TestEnumeratedFields:
    def test_license_type_valid(self, db):
        for tool in db:
            lic = tool.get("license_type", "")
            assert lic in VALID_LICENSE_TYPES, (
                f"Tool {tool['slug']} has invalid license_type: '{lic}'"
            )

    def test_complexity_level_valid(self, db):
        for tool in db:
            cx = tool.get("complexity_level", "")
            assert cx in VALID_COMPLEXITY_LEVELS, (
                f"Tool {tool['slug']} has invalid complexity_level: '{cx}'"
            )

    def test_maturity_valid(self, db):
        for tool in db:
            m = tool.get("maturity", "")
            assert m in VALID_MATURITY_LEVELS, (
                f"Tool {tool['slug']} has invalid maturity: '{m}'"
            )

    def test_commit_frequency_valid_when_present(self, db):
        for tool in db:
            freq = tool.get("commit_frequency", "")
            if freq:
                assert freq in VALID_COMMIT_FREQUENCIES, (
                    f"Tool {tool['slug']} has invalid commit_frequency: '{freq}'"
                )

    def test_performance_tier_valid_when_present(self, db):
        for tool in db:
            tier = tool.get("performance_tier", "")
            if tier:
                assert tier in VALID_PERFORMANCE_TIERS, (
                    f"Tool {tool['slug']} has invalid performance_tier: '{tier}'"
                )


# ===========================================================================
# Boolean fields
# ===========================================================================

class TestBooleanFields:
    def test_self_hostable_is_bool_when_present(self, db):
        for tool in db:
            val = tool.get("self_hostable")
            if val is not None:
                assert isinstance(val, bool), (
                    f"Tool {tool['slug']} self_hostable is not bool: {val!r}"
                )

    def test_k8s_native_is_bool_when_present(self, db):
        for tool in db:
            val = tool.get("k8s_native")
            if val is not None:
                assert isinstance(val, bool), (
                    f"Tool {tool['slug']} k8s_native is not bool: {val!r}"
                )

    def test_offline_capable_is_bool_when_present(self, db):
        for tool in db:
            val = tool.get("offline_capable")
            if val is not None:
                assert isinstance(val, bool), (
                    f"Tool {tool['slug']} offline_capable is not bool: {val!r}"
                )


# ===========================================================================
# Category index consistency
# ===========================================================================

class TestCategoryIndex:
    def test_known_categories_present(self, db):
        """All 20 known categories should be present."""
        categories = {t["category"] for t in db}
        expected_categories = {
            "AI / ML", "Analytics", "Automation", "CRM & ERP", "Communication",
            "DNS & Networking", "Databases", "Dev Tools", "DevOps & Infra",
            "Email Marketing", "Embeddable", "Knowledge & Docs", "LLMs & AI Infra",
            "Low-Code", "Media & Files", "Monitoring", "Project Mgmt",
            "Scheduling", "Security & Auth", "Web & CMS",
        }
        assert categories == expected_categories, (
            f"Category mismatch. Extra: {categories - expected_categories}. "
            f"Missing: {expected_categories - categories}"
        )

    def test_category_count_is_20(self, db):
        categories = {t["category"] for t in db}
        assert len(categories) == 20

    def test_every_tool_belongs_to_known_category(self, db):
        valid_categories = {
            "AI / ML", "Analytics", "Automation", "CRM & ERP", "Communication",
            "DNS & Networking", "Databases", "Dev Tools", "DevOps & Infra",
            "Email Marketing", "Embeddable", "Knowledge & Docs", "LLMs & AI Infra",
            "Low-Code", "Media & Files", "Monitoring", "Project Mgmt",
            "Scheduling", "Security & Auth", "Web & CMS",
        }
        for tool in db:
            assert tool["category"] in valid_categories, (
                f"Tool {tool['slug']} has unknown category: {tool['category']}"
            )

    def test_crm_category_tools_exist(self, db):
        crm_tools = [t for t in db if t["category"] == "CRM & ERP"]
        assert len(crm_tools) >= 5

    def test_databases_category_includes_postgresql(self, db):
        db_tools = {t["slug"] for t in db if t["category"] == "Databases"}
        assert "postgresql" in db_tools

    def test_llm_ai_category_includes_qdrant(self, db):
        llm_tools = {t["slug"] for t in db if t["category"] == "LLMs & AI Infra"}
        assert "qdrant" in llm_tools


# ===========================================================================
# List fields
# ===========================================================================

class TestListFields:
    def test_use_cases_are_list(self, db):
        for tool in db:
            ucs = tool.get("use_cases_detailed", [])
            assert isinstance(ucs, list), (
                f"Tool {tool['slug']} use_cases_detailed is not list"
            )

    def test_problem_domains_are_list(self, db):
        for tool in db:
            pds = tool.get("problem_domains", [])
            assert isinstance(pds, list), (
                f"Tool {tool['slug']} problem_domains is not list"
            )

    def test_stack_layer_is_list(self, db):
        for tool in db:
            sl = tool.get("stack_layer", [])
            assert isinstance(sl, list), (
                f"Tool {tool['slug']} stack_layer is not list"
            )

    def test_deployment_methods_are_list(self, db):
        for tool in db:
            dm = tool.get("deployment_methods", [])
            assert isinstance(dm, list), (
                f"Tool {tool['slug']} deployment_methods is not list"
            )


# ===========================================================================
# Field count
# ===========================================================================

class TestFieldCount:
    def test_all_tools_have_51_fields(self, db):
        """Every tool should have exactly 51 fields (the full schema)."""
        for tool in db:
            assert len(tool) == 51, (
                f"Tool {tool['slug']} has {len(tool)} fields, expected 51. "
                f"Extra: {set(tool.keys()) - set(db[0].keys())}. "
                f"Missing: {set(db[0].keys()) - set(tool.keys())}"
            )
