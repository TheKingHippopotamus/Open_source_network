"""
tests/test_health.py
====================
Tests for engine/health.py:
  - HealthScorer construction
  - All 244 tools get a health score
  - Scores are in [0.0, 1.0]
  - Grade assignment thresholds
  - Risk band thresholds
  - Known-tool scores (PostgreSQL and other mature tools)
  - Dimension breakdown validity
  - compare_health() output format
  - get_risk_tools() threshold filtering
  - all_scores() ordering
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from engine.health import (
    HealthScorer,
    WEIGHTS,
    GRADE_THRESHOLDS,
    RISK_BANDS,
    _assign_grade,
    _assign_risk_band,
    _clamp,
    _log_normalise,
    _months_since,
)


# ===========================================================================
# Helper functions
# ===========================================================================

class TestHelpers:
    def test_clamp_lower_bound(self):
        assert _clamp(-0.5) == 0.0

    def test_clamp_upper_bound(self):
        assert _clamp(1.5) == 1.0

    def test_clamp_in_range(self):
        assert _clamp(0.7) == 0.7

    def test_clamp_custom_bounds(self):
        assert _clamp(5.0, lo=0.0, hi=10.0) == 5.0
        assert _clamp(-1.0, lo=0.0, hi=10.0) == 0.0
        assert _clamp(15.0, lo=0.0, hi=10.0) == 10.0

    def test_log_normalise_zero_value(self):
        assert _log_normalise(0, 0, 100000) == 0.0

    def test_log_normalise_max_value(self):
        result = _log_normalise(100000, 0, 100000)
        assert abs(result - 1.0) < 1e-9

    def test_log_normalise_between(self):
        result = _log_normalise(5000, 0, 100000)
        assert 0.0 < result < 1.0

    def test_log_normalise_equal_min_max(self):
        assert _log_normalise(100, 100, 100) == 0.0

    def test_months_since_valid_date(self):
        # "2025-09" is 6 months before March 2026 reference date
        months = _months_since("2025-09")
        assert months is not None
        assert 5 < months < 7

    def test_months_since_empty_returns_none(self):
        assert _months_since("") is None

    def test_months_since_malformed_returns_none(self):
        assert _months_since("bad-date") is None

    def test_months_since_future_returns_negative_or_small(self):
        # "2030-01" is in the future relative to 2026-03-01
        months = _months_since("2030-01")
        assert months is not None
        assert months < 0

    def test_assign_grade_thresholds(self):
        assert _assign_grade(0.90) == "A"
        assert _assign_grade(0.85) == "A"
        assert _assign_grade(0.80) == "B"
        assert _assign_grade(0.70) == "B"
        assert _assign_grade(0.60) == "C"
        assert _assign_grade(0.50) == "C"
        assert _assign_grade(0.40) == "D"
        assert _assign_grade(0.30) == "D"
        assert _assign_grade(0.29) == "F"
        assert _assign_grade(0.00) == "F"

    def test_assign_risk_band_thresholds(self):
        assert _assign_risk_band(0.80) == "green"
        assert _assign_risk_band(0.70) == "green"
        assert _assign_risk_band(0.60) == "yellow"
        assert _assign_risk_band(0.50) == "yellow"
        assert _assign_risk_band(0.40) == "orange"
        assert _assign_risk_band(0.30) == "orange"
        assert _assign_risk_band(0.29) == "red"
        assert _assign_risk_band(0.00) == "red"


# ===========================================================================
# Weight constants
# ===========================================================================

class TestWeightConstants:
    def test_weights_sum_to_one(self):
        total = sum(WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_all_weights_positive(self):
        for dim, w in WEIGHTS.items():
            assert w > 0.0, f"Weight for {dim} is non-positive: {w}"

    def test_expected_dimensions_present(self):
        expected = {"activity", "community", "maturity", "backing", "license", "documentation"}
        assert expected == set(WEIGHTS.keys())


# ===========================================================================
# HealthScorer construction
# ===========================================================================

class TestHealthScorerConstruction:
    def test_requires_non_empty_tools(self):
        with pytest.raises(ValueError):
            HealthScorer([])

    def test_builds_from_full_db(self, db):
        scorer = HealthScorer(db)
        assert scorer is not None

    def test_all_tools_get_scores(self, db, health_scorer):
        scored_slugs = set(health_scorer._scores.keys())
        db_slugs = {t["slug"] for t in db}
        assert scored_slugs == db_slugs

    def test_total_scored_equals_244(self, health_scorer):
        assert len(health_scorer._scores) == 244


# ===========================================================================
# Score validity
# ===========================================================================

class TestScoreValidity:
    def test_all_overall_scores_in_0_1(self, health_scorer):
        for slug, s in health_scorer._scores.items():
            assert 0.0 <= s["overall"] <= 1.0, (
                f"{slug} overall={s['overall']} is out of [0, 1]"
            )

    def test_all_dimension_scores_in_0_1(self, health_scorer):
        for slug, s in health_scorer._scores.items():
            for dim, data in s["dimensions"].items():
                score = data["score"]
                assert 0.0 <= score <= 1.0, (
                    f"{slug}.{dim} score={score} is out of [0, 1]"
                )

    def test_grade_is_valid_letter(self, health_scorer):
        valid_grades = {"A", "B", "C", "D", "F"}
        for slug, s in health_scorer._scores.items():
            assert s["grade"] in valid_grades, (
                f"{slug} has invalid grade: {s['grade']}"
            )

    def test_risk_band_is_valid(self, health_scorer):
        valid_bands = {"green", "yellow", "orange", "red"}
        for slug, s in health_scorer._scores.items():
            assert s["risk_band"] in valid_bands, (
                f"{slug} has invalid risk_band: {s['risk_band']}"
            )

    def test_grade_consistent_with_thresholds(self, health_scorer):
        for slug, s in health_scorer._scores.items():
            overall = s["overall"]
            grade = s["grade"]
            expected = _assign_grade(overall)
            assert grade == expected, (
                f"{slug}: grade={grade} but _assign_grade({overall})={expected}"
            )

    def test_risk_band_consistent_with_thresholds(self, health_scorer):
        for slug, s in health_scorer._scores.items():
            overall = s["overall"]
            band = s["risk_band"]
            expected = _assign_risk_band(overall)
            assert band == expected, (
                f"{slug}: risk_band={band} but expected {expected}"
            )


# ===========================================================================
# Known-tool scores
# ===========================================================================

class TestKnownToolScores:
    def test_postgresql_overall_above_0_7(self, health_scorer):
        s = health_scorer.score("postgresql")
        assert s["overall"] > 0.7, (
            f"PostgreSQL overall={s['overall']} — expected healthy tool (>0.7)"
        )

    def test_tensorflow_grade_a_or_b(self, health_scorer):
        s = health_scorer.score("tensorflow")
        assert s["grade"] in ("A", "B"), (
            f"TensorFlow grade={s['grade']} — expected A or B for Google-backed mature project"
        )

    def test_pytorch_in_green_band(self, health_scorer):
        s = health_scorer.score("pytorch")
        assert s["risk_band"] == "green", (
            f"PyTorch risk_band={s['risk_band']} — expected green"
        )

    def test_redis_overall_above_0_7(self, health_scorer):
        s = health_scorer.score("redis")
        assert s["overall"] > 0.7

    def test_scikit_learn_grade_not_f(self, health_scorer):
        s = health_scorer.score("scikit-learn")
        assert s["grade"] != "F"

    def test_mlflow_overall_above_0_85(self, health_scorer):
        # mlflow is known to score very high in our dataset
        s = health_scorer.score("mlflow")
        assert s["overall"] > 0.85

    def test_missing_slug_raises_keyerror(self, health_scorer):
        with pytest.raises(KeyError):
            health_scorer.score("no-such-slug-xyz")


# ===========================================================================
# Dimension breakdown
# ===========================================================================

class TestDimensionBreakdown:
    def test_all_six_dimensions_present(self, health_scorer):
        s = health_scorer.score("postgresql")
        expected = {"activity", "community", "maturity", "backing", "license", "documentation"}
        assert expected == set(s["dimensions"].keys())

    def test_each_dimension_has_score_weight_factors(self, health_scorer):
        s = health_scorer.score("postgresql")
        for dim, data in s["dimensions"].items():
            assert "score" in data, f"Missing 'score' in dimension {dim}"
            assert "weight" in data, f"Missing 'weight' in dimension {dim}"
            assert "factors" in data, f"Missing 'factors' in dimension {dim}"

    def test_dimension_weights_match_constants(self, health_scorer):
        s = health_scorer.score("postgresql")
        for dim, data in s["dimensions"].items():
            assert data["weight"] == WEIGHTS[dim], (
                f"{dim} weight mismatch: {data['weight']} != {WEIGHTS[dim]}"
            )

    def test_weighted_sum_approximates_overall(self, health_scorer):
        for slug in ["postgresql", "redis", "qdrant", "tensorflow"]:
            s = health_scorer.score(slug)
            weighted_sum = sum(
                s["dimensions"][dim]["score"] * s["dimensions"][dim]["weight"]
                for dim in s["dimensions"]
            )
            assert abs(weighted_sum - s["overall"]) < 1e-3, (
                f"{slug}: weighted sum {weighted_sum} != overall {s['overall']}"
            )

    def test_factors_are_non_empty_list(self, health_scorer):
        s = health_scorer.score("tensorflow")
        for dim, data in s["dimensions"].items():
            assert isinstance(data["factors"], list)
            assert len(data["factors"]) >= 1, f"No factors for dimension {dim}"

    def test_factor_has_name_value_score(self, health_scorer):
        s = health_scorer.score("postgresql")
        for dim, data in s["dimensions"].items():
            for factor in data["factors"]:
                assert "name" in factor, f"Factor missing 'name' in {dim}"

    def test_summary_is_non_empty_string(self, health_scorer):
        s = health_scorer.score("postgresql")
        assert isinstance(s["summary"], str)
        assert len(s["summary"]) > 10


# ===========================================================================
# compare_health
# ===========================================================================

class TestCompareHealth:
    def test_returns_string(self, health_scorer):
        result = health_scorer.compare_health(["postgresql", "mysql"])
        assert isinstance(result, str)

    def test_contains_markdown_table(self, health_scorer):
        result = health_scorer.compare_health(["postgresql", "mysql", "mariadb"])
        assert "| Rank |" in result

    def test_contains_all_requested_slugs(self, health_scorer):
        slugs = ["postgresql", "redis", "qdrant"]
        result = health_scorer.compare_health(slugs)
        for slug in slugs:
            assert slug in result

    def test_sorted_descending_by_overall(self, health_scorer):
        slugs = ["qdrant", "postgresql", "redis", "milvus"]
        result = health_scorer.compare_health(slugs)
        # Parse rank column: first mentioned slug should have higher overall
        lines = [l for l in result.split("\n") if l.startswith("| 1 ")]
        assert len(lines) == 1, "Expected exactly one rank-1 row"

    def test_unknown_slugs_skipped(self, health_scorer):
        result = health_scorer.compare_health(["postgresql", "no-such-slug-xyz"])
        assert "no-such-slug-xyz" not in result
        assert "postgresql" in result

    def test_empty_slugs_returns_message(self, health_scorer):
        result = health_scorer.compare_health([])
        assert "No valid" in result or "no valid" in result.lower()

    def test_all_unknown_slugs_returns_message(self, health_scorer):
        result = health_scorer.compare_health(["no-such-1", "no-such-2"])
        assert "No valid" in result or "_No valid" in result

    def test_single_tool_still_produces_table(self, health_scorer):
        result = health_scorer.compare_health(["postgresql"])
        assert "| 1 " in result

    def test_contains_summaries_section(self, health_scorer):
        result = health_scorer.compare_health(["postgresql", "mysql"])
        assert "Summaries" in result


# ===========================================================================
# get_risk_tools
# ===========================================================================

class TestGetRiskTools:
    def test_returns_list(self, health_scorer):
        tools = health_scorer.get_risk_tools()
        assert isinstance(tools, list)

    def test_all_below_threshold(self, health_scorer):
        threshold = 0.5
        tools = health_scorer.get_risk_tools(threshold=threshold)
        for tool in tools:
            assert tool["overall"] < threshold, (
                f"{tool['slug']} overall={tool['overall']} is above threshold {threshold}"
            )

    def test_sorted_ascending(self, health_scorer):
        tools = health_scorer.get_risk_tools(threshold=0.7)
        scores = [t["overall"] for t in tools]
        assert scores == sorted(scores), "get_risk_tools() should be sorted ascending"

    def test_high_threshold_includes_more_tools(self, health_scorer):
        tools_70 = health_scorer.get_risk_tools(threshold=0.7)
        tools_50 = health_scorer.get_risk_tools(threshold=0.5)
        assert len(tools_70) >= len(tools_50)

    def test_zero_threshold_returns_empty(self, health_scorer):
        # No tool can have overall < 0.0
        tools = health_scorer.get_risk_tools(threshold=0.0)
        assert tools == []

    def test_one_threshold_returns_all(self, health_scorer):
        tools = health_scorer.get_risk_tools(threshold=1.01)
        assert len(tools) == 244

    def test_results_contain_full_score_dicts(self, health_scorer):
        tools = health_scorer.get_risk_tools(threshold=0.7)
        if tools:
            t = tools[0]
            assert "slug" in t
            assert "overall" in t
            assert "grade" in t
            assert "risk_band" in t
            assert "dimensions" in t


# ===========================================================================
# all_scores
# ===========================================================================

class TestAllScores:
    def test_returns_list_of_length_244(self, health_scorer):
        scores = health_scorer.all_scores()
        assert len(scores) == 244

    def test_sorted_descending(self, health_scorer):
        scores = health_scorer.all_scores()
        overalls = [s["overall"] for s in scores]
        assert overalls == sorted(overalls, reverse=True)

    def test_first_is_highest_score(self, health_scorer):
        scores = health_scorer.all_scores()
        assert scores[0]["overall"] >= scores[-1]["overall"]

    def test_all_have_required_keys(self, health_scorer):
        scores = health_scorer.all_scores()
        required = {"slug", "overall", "grade", "risk_band", "dimensions", "summary"}
        for s in scores:
            assert required.issubset(set(s.keys()))
