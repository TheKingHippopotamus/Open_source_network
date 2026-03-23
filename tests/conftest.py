"""
Shared pytest fixtures for the Open Source Network test suite.
"""

import json
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_DB_PATH = _REPO_ROOT / "db.json"


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def db():
    """Load the full 244-tool database once per session."""
    with open(_DB_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def slug_index(db):
    """Dict mapping slug -> tool record for fast lookup."""
    return {t["slug"]: t for t in db}


@pytest.fixture(scope="session")
def sample_tools(db):
    """A small 20-tool subset for fast unit tests."""
    return db[:20]


# ---------------------------------------------------------------------------
# Engine fixtures (session-scoped so they are built once, reused everywhere)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def scoring_engine(db):
    """A ScoringEngine built from the full DB."""
    from engine.scoring import ScoringEngine
    return ScoringEngine(db)


@pytest.fixture(scope="session")
def graph_engine(db):
    """A GraphEngine built from the full DB."""
    from engine.graph import GraphEngine
    return GraphEngine(db)


@pytest.fixture(scope="session")
def health_scorer(db):
    """A HealthScorer pre-computed over the full DB."""
    from engine.health import HealthScorer
    return HealthScorer(db)


@pytest.fixture(scope="session")
def explainer(scoring_engine, graph_engine, health_scorer, slug_index):
    """A RecommendationExplainer wired to all three engines."""
    from engine.explain import RecommendationExplainer
    return RecommendationExplainer(
        scoring_engine=scoring_engine,
        graph_engine=graph_engine,
        health_scorer=health_scorer,
        slug_index=slug_index,
    )
