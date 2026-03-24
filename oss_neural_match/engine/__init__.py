"""
Open Source Network — Engine Module
=================================
Modular intelligence layer for the Open Source Network MCP server.

Components:
- scoring: BM25 + dense hybrid search scoring
- graph: Knowledge graph algorithms (PageRank, communities, link prediction)
- health: OSS health scoring from metadata
- explain: Recommendation explainability
"""

from oss_neural_match.engine.scoring import ScoringEngine

try:
    from oss_neural_match.engine.graph import GraphEngine
except ImportError:
    GraphEngine = None  # type: ignore[assignment,misc]

try:
    from oss_neural_match.engine.health import HealthScorer
except ImportError:
    HealthScorer = None  # type: ignore[assignment,misc]

try:
    from oss_neural_match.engine.explain import RecommendationExplainer
except ImportError:
    RecommendationExplainer = None  # type: ignore[assignment,misc]

__all__ = ['ScoringEngine', 'GraphEngine', 'HealthScorer', 'RecommendationExplainer']
