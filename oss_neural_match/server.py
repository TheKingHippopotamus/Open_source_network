"""
Open Source Network — MCP Server for Claude Code
===============================================
A 244-tool open-source intelligence engine.
Claude Code (Max plan) connects via stdio and gets full semantic search,
filtering, stack building, and graph traversal capabilities.

Author: KingHippo (https://github.com/TheKingHippopotamus)

Usage with Claude Code:
  claude mcp add open-source-network -- python /path/to/server.py
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, List, Dict, Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict

# ============================================================
# ENGINE MODULE IMPORTS — graceful degradation if not present
# ============================================================

# ScoringEngine (engine/scoring.py) is the most likely to exist.
# We import each engine class individually so a missing graph.py / health.py /
# explain.py does not prevent the scoring upgrade from being applied.

_ENGINE_AVAILABLE = False

try:
    from oss_neural_match.engine.scoring import ScoringEngine
    _ENGINE_AVAILABLE = True
except ImportError:
    ScoringEngine = None  # type: ignore[assignment,misc]

try:
    from oss_neural_match.engine.graph import GraphEngine
    _GRAPH_AVAILABLE = True
except ImportError:
    GraphEngine = None  # type: ignore[assignment,misc]
    _GRAPH_AVAILABLE = False

try:
    from oss_neural_match.engine.health import HealthScorer
    _HEALTH_AVAILABLE = True
except ImportError:
    HealthScorer = None  # type: ignore[assignment,misc]
    _HEALTH_AVAILABLE = False

try:
    from oss_neural_match.engine.explain import RecommendationExplainer
    _EXPLAIN_AVAILABLE = True
except ImportError:
    RecommendationExplainer = None  # type: ignore[assignment,misc]
    _EXPLAIN_AVAILABLE = False


# ============================================================
# INIT
# ============================================================

mcp = FastMCP("open_source_network_mcp")

DB_PATH = Path(__file__).parent / "db.json"
DB: List[Dict[str, Any]] = []
SLUG_INDEX: Dict[str, Dict] = {}
TAG_INDEX: Dict[str, List[str]] = defaultdict(list)
DOMAIN_INDEX: Dict[str, List[str]] = defaultdict(list)
CATEGORY_INDEX: Dict[str, List[str]] = defaultdict(list)

# Legacy TF-IDF state (kept as fallback when engine/scoring.py is absent)
IDF: Dict[str, float] = {}
TOOL_TFIDF: Dict[str, Dict[str, float]] = {}

# Engine instances — None until _load_db() initialises them
SCORING: Optional[Any] = None
GRAPH: Optional[Any] = None
HEALTH: Optional[Any] = None
EXPLAINER: Optional[Any] = None


def _load_db() -> None:
    global DB, SLUG_INDEX, TAG_INDEX, DOMAIN_INDEX, CATEGORY_INDEX
    global IDF, TOOL_TFIDF
    global SCORING, GRAPH, HEALTH, EXPLAINER

    with open(DB_PATH, "r") as f:
        DB = json.load(f)

    # Build lookup indices
    SLUG_INDEX = {t["slug"]: t for t in DB}
    TAG_INDEX.clear()
    DOMAIN_INDEX.clear()
    CATEGORY_INDEX.clear()

    for t in DB:
        for tag in t.get("tags", []):
            TAG_INDEX[tag].append(t["slug"])
        for d in t.get("problem_domains", []):
            DOMAIN_INDEX[d].append(t["slug"])
        CATEGORY_INDEX[t["category"]].append(t["slug"])

    # Always build legacy TF-IDF as a zero-cost fallback
    all_terms: Counter = Counter()
    doc_terms: Dict[str, Counter] = {}
    for t in DB:
        terms = _extract_terms(t)
        doc_terms[t["slug"]] = terms
        all_terms.update(set(terms.keys()))

    n = len(DB)
    IDF = {term: math.log(n / (1 + count)) for term, count in all_terms.items()}
    TOOL_TFIDF = {}
    for slug, terms in doc_terms.items():
        tfidf: Dict[str, float] = {}
        max_tf = max(terms.values()) if terms else 1
        for term, count in terms.items():
            tf = 0.5 + 0.5 * (count / max_tf)
            tfidf[term] = tf * IDF.get(term, 0.0)
        TOOL_TFIDF[slug] = tfidf

    # Attempt to initialise engine modules
    synonyms_path = Path(__file__).parent / "data" / "synonyms.json"
    try:
        synonyms = json.load(open(synonyms_path)) if synonyms_path.exists() else None
    except (OSError, json.JSONDecodeError):
        synonyms = None

    embeddings_dir = Path(__file__).parent / "data"

    if _ENGINE_AVAILABLE and ScoringEngine is not None:
        try:
            SCORING = ScoringEngine(DB, synonyms=synonyms, embeddings_dir=embeddings_dir)
        except Exception:
            SCORING = None

    if _GRAPH_AVAILABLE and GraphEngine is not None:
        try:
            GRAPH = GraphEngine(DB)
        except Exception:
            GRAPH = None

    if _HEALTH_AVAILABLE and HealthScorer is not None:
        try:
            HEALTH = HealthScorer(DB)
        except Exception:
            HEALTH = None

    if _EXPLAIN_AVAILABLE and RecommendationExplainer is not None and SCORING is not None:
        try:
            EXPLAINER = RecommendationExplainer(SCORING, GRAPH, HEALTH)
        except Exception:
            EXPLAINER = None


# ============================================================
# SCORING HELPERS
# ============================================================

def _extract_terms(t: Dict) -> Counter:
    """Extract weighted terms from a tool's semantic fields (legacy TF-IDF)."""
    text_parts: List[str] = []
    # High weight: tags and problem domains (repeat for emphasis)
    for tag in t.get("tags", []):
        text_parts.extend(tag.replace("-", " ").split() * 3)
    for d in t.get("problem_domains", []):
        text_parts.extend(d.replace("-", " ").split() * 2)
    # Medium weight: tagline, use cases
    text_parts.extend(t.get("tagline", "").lower().split())
    for uc in t.get("use_cases_detailed", []):
        text_parts.extend(uc.lower().split())
    # Low weight: replaces, category
    for r in t.get("replaces", []):
        text_parts.extend(r.lower().split())
    text_parts.extend(t.get("category", "").lower().split())
    text_parts.extend(t.get("sub_category", "").lower().split())
    # Clean
    terms: Counter = Counter()
    for w in text_parts:
        w = re.sub(r'[^a-z0-9+#]', '', w)
        if len(w) > 1:
            terms[w] += 1
    return terms


def _legacy_score_query(query: str, slug: str) -> float:
    """Original TF-IDF + exact-match scorer. Preserved verbatim as fallback."""
    t = SLUG_INDEX.get(slug)
    if not t:
        return 0.0

    q_lower = query.lower().strip()
    q_words = re.sub(r'[^a-z0-9+# -]', '', q_lower).split()

    # === EXACT MATCH BOOST (strong signal) ===
    exact_bonus = 0.0
    tool_tags = set(t.get("tags", []))
    tool_domains = set(t.get("problem_domains", []))
    tool_all_text = tool_tags | tool_domains

    for w in q_words:
        # Direct tag/domain hit
        if w in tool_all_text:
            exact_bonus += 0.35
        # Hyphenated query term matches tag (e.g. "email-marketing" -> "email-marketing")
        for tag in tool_all_text:
            if w in tag or tag in w:
                exact_bonus += 0.15
                break
    # Multi-word compound match (e.g. "email marketing" -> "email-marketing" tag)
    q_hyphenated = q_lower.replace(" ", "-")
    for tag in tool_all_text:
        if q_hyphenated == tag or q_hyphenated in tag:
            exact_bonus += 0.5

    # Category/subcategory match
    cat_words = (t.get("category", "") + " " + t.get("sub_category", "")).lower()
    for w in q_words:
        if w in cat_words:
            exact_bonus += 0.1

    # === TF-IDF COSINE (broad signal) ===
    q_terms = Counter(re.sub(r'[^a-z0-9+# ]', '', q_lower).split())
    # Also add split hyphenated terms
    for w in q_words:
        if '-' in w:
            for part in w.split('-'):
                if len(part) > 1:
                    q_terms[part] += 1

    q_tfidf: Dict[str, float] = {}
    max_qtf = max(q_terms.values()) if q_terms else 1
    for term, count in q_terms.items():
        tf = 0.5 + 0.5 * (count / max_qtf)
        q_tfidf[term] = tf * IDF.get(term, 0.0)

    t_tfidf = TOOL_TFIDF.get(slug, {})
    dot = sum(q_tfidf.get(t, 0) * t_tfidf.get(t, 0) for t in set(q_tfidf) | set(t_tfidf))
    mag_q = math.sqrt(sum(v ** 2 for v in q_tfidf.values())) or 1
    mag_t = math.sqrt(sum(v ** 2 for v in t_tfidf.values())) or 1
    cosine = dot / (mag_q * mag_t)

    return cosine + exact_bonus


def _score_query(query: str, slug: str) -> float:
    """Score a tool against a query. Uses ScoringEngine when available, falls back to legacy TF-IDF."""
    if SCORING is not None:
        try:
            return SCORING.score(query, slug)
        except Exception:
            pass
    return _legacy_score_query(query, slug)


# ============================================================
# FORMATTING HELPERS
# ============================================================

def _format_tool_brief(t: Dict) -> str:
    stars = f" \u2b50{t['github_stars'] // 1000}k" if t.get('github_stars', 0) > 0 else ""
    result = (
        f"**{t['name']}** ({t['category']} / {t['sub_category']}){stars}\n"
        f"  {t['tagline']}\n"
        f"  License: {t['license']} ({t['license_type']}) | RAM: {t['min_ram_mb']}MB | "
        f"Complexity: {t['complexity_level']} | Maturity: {t['maturity']}\n"
        f"  Tags: {', '.join(t.get('tags', [])[:8])}\n"
        f"  Replaces: {', '.join(t.get('replaces', [])[:3])}"
    )
    aps = t.get('anti_patterns', [])
    if aps:
        result += f"\n  \u26a0\ufe0f Watch out: {aps[0]}"
    return result


def _format_tool_full(t: Dict) -> str:
    sections = [f"# {t['name']}\n"]
    sections.append(f"**{t['tagline']}**\n")
    sections.append(f"Category: {t['category']} / {t['sub_category']}")
    sections.append(f"License: {t['license']} ({t['license_type']})")
    sections.append(f"Language: {', '.join(t.get('language', []))}")
    sections.append(f"Website: {t.get('website', 'N/A')}")
    sections.append(f"Repo: {t.get('repo_url', 'N/A')}")
    sections.append(f"Stars: {t.get('github_stars', 0)} | Contributors: {t.get('contributors_count', 0)}")
    sections.append(f"Backing: {t.get('backing_org', 'Community')} ({t.get('funding_model', 'community')})")
    sections.append(f"Maturity: {t.get('maturity', 'stable')} | Docs: {t.get('docs_quality', 'good')}")
    sections.append("\n## Technical")
    sections.append(f"Min RAM: {t['min_ram_mb']}MB | CPU: {t.get('min_cpu_cores', 1)} cores")
    sections.append(f"Scaling: {t.get('scaling_pattern', 'single_node')}")
    sections.append(f"Deploy: {', '.join(t.get('deployment_methods', []))}")
    sections.append(
        f"Self-hostable: {t.get('self_hostable', True)} | "
        f"K8s: {t.get('k8s_native', False)} | "
        f"Offline: {t.get('offline_capable', False)}"
    )
    sections.append(f"API: {', '.join(t.get('api_type', []))}")
    sections.append(f"SDKs: {', '.join(t.get('sdk_languages', []))}")
    sections.append("\n## Ecosystem")
    if t.get('integrates_with'):
        sections.append(f"Integrates with: {', '.join(t['integrates_with'])}")
    if t.get('complements'):
        sections.append(f"Complements: {', '.join(t['complements'])}")
    if t.get('similar_to'):
        sections.append(f"Similar to: {', '.join(t['similar_to'])}")
    if t.get('conflicts_with'):
        sections.append(f"\u26a0\ufe0f Conflicts with: {', '.join(t['conflicts_with'])}")
    if t.get('replaces'):
        sections.append(f"Replaces: {', '.join(t['replaces'])}")
    sections.append("\n## Matching Profile")
    sections.append(f"Complexity: {t.get('complexity_level', 'intermediate')}")
    sections.append(f"Team fit: {', '.join(t.get('team_size_fit', []))}")
    sections.append(f"Industries: {', '.join(t.get('industry_verticals', []))}")
    sections.append(f"Performance: {t.get('performance_tier', 'medium')}")
    sections.append(f"Stack layers: {', '.join(t.get('stack_layer', []))}")
    sections.append(f"Tags: {', '.join(t.get('tags', []))}")
    sections.append("\n## Use Cases")
    for uc in t.get('use_cases_detailed', []):
        sections.append(f"- {uc}")
    if t.get('anti_patterns'):
        sections.append("\n## \u26a0\ufe0f Anti-Patterns (when NOT to use)")
        for ap in t['anti_patterns']:
            sections.append(f"- {ap}")
    return "\n".join(sections)


# ============================================================
# MCP TOOLS
# ============================================================

class SearchInput(BaseModel):
    """Search the OSS database with natural language."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    query: str = Field(
        ...,
        description="Natural language search query, e.g. 'lightweight CRM for small team' or 'vector database for RAG'",
        min_length=2,
    )
    limit: int = Field(default=10, description="Max results to return", ge=1, le=50)
    category: Optional[str] = Field(
        default=None,
        description="Filter by category, e.g. 'Databases', 'LLMs & AI Infra', 'CRM & ERP'",
    )
    max_ram_mb: Optional[int] = Field(
        default=None,
        description="Max RAM in MB the tool should need, e.g. 2048 for 2GB VPS",
    )
    license_type: Optional[str] = Field(
        default=None,
        description="License filter: 'permissive', 'copyleft', 'source-available', 'fair-code'",
    )
    complexity: Optional[str] = Field(
        default=None,
        description="Max complexity: 'beginner', 'intermediate', 'advanced', 'expert'",
    )
    self_hosted_only: bool = Field(default=False, description="Only return self-hostable tools")


@mcp.tool(
    name="oss_search",
    annotations={"title": "Search OSS Tools", "readOnlyHint": True, "openWorldHint": False},
)
async def oss_search(params: SearchInput) -> str:
    """Search the 244-tool open-source database by natural language query with optional filters.

    Uses hybrid BM25 + exact-match scoring (or TF-IDF fallback) across tags,
    problem domains, use cases, and descriptions.
    Supports filtering by category, RAM requirements, license type, and complexity.

    Returns ranked results with name, category, tagline, license, RAM, key tags,
    and a watch-out anti-pattern when one exists.
    """
    cx_order = {"beginner": 0, "intermediate": 1, "advanced": 2, "expert": 3}

    candidates = DB
    if params.category:
        cat_lower = params.category.lower()
        candidates = [t for t in candidates if cat_lower in t["category"].lower()]
    if params.max_ram_mb:
        candidates = [t for t in candidates if t.get("min_ram_mb", 256) <= params.max_ram_mb]
    if params.license_type:
        candidates = [t for t in candidates if t.get("license_type", "") == params.license_type]
    if params.complexity:
        max_cx = cx_order.get(params.complexity, 3)
        candidates = [
            t for t in candidates
            if cx_order.get(t.get("complexity_level", "intermediate"), 1) <= max_cx
        ]
    if params.self_hosted_only:
        candidates = [t for t in candidates if t.get("self_hostable", True)]

    scored = [(t, _score_query(params.query, t["slug"])) for t in candidates]
    scored.sort(key=lambda x: -x[1])
    top = scored[:params.limit]

    if not top:
        return f"No tools found matching '{params.query}' with the given filters."

    lines = [f"## Search results for: \"{params.query}\"\n"]
    lines.append(f"*{len(candidates)} tools matched filters, showing top {len(top)} by relevance*\n")
    for i, (t, score) in enumerate(top, 1):
        lines.append(f"### {i}. {_format_tool_brief(t)}")
        lines.append(f"  Relevance: {score:.3f}\n")

    return "\n".join(lines)


class GetToolInput(BaseModel):
    """Get full details of a specific tool."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    slug: str = Field(
        ...,
        description="Tool slug (URL-safe name), e.g. 'postgresql', 'supabase', 'n8n'",
    )


@mcp.tool(
    name="oss_get_tool",
    annotations={"title": "Get Tool Details", "readOnlyHint": True, "openWorldHint": False},
)
async def oss_get_tool(params: GetToolInput) -> str:
    """Get complete 51-field details for a specific tool by slug.

    Returns all metadata including technical specs, ecosystem connections,
    community health, use cases, and anti-patterns.
    """
    t = SLUG_INDEX.get(params.slug)
    if not t:
        suggestions = [s for s in SLUG_INDEX if params.slug in s][:5]
        return (
            f"Tool '{params.slug}' not found. Did you mean: {', '.join(suggestions)}?"
            if suggestions
            else f"Tool '{params.slug}' not found."
        )
    return _format_tool_full(t)


class FindStackInput(BaseModel):
    """Find a compatible tool stack for a use case."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    needs: List[str] = Field(
        ...,
        description="List of needs, e.g. ['authentication', 'crm', 'email-marketing', 'project-management']",
        min_length=1,
        max_length=10,
    )
    max_ram_mb: int = Field(
        default=8192,
        description="Total RAM budget in MB for all tools combined",
    )
    team_size: str = Field(
        default="small",
        description="Team size: 'solo', 'small', 'medium', 'enterprise'",
    )
    max_complexity: str = Field(
        default="intermediate",
        description="Max complexity level the team can handle",
    )
    license_preference: Optional[str] = Field(
        default=None,
        description="Preferred license type: 'permissive', 'copyleft', 'source-available'",
    )


@mcp.tool(
    name="oss_find_stack",
    annotations={"title": "Build Compatible Stack", "readOnlyHint": True, "openWorldHint": False},
)
async def oss_find_stack(params: FindStackInput) -> str:
    """Build a compatible tool stack based on specified needs and constraints.

    For each need, finds the best matching tool considering:
    1. Semantic relevance to the need
    2. RAM budget (cumulative across all tools)
    3. Team size and complexity constraints
    4. License preferences
    5. Cross-tool compatibility (integrates_with, complements, conflicts_with)

    Returns a ranked stack with total RAM usage, compatibility analysis, and warnings.
    """
    cx_order = {"beginner": 0, "intermediate": 1, "advanced": 2, "expert": 3}
    max_cx = cx_order.get(params.max_complexity, 1)

    stack = []
    used_ram = 0
    selected_slugs: set = set()

    for need in params.needs:
        candidates = [
            t for t in DB
            if cx_order.get(t.get("complexity_level", "intermediate"), 1) <= max_cx
            and params.team_size in t.get("team_size_fit", [])
            and t.get("self_hostable", True)
            and t["slug"] not in selected_slugs
        ]
        if params.license_preference:
            pref = [t for t in candidates if t.get("license_type") == params.license_preference]
            if pref:
                candidates = pref

        scored = []
        for t in candidates:
            base_score = _score_query(need, t["slug"])
            # Bonus for integrating with already-selected tools
            compat_bonus = 0.0
            for sel_slug in selected_slugs:
                if sel_slug in t.get("integrates_with", []) or sel_slug in t.get("complements", []):
                    compat_bonus += 0.15
                if sel_slug in t.get("conflicts_with", []):
                    compat_bonus -= 0.5
            # Penalty for RAM overshoot
            ram_left = params.max_ram_mb - used_ram
            ram_penalty = -0.3 if t.get("min_ram_mb", 256) > ram_left else 0.0
            scored.append((t, base_score + compat_bonus + ram_penalty))

        scored.sort(key=lambda x: -x[1])
        if scored:
            best = scored[0][0]
            stack.append({
                "need": need,
                "tool": best,
                "score": scored[0][1],
                "alternatives": [s[0]["name"] for s in scored[1:4]],
            })
            selected_slugs.add(best["slug"])
            used_ram += best.get("min_ram_mb", 256)

    # ---- Compatibility analysis ----
    warnings: List[str] = []
    connections = 0

    # Prefer graph engine when available; fall back to inline loop
    if GRAPH is not None:
        try:
            stack_slugs = [item["tool"]["slug"] for item in stack]
            cohesion_result = GRAPH.stack_cohesion(stack_slugs)
            # GraphEngine.stack_cohesion() returns:
            #   {cohesion_pct, connections (list of pairs), conflicts (list), ...}
            raw_connections = cohesion_result.get("connections", [])
            # connections may be an int (count) or a list of pairs — normalise to int
            connections = raw_connections if isinstance(raw_connections, int) else len(raw_connections)
            cohesion = float(cohesion_result.get("cohesion_pct", 0.0))
            # Surface conflicts returned by the graph engine
            for conflict in cohesion_result.get("conflicts", []):
                if isinstance(conflict, str):
                    warnings.append(f"\u26a0\ufe0f CONFLICT: {conflict}")
                elif isinstance(conflict, (list, tuple)) and len(conflict) >= 2:
                    warnings.append(f"\u26a0\ufe0f CONFLICT: {conflict[0]} conflicts with {conflict[1]}")
        except Exception:
            # Graph engine present but stack_cohesion failed — fall through to inline
            connections, cohesion, warnings = _inline_stack_cohesion(stack)
    else:
        connections, cohesion, warnings = _inline_stack_cohesion(stack)

    # ---- Format output ----
    lines = ["# Recommended Stack\n"]
    lines.append(
        f"*Team: {params.team_size} | RAM budget: {params.max_ram_mb}MB | "
        f"Max complexity: {params.max_complexity}*\n"
    )

    total_ram = 0
    for item in stack:
        t = item["tool"]
        total_ram += t.get("min_ram_mb", 256)
        lines.append(f"## {item['need'].title()} \u2192 **{t['name']}**")
        lines.append(f"  {t['tagline']}")
        lines.append(
            f"  RAM: {t['min_ram_mb']}MB | License: {t['license']} ({t['license_type']}) | "
            f"Complexity: {t['complexity_level']}"
        )
        lines.append(f"  Deploy: {', '.join(t.get('deployment_methods', []))}")
        if item["alternatives"]:
            lines.append(f"  Alternatives: {', '.join(item['alternatives'])}")
        lines.append("")

    lines.append("## Stack Summary")
    lines.append(
        f"- Total RAM: **{total_ram}MB** / {params.max_ram_mb}MB "
        f"({total_ram * 100 // params.max_ram_mb}% of budget)"
    )
    lines.append(f"- Tools: {len(stack)}")
    lines.append(f"- Compatibility connections: {connections} ({cohesion:.0f}% cohesion)")

    if warnings:
        lines.append("\n## \u26a0\ufe0f Warnings")
        for w in warnings:
            lines.append(f"- {w}")

    return "\n".join(lines)


def _inline_stack_cohesion(stack: List[Dict]) -> tuple:
    """
    Compute compatibility stats inline when GraphEngine is unavailable.
    Returns (connections: int, cohesion_pct: float, warnings: List[str]).
    """
    warnings: List[str] = []
    connections = 0

    for i, a in enumerate(stack):
        for j, b in enumerate(stack):
            if i >= j:
                continue
            a_slug = a["tool"]["slug"]
            b_slug = b["tool"]["slug"]
            if (
                b_slug in a["tool"].get("integrates_with", [])
                or b_slug in a["tool"].get("complements", [])
            ):
                connections += 1
            if (
                a_slug in b["tool"].get("integrates_with", [])
                or a_slug in b["tool"].get("complements", [])
            ):
                connections += 1
            if b_slug in a["tool"].get("conflicts_with", []):
                warnings.append(
                    f"\u26a0\ufe0f CONFLICT: {a['tool']['name']} conflicts with {b['tool']['name']}"
                )

    max_possible = len(stack) * (len(stack) - 1)
    cohesion = (connections / max_possible * 100) if max_possible > 0 else 0.0
    return connections, cohesion, warnings


class CompatibleInput(BaseModel):
    """Find tools compatible with a given tool."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    slug: str = Field(
        ...,
        description="Tool slug to find companions for, e.g. 'supabase'",
    )
    limit: int = Field(default=10, ge=1, le=30)


@mcp.tool(
    name="oss_find_compatible",
    annotations={"title": "Find Compatible Tools", "readOnlyHint": True, "openWorldHint": False},
)
async def oss_find_compatible(params: CompatibleInput) -> str:
    """Find tools that integrate with, complement, or are similar to a given tool.

    Uses graph edges (integrates_with, complements, similar_to) to find
    the best companions. Also warns about conflicting tools.
    """
    t = SLUG_INDEX.get(params.slug)
    if not t:
        return f"Tool '{params.slug}' not found."

    results: Dict[str, List[Dict]] = {"integrates": [], "complements": [], "similar": [], "conflicts": []}
    for slug in t.get("integrates_with", []):
        if slug in SLUG_INDEX:
            results["integrates"].append(SLUG_INDEX[slug])
    for slug in t.get("complements", []):
        if slug in SLUG_INDEX:
            results["complements"].append(SLUG_INDEX[slug])
    for slug in t.get("similar_to", []):
        if slug in SLUG_INDEX:
            results["similar"].append(SLUG_INDEX[slug])
    for slug in t.get("conflicts_with", []):
        if slug in SLUG_INDEX:
            results["conflicts"].append(SLUG_INDEX[slug])

    # Also find reverse connections (tools that mention THIS tool)
    reverse_integrates = []
    for other in DB:
        if other["slug"] == params.slug:
            continue
        if params.slug in other.get("integrates_with", []) or params.slug in other.get("complements", []):
            if other["slug"] not in [r["slug"] for r in results["integrates"] + results["complements"]]:
                reverse_integrates.append(other)

    lines = [f"# Compatible tools for: {t['name']}\n"]

    if results["integrates"]:
        lines.append(f"## \U0001f517 Direct integrations ({len(results['integrates'])})")
        for r in results["integrates"][:params.limit]:
            lines.append(f"- **{r['name']}** ({r['category']}) \u2014 {r['tagline']}")

    if results["complements"]:
        lines.append(f"\n## \U0001f91d Best used together ({len(results['complements'])})")
        for r in results["complements"][:params.limit]:
            lines.append(f"- **{r['name']}** ({r['category']}) \u2014 {r['tagline']}")

    if reverse_integrates:
        lines.append(f"\n## \u2190 Also integrates with {t['name']} ({len(reverse_integrates)})")
        for r in reverse_integrates[:params.limit]:
            lines.append(f"- **{r['name']}** ({r['category']}) \u2014 {r['tagline']}")

    if results["similar"]:
        lines.append(f"\n## \U0001f504 Alternatives ({len(results['similar'])})")
        for r in results["similar"][:params.limit]:
            lines.append(f"- **{r['name']}** ({r['category']}) \u2014 {r['tagline']}")

    if results["conflicts"]:
        lines.append("\n## \u26a0\ufe0f Conflicts (avoid combining)")
        for r in results["conflicts"]:
            lines.append(f"- **{r['name']}** \u2014 {r['tagline']}")

    total = len(results["integrates"]) + len(results["complements"]) + len(reverse_integrates)
    if total == 0:
        lines.append("\nNo direct graph connections found. Try `oss_search` with related terms.")

    return "\n".join(lines)


class CompareInput(BaseModel):
    """Compare two or more tools side by side."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    slugs: List[str] = Field(
        ...,
        description="List of tool slugs to compare, e.g. ['postgresql', 'mysql', 'mariadb']",
        min_length=2,
        max_length=6,
    )


@mcp.tool(
    name="oss_compare",
    annotations={"title": "Compare Tools", "readOnlyHint": True, "openWorldHint": False},
)
async def oss_compare(params: CompareInput) -> str:
    """Compare multiple tools side by side across all dimensions.

    Shows a matrix comparing: license, RAM, complexity, maturity, stars,
    scaling pattern, key features, and anti-patterns for each tool.
    """
    tools = [SLUG_INDEX[s] for s in params.slugs if s in SLUG_INDEX]
    if len(tools) < 2:
        return f"Need at least 2 valid tools. Found: {[t['name'] for t in tools]}"

    lines = [f"# Comparison: {' vs '.join(t['name'] for t in tools)}\n"]
    fields = [
        ("Category", lambda t: f"{t['category']} / {t['sub_category']}"),
        ("License", lambda t: f"{t['license']} ({t['license_type']})"),
        ("Language", lambda t: ", ".join(t.get("language", []))),
        ("Min RAM", lambda t: f"{t['min_ram_mb']}MB"),
        ("Scaling", lambda t: t.get("scaling_pattern", "N/A")),
        ("Complexity", lambda t: t.get("complexity_level", "N/A")),
        ("Maturity", lambda t: t.get("maturity", "N/A")),
        ("Stars", lambda t: f"{t.get('github_stars', 0):,}"),
        ("Contributors", lambda t: str(t.get("contributors_count", 0))),
        ("Backing", lambda t: t.get("backing_org", "Community")),
        ("Funding", lambda t: t.get("funding_model", "community")),
        ("Self-hosted", lambda t: "\u2705" if t.get("self_hostable") else "\u274c"),
        ("K8s native", lambda t: "\u2705" if t.get("k8s_native") else "\u274c"),
        ("Offline", lambda t: "\u2705" if t.get("offline_capable") else "\u274c"),
        ("Deploy", lambda t: ", ".join(t.get("deployment_methods", []))),
        ("Team fit", lambda t: ", ".join(t.get("team_size_fit", []))),
        ("Performance", lambda t: t.get("performance_tier", "N/A")),
        ("Docs quality", lambda t: t.get("docs_quality", "N/A")),
    ]

    # Build markdown table
    header = "| Dimension | " + " | ".join(t["name"] for t in tools) + " |"
    sep = "|---|" + "|".join(["---"] * len(tools)) + "|"
    lines.append(header)
    lines.append(sep)
    for label, fn in fields:
        row = f"| **{label}** | " + " | ".join(fn(t) for t in tools) + " |"
        lines.append(row)

    # Key tags comparison
    lines.append("\n## Tags")
    for t in tools:
        lines.append(f"**{t['name']}**: {', '.join(t.get('tags', [])[:10])}")

    # Anti-patterns
    lines.append("\n## \u26a0\ufe0f Anti-patterns")
    for t in tools:
        aps = t.get("anti_patterns", [])
        if aps:
            lines.append(f"**{t['name']}**:")
            for ap in aps[:3]:
                lines.append(f"  - {ap}")

    return "\n".join(lines)


class ListCategoriesInput(BaseModel):
    """List all categories and tool counts."""
    model_config = ConfigDict(extra='forbid')
    include_tools: bool = Field(default=False, description="Include tool names in each category")


@mcp.tool(
    name="oss_list_categories",
    annotations={"title": "List All Categories", "readOnlyHint": True, "openWorldHint": False},
)
async def oss_list_categories(params: ListCategoriesInput) -> str:
    """List all categories in the database with tool counts.

    Provides an overview of the entire 244-tool database organized by category
    and sub-category, useful for understanding what's available before searching.
    """
    lines = [f"# Open Source Network Database \u2014 {len(DB)} tools\n"]
    cat_counts: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for t in DB:
        cat_counts[t["category"]][t["sub_category"]].append(t["name"])

    for cat in sorted(cat_counts, key=lambda c: -sum(len(v) for v in cat_counts[c].values())):
        total = sum(len(v) for v in cat_counts[cat].values())
        lines.append(f"## {cat} ({total} tools)")
        for sub in sorted(cat_counts[cat]):
            tools = cat_counts[cat][sub]
            if params.include_tools:
                lines.append(f"  **{sub}** ({len(tools)}): {', '.join(sorted(tools))}")
            else:
                lines.append(f"  - {sub} ({len(tools)})")
        lines.append("")

    return "\n".join(lines)


class BrowseTagsInput(BaseModel):
    """Browse tools by tag."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    tag: Optional[str] = Field(
        default=None,
        description="Specific tag to look up, e.g. 'rag', 'crm', 'kubernetes'",
    )
    search: Optional[str] = Field(
        default=None,
        description="Search across all tags for partial match",
    )
    limit: int = Field(default=20, ge=1, le=100)


@mcp.tool(
    name="oss_browse_tags",
    annotations={"title": "Browse Tags", "readOnlyHint": True, "openWorldHint": False},
)
async def oss_browse_tags(params: BrowseTagsInput) -> str:
    """Browse the tag taxonomy. List all tags, search for partial matches,
    or get all tools with a specific tag.

    The tag system is the core of semantic matching — understanding available
    tags helps build better search queries.
    """
    if params.tag:
        slugs = TAG_INDEX.get(params.tag, [])
        if not slugs:
            # Try partial match
            matches = [(tag, sls) for tag, sls in TAG_INDEX.items() if params.tag in tag]
            if matches:
                lines = [f"Tag '{params.tag}' not found exactly. Partial matches:"]
                for tag, sls in matches[:10]:
                    lines.append(f"  - **{tag}** ({len(sls)} tools)")
                return "\n".join(lines)
            return f"Tag '{params.tag}' not found."
        tools = [SLUG_INDEX[s] for s in slugs if s in SLUG_INDEX]
        lines = [f"# Tag: {params.tag} ({len(tools)} tools)\n"]
        for t in tools:
            lines.append(f"- **{t['name']}** ({t['category']}) \u2014 {t['tagline']}")
        return "\n".join(lines)

    if params.search:
        matches = [
            (tag, len(slugs))
            for tag, slugs in TAG_INDEX.items()
            if params.search.lower() in tag
        ]
        matches.sort(key=lambda x: -x[1])
        lines = [f"# Tags matching '{params.search}' ({len(matches)} found)\n"]
        for tag, count in matches[:params.limit]:
            lines.append(f"- **{tag}** ({count} tools)")
        return "\n".join(lines)

    # List top tags by frequency
    tag_counts = [(tag, len(slugs)) for tag, slugs in TAG_INDEX.items()]
    tag_counts.sort(key=lambda x: -x[1])
    lines = [f"# Top {params.limit} tags (of {len(TAG_INDEX)} total)\n"]
    for tag, count in tag_counts[:params.limit]:
        lines.append(f"- **{tag}** ({count} tools)")
    return "\n".join(lines)


class StatsInput(BaseModel):
    """Get database statistics."""
    model_config = ConfigDict(extra='forbid')


@mcp.tool(
    name="oss_stats",
    annotations={"title": "Database Statistics", "readOnlyHint": True, "openWorldHint": False},
)
async def oss_stats(params: StatsInput) -> str:
    """Get comprehensive statistics about the Open Source Network database.

    Returns total tools, fields per tool, graph edges, tag counts,
    problem domain counts, and category distribution.
    """
    total_edges = sum(
        len(t.get('integrates_with', [])) + len(t.get('complements', [])) +
        len(t.get('similar_to', [])) + len(t.get('conflicts_with', []))
        for t in DB
    )
    total_tags = sum(len(t.get('tags', [])) for t in DB)
    total_domains = sum(len(t.get('problem_domains', [])) for t in DB)
    total_anti = sum(len(t.get('anti_patterns', [])) for t in DB)
    total_uc = sum(len(t.get('use_cases_detailed', [])) for t in DB)

    engine_status = "hybrid BM25" if _ENGINE_AVAILABLE and SCORING is not None else "legacy TF-IDF"

    lines = [
        "# Open Source Network Database Stats\n",
        f"- **Total tools**: {len(DB)}",
        f"- **Fields per tool**: {len(DB[0]) if DB else 0}",
        f"- **Total data points**: {len(DB) * (len(DB[0]) if DB else 0):,}",
        f"- **Categories**: {len(CATEGORY_INDEX)}",
        f"- **Unique tags**: {len(TAG_INDEX)}",
        f"- **Unique problem domains**: {len(DOMAIN_INDEX)}",
        f"- **Graph edges**: {total_edges:,}",
        f"- **Tag entries**: {total_tags:,}",
        f"- **Problem domains**: {total_domains:,}",
        f"- **Anti-patterns**: {total_anti:,}",
        f"- **Use cases**: {total_uc:,}",
        f"- **Scoring engine**: {engine_status}",
    ]
    return "\n".join(lines)


# ============================================================
# NEW TOOL: oss_health_score
# ============================================================

class HealthScoreInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    slug: str = Field(..., description="Tool slug to check health score for")


@mcp.tool(
    name="oss_health_score",
    annotations={"title": "OSS Health Score", "readOnlyHint": True, "openWorldHint": False},
)
async def oss_health_score(params: HealthScoreInput) -> str:
    """Get the health score for an open-source tool.

    Returns a composite score (0-1) with grade, risk band, and per-dimension
    breakdown covering community activity, maintenance, documentation, and
    ecosystem factors.

    Requires the engine/health.py module. Falls back to a metadata summary
    when the module is not available.
    """
    t = SLUG_INDEX.get(params.slug)
    if not t:
        suggestions = [s for s in SLUG_INDEX if params.slug in s][:5]
        return (
            f"Tool '{params.slug}' not found. Did you mean: {', '.join(suggestions)}?"
            if suggestions
            else f"Tool '{params.slug}' not found."
        )

    if HEALTH is not None:
        try:
            result = HEALTH.score(params.slug)
            # Normalise across two possible key names for the composite score.
            # engine/health.py uses "overall"; future versions may use "composite".
            composite = result.get("overall", result.get("composite", 0.0))
            grade = result.get("grade", "N/A")
            risk_band = result.get("risk_band", "unknown")
            dimensions: Dict = result.get("dimensions", {})
            summary = result.get("summary", "")

            lines = [f"# Health Score: {t['name']}\n"]
            lines.append(
                f"**Score**: {composite:.3f} | **Grade**: {grade} | **Risk**: {risk_band}\n"
            )
            if summary:
                lines.append(f"_{summary}_\n")

            if dimensions:
                lines.append("## Dimension Breakdown")
                for dim, dim_data in dimensions.items():
                    if isinstance(dim_data, dict):
                        dim_score = dim_data.get("score", 0.0)
                        weight = dim_data.get("weight", 0.0)
                        bar = _score_bar(float(dim_score))
                        lines.append(
                            f"- **{dim}**: {bar} {float(dim_score):.3f}"
                            f"  (weight {weight:.0%})"
                        )
                        # Surface individual factors
                        for factor in dim_data.get("factors", []):
                            f_name = factor.get("name", "")
                            f_val = factor.get("value", "")
                            f_score = factor.get("score", "")
                            score_str = f" → {f_score:.3f}" if isinstance(f_score, (int, float)) else ""
                            lines.append(f"    - {f_name}: {f_val}{score_str}")
                    else:
                        # Scalar fallback
                        bar = _score_bar(float(dim_data))
                        lines.append(f"- **{dim}**: {bar} {float(dim_data):.3f}")

            # Surface any low-scoring dimensions
            low_dims = []
            for dim, dim_data in dimensions.items():
                dim_score = dim_data.get("score", dim_data) if isinstance(dim_data, dict) else dim_data
                if isinstance(dim_score, (int, float)) and float(dim_score) < 0.4:
                    low_dims.append(dim)
            if low_dims:
                lines.append(f"\n**Low-scoring areas**: {', '.join(low_dims)}")

            return "\n".join(lines)

        except Exception:
            # Engine present but failed — fall through to metadata fallback
            pass

    # Metadata-based fallback when HealthScorer is unavailable
    return _health_from_metadata(t)


def _score_bar(score: float, width: int = 10) -> str:
    """Render a simple ASCII progress bar for a 0-1 score."""
    filled = round(score * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _health_from_metadata(t: Dict) -> str:
    """Derive a rough health summary purely from static metadata fields."""
    lines = [f"# Health Summary: {t['name']} (metadata-based)\n"]
    lines.append(
        "*Full HealthScorer engine not available. Showing static metadata signals.*\n"
    )

    stars = t.get("github_stars", 0)
    contributors = t.get("contributors_count", 0)
    maturity = t.get("maturity", "unknown")
    docs = t.get("docs_quality", "unknown")
    funding = t.get("funding_model", "community")
    backing = t.get("backing_org", "Community")

    lines.append(f"- **Stars**: {stars:,}")
    lines.append(f"- **Contributors**: {contributors}")
    lines.append(f"- **Maturity**: {maturity}")
    lines.append(f"- **Docs quality**: {docs}")
    lines.append(f"- **Funding model**: {funding}")
    lines.append(f"- **Backing org**: {backing}")

    # Simple heuristic risk assessment
    risk = "low"
    risk_reasons: List[str] = []
    if stars < 500:
        risk = "medium"
        risk_reasons.append("low GitHub stars")
    if contributors < 10:
        risk = "medium"
        risk_reasons.append("few contributors")
    if maturity in ("experimental", "beta"):
        risk = "high"
        risk_reasons.append(f"maturity is {maturity}")
    if funding == "community" and backing == "Community" and stars < 1000:
        if risk == "low":
            risk = "medium"
        risk_reasons.append("no commercial backing")

    lines.append(f"\n**Risk estimate**: {risk}" + (f" ({', '.join(risk_reasons)})" if risk_reasons else ""))
    return "\n".join(lines)


# ============================================================
# NEW TOOL: oss_explain_recommendation
# ============================================================

class ExplainInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    query: str = Field(..., description="The search query that produced this recommendation")
    slug: str = Field(..., description="The tool slug to explain")


@mcp.tool(
    name="oss_explain_recommendation",
    annotations={"title": "Explain Recommendation", "readOnlyHint": True, "openWorldHint": False},
)
async def oss_explain_recommendation(params: ExplainInput) -> str:
    """Explain why a specific tool was recommended for a given query.

    Shows score breakdown, matching terms, graph context, health assessment,
    and anti-patterns.

    Uses the RecommendationExplainer engine when available; falls back to a
    structured breakdown derived directly from ScoringEngine.explain_score()
    and static metadata.
    """
    t = SLUG_INDEX.get(params.slug)
    if not t:
        suggestions = [s for s in SLUG_INDEX if params.slug in s][:5]
        return (
            f"Tool '{params.slug}' not found. Did you mean: {', '.join(suggestions)}?"
            if suggestions
            else f"Tool '{params.slug}' not found."
        )

    # --- Path 1: Full explainer engine ---
    if EXPLAINER is not None:
        try:
            explanation = EXPLAINER.explain(params.query, params.slug, t)
            # explanation is expected to be a dict or a formatted string
            if isinstance(explanation, str):
                return explanation
            return _format_explanation_dict(explanation, t)
        except Exception:
            pass  # fall through to Path 2

    # --- Path 2: ScoringEngine.explain_score() ---
    if SCORING is not None:
        try:
            detail = SCORING.explain_score(params.query, params.slug)
            return _format_scoring_explanation(detail, t)
        except Exception:
            pass  # fall through to Path 3

    # --- Path 3: Pure metadata fallback ---
    return _explain_from_metadata(params.query, t)


def _format_explanation_dict(explanation: Dict, t: Dict) -> str:
    """Format a dict returned by RecommendationExplainer.explain()."""
    lines = [f"# Why {t['name']} was recommended\n"]
    for key, value in explanation.items():
        if isinstance(value, (list, dict)):
            lines.append(f"**{key}**: {json.dumps(value, ensure_ascii=False)}")
        else:
            lines.append(f"**{key}**: {value}")
    return "\n".join(lines)


def _format_scoring_explanation(detail: Dict, t: Dict) -> str:
    """Format the dict returned by ScoringEngine.explain_score()."""
    lines = [f"# Why {t['name']} was recommended\n"]

    lines.append(f"**Query**: {detail.get('query', '')}")
    lines.append(f"**Expanded tokens**: {', '.join(detail.get('tokens', []))}\n")

    lines.append("## Score Breakdown")
    lines.append(f"- BM25 (normalised):  {detail.get('bm25_normalised', 0):.4f}")
    lines.append(f"- Exact match bonus:  {detail.get('exact_bonus', 0):.4f}")
    lines.append(f"- Dense similarity:   {detail.get('dense_score', 0):.4f}"
                 f"  ({'enabled' if detail.get('dense_available') else 'not available'})")
    lines.append(f"- **Final score**:    {detail.get('final_score', 0):.4f}")
    lines.append(f"- Weights applied:    {detail.get('weights', {})}\n")

    top_terms = detail.get("top_bm25_terms", [])
    if top_terms:
        lines.append("## Top BM25 Terms")
        for term, score in top_terms:
            lines.append(f"  - `{term}`: {score:.4f}")
        lines.append("")

    exact_hits = detail.get("exact_hits", [])
    if exact_hits:
        lines.append("## Exact Match Signals")
        for hit in exact_hits:
            tok = hit.get("token", "")
            reasons = []
            if hit.get("direct"):
                reasons.append("direct tag/domain match")
            if hit.get("partial"):
                reasons.append(f"partial match in '{hit['partial']}'")
            if hit.get("category"):
                reasons.append("category match")
            if hit.get("compound"):
                reasons.append("compound query match")
            lines.append(f"  - `{tok}`: {', '.join(reasons)}")
        lines.append("")

    # Ecosystem context
    integrates = t.get("integrates_with", [])
    complements = t.get("complements", [])
    if integrates or complements:
        lines.append("## Ecosystem Fit")
        if integrates:
            lines.append(f"  Integrates with: {', '.join(integrates[:5])}")
        if complements:
            lines.append(f"  Complements: {', '.join(complements[:5])}")
        lines.append("")

    # Anti-patterns
    aps = t.get("anti_patterns", [])
    if aps:
        lines.append(f"## When NOT to use {t['name']}")
        for ap in aps:
            lines.append(f"  - {ap}")

    return "\n".join(lines)


def _explain_from_metadata(query: str, t: Dict) -> str:
    """Pure metadata explanation when no engine is available."""
    lines = [f"# Why {t['name']} was recommended\n"]
    lines.append(f"**Query**: {query}")
    lines.append("*(Detailed scoring unavailable — showing metadata signals)*\n")

    q_lower = query.lower()
    q_words = set(re.sub(r'[^a-z0-9 ]', '', q_lower).split())

    tool_tags = set(t.get("tags", []))
    tool_domains = set(t.get("problem_domains", []))
    all_terms = tool_tags | tool_domains

    matching_tags = [tag for tag in all_terms if any(w in tag or tag in w for w in q_words)]
    if matching_tags:
        lines.append("## Matching Tags / Domains")
        for tag in matching_tags[:10]:
            lines.append(f"  - {tag}")
        lines.append("")

    lines.append("## Tool Profile")
    lines.append(f"  - Category: {t.get('category')} / {t.get('sub_category')}")
    lines.append(f"  - Tagline: {t.get('tagline')}")
    lines.append(f"  - Maturity: {t.get('maturity')} | Complexity: {t.get('complexity_level')}")
    lines.append(f"  - Stars: {t.get('github_stars', 0):,} | Contributors: {t.get('contributors_count', 0)}")
    lines.append("")

    aps = t.get("anti_patterns", [])
    if aps:
        lines.append(f"## When NOT to use {t['name']}")
        for ap in aps:
            lines.append(f"  - {ap}")

    return "\n".join(lines)


# ============================================================
# STARTUP
# ============================================================

_load_db()

def main():
    """Entry point for the CLI command."""
    mcp.run()


if __name__ == "__main__":
    main()
