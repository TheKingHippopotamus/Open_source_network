"""
engine/explain.py — Recommendation Explainer
=============================================
Produces clear, opinionated, senior-engineer-quality explanations of why
a tool was (or was not) recommended for a given query or stack.

No external dependencies. Designed to work standalone or alongside
scoring_engine, graph_engine, and health_scorer if those exist. Falls back
to computing everything inline from the tool dict when they do not.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Confidence thresholds (tuned against TF-IDF + exact-match scoring range)
# ---------------------------------------------------------------------------
_CONFIDENCE_HIGH = 0.75
_CONFIDENCE_MED = 0.35

# Complexity order used for display and comparisons
_CX_ORDER = {"beginner": 0, "intermediate": 1, "advanced": 2, "expert": 3}
_CX_LABEL = {
    "beginner": "beginner-friendly",
    "intermediate": "intermediate",
    "advanced": "advanced",
    "expert": "expert-only",
}

# Performance tier labels
_PERF_LABEL = {
    "lightweight": "lightweight / resource-efficient",
    "medium": "mid-tier performance",
    "heavy": "resource-heavy",
    "enterprise_grade": "enterprise-grade performance",
}

# Funding model signals
_FUNDING_RISK = {
    "corporate": ("low", "corporate-backed — well-funded, roadmap driven by the backer"),
    "foundation": ("low", "foundation-backed — community-governed, long-term stability"),
    "vc_backed": ("medium", "VC-backed — rapid growth but potential pivot or acquisition risk"),
    "open_core": ("medium", "open-core model — free tier exists but key features may be commercial"),
    "community": ("medium", "community-maintained — no dedicated funding, depends on volunteer effort"),
}

# License type labels
_LICENSE_NOTES = {
    "permissive": "permissive (MIT / Apache / BSD) — use commercially without restriction",
    "copyleft": "copyleft (GPL family) — derivative works must also be open source; check your use case",
    "source-available": "source-available — can read the code but commercial use may be restricted; verify the license",
    "fair-code": "fair-code — free for small use, commercial license required at scale; check thresholds carefully",
}

# RAM tiers for human-readable framing
_RAM_BUCKETS = [
    (64,    "tiny — runs on a Raspberry Pi or shared hosting"),
    (256,   "very lightweight — fits in the smallest VPS"),
    (512,   "lightweight — comfortable on a 1 GB VPS"),
    (1024,  "moderate — standard 2 GB VPS is fine"),
    (2048,  "moderate — 2–4 GB VPS, typical for a dev server"),
    (4096,  "medium — 4–8 GB VPS is the practical minimum"),
    (8192,  "substantial — 8+ GB; dedicated server recommended"),
    (16384, "heavy — 16 GB+ RAM; plan your infrastructure accordingly"),
    (math.inf, "very heavy — 32+ GB; dedicated hardware or large cloud instance required"),
]


# ---------------------------------------------------------------------------
# Inline scoring helpers (mirrors server.py logic; used when scoring_engine
# is unavailable or does not expose explain_score())
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.sub(r'[^a-z0-9+# ]', '', text.lower()).split()


def _query_words(query: str) -> List[str]:
    return _tokenise(query)


def _tool_corpus(tool: Dict) -> Counter:
    """Reconstruct the term bag from a tool dict (mirrors _extract_terms)."""
    parts: List[str] = []
    for tag in tool.get("tags", []):
        parts.extend(tag.replace("-", " ").split() * 3)
    for d in tool.get("problem_domains", []):
        parts.extend(d.replace("-", " ").split() * 2)
    parts.extend(tool.get("tagline", "").lower().split())
    for uc in tool.get("use_cases_detailed", []):
        parts.extend(uc.lower().split())
    for r in tool.get("replaces", []):
        parts.extend(r.lower().split())
    parts.extend(tool.get("category", "").lower().split())
    parts.extend(tool.get("sub_category", "").lower().split())
    terms: Counter = Counter()
    for w in parts:
        w = re.sub(r'[^a-z0-9+#]', '', w)
        if len(w) > 1:
            terms[w] += 1
    return terms


def _score_breakdown(query: str, tool: Dict) -> Dict[str, Any]:
    """
    Compute a detailed score breakdown for (query, tool) inline.

    Returns a dict with:
        total          : float  — combined score
        exact_bonus    : float
        cosine         : float
        matched_tags   : List[str]  — tool tags/domains that matched query terms
        matched_terms  : List[Tuple[str, str]]  — (query_word, matching_tag_or_domain)
        category_hits  : List[str]
    """
    q_lower = query.lower().strip()
    q_words = _query_words(q_lower)
    tool_tags = set(tool.get("tags", []))
    tool_domains = set(tool.get("problem_domains", []))
    tool_all = tool_tags | tool_domains

    exact_bonus = 0.0
    matched_tags: List[str] = []
    matched_terms: List[Tuple[str, str]] = []

    for w in q_words:
        if w in tool_all:
            exact_bonus += 0.35
            matched_tags.append(w)
            matched_terms.append((w, w))
        else:
            for tag in tool_all:
                if w in tag or tag in w:
                    exact_bonus += 0.15
                    matched_terms.append((w, tag))
                    if tag not in matched_tags:
                        matched_tags.append(tag)
                    break

    q_hyphenated = q_lower.replace(" ", "-")
    for tag in tool_all:
        if q_hyphenated == tag or q_hyphenated in tag:
            exact_bonus += 0.5
            if tag not in matched_tags:
                matched_tags.append(tag)

    cat_words = (tool.get("category", "") + " " + tool.get("sub_category", "")).lower()
    category_hits: List[str] = []
    for w in q_words:
        if w in cat_words:
            exact_bonus += 0.1
            category_hits.append(w)

    # Lightweight cosine (no global IDF — use raw TF)
    q_terms = Counter(q_words)
    t_terms = _tool_corpus(tool)
    common = set(q_terms) & set(t_terms)
    dot = sum(q_terms[w] * t_terms[w] for w in common)
    mag_q = math.sqrt(sum(v ** 2 for v in q_terms.values())) or 1.0
    mag_t = math.sqrt(sum(v ** 2 for v in t_terms.values())) or 1.0
    cosine = dot / (mag_q * mag_t)

    return {
        "total": cosine + exact_bonus,
        "exact_bonus": exact_bonus,
        "cosine": cosine,
        "matched_tags": matched_tags,
        "matched_terms": matched_terms,
        "category_hits": category_hits,
    }


# ---------------------------------------------------------------------------
# Health scoring helper (used when health_scorer is unavailable)
# ---------------------------------------------------------------------------

def _health_score_inline(tool: Dict) -> Tuple[float, List[str], List[str]]:
    """
    Compute a health score 0–1 from the tool dict directly.

    Returns (score, strengths, concerns).
    """
    score = 0.5  # baseline
    strengths: List[str] = []
    concerns: List[str] = []

    stars = tool.get("github_stars", 0)
    if stars >= 20000:
        score += 0.15
        strengths.append(f"{stars:,} GitHub stars — widely adopted")
    elif stars >= 5000:
        score += 0.08
        strengths.append(f"{stars:,} GitHub stars — solid community")
    elif stars < 500 and stars > 0:
        score -= 0.08
        concerns.append(f"Only {stars:,} GitHub stars — limited community signal")

    contributors = tool.get("contributors_count", 0)
    if contributors >= 500:
        score += 0.1
        strengths.append(f"{contributors:,} contributors — broad contributor base")
    elif contributors >= 50:
        score += 0.05
    elif contributors < 10 and contributors > 0:
        score -= 0.05
        concerns.append(f"Only {contributors} contributors — bus-factor risk")

    commit_freq = tool.get("commit_frequency", "")
    if commit_freq == "daily":
        score += 0.1
        strengths.append("Daily commits — actively maintained")
    elif commit_freq == "weekly":
        score += 0.05
        strengths.append("Weekly commits — regularly maintained")
    elif commit_freq == "monthly":
        score -= 0.05
        concerns.append("Monthly commit cadence — slower pace of development")

    maturity = tool.get("maturity", "stable")
    if maturity == "mature":
        score += 0.05
        strengths.append("Mature project with long production track record")

    docs = tool.get("docs_quality", "")
    if docs == "excellent":
        score += 0.05
        strengths.append("Excellent documentation")
    elif docs in ("poor", ""):
        score -= 0.05
        concerns.append("Documentation quality is below average")

    funding_model = tool.get("funding_model", "community")
    risk_level, _ = _FUNDING_RISK.get(funding_model, ("medium", ""))
    if risk_level == "low":
        score += 0.05
    elif risk_level == "medium" and funding_model == "community":
        concerns.append("Community-only funding — no dedicated maintainers")

    release_year = tool.get("first_release_year", 0)
    if release_year and (2026 - release_year) >= 5:
        strengths.append(f"Battle-tested — in production since {release_year}")

    return min(1.0, max(0.0, score)), strengths, concerns


# ---------------------------------------------------------------------------
# Graph context helper
# ---------------------------------------------------------------------------

def _graph_summary(tool: Dict, slug_index: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Summarise graph connections from the tool dict.
    slug_index is optional; if provided, neighbour names are resolved.
    """
    def _names(slugs: List[str]) -> List[str]:
        if slug_index:
            return [slug_index[s]["name"] for s in slugs if s in slug_index]
        return slugs

    return {
        "integrates_with": _names(tool.get("integrates_with", [])),
        "complements": _names(tool.get("complements", [])),
        "similar_to": _names(tool.get("similar_to", [])),
        "conflicts_with": _names(tool.get("conflicts_with", [])),
    }


def _ram_label(ram_mb: int) -> str:
    for threshold, label in _RAM_BUCKETS:
        if ram_mb <= threshold:
            return label
    return "very heavy"


def _confidence_label(score: float) -> str:
    if score >= _CONFIDENCE_HIGH:
        return "HIGH"
    if score >= _CONFIDENCE_MED:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# RecommendationExplainer
# ---------------------------------------------------------------------------

class RecommendationExplainer:
    """
    Explains why a tool was (or was not) recommended.

    Accepts optional references to scoring_engine, graph_engine, and
    health_scorer. If those objects expose the expected methods, they are
    used. Otherwise, everything is computed inline from the tool dict.
    """

    def __init__(
        self,
        scoring_engine: Any = None,
        graph_engine: Any = None,
        health_scorer: Any = None,
        slug_index: Optional[Dict[str, Dict]] = None,
    ) -> None:
        self._scoring = scoring_engine
        self._graph = graph_engine
        self._health = health_scorer
        # slug_index lets graph summaries resolve names instead of slugs.
        # server.py populates SLUG_INDEX globally; callers may pass it in.
        self._slug_index = slug_index or {}

    # ------------------------------------------------------------------
    # Internal helpers that delegate to engines or fall back to inline
    # ------------------------------------------------------------------

    def _get_score_breakdown(self, query: str, tool: Dict) -> Dict[str, Any]:
        """Try scoring_engine.explain_score() first, fall back inline."""
        if self._scoring and hasattr(self._scoring, "explain_score"):
            try:
                return self._scoring.explain_score(query, tool.get("slug", ""))
            except Exception:
                pass
        return _score_breakdown(query, tool)

    def _get_graph(self, tool: Dict) -> Dict[str, Any]:
        """Try graph_engine.get_neighbors() first, fall back inline."""
        if self._graph and hasattr(self._graph, "get_neighbors"):
            try:
                return self._graph.get_neighbors(tool.get("slug", ""))
            except Exception:
                pass
        return _graph_summary(tool, self._slug_index)

    def _get_health(self, tool: Dict) -> Tuple[float, List[str], List[str]]:
        """Try health_scorer.score() first, fall back inline."""
        if self._health and hasattr(self._health, "score"):
            try:
                result = self._health.score(tool.get("slug", ""))
                # Normalise: expect (score, strengths, concerns) or dict
                if isinstance(result, tuple):
                    return result
                if isinstance(result, dict):
                    return (
                        result.get("score", 0.5),
                        result.get("strengths", []),
                        result.get("concerns", []),
                    )
            except Exception:
                pass
        return _health_score_inline(tool)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(self, query: str, slug: str, tool: Dict) -> str:
        """
        Generate a structured explanation of why a tool was recommended.

        Returns formatted markdown covering:
        - Overall relevance score and breakdown
        - Which query terms matched which tool attributes
        - Graph context (integrations, conflicts)
        - Health assessment
        - Anti-patterns to be aware of
        - Confidence level (HIGH / MEDIUM / LOW)
        """
        breakdown = self._get_score_breakdown(query, tool)
        graph = self._get_graph(tool)
        health_score, strengths, concerns = self._get_health(tool)

        total = breakdown.get("total", 0.0)
        exact = breakdown.get("exact_bonus", 0.0)
        cosine = breakdown.get("cosine", 0.0)
        matched_tags = breakdown.get("matched_tags", [])
        matched_terms = breakdown.get("matched_terms", [])
        category_hits = breakdown.get("category_hits", [])
        confidence = _confidence_label(total)

        name = tool.get("name", slug)
        tagline = tool.get("tagline", "")
        category = tool.get("category", "")
        sub_category = tool.get("sub_category", "")
        ram_mb = tool.get("min_ram_mb", 0)
        complexity = tool.get("complexity_level", "intermediate")
        license_type = tool.get("license_type", "")
        license_name = tool.get("license", "")
        maturity = tool.get("maturity", "stable")
        perf_tier = tool.get("performance_tier", "medium")
        anti_patterns = tool.get("anti_patterns", [])
        use_cases = tool.get("use_cases_detailed", [])
        funding_model = tool.get("funding_model", "community")
        tags = tool.get("tags", [])
        problem_domains = tool.get("problem_domains", [])

        lines: List[str] = []

        # Header
        lines.append(f"## Why {name}?")
        lines.append(f"*{tagline}*")
        lines.append("")

        # Confidence banner
        conf_context = {
            "HIGH": "Strong match. This tool is well-aligned to your query.",
            "MEDIUM": "Reasonable match. Worth evaluating — check the caveats below.",
            "LOW": "Weak match. This tool may not be what you need; review carefully.",
        }[confidence]
        lines.append(f"**Confidence: {confidence}** — {conf_context}")
        lines.append("")

        # Score breakdown
        lines.append("### Relevance Score Breakdown")
        lines.append(f"| Component | Score |")
        lines.append(f"|---|---|")
        lines.append(f"| Semantic similarity (TF-IDF cosine) | {cosine:.3f} |")
        lines.append(f"| Exact tag / domain match bonus | {exact:.3f} |")
        lines.append(f"| **Total relevance** | **{total:.3f}** |")
        lines.append("")

        # Term matching
        if matched_tags or matched_terms or category_hits:
            lines.append("### What Matched Your Query")
            if matched_tags:
                unique_tags = list(dict.fromkeys(matched_tags))  # preserve order, dedup
                lines.append(
                    f"- **Tags / domains matched**: `{'`, `'.join(unique_tags)}`"
                )
            if matched_terms:
                # Show the most informative non-trivial mappings
                informative = [
                    (qw, tm) for qw, tm in matched_terms if qw != tm
                ][:4]
                if informative:
                    term_strs = [f'"{qw}" → `{tm}`' for qw, tm in informative]
                    lines.append(f"- **Term mappings**: {', '.join(term_strs)}")
            if category_hits:
                lines.append(
                    f"- **Category match**: query terms `{', '.join(category_hits)}` "
                    f"hit the `{category} / {sub_category}` classification"
                )
            if not matched_tags and not category_hits:
                lines.append(
                    "- Matched via broad semantic similarity rather than exact tag overlap — "
                    "review use cases below to confirm fit"
                )
            lines.append("")
        else:
            lines.append("### What Matched Your Query")
            lines.append(
                "- Matched through semantic similarity only. No direct tag or domain overlap "
                "with your query terms — treat this as a lower-confidence suggestion."
            )
            lines.append("")

        # Technical fit
        lines.append("### Technical Profile")
        ram_desc = _ram_label(ram_mb)
        lines.append(f"- **RAM**: {ram_mb} MB — {ram_desc}")
        lines.append(
            f"- **Complexity**: {_CX_LABEL.get(complexity, complexity)} — "
            + {
                "beginner": "easy to get started, good docs, low ops overhead",
                "intermediate": "some setup required; expect a half-day to get productive",
                "advanced": "significant ops complexity; allocate time for tuning and monitoring",
                "expert": "steep learning curve; only choose this if you have the expertise in-house",
            }.get(complexity, "")
        )
        lines.append(f"- **Performance tier**: {_PERF_LABEL.get(perf_tier, perf_tier)}")
        lines.append(
            f"- **License**: {license_name} — {_LICENSE_NOTES.get(license_type, license_type)}"
        )
        lines.append(f"- **Maturity**: {maturity}")

        funding_risk, funding_desc = _FUNDING_RISK.get(
            funding_model, ("medium", f"{funding_model} funding model")
        )
        lines.append(f"- **Sustainability**: {funding_desc}")
        lines.append("")

        # Graph context
        lines.append("### Ecosystem Fit")
        integrates = graph.get("integrates_with", [])
        complements = graph.get("complements", [])
        conflicts = graph.get("conflicts_with", [])
        similar = graph.get("similar_to", [])

        if integrates:
            lines.append(f"- **Integrates with**: {', '.join(integrates[:8])}")
        if complements:
            lines.append(f"- **Works especially well with**: {', '.join(complements[:6])}")
        if similar:
            lines.append(f"- **Alternatives to compare**: {', '.join(similar[:5])}")
        if conflicts:
            lines.append(
                f"- **Avoid combining with**: {', '.join(conflicts)} "
                f"(direct overlap — pick one)"
            )
        if not integrates and not complements:
            lines.append(
                "- Limited explicit graph connections in the database. "
                "Check the project's docs for integration guides."
            )
        lines.append("")

        # Health
        health_pct = int(health_score * 100)
        health_label = (
            "Healthy project" if health_score >= 0.7
            else "Reasonable health" if health_score >= 0.5
            else "Some health concerns"
        )
        lines.append(f"### Community Health: {health_label} ({health_pct}/100)")
        for s in strengths[:4]:
            lines.append(f"- {s}")
        for c in concerns[:3]:
            lines.append(f"- **Concern**: {c}")
        lines.append("")

        # Use cases — confirm fit
        if use_cases:
            lines.append("### Confirmed Use Cases")
            for uc in use_cases[:4]:
                lines.append(f"- {uc}")
            lines.append("")

        # Anti-patterns — be honest
        if anti_patterns:
            lines.append("### When NOT to Use This")
            for ap in anti_patterns:
                lines.append(f"- {ap}")
            lines.append("")

        # Bottom line
        lines.append("### Bottom Line")
        if confidence == "HIGH":
            lines.append(
                f"{name} is a strong fit for \"{query}\". "
                f"The tag and domain overlap is direct, the project is {maturity}, "
                f"and the ecosystem connections support integration with adjacent tools."
            )
        elif confidence == "MEDIUM":
            lines.append(
                f"{name} covers part of what \"{query}\" requires. "
                f"Verify the specific use cases above map to your situation before committing. "
                f"Consider the anti-patterns — they are the most common reasons teams regret this choice."
            )
        else:
            lines.append(
                f"{name} is a weak semantic match for \"{query}\". "
                f"The scoring engine found some signal but no strong overlap. "
                f"It may still be the right answer if your query is unusual — "
                f"but compare it against alternatives before deciding."
            )

        return "\n".join(lines)

    def explain_stack(self, needs: List[str], stack: List[Dict]) -> str:
        """
        Explain why a particular stack was recommended.

        Returns a narrative covering:
        - Why each tool was chosen for its need
        - How tools work together (graph connections)
        - Overall stack health
        - Potential issues or gaps
        """
        if not stack:
            return "No stack to explain — the stack list is empty."

        lines: List[str] = []
        lines.append("## Stack Explanation")
        lines.append("")

        # Pair needs with tools (stack may be list of tool dicts or {need, tool} dicts)
        pairs: List[Tuple[str, Dict]] = []
        for i, item in enumerate(stack):
            if isinstance(item, dict) and "tool" in item:
                need = item.get("need", needs[i] if i < len(needs) else f"need-{i+1}")
                tool = item["tool"]
            else:
                need = needs[i] if i < len(needs) else f"need-{i+1}"
                tool = item
            pairs.append((need, tool))

        # Per-tool rationale
        lines.append("### Tool Selection Rationale")
        lines.append("")
        for need, tool in pairs:
            name = tool.get("name", tool.get("slug", "unknown"))
            tagline = tool.get("tagline", "")
            complexity = tool.get("complexity_level", "intermediate")
            ram_mb = tool.get("min_ram_mb", 0)
            license_type = tool.get("license_type", "")
            tags = tool.get("tags", [])
            domains = tool.get("problem_domains", [])

            # Find overlap between need and tool's vocabulary
            need_words = set(_query_words(need))
            tag_words = set()
            for tag in tags:
                tag_words.update(tag.replace("-", " ").split())
            for d in domains:
                tag_words.update(d.replace("-", " ").split())
            overlap = need_words & tag_words

            rationale = ""
            if overlap:
                rationale = (
                    f"Direct match on `{'`, `'.join(sorted(overlap))}`. "
                )
            rationale += (
                f"{_CX_LABEL.get(complexity, complexity).capitalize()} setup, "
                f"{ram_mb} MB RAM, {license_type} license."
            )

            lines.append(f"**{need.title()} → {name}**")
            lines.append(f"  *{tagline}*")
            lines.append(f"  {rationale}")
            lines.append("")

        # Cross-tool integration map
        lines.append("### How These Tools Work Together")
        lines.append("")
        tool_map = {t.get("slug", t.get("name", "")): t for _, t in pairs}
        tool_names = {t.get("slug", t.get("name", "")): t.get("name", "") for _, t in pairs}
        connections_found: List[str] = []
        conflicts_found: List[str] = []

        for slug_a, tool_a in tool_map.items():
            name_a = tool_a.get("name", slug_a)
            for slug_b, tool_b in tool_map.items():
                if slug_a >= slug_b:  # avoid duplicates
                    continue
                name_b = tool_b.get("name", slug_b)
                a_integrates = set(tool_a.get("integrates_with", []) + tool_a.get("complements", []))
                b_integrates = set(tool_b.get("integrates_with", []) + tool_b.get("complements", []))

                if slug_b in a_integrates or slug_a in b_integrates:
                    connections_found.append(f"- **{name_a}** + **{name_b}**: confirmed integration")
                if slug_b in set(tool_a.get("conflicts_with", [])):
                    conflicts_found.append(
                        f"- **{name_a}** conflicts with **{name_b}** — these solve the same problem; pick one"
                    )

        if connections_found:
            lines.append("Confirmed connections in this stack:")
            lines.extend(connections_found)
        else:
            lines.append(
                "No direct graph connections were found between the selected tools. "
                "This is not necessarily a problem — many tools coexist without explicit integration — "
                "but verify through each project's documentation."
            )
        lines.append("")

        if conflicts_found:
            lines.append("**Conflicts detected — action required:**")
            lines.extend(conflicts_found)
            lines.append("")

        # Stack health summary
        lines.append("### Overall Stack Health")
        health_scores: List[float] = []
        for _, tool in pairs:
            h, _, _ = self._get_health(tool)
            health_scores.append(h)

        avg_health = sum(health_scores) / len(health_scores) if health_scores else 0.0
        health_pct = int(avg_health * 100)
        health_label = (
            "Healthy" if avg_health >= 0.7
            else "Acceptable" if avg_health >= 0.5
            else "Needs review"
        )
        lines.append(f"Average community health: **{health_label}** ({health_pct}/100)")
        lines.append("")

        # RAM budget
        total_ram = sum(t.get("min_ram_mb", 0) for _, t in pairs)
        lines.append(f"Combined minimum RAM: **{total_ram} MB** ({total_ram / 1024:.1f} GB)")
        lines.append(
            "Add 25–50% overhead for OS, logs, and traffic spikes. "
            f"A {_recommended_vps_size(total_ram)} server is the practical minimum."
        )
        lines.append("")

        # Complexity spread
        cx_levels = [t.get("complexity_level", "intermediate") for _, t in pairs]
        max_cx = max(cx_levels, key=lambda c: _CX_ORDER.get(c, 1))
        lines.append(
            f"Complexity ceiling: **{_CX_LABEL.get(max_cx, max_cx)}** "
            f"(driven by {[t.get('name') for _, t in pairs if t.get('complexity_level') == max_cx][0]})"
        )
        lines.append("")

        # Gaps
        lines.append("### Potential Gaps and Issues")
        gaps = _identify_gaps(pairs)
        if gaps:
            for gap in gaps:
                lines.append(f"- {gap}")
        else:
            lines.append("- No obvious structural gaps identified. Validate your specific constraints.")
        lines.append("")

        return "\n".join(lines)

    def why_not(self, query: str, slug: str, tool: Dict) -> str:
        """
        Explain why a tool might NOT be the right choice.

        Surfaces anti-patterns relevant to the query, complexity warnings,
        RAM / resource concerns, license risks, and better alternatives.
        """
        name = tool.get("name", slug)
        tagline = tool.get("tagline", "")
        anti_patterns = tool.get("anti_patterns", [])
        similar_to = tool.get("similar_to", [])
        complexity = tool.get("complexity_level", "intermediate")
        ram_mb = tool.get("min_ram_mb", 0)
        license_type = tool.get("license_type", "")
        license_name = tool.get("license", "")
        performance_tier = tool.get("performance_tier", "medium")
        funding_model = tool.get("funding_model", "community")
        conflicts = tool.get("conflicts_with", [])
        team_fit = tool.get("team_size_fit", [])
        maturity = tool.get("maturity", "stable")

        # Score to assess fit
        breakdown = self._get_score_breakdown(query, tool)
        total = breakdown.get("total", 0.0)
        confidence = _confidence_label(total)

        lines: List[str] = []
        lines.append(f"## Why {name} Might Not Be Right for You")
        lines.append(f"*{tagline}*")
        lines.append("")

        if confidence == "HIGH":
            lines.append(
                f"**Note**: {name} is actually a strong match for \"{query}\" (score: {total:.3f}). "
                f"This `why_not` surfaces edge cases and risks — not a rejection."
            )
        elif confidence == "LOW":
            lines.append(
                f"**{name} scores low for \"{query}\"** ({total:.3f}) — there are likely better options."
            )
        lines.append("")

        reasons: List[str] = []

        # Anti-patterns — filter for query relevance
        query_words = set(_query_words(query))
        relevant_aps = []
        for ap in anti_patterns:
            ap_words = set(_query_words(ap))
            if query_words & ap_words or len(anti_patterns) <= 2:
                relevant_aps.append(ap)
        if not relevant_aps:
            relevant_aps = anti_patterns  # show all if none match

        if relevant_aps:
            lines.append("### Anti-Patterns for Your Query")
            for ap in relevant_aps:
                lines.append(f"- {ap}")
            lines.append("")

        # Complexity warning
        if _CX_ORDER.get(complexity, 1) >= 2:
            lines.append("### Complexity Warning")
            detail = {
                "advanced": (
                    f"{name} requires advanced operational skill. "
                    "Plan for a dedicated ops engineer or significant DevOps time. "
                    "Initial setup is rarely the hard part — ongoing tuning and incident response is."
                ),
                "expert": (
                    f"{name} is expert-level infrastructure. "
                    "Without in-house expertise, the learning curve will cost you more time "
                    "than the tool saves. Evaluate simpler alternatives first."
                ),
            }.get(complexity, "")
            if detail:
                lines.append(detail)
                lines.append("")
            reasons.append(f"complexity: {complexity}")

        # RAM concern
        if ram_mb >= 4096:
            lines.append("### Resource Requirements")
            lines.append(
                f"{name} needs at least {ram_mb} MB RAM — {_ram_label(ram_mb)}. "
                f"On a small VPS or constrained environment this will be a problem. "
                f"Factor in the hosting cost before committing."
            )
            lines.append("")
            reasons.append(f"RAM: {ram_mb} MB minimum")

        # License risk
        if license_type in ("copyleft", "source-available", "fair-code"):
            lines.append("### License Risk")
            lines.append(_LICENSE_NOTES.get(license_type, license_type))
            if license_type == "copyleft":
                lines.append(
                    f"If you are building proprietary software, check whether your use "
                    f"of {name} ({license_name}) triggers copyleft requirements. Get legal clarity before production use."
                )
            elif license_type in ("source-available", "fair-code"):
                lines.append(
                    f"Commercial restrictions may apply at scale. "
                    f"Review the {license_name} license carefully before building a product around {name}."
                )
            lines.append("")
            reasons.append(f"license: {license_type}")

        # Funding / sustainability concern
        if funding_model == "community":
            lines.append("### Sustainability Concern")
            lines.append(
                f"{name} is community-maintained with no dedicated funding. "
                "This is fine for many use cases but means: slower issue resolution, "
                "potential for the project to go dormant, and no SLA. "
                "If this is business-critical infrastructure, plan a fork-and-maintain strategy."
            )
            lines.append("")
            reasons.append("community-only funding")
        elif funding_model == "vc_backed":
            lines.append("### Business Risk")
            lines.append(
                f"{name} is VC-backed. That means fast development today but potential "
                "for pivot, acquisition, or licence change tomorrow. "
                "Mirror your deployment setup so you can swap it out if the business model changes."
            )
            lines.append("")

        # Performance tier mismatch
        if performance_tier == "heavy" and "lightweight" in query.lower():
            lines.append("### Performance Tier Mismatch")
            lines.append(
                f"Your query mentions lightweight constraints, but {name} is classified as "
                f"resource-heavy. Consider lighter alternatives before committing."
            )
            lines.append("")
            reasons.append("heavy resource footprint vs lightweight query intent")

        # Better alternatives
        if similar_to:
            lines.append("### Better Alternatives for This Query")
            alt_scores: List[Tuple[str, float]] = []
            for alt_slug in similar_to:
                if self._slug_index and alt_slug in self._slug_index:
                    alt_tool = self._slug_index[alt_slug]
                    alt_bd = self._get_score_breakdown(query, alt_tool)
                    alt_scores.append((alt_slug, alt_bd.get("total", 0.0), alt_tool))
                else:
                    alt_scores.append((alt_slug, 0.0, {"name": alt_slug}))

            # Sort by descending score
            alt_scores_sorted = sorted(alt_scores, key=lambda x: -x[1])

            for alt_slug, alt_score, alt_tool in alt_scores_sorted[:4]:
                alt_name = alt_tool.get("name", alt_slug)
                alt_tagline = alt_tool.get("tagline", "")
                score_note = f" (relevance: {alt_score:.3f})" if alt_score > 0 else ""
                lines.append(f"- **{alt_name}**{score_note}: {alt_tagline}")
            lines.append("")

        # Summary
        lines.append("### Summary")
        if not reasons and confidence == "HIGH":
            lines.append(
                f"{name} is a solid choice for \"{query}\". "
                f"The concerns above are minor. Use this section as a checklist, not a veto."
            )
        elif reasons:
            joined = "; ".join(reasons)
            lines.append(
                f"Key risk factors for \"{query}\": {joined}. "
                f"These are not automatic disqualifiers, but weigh them honestly "
                f"against your team's constraints before committing."
            )
        else:
            lines.append(
                f"{name} has no severe red flags for \"{query}\", "
                f"but the match score is {confidence.lower()}. "
                f"Explore alternatives before deciding."
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private utilities used by explain_stack
# ---------------------------------------------------------------------------

def _recommended_vps_size(ram_mb: int) -> str:
    """Return a human-readable VPS size recommendation."""
    if ram_mb <= 512:
        return "1 GB"
    if ram_mb <= 1536:
        return "2 GB"
    if ram_mb <= 3072:
        return "4 GB"
    if ram_mb <= 6144:
        return "8 GB"
    if ram_mb <= 12288:
        return "16 GB"
    if ram_mb <= 24576:
        return "32 GB"
    return "64 GB+"


def _identify_gaps(pairs: List[Tuple[str, Dict]]) -> List[str]:
    """
    Heuristic gap analysis for a stack: look for missing common layers.

    Checks for observability, reverse proxy, TLS/auth, backup strategy.
    """
    gaps: List[str] = []
    all_tags: set = set()
    all_domains: set = set()
    all_layers: set = set()
    all_categories: set = set()

    for _, tool in pairs:
        all_tags.update(tool.get("tags", []))
        all_domains.update(tool.get("problem_domains", []))
        all_layers.update(tool.get("stack_layer", []))
        all_categories.add(tool.get("category", "").lower())

    # Observability
    obs_signals = {"monitoring", "metrics", "observability", "tracing", "logging", "prometheus", "grafana"}
    if not (all_tags & obs_signals or all_domains & obs_signals):
        gaps.append(
            "No observability layer detected. Add Prometheus + Grafana or an equivalent "
            "before going to production — you need to see what is happening inside your stack."
        )

    # Reverse proxy / ingress
    proxy_signals = {"reverse-proxy", "ingress", "load-balancer", "nginx", "traefik", "caddy"}
    if not (all_tags & proxy_signals or all_domains & proxy_signals):
        gaps.append(
            "No reverse proxy or ingress controller in the stack. "
            "Consider Caddy (automatic HTTPS) or Nginx Proxy Manager for TLS termination and routing."
        )

    # Auth — if the stack has user-facing tools but no auth layer
    user_facing_cats = {"crm & erp", "communication", "project management", "analytics"}
    auth_signals = {"authentication", "auth", "sso", "oauth", "oidc", "identity"}
    if user_facing_cats & all_categories and not (all_tags & auth_signals or all_domains & auth_signals):
        gaps.append(
            "User-facing tools in the stack but no dedicated auth / SSO layer. "
            "Consider Authentik or Keycloak to centralise identity management."
        )

    # Backup
    backup_signals = {"backup", "restore", "disaster-recovery"}
    if not (all_tags & backup_signals or all_domains & backup_signals):
        gaps.append(
            "No backup solution included. Self-hosted stacks need an explicit backup strategy. "
            "At minimum, automate database dumps + off-site storage (Restic + B2 or S3)."
        )

    return gaps
