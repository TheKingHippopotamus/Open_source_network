"""
OSS Neural Match — Health Scorer
=================================
Computes a multi-dimensional health score for each open-source tool using
only the fields present in db.json.  No external API calls are made.

Scoring formula (weighted sum of six dimensions):

    overall = 0.25*activity + 0.20*community + 0.20*maturity
            + 0.15*backing  + 0.10*license   + 0.10*documentation

Each dimension score is a float in [0.0, 1.0].

Public API
----------
    scorer = HealthScorer(tools)          # pre-computes all scores at init
    scorer.score("tensorflow")            # -> full score dict for one tool
    scorer.compare_health(["a", "b"])     # -> markdown comparison table
    scorer.get_risk_tools(threshold=0.4)  # -> tools below threshold
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Dimension weights — must sum to 1.0
WEIGHTS: Dict[str, float] = {
    "activity":      0.25,
    "community":     0.20,
    "maturity":      0.20,
    "backing":       0.15,
    "license":       0.10,
    "documentation": 0.10,
}

# Grade thresholds (inclusive lower bound)
GRADE_THRESHOLDS: List[tuple] = [
    (0.85, "A"),
    (0.70, "B"),
    (0.50, "C"),
    (0.30, "D"),
    (0.00, "F"),
]

# Risk band thresholds (inclusive lower bound)
RISK_BANDS: List[tuple] = [
    (0.70, "green"),
    (0.50, "yellow"),
    (0.30, "orange"),
    (0.00, "red"),
]

# Major corporate backers known for sustained OSS investment
MAJOR_CORPS = {
    "google", "meta", "microsoft", "amazon", "aws", "apple", "ibm",
    "oracle", "intel", "nvidia", "netflix", "stripe", "salesforce",
    "databricks", "snowflake", "cloudflare", "elastic", "mongodb",
    "hashicorp", "redhat", "red hat", "vmware", "broadcom",
}

# Foundation keywords indicating nonprofit stewardship
FOUNDATION_KEYWORDS = {
    "foundation", "apache", "linux", "cncf", "openjs", "gnome",
    "eclipse", "fsf", "software freedom", "asf",
}

# Reference date for computing time-based scores.
# Falls back to datetime.utcnow() if the import fails in unusual environments.
_REFERENCE_DATE = datetime(2026, 3, 1)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp `value` to [lo, hi]."""
    return max(lo, min(hi, value))


def _log_normalise(value: float, min_val: float, max_val: float) -> float:
    """
    Logarithmic normalisation to [0, 1].

    Maps `value` to the interval using log1p so that large differences at
    the top of the range compress nicely (e.g. 1 000 vs 10 000 stars does
    not dominate 1 000 vs 2 000 stars).

    Returns 0.0 when max_val == min_val or value <= 0.
    """
    if max_val <= min_val or value <= 0:
        return 0.0
    log_val = math.log1p(max(0.0, value - min_val))
    log_max = math.log1p(max_val - min_val)
    return _clamp(log_val / log_max if log_max > 0 else 0.0)


def _months_since(date_str: str) -> Optional[float]:
    """
    Parse a 'YYYY-MM' date string and return the number of months elapsed
    since _REFERENCE_DATE.  Returns None if the string is empty or malformed.
    """
    if not date_str or len(date_str) < 7:
        return None
    try:
        year = int(date_str[:4])
        month = int(date_str[5:7])
        release = datetime(year, month, 1)
        delta = _REFERENCE_DATE - release
        return delta.days / 30.44  # average days per month
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# DIMENSION SCORERS
# ---------------------------------------------------------------------------

def _score_activity(tool: Dict) -> tuple[float, list]:
    """
    Activity dimension — measures how actively the project is maintained.

    Factors (scored individually, then averaged):
      1. commit_frequency  — cadence of commits (daily/weekly/monthly/quarterly)
      2. last_release_date — recency of last published release (sparse field)
      3. latest_version    — whether a versioned release exists at all (sparse)

    Because last_release_date and latest_version are populated for only ~4
    of 244 tools, they are treated as bonus evidence rather than mandatory
    signals.  When absent the factor is excluded from the average so the
    commit_frequency alone can still produce a full-range activity score.
    """
    factors: list = []
    scores: list = []

    # --- commit_frequency ---
    freq_map = {
        "daily":     1.0,
        "weekly":    0.8,
        "monthly":   0.5,
        "quarterly": 0.3,
        "yearly":    0.1,
    }
    freq = tool.get("commit_frequency", "")
    freq_score = freq_map.get(freq.lower(), 0.2) if freq else 0.2
    factors.append({"name": "commit_frequency", "value": freq or "unknown", "score": freq_score})
    scores.append(freq_score)

    # --- last_release_date (optional — only 4 tools in dataset have this) ---
    last_release = tool.get("last_release_date", "")
    months_ago = _months_since(last_release)
    if months_ago is not None:
        if months_ago <= 6:
            release_score = 1.0
        elif months_ago <= 12:
            release_score = 0.7
        elif months_ago <= 24:
            release_score = 0.4
        else:
            release_score = 0.1
        factors.append({
            "name":  "last_release_date",
            "value": last_release,
            "score": release_score,
        })
        scores.append(release_score)

    # --- latest_version (optional — bonus for explicit versioning) ---
    version = tool.get("latest_version", "")
    if version:
        version_score = 0.5
        factors.append({"name": "latest_version", "value": version, "score": version_score})
        # Version presence is a partial bonus — weight it at 0.5 relative to others
        scores.append(version_score * 0.5)

    if not scores:
        return 0.2, factors

    activity_score = _clamp(sum(scores) / len(scores) if scores else 0.0)
    return activity_score, factors


def _score_community(
    tool: Dict,
    global_min_stars: float,
    global_max_stars: float,
) -> tuple[float, list]:
    """
    Community dimension — size and health of the contributor ecosystem.

    Factors:
      1. github_stars       — log-normalised against global min/max in dataset
      2. contributors_count — stepped scoring (0 treated as unknown -> 0.2)
      3. plugin_ecosystem   — qualitative breadth of integrations
    """
    factors: list = []

    # --- github_stars ---
    stars = tool.get("github_stars", 0) or 0
    stars_score = _log_normalise(float(stars), global_min_stars, global_max_stars)
    factors.append({"name": "github_stars", "value": stars, "score": round(stars_score, 4)})

    # --- contributors_count ---
    contribs = tool.get("contributors_count", 0) or 0
    if contribs <= 0:
        contrib_score = 0.2      # unknown — give benefit of the doubt
    elif contribs >= 100:
        contrib_score = 1.0
    elif contribs >= 50:
        contrib_score = 0.7
    elif contribs >= 20:
        contrib_score = 0.5
    elif contribs >= 5:
        contrib_score = 0.3
    else:
        contrib_score = 0.1
    factors.append({"name": "contributors_count", "value": contribs, "score": contrib_score})

    # --- plugin_ecosystem ---
    ecosystem_map = {
        "massive": 1.0,
        "large":   1.0,
        "medium":  0.6,
        "small":   0.3,
        "none":    0.1,
    }
    ecosystem = tool.get("plugin_ecosystem", "")
    ecosystem_score = ecosystem_map.get(ecosystem.lower(), 0.3) if ecosystem else 0.3
    factors.append({
        "name":  "plugin_ecosystem",
        "value": ecosystem or "unknown",
        "score": ecosystem_score,
    })

    community_score = _clamp((stars_score + contrib_score + ecosystem_score) / 3.0)
    return community_score, factors


def _score_maturity(tool: Dict) -> tuple[float, list]:
    """
    Maturity dimension — project age, stability, and lifecycle stage.

    Factors:
      1. maturity field     — explicit maturity label
      2. first_release_year — project age; older generally == more battle-tested
      3. experimental_age_penalty — experimental projects that are also old
                                    may be abandoned (penalised)
    """
    factors: list = []

    # --- maturity label ---
    maturity_map = {
        "mature":       1.0,
        "stable":       0.7,
        "growing":      0.5,
        "experimental": 0.2,
    }
    maturity_label = tool.get("maturity", "")
    maturity_score = maturity_map.get(maturity_label.lower(), 0.4) if maturity_label else 0.4
    factors.append({
        "name":  "maturity",
        "value": maturity_label or "unknown",
        "score": maturity_score,
    })

    # --- first_release_year ---
    release_year = tool.get("first_release_year", 0) or 0
    current_year = _REFERENCE_DATE.year
    age = current_year - release_year if release_year > 0 else 0

    if age <= 0:
        age_score = 0.2
    elif release_year >= 2020:
        age_score = 0.3
    elif release_year >= 2015:
        age_score = 0.6
    elif release_year >= 2010:
        age_score = 0.8
    else:
        age_score = 1.0

    factors.append({
        "name":  "first_release_year",
        "value": release_year or "unknown",
        "score": age_score,
    })

    # --- experimental + old project penalty (possible abandonment signal) ---
    penalty = 0.0
    if maturity_label.lower() == "experimental" and age >= 5:
        penalty = 0.15
        factors.append({
            "name":  "stale_experimental_penalty",
            "value": f"experimental project aged {age} years",
            "score": -penalty,
        })

    maturity_score_final = _clamp((maturity_score + age_score) / 2.0 - penalty)
    return maturity_score_final, factors


def _score_backing(tool: Dict) -> tuple[float, list]:
    """
    Backing dimension — financial and organisational sustainability.

    Factors:
      1. backing_org   — heuristic classification: major corp / foundation /
                         startup-with-name / pure community
      2. funding_model — explicit funding model label
    """
    factors: list = []

    # --- backing_org classification ---
    backing_raw = tool.get("backing_org", "") or ""
    backing_lower = backing_raw.lower()

    # Check for major corporate backer (any of the well-known names appear)
    is_major_corp = any(corp in backing_lower for corp in MAJOR_CORPS)
    is_foundation = any(kw in backing_lower for kw in FOUNDATION_KEYWORDS)
    is_pure_community = backing_lower in {"community", ""}

    if is_major_corp:
        org_score = 1.0
        org_label = "major_corp"
    elif is_foundation:
        org_score = 0.8
        org_label = "foundation"
    elif is_pure_community:
        org_score = 0.3
        org_label = "community"
    else:
        # Named startup, indie developer, or known-but-smaller company
        org_score = 0.5
        org_label = "startup_or_indie"

    factors.append({
        "name":  "backing_org",
        "value": backing_raw or "unknown",
        "score": org_score,
        "classification": org_label,
    })

    # --- funding_model ---
    funding_map = {
        "corporate":  0.9,
        "foundation": 0.8,
        "vc_backed":  0.6,
        "open_core":  0.5,
        "community":  0.3,
    }
    funding = tool.get("funding_model", "") or ""
    funding_score = funding_map.get(funding.lower(), 0.3) if funding else 0.3
    factors.append({
        "name":  "funding_model",
        "value": funding or "unknown",
        "score": funding_score,
    })

    backing_score = _clamp((org_score + funding_score) / 2.0)
    return backing_score, factors


def _score_license(tool: Dict) -> tuple[float, list]:
    """
    License dimension — openness and commercial adoption risk.

    Factors:
      1. license_type      — permissive / copyleft / source-available / fair-code
      2. vendor_lockin_risk — none / low / medium / high
    """
    factors: list = []

    # --- license_type ---
    license_map = {
        "permissive":       1.0,
        "copyleft":         0.7,
        "fair-code":        0.6,
        "source-available": 0.5,
    }
    lic_type = tool.get("license_type", "") or ""
    lic_score = license_map.get(lic_type.lower(), 0.5) if lic_type else 0.5
    factors.append({
        "name":  "license_type",
        "value": lic_type or "unknown",
        "score": lic_score,
    })

    # --- vendor_lockin_risk ---
    lockin_map = {
        "none":   1.0,
        "low":    0.8,
        "medium": 0.5,
        "high":   0.2,
    }
    lockin = tool.get("vendor_lockin_risk", "") or ""
    lockin_score = lockin_map.get(lockin.lower(), 0.8) if lockin else 0.8
    factors.append({
        "name":  "vendor_lockin_risk",
        "value": lockin or "unknown",
        "score": lockin_score,
    })

    license_score = _clamp((lic_score + lockin_score) / 2.0)
    return license_score, factors


def _score_documentation(tool: Dict) -> tuple[float, list]:
    """
    Documentation dimension — quality of official docs.

    Factor:
      1. docs_quality — excellent / good / fair / poor
    """
    factors: list = []

    docs_map = {
        "excellent": 1.0,
        "good":      0.7,
        "fair":      0.4,
        "poor":      0.1,
    }
    docs = tool.get("docs_quality", "") or ""
    docs_score = docs_map.get(docs.lower(), 0.4) if docs else 0.4
    factors.append({
        "name":  "docs_quality",
        "value": docs or "unknown",
        "score": docs_score,
    })

    return docs_score, factors


# ---------------------------------------------------------------------------
# CLASSIFICATION HELPERS
# ---------------------------------------------------------------------------

def _assign_grade(overall: float) -> str:
    """Return letter grade for an overall score."""
    for threshold, grade in GRADE_THRESHOLDS:
        if overall >= threshold:
            return grade
    return "F"


def _assign_risk_band(overall: float) -> str:
    """Return risk band for an overall score."""
    for threshold, band in RISK_BANDS:
        if overall >= threshold:
            return band
    return "red"


def _build_summary(tool: Dict, overall: float, grade: str, risk_band: str) -> str:
    """Return a one-line human-readable health summary."""
    name = tool.get("name", tool.get("slug", "Unknown"))
    maturity = tool.get("maturity", "").capitalize() or "Unknown maturity"
    backing = tool.get("backing_org", "") or "community"
    pct = round(overall * 100)
    return (
        f"{name} scores {pct}% (grade {grade}, {risk_band} band): "
        f"{maturity} project backed by {backing}."
    )


# ---------------------------------------------------------------------------
# HEALTH SCORER
# ---------------------------------------------------------------------------

class HealthScorer:
    """
    Pre-computes and caches OSS health scores for all tools at construction
    time so that repeated calls to ``score()`` are O(1) lookups.

    Parameters
    ----------
    tools : List[Dict]
        Full tool records loaded from db.json.  Each record must contain
        at minimum a ``"slug"`` key.  All other scoring fields are optional
        and degrade gracefully when absent.
    """

    def __init__(self, tools: List[Dict]) -> None:
        if not tools:
            raise ValueError("HealthScorer requires at least one tool record.")

        self._tools: Dict[str, Dict] = {t["slug"]: t for t in tools}

        # Pre-compute global star range for log-normalisation
        all_stars = [float(t.get("github_stars", 0) or 0) for t in tools]
        self._min_stars = min(s for s in all_stars if s > 0) if any(s > 0 for s in all_stars) else 0.0
        self._max_stars = max(all_stars) if all_stars else 1.0

        # Pre-compute and cache all scores
        self._scores: Dict[str, Dict] = {}
        for tool in tools:
            slug = tool["slug"]
            self._scores[slug] = self._compute(tool)

    # -----------------------------------------------------------------------
    # INTERNAL: single-tool computation
    # -----------------------------------------------------------------------

    def _compute(self, tool: Dict) -> Dict:
        """
        Compute the full health score dict for a single tool record.

        This is called once per tool at construction time and the result is
        cached.  External callers should use ``score()`` instead.
        """
        activity_score,   activity_factors   = _score_activity(tool)
        community_score,  community_factors  = _score_community(
            tool, self._min_stars, self._max_stars
        )
        maturity_score,   maturity_factors   = _score_maturity(tool)
        backing_score,    backing_factors     = _score_backing(tool)
        license_score,    license_factors     = _score_license(tool)
        docs_score,       docs_factors        = _score_documentation(tool)

        overall = _clamp(
            WEIGHTS["activity"]      * activity_score
            + WEIGHTS["community"]   * community_score
            + WEIGHTS["maturity"]    * maturity_score
            + WEIGHTS["backing"]     * backing_score
            + WEIGHTS["license"]     * license_score
            + WEIGHTS["documentation"] * docs_score
        )

        grade     = _assign_grade(overall)
        risk_band = _assign_risk_band(overall)
        summary   = _build_summary(tool, overall, grade, risk_band)

        return {
            "slug":      tool["slug"],
            "overall":   round(overall, 4),
            "grade":     grade,
            "risk_band": risk_band,
            "dimensions": {
                "activity": {
                    "score":   round(activity_score, 4),
                    "weight":  WEIGHTS["activity"],
                    "factors": activity_factors,
                },
                "community": {
                    "score":   round(community_score, 4),
                    "weight":  WEIGHTS["community"],
                    "factors": community_factors,
                },
                "maturity": {
                    "score":   round(maturity_score, 4),
                    "weight":  WEIGHTS["maturity"],
                    "factors": maturity_factors,
                },
                "backing": {
                    "score":   round(backing_score, 4),
                    "weight":  WEIGHTS["backing"],
                    "factors": backing_factors,
                },
                "license": {
                    "score":   round(license_score, 4),
                    "weight":  WEIGHTS["license"],
                    "factors": license_factors,
                },
                "documentation": {
                    "score":   round(docs_score, 4),
                    "weight":  WEIGHTS["documentation"],
                    "factors": docs_factors,
                },
            },
            "summary": summary,
        }

    # -----------------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------------

    def score(self, slug: str) -> Dict:
        """
        Return the health score dict for a tool identified by ``slug``.

        Returns
        -------
        Dict with keys:
          slug       : str
          overall    : float  (0.0–1.0)
          grade      : str    ('A', 'B', 'C', 'D', 'F')
          risk_band  : str    ('green', 'yellow', 'orange', 'red')
          dimensions : dict   (per-dimension score + factors)
          summary    : str    (one-line human-readable summary)

        Raises
        ------
        KeyError
            If ``slug`` is not found in the loaded tool database.
        """
        if slug not in self._scores:
            raise KeyError(f"Tool slug '{slug}' not found in health scorer index.")
        return self._scores[slug]

    def compare_health(self, slugs: List[str]) -> str:
        """
        Compare health scores of multiple tools.

        Returns a formatted Markdown table with one row per tool, sorted by
        overall score descending.  Unknown slugs are silently skipped.

        Parameters
        ----------
        slugs : List[str]
            Tool slugs to compare.  Duplicates are de-duplicated.

        Returns
        -------
        str
            Markdown-formatted comparison table.
        """
        seen: set = set()
        rows: List[Dict] = []
        for slug in slugs:
            if slug in seen:
                continue
            seen.add(slug)
            if slug not in self._scores:
                continue
            rows.append(self._scores[slug])

        if not rows:
            return "_No valid tool slugs provided for comparison._"

        rows.sort(key=lambda r: r["overall"], reverse=True)

        # Header
        lines = [
            "| Rank | Tool | Overall | Grade | Risk | Activity | Community | Maturity | Backing | License | Docs |",
            "|------|------|---------|-------|------|----------|-----------|----------|---------|---------|------|",
        ]

        for rank, row in enumerate(rows, start=1):
            d = row["dimensions"]
            lines.append(
                f"| {rank} "
                f"| {row['slug']} "
                f"| {row['overall']:.2f} "
                f"| {row['grade']} "
                f"| {row['risk_band']} "
                f"| {d['activity']['score']:.2f} "
                f"| {d['community']['score']:.2f} "
                f"| {d['maturity']['score']:.2f} "
                f"| {d['backing']['score']:.2f} "
                f"| {d['license']['score']:.2f} "
                f"| {d['documentation']['score']:.2f} |"
            )

        # Summary section
        lines.append("")
        lines.append("### Summaries")
        lines.append("")
        for row in rows:
            lines.append(f"- **{row['slug']}**: {row['summary']}")

        return "\n".join(lines)

    def get_risk_tools(self, threshold: float = 0.4) -> List[Dict]:
        """
        Return all tools whose overall health score falls below ``threshold``.

        Results are sorted by overall score ascending (lowest / highest-risk
        tools first).

        Parameters
        ----------
        threshold : float
            Score boundary.  Tools with ``overall < threshold`` are returned.
            Defaults to 0.4 (orange/red band boundary).

        Returns
        -------
        List[Dict]
            List of full health score dicts, sorted ascending by overall.
        """
        risky = [s for s in self._scores.values() if s["overall"] < threshold]
        risky.sort(key=lambda r: r["overall"])
        return risky

    def all_scores(self) -> List[Dict]:
        """
        Return health score dicts for every tool, sorted by overall score
        descending.

        Useful for building leaderboards or bulk exports.
        """
        return sorted(self._scores.values(), key=lambda r: r["overall"], reverse=True)
