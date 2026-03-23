#!/usr/bin/env python3
"""
Generate stack recipes — pre-built tool combinations for common use cases.

Each recipe declares a set of "needs" (semantic queries) plus constraints
(max RAM, team size, max complexity). The same scoring algorithm used by
oss_find_stack in server.py resolves the best tool for each need.

Output: website/src/data/stack_recipes.json
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "db.json"
OUTPUT_PATH = REPO_ROOT / "website" / "src" / "data" / "stack_recipes.json"

# ---------------------------------------------------------------------------
# Recipe definitions
# ---------------------------------------------------------------------------
RECIPES: list[dict] = [
    {
        "id": "self-hosted-crm",
        "name": "Self-Hosted CRM Stack",
        "description": "Complete customer relationship management for small businesses — contacts, email campaigns, and analytics under one roof.",
        "needs": ["crm", "email-marketing", "analytics", "authentication"],
        "audience": "Small business owners and startups",
        "use_case": "Manage customer relationships without SaaS lock-in",
        "max_ram_mb": 4096,
        "team_size": "small",
        "max_complexity": "intermediate",
    },
    {
        "id": "ai-ml-platform",
        "name": "AI/ML Development Platform",
        "description": "Full stack for training, tracking, and serving ML models — from experiment to production.",
        "needs": ["ml-framework", "vector-database", "model-serving", "monitoring"],
        "audience": "Data scientists and ML engineers",
        "use_case": "Train, version, and deploy machine-learning models at scale",
        "max_ram_mb": 16384,
        "team_size": "medium",
        "max_complexity": "advanced",
    },
    {
        "id": "devops-pipeline",
        "name": "DevOps Automation Pipeline",
        "description": "CI/CD, container orchestration, secrets management, and observability for modern engineering teams.",
        "needs": ["ci-cd", "container-orchestration", "secrets-management", "metrics-monitoring"],
        "audience": "DevOps and platform engineers",
        "use_case": "Automate build-test-deploy with full observability",
        "max_ram_mb": 8192,
        "team_size": "medium",
        "max_complexity": "advanced",
    },
    {
        "id": "self-hosted-productivity",
        "name": "Self-Hosted Productivity Suite",
        "description": "Documents, project management, communication, and file storage — fully self-hosted.",
        "needs": ["document-collaboration", "project-management", "team-chat", "file-storage"],
        "audience": "Remote teams and privacy-conscious organisations",
        "use_case": "Replace Google Workspace / Microsoft 365 with owned infrastructure",
        "max_ram_mb": 6144,
        "team_size": "small",
        "max_complexity": "intermediate",
    },
    {
        "id": "ecommerce-stack",
        "name": "E-Commerce Stack",
        "description": "Headless commerce engine, payments, search, and analytics for online stores.",
        "needs": ["e-commerce", "search", "analytics", "authentication"],
        "audience": "E-commerce teams and digital agencies",
        "use_case": "Launch and scale an online store without proprietary SaaS fees",
        "max_ram_mb": 4096,
        "team_size": "small",
        "max_complexity": "intermediate",
    },
    {
        "id": "content-publishing",
        "name": "Content Publishing Platform",
        "description": "Headless CMS, media management, search, and CDN-ready image processing.",
        "needs": ["cms", "media-management", "search", "authentication"],
        "audience": "Content teams, media companies, and digital publishers",
        "use_case": "Publish content at scale with full editorial control",
        "max_ram_mb": 4096,
        "team_size": "small",
        "max_complexity": "intermediate",
    },
    {
        "id": "data-engineering-pipeline",
        "name": "Data Engineering Pipeline",
        "description": "Workflow orchestration, columnar storage, data transformation, and BI dashboard.",
        "needs": ["workflow-orchestration", "data-warehouse", "data-transformation", "business-intelligence"],
        "audience": "Data engineers and analytics teams",
        "use_case": "Build reliable ELT pipelines feeding a self-hosted analytics layer",
        "max_ram_mb": 16384,
        "team_size": "medium",
        "max_complexity": "advanced",
    },
    {
        "id": "monitoring-observability",
        "name": "Monitoring & Observability Stack",
        "description": "Metrics, logs, distributed traces, and alerting — the full three-pillars observability setup.",
        "needs": ["metrics-monitoring", "log-management", "distributed-tracing", "alerting"],
        "audience": "SRE teams and platform engineers",
        "use_case": "Full-stack observability with unified dashboards and on-call alerts",
        "max_ram_mb": 8192,
        "team_size": "medium",
        "max_complexity": "advanced",
    },
    {
        "id": "internal-tools-platform",
        "name": "Internal Tools Platform",
        "description": "Low-code UI builder, automation engine, database, and authentication for internal apps.",
        "needs": ["low-code", "workflow-automation", "database", "authentication"],
        "audience": "Engineering and operations teams building internal tools",
        "use_case": "Ship admin dashboards, approval workflows, and CRUD apps in days",
        "max_ram_mb": 4096,
        "team_size": "small",
        "max_complexity": "intermediate",
    },
    {
        "id": "self-hosted-communication",
        "name": "Self-Hosted Communication Hub",
        "description": "Team messaging, video calls, and transactional email — no data leaving your servers.",
        "needs": ["team-chat", "video-conferencing", "transactional-email", "authentication"],
        "audience": "Privacy-first teams and regulated industries",
        "use_case": "Replace Slack, Zoom, and SendGrid with owned communication infrastructure",
        "max_ram_mb": 6144,
        "team_size": "small",
        "max_complexity": "intermediate",
    },
    {
        "id": "knowledge-management",
        "name": "Knowledge Management System",
        "description": "Wiki, search, AI-powered Q&A, and document storage for team knowledge.",
        "needs": ["wiki", "search", "document-storage", "authentication"],
        "audience": "Engineering teams and knowledge-intensive organisations",
        "use_case": "Centralise institutional knowledge with semantic search and AI retrieval",
        "max_ram_mb": 4096,
        "team_size": "small",
        "max_complexity": "intermediate",
    },
    {
        "id": "iot-platform",
        "name": "IoT Data Platform",
        "description": "Message broker, time-series database, dashboards, and automation rules for IoT fleets.",
        "needs": ["message-broker", "time-series-database", "dashboard", "automation"],
        "audience": "IoT developers and industrial engineers",
        "use_case": "Collect, store, visualise, and act on sensor data streams",
        "max_ram_mb": 4096,
        "team_size": "small",
        "max_complexity": "intermediate",
    },
    {
        "id": "security-platform",
        "name": "Security & Identity Platform",
        "description": "Identity provider, secrets vault, vulnerability scanning, and audit logging.",
        "needs": ["identity-provider", "secrets-management", "vulnerability-scanning", "audit-logging"],
        "audience": "Security teams and compliance-driven organisations",
        "use_case": "Centralise identity, secrets, and security posture management",
        "max_ram_mb": 4096,
        "team_size": "medium",
        "max_complexity": "advanced",
    },
    {
        "id": "llm-app-stack",
        "name": "LLM Application Stack",
        "description": "Local LLM inference, vector store, orchestration layer, and embedding pipeline.",
        "needs": ["llm-inference", "vector-database", "embedding", "api-gateway"],
        "audience": "AI application developers",
        "use_case": "Build RAG pipelines and LLM-powered apps fully on-premise",
        "max_ram_mb": 32768,
        "team_size": "small",
        "max_complexity": "advanced",
    },
    {
        "id": "startup-saas-backend",
        "name": "Startup SaaS Backend",
        "description": "PostgreSQL, authentication, object storage, and background job queue — the standard SaaS foundation.",
        "needs": ["relational-database", "authentication", "object-storage", "job-queue"],
        "audience": "Early-stage startups and indie hackers",
        "use_case": "Ship a production-ready SaaS backend in a weekend",
        "max_ram_mb": 2048,
        "team_size": "solo",
        "max_complexity": "intermediate",
    },
    {
        "id": "api-platform",
        "name": "API Platform",
        "description": "API gateway, rate limiting, documentation, and analytics for internal or public APIs.",
        "needs": ["api-gateway", "rate-limiting", "api-documentation", "analytics"],
        "audience": "Platform teams and API-first companies",
        "use_case": "Manage and monetise APIs with observability and developer portal",
        "max_ram_mb": 4096,
        "team_size": "medium",
        "max_complexity": "advanced",
    },
    {
        "id": "feature-flag-platform",
        "name": "Feature Flag & Experimentation Platform",
        "description": "Feature flags, A/B testing, product analytics, and session replay for growth teams.",
        "needs": ["feature-flags", "a-b-testing", "product-analytics", "session-replay"],
        "audience": "Growth engineering and product teams",
        "use_case": "Ship features safely with gradual rollouts and data-driven decisions",
        "max_ram_mb": 4096,
        "team_size": "medium",
        "max_complexity": "intermediate",
    },
    {
        "id": "headless-ecommerce-enterprise",
        "name": "Enterprise Headless Commerce",
        "description": "High-performance commerce API, PIM, CDN, and recommendation engine for large catalogues.",
        "needs": ["headless-commerce", "product-information-management", "search", "recommendation-engine"],
        "audience": "Enterprise e-commerce teams",
        "use_case": "Power multi-channel commerce experiences with a composable architecture",
        "max_ram_mb": 16384,
        "team_size": "enterprise",
        "max_complexity": "expert",
    },
    {
        "id": "financial-data-stack",
        "name": "Financial Data & Reporting Stack",
        "description": "Time-series database, data warehouse, BI, and workflow orchestration for financial data.",
        "needs": ["time-series-database", "data-warehouse", "business-intelligence", "workflow-orchestration"],
        "audience": "Fintech teams and financial analysts",
        "use_case": "Ingest, store, and analyse financial metrics and trading data",
        "max_ram_mb": 16384,
        "team_size": "medium",
        "max_complexity": "advanced",
    },
    {
        "id": "self-hosted-analytics",
        "name": "Self-Hosted Web Analytics",
        "description": "Privacy-first web analytics, event tracking, and A/B testing — no third-party cookies.",
        "needs": ["web-analytics", "event-tracking", "a-b-testing", "dashboard"],
        "audience": "Privacy-conscious businesses and GDPR-regulated companies",
        "use_case": "Understand user behaviour without sending data to Google",
        "max_ram_mb": 2048,
        "team_size": "solo",
        "max_complexity": "beginner",
    },
    {
        "id": "video-platform",
        "name": "Self-Hosted Video Platform",
        "description": "Video hosting, transcoding, live streaming, and CDN integration.",
        "needs": ["video-hosting", "media-transcoding", "live-streaming", "cdn"],
        "audience": "Media companies and education platforms",
        "use_case": "Host and stream video content without YouTube or Vimeo lock-in",
        "max_ram_mb": 8192,
        "team_size": "small",
        "max_complexity": "advanced",
    },
    {
        "id": "developer-portal",
        "name": "Developer Portal",
        "description": "API docs, changelog, status page, and support for external developers.",
        "needs": ["api-documentation", "changelog", "status-page", "support-desk"],
        "audience": "Developer-relations and platform teams",
        "use_case": "Create a world-class developer experience for external API consumers",
        "max_ram_mb": 2048,
        "team_size": "small",
        "max_complexity": "beginner",
    },
    {
        "id": "data-science-notebook",
        "name": "Data Science Notebook Environment",
        "description": "Interactive notebooks, versioned datasets, experiment tracking, and collaborative review.",
        "needs": ["notebook", "experiment-tracking", "data-versioning", "collaboration"],
        "audience": "Data scientists and research teams",
        "use_case": "Run, track, and share data science experiments in a reproducible environment",
        "max_ram_mb": 8192,
        "team_size": "small",
        "max_complexity": "intermediate",
    },
    {
        "id": "gitops-platform",
        "name": "GitOps Platform",
        "description": "Git hosting, CI runner, container registry, and deployment operator for GitOps workflows.",
        "needs": ["git-hosting", "ci-runner", "container-registry", "gitops-operator"],
        "audience": "Platform engineering teams adopting GitOps",
        "use_case": "Self-host the full GitOps toolchain — no GitHub dependency",
        "max_ram_mb": 8192,
        "team_size": "medium",
        "max_complexity": "advanced",
    },
]


# ---------------------------------------------------------------------------
# TF-IDF stack-resolution engine (mirrors server.py oss_find_stack)
# ---------------------------------------------------------------------------

def _build_indices(db: list[dict]) -> tuple[dict, dict, dict]:
    slug_index = {t["slug"]: t for t in db}
    idf: dict[str, float] = {}
    tool_tfidf: dict[str, dict[str, float]] = {}

    all_terms: Counter = Counter()
    doc_terms: dict[str, Counter] = {}
    for t in db:
        terms = _extract_terms(t)
        doc_terms[t["slug"]] = terms
        all_terms.update(set(terms.keys()))

    n = len(db)
    idf = {term: math.log(n / (1 + count)) for term, count in all_terms.items()}

    for slug, terms in doc_terms.items():
        tfidf: dict[str, float] = {}
        max_tf = max(terms.values()) if terms else 1
        for term, count in terms.items():
            tf = 0.5 + 0.5 * (count / max_tf)
            tfidf[term] = tf * idf.get(term, 0)
        tool_tfidf[slug] = tfidf

    return slug_index, idf, tool_tfidf


def _extract_terms(t: dict) -> Counter:
    text_parts: list[str] = []
    for tag in t.get("tags", []):
        text_parts.extend(tag.replace("-", " ").split() * 3)
    for d in t.get("problem_domains", []):
        text_parts.extend(d.replace("-", " ").split() * 2)
    text_parts.extend(t.get("tagline", "").lower().split())
    for uc in t.get("use_cases_detailed", []):
        text_parts.extend(uc.lower().split())
    for r in t.get("replaces", []):
        text_parts.extend(r.lower().split())
    text_parts.extend(t.get("category", "").lower().split())
    text_parts.extend(t.get("sub_category", "").lower().split())

    terms: Counter = Counter()
    for w in text_parts:
        w = re.sub(r"[^a-z0-9+#]", "", w)
        if len(w) > 1:
            terms[w] += 1
    return terms


def _score_query(query: str, slug: str, slug_index: dict, idf: dict, tool_tfidf: dict) -> float:
    t = slug_index.get(slug)
    if not t:
        return 0.0

    q_lower = query.lower().strip()
    q_words = re.sub(r"[^a-z0-9+# -]", "", q_lower).split()

    exact_bonus = 0.0
    tool_tags = set(t.get("tags", []))
    tool_domains = set(t.get("problem_domains", []))
    tool_all_text = tool_tags | tool_domains

    for w in q_words:
        if w in tool_all_text:
            exact_bonus += 0.35
        for tag in tool_all_text:
            if w in tag or tag in w:
                exact_bonus += 0.15
                break

    q_hyphenated = q_lower.replace(" ", "-")
    for tag in tool_all_text:
        if q_hyphenated == tag or q_hyphenated in tag:
            exact_bonus += 0.5

    cat_words = (t.get("category", "") + " " + t.get("sub_category", "")).lower()
    for w in q_words:
        if w in cat_words:
            exact_bonus += 0.1

    q_terms: Counter = Counter(re.sub(r"[^a-z0-9+# ]", "", q_lower).split())
    for w in q_words:
        if "-" in w:
            for part in w.split("-"):
                if len(part) > 1:
                    q_terms[part] += 1

    q_tfidf: dict[str, float] = {}
    max_qtf = max(q_terms.values()) if q_terms else 1
    for term, count in q_terms.items():
        tf = 0.5 + 0.5 * (count / max_qtf)
        q_tfidf[term] = tf * idf.get(term, 0)

    t_tfidf = tool_tfidf.get(slug, {})
    dot = sum(q_tfidf.get(term, 0) * t_tfidf.get(term, 0) for term in set(q_tfidf) | set(t_tfidf))
    mag_q = math.sqrt(sum(v**2 for v in q_tfidf.values())) or 1
    mag_t = math.sqrt(sum(v**2 for v in t_tfidf.values())) or 1
    cosine = dot / (mag_q * mag_t)

    return cosine + exact_bonus


def resolve_stack(recipe: dict, db: list[dict], slug_index: dict, idf: dict, tool_tfidf: dict) -> list[dict]:
    cx_order = {"beginner": 0, "intermediate": 1, "advanced": 2, "expert": 3}
    max_cx = cx_order.get(recipe.get("max_complexity", "intermediate"), 1)
    team_size = recipe.get("team_size", "small")
    max_ram = recipe.get("max_ram_mb", 8192)

    stack: list[dict] = []
    used_ram = 0
    selected_slugs: set[str] = set()

    for need in recipe["needs"]:
        candidates = [
            t for t in db
            if cx_order.get(t.get("complexity_level", "intermediate"), 1) <= max_cx
            and (team_size in t.get("team_size_fit", []) or team_size == "enterprise")
            and t.get("self_hostable", True)
            and t["slug"] not in selected_slugs
        ]

        scored: list[tuple[dict, float]] = []
        for t in candidates:
            base_score = _score_query(need, t["slug"], slug_index, idf, tool_tfidf)
            compat_bonus = 0.0
            for sel_slug in selected_slugs:
                if sel_slug in t.get("integrates_with", []) or sel_slug in t.get("complements", []):
                    compat_bonus += 0.15
                if sel_slug in t.get("conflicts_with", []):
                    compat_bonus -= 0.5
            ram_left = max_ram - used_ram
            ram_penalty = -0.3 if t.get("min_ram_mb", 256) > ram_left else 0.0
            scored.append((t, base_score + compat_bonus + ram_penalty))

        scored.sort(key=lambda x: -x[1])

        if scored:
            best, best_score = scored[0]
            alternatives = [s[0]["name"] for s in scored[1:4]]
            stack.append(
                {
                    "need": need,
                    "slug": best["slug"],
                    "name": best["name"],
                    "tagline": best["tagline"],
                    "category": best["category"],
                    "sub_category": best["sub_category"],
                    "license": best.get("license", ""),
                    "license_type": best.get("license_type", ""),
                    "min_ram_mb": best.get("min_ram_mb", 256),
                    "complexity_level": best.get("complexity_level", ""),
                    "deployment_methods": best.get("deployment_methods", []),
                    "github_stars": best.get("github_stars", 0),
                    "score": round(best_score, 4),
                    "alternatives": alternatives,
                }
            )
            selected_slugs.add(best["slug"])
            used_ram += best.get("min_ram_mb", 256)

    return stack


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate() -> None:
    db: list[dict] = json.loads(DB_PATH.read_text())
    slug_index, idf, tool_tfidf = _build_indices(db)

    output: list[dict] = []
    for recipe in RECIPES:
        stack = resolve_stack(recipe, db, slug_index, idf, tool_tfidf)
        total_ram = sum(item["min_ram_mb"] for item in stack)

        # Compute cohesion
        slugs_in_stack = [item["slug"] for item in stack]
        connections = 0
        for i, sl_a in enumerate(slugs_in_stack):
            tool_a = slug_index.get(sl_a, {})
            for sl_b in slugs_in_stack[i + 1 :]:
                if sl_b in tool_a.get("integrates_with", []) or sl_b in tool_a.get("complements", []):
                    connections += 1
                tool_b = slug_index.get(sl_b, {})
                if sl_a in tool_b.get("integrates_with", []) or sl_a in tool_b.get("complements", []):
                    connections += 1

        n_pairs = len(slugs_in_stack) * (len(slugs_in_stack) - 1)
        cohesion = round(connections / n_pairs * 100, 1) if n_pairs > 0 else 0.0

        output.append(
            {
                "id": recipe["id"],
                "name": recipe["name"],
                "description": recipe["description"],
                "audience": recipe["audience"],
                "use_case": recipe["use_case"],
                "max_ram_mb": recipe["max_ram_mb"],
                "team_size": recipe.get("team_size", "small"),
                "max_complexity": recipe.get("max_complexity", "intermediate"),
                "total_ram_mb": total_ram,
                "cohesion_pct": cohesion,
                "tools": stack,
            }
        )
        print(
            f"  {recipe['id']}: {len(stack)} tools, {total_ram}MB RAM, {cohesion}% cohesion"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nGenerated {len(output)} stack recipes -> {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()
