<p align="center">
  <img src="https://img.shields.io/badge/Tools-244-blue?style=for-the-badge" alt="244 Tools" />
  <img src="https://img.shields.io/badge/Graph%20Edges-919-purple?style=for-the-badge" alt="919 Edges" />
  <img src="https://img.shields.io/badge/Tags-1%2C044-green?style=for-the-badge" alt="1,044 Tags" />
  <img src="https://img.shields.io/badge/Fields%20Per%20Tool-51-orange?style=for-the-badge" alt="51 Fields" />
</p>

<h1 align="center">Open Source Network</h1>

<p align="center">
  <strong>The open-source stack advisor inside Claude Code.</strong><br/>
  244 tools. 919 compatibility edges. 51 dimensions per tool.<br/>
  Semantic search, stack building, health scoring, and graph intelligence — zero external APIs.
</p>

<p align="center">
  <a href="https://pypi.org/project/open-source-network/"><img src="https://img.shields.io/pypi/v/open-source-network?color=blue&label=PyPI" alt="PyPI" /></a>
  <a href="https://github.com/TheKingHippopotamus/Open_source_network/actions"><img src="https://github.com/TheKingHippopotamus/Open_source_network/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://oss-neural-match.pages.dev"><img src="https://img.shields.io/badge/Live%20Site-pages.dev-orange" alt="Website" /></a>
  <a href="https://github.com/TheKingHippopotamus/Open_source_network/blob/main/LICENSE"><img src="https://img.shields.io/github/license/TheKingHippopotamus/Open_source_network" alt="License" /></a>
</p>

---

## What Is This?

Open Source Network is an **MCP server** that gives Claude Code deep knowledge about 244 open-source tools. Instead of Claude guessing from training data, it gets structured, curated intelligence — compatibility graphs, health scores, anti-patterns, and stack-building algorithms.

**Ask it anything:**
- *"Find me a self-hosted CRM that runs on 2GB RAM"*
- *"Build me a stack: auth, database, CRM, email marketing — under 4GB total"*
- *"Compare PostgreSQL vs MySQL vs MariaDB"*
- *"What integrates with n8n?"*
- *"Is this tool healthy enough for production?"*

---

## Install in 30 Seconds

### Option 1: pip (recommended)
```bash
pip install open-source-network
claude mcp add open-source-network -- open-source-network
```

### Option 2: From source
```bash
git clone https://github.com/TheKingHippopotamus/Open_source_network.git
cd Open_source_network
pip install -r requirements.txt
claude mcp add open-source-network -- python server.py
```

That's it. Open Claude Code and start asking about tools.

---

## 10 MCP Tools

| Tool | What It Does |
|------|-------------|
| `oss_search` | Semantic search with filters (category, RAM, license, complexity) |
| `oss_get_tool` | Full 51-field profile for any tool |
| `oss_find_stack` | Build a compatible stack from needs + constraints |
| `oss_find_compatible` | Graph traversal — integrations, complements, conflicts |
| `oss_compare` | Side-by-side comparison matrix (2-6 tools) |
| `oss_list_categories` | Browse 20 categories and 111 sub-categories |
| `oss_browse_tags` | Explore 1,044 semantic tags |
| `oss_stats` | Database statistics and health check |
| `oss_health_score` | Composite A-F health grade with 6-dimension breakdown |
| `oss_explain_recommendation` | Why was this tool recommended? Full reasoning. |

---

## The Intelligence Engine

This isn't a simple keyword search. Under the hood:

```
                    ┌─────────────────────────┐
                    │     10 MCP TOOLS        │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   INTELLIGENCE ENGINE   │
                    │                         │
                    │  BM25 + Dense Hybrid    │
                    │  Scoring Engine         │
                    │         +               │
                    │  Graph Engine           │
                    │  (PageRank, community   │
                    │   detection, link       │
                    │   prediction)           │
                    │         +               │
                    │  Health Scorer          │
                    │  (6 dimensions, A-F)    │
                    │         +               │
                    │  Recommendation         │
                    │  Explainer              │
                    │         +               │
                    │  481 Technical Synonyms │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   244 TOOLS × 51 FIELDS │
                    │   919 Graph Edges       │
                    │   1,044 Tags            │
                    │   817 Problem Domains   │
                    │   496 Anti-Patterns     │
                    └─────────────────────────┘
```

### Search: BM25 + Synonym Expansion
Queries like `"k8s deployment"` automatically expand to match `kubernetes`, `container-orchestration`, `helm`, etc. via a 481-entry synonym corpus. BM25 scoring replaces TF-IDF for better relevance ranking.

### Graph: 919 Relationship Edges
Every tool knows what it integrates with, complements, competes with, and conflicts with. PageRank identifies ecosystem anchors. Community detection reveals natural tool clusters. Link prediction discovers missing connections.

### Health Scoring: 6 Dimensions
Each tool gets a composite health score based on:

| Dimension | Weight | Signals |
|-----------|--------|---------|
| Activity | 25% | Commit frequency, last release date |
| Community | 20% | Stars, contributors, plugin ecosystem |
| Maturity | 20% | Age, stability level |
| Backing | 15% | Organization, funding model |
| License | 10% | Permissiveness, vendor lock-in risk |
| Documentation | 10% | Docs quality rating |

### Stack Building: Constraint Satisfaction
Tell it what you need, and it builds a compatible stack:
- Respects RAM budgets (cumulative across all tools)
- Checks cross-tool compatibility via graph edges
- Penalizes conflicts, rewards integrations
- Suggests alternatives for every pick

---

## The Database: 51 Fields Per Tool

Every tool is described across **51 structured dimensions**:

<details>
<summary><strong>Click to see all 51 fields</strong></summary>

**Identity:** name, slug, tagline, description, category, sub_category, logo_url

**Repository:** website, repo_url, license

**Technical:** license_type, language, framework, api_type, min_ram_mb, min_cpu_cores, scaling_pattern, data_model, protocols, deployment_methods

**Operational:** self_hostable, k8s_native, offline_capable, sdk_languages, plugin_ecosystem

**Ecosystem Graph:** integrates_with, complements, similar_to, conflicts_with, replaces

**Community:** github_stars, contributors_count, commit_frequency, first_release_year, latest_version, last_release_date, backing_org, funding_model, docs_quality, maturity

**Fit:** complexity_level, team_size_fit, industry_verticals, performance_tier, vendor_lockin_risk, pricing_model

**Semantic:** tags, problem_domains, use_cases_detailed, anti_patterns, stack_layer

</details>

---

## 20 Categories, 244 Tools

| Category | Tools | Examples |
|----------|-------|---------|
| Databases | 25 | PostgreSQL, Redis, MongoDB, ClickHouse, Qdrant |
| Embeddable | 47 | TipTap, Mermaid, Excalidraw, BlockNote |
| LLMs & AI Infra | 18 | Ollama, vLLM, LangChain, Haystack |
| DevOps & Infra | 15 | ArgoCD, Terraform, Ansible, Traefik |
| AI / ML | 16 | PyTorch, TensorFlow, Hugging Face, OpenCV |
| CRM & ERP | 11 | EspoCRM, Twenty, ERPNext, Dolibarr |
| Monitoring | 12 | Prometheus, Grafana, Uptime Kuma, Sentry |
| Low-Code | 13 | n8n, Appsmith, Supabase, NocoDB |
| Communication | 8 | Mattermost, Matrix, Chatwoot |
| Web & CMS | 10 | WordPress, Ghost, Strapi, Payload |
| Media & Files | 11 | MinIO, Immich, Jellyfin |
| Analytics | 7 | Plausible, PostHog, Metabase |
| DNS & Networking | 8 | Caddy, Traefik, WireGuard |
| Knowledge & Docs | 8 | Outline, BookStack, Wiki.js |
| Security & Auth | 6 | Keycloak, Authentik, Vault |
| Dev Tools | 9 | Gitea, Hoppscotch, Redash |
| Automation | 7 | n8n, Airflow, Temporal |
| Email Marketing | 5 | Mautic, Listmonk, Postal |
| Project Mgmt | 7 | Plane, Taiga, OpenProject |
| Scheduling | 1 | Cal.com |

---

## Live Website

**https://oss-neural-match.pages.dev**

477 auto-generated pages — tool profiles, side-by-side comparisons, category guides — all built from the same database that powers the MCP tools.

---

## Example Conversations

### Find a tool
> **You:** Find me a lightweight vector database I can self-host on a 2GB VPS
>
> **Claude (using oss_search):** Here are the top matches — Qdrant (512MB RAM, Rust-based, REST+gRPC), Milvus, Weaviate...

### Build a stack
> **You:** I need auth, a database, CRM, and email marketing. Budget: 4GB RAM, small team.
>
> **Claude (using oss_find_stack):** Recommended stack: Lucia (16MB) + PostgreSQL (256MB) + EspoCRM (512MB) + Listmonk (256MB) = 1,040MB total, 25% of budget. All deploy via Docker.

### Health check
> **You:** Is EspoCRM safe for production?
>
> **Claude (using oss_health_score):** EspoCRM scores 0.72 (Grade B, green band). Strong: daily commits, stable maturity. Watch: community-funded, AGPL license.

### Deep comparison
> **You:** Compare n8n vs Airflow vs Temporal
>
> **Claude (using oss_compare):** Side-by-side across 18 dimensions — n8n is beginner-friendly (256MB), Airflow is enterprise-grade (2GB), Temporal handles complex workflows...

---

## Architecture

```
open-source-network/
├── server.py                 # MCP server (10 tools)
├── db.json                   # 244-tool database (566KB)
├── engine/
│   ├── scoring.py            # BM25 + dense hybrid scorer
│   ├── graph.py              # PageRank, communities, link prediction
│   ├── health.py             # 6-dimension health scoring
│   ├── explain.py            # Recommendation explainer
│   └── embeddings.py         # Optional dense vectors
├── data/
│   └── synonyms.json         # 481 technical synonym mappings
├── website/                  # Astro static site (477 pages)
├── api/                      # Cloudflare Worker REST API
├── scripts/                  # Data refresh & generation
├── tests/                    # 322 tests
└── .github/workflows/        # CI, deploy, publish, data refresh
```

---

## Development

```bash
git clone https://github.com/TheKingHippopotamus/Open_source_network.git
cd Open_source_network
pip install -r requirements.txt -r requirements-dev.txt

# Run tests (322 tests, ~2s)
python -m pytest tests/ -v

# Run the MCP server locally
python server.py

# Build the website (477 pages, ~2s)
cd website && npm install && npm run build

# Preview the website
npm run preview  # → http://localhost:4321
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

**Quick contributions (5-10 min):**
- Add a tool to `db.json` (all 51 fields)
- Improve existing tool data (fix tags, add anti-patterns)
- Add synonym mappings to `data/synonyms.json`

**Larger contributions:**
- Add a new MCP tool
- Improve the scoring algorithm
- Add comparison pages to the website

---

## How It Works (No External APIs)

This is a key design principle: **Claude IS the reasoning engine.** The MCP server provides structured data and algorithms. No OpenAI embeddings, no Ollama, no vector database required.

Everything runs locally:
- Search: BM25 scoring computed in-process
- Graph: PageRank and community detection in pure Python
- Health: Composite scores from static metadata
- Synonyms: 481-entry JSON dictionary loaded at startup

The entire server starts in <1 second and answers queries in <100ms.

---

## Infrastructure

| Layer | Service | Cost |
|-------|---------|------|
| Package | [PyPI](https://pypi.org/project/open-source-network/) | Free |
| Website | Cloudflare Pages | Free (unlimited bandwidth) |
| API | Cloudflare Workers | Free (100K req/day) |
| CI/CD | GitHub Actions | Free (public repo) |
| Data Refresh | GitHub Actions cron | Free (weekly) |
| **Total** | | **$0/month** |

---

<p align="center">
  <strong>Built by <a href="https://github.com/TheKingHippopotamus">KingHippo</a></strong><br/>
  <sub>Powered by Claude Code + NEXUS AI</sub>
</p>
