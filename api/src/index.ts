/**
 * OSS Neural Match — Cloudflare Worker API Gateway
 * =================================================
 * Serves the tool database as a REST API with:
 *   - Rate limiting via D1 (free: 100/hr anon, 100/hr free-tier key, 1000/hr pro)
 *   - API key authentication (X-API-Key header or ?api_key= query param)
 *   - Full security headers (HSTS, CSP, X-Frame-Options, etc.) via cors.ts
 *   - Keyword search across all tool fields
 *   - Stack recommendation, comparison, health, categories, tags, and stats endpoints
 *
 * Architecture:
 *   auth.ts  — key extraction, SHA-256 hashing, D1 rate-limit logic
 *   cors.ts  — CORS + security headers, response builders
 *   index.ts — routing, search logic, route handlers (this file)
 *
 * The tool database (db.json) is imported as a static module and held in
 * memory for the Worker's lifetime. Indices are built once at module init.
 * All route handlers are synchronous over the in-memory data; only the
 * D1 rate-limit path is async.
 */

import toolDb from "../../db.json";
import {
  authenticate,
  recordUsage,
  cleanupOldWindows,
  type AuthResult,
} from "./auth";
import {
  buildResponseHeaders,
  buildRateLimitHeaders,
  handlePreflight,
  errorResponse,
} from "./cors";

// ─────────────────────────────────────────────────────────────────────────────
// Env bindings
// ─────────────────────────────────────────────────────────────────────────────

export interface Env {
  DB: D1Database;
  /** Overrides default 100 req/hour for the free tier (set in wrangler.toml vars). */
  RATE_LIMIT_FREE?: string;
  /** Overrides default 1000 req/hour for the pro tier (set in wrangler.toml vars). */
  RATE_LIMIT_PRO?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool data model (subset of db.json fields used at runtime)
// ─────────────────────────────────────────────────────────────────────────────

interface Tool {
  name: string;
  slug: string;
  tagline: string;
  description: string;
  category: string;
  sub_category: string;
  logo_url: string;
  website: string;
  repo_url: string;
  license: string;
  license_type: string;
  language: string[];
  framework: string[];
  api_type: string[];
  min_ram_mb: number;
  min_cpu_cores: number;
  scaling_pattern: string;
  data_model: string[];
  protocols: string[];
  deployment_methods: string[];
  self_hostable: boolean;
  k8s_native: boolean;
  offline_capable: boolean;
  integrates_with: string[];
  complements: string[];
  replaces: string[];
  similar_to: string[];
  conflicts_with: string[];
  sdk_languages: string[];
  plugin_ecosystem: string;
  github_stars: number;
  contributors_count: number;
  commit_frequency: string;
  first_release_year: number;
  latest_version: string;
  last_release_date: string;
  backing_org: string;
  funding_model: string;
  docs_quality: string;
  maturity: string;
  complexity_level: string;
  team_size_fit: string[];
  industry_verticals: string[];
  performance_tier: string;
  vendor_lockin_risk: string;
  pricing_model: string;
  tags: string[];
  problem_domains: string[];
  use_cases_detailed: string[];
  anti_patterns: string[];
  stack_layer: string[];
}

// ─────────────────────────────────────────────────────────────────────────────
// In-memory database — built once at module init
// ─────────────────────────────────────────────────────────────────────────────

const DB_TOOLS = toolDb as Tool[];

const SLUG_INDEX = new Map<string, Tool>(DB_TOOLS.map((t) => [t.slug, t]));

// Category index: lowercase category string → tools in that category
const CATEGORY_INDEX = new Map<string, Tool[]>();

// Tag index: lowercase tag string → tools with that tag
const TAG_INDEX = new Map<string, Tool[]>();

for (const tool of DB_TOOLS) {
  const catKey = tool.category.toLowerCase();
  if (!CATEGORY_INDEX.has(catKey)) CATEGORY_INDEX.set(catKey, []);
  CATEGORY_INDEX.get(catKey)!.push(tool);

  for (const tag of tool.tags ?? []) {
    const tagKey = tag.toLowerCase();
    if (!TAG_INDEX.has(tagKey)) TAG_INDEX.set(tagKey, []);
    TAG_INDEX.get(tagKey)!.push(tool);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Response helpers (wrapping cors.ts for the standard API envelope)
// ─────────────────────────────────────────────────────────────────────────────

interface ApiMeta {
  took_ms: number;
  remaining_requests: number;
  tier: string;
}

function okResponse(
  data: unknown,
  meta: ApiMeta,
  auth: AuthResult,
  status = 200
): Response {
  const rateLimitHeaders = buildRateLimitHeaders(
    auth.remainingRequests + (auth.remainingRequests === 0 ? 0 : 1), // approximate limit
    auth.remainingRequests
  );
  return new Response(JSON.stringify({ ok: true, data, meta }), {
    status,
    headers: buildResponseHeaders(rateLimitHeaders),
  });
}

function apiError(
  message: string,
  code: string,
  status: number,
  extra: Record<string, string> = {}
): Response {
  return new Response(JSON.stringify({ ok: false, error: message, code }), {
    status,
    headers: buildResponseHeaders(extra),
  });
}

function notFound(resource: string): Response {
  return apiError(`${resource} not found`, "NOT_FOUND", 404);
}

function badRequest(message: string): Response {
  return apiError(message, "BAD_REQUEST", 400);
}

// ─────────────────────────────────────────────────────────────────────────────
// Search helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Tokenises a query string into lowercase words, stripping punctuation.
 * Handles hyphenated compound terms by adding both the compound and its parts.
 * The MCP server runs full TF-IDF; this lightweight variant keeps Workers
 * CPU budget low (no floating-point heavy loops).
 */
function tokenise(query: string): string[] {
  const lower = query.toLowerCase().trim();
  const raw = lower.replace(/[^a-z0-9+# -]/g, "").split(/\s+/);
  const terms: string[] = [];
  for (const w of raw) {
    if (w.length < 2) continue;
    terms.push(w);
    if (w.includes("-")) {
      for (const part of w.split("-")) {
        if (part.length > 1) terms.push(part);
      }
    }
  }
  return [...new Set(terms)];
}

/**
 * Scores a tool against a set of query terms.
 * Scoring heuristic (higher = better match):
 *   +3   exact slug word match
 *   +2   term in tags or problem_domains
 *   +1   term in name, tagline, or category
 *   +0.5 term in description or use_cases_detailed
 */
function scoreToolKeyword(tool: Tool, terms: string[]): number {
  if (terms.length === 0) return 1; // no query → return all tools

  let score = 0;
  const slugWords = tool.slug.toLowerCase().split("-");
  const tagSet = new Set((tool.tags ?? []).map((t) => t.toLowerCase()));
  const domainSet = new Set(
    (tool.problem_domains ?? []).map((d) => d.toLowerCase())
  );
  const nameStr = tool.name.toLowerCase();
  const taglineStr = tool.tagline.toLowerCase();
  const catStr = (tool.category + " " + tool.sub_category).toLowerCase();
  const descStr = tool.description.toLowerCase();
  const useCasesStr = (tool.use_cases_detailed ?? []).join(" ").toLowerCase();

  for (const term of terms) {
    if (slugWords.includes(term) || tool.slug === term) {
      score += 3;
    } else if (tagSet.has(term) || domainSet.has(term)) {
      score += 2;
    } else if (
      nameStr.includes(term) ||
      taglineStr.includes(term) ||
      catStr.includes(term)
    ) {
      score += 1;
    } else if (descStr.includes(term) || useCasesStr.includes(term)) {
      score += 0.5;
    }
  }

  return score;
}

// ─────────────────────────────────────────────────────────────────────────────
// Route handlers — all synchronous over in-memory data
// ─────────────────────────────────────────────────────────────────────────────

/**
 * GET /api/v1/search?q=<query>&limit=10&category=<cat>&max_ram=<mb>
 */
function handleSearch(url: URL): Tool[] {
  const q = url.searchParams.get("q") ?? "";
  const limit = Math.min(
    parseInt(url.searchParams.get("limit") ?? "10", 10),
    50
  );
  const categoryFilter =
    url.searchParams.get("category")?.toLowerCase() ?? null;
  const maxRam = url.searchParams.get("max_ram")
    ? parseInt(url.searchParams.get("max_ram")!, 10)
    : null;

  const terms = tokenise(q);

  // Pre-filter by category (fast index lookup)
  let candidates: Tool[];
  if (categoryFilter) {
    const exact = CATEGORY_INDEX.get(categoryFilter);
    if (exact) {
      candidates = exact;
    } else {
      // Substring fallback: "ai" matches "AI / ML"
      candidates = DB_TOOLS.filter((t) =>
        t.category.toLowerCase().includes(categoryFilter)
      );
    }
  } else {
    candidates = DB_TOOLS;
  }

  // Pre-filter by RAM constraint
  if (maxRam !== null) {
    candidates = candidates.filter((t) => t.min_ram_mb <= maxRam);
  }

  // Score, filter zero-scores, sort, truncate
  return candidates
    .map((t) => ({ tool: t, score: scoreToolKeyword(t, terms) }))
    .filter((s) => s.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((s) => s.tool);
}

/**
 * GET /api/v1/tools/:slug
 */
function handleToolGet(slug: string): Tool | null {
  return SLUG_INDEX.get(slug) ?? null;
}

/**
 * GET /api/v1/tools/:slug/health
 * Lightweight activity/health snapshot — no external calls, derived from db.json.
 */
function handleToolHealth(slug: string): Record<string, unknown> | null {
  const tool = SLUG_INDEX.get(slug);
  if (!tool) return null;

  const now = new Date();
  // last_release_date is "YYYY-MM" — normalise to first of that month
  const lastRelease = tool.last_release_date
    ? new Date(tool.last_release_date + "-01")
    : null;
  const monthsSinceRelease =
    lastRelease && !isNaN(lastRelease.getTime())
      ? Math.floor(
          (now.getTime() - lastRelease.getTime()) / (1000 * 60 * 60 * 24 * 30)
        )
      : null;

  let activityStatus: "active" | "moderate" | "slow";
  if (tool.commit_frequency === "daily") {
    activityStatus = "active";
  } else if (tool.commit_frequency === "weekly") {
    activityStatus =
      monthsSinceRelease !== null && monthsSinceRelease < 6
        ? "active"
        : "moderate";
  } else {
    activityStatus =
      monthsSinceRelease !== null && monthsSinceRelease > 12 ? "slow" : "moderate";
  }

  return {
    slug: tool.slug,
    name: tool.name,
    maturity: tool.maturity,
    commit_frequency: tool.commit_frequency,
    last_release_date: tool.last_release_date,
    latest_version: tool.latest_version,
    months_since_release: monthsSinceRelease,
    github_stars: tool.github_stars,
    contributors_count: tool.contributors_count,
    activity_status: activityStatus,
    docs_quality: tool.docs_quality,
    backing_org: tool.backing_org,
    funding_model: tool.funding_model,
  };
}

/**
 * GET /api/v1/compare?slugs=tool1,tool2,tool3
 * Side-by-side comparison matrix for 2–5 tools.
 */
function handleCompare(url: URL): Record<string, unknown> | null {
  const raw = url.searchParams.get("slugs") ?? "";
  const slugs = raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .slice(0, 5);

  if (slugs.length < 2) return null;

  const tools = slugs
    .map((s) => SLUG_INDEX.get(s))
    .filter((t): t is Tool => t !== undefined);

  if (tools.length < 2) return null;

  const COMPARISON_FIELDS: (keyof Tool)[] = [
    "name",
    "category",
    "sub_category",
    "license",
    "license_type",
    "min_ram_mb",
    "min_cpu_cores",
    "scaling_pattern",
    "self_hostable",
    "k8s_native",
    "offline_capable",
    "github_stars",
    "maturity",
    "complexity_level",
    "performance_tier",
    "vendor_lockin_risk",
    "pricing_model",
    "docs_quality",
    "backing_org",
    "funding_model",
  ];

  const matrix: Record<string, Record<string, unknown>> = {};
  for (const field of COMPARISON_FIELDS) {
    matrix[field] = {};
    for (const tool of tools) {
      matrix[field][tool.slug] = tool[field];
    }
  }

  // Surface declared conflicts between the compared set
  const conflicts: Array<{ between: [string, string] }> = [];
  for (let i = 0; i < tools.length; i++) {
    for (let j = i + 1; j < tools.length; j++) {
      const a = tools[i];
      const b = tools[j];
      if (
        (a.conflicts_with ?? []).includes(b.slug) ||
        (b.conflicts_with ?? []).includes(a.slug)
      ) {
        conflicts.push({ between: [a.slug, b.slug] });
      }
    }
  }

  return {
    slugs: tools.map((t) => t.slug),
    names: tools.map((t) => t.name),
    matrix,
    conflicts,
  };
}

/**
 * GET /api/v1/categories
 * Returns all unique categories with tool counts, sorted by count descending.
 */
function handleCategories(): Array<{ category: string; count: number }> {
  return [...CATEGORY_INDEX.entries()]
    .map(([category, tools]) => ({ category, count: tools.length }))
    .sort((a, b) => b.count - a.count);
}

/**
 * GET /api/v1/tags?search=<term>
 * Returns all tags (optionally filtered by substring), sorted by frequency.
 */
function handleTags(url: URL): Array<{ tag: string; count: number }> {
  const search = url.searchParams.get("search")?.toLowerCase() ?? "";
  const result: Array<{ tag: string; count: number }> = [];

  for (const [tag, tools] of TAG_INDEX.entries()) {
    if (!search || tag.includes(search)) {
      result.push({ tag, count: tools.length });
    }
  }

  return result.sort((a, b) => b.count - a.count);
}

/**
 * GET /api/v1/stats
 * Aggregate statistics about the entire tool database.
 */
function handleStats(): Record<string, unknown> {
  const licenseBreakdown: Record<string, number> = {};
  const maturityBreakdown: Record<string, number> = {};
  const languageFreq: Record<string, number> = {};
  let totalStars = 0;
  let selfHostableCount = 0;
  let k8sNativeCount = 0;

  for (const t of DB_TOOLS) {
    licenseBreakdown[t.license_type] =
      (licenseBreakdown[t.license_type] ?? 0) + 1;
    maturityBreakdown[t.maturity] = (maturityBreakdown[t.maturity] ?? 0) + 1;
    for (const lang of t.language ?? []) {
      languageFreq[lang] = (languageFreq[lang] ?? 0) + 1;
    }
    totalStars += t.github_stars ?? 0;
    if (t.self_hostable) selfHostableCount++;
    if (t.k8s_native) k8sNativeCount++;
  }

  const topLanguages = Object.entries(languageFreq)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10)
    .map(([lang, count]) => ({ lang, count }));

  const categoryCounts = Object.fromEntries(
    [...CATEGORY_INDEX.entries()].map(([cat, tools]) => [cat, tools.length])
  );

  return {
    total_tools: DB_TOOLS.length,
    total_github_stars: totalStars,
    self_hostable_count: selfHostableCount,
    k8s_native_count: k8sNativeCount,
    license_breakdown: licenseBreakdown,
    maturity_breakdown: maturityBreakdown,
    top_languages: topLanguages,
    category_counts: categoryCounts,
  };
}

/**
 * GET /api/v1/stack?needs=auth,crm,email&max_ram=4096
 *
 * Recommends the highest-scoring non-conflicting tool per stated need.
 * Each need is tokenised and matched against tags and problem_domains.
 * Conflict detection runs after selection to warn the caller.
 */
function handleStack(url: URL): Record<string, unknown> | null {
  const needsRaw = url.searchParams.get("needs") ?? "";
  const needs = needsRaw
    .split(",")
    .map((n) => n.trim().toLowerCase())
    .filter(Boolean);

  if (needs.length === 0) return null;

  const maxRam = url.searchParams.get("max_ram")
    ? parseInt(url.searchParams.get("max_ram")!, 10)
    : null;

  const selected: Array<{ need: string; tool: Tool; score: number }> = [];
  const usedSlugs = new Set<string>();

  for (const need of needs) {
    const terms = tokenise(need);
    let candidates = DB_TOOLS;

    if (maxRam !== null) {
      candidates = candidates.filter((t) => t.min_ram_mb <= maxRam);
    }

    const best = candidates
      .filter((t) => !usedSlugs.has(t.slug))
      .map((t) => ({ tool: t, score: scoreToolKeyword(t, terms) }))
      .filter((s) => s.score > 0)
      .sort((a, b) => b.score - a.score)[0];

    if (best) {
      selected.push({ need, tool: best.tool, score: best.score });
      usedSlugs.add(best.tool.slug);
    }
  }

  // Detect conflicts between selected tools
  const conflicts: Array<{ between: [string, string] }> = [];
  for (let i = 0; i < selected.length; i++) {
    for (let j = i + 1; j < selected.length; j++) {
      const a = selected[i].tool;
      const b = selected[j].tool;
      if (
        (a.conflicts_with ?? []).includes(b.slug) ||
        (b.conflicts_with ?? []).includes(a.slug)
      ) {
        conflicts.push({ between: [a.slug, b.slug] });
      }
    }
  }

  return {
    needs,
    stack: selected.map((s) => ({
      need: s.need,
      tool: {
        slug: s.tool.slug,
        name: s.tool.name,
        tagline: s.tool.tagline,
        category: s.tool.category,
        sub_category: s.tool.sub_category,
        license: s.tool.license,
        min_ram_mb: s.tool.min_ram_mb,
        self_hostable: s.tool.self_hostable,
        maturity: s.tool.maturity,
        website: s.tool.website,
      },
      score: Math.round(s.score * 100) / 100,
    })),
    conflicts,
    warnings:
      conflicts.length > 0
        ? [
            "Some selected tools declare conflicts with each other. Review before combining.",
          ]
        : [],
    total_min_ram_mb: selected.reduce((acc, s) => acc + s.tool.min_ram_mb, 0),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Router / dispatcher
// ─────────────────────────────────────────────────────────────────────────────

async function dispatch(
  request: Request,
  env: Env,
  ctx: ExecutionContext
): Promise<Response> {
  const startMs = Date.now();
  const url = new URL(request.url);
  const path = url.pathname;

  // Authenticate and apply rate limiting
  const auth = await authenticate(request, env.DB);

  if (auth.remainingRequests < 0) {
    // Auth module signals rate limit exceeded with -1
    return apiError("Rate limit exceeded", "RATE_LIMIT_EXCEEDED", 429, {
      "Retry-After": "3600",
      ...buildRateLimitHeaders(0, 0),
    });
  }

  const elapsed = (): number => Date.now() - startMs;

  const meta = (): ApiMeta => ({
    took_ms: elapsed(),
    remaining_requests: auth.remainingRequests,
    tier: auth.tier,
  });

  // ── Route matching ─────────────────────────────────────────────────────────

  // GET /api/v1/search
  if (path === "/api/v1/search") {
    const results = handleSearch(url);
    const q = url.searchParams.get("q") ?? undefined;
    ctx.waitUntil(recordUsage(env.DB, auth.keyHash, "/api/v1/search", elapsed(), q));
    return okResponse(results, meta(), auth);
  }

  // GET /api/v1/categories
  if (path === "/api/v1/categories") {
    ctx.waitUntil(recordUsage(env.DB, auth.keyHash, "/api/v1/categories", elapsed()));
    return okResponse(handleCategories(), meta(), auth);
  }

  // GET /api/v1/tags
  if (path === "/api/v1/tags") {
    ctx.waitUntil(recordUsage(env.DB, auth.keyHash, "/api/v1/tags", elapsed()));
    return okResponse(handleTags(url), meta(), auth);
  }

  // GET /api/v1/stats
  if (path === "/api/v1/stats") {
    ctx.waitUntil(recordUsage(env.DB, auth.keyHash, "/api/v1/stats", elapsed()));
    return okResponse(handleStats(), meta(), auth);
  }

  // GET /api/v1/compare
  if (path === "/api/v1/compare") {
    const result = handleCompare(url);
    if (!result) {
      return badRequest("Provide at least 2 valid tool slugs via ?slugs=a,b");
    }
    ctx.waitUntil(
      recordUsage(
        env.DB,
        auth.keyHash,
        "/api/v1/compare",
        elapsed(),
        url.searchParams.get("slugs") ?? undefined
      )
    );
    return okResponse(result, meta(), auth);
  }

  // GET /api/v1/stack
  if (path === "/api/v1/stack") {
    const result = handleStack(url);
    if (!result) {
      return badRequest("Provide at least one need via ?needs=auth,crm");
    }
    ctx.waitUntil(
      recordUsage(
        env.DB,
        auth.keyHash,
        "/api/v1/stack",
        elapsed(),
        url.searchParams.get("needs") ?? undefined
      )
    );
    return okResponse(result, meta(), auth);
  }

  // GET /api/v1/tools/:slug/health  (must match before /tools/:slug)
  const healthMatch = path.match(/^\/api\/v1\/tools\/([^/]+)\/health$/);
  if (healthMatch) {
    const slug = decodeURIComponent(healthMatch[1]);
    const result = handleToolHealth(slug);
    if (!result) return notFound(`Tool "${slug}"`);
    ctx.waitUntil(
      recordUsage(env.DB, auth.keyHash, "/api/v1/tools/:slug/health", elapsed(), slug)
    );
    return okResponse(result, meta(), auth);
  }

  // GET /api/v1/tools/:slug
  const toolMatch = path.match(/^\/api\/v1\/tools\/([^/]+)$/);
  if (toolMatch) {
    const slug = decodeURIComponent(toolMatch[1]);
    const tool = handleToolGet(slug);
    if (!tool) return notFound(`Tool "${slug}"`);
    ctx.waitUntil(
      recordUsage(env.DB, auth.keyHash, "/api/v1/tools/:slug", elapsed(), slug)
    );
    return okResponse(tool, meta(), auth);
  }

  // API root — discovery endpoint
  if (path === "/" || path === "/api/v1" || path === "/api/v1/") {
    const limitFree = parseInt(env.RATE_LIMIT_FREE ?? "100", 10);
    const limitPro = parseInt(env.RATE_LIMIT_PRO ?? "1000", 10);
    return okResponse(
      {
        name: "OSS Neural Match API",
        version: "1.0.0",
        endpoints: [
          "GET /api/v1/search?q=<query>&limit=10&category=<cat>&max_ram=<mb>",
          "GET /api/v1/tools/:slug",
          "GET /api/v1/tools/:slug/health",
          "GET /api/v1/compare?slugs=tool1,tool2,tool3",
          "GET /api/v1/categories",
          "GET /api/v1/tags?search=<term>",
          "GET /api/v1/stats",
          "GET /api/v1/stack?needs=auth,crm,email&max_ram=4096",
        ],
        auth: {
          header: "X-API-Key",
          query_param: "api_key",
          no_key: "anonymous (rate limited by IP)",
        },
        rate_limits: {
          anonymous: "50 requests/hour",
          free: `${limitFree} requests/hour`,
          pro: `${limitPro} requests/hour`,
          team: "5000 requests/hour",
          enterprise: "10000 requests/hour",
        },
      },
      meta(),
      auth
    );
  }

  return apiError("Endpoint not found", "NOT_FOUND", 404);
}

// ─────────────────────────────────────────────────────────────────────────────
// Worker entry point
// ─────────────────────────────────────────────────────────────────────────────

export default {
  /**
   * Handles HTTP requests.
   * All routes are GET-only; preflight OPTIONS is handled before auth.
   */
  async fetch(
    request: Request,
    env: Env,
    ctx: ExecutionContext
  ): Promise<Response> {
    // CORS preflight — must succeed without auth
    if (request.method === "OPTIONS") {
      return handlePreflight();
    }

    // Only GET is supported — this is a read-only API
    if (request.method !== "GET") {
      return errorResponse(405, "Method not allowed");
    }

    try {
      return await dispatch(request, env, ctx);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "An unexpected error occurred";
      return errorResponse(500, message);
    }
  },

  /**
   * Cron trigger handler for stale rate-limit window cleanup.
   * Configure in wrangler.toml:
   *   [[triggers.crons]]
   *   crons = ["0 * * * *"]   # every hour
   */
  async scheduled(
    _controller: ScheduledController,
    env: Env,
    ctx: ExecutionContext
  ): Promise<void> {
    ctx.waitUntil(cleanupOldWindows(env.DB));
  },
} satisfies ExportedHandler<Env>;
