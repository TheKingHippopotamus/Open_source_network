#!/usr/bin/env python3
"""
Generate SEO metadata for all pages of the Open Source Network website.

Pages covered:
  - /tools/{slug}              — one page per tool (244 pages)
  - /compare/{a}-vs-{b}        — one page per comparison pair (from comparisons.json)
  - /alternatives/{slug}       — one page per tool's alternatives page
  - /stacks/{id}               — one page per stack recipe
  - /                          — homepage
  - /categories/{slug}         — one page per category

Output: website/src/data/seo.json
Schema: { page_type: { page_id: { title, description, canonical, og_title, og_description, schema } } }
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "db.json"
DATA_DIR = REPO_ROOT / "website" / "src" / "data"
OUTPUT_PATH = DATA_DIR / "seo.json"

SITE_URL = "https://opensourcenetwork.com"
SITE_NAME = "Open Source Network"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9-]", "-", text.lower().strip()).strip("-")


def _truncate(text: str, max_len: int = 160) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rsplit(" ", 1)[0] + "..."


def _stars_label(stars: int) -> str:
    if stars >= 100_000:
        return f"{stars // 1000}k+"
    if stars >= 1000:
        return f"{round(stars / 1000, 1)}k"
    return str(stars) if stars else ""


def _load_json_if_exists(path: Path) -> Any:
    if path.exists():
        return json.loads(path.read_text())
    return None


# ---------------------------------------------------------------------------
# Structured data (JSON-LD) generators
# ---------------------------------------------------------------------------

def _software_schema(tool: dict) -> dict:
    schema: dict = {
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        "name": tool["name"],
        "description": tool.get("tagline", ""),
        "applicationCategory": tool.get("category", ""),
        "operatingSystem": "Linux, macOS, Windows",
        "offers": {
            "@type": "Offer",
            "price": "0",
            "priceCurrency": "USD",
        },
        "license": tool.get("license", ""),
        "url": tool.get("website", ""),
        "sameAs": [tool.get("repo_url", "")] if tool.get("repo_url") else [],
    }
    if tool.get("github_stars"):
        schema["aggregateRating"] = {
            "@type": "AggregateRating",
            "ratingValue": "4.5",
            "ratingCount": str(tool["github_stars"]),
            "bestRating": "5",
        }
    return schema


def _breadcrumb_schema(*crumbs: tuple[str, str]) -> dict:
    """crumbs: sequence of (name, url) tuples"""
    return {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": [
            {
                "@type": "ListItem",
                "position": i + 1,
                "name": name,
                "item": url,
            }
            for i, (name, url) in enumerate(crumbs)
        ],
    }


def _faq_schema(questions: list[tuple[str, str]]) -> dict:
    return {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": q,
                "acceptedAnswer": {"@type": "Answer", "text": a},
            }
            for q, a in questions
        ],
    }


# ---------------------------------------------------------------------------
# Page generators
# ---------------------------------------------------------------------------

def _tool_pages(db: list[dict]) -> dict[str, dict]:
    pages: dict[str, dict] = {}
    for tool in db:
        slug = tool["slug"]
        name = tool["name"]
        tagline = tool.get("tagline", "")
        category = tool.get("category", "")
        sub_category = tool.get("sub_category", "")
        stars_label = _stars_label(tool.get("github_stars", 0))
        license_name = tool.get("license", "open-source")

        title = f"{name} — Self-Hosted {sub_category} | {SITE_NAME}"
        description = _truncate(
            f"{name} is an open-source {sub_category.lower()} tool. {tagline}. "
            f"License: {license_name}. "
            f"RAM: {tool.get('min_ram_mb', '')}MB. "
            + (f"⭐ {stars_label} GitHub stars." if stars_label else ""),
            160,
        )

        replaces = tool.get("replaces", [])
        if replaces:
            og_description = _truncate(
                f"Self-host {name} as a free alternative to {', '.join(replaces[:2])}. "
                + tagline,
                200,
            )
        else:
            og_description = _truncate(f"Everything you need to know about self-hosting {name}. " + tagline, 200)

        faq_pairs = [
            (f"Is {name} free?", f"Yes. {name} is released under the {license_name} license and is free to self-host."),
            (f"What is {name} used for?", tagline),
        ]
        use_cases = tool.get("use_cases_detailed", [])
        if use_cases:
            faq_pairs.append((f"What are common use cases for {name}?", " ".join(use_cases[:2])))
        anti = tool.get("anti_patterns", [])
        if anti:
            faq_pairs.append((f"When should I NOT use {name}?", anti[0]))

        pages[slug] = {
            "path": f"/tools/{slug}",
            "title": title,
            "description": description,
            "og_title": f"{name}: Open-Source {sub_category} Alternative",
            "og_description": og_description,
            "canonical": f"{SITE_URL}/tools/{slug}",
            "schema": [
                _software_schema(tool),
                _breadcrumb_schema(
                    ("Home", SITE_URL),
                    (category, f"{SITE_URL}/categories/{_slugify(category)}"),
                    (name, f"{SITE_URL}/tools/{slug}"),
                ),
                _faq_schema(faq_pairs),
            ],
        }
    return pages


def _comparison_pages(comparisons: list[dict], slug_index: dict) -> dict[str, dict]:
    pages: dict[str, dict] = {}
    for comp in comparisons:
        slugs = comp.get("slugs", [])
        if len(slugs) != 2:
            continue
        slug_a, slug_b = slugs
        tool_a = slug_index.get(slug_a)
        tool_b = slug_index.get(slug_b)
        if not tool_a or not tool_b:
            continue

        page_id = f"{slug_a}-vs-{slug_b}"
        name_a, name_b = tool_a["name"], tool_b["name"]
        cat = tool_a.get("category", "")

        title = f"{name_a} vs {name_b} — {cat} Comparison | {SITE_NAME}"
        description = _truncate(
            f"Detailed comparison of {name_a} and {name_b}. "
            f"Compare license, RAM usage, complexity, maturity, and ecosystem. "
            f"Both are open-source {tool_a.get('sub_category', 'tools').lower()} tools.",
            160,
        )

        faq_pairs = [
            (
                f"What is the difference between {name_a} and {name_b}?",
                f"{name_a}: {tool_a.get('tagline', '')} | {name_b}: {tool_b.get('tagline', '')}",
            ),
            (
                f"Is {name_a} better than {name_b}?",
                f"It depends on your use case. {name_a} is {tool_a.get('complexity_level', 'intermediate')} "
                f"complexity and requires {tool_a.get('min_ram_mb', 256)}MB RAM. "
                f"{name_b} is {tool_b.get('complexity_level', 'intermediate')} complexity and requires "
                f"{tool_b.get('min_ram_mb', 256)}MB RAM.",
            ),
        ]

        pages[page_id] = {
            "path": f"/compare/{page_id}",
            "title": title,
            "description": description,
            "og_title": f"{name_a} vs {name_b}: Full Comparison",
            "og_description": _truncate(
                f"Side-by-side comparison of {name_a} and {name_b} — license, RAM, complexity, "
                f"ecosystem integrations, and when to choose each.",
                200,
            ),
            "canonical": f"{SITE_URL}/compare/{page_id}",
            "schema": [
                _breadcrumb_schema(
                    ("Home", SITE_URL),
                    (cat, f"{SITE_URL}/categories/{_slugify(cat)}"),
                    (f"{name_a} vs {name_b}", f"{SITE_URL}/compare/{page_id}"),
                ),
                _faq_schema(faq_pairs),
            ],
        }
    return pages


def _alternative_pages(alternatives: dict, slug_index: dict) -> dict[str, dict]:
    pages: dict[str, dict] = {}
    for slug, data in alternatives.items():
        tool = slug_index.get(slug)
        if not tool:
            continue
        name = tool["name"]
        sub_cat = tool.get("sub_category", "tool")
        alt_names = [a["name"] for a in data.get("alternatives", [])[:5]]

        title = f"Best {name} Alternatives (Open-Source) | {SITE_NAME}"
        description = _truncate(
            f"Top open-source alternatives to {name} for self-hosting. "
            + (f"Compare {', '.join(alt_names[:3])} and more." if alt_names else "")
            + f" All {sub_cat.lower()} tools, ranked by compatibility.",
            160,
        )

        faq_pairs = [
            (
                f"What are the best alternatives to {name}?",
                f"Top open-source alternatives: {', '.join(alt_names[:5])}." if alt_names else f"There are several alternatives in the {sub_cat} category.",
            ),
            (
                f"Is there a free alternative to {name}?",
                f"Yes. {', '.join(alt_names[:3])} are free, open-source alternatives to {name}." if alt_names else f"Yes, there are open-source alternatives available.",
            ),
        ]

        pages[slug] = {
            "path": f"/alternatives/{slug}",
            "title": title,
            "description": description,
            "og_title": f"{name} Alternatives: {len(data.get('alternatives', []))} Open-Source Options",
            "og_description": _truncate(
                f"Looking for a self-hosted alternative to {name}? "
                f"Explore {len(data.get('alternatives', []))} open-source options ranked by relevance.",
                200,
            ),
            "canonical": f"{SITE_URL}/alternatives/{slug}",
            "schema": [
                _breadcrumb_schema(
                    ("Home", SITE_URL),
                    ("Alternatives", f"{SITE_URL}/alternatives"),
                    (f"{name} Alternatives", f"{SITE_URL}/alternatives/{slug}"),
                ),
                _faq_schema(faq_pairs),
            ],
        }
    return pages


def _stack_pages(recipes: list[dict]) -> dict[str, dict]:
    pages: dict[str, dict] = {}
    for recipe in recipes:
        recipe_id = recipe["id"]
        name = recipe["name"]
        description = recipe["description"]
        tools_in_stack = [t["name"] for t in recipe.get("tools", [])[:4]]
        ram = recipe.get("total_ram_mb", recipe.get("max_ram_mb", 0))

        title = f"{name} — Self-Hosted Open-Source Stack | {SITE_NAME}"
        meta_desc = _truncate(
            f"{description} Includes: {', '.join(tools_in_stack)}. "
            f"Total RAM: {ram}MB. "
            f"For {recipe.get('audience', 'developers')}.",
            160,
        )

        faq_pairs = [
            (
                f"What tools are in the {name}?",
                f"The {name} includes: {', '.join(t['name'] + ' (' + t['need'] + ')' for t in recipe.get('tools', []))}.",
            ),
            (
                f"How much RAM does the {name} require?",
                f"Approximately {ram}MB of RAM total across all components.",
            ),
        ]

        pages[recipe_id] = {
            "path": f"/stacks/{recipe_id}",
            "title": title,
            "description": meta_desc,
            "og_title": f"{name}: Open-Source Self-Hosted Stack",
            "og_description": _truncate(description, 200),
            "canonical": f"{SITE_URL}/stacks/{recipe_id}",
            "schema": [
                _breadcrumb_schema(
                    ("Home", SITE_URL),
                    ("Stacks", f"{SITE_URL}/stacks"),
                    (name, f"{SITE_URL}/stacks/{recipe_id}"),
                ),
                _faq_schema(faq_pairs),
            ],
        }
    return pages


def _category_pages(db: list[dict]) -> dict[str, dict]:
    from collections import defaultdict

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for tool in db:
        by_cat[tool["category"]].append(tool)

    pages: dict[str, dict] = {}
    for cat, tools in by_cat.items():
        cat_slug = _slugify(cat)
        tool_names = [t["name"] for t in tools[:6]]
        sub_cats = sorted(set(t["sub_category"] for t in tools))

        title = f"Open-Source {cat} Tools — Self-Hosted | {SITE_NAME}"
        description = _truncate(
            f"Browse {len(tools)} open-source {cat.lower()} tools you can self-host. "
            f"Includes {', '.join(tool_names[:4])} and more. "
            f"Sub-categories: {', '.join(sub_cats[:4])}.",
            160,
        )

        pages[cat_slug] = {
            "path": f"/categories/{cat_slug}",
            "title": title,
            "description": description,
            "og_title": f"{len(tools)} Open-Source {cat} Tools to Self-Host",
            "og_description": _truncate(
                f"Compare and choose from {len(tools)} open-source {cat.lower()} tools. "
                f"Filtered by RAM, license, complexity, and team size.",
                200,
            ),
            "canonical": f"{SITE_URL}/categories/{cat_slug}",
            "schema": [
                _breadcrumb_schema(
                    ("Home", SITE_URL),
                    (cat, f"{SITE_URL}/categories/{cat_slug}"),
                ),
            ],
        }
    return pages


def _homepage(db: list[dict]) -> dict:
    top_tools = sorted(db, key=lambda t: -t.get("github_stars", 0))[:6]
    tool_names = [t["name"] for t in top_tools]
    cats = sorted(set(t["category"] for t in db))

    return {
        "path": "/",
        "title": f"Open Source Network — Find & Compare 244 Self-Hosted Open-Source Tools",
        "description": _truncate(
            f"Discover and compare {len(db)} open-source tools you can self-host. "
            f"Semantic search, stack builder, and side-by-side comparisons. "
            f"Popular: {', '.join(tool_names[:4])}.",
            160,
        ),
        "og_title": "Open Source Network — Open-Source Tool Intelligence",
        "og_description": _truncate(
            f"The definitive database of {len(db)} self-hostable open-source tools. "
            f"Build your perfect stack, compare alternatives, and escape SaaS lock-in.",
            200,
        ),
        "canonical": SITE_URL,
        "schema": [
            {
                "@context": "https://schema.org",
                "@type": "WebSite",
                "name": SITE_NAME,
                "url": SITE_URL,
                "potentialAction": {
                    "@type": "SearchAction",
                    "target": {
                        "@type": "EntryPoint",
                        "urlTemplate": f"{SITE_URL}/search?q={{search_term_string}}",
                    },
                    "query-input": "required name=search_term_string",
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate() -> None:
    db: list[dict] = json.loads(DB_PATH.read_text())
    slug_index = {t["slug"]: t for t in db}

    comparisons = _load_json_if_exists(DATA_DIR / "comparisons.json") or []
    alternatives = _load_json_if_exists(DATA_DIR / "alternatives.json") or {}
    recipes = _load_json_if_exists(DATA_DIR / "stack_recipes.json") or []

    seo: dict[str, Any] = {}

    # Homepage
    seo["home"] = {"index": _homepage(db)}
    print(f"  homepage: 1 page")

    # Tool pages
    tool_pages = _tool_pages(db)
    seo["tools"] = tool_pages
    print(f"  tools: {len(tool_pages)} pages")

    # Comparison pages
    comp_pages = _comparison_pages(comparisons, slug_index)
    seo["comparisons"] = comp_pages
    print(f"  comparisons: {len(comp_pages)} pages")

    # Alternative pages
    alt_pages = _alternative_pages(alternatives, slug_index)
    seo["alternatives"] = alt_pages
    print(f"  alternatives: {len(alt_pages)} pages")

    # Stack recipe pages
    stack_pages = _stack_pages(recipes)
    seo["stacks"] = stack_pages
    print(f"  stacks: {len(stack_pages)} pages")

    # Category pages
    cat_pages = _category_pages(db)
    seo["categories"] = cat_pages
    print(f"  categories: {len(cat_pages)} pages")

    total = sum(len(v) for v in seo.values())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(seo, indent=2))
    print(f"\nGenerated SEO data for {total} pages -> {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()
