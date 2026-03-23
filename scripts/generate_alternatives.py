#!/usr/bin/env python3
"""
Generate "Best X Alternatives" data for every tool in db.json.

For each tool, alternatives are gathered from three sources (ranked):
  1. similar_to edges declared on the tool itself
  2. Reverse similar_to edges — tools that declare THIS tool as similar
  3. Tools in the same sub_category not already captured above

Each alternative entry carries a relevance score so the website can order them.

Output: website/src/data/alternatives.json
Schema: {slug: {name, tagline, alternatives: [{slug, name, score, reason}]}}
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "db.json"
OUTPUT_PATH = REPO_ROOT / "website" / "src" / "data" / "alternatives.json"
MAX_ALTERNATIVES = 10


def generate() -> None:
    db: list[dict] = json.loads(DB_PATH.read_text())
    slug_index = {t["slug"]: t for t in db}

    # Build reverse similar_to map
    reverse_sim: dict[str, list[str]] = defaultdict(list)
    for tool in db:
        for sim_slug in tool.get("similar_to", []):
            if sim_slug in slug_index:
                reverse_sim[sim_slug].append(tool["slug"])

    # Build sub_category groups
    by_subcat: dict[str, list[str]] = defaultdict(list)
    for tool in db:
        by_subcat[f"{tool['category']}/{tool['sub_category']}"].append(tool["slug"])

    output: dict[str, dict] = {}

    for tool in db:
        slug = tool["slug"]
        seen: set[str] = {slug}
        alts: list[dict] = []

        # --- Source 1: declared similar_to edges (score = 1.0) ---
        for sim_slug in tool.get("similar_to", []):
            if sim_slug not in seen and sim_slug in slug_index:
                alts.append({"slug": sim_slug, "score": 1.0, "reason": "similar_to"})
                seen.add(sim_slug)

        # --- Source 2: reverse similar_to (score = 0.9) ---
        for rev_slug in reverse_sim.get(slug, []):
            if rev_slug not in seen:
                alts.append({"slug": rev_slug, "score": 0.9, "reason": "reverse similar_to"})
                seen.add(rev_slug)

        # --- Source 3: same sub_category (score = 0.6 + tag overlap bonus) ---
        subcat_key = f"{tool['category']}/{tool['sub_category']}"
        own_tags = set(tool.get("tags", []))
        for peer_slug in by_subcat.get(subcat_key, []):
            if peer_slug not in seen:
                peer = slug_index[peer_slug]
                shared_tags = own_tags & set(peer.get("tags", []))
                tag_bonus = min(len(shared_tags) * 0.05, 0.3)
                alts.append(
                    {
                        "slug": peer_slug,
                        "score": round(0.6 + tag_bonus, 3),
                        "reason": f"same sub-category ({tool['sub_category']})",
                    }
                )
                seen.add(peer_slug)

        # Sort by score desc, cap at MAX_ALTERNATIVES
        alts.sort(key=lambda x: -x["score"])
        alts = alts[:MAX_ALTERNATIVES]

        # Enrich with display fields
        enriched = []
        for alt in alts:
            peer = slug_index.get(alt["slug"])
            if not peer:
                continue
            enriched.append(
                {
                    "slug": alt["slug"],
                    "name": peer["name"],
                    "tagline": peer["tagline"],
                    "category": peer["category"],
                    "sub_category": peer["sub_category"],
                    "license_type": peer.get("license_type", ""),
                    "min_ram_mb": peer.get("min_ram_mb", 0),
                    "complexity_level": peer.get("complexity_level", ""),
                    "github_stars": peer.get("github_stars", 0),
                    "score": alt["score"],
                    "reason": alt["reason"],
                }
            )

        output[slug] = {
            "slug": slug,
            "name": tool["name"],
            "tagline": tool["tagline"],
            "category": tool["category"],
            "sub_category": tool["sub_category"],
            "alternatives": enriched,
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Generated alternatives for {len(output)} tools -> {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()
