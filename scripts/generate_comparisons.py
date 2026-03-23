#!/usr/bin/env python3
"""
Generate top comparison pairs from db.json.

Prioritises pairs by strength of connection:
  Priority 1 — Tools connected by mutual similar_to edges (strongest signal)
  Priority 2 — Tools connected by one-directional similar_to edge
  Priority 3 — Tools in the same sub_category
  Priority 4 — Tools sharing 3+ tags and same category

Output: website/src/data/comparisons.json
Each entry: {"slugs": [a, b], "priority": N, "reason": "..."}
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "db.json"
OUTPUT_PATH = REPO_ROOT / "website" / "src" / "data" / "comparisons.json"
MAX_PAIRS = 500


def _pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted([a, b]))  # type: ignore[return-value]


def generate() -> None:
    db: list[dict] = json.loads(DB_PATH.read_text())
    slug_index = {t["slug"]: t for t in db}

    # ------------------------------------------------------------------ #
    # Accumulate raw pairs with reason and base priority weight            #
    # ------------------------------------------------------------------ #
    # pair -> (weight, reason_string)
    pair_data: dict[tuple[str, str], dict] = {}

    def _add(a: str, b: str, weight: int, reason: str) -> None:
        key = _pair(a, b)
        if key[0] == key[1]:
            return
        if key not in pair_data:
            pair_data[key] = {"weight": 0, "reasons": []}
        pair_data[key]["weight"] += weight
        if reason not in pair_data[key]["reasons"]:
            pair_data[key]["reasons"].append(reason)

    # Build a set of all similar_to slugs for fast mutual-edge detection
    sim_map: dict[str, set[str]] = defaultdict(set)
    for tool in db:
        for sim_slug in tool.get("similar_to", []):
            if sim_slug in slug_index:
                sim_map[tool["slug"]].add(sim_slug)

    # Priority 1 — mutual similar_to (both sides point at each other)
    for slug, sims in sim_map.items():
        for sim_slug in sims:
            if slug in sim_map.get(sim_slug, set()):
                _add(slug, sim_slug, 8, "mutual similar_to")

    # Priority 2 — one-directional similar_to
    for slug, sims in sim_map.items():
        for sim_slug in sims:
            _add(slug, sim_slug, 4, "similar_to")

    # Priority 3 — same sub_category
    by_subcat: dict[str, list[str]] = defaultdict(list)
    for tool in db:
        key = f"{tool['category']}/{tool['sub_category']}"
        by_subcat[key].append(tool["slug"])

    for group_key, slugs in by_subcat.items():
        if len(slugs) < 2:
            continue
        reason = f"same sub-category: {group_key.split('/')[-1]}"
        for i, a in enumerate(slugs):
            for b in slugs[i + 1 :]:
                _add(a, b, 2, reason)

    # Priority 4 — same category + 3+ shared tags
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for tool in db:
        by_cat[tool["category"]].append(tool)

    for cat_tools in by_cat.values():
        for i, tool_a in enumerate(cat_tools):
            tags_a = set(tool_a.get("tags", []))
            for tool_b in cat_tools[i + 1 :]:
                tags_b = set(tool_b.get("tags", []))
                shared = tags_a & tags_b
                if len(shared) >= 3:
                    _add(
                        tool_a["slug"],
                        tool_b["slug"],
                        1,
                        f"shared tags: {', '.join(sorted(shared)[:4])}",
                    )

    # ------------------------------------------------------------------ #
    # Sort and cap                                                          #
    # ------------------------------------------------------------------ #
    sorted_pairs = sorted(
        pair_data.items(), key=lambda kv: -kv[1]["weight"]
    )[:MAX_PAIRS]

    output = []
    for (slug_a, slug_b), meta in sorted_pairs:
        tool_a = slug_index.get(slug_a)
        tool_b = slug_index.get(slug_b)
        if not tool_a or not tool_b:
            continue
        output.append(
            {
                "slugs": [slug_a, slug_b],
                "names": [tool_a["name"], tool_b["name"]],
                "category": tool_a["category"],
                "priority": meta["weight"],
                "reason": meta["reasons"][0] if meta["reasons"] else "",
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Generated {len(output)} comparison pairs -> {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()
