#!/usr/bin/env python3
"""
Refresh GitHub data for all tools in db.json.
Run weekly via GitHub Actions or manually.

Usage:
    python scripts/refresh_github_data.py
    python scripts/refresh_github_data.py --dry-run
    python scripts/refresh_github_data.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "db.json"
SIGNALS_PATH = REPO_ROOT / "data" / "signals.json"

# ---------------------------------------------------------------------------
# GitHub API
# ---------------------------------------------------------------------------
GITHUB_API = "https://api.github.com"
GITHUB_REPO_RE = re.compile(
    r"^https?://github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+?)(?:\.git)?/?$"
)

# Global rate-limit state — updated from every response header.
_requests_made: int = 0
_rate_limit_remaining: int = 5000
_token: str | None = os.environ.get("GITHUB_TOKEN")


def _build_opener() -> urllib.request.OpenerDirector:
    """Build a urllib opener with the GitHub API headers baked in."""
    opener = urllib.request.build_opener()
    opener.addheaders = [
        ("Accept", "application/vnd.github+json"),
        ("X-GitHub-Api-Version", "2022-11-28"),
        ("User-Agent", "oss-neural-match-refresh/1.0"),
    ]
    if _token:
        opener.addheaders.append(("Authorization", f"Bearer {_token}"))
    return opener


_opener: urllib.request.OpenerDirector | None = None


def _get_opener() -> urllib.request.OpenerDirector:
    global _opener
    if _opener is None:
        _opener = _build_opener()
        if _token:
            print("[auth] GitHub token found — authenticated requests (5,000/hr limit)")
        else:
            print("[warn] No GITHUB_TOKEN env var — unauthenticated (60/hr limit)", file=sys.stderr)
    return _opener


def _github_get(path: str, params: dict[str, str] | None = None) -> tuple[dict[str, Any] | None, dict[str, str]]:
    """
    GET from the GitHub API.

    Returns (parsed_json_or_None, response_headers).
    Handles 404 (repo gone), 403 (rate-limited), and network errors gracefully.
    Updates global rate-limit counters from each response.
    """
    global _requests_made, _rate_limit_remaining

    # Back off when the hourly budget is almost exhausted.
    if _rate_limit_remaining <= 100:
        print(
            f"[rate-limit] Only {_rate_limit_remaining} requests remaining — sleeping 60s",
            file=sys.stderr,
        )
        time.sleep(60)

    url = f"{GITHUB_API}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    opener = _get_opener()

    try:
        with opener.open(url, timeout=15) as resp:
            _requests_made += 1
            headers: dict[str, str] = {k.lower(): v for k, v in resp.headers.items()}

            # Refresh remaining budget.
            remaining_hdr = headers.get("x-ratelimit-remaining")
            if remaining_hdr is not None:
                _rate_limit_remaining = int(remaining_hdr)

            raw = resp.read()
            if not raw:
                return None, headers
            return json.loads(raw), headers

    except urllib.error.HTTPError as exc:
        _requests_made += 1

        # Update rate-limit from error response headers too.
        remaining_hdr = exc.headers.get("X-RateLimit-Remaining")
        if remaining_hdr is not None:
            _rate_limit_remaining = int(remaining_hdr)

        if exc.code == 404:
            # Repo moved or deleted — not an error worth logging loudly.
            return None, {}

        if exc.code == 403:
            reset_ts = exc.headers.get("X-RateLimit-Reset")
            wait = 60
            if reset_ts:
                wait = max(int(reset_ts) - int(time.time()) + 5, 5)
            print(f"[rate-limit] 403 from GitHub — sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
            return None, {}

        if exc.code == 202:
            # GitHub is computing stats asynchronously; caller handles retry.
            return None, {"x-status": "202"}

        print(f"[error] HTTP {exc.code} for {url}", file=sys.stderr)
        return None, {}

    except (urllib.error.URLError, OSError) as exc:
        print(f"[error] Network error for {url}: {exc}", file=sys.stderr)
        return None, {}


def _parse_repo_path(repo_url: str) -> tuple[str, str] | None:
    """Return (owner, repo) from a GitHub URL, or None for non-GitHub URLs."""
    if not repo_url:
        return None
    match = GITHUB_REPO_RE.match(repo_url.strip())
    if not match:
        return None
    return match.group(1), match.group(2)


# ---------------------------------------------------------------------------
# Per-endpoint fetch helpers
# ---------------------------------------------------------------------------

def _fetch_repo(owner: str, repo: str) -> dict[str, Any]:
    """
    Fetch core repo metadata: stars, forks, open_issues, language,
    updated_at (last push), and archived flag.
    """
    data, _ = _github_get(f"/repos/{owner}/{repo}")
    if not data:
        return {}

    result: dict[str, Any] = {
        "github_stars": data.get("stargazers_count"),
        "forks": data.get("forks_count"),
        "open_issues": data.get("open_issues_count"),
        "archived": data.get("archived", False),
    }

    # pushed_at is the most reliable "last activity" proxy.
    pushed_at = data.get("pushed_at")
    if pushed_at:
        result["last_commit_date"] = pushed_at[:10]  # YYYY-MM-DD

    return result


def _fetch_latest_release(owner: str, repo: str) -> dict[str, Any]:
    """
    Fetch latest release: tag_name (version string) and published_at (date).
    Returns empty dict when no releases exist (404).
    """
    data, _ = _github_get(f"/repos/{owner}/{repo}/releases/latest")
    if not data:
        return {}

    result: dict[str, Any] = {}
    tag = data.get("tag_name", "")
    if tag:
        # Strip common 'v' prefix so "v1.2.3" → "1.2.3".
        result["latest_release"] = tag.lstrip("v") if tag.startswith("v") else tag

    published = data.get("published_at")
    if published:
        result["latest_release_date"] = published[:10]  # YYYY-MM-DD

    return result


def _fetch_contributors_count(owner: str, repo: str) -> int | None:
    """
    Derive the total contributor count from the Link header pagination.

    GitHub's contributors endpoint returns per_page=1 with a Link header
    pointing to the last page — that page number equals the contributor count.
    Includes anonymous contributors (anon=true) per the task spec.

    A 202 means GitHub is computing the stats cache; we retry once.
    """
    global _requests_made

    path = f"/repos/{owner}/{repo}/contributors"
    params = {"per_page": "1", "anon": "true"}

    data, headers = _github_get(path, params=params)

    # Handle 202 cache-building: wait briefly and retry once.
    if data is None and headers.get("x-status") == "202":
        time.sleep(3)
        data, headers = _github_get(path, params=params)

    if data is None:
        return None

    link_header = headers.get("link", "")
    last_page_match = re.search(r'page=(\d+)>;\s*rel="last"', link_header)
    if last_page_match:
        return int(last_page_match.group(1))

    # No Link header — all contributors fit on page 1.
    if isinstance(data, list):
        return len(data)

    return None


# ---------------------------------------------------------------------------
# Per-tool orchestration
# ---------------------------------------------------------------------------

def fetch_tool_signals(tool: dict[str, Any]) -> dict[str, Any] | None:
    """
    Fetch all GitHub signals for a single tool.

    Returns a signals dict on success, or None if the tool has no GitHub URL
    (indicating it should be skipped rather than counted as an error).
    """
    repo_url = tool.get("repo_url", "")
    parsed = _parse_repo_path(repo_url)

    if not parsed:
        return None  # Non-GitHub — skip silently.

    owner, repo = parsed

    signals: dict[str, Any] = {}

    # 1. Core repo metadata.
    repo_signals = _fetch_repo(owner, repo)
    if not repo_signals:
        # 404 or network error — treat as error, not skip.
        return {}
    signals.update(repo_signals)

    # 2. Latest release.
    release_signals = _fetch_latest_release(owner, repo)
    signals.update(release_signals)

    # 3. Contributors count.
    count = _fetch_contributors_count(owner, repo)
    if count is not None:
        signals["contributors_count"] = count

    # 4. Timestamp.
    signals["fetched_at"] = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return signals


# ---------------------------------------------------------------------------
# db.json update logic
# ---------------------------------------------------------------------------

# Mapping from signals.json keys → db.json field names.
_DB_FIELD_MAP: dict[str, str] = {
    "github_stars": "github_stars",
    "contributors_count": "contributors_count",
    "latest_release": "latest_version",
    "latest_release_date": "last_release_date",
}


def _is_meaningfully_different(field: str, old_val: Any, new_val: Any) -> bool:
    """
    Return True only when the new value is materially different from the old.

    - Numeric fields (stars, contributors): require >= 1% change or absolute
      difference > 10 to avoid noisy micro-updates.
    - String fields: simple inequality check, ignoring None/empty new values.
    """
    if new_val is None or new_val == "":
        return False
    if old_val is None or old_val == "":
        return True

    if isinstance(new_val, (int, float)) and isinstance(old_val, (int, float)):
        if old_val == 0:
            return new_val != 0
        abs_diff = abs(new_val - old_val)
        pct_diff = abs_diff / abs(old_val)
        return abs_diff > 10 or pct_diff >= 0.01

    return str(new_val) != str(old_val)


def apply_signals_to_tool(tool: dict[str, Any], signals: dict[str, Any]) -> list[str]:
    """
    Write new signal values into the tool dict where they are meaningfully different.

    Returns a list of field names that were actually updated.
    """
    changed: list[str] = []
    for signal_key, db_key in _DB_FIELD_MAP.items():
        new_val = signals.get(signal_key)
        if new_val is None:
            continue
        old_val = tool.get(db_key)
        if _is_meaningfully_different(db_key, old_val, new_val):
            tool[db_key] = new_val
            changed.append(db_key)
    return changed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh GitHub data for all tools in db.json.\n"
            "Writes data/signals.json and updates db.json in-place."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch from GitHub but do NOT write any files.",
    )
    parser.add_argument(
        "--limit",
        metavar="N",
        type=int,
        default=None,
        help="Process only the first N GitHub tools (useful for testing).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if not DB_PATH.exists():
        print(f"[error] db.json not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    with DB_PATH.open(encoding="utf-8") as fh:
        tools: list[dict[str, Any]] = json.load(fh)

    total_tools = len(tools)
    print(f"[init] Loaded {total_tools} tools from db.json")
    if args.dry_run:
        print("[init] DRY RUN — no files will be written")
    if args.limit:
        print(f"[init] Processing limit: {args.limit} GitHub tools")

    SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_signals: dict[str, dict[str, Any]] = {}

    stats = {"updated": 0, "skipped": 0, "errors": 0, "db_fields_changed": 0}
    github_processed = 0  # Count of tools with a GitHub URL we attempted.

    for i, tool in enumerate(tools):
        slug = tool.get("slug") or tool.get("name", f"tool_{i}")

        # Progress report every 10 tools.
        if i > 0 and i % 10 == 0:
            print(
                f"[progress] {i}/{total_tools} tools processed — "
                f"updated={stats['updated']} skipped={stats['skipped']} errors={stats['errors']}"
            )

        # Check limit against GitHub-eligible tools, not total tools.
        if args.limit is not None and github_processed >= args.limit:
            # Count remaining as skipped for an accurate summary.
            repo_url = tool.get("repo_url", "")
            if "github.com" in repo_url:
                stats["skipped"] += 1
            continue

        repo_url = tool.get("repo_url", "")
        if "github.com" not in repo_url:
            stats["skipped"] += 1
            continue

        github_processed += 1
        signals = fetch_tool_signals(tool)

        if signals is None:
            # Non-GitHub URL (regex didn't match despite containing "github.com").
            stats["skipped"] += 1
            continue

        if not signals:
            # Fetch returned empty — 404 or network error.
            print(f"[error] No data returned for {slug} ({repo_url})", file=sys.stderr)
            stats["errors"] += 1
            continue

        all_signals[slug] = signals

        changed_fields = apply_signals_to_tool(tool, signals)
        if changed_fields:
            stats["db_fields_changed"] += len(changed_fields)

        stats["updated"] += 1

    # Final progress line.
    print(
        f"[progress] {total_tools}/{total_tools} tools processed — "
        f"updated={stats['updated']} skipped={stats['skipped']} errors={stats['errors']}"
    )

    if not args.dry_run:
        # Write db.json back with updated fields.
        with DB_PATH.open("w", encoding="utf-8") as fh:
            json.dump(tools, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        print(f"[write] db.json updated ({stats['db_fields_changed']} field changes)")

        # Write signals.json.
        with SIGNALS_PATH.open("w", encoding="utf-8") as fh:
            json.dump(all_signals, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        print(f"[write] data/signals.json written ({len(all_signals)} tool entries)")
    else:
        print(f"[dry-run] Would write {len(all_signals)} entries to data/signals.json")
        print(f"[dry-run] Would apply {stats['db_fields_changed']} field changes to db.json")

    # Summary banner.
    print()
    print("=" * 60)
    print("SUMMARY")
    print(f"  Tools updated : {stats['updated']}")
    print(f"  Tools skipped : {stats['skipped']}")
    print(f"  Errors        : {stats['errors']}")
    print(f"  DB field edits: {stats['db_fields_changed']}")
    print(f"  API requests  : {_requests_made}")
    print(f"  Rate limit rem: {_rate_limit_remaining}")
    print("=" * 60)

    if stats["errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
