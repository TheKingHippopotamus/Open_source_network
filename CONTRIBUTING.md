# Contributing to OSS Neural Match

Thanks for helping improve this project. Contributions are small and targeted — no massive PRs, please.

## How to add a tool (~30 min)

Edit `db.json` and add a new entry with all 51 fields. Every field is required — the scoring engine relies on them.

Key fields to get right:
- `slug` — lowercase, hyphenated, unique (e.g. `"my-tool"`)
- `tags` — the most important relevance signal, be specific
- `problem_domains` — what problem category does this solve?
- `use_cases_detailed` — 3-5 concrete sentences about real usage
- `anti_patterns` — when NOT to use this tool (just as valuable as use cases)
- `replaces` — what tools does this replace or compete with?

Validate your entry: `python -c "import json; json.load(open('db.json'))"` — it must parse clean.

## How to improve data (~10-15 min)

- Fix wrong tags: just edit the `tags` array in the relevant entry
- Add missing anti-patterns: add to `anti_patterns` array
- Fix a wrong description: edit `tagline` or `use_cases_detailed`
- Add a missing `replaces` entry: helps users find alternatives

No special setup needed for data-only changes.

## How to report bugs

Open a GitHub Issue with:
1. What you searched for
2. What you expected to get
3. What you actually got

For scoring bugs, run `explain_score` via the MCP tool and paste the output.

## Contribution time estimates

| Type | Time |
|------|------|
| Fix a tag or description | 5 min |
| Add an anti-pattern | 10 min |
| Add a new tool (full 51 fields) | 30 min |
| Improve synonyms.json | 15 min |
| Bug report | 5 min |

## Questions

Use [GitHub Discussions](../../discussions) — not Issues — for questions about architecture, roadmap, or "should I add X tool?"

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest
```
