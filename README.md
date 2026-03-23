# Open Source Network — MCP Server

A 244-tool open-source intelligence engine that plugs into Claude Code.
Ask natural language questions, get ranked tool stacks with compatibility analysis.

## Quick Start

### 1. Install dependencies

```bash
cd open-source-network-mcp
pip install -r requirements.txt
```

### 2. Register with Claude Code

```bash
claude mcp add open-source-network -- python /full/path/to/open-source-network-mcp/server.py
```

### 3. Use it

Open Claude Code and ask:

```
Search for a lightweight CRM I can self-host on a 2GB VPS
```

```
Build me a stack: I need auth, database, CRM, email marketing, 
and project management for a 5-person startup on 8GB RAM Hetzner VPS
```

```
Compare postgresql vs mysql vs mariadb
```

```
What tools are compatible with supabase?
```

## Available Tools (8)

| Tool | Description |
|------|-------------|
| `oss_search` | Semantic search with filters (RAM, license, complexity, category) |
| `oss_get_tool` | Full 51-field details for any tool by slug |
| `oss_find_stack` | Build compatible stacks with RAM budgets and compatibility scoring |
| `oss_find_compatible` | Graph traversal — find what integrates/complements a tool |
| `oss_compare` | Side-by-side comparison matrix of 2-6 tools |
| `oss_list_categories` | Browse all 20 categories and 244 tools |
| `oss_browse_tags` | Explore the tag taxonomy (278 unique tags) |
| `oss_stats` | Database statistics and health check |

## Database Stats

- **244 tools** across 20 categories
- **51 fields** per tool (6 dimensions)
- **12,444** total data points
- **919** graph edges (integrates/complements/similar/conflicts)
- **1,645** tag entries across 278 unique tags
- **987** problem domain entries
- **496** anti-patterns
- **808** use case descriptions

## How It Works

The matching engine uses **TF-IDF scoring** across semantic fields (tags, problem domains, 
use cases, descriptions) combined with **graph traversal** for stack compatibility.

Pipeline: `Query → Hard Filters → TF-IDF Ranking → Graph Compatibility → Ranked Output`

No external APIs needed. No Ollama. No embeddings server. 
Claude Code IS the reasoning engine — the MCP just gives it the data superpowers.

## Example Queries

### Search
```
oss_search("vector database for RAG", max_ram_mb=4096, license_type="permissive")
```

### Stack Building
```
oss_find_stack(needs=["auth","crm","email","wiki","monitoring"], max_ram_mb=8192, team_size="small")
```

### Compatibility
```
oss_find_compatible("n8n")  → shows everything n8n integrates with
```

### Compare
```
oss_compare(["qdrant","milvus","chroma","weaviate"])
```

## File Structure

```
open-source-network-mcp/
├── server.py          # MCP server (FastMCP, 8 tools, TF-IDF engine)
├── db.json            # 244-tool database (51 fields each)
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Creator

Created and maintained by **KingHippo** ([@TheKingHippopotamus](https://github.com/TheKingHippopotamus))

## License

MIT — see [LICENSE](LICENSE) for details.

## Creator

Created and maintained by **[KingHippo](https://github.com/TheKingHippopotamus)**.
