"""
Open Source Network — Graph Intelligence Engine
=============================================
Pure-stdlib graph algorithms over the 244-tool ecosystem graph.

Graph topology
--------------
Four directed edge types are stored on each tool record:

  integrates_with   strong technical coupling (A uses B's API/protocol)
  complements       works well alongside (weaker, usage-pattern coupling)
  similar_to        functional alternatives (undirected by convention)
  conflicts_with    mutually exclusive or broken together

Because many edges point to slugs that are not in the DB (121 dangling refs
out of 919 total edges), every algorithm here is defensive: unknown slugs are
silently skipped so callers never receive KeyErrors.

Algorithms
----------
  PageRank          iterative power method (no networkx)
  Community         label-propagation (no python-louvain)
  Link prediction   Jaccard coefficient on union neighbour sets

Performance contract: all methods run in < 1 s on 244 nodes / 919 edges.

Usage
-----
    from engine.graph import GraphEngine
    ge = GraphEngine(tools)          # pass the list from db.json
    pr = ge.compute_pagerank()       # {slug: float}
    comms = ge.detect_communities()  # {slug: int}
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Edge types that carry "positive" coupling signal used for PageRank /
# community detection / bridge analysis.  similar_to and conflicts_with are
# intentionally excluded from the authority-flow graph — alternatives don't
# transfer authority; conflicts are negative signal handled separately.
POSITIVE_EDGE_TYPES: Tuple[str, ...] = ("integrates_with", "complements")

ALL_EDGE_TYPES: Tuple[str, ...] = (
    "integrates_with",
    "complements",
    "similar_to",
    "conflicts_with",
)


# ===========================================================================
# GraphEngine
# ===========================================================================

class GraphEngine:
    """
    Knowledge-graph intelligence layer over the OSS tool ecosystem.

    The graph is built once at construction time from the raw tool list.
    All public methods are read-only and safe to call repeatedly with no
    re-computation cost (except compute_pagerank and detect_communities,
    which are computed fresh on each call but remain fast at this scale).

    Parameters
    ----------
    tools : list[dict]
        Raw tool records from db.json.  The only required keys are 'slug'
        and any of the four edge-type keys.  All other keys are ignored.
    """

    def __init__(self, tools: List[Dict]) -> None:
        # Canonical set of slugs that exist in the DB.
        # Dangling edges (refs to unknown slugs) are silently dropped.
        self._slugs: Set[str] = {t["slug"] for t in tools}

        # Adjacency: forward edges per type
        # adj[edge_type][slug] = [target_slug, ...]  (only known slugs)
        self._adj: Dict[str, Dict[str, List[str]]] = {
            et: defaultdict(list) for et in ALL_EDGE_TYPES
        }

        # Reverse adjacency for integrates_with and complements
        # rev_adj[edge_type][slug] = [source_slug, ...]
        self._rev_adj: Dict[str, Dict[str, List[str]]] = {
            et: defaultdict(list) for et in ALL_EDGE_TYPES
        }

        for tool in tools:
            src = tool["slug"]
            for et in ALL_EDGE_TYPES:
                for tgt in tool.get(et, []):
                    if tgt in self._slugs:
                        self._adj[et][src].append(tgt)
                        self._rev_adj[et][tgt].append(src)

        # Pre-compute the union neighbour set per node used by several
        # algorithms (positive-edge neighbours only).
        self._positive_neighbors: Dict[str, Set[str]] = {}
        for slug in self._slugs:
            nb: Set[str] = set()
            for et in POSITIVE_EDGE_TYPES:
                nb.update(self._adj[et].get(slug, []))
                nb.update(self._rev_adj[et].get(slug, []))
            self._positive_neighbors[slug] = nb

    # -----------------------------------------------------------------------
    # Core graph operations
    # -----------------------------------------------------------------------

    def get_neighbors(
        self,
        slug: str,
        edge_types: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Get the forward neighbours of *slug* grouped by edge type.

        Parameters
        ----------
        slug : str
            The source tool slug.
        edge_types : list[str] | None
            Subset of edge types to include.  Defaults to all four types.
            Invalid type names are silently ignored.

        Returns
        -------
        dict[str, list[str]]
            ``{edge_type: [target_slug, ...]}`` — only non-empty types are
            included.  Unknown *slug* returns an empty dict.
        """
        types = edge_types if edge_types is not None else list(ALL_EDGE_TYPES)
        result: Dict[str, List[str]] = {}
        for et in types:
            if et not in self._adj:
                continue
            targets = self._adj[et].get(slug, [])
            if targets:
                result[et] = list(targets)
        return result

    def get_reverse_neighbors(self, slug: str) -> Dict[str, List[str]]:
        """
        Find tools that declare *slug* in their own edge lists.

        Returns
        -------
        dict[str, list[str]]
            ``{edge_type: [source_slug, ...]}`` for every edge type that has
            at least one inbound reference to *slug*.
        """
        result: Dict[str, List[str]] = {}
        for et in ALL_EDGE_TYPES:
            sources = self._rev_adj[et].get(slug, [])
            if sources:
                result[et] = list(sources)
        return result

    def has_edge(
        self,
        slug_a: str,
        slug_b: str,
        edge_type: Optional[str] = None,
    ) -> bool:
        """
        Return True if a directed edge exists from *slug_a* to *slug_b*.

        Parameters
        ----------
        slug_a : str
            Source slug.
        slug_b : str
            Target slug.
        edge_type : str | None
            If given, only check that specific edge type.  If None, return
            True if *any* edge type connects the two nodes.
        """
        types = [edge_type] if edge_type is not None else list(ALL_EDGE_TYPES)
        for et in types:
            if et not in self._adj:
                continue
            if slug_b in self._adj[et].get(slug_a, []):
                return True
        return False

    # -----------------------------------------------------------------------
    # PageRank
    # -----------------------------------------------------------------------

    def compute_pagerank(
        self,
        damping: float = 0.85,
        iterations: int = 100,
    ) -> Dict[str, float]:
        """
        Compute PageRank over the *positive* (integrates_with + complements)
        directed graph using the iterative power method.

        The graph is treated as directed: authority flows along outbound
        integrates_with and complements edges.  Dangling nodes (no outbound
        positive edges) redistribute their rank uniformly to all nodes,
        preventing rank sinks.

        Parameters
        ----------
        damping : float
            Damping factor (standard default 0.85).
        iterations : int
            Power-method iterations.  Convergence on 244 nodes is typically
            reached well before 100 iterations.

        Returns
        -------
        dict[str, float]
            ``{slug: pagerank_score}`` normalised so scores sum to 1.0.
        """
        nodes = list(self._slugs)
        n = len(nodes)
        if n == 0:
            return {}

        idx: Dict[str, int] = {slug: i for i, slug in enumerate(nodes)}

        # Build out-edges using positive edge types only (directed)
        out_edges: List[List[int]] = [[] for _ in range(n)]
        for et in POSITIVE_EDGE_TYPES:
            for src, targets in self._adj[et].items():
                if src not in idx:
                    continue
                si = idx[src]
                for tgt in targets:
                    if tgt in idx:
                        ti = idx[tgt]
                        if ti not in out_edges[si]:
                            out_edges[si].append(ti)

        out_degree = [len(out_edges[i]) for i in range(n)]
        uniform = 1.0 / n

        # Initialise rank vector
        rank = [uniform] * n
        teleport = (1.0 - damping) * uniform

        for _ in range(iterations):
            new_rank = [teleport] * n

            # Dangling mass (nodes with no outbound positive edges)
            dangling_mass = 0.0
            for i in range(n):
                if out_degree[i] == 0:
                    dangling_mass += rank[i]

            dangling_contribution = damping * dangling_mass * uniform
            for i in range(n):
                new_rank[i] += dangling_contribution

            # Normal propagation
            for i in range(n):
                if out_degree[i] > 0:
                    contribution = damping * rank[i] / out_degree[i]
                    for j in out_edges[i]:
                        new_rank[j] += contribution

            rank = new_rank

        return {nodes[i]: rank[i] for i in range(n)}

    # -----------------------------------------------------------------------
    # Community detection
    # -----------------------------------------------------------------------

    def detect_communities(self) -> Dict[str, int]:
        """
        Assign community IDs via label-propagation on the undirected positive
        graph (integrates_with + complements treated as undirected).

        Algorithm
        ---------
        1. Initialise each node with a unique label equal to its own index.
        2. For a fixed number of passes, iterate over nodes in random order.
           Each node adopts the most-frequent label among its neighbours.
           Ties are broken by choosing the smallest label.
        3. Remap labels to contiguous integers starting from 0.

        Isolated nodes (no positive-edge connections) each form their own
        community.

        Returns
        -------
        dict[str, int]
            ``{slug: community_id}`` where IDs are contiguous integers.
        """
        nodes = list(self._slugs)
        n = len(nodes)
        if n == 0:
            return {}

        idx: Dict[str, int] = {slug: i for i, slug in enumerate(nodes)}

        # Undirected adjacency lists (positive edges only)
        undirected: List[List[int]] = [[] for _ in range(n)]
        for et in POSITIVE_EDGE_TYPES:
            for src, targets in self._adj[et].items():
                if src not in idx:
                    continue
                si = idx[src]
                for tgt in targets:
                    if tgt not in idx:
                        continue
                    ti = idx[tgt]
                    if ti not in undirected[si]:
                        undirected[si].append(ti)
                    if si not in undirected[ti]:
                        undirected[ti].append(si)

        # Initialise labels
        labels = list(range(n))

        # Propagation — 20 passes is more than enough for 244 nodes
        order = list(range(n))
        for _ in range(20):
            random.shuffle(order)
            changed = False
            for i in order:
                neighbours = undirected[i]
                if not neighbours:
                    continue
                freq: Dict[int, int] = {}
                for j in neighbours:
                    lbl = labels[j]
                    freq[lbl] = freq.get(lbl, 0) + 1
                max_freq = max(freq.values())
                best = min(lbl for lbl, f in freq.items() if f == max_freq)
                if best != labels[i]:
                    labels[i] = best
                    changed = True
            if not changed:
                break

        # Remap to contiguous IDs
        unique_labels = sorted(set(labels))
        remap = {old: new for new, old in enumerate(unique_labels)}
        return {nodes[i]: remap[labels[i]] for i in range(n)}

    # -----------------------------------------------------------------------
    # Stack analysis
    # -----------------------------------------------------------------------

    def stack_cohesion(self, slugs: List[str]) -> Dict:
        """
        Analyse the internal compatibility of a set of tools.

        Cohesion is defined as the fraction of directed positive-edge pairs
        that actually exist out of the maximum possible directed pairs.

        Parameters
        ----------
        slugs : list[str]
            Tool slugs forming the candidate stack.  Unknown slugs are
            silently filtered out before analysis.

        Returns
        -------
        dict with keys:
            cohesion_pct : float
                Percentage of possible positive-edge pairs that exist
                (0.0 – 100.0).
            connections : list[tuple[str, str, str]]
                ``(slug_a, slug_b, edge_type)`` for every positive directed
                edge within the set.
            conflicts : list[tuple[str, str]]
                ``(slug_a, slug_b)`` for every conflicts_with edge within
                the set.
            missing_links : list[tuple[str, str]]
                Ordered pairs ``(slug_a, slug_b)`` that have no direct
                positive edge but share at least one common positive
                neighbour (i.e. one hop away), sorted by shared-neighbour
                count descending.
            bridge_tools : list[str]
                Up to 5 external tools that would maximally improve
                connectivity (delegates to ``find_bridge_tools``).
        """
        known = [s for s in slugs if s in self._slugs]
        stack_set = set(known)
        n = len(known)

        connections: List[Tuple[str, str, str]] = []
        conflicts: List[Tuple[str, str]] = []

        # Collect directed connections and conflicts
        for slug_a in known:
            for et in POSITIVE_EDGE_TYPES:
                for slug_b in self._adj[et].get(slug_a, []):
                    if slug_b in stack_set and slug_b != slug_a:
                        connections.append((slug_a, slug_b, et))
            for slug_b in self._adj["conflicts_with"].get(slug_a, []):
                if slug_b in stack_set:
                    conflicts.append((slug_a, slug_b))

        # Cohesion: directed pairs that could exist = n*(n-1)
        max_possible = n * (n - 1)
        cohesion_pct = (len(connections) / max_possible * 100.0) if max_possible > 0 else 0.0

        # Missing links: pairs with no positive edge but with shared neighbours
        connection_set = {(a, b) for a, b, _ in connections}
        missing: List[Tuple[str, str, int]] = []
        for i, slug_a in enumerate(known):
            for slug_b in known[i + 1:]:
                # Check both directions
                if (slug_a, slug_b) not in connection_set and (slug_b, slug_a) not in connection_set:
                    # Count shared positive neighbours (outside the stack)
                    nb_a = self._positive_neighbors.get(slug_a, set())
                    nb_b = self._positive_neighbors.get(slug_b, set())
                    shared = len(nb_a & nb_b - stack_set)
                    if shared > 0:
                        missing.append((slug_a, slug_b, shared))

        missing.sort(key=lambda x: -x[2])
        missing_links = [(a, b) for a, b, _ in missing]

        bridge_tools = [d["slug"] for d in self.find_bridge_tools(known, limit=5)]

        return {
            "cohesion_pct": round(cohesion_pct, 2),
            "connections": connections,
            "conflicts": conflicts,
            "missing_links": missing_links,
            "bridge_tools": bridge_tools,
        }

    def find_bridge_tools(self, slugs: List[str], limit: int = 5) -> List[Dict]:
        """
        Find external tools that would most improve stack connectivity.

        A bridge tool scores points for each stack member it has a positive
        edge to (in either direction).  Only tools NOT already in the stack
        are considered.

        Parameters
        ----------
        slugs : list[str]
            The current stack.
        limit : int
            Maximum number of bridge tools to return.

        Returns
        -------
        list[dict]
            Each dict: ``{"slug": str, "connects_to": int, "edge_types": list[str]}``
            sorted by *connects_to* descending.
        """
        stack_set = set(s for s in slugs if s in self._slugs)
        if not stack_set:
            return []

        # Candidate bridge tools: every known slug NOT in the stack
        candidates: Dict[str, Dict] = {}
        for candidate in self._slugs - stack_set:
            edge_types_found: Set[str] = set()
            connects_to = 0
            for stack_slug in stack_set:
                for et in POSITIVE_EDGE_TYPES:
                    # candidate → stack member
                    if stack_slug in self._adj[et].get(candidate, []):
                        connects_to += 1
                        edge_types_found.add(et)
                        break
                    # stack member → candidate
                    if candidate in self._adj[et].get(stack_slug, []):
                        connects_to += 1
                        edge_types_found.add(et)
                        break
            if connects_to >= 2:
                candidates[candidate] = {
                    "slug": candidate,
                    "connects_to": connects_to,
                    "edge_types": sorted(edge_types_found),
                }

        ranked = sorted(candidates.values(), key=lambda x: -x["connects_to"])
        return ranked[:limit]

    # -----------------------------------------------------------------------
    # Link prediction
    # -----------------------------------------------------------------------

    def predict_links(self, limit: int = 50) -> List[Dict]:
        """
        Predict missing positive edges using the Jaccard coefficient on
        combined positive neighbour sets (forward + reverse, all positive
        edge types).

        For a pair (u, v):

            Jaccard(u, v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|

        where N(x) is the union of all known positive neighbours of x
        (inbound + outbound, excluding x itself).

        Only pairs where no positive edge currently exists in either
        direction are considered.  Pairs must have |N(u) ∩ N(v)| >= 1
        (at least one shared neighbour) to appear in results.

        Parameters
        ----------
        limit : int
            Maximum number of predictions to return.

        Returns
        -------
        list[dict]
            Each dict: ``{"slug_a": str, "slug_b": str, "score": float,
            "shared_neighbors": int}`` sorted by *score* descending.
        """
        nodes = list(self._slugs)
        n = len(nodes)

        # Filter to nodes that have at least one positive-edge neighbour
        # (isolated nodes would always score 0 — skip for efficiency)
        active = [slug for slug in nodes if self._positive_neighbors[slug]]
        m = len(active)

        # Build the existing positive-edge set for quick "edge exists?" lookup
        existing: Set[Tuple[str, str]] = set()
        for et in POSITIVE_EDGE_TYPES:
            for src, targets in self._adj[et].items():
                for tgt in targets:
                    existing.add((src, tgt))
                    existing.add((tgt, src))  # treat as undirected for "exists"

        results: List[Dict] = []

        # O(m²) but m ≤ 244 so worst case ~30k iterations — well under 1 s
        for i in range(m):
            slug_a = active[i]
            nb_a = self._positive_neighbors[slug_a]
            for j in range(i + 1, m):
                slug_b = active[j]
                # Skip pairs that already have a positive edge
                if (slug_a, slug_b) in existing:
                    continue
                nb_b = self._positive_neighbors[slug_b]
                inter = nb_a & nb_b
                shared = len(inter)
                if shared == 0:
                    continue
                union_size = len(nb_a | nb_b)
                jaccard = shared / union_size if union_size > 0 else 0.0
                results.append(
                    {
                        "slug_a": slug_a,
                        "slug_b": slug_b,
                        "score": round(jaccard, 4),
                        "shared_neighbors": shared,
                    }
                )

        results.sort(key=lambda x: (-x["score"], -x["shared_neighbors"]))
        return results[:limit]

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def stats(self) -> Dict:
        """
        Return summary statistics for the graph.

        Returns
        -------
        dict with keys:
            nodes : int                  Total tools in the graph.
            edges : dict[str, int]       Edge count per type (known slugs only).
            total_edges : int            Sum across all edge types.
            avg_out_degree : float       Mean outbound degree (all edge types).
            avg_positive_degree : float  Mean degree on the positive subgraph.
            isolated_nodes : int         Nodes with zero positive-edge connections.
            dangling_refs_skipped : int  Edges that referenced unknown slugs
                                         (counted at construction time implicitly;
                                         reported as total_edges_raw - total_edges).
            most_connected : list[dict]  Top 10 nodes by positive-edge degree.
        """
        n = len(self._slugs)

        edge_counts: Dict[str, int] = {}
        for et in ALL_EDGE_TYPES:
            edge_counts[et] = sum(len(v) for v in self._adj[et].values())

        total_edges = sum(edge_counts.values())

        total_out = sum(
            sum(len(v) for v in self._adj[et].values())
            for et in ALL_EDGE_TYPES
        )
        avg_out = total_out / n if n > 0 else 0.0

        # Positive degree = |forward positive edges| + |reverse positive edges|
        pos_degrees = {
            slug: len(self._positive_neighbors[slug])
            for slug in self._slugs
        }
        avg_pos = sum(pos_degrees.values()) / n if n > 0 else 0.0
        isolated = sum(1 for d in pos_degrees.values() if d == 0)

        top10 = sorted(pos_degrees.items(), key=lambda x: -x[1])[:10]
        most_connected = [
            {"slug": slug, "positive_degree": deg} for slug, deg in top10
        ]

        return {
            "nodes": n,
            "edges": edge_counts,
            "total_edges": total_edges,
            "avg_out_degree": round(avg_out, 2),
            "avg_positive_degree": round(avg_pos, 2),
            "isolated_nodes": isolated,
            "most_connected": most_connected,
        }
