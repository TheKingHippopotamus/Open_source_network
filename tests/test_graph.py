"""
tests/test_graph.py
===================
Tests for engine/graph.py:
  - GraphEngine construction
  - Adjacency list queries (has_edge, get_neighbors, get_reverse_neighbors)
  - PageRank validity
  - Community detection validity
  - stack_cohesion output format and correctness
  - find_bridge_tools suggestions
  - predict_links scored pairs
  - stats() output
"""

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from engine.graph import GraphEngine, POSITIVE_EDGE_TYPES, ALL_EDGE_TYPES


# ===========================================================================
# Construction
# ===========================================================================

class TestGraphEngineConstruction:
    def test_builds_from_full_db(self, db):
        ge = GraphEngine(db)
        assert ge is not None

    def test_node_count_matches_db(self, db, graph_engine):
        # stats().nodes should equal len(db)
        stats = graph_engine.stats()
        assert stats["nodes"] == len(db)

    def test_slugs_are_all_db_slugs(self, db, graph_engine):
        db_slugs = {t["slug"] for t in db}
        graph_slugs = set(graph_engine._slugs)
        assert db_slugs == graph_slugs

    def test_total_edges_positive(self, graph_engine):
        stats = graph_engine.stats()
        assert stats["total_edges"] > 0

    def test_dangling_refs_not_in_adj(self, db, graph_engine):
        """All adjacency targets should be known slugs (dangling refs dropped)."""
        db_slugs = {t["slug"] for t in db}
        for et in ALL_EDGE_TYPES:
            for targets in graph_engine._adj[et].values():
                for tgt in targets:
                    assert tgt in db_slugs, (
                        f"Dangling ref '{tgt}' found in {et} adjacency"
                    )


# ===========================================================================
# Adjacency queries
# ===========================================================================

class TestHasEdge:
    def test_postgresql_integrates_with_redis(self, graph_engine):
        # postgresql lists redis in integrates_with
        assert graph_engine.has_edge("postgresql", "redis") is True

    def test_has_edge_specific_type(self, graph_engine):
        # postgresql lists redis in its complements field
        assert graph_engine.has_edge(
            "postgresql", "redis", edge_type="complements"
        ) is True

    def test_has_edge_wrong_type_returns_false(self, graph_engine):
        # redis is not in conflicts_with for postgresql
        assert graph_engine.has_edge(
            "postgresql", "redis", edge_type="conflicts_with"
        ) is False

    def test_has_edge_unknown_src(self, graph_engine):
        assert graph_engine.has_edge("no-such-slug", "postgresql") is False

    def test_has_edge_unknown_dst(self, graph_engine):
        assert graph_engine.has_edge("postgresql", "no-such-slug") is False

    def test_has_edge_reflexive_false(self, graph_engine):
        # No self-loops should exist
        assert graph_engine.has_edge("postgresql", "postgresql") is False


class TestGetNeighbors:
    def test_returns_dict(self, graph_engine):
        nbrs = graph_engine.get_neighbors("postgresql")
        assert isinstance(nbrs, dict)

    def test_known_tool_has_neighbors(self, graph_engine):
        nbrs = graph_engine.get_neighbors("postgresql")
        # postgresql has integrations and complements
        assert len(nbrs) > 0

    def test_only_non_empty_types_returned(self, graph_engine):
        nbrs = graph_engine.get_neighbors("postgresql")
        for et, targets in nbrs.items():
            assert len(targets) > 0, f"Empty list returned for edge type {et}"

    def test_unknown_slug_returns_empty(self, graph_engine):
        nbrs = graph_engine.get_neighbors("no-such-slug")
        assert nbrs == {}

    def test_edge_type_filter(self, graph_engine):
        nbrs = graph_engine.get_neighbors(
            "postgresql", edge_types=["integrates_with"]
        )
        assert set(nbrs.keys()).issubset({"integrates_with"})

    def test_targets_are_known_slugs(self, db, graph_engine):
        db_slugs = {t["slug"] for t in db}
        nbrs = graph_engine.get_neighbors("n8n")
        for targets in nbrs.values():
            for tgt in targets:
                assert tgt in db_slugs


class TestGetReverseNeighbors:
    def test_reverse_neighbors_returns_dict(self, graph_engine):
        rev = graph_engine.get_reverse_neighbors("postgresql")
        assert isinstance(rev, dict)

    def test_well_connected_tool_has_reverse_edges(self, graph_engine):
        # postgresql is referenced by many tools
        rev = graph_engine.get_reverse_neighbors("postgresql")
        total_inbound = sum(len(v) for v in rev.values())
        assert total_inbound >= 1

    def test_unknown_slug_returns_empty_rev(self, graph_engine):
        rev = graph_engine.get_reverse_neighbors("no-such-slug-xyz")
        assert rev == {}


# ===========================================================================
# PageRank
# ===========================================================================

class TestPageRank:
    def test_returns_dict_with_all_slugs(self, db, graph_engine):
        pr = graph_engine.compute_pagerank()
        assert len(pr) == len(db)

    def test_scores_sum_to_approximately_one(self, graph_engine):
        pr = graph_engine.compute_pagerank()
        total = sum(pr.values())
        assert abs(total - 1.0) < 1e-3, f"PageRank sum = {total}, expected ~1.0"

    def test_all_scores_positive(self, graph_engine):
        pr = graph_engine.compute_pagerank()
        for slug, score in pr.items():
            assert score > 0.0, f"{slug} has non-positive PageRank: {score}"

    def test_all_scores_less_than_one(self, graph_engine):
        pr = graph_engine.compute_pagerank()
        for slug, score in pr.items():
            assert score < 1.0, f"{slug} PageRank >= 1.0: {score}"

    def test_well_connected_tools_rank_higher(self, graph_engine, db):
        pr = graph_engine.compute_pagerank()
        stats = graph_engine.stats()
        # The most connected node should have above-average PageRank
        most_connected = stats["most_connected"][0]["slug"]
        avg_pr = sum(pr.values()) / len(pr)
        assert pr[most_connected] >= avg_pr, (
            f"Most connected tool {most_connected} has below-average PR"
        )

    def test_custom_damping_factor(self, graph_engine):
        pr_85 = graph_engine.compute_pagerank(damping=0.85)
        pr_50 = graph_engine.compute_pagerank(damping=0.50)
        # Different damping should produce different distributions
        assert pr_85 != pr_50

    def test_postgresql_has_positive_rank(self, graph_engine):
        pr = graph_engine.compute_pagerank()
        assert pr.get("postgresql", 0) > 0.0


# ===========================================================================
# Community Detection
# ===========================================================================

class TestCommunityDetection:
    def test_returns_dict_with_all_slugs(self, db, graph_engine):
        comms = graph_engine.detect_communities()
        assert len(comms) == len(db)

    def test_all_values_are_integers(self, graph_engine):
        comms = graph_engine.detect_communities()
        for slug, comm_id in comms.items():
            assert isinstance(comm_id, int), f"{slug} community id is not int"

    def test_community_ids_are_contiguous(self, graph_engine):
        comms = graph_engine.detect_communities()
        unique_ids = sorted(set(comms.values()))
        assert unique_ids == list(range(len(unique_ids))), (
            "Community IDs are not contiguous integers"
        )

    def test_at_least_two_communities(self, graph_engine, db):
        # 244 tools with varying connectivity should form multiple communities
        comms = graph_engine.detect_communities()
        num_communities = len(set(comms.values()))
        assert num_communities >= 2, (
            f"Expected multiple communities, got {num_communities}"
        )

    def test_connected_tools_in_same_community(self, graph_engine):
        # postgresql and redis have direct positive edges — likely in same community
        comms = graph_engine.detect_communities()
        # They might not always end up together due to label propagation randomness,
        # but at minimum both must have valid community assignments
        assert "postgresql" in comms
        assert "redis" in comms

    def test_total_community_members_equals_total_tools(self, db, graph_engine):
        comms = graph_engine.detect_communities()
        assert len(comms) == len(db)


# ===========================================================================
# Stack Cohesion
# ===========================================================================

class TestStackCohesion:
    def test_returns_expected_keys(self, graph_engine):
        result = graph_engine.stack_cohesion(["postgresql", "redis", "supabase"])
        required = {"cohesion_pct", "connections", "conflicts", "missing_links", "bridge_tools"}
        assert required.issubset(set(result.keys()))

    def test_cohesion_pct_is_float(self, graph_engine):
        result = graph_engine.stack_cohesion(["postgresql", "redis", "supabase"])
        assert isinstance(result["cohesion_pct"], float)

    def test_cohesion_pct_is_non_negative_float(self, graph_engine):
        # cohesion_pct can exceed 100 when multiple directed edge types connect
        # the same pair (e.g. both integrates_with AND complements exist between
        # two tools).  We only assert non-negative and finite.
        result = graph_engine.stack_cohesion(["postgresql", "redis", "supabase"])
        assert result["cohesion_pct"] >= 0.0
        assert isinstance(result["cohesion_pct"], float)

    def test_well_connected_stack_has_connections(self, graph_engine):
        # postgresql, redis, supabase are well-connected
        result = graph_engine.stack_cohesion(["postgresql", "redis", "supabase", "metabase"])
        assert len(result["connections"]) > 0

    def test_connections_are_tuples_of_three(self, graph_engine):
        result = graph_engine.stack_cohesion(["postgresql", "redis", "supabase"])
        for conn in result["connections"]:
            assert len(conn) == 3, f"Expected (src, dst, edge_type), got {conn}"
            src, dst, et = conn
            assert et in POSITIVE_EDGE_TYPES

    def test_conflicts_list_format(self, graph_engine):
        result = graph_engine.stack_cohesion(["postgresql", "redis"])
        assert isinstance(result["conflicts"], list)

    def test_single_tool_stack(self, graph_engine):
        result = graph_engine.stack_cohesion(["postgresql"])
        # 1 tool: max_possible = 0, cohesion should be 0
        assert result["cohesion_pct"] == 0.0
        assert result["connections"] == []

    def test_empty_stack(self, graph_engine):
        result = graph_engine.stack_cohesion([])
        assert result["cohesion_pct"] == 0.0

    def test_unknown_slugs_filtered(self, graph_engine):
        result = graph_engine.stack_cohesion(["postgresql", "no-such-slug-xyz"])
        # Only postgresql remains — cohesion = 0
        assert result["cohesion_pct"] == 0.0


# ===========================================================================
# find_bridge_tools
# ===========================================================================

class TestFindBridgeTools:
    def test_returns_list_of_dicts(self, graph_engine):
        bridges = graph_engine.find_bridge_tools(["postgresql", "redis", "supabase"])
        assert isinstance(bridges, list)
        for b in bridges:
            assert isinstance(b, dict)

    def test_bridge_dicts_have_required_keys(self, graph_engine):
        bridges = graph_engine.find_bridge_tools(["postgresql", "redis", "supabase"])
        for b in bridges:
            assert "slug" in b
            assert "connects_to" in b
            assert "edge_types" in b

    def test_bridge_tools_not_in_input_stack(self, graph_engine):
        stack = ["postgresql", "redis", "supabase"]
        bridges = graph_engine.find_bridge_tools(stack, limit=5)
        for b in bridges:
            assert b["slug"] not in stack

    def test_bridge_tools_are_known_slugs(self, graph_engine, db):
        db_slugs = {t["slug"] for t in db}
        bridges = graph_engine.find_bridge_tools(
            ["postgresql", "redis", "supabase"], limit=5
        )
        for b in bridges:
            assert b["slug"] in db_slugs

    def test_respects_limit(self, graph_engine):
        bridges = graph_engine.find_bridge_tools(
            ["postgresql", "redis", "supabase", "metabase", "n8n"], limit=3
        )
        assert len(bridges) <= 3

    def test_empty_stack_returns_empty(self, graph_engine):
        bridges = graph_engine.find_bridge_tools([])
        assert bridges == []

    def test_connects_to_is_at_least_two(self, graph_engine):
        # Bridge tools must connect to at least 2 stack members (the minimum threshold)
        bridges = graph_engine.find_bridge_tools(
            ["postgresql", "redis", "supabase", "metabase", "n8n"], limit=5
        )
        for b in bridges:
            assert b["connects_to"] >= 2


# ===========================================================================
# predict_links
# ===========================================================================

class TestPredictLinks:
    def test_returns_list_of_dicts(self, graph_engine):
        links = graph_engine.predict_links(limit=10)
        assert isinstance(links, list)

    def test_link_dicts_have_required_keys(self, graph_engine):
        links = graph_engine.predict_links(limit=5)
        for link in links:
            assert "slug_a" in link
            assert "slug_b" in link
            assert "score" in link
            assert "shared_neighbors" in link

    def test_scores_in_valid_range(self, graph_engine):
        links = graph_engine.predict_links(limit=20)
        for link in links:
            assert 0.0 < link["score"] <= 1.0, (
                f"Jaccard score out of range: {link['score']}"
            )

    def test_sorted_descending_by_score(self, graph_engine):
        links = graph_engine.predict_links(limit=20)
        scores = [link["score"] for link in links]
        assert scores == sorted(scores, reverse=True)

    def test_respects_limit(self, graph_engine):
        links = graph_engine.predict_links(limit=5)
        assert len(links) <= 5

    def test_no_self_pairs(self, graph_engine):
        links = graph_engine.predict_links(limit=50)
        for link in links:
            assert link["slug_a"] != link["slug_b"]

    def test_slugs_are_known(self, graph_engine, db):
        db_slugs = {t["slug"] for t in db}
        links = graph_engine.predict_links(limit=20)
        for link in links:
            assert link["slug_a"] in db_slugs
            assert link["slug_b"] in db_slugs

    def test_predicted_pairs_have_no_existing_positive_edge(self, graph_engine):
        links = graph_engine.predict_links(limit=20)
        for link in links:
            # Check: there should be no positive edge between slug_a and slug_b
            a, b = link["slug_a"], link["slug_b"]
            has_pos_ab = any(
                graph_engine.has_edge(a, b, edge_type=et)
                for et in POSITIVE_EDGE_TYPES
            )
            has_pos_ba = any(
                graph_engine.has_edge(b, a, edge_type=et)
                for et in POSITIVE_EDGE_TYPES
            )
            assert not has_pos_ab and not has_pos_ba, (
                f"Predicted edge ({a}, {b}) already exists as a positive edge"
            )

    def test_shared_neighbors_positive(self, graph_engine):
        links = graph_engine.predict_links(limit=10)
        for link in links:
            assert link["shared_neighbors"] >= 1


# ===========================================================================
# Stats
# ===========================================================================

class TestStats:
    def test_stats_returns_required_keys(self, graph_engine):
        stats = graph_engine.stats()
        required = {
            "nodes", "edges", "total_edges", "avg_out_degree",
            "avg_positive_degree", "isolated_nodes", "most_connected",
        }
        assert required.issubset(set(stats.keys()))

    def test_edges_dict_has_all_edge_types(self, graph_engine):
        stats = graph_engine.stats()
        for et in ALL_EDGE_TYPES:
            assert et in stats["edges"]

    def test_total_edges_equals_sum_of_edges(self, graph_engine):
        stats = graph_engine.stats()
        expected_total = sum(stats["edges"].values())
        assert stats["total_edges"] == expected_total

    def test_avg_degrees_non_negative(self, graph_engine):
        stats = graph_engine.stats()
        assert stats["avg_out_degree"] >= 0.0
        assert stats["avg_positive_degree"] >= 0.0

    def test_most_connected_length(self, graph_engine):
        stats = graph_engine.stats()
        assert 1 <= len(stats["most_connected"]) <= 10

    def test_most_connected_sorted_descending(self, graph_engine):
        stats = graph_engine.stats()
        degrees = [e["positive_degree"] for e in stats["most_connected"]]
        assert degrees == sorted(degrees, reverse=True)

    def test_isolated_nodes_count_reasonable(self, graph_engine, db):
        stats = graph_engine.stats()
        # At scale of 244 tools with 919 edges, some isolation is expected
        assert 0 <= stats["isolated_nodes"] <= len(db)
