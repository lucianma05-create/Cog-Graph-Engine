import unittest

from engine.schema import Edge, EdgeUpdate, InitOutput, NodeDraft, NodeUpdate
from engine.updater import CognitiveGraph, apply_updates, build_initial_graph


def up(node_id=None, level_probs=None, content=None, op="update", node=None):
    return NodeUpdate(op=op, node_id=node_id, level_probs=level_probs,
                      content=content, node=node)


def eup(op, frm, to, rel):
    return EdgeUpdate(op=op, edge=Edge(frm=frm, to=to, relation=rel))


def draft(nid, ntype, content, probs):
    return NodeDraft(id=nid, type=ntype, content=content, level_probs=probs)


def graph_with_b1d1i1():
    init = InitOutput(
        nodes=[draft("B1", "belief", "charity is trustworthy", [0, 0.1, 0.2, 0.3, 0.4]),
               draft("D1", "desire", "help children", [0.1, 0.2, 0.3, 0.3, 0.1]),
               draft("I1", "intention", "donate $10", [0.5, 0.3, 0.2, 0, 0])],
        edges=[Edge(frm="B1", to="I1", relation="facilitates"),
               Edge(frm="I1", to="D1", relation="means_for")],
    )
    result = build_initial_graph(init)
    assert not result.ops_rejected, result.ops_rejected
    return result.graph


class TestUpdaterBasics(unittest.TestCase):
    def test_build_initial_graph(self):
        g = graph_with_b1d1i1()
        self.assertEqual(len(g.active_nodes()), 3)
        self.assertEqual(len(g.edges), 2)
        self.assertAlmostEqual(g.nodes["B1"].strength, 3.0, places=3)

    def test_update_strength_and_delta(self):
        g = graph_with_b1d1i1()
        r = apply_updates(g, [up("B1", level_probs=[0, 0, 0, 1, 0])], [])
        self.assertNotIn("B1", r.deltas)  # 3.0 -> 3.0, no change, no delta entry
        self.assertEqual(r.graph.nodes["B1"].strength, 3.0)
        r2 = apply_updates(g, [up("B1", level_probs=[1, 0, 0, 0, 0])], [])
        self.assertAlmostEqual(r2.deltas["B1"], -3.0, places=3)
        self.assertIn("B1", r2.graph.deactivated)  # auto-deactivate at strength 0
        auto = [o for o in r2.ops_applied if o["op"] == "deactivate"]
        self.assertEqual(len(auto), 1)
        self.assertTrue(auto[0]["auto"])

    def test_content_only_update_no_delta(self):
        g = graph_with_b1d1i1()
        r = apply_updates(g, [up("B1", content="charity is trustworthy, mostly")], [])
        self.assertNotIn("B1", r.deltas)
        self.assertEqual(r.graph.nodes["B1"].content, "charity is trustworthy, mostly")
        self.assertAlmostEqual(r.graph.nodes["B1"].strength, 3.0, places=3)

    def test_noise_jitter_rejected(self):
        """Strength moves below MIN_ABS_DELTA are rejected as noise."""
        g = graph_with_b1d1i1()  # B1 strength 3.0
        r = apply_updates(g, [up("B1", level_probs=[0, 0, 0.05, 0.8, 0.15])], [])  # 3.1
        self.assertEqual(len(r.ops_rejected), 1)
        self.assertIn("significance threshold", r.ops_rejected[0]["reason"])
        self.assertAlmostEqual(r.graph.nodes["B1"].strength, 3.0, places=3)  # unchanged
        # a real move passes
        r2 = apply_updates(g, [up("B1", level_probs=[0, 0, 0, 0.4, 0.6])], [])  # 3.6
        self.assertEqual(len(r2.ops_rejected), 0)
        self.assertAlmostEqual(r2.deltas["B1"], 0.6, places=3)

    def test_jitter_applied_when_content_changes(self):
        """A content edit is evidence of a real cognitive change: its
        sub-threshold strength move is APPLIED (facts-driven small moves).
        Re-sending the same content is no bypass."""
        g = graph_with_b1d1i1()
        r = apply_updates(g, [up("B1", content="charity now seems less trustworthy",
                                 level_probs=[0, 0, 0.02, 0.8, 0.18])], [])  # 3.0 -> 3.16
        self.assertEqual(len(r.ops_rejected), 0)
        self.assertEqual(r.graph.nodes["B1"].content, "charity now seems less trustworthy")
        self.assertAlmostEqual(r.graph.nodes["B1"].strength, 3.16, places=3)  # applied
        self.assertAlmostEqual(r.deltas["B1"], 0.16, places=3)
        self.assertFalse(any("frozen" in n for n in r.notes))
        # same content re-sent with a jitter: no content change => rejected
        r2 = apply_updates(g, [up("B1", content="charity is trustworthy",
                                  level_probs=[0, 0, 0.02, 0.8, 0.18])], [])
        self.assertEqual(len(r2.ops_rejected), 1)
        self.assertIn("significance threshold", r2.ops_rejected[0]["reason"])

    def test_reactivation_bypasses_threshold(self):
        g = graph_with_b1d1i1()
        g2 = apply_updates(g, [up("B1", op="deactivate")], []).graph
        r2 = apply_updates(g2, [up("B1", level_probs=[0.2, 0.3, 0.3, 0.2, 0])], [])  # 1.5
        self.assertEqual(len(r2.ops_rejected), 0)
        self.assertNotIn("B1", r2.graph.deactivated)
        self.assertAlmostEqual(r2.deltas["B1"], 1.5, places=3)

    def test_deactivate_keeps_id_and_cascades_edges(self):
        g = graph_with_b1d1i1()
        r = apply_updates(g, [up("B1", op="deactivate")], [])
        self.assertIn("B1", r.graph.deactivated)
        self.assertIn("B1", r.graph.nodes)          # id kept
        self.assertEqual(r.graph.nodes["B1"].strength, 0.0)
        self.assertNotIn(("B1", "I1", "facilitates"), r.graph.edges)  # cascade
        removed = [o for o in r.ops_applied if o["op"] == "remove"]
        self.assertEqual(len(removed), 1)
        self.assertTrue(removed[0]["auto"])

    def test_reactivate_deactivated_node(self):
        g = graph_with_b1d1i1()
        g = apply_updates(g, [up("B1", op="deactivate")], []).graph
        r = apply_updates(g, [up("B1", level_probs=[0, 0, 0, 0, 1])], [])
        self.assertNotIn("B1", r.graph.deactivated)
        self.assertAlmostEqual(r.graph.nodes["B1"].strength, 4.0, places=3)
        self.assertAlmostEqual(r.deltas["B1"], 4.0, places=3)  # baseline 0

    def test_empty_updates_neutral_turn(self):
        g = graph_with_b1d1i1()
        r = apply_updates(g, [], [])
        self.assertEqual(r.ops_applied, [])
        self.assertEqual(r.graph.snapshot(), g.snapshot())

    def test_deterministic(self):
        g = graph_with_b1d1i1()
        node_ops = [up("B1", level_probs=[0, 0, 0.2, 0.4, 0.4]),
                    up(op="add", node=draft("D2", "desire", "save money", [0, 0, 0, 1, 0]))]
        edge_ops = [eup("add", "D1", "D2", "conflicts_with")]
        r1 = apply_updates(g, node_ops, edge_ops)
        r2 = apply_updates(g, node_ops, edge_ops)
        self.assertEqual(r1.graph.snapshot(), r2.graph.snapshot())
        self.assertEqual(r1.deltas, r2.deltas)
        self.assertEqual(r1.ops_applied, r2.ops_applied)


class TestUpdaterValidation(unittest.TestCase):
    def test_dangling_node_id_rejected(self):
        g = graph_with_b1d1i1()
        r = apply_updates(g, [up("B9", level_probs=[0, 0, 0, 1, 0])], [])
        self.assertEqual(len(r.ops_rejected), 1)
        self.assertIn("unknown node_id", r.ops_rejected[0]["reason"])

    def test_update_deactivated_rejected_for_deactivate_op(self):
        g = graph_with_b1d1i1()
        g = apply_updates(g, [up("B1", op="deactivate")], []).graph
        r = apply_updates(g, [up("B1", op="deactivate")], [])
        self.assertEqual(len(r.ops_rejected), 1)
        self.assertIn("already deactivated", r.ops_rejected[0]["reason"])

    def test_duplicate_add_id_renamed(self):
        g = graph_with_b1d1i1()
        r = apply_updates(g, [up(op="add", node=draft("B1", "belief", "new thing", [0, 0, 0, 1, 0]))], [])
        self.assertEqual(len(r.ops_applied), 1)
        new_id = r.ops_applied[0]["node_id"]
        self.assertNotEqual(new_id, "B1")
        self.assertTrue(any(n.startswith("id_renamed B1 ->") for n in r.notes))

    def test_id_prefix_type_mismatch_rejected(self):
        # pydantic already rejects this at the LLM boundary; the updater's
        # defensive path is exercised via unvalidated (model_construct) ops.
        g = graph_with_b1d1i1()
        raw_node = NodeDraft.model_construct(id="D9", type="belief", content="x",
                                             level_probs=[0, 0, 0, 1, 0])
        raw_op = NodeUpdate.model_construct(op="add", node_id=None, node=raw_node,
                                            level_probs=None, content=None)
        r = apply_updates(g, [raw_op], [])
        self.assertEqual(len(r.ops_rejected), 1)

    def test_bad_level_probs_rejected(self):
        g = graph_with_b1d1i1()
        raw_node = NodeDraft.model_construct(id="B5", type="belief", content="x",
                                             level_probs=[0.1, 0.1])
        raw_op = NodeUpdate.model_construct(op="add", node_id=None, node=raw_node,
                                            level_probs=None, content=None)
        r = apply_updates(g, [raw_op], [])
        self.assertEqual(len(r.ops_rejected), 1)
        raw2 = NodeDraft.model_construct(id="B5", type="belief", content="x",
                                         level_probs=[-0.1, 0, 0, 1, 0.1])
        raw_op2 = NodeUpdate.model_construct(op="add", node_id=None, node=raw2,
                                             level_probs=None, content=None)
        r = apply_updates(g, [raw_op2], [])
        self.assertEqual(len(r.ops_rejected), 1)

    def test_means_for_illegal_pair_rejected(self):
        g = graph_with_b1d1i1()
        r = apply_updates(g, [], [eup("add", "B1", "D1", "means_for")])
        self.assertEqual(len(r.ops_rejected), 1)
        self.assertIn("illegal relation", r.ops_rejected[0]["reason"])
        ok = apply_updates(g, [], [eup("add", "I1", "D1", "means_for")])  # already exists
        self.assertIn("duplicate", ok.ops_rejected[0]["reason"])

    def test_forward_directionality_only(self):
        """facilitates/inhibits: B->D, D->I, B->I legal; same-level and
        backward pairs illegal (user-specified rule: cognition flows B->D->I)."""
        # legal forward pairs (fresh graph per pair; B1->I1 already exists in
        # graph_with_b1d1i1, so use a relation-legal pair not yet present)
        for frm, to in [("B1", "D1"), ("D1", "I1"), ("B1", "I1")]:
            r = apply_updates(graph_with_b1d1i1(), [], [eup("add", frm, to, "facilitates")])
            if frm == "B1" and to == "I1":
                self.assertIn("duplicate", r.ops_rejected[0]["reason"])  # legal but dup
                continue
            self.assertEqual(len(r.ops_rejected), 0, f"{frm}->{to} should be legal: {r.ops_rejected}")
            self.assertEqual(len(r.ops_applied), 1)
        # illegal same-level / backward pairs (fresh graph each time)
        for frm, to in [("B1", "B1"), ("D1", "D1"), ("I1", "I1"),  # same-level
                        ("D1", "B1"), ("I1", "B1"), ("I1", "D1")]:  # backward
            r = apply_updates(graph_with_b1d1i1(), [], [eup("add", frm, to, "facilitates")])
            self.assertEqual(len(r.ops_rejected), 1, f"{frm}->{to} should be illegal")
            self.assertIn("illegal relation", r.ops_rejected[0]["reason"])
        # inhibits follows the same directionality
        r = apply_updates(graph_with_b1d1i1(), [], [eup("add", "D1", "B1", "inhibits")])
        self.assertEqual(len(r.ops_rejected), 1)
        r = apply_updates(graph_with_b1d1i1(), [], [eup("add", "B1", "D1", "inhibits")])
        self.assertEqual(len(r.ops_applied), 1)

    def test_conflicts_with_only_desires(self):
        g = graph_with_b1d1i1()
        r = apply_updates(g, [], [eup("add", "B1", "I1", "conflicts_with")])
        self.assertEqual(len(r.ops_rejected), 1)
        g2 = apply_updates(g, [up(op="add", node=draft("D2", "desire", "save money", [0, 0, 0, 1, 0]))], []).graph
        r2 = apply_updates(g2, [], [eup("add", "D1", "D2", "conflicts_with")])
        self.assertEqual(len(r2.ops_applied), 1)

    def test_conflicts_with_reverse_is_duplicate(self):
        g = graph_with_b1d1i1()
        g = apply_updates(g, [up(op="add", node=draft("D2", "desire", "save money", [0, 0, 0, 1, 0]))], []).graph
        g = apply_updates(g, [], [eup("add", "D1", "D2", "conflicts_with")]).graph
        r = apply_updates(g, [], [eup("add", "D2", "D1", "conflicts_with")])
        self.assertEqual(len(r.ops_rejected), 1)
        self.assertIn("duplicate", r.ops_rejected[0]["reason"])

    def test_remove_missing_edge_rejected(self):
        g = graph_with_b1d1i1()
        r = apply_updates(g, [], [eup("remove", "B1", "I1", "inhibits")])
        self.assertEqual(len(r.ops_rejected), 1)
        self.assertIn("edge not found", r.ops_rejected[0]["reason"])

    def test_edge_to_deactivated_endpoint_rejected(self):
        g = graph_with_b1d1i1()
        g = apply_updates(g, [up("B1", op="deactivate")], []).graph
        r = apply_updates(g, [], [eup("add", "B1", "D1", "facilitates")])
        self.assertEqual(len(r.ops_rejected), 1)
        self.assertIn("inactive", r.ops_rejected[0]["reason"])

    def test_same_turn_add_then_edge(self):
        g = graph_with_b1d1i1()
        r = apply_updates(
            g,
            [up(op="add", node=draft("D2", "desire", "save money", [0, 0, 0, 1, 0]))],
            [eup("add", "D1", "D2", "conflicts_with")],
        )
        self.assertEqual(len(r.ops_applied), 2)
        self.assertIn(("D1", "D2", "conflicts_with"), r.graph.edges)


if __name__ == "__main__":
    unittest.main()


class TestCommitmentGuard(unittest.TestCase):
    def test_deactivate_intention_with_live_means_for_rejected(self):
        g = graph_with_b1d1i1()
        r = apply_updates(g, [up("I1", op="deactivate")], [])
        self.assertEqual(len(r.ops_rejected), 1)
        self.assertIn("commitment guard", r.ops_rejected[0]["reason"])
        self.assertNotIn("I1", r.graph.deactivated)

    def test_update_to_zero_also_guarded(self):
        g = graph_with_b1d1i1()
        r = apply_updates(g, [up("I1", level_probs=[1, 0, 0, 0, 0])], [])
        self.assertEqual(len(r.ops_rejected), 1)
        self.assertIn("commitment guard", r.ops_rejected[0]["reason"])
        self.assertNotIn("I1", r.graph.deactivated)

    def test_desire_dying_same_turn_unblocks(self):
        g = graph_with_b1d1i1()
        r = apply_updates(g, [up("I1", op="deactivate"),
                              up("D1", level_probs=[1, 0, 0, 0, 0])], [])
        self.assertEqual(len(r.ops_rejected), 0)
        self.assertIn("I1", r.graph.deactivated)

    def test_means_for_edge_removed_same_turn_unblocks(self):
        g = graph_with_b1d1i1()
        r = apply_updates(g, [up("I1", op="deactivate")],
                          [eup("remove", "I1", "D1", "means_for")])
        self.assertEqual(len(r.ops_rejected), 0)
        self.assertIn("I1", r.graph.deactivated)

    def test_replacement_intention_unblocks(self):
        """A NEW intention adopted this turn WITH a means_for link to the SAME
        desire is a true replacement (the blanket exemption was removed in
        review: an unrelated or unlinked add must NOT unblock)."""
        g = graph_with_b1d1i1()
        r = apply_updates(g, [up("I1", op="deactivate"),
                              up(op="add", node=draft("I2", "intention", "donate $5",
                                                      [0.2, 0.3, 0.3, 0.2, 0]))],
                          [eup("add", "I2", "D1", "means_for")])
        self.assertEqual(len(r.ops_rejected), 0)
        self.assertIn("I1", r.graph.deactivated)

    def test_unlinked_intention_add_does_not_unblock(self):
        """Regression (code review): adding an unrelated intention must not
        exempt dropping a means_for-served intention."""
        g = graph_with_b1d1i1()
        r = apply_updates(g, [up("I1", op="deactivate"),
                              up(op="add", node=draft("I2", "intention", "buy milk",
                                                      [0.2, 0.3, 0.3, 0.2, 0]))], [])
        self.assertEqual(len(r.ops_rejected), 1)
        self.assertIn("commitment guard", r.ops_rejected[0]["reason"])
        self.assertNotIn("I1", r.graph.deactivated)

    def test_done_unblocks(self):
        g = graph_with_b1d1i1()
        r = apply_updates(g, [up("I1", op="deactivate")], [], done=True)
        self.assertEqual(len(r.ops_rejected), 0)
        self.assertIn("I1", r.graph.deactivated)

    def test_flexible_commitment_disables_guard(self):
        g = graph_with_b1d1i1()
        r = apply_updates(g, [up("I1", op="deactivate")], [], commitment="flexible")
        self.assertEqual(len(r.ops_rejected), 0)
        self.assertIn("I1", r.graph.deactivated)

    def test_belief_deactivation_never_blocked(self):
        g = graph_with_b1d1i1()
        r = apply_updates(g, [up("B1", op="deactivate")], [])
        self.assertEqual(len(r.ops_rejected), 0)
