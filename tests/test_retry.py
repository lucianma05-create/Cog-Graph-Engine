"""Conditional-retry tests: rewrite the utterance against the REAL post-apply
graph when the engine rejected substantive ops or fired ⚠ audit warnings."""

import json
import unittest
from pathlib import Path
from unittest import mock

from engine.schema import (Edge, EdgeUpdate, InitOutput, NodeDraft, NodeUpdate,
                           RewriteOutput, Seed, TurnOutput)
from engine.simulator import needs_rewrite
from engine.updater import build_initial_graph

ROOT = Path(__file__).parent.parent

SEED = Seed.model_validate(json.loads((ROOT / "seeds" / "craigslist_01.json")
                                      .read_text(encoding="utf-8")))


class TestNeedsRewrite(unittest.TestCase):
    def test_substantive_rejection_triggers(self):
        rej = [{"op": {"op": "add", "edge": {"from": "I1", "to": "B1",
                                             "relation": "facilitates"}},
                "reason": "illegal relation facilitates for endpoint types"}]
        self.assertTrue(needs_rewrite(rej, []))

    def test_commitment_guard_triggers(self):
        rej = [{"op": {"op": "deactivate", "node_id": "I1"},
                "reason": "commitment guard: I1 still serves active desire D1"}]
        self.assertTrue(needs_rewrite(rej, []))

    def test_benign_jitter_does_not_trigger(self):
        rej = [{"op": {"op": "update", "node_id": "B1"},
                "reason": "below significance threshold (|Δ|=0.15 < 0.4)"}]
        self.assertFalse(needs_rewrite(rej, []))

    def test_audit_warning_triggers(self):
        self.assertTrue(needs_rewrite([], ["⚠ price offer moved up with no worth-belief rise"]))

    def test_plain_notes_do_not_trigger(self):
        self.assertFalse(needs_rewrite([], ["id_renamed B1 -> B2"]))


class TestConditionalRetry(unittest.TestCase):
    """End-to-end: first call proposes an illegal edge (substantive rejection)
    -> the retry path rewrites the utterance via generate_rewrite; the turn
    record carries the rewritten utterance and retries=1."""

    def test_step_rewrites_utterance_after_rejection(self):
        g0 = build_initial_graph(InitOutput(
            nodes=[NodeDraft(id="B1", type="belief", content="card is new", strength=3.0),
                   NodeDraft(id="D1", type="desire", content="I want to pay around $69", strength=2.6)],
            edges=[Edge(frm="B1", to="D1", relation="facilitates")],
        )).graph

        bad = TurnOutput(
            node_updates=[],
            edge_updates=[EdgeUpdate(op="add", edge=Edge(
                frm="D1", to="B1", relation="facilitates"))],  # illegal D->B
            appraisal={"goal_congruence": 0.0, "controllability": 0.5, "goal_conflict": 0.0},
            emotion={"category": "neutral", "valence": 0.0, "arousal": 0.2,
                     "appraisal_target": "#agent_reply"},
            user_utterance="I now think the card is worth more because I want it.",
            done=False,
        )

        import engine.simulator as sim
        sim_obj = sim.UserSimulator(SEED, None)
        sim_obj.graph = g0
        sim_obj.initial_log = {"graph": g0.snapshot(), "ops_rejected": [], "notes": []}
        with mock.patch.object(sim.llm, "generate_turn",
                               return_value=(bad, {"edge_updates": []})), \
             mock.patch.object(sim.llm, "generate_rewrite", return_value=(
                 RewriteOutput(user_utterance="My offer is $69 and I will hold there.",
                               done=False), {"user_utterance": "x"})):
            turn = sim_obj.step("What is your offer?", mode="manual")

        self.assertEqual(turn["retries"], 1)
        self.assertEqual(turn["user_utterance"], "My offer is $69 and I will hold there.")
        self.assertIn("retries", turn["validation"])
        # the illegal edge was rejected and never entered the graph
        self.assertNotIn(("D1", "B1", "facilitates"), sim_obj.graph.edges)
        self.assertTrue(any("illegal relation" in r["reason"] for r in turn["ops_rejected"]))


if __name__ == "__main__":
    unittest.main()
