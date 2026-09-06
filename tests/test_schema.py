"""Lock the insight.md schema: the rejected 0904 fields/edges must fail."""

import copy
import unittest

from pydantic import ValidationError

from engine.schema import (InitOutput, Node, NodeDraft, Seed, StepRequest,
                           TurnOutput)

VALID_TURN = {
    "node_updates": [
        {"op": "update", "node_id": "B1", "strength": 3.0},
        {"op": "add", "node": {"id": "I2", "type": "intention",
                               "content": "look at other jobs", "strength": 3.0}},
    ],
    "edge_updates": [
        {"op": "add", "edge": {"from": "B1", "to": "I2", "relation": "facilitates"}},
    ],
    "appraisal": {"goal_congruence": 0.6, "controllability": 0.3, "goal_conflict": 0.2},
    "emotion": {"category": "hope", "valence": 0.4, "arousal": 0.3,
                "appraisal_target": "#agent_reply"},
    "user_utterance": "That actually helps.",
    "done": False,
}


class TestTurnOutput(unittest.TestCase):
    def test_valid_turn(self):
        t = TurnOutput.model_validate(VALID_TURN)
        self.assertEqual(t.node_updates[0].node_id, "B1")
        self.assertEqual(t.user_utterance, "That actually helps.")

    def test_0904_fields_rejected_on_node(self):
        bad = copy.deepcopy(VALID_TURN)
        bad["node_updates"][1]["node"]["confidence"] = 0.8
        with self.assertRaises(ValidationError):
            TurnOutput.model_validate(bad)
        bad = copy.deepcopy(VALID_TURN)
        bad["node_updates"][1]["node"]["observability"] = "explicit"
        with self.assertRaises(ValidationError):
            TurnOutput.model_validate(bad)
        bad = copy.deepcopy(VALID_TURN)
        bad["node_updates"][1]["node"]["importance"] = 0.5
        with self.assertRaises(ValidationError):
            TurnOutput.model_validate(bad)
        bad = copy.deepcopy(VALID_TURN)
        bad["node_updates"][1]["node"]["goal_impact"] = 0.3
        with self.assertRaises(ValidationError):
            TurnOutput.model_validate(bad)

    def test_0904_relations_rejected(self):
        for rel in ("based_on", "elicits"):
            bad = copy.deepcopy(VALID_TURN)
            bad["edge_updates"][0]["edge"]["relation"] = rel
            with self.assertRaises(ValidationError):
                TurnOutput.model_validate(bad)

    def test_add_node_requires_strength(self):
        bad = copy.deepcopy(VALID_TURN)
        del bad["node_updates"][1]["node"]["strength"]
        with self.assertRaises(ValidationError):
            TurnOutput.model_validate(bad)

    def test_add_requires_node(self):
        bad = copy.deepcopy(VALID_TURN)
        bad["node_updates"][0] = {"op": "add"}
        with self.assertRaises(ValidationError):
            TurnOutput.model_validate(bad)

    def test_update_requires_node_id_and_signal(self):
        bad = copy.deepcopy(VALID_TURN)
        bad["node_updates"][0] = {"op": "update", "strength": 3.0}
        with self.assertRaises(ValidationError):
            TurnOutput.model_validate(bad)
        bad = copy.deepcopy(VALID_TURN)
        bad["node_updates"][0] = {"op": "update", "node_id": "B1"}
        with self.assertRaises(ValidationError):
            TurnOutput.model_validate(bad)

    def test_strength_out_of_range_rejected(self):
        bad = copy.deepcopy(VALID_TURN)
        bad["node_updates"][0]["strength"] = 4.5
        with self.assertRaises(ValidationError):
            TurnOutput.model_validate(bad)

    def test_top_level_extra_rejected(self):
        bad = copy.deepcopy(VALID_TURN)
        bad["reasoning"] = "no free-form fields"
        with self.assertRaises(ValidationError):
            TurnOutput.model_validate(bad)

    def test_emotion_category_closed_set(self):
        good = copy.deepcopy(VALID_TURN)
        good["emotion"]["category"] = "warmth"
        TurnOutput.model_validate(good)
        for bad_label in ("hopeful", "mild discomfort", "reflectiveness", "annoyance"):
            bad = copy.deepcopy(VALID_TURN)
            bad["emotion"]["category"] = bad_label
            with self.assertRaises(ValidationError, msg=bad_label):
                TurnOutput.model_validate(bad)
        # the LLM-facing tool schema also constrains it
        from engine.schema import SIMULATE_USER_TURN_SCHEMA
        self.assertEqual(SIMULATE_USER_TURN_SCHEMA["properties"]["emotion"]
                         ["properties"]["category"]["enum"][0], "neutral")

    def test_done_flag(self):
        ok = TurnOutput.model_validate(VALID_TURN)
        self.assertFalse(ok.done)
        self.assertIsNone(ok.done_reason)
        # the LLM-facing tool schema requires "done" (pydantic defaults False)
        from engine.schema import SIMULATE_USER_TURN_SCHEMA
        self.assertIn("done", SIMULATE_USER_TURN_SCHEMA["required"])
        self.assertIn("done", SIMULATE_USER_TURN_SCHEMA["properties"])
        done = copy.deepcopy(VALID_TURN)
        done["done"] = True
        done["done_reason"] = "the user agreed to donate $10"
        t = TurnOutput.model_validate(done)
        self.assertTrue(t.done)
        self.assertEqual(t.done_reason, "the user agreed to donate $10")


class TestInitOutput(unittest.TestCase):
    def test_valid_init(self):
        i = InitOutput.model_validate({
            "nodes": [{"id": "B1", "type": "belief", "content": "job is stressful",
                       "strength": 3.0}],
            "edges": [],
        })
        self.assertEqual(len(i.nodes), 1)

    def test_id_prefix_mismatch_rejected(self):
        with self.assertRaises(ValidationError):
            InitOutput.model_validate({
                "nodes": [{"id": "D1", "type": "belief", "content": "x",
                           "strength": 3.0}],
                "edges": [],
            })

    def test_illegal_relation_rejected(self):
        with self.assertRaises(ValidationError):
            InitOutput.model_validate({
                "nodes": [], "edges": [{"from": "B1", "to": "I1", "relation": "elicits"}],
            })


class TestNode(unittest.TestCase):
    def test_node_strength_bounds(self):
        Node(id="B1", type="belief", content="x", strength=2.5)
        with self.assertRaises(ValidationError):
            Node(id="B1", type="belief", content="x", strength=4.5)


class TestSeed(unittest.TestCase):
    def test_seed_with_private_persona(self):
        s = Seed.model_validate({
            "seed_id": "x_01", "task": "price_negotiation",
            "persona": "The user is the BUYER.",
            "private_persona": "target $69",
            "scenario": "seller lists item for $75", "u0": "hi interested",
        })
        self.assertIn("target $69", s.simulator_persona())
        self.assertNotIn("target $69", s.persona)


class TestStepRequest(unittest.TestCase):
    def test_manual_requires_reply(self):
        with self.assertRaises(ValidationError):
            StepRequest.model_validate({"seed_id": "x", "mode": "manual"})
        StepRequest.model_validate({"seed_id": "x", "mode": "auto"})
        StepRequest.model_validate({"seed_id": "x", "mode": "auto_editable", "system_prompt": "be kind"})


if __name__ == "__main__":
    unittest.main()
