import unittest

from engine.simulator import price_audit_note


class TestPriceAudit(unittest.TestCase):
    LP = 75.0  # listed price, like craigslist_01

    def test_non_negotiation_tasks_ignored(self):
        self.assertIsNone(price_audit_note("emotional_support", "I'd pay $70", None, False, self.LP))

    def test_no_price_in_utterance(self):
        self.assertIsNone(price_audit_note("price_negotiation", "That sounds fair.", 69.0, False, self.LP))

    def test_first_offer_no_warning(self):
        self.assertIsNone(price_audit_note("price_negotiation", "I can do $69", None, False, self.LP))

    def test_offer_up_with_worth_belief_rise_ok(self):
        note = price_audit_note("price_negotiation", "OK, $70 works", 69.0, True, self.LP)
        self.assertIsNone(note)

    def test_offer_up_with_only_intention_rise_flagged(self):
        # acceptance intention moving up is circular, not justification
        note = price_audit_note("price_negotiation", "I can meet you at $70", 69.0, False, self.LP)
        self.assertIsNotNone(note)
        self.assertIn("PRICE EXPECTATION REVISION", note)

    def test_offer_down_or_equal_ok(self):
        self.assertIsNone(price_audit_note("price_negotiation", "How about $65?", 69.0, False, self.LP))
        self.assertIsNone(price_audit_note("price_negotiation", "Let's do $69 then", 69.0, False, self.LP))

    def test_quoting_sellers_higher_anchor_is_not_an_offer(self):
        # "I'm not looking to pay $85" quotes the seller's anchor, not an offer
        note = price_audit_note("price_negotiation",
                                "My offer is $69 — I'm not looking to pay $85.", 69.0, False, self.LP)
        self.assertIsNone(note)


if __name__ == "__main__":
    unittest.main()


from engine.schema import Edge
from engine.simulator import propagation_audit_note, reactance_audit_note


class TestReactanceAudit(unittest.TestCase):
    def test_no_profile_no_flag(self):
        self.assertIsNone(reactance_audit_note(None, "Don't you care?", 0.8, {}, {}, []))

    def test_normal_profile_no_flag(self):
        self.assertIsNone(reactance_audit_note({"reactance": "normal"},
                                               "Don't you care?", 0.8, {}, {}, []))

    def test_pronounced_pressure_rising_intention_flags(self):
        note = reactance_audit_note({"reactance": "pronounced"},
                                    "Don't you care about the children?",
                                    0.8, {"I1": 0.6}, {"I1": "intention"}, [])
        self.assertIsNotNone(note)
        self.assertIn("reactance", note)

    def test_no_pressure_language_no_flag(self):
        self.assertIsNone(reactance_audit_note({"reactance": "pronounced"},
                                               "Here are the audited facts.",
                                               0.8, {"I1": 0.6}, {"I1": "intention"}, []))

    def test_low_goal_conflict_no_flag(self):
        self.assertIsNone(reactance_audit_note({"reactance": "pronounced"},
                                               "Don't you care?", 0.4,
                                               {"I1": 0.6}, {"I1": "intention"}, []))

    def test_belief_rise_not_intention_no_flag(self):
        self.assertIsNone(reactance_audit_note({"reactance": "pronounced"},
                                               "Don't you care?", 0.8,
                                               {"B1": 0.6}, {"B1": "belief"}, []))

    def test_new_intention_under_pressure_flags(self):
        note = reactance_audit_note({"reactance": "pronounced"},
                                    "Take it or leave it.", 0.8,
                                    {}, {}, ["I2"])
        self.assertIsNotNone(note)
        self.assertIn("I2", note)


class TestPropagationAudit(unittest.TestCase):
    def test_facilitates_opposite_moves_flagged(self):
        notes = propagation_audit_note(
            [Edge(frm="B1", to="D1", relation="facilitates")],
            {"B1": 0.6, "D1": -0.5})
        self.assertEqual(len(notes), 1)
        self.assertIn("B1-facilitates->D1", notes[0])

    def test_facilitates_same_direction_ok(self):
        self.assertEqual(propagation_audit_note(
            [Edge(frm="B1", to="D1", relation="facilitates")],
            {"B1": 0.6, "D1": 0.5}), [])

    def test_inhibits_same_direction_flagged(self):
        notes = propagation_audit_note(
            [Edge(frm="B1", to="D1", relation="inhibits")],
            {"B1": 0.6, "D1": 0.5})
        self.assertEqual(len(notes), 1)
        self.assertIn("B1-inhibits->D1", notes[0])

    def test_small_moves_ignored(self):
        self.assertEqual(propagation_audit_note(
            [Edge(frm="B1", to="D1", relation="facilitates")],
            {"B1": 0.3, "D1": -0.5}), [])

    def test_means_for_and_conflicts_ignored(self):
        self.assertEqual(propagation_audit_note(
            [Edge(frm="I1", to="D1", relation="means_for"),
             Edge(frm="D1", to="D2", relation="conflicts_with")],
            {"I1": 0.6, "D1": -0.5, "D2": 0.6}), [])


class TestPriceAuditConcessions(unittest.TestCase):
    LP = 75.0

    def test_small_raise_with_concession_exempt(self):
        # probe C false positive: "$70 + throw in the adapter" is not a real rise
        self.assertIsNone(price_audit_note("price_negotiation",
                                           "Can we meet at $70 and you throw in the SD card adapter?",
                                           69.0, False, self.LP))

    def test_small_raise_with_if_you_include_exempt(self):
        self.assertIsNone(price_audit_note("price_negotiation",
                                           "I'll do $70 if you include a case for it",
                                           69.0, False, self.LP))

    def test_large_raise_with_concession_still_flagged(self):
        note = price_audit_note("price_negotiation",
                                "OK, $76 — and throw in the adapter too",
                                69.0, False, self.LP)
        self.assertIsNotNone(note)
        self.assertIn("PRICE EXPECTATION REVISION", note)

    def test_bare_raise_without_concession_flagged(self):
        note = price_audit_note("price_negotiation", "I can meet you at $70",
                                69.0, False, self.LP)
        self.assertIsNotNone(note)





class TestReviewRegressions(unittest.TestCase):
    """Regression tests from the 2026-09-05 code review."""

    def test_reactance_subthreshold_delta_no_flag(self):
        # 内容修改豁免会放行 +0.15 的意图移动；审计不得把它当 reactance 违规
        note = reactance_audit_note({"reactance": "pronounced"},
                                    "Don't you care about the children?", 0.8,
                                    {"I1": 0.15}, {"I1": "intention"}, [])
        self.assertIsNone(note)

    def test_propagation_uses_post_apply_edges(self):
        # 同轮切断矛盾边的重构不再被标记（post-apply 边集里该边已不存在）
        notes = propagation_audit_note([], {"B1": 0.6, "D1": -0.5})
        self.assertEqual(notes, [])

    def test_engine_feedback_labels_remove_edge(self):
        from engine.schema import Seed
        from engine.simulator import UserSimulator
        seed = Seed.model_validate({
            "seed_id": "t", "task": "price_negotiation",
            "persona": "p", "scenario": "s", "u0": "hi"})
        sim = UserSimulator(seed, None)
        sim.turns = [{"ops_rejected": [
            {"op": {"op": "remove",
                    "edge": {"from": "I1", "to": "D1", "relation": "means_for"}},
             "reason": "edge not found"}], "notes": []}]
        fb = sim._engine_feedback()
        self.assertTrue(fb and any("remove edge I1 -means_for-> D1" in x for x in fb))


if __name__ == "__main__":
    unittest.main()
