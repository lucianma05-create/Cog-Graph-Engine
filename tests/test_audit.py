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
