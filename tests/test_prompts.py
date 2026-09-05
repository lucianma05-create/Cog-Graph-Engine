import json
import unittest
from pathlib import Path

from engine.prompts import (AGENT_DEFAULT, TASK_RULES, build_init_system,
                            build_turn_system, render_agent_user, render_init_user,
                            render_turn_user)
from engine.schema import Seed, Task
from engine.updater import CognitiveGraph

ROOT = Path(__file__).parent.parent

TASK_KEYWORDS = {
    Task.emotional_support: ("support-seeker", "distress"),
    Task.persuasion_donation: ("persuadee", "DONATION INTENTION"),
    Task.price_negotiation: ("BUYER", "target price", "PRICE EXPECTATION REVISION"),
}


class TestPromptAssembly(unittest.TestCase):
    def test_three_tasks_have_three_rule_blocks(self):
        self.assertEqual(set(TASK_RULES), {t for t in Task})
        self.assertEqual(set(AGENT_DEFAULT), {t for t in Task})
        blocks = [TASK_RULES[t] for t in Task]
        self.assertEqual(len(set(blocks)), 3)  # distinct content

    def test_task_rules_injected_into_systems(self):
        for task in Task:
            for builder in (build_init_system, build_turn_system):
                system = builder(task)
                flat = " ".join(system.split())  # ignore line wrapping
                for kw in TASK_KEYWORDS[task]:
                    self.assertIn(kw, flat, f"{builder.__name__}({task}) missing {kw!r}")
                # no cross-contamination: each system contains only its own rules
                for other, kws in TASK_KEYWORDS.items():
                    if other is task:
                        continue
                    for kw in kws:
                        self.assertNotIn(kw, flat,
                                         f"{builder.__name__}({task}) leaked {other} rule {kw!r}")


class TestAgentContext(unittest.TestCase):
    def _seed(self, name: str) -> Seed:
        path = ROOT / "seeds" / f"{name}.json"
        if not path.exists():
            self.skipTest("seeds not extracted")
        return Seed.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def test_negotiation_agent_context_has_private_position(self):
        seed = self._seed("craigslist_01")
        ctx = render_agent_user(seed, [])
        self.assertIn("YOUR PRIVATE NEGOTIATION POSITION", ctx)
        self.assertIn(seed.agent_private, ctx)

    def test_agent_private_never_leaks_into_simulator_prompts(self):
        seed = self._seed("craigslist_01")
        turn = render_turn_user(seed, CognitiveGraph(), {"goal_congruence": 0, "controllability": 0.5,
                                                         "goal_conflict": 0},
                                {"category": "neutral", "valence": 0, "arousal": 0.2,
                                 "appraisal_target": "#"}, [], "hi")
        self.assertNotIn(seed.agent_private, turn)
        self.assertNotIn("YOUR PRIVATE NEGOTIATION POSITION", build_turn_system(seed.task))
        self.assertNotIn("YOUR PRIVATE NEGOTIATION POSITION", build_init_system(seed.task))

    def test_simulator_private_persona_never_in_agent_context(self):
        seed = self._seed("craigslist_01")
        ctx = render_agent_user(seed, [])
        self.assertNotIn(seed.private_persona, ctx)

    def test_engine_feedback_rendered_and_absent_by_default(self):
        seed = self._seed("craigslist_01")
        kwargs = dict(appraisal={"goal_congruence": 0, "controllability": 0.5,
                                 "goal_conflict": 0},
                      emotion={"category": "neutral", "valence": 0, "arousal": 0.2,
                               "appraisal_target": "#"},
                      history=[], agent_reply="hi")
        plain = render_turn_user(seed, CognitiveGraph(), **kwargs)
        self.assertNotIn("ENGINE FEEDBACK", plain)
        with_fb = render_turn_user(
            seed, CognitiveGraph(), **kwargs,
            feedback=["add edge I1 -facilitates-> B2 REJECTED: illegal relation facilitates for endpoint types"])
        self.assertIn("ENGINE FEEDBACK", with_fb)
        self.assertIn("illegal relation", with_fb)

    def test_first_person_stance_in_systems(self):
        seed = self._seed("craigslist_01")
        init = build_init_system(seed.task)
        turn = build_turn_system(seed.task)
        for s in (init, turn):
            self.assertIn("first-person", s)
            self.assertIn("AS THE USER HOLDS IT", s)
        self.assertIn("CAUSAL DISCIPLINE", turn)

    def test_cognitive_style_in_simulator_prompts_only(self):
        seed = self._seed("p4g_01")
        self.assertTrue(seed.cognitive_style)
        marker = seed.cognitive_style[:40]
        turn = render_turn_user(seed, CognitiveGraph(),
                                {"goal_congruence": 0, "controllability": 0.5,
                                 "goal_conflict": 0},
                                {"category": "neutral", "valence": 0, "arousal": 0.2,
                                 "appraisal_target": "#"}, [], "hi")
        self.assertIn("COGNITIVE STYLE", turn)
        self.assertIn(marker, turn)
        self.assertIn("COGNITIVE STYLE", render_init_user(seed))
        # the style is the user's private cognition: never in the agent context
        ctx = render_agent_user(seed, [])
        self.assertNotIn("COGNITIVE STYLE", ctx)
        self.assertNotIn(marker, ctx)


if __name__ == "__main__":
    unittest.main()
