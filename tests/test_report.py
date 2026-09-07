import json
import unittest
from pathlib import Path

from engine.report import render_report
from engine.schema import Seed

ROOT = Path(__file__).parent.parent


def _fake_log() -> dict:
    return {
        "schema_version": 3,
        "seed_id": "t_01",
        "task": "emotional_support",
        "created_at": "2026-09-04T00:00:00Z",
        "initial": {
            "graph": {"nodes": [{"id": "B1", "type": "belief", "content": "job is stressful", "strength": 3.0}],
                      "edges": []},
            "deactivated": [],
            "appraisal": {"goal_congruence": 0, "controllability": 0.5, "goal_conflict": 0},
            "emotion": {"category": "neutral", "valence": 0, "arousal": 0.2,
                        "appraisal_target": "#session_start"},
        },
        "turns": [{
            "turn_index": 1, "mode": "manual", "agent_reply": "What troubles you?",
            "user_utterance": "My job.", "appraisal": {"goal_congruence": 0, "controllability": 0.5,
                                                       "goal_conflict": 0},
            "emotion": {"category": "sadness", "valence": -0.3, "arousal": 0.4,
                        "appraisal_target": "B1"},
            "done": True, "done_reason": "the user felt relieved and said goodbye",
            "graph_after": {"nodes": [{"id": "B1", "type": "belief", "content": "job is stressful",
                                       "strength": 3.2}], "edges": []},
            "deactivated_after": [], "deltas": {"B1": 0.2},
            "ops_applied": [{"op": "update", "node_id": "B1", "auto": False}],
            "ops_rejected": [], "notes": [], "retries": 0, "retry_raw": None,
            "validation": {"schema_ok": True, "retries": 0},
        }],
    }


class TestReport(unittest.TestCase):
    def test_report_contains_trajectory(self):
        seed_path = ROOT / "seeds" / "esconv_01.json"
        if not seed_path.exists():
            self.skipTest("seeds not extracted")
        seed = Seed.model_validate(json.loads(seed_path.read_text(encoding="utf-8")))
        md = render_report(seed, _fake_log())
        self.assertIn("交互轨迹报告", md)
        self.assertIn("G0 基础图", md)
        self.assertIn("第 1 轮", md)
        self.assertIn("完成 ✓", md)
        self.assertIn("the user felt relieved and said goodbye", md)
        self.assertIn("agent 回复：What troubles you?", md)
        self.assertIn("模拟用户回复：My job.", md)


if __name__ == "__main__":
    unittest.main()
