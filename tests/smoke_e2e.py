"""Live end-to-end smoke test (needs the LLM proxy env vars).

Run:  python tests/smoke_e2e.py --live --turns 3
For each of the 6 seeds: Init + N turns (first turns replay the real agent
replies via mode=manual, last turn uses mode=auto). Asserts:
  - schema_ok on every turn (implicit: step() raises otherwise)
  - no schema-class rejections (only duplicate/no-op tolerated)
  - node ids unique, session log reloadable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.schema import Seed
from engine.simulator import UserSimulator

USER_ROLES = {"emotional_support": "seeker", "persuasion_donation": "persuadee",
              "price_negotiation": "buyer"}
SCHEMA_CLASS_REASONS = ("schema", "pattern", "missing", "unknown node_id", "illegal relation")


def agent_replies_after_prefix(seed: Seed) -> list[str]:
    """Real agent utterances following the unintervened prefix."""
    replies: list[str] = []
    for u in seed.reference_transcript[len(seed.pre_context):]:
        if u.role != USER_ROLES[seed.task.value]:
            replies.append(u.text)
    return replies


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="actually call the LLM (default: structure check only)")
    ap.add_argument("--turns", type=int, default=3)
    args = ap.parse_args()

    seeds_dir = ROOT / "seeds"
    failures = []
    for seed_path in sorted(seeds_dir.glob("*.json")):
        seed = Seed.model_validate(json.loads(seed_path.read_text(encoding="utf-8")))
        sim = UserSimulator(seed, ROOT / "sessions" / f"{seed.seed_id}.smoke.json")
        print(f"\n=== {seed.seed_id} ({seed.task.value}) ===")
        if not args.live:
            print("  [structure-only] Init skipped (no --live)")
        else:
            sim.init()
            n, e = len(sim.graph.active_nodes()), len(sim.graph.edges)
            print(f"  Init: {n} nodes, {e} edges")
            ids = [x["id"] for x in sim.graph.snapshot()["nodes"]]
            assert len(ids) == len(set(ids)), f"duplicate ids: {ids}"

        replies = agent_replies_after_prefix(seed)
        for i in range(min(args.turns, len(replies))):
            if not args.live:
                sim.history.append({"role": "agent", "text": replies[i]})
                continue
            turn = sim.step(replies[i], mode="manual")
            print(f"  turn {i + 1}: sim_user={turn['user_utterance'][:60]!r} "
                  f"| deltas={turn['deltas']} | rejected={len(turn['ops_rejected'])}")
            for r in turn["ops_rejected"]:
                if any(k in r.get("reason", "") for k in SCHEMA_CLASS_REASONS):
                    failures.append((seed.seed_id, "schema-class rejection", r))

        if args.live and replies:  # one auto-mode turn
            from engine import agent
            prompt = agent.default_system_prompt(seed.task)
            reply = agent.generate_agent_reply(seed, sim.history, prompt)
            turn = sim.step(reply, mode="auto")
            print(f"  auto turn: agent={reply[:50]!r} sim_user={turn['user_utterance'][:50]!r}")

        # log reload check (structure-only mode never called step(), so persist once here)
        sim._save()
        log = json.loads(sim.session_file.read_text(encoding="utf-8"))
        assert log["seed_id"] == seed.seed_id
        assert len(log["turns"]) == len(sim.turns)
        print(f"  log ok: {len(log['turns'])} turns persisted")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" ", f)
        sys.exit(1)
    print("\nALL SEEDS OK")


if __name__ == "__main__":
    main()
