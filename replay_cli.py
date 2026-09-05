#!/usr/bin/env python3
"""Headless replay: feed the seed's REAL agent replies to the simulator turn by
turn, and compare the simulated user replies with the real user replies.

This is the quick "behavior realism" check: does the BDI-E simulator, driven by
exactly the same agent utterances as the real conversation, produce a user
trajectory that looks like the real one?

Usage:
  python replay_cli.py --seed esconv_01 --turns 6
  python replay_cli.py --seed p4g_01 --turns 6 --compare-g0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from engine.schema import Seed
from engine.simulator import UserSimulator

USER_ROLES = {"emotional_support": "seeker", "persuasion_donation": "persuadee",
              "price_negotiation": "buyer"}


def _wrap(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--no-llm-graph", action="store_true",
                    help="skip Init+step LLM calls (structure check only)")
    ap.add_argument("--compare-g0", action="store_true",
                    help="also print the CogWM hindsight global_bdi for manual comparison")
    args = ap.parse_args()

    seed_path = ROOT / "seeds" / f"{args.seed}.json"
    if not seed_path.exists():
        sys.exit(f"seed not found: {seed_path}")
    seed = Seed.model_validate_json(seed_path.read_text(encoding="utf-8"))
    user_role = USER_ROLES[seed.task.value]

    # Align with the live session: the simulation continues right AFTER the
    # unintervened prefix (pre_context), so collect the real conversation's
    # agent/user utterances that follow the prefix.
    offset = len(seed.pre_context)
    agent_replies: list[str] = []
    real_user_replies: list[str] = []
    for u in seed.reference_transcript[offset:]:
        is_user = u.role == user_role
        (real_user_replies if is_user else agent_replies).append(u.text)

    sim = UserSimulator(seed, ROOT / "sessions" / f"{args.seed}.replay.json")
    print(f"[init] {args.seed} — Init(P, S, H0) ...")
    if not args.no_llm_graph:
        sim.init()
    print(f"[init] G0: {len(sim.graph.active_nodes())} nodes, {len(sim.graph.edges)} edges")
    for n in sim.graph.active_nodes():
        print(f"        {n.id} [{n.type.value} s={n.strength}] {_wrap(n.content, 80)}")

    if args.compare_g0:
        print("\n[compare-g0] CogWM hindsight global_bdi (manual comparison only):")
        import json
        gb = None
        for line in open("/data/user21300120/mmh/CogWM/bdi-annotation/output/" +
                         ("esconv.jsonl" if seed.task.value == "emotional_support"
                          else "persuasionforgood.jsonl"), encoding="utf-8"):
            rec = json.loads(line)
            if rec["id"] == seed.notes.get("source_record_id"):
                gb = rec.get("global_bdi")
                break
        if gb:
            for kind in ("beliefs", "desires", "intentions"):
                for x in gb.get(kind, []):
                    score = x.get(next((k for k in x if k.endswith("_score")), "?"))
                    print(f"        {x['id']} [{kind} score={score}] {_wrap(x['content'], 90)}")
        else:
            print("        (not found)")

    n = min(args.turns, len(agent_replies))
    print(f"\n[replay] {n} turns with real agent replies:")
    for i in range(n):
        agent_reply = agent_replies[i]
        real_user = real_user_replies[i] if i < len(real_user_replies) else "(end)"
        if args.no_llm_graph:
            sim.history.append({"role": "agent", "text": agent_reply})
            sim.history.append({"role": "user", "text": real_user})
            print(f"\n--- turn {i + 1} ---\nagent: {_wrap(agent_reply, 90)}")
            print(f"real user: {_wrap(real_user, 90)}")
            continue
        turn = sim.step(agent_reply, mode="manual")
        print(f"\n--- turn {i + 1} ---")
        print(f"agent:        {_wrap(agent_reply, 90)}")
        print(f"real user:    {_wrap(real_user, 90)}")
        print(f"sim user:     {_wrap(turn['user_utterance'], 90)}")
        if turn.get("done"):
            print(f"*** 模拟器判定完成：{turn.get('done_reason') or '（无说明）'} ***")
        print(f"appraisal:    {turn['appraisal']}")
        print(f"emotion:      {turn['emotion']['category']} v={turn['emotion']['valence']} "
              f"a={turn['emotion']['arousal']} i={turn['emotion']['intensity']}")
        print(f"graph:        {len(turn['graph_after']['nodes'])} active nodes, "
              f"{len(turn['graph_after']['edges'])} edges | deltas={turn['deltas']}")
        for o in turn["ops_applied"]:
            desc = o.get("node_id") or (o.get("edge") and f"{o['edge']['from']}->{o['edge']['to']}")
            print(f"  op {o['op']:<10} {desc}" + (" [auto]" if o.get("auto") else ""))
        if turn["ops_rejected"]:
            print(f"  rejected: {turn['ops_rejected']}")
        if turn.get("done"):
            print(f"\n[replay] simulator signaled completion at turn {i + 1}; stopping.")
            break


if __name__ == "__main__":
    main()
