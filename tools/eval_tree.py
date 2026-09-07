#!/usr/bin/env python3
"""Sparse tree rollout: from ONE cached G0, branch the dialogue at every turn
with 2 contrasting agent strategies and observe whether the cognitive state
trajectories diverge (causality) and reach different outcomes (discrimination).

This is the 可操纵性 quality metric from insight.md §8, tested on a shared
prefix: nodes at the same depth share the SAME user state, so any trajectory
divergence is attributable to the agent reply alone.

Current tree: craigslist_01, depth 4, branching 2 (facts-concession lane F vs
pressure-scarcity lane P). 4 full paths, shared prefixes computed once — 15
simulator LLM calls total. Agent replies are hand-written per lane (replay
style): a sampling agent would be near-deterministic and produce no
divergence.

Usage:
  python tools/eval_tree.py
  python tools/eval_tree.py --seed craigslist_01 --depth 4
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.schema import Seed
from engine.simulator import UserSimulator

TMP = ROOT / "sessions" / "_tree_tmp.json"

# Lane replies: F = facts + small concessions, P = pressure + scarcity.
# Written against craigslist_01's real transcript (listed $75, buyer target ~$69).
LANES = {
    "F": [
        "Hi! Yes, the card is brand new and still sealed. The $75 price reflects that — what were you thinking?",
        "To be fair, the same 200GB SanDisk Ultra goes for $85+ new on Amazon right now, so $75 is already under market. It's sealed, with the warranty card.",
        "Since you can pick it up today, I can come down a little — how about $72?",
        "OK, meet me in the middle: $70 and it's yours. I can do the handoff this evening.",
    ],
    "P": [
        "Hey. Price is $75 firm. I've got three other buyers messaging me about it right now.",
        "Look, someone just offered $72. I can't hold it forever — if you want it you need to move fast, it'll be gone by tonight.",
        "Last chance: $72 and it's yours today. Otherwise I'm selling to the other guy.",
        "Final call. I'm leaving in ten minutes — $72 or the other buyer takes it. Your call.",
    ],
}

# Per-seed lane scripts for the newer negotiation seeds (F=facts+concessions,
# P=pressure+scarcity), written against each seed's item and listed price.
LANES_BY_SEED = {
    "craigslist_03": {
        "F": [
            "Hi! It's a display shelf with LED lights — holds about 20 books, works perfectly. What's your budget?",
            "The wood is solid and it's barely used — these go for $60+ new, so $49 is already below that.",
            "Since you can pick it up today, I can do $47.",
            "OK, $45 and it's yours — meet you this evening.",
        ],
        "P": [
            "Hey. $49 firm, it's a good display. Several people have messaged about it.",
            "Someone just offered $46. If you want it, decide now — first come first served.",
            "Last call: $47 takes it today, otherwise I'm going with the other buyer.",
            "Final offer. I'm packing it for the other guy tonight — $47 or it's gone.",
        ],
    },
    "craigslist_04": {
        "F": [
            "Hi! Yes — Gravity brand board, barely ridden, the wheels are new. What's your offer?",
            "The wheels are premium quality and the deck has no cracks — comparable boards sell for $180-220, so it's basically new.",
            "I can come down a bit since you asked — $150?",
            "Meet me at $135 and it's yours.",
        ],
        "P": [
            "Hey. $200 firm, the board is basically brand new. Got three other people asking about it.",
            "Someone just offered $150. If you want it, move now — it'll be gone today.",
            "Last chance: $150 today, otherwise the other buyer takes it.",
            "Final call — I'm meeting the other buyer in an hour. $150 or it's gone.",
        ],
    },
    "craigslist_05": {
        "F": [
            "Hello! It's in very good condition — no accidents, and I have all the service records. What were you thinking?",
            "The maintenance is fully documented — new tires last month, oil changes every 5k. KBB puts it around $14k private sale.",
            "Since you're serious, I could do $13,000.",
            "OK — $12,500 and it's yours, records included.",
        ],
        "P": [
            "Hi. $14,800 firm — I have another buyer coming to see it Saturday.",
            "He's offered $13,500. If you want it, I need your answer before Saturday — it won't last.",
            "Last chance: $13,500 or the Saturday buyer takes it.",
            "Final call — he's coming at noon. $13,500 or it's gone.",
        ],
    },
}


def _seed(sid: str) -> Seed:
    p = ROOT / "seeds" / f"{sid}.json"
    return Seed.model_validate(json.loads(p.read_text(encoding="utf-8")))


def run_tree(seed_id: str, depth: int) -> dict:
    """Roll out the shared-prefix tree. Branches that reach done=true stop
    expanding (stepping a concluded session would produce contradictory
    turns); their final record is carried forward to the remaining depths so
    aggregation sees the real outcome."""
    lanes = LANES_BY_SEED.get(seed_id, LANES)
    if depth > len(next(iter(lanes.values()))):
        raise SystemExit(f"--depth {depth} exceeds the lane script length "
                         f"{len(next(iter(lanes.values())))} — add replies to LANES first")
    seed = _seed(seed_id)
    sim = UserSimulator(seed, TMP)
    sim.init()                      # cached, deterministic G0
    sim.session_file = None         # in-memory branching; no log writes

    # frontier: list of (sim_state, path_label); done branches freeze
    frontier = [(sim, "")]
    # turn-level states: depth -> label -> record (shared prefixes appear once)
    tree: dict[int, dict[str, dict]] = {}
    for turn in range(1, depth + 1):
        nxt = []
        for sim_state, label in frontier:
            for lane, lane_replies in lanes.items():
                child = copy.deepcopy(sim_state)
                reply = lane_replies[turn - 1]
                path = label + lane
                try:
                    child.step(reply, mode="manual")
                except Exception as e:  # LLM schema 瞬时失败：标记后继续其他分支
                    tree.setdefault(turn, {})[path] = {
                        "user": f"[LLM FAILED: {type(e).__name__}]",
                        "emotion": "?", "goal_congruence": 0.0, "goal_conflict": 0.0,
                        "controllability": 0.0, "deltas": {}, "notes": [],
                        "done": True, "done_reason": f"llm_failure:{type(e).__name__}",
                    }
                    continue
                rec = _summarize(child)
                tree.setdefault(turn, {})[path] = rec
                if not rec["done"]:
                    nxt.append((child, path))
        frontier = nxt
        if not frontier:
            break
    # carry final records of early-done branches down to the requested depth
    for d in range(2, depth + 1):
        for p_prev, rec in tree.get(d - 1, {}).items():
            if rec["done"] and p_prev not in tree.get(d, {}):
                tree.setdefault(d, {})[p_prev] = rec
    return tree


def classify_outcome(rec: dict) -> str:
    if not rec.get("done"):
        return "undetermined"
    reason = (rec.get("done_reason") or "").lower()
    if reason.startswith("llm_failure"):
        return "failed"
    if any(k in reason for k in ("walk away", "walking away", "breakdown", "refuse")):
        return "walk_away"
    return "deal"


def print_tree(tree: dict, depth: int, seed_id: str) -> None:
    print(f"seed={seed_id}  depth={depth}  branching={len(LANES_BY_SEED.get(seed_id, LANES))}")
    for d in range(1, depth + 1):
        print(f"\n===== turn {d} =====")
        for path, rec in tree.get(d, {}).items():
            lane = path[-1] if path else "?"
            print(f"  [{path:>4}] ({'F' if lane == 'F' else 'P'}回) user: {rec['user'][:95]}")
            print(f"         emotion={rec['emotion']:<11} goal_c={rec['goal_congruence']:+.2f} "
                  f"goal_conflict={rec['goal_conflict']:.2f} controll={rec['controllability']:.2f} "
                  f"deltas={rec['deltas'] or '-'}")
            for n in rec["notes"]:
                print(f"         ⚠ {n[:90]}")
            if rec["done"]:
                print(f"         *** DONE: {rec['done_reason']} ***")
    print("\n===== 结局 =====")
    for path, rec in tree.get(depth, {}).items():
        print(f"  {path}: {classify_outcome(rec)} | {rec['done_reason'] or '-'}")


def print_aggregate(trees: list[dict], depth: int) -> None:
    """Across N runs: outcome counts per path, and by pressure exposure."""
    from collections import Counter
    paths = sorted(trees[0].get(depth, {}).keys())
    print(f"\n===== 聚合 {len(trees)} 棵树：各路径结局 =====")
    print(f"{'path':>6} {'deal':>5} {'walk':>5} {'undet':>5}")
    per_path = {}
    for p in paths:
        c = Counter(classify_outcome(t.get(depth, {}).get(p, {})) for t in trees)
        per_path[p] = c
        print(f"{p:>6} {c['deal']:>5} {c['walk_away']:>5} {c['undetermined']:>5}")
    print(f"\n===== 按施压暴露量（路径中 P 回合数）聚合 =====")
    print(f"{'P回合数':>7} {'样本':>5} {'deal率':>7} {'walk率':>7}")
    for k in range(depth + 1):
        c = Counter()
        for p in paths:
            if p.count("P") == k:
                c.update(per_path[p])
        n = sum(c.values())
        if n:
            print(f"{k:>7} {n:>5} {c['deal'] / n:>7.1%} {c['walk_away'] / n:>7.1%}")


def _summarize(sim: UserSimulator) -> dict:
    t = sim.turns[-1]
    notes = [n for n in t.get("notes") or [] if isinstance(n, str) and n.startswith("⚠")]
    return {
        "user": t["user_utterance"],
        "emotion": t["emotion"]["category"],
        "goal_congruence": round(t["appraisal"].get("goal_congruence", 0), 2),
        "goal_conflict": round(t["appraisal"].get("goal_conflict", 0), 2),
        "controllability": round(t["appraisal"].get("controllability", 0), 2),
        "deltas": t.get("deltas") or {},
        "notes": notes,
        "done": bool(t.get("done")),
        "done_reason": t.get("done_reason"),
        "nodes": t["graph_after"]["nodes"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", default="craigslist_01")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--runs", type=int, default=1,
                    help="number of full trees to roll out (each = ~15 LLM calls)")
    args = ap.parse_args()
    trees = [run_tree(args.seed, args.depth) for _ in range(args.runs)]
    if args.runs == 1:
        print_tree(trees[0], args.depth, args.seed)
    else:
        print(f"seed={args.seed} depth={args.depth} runs={args.runs} "
              f"(leaf outcomes per run, aggregated below)")
        print_aggregate(trees, args.depth)
    TMP.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
