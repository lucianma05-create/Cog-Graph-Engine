#!/usr/bin/env python3
"""Style-conformance probe suite (LLM-driven): does the simulator behave per
the seed's COGNITIVE STYLE block? Three probes, one per style dimension:

  A. reactance    — p4g_02 (freedom 6.0): guilt-pressure must NOT raise
                    commitment; goal_conflict should rise; the user pushes back.
  B. facts-first  — p4g_01: flattery without facts must not move beliefs or
                    create intentions; a factual appeal should produce a
                    positive belief/desire move (content-revision exemption
                    lets small facts-driven moves through).
  C. commitment   — craigslist_01: an ultimatum must not be accepted without
                    justification; the price anchor holds (no price-audit flag
                    when the counter-offer asks for a concession); the target
                    intention survives.

Usage:
  python tools/eval_probes.py            # all probes (each ~2 LLM calls)
  python tools/eval_probes.py --probe A
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

TMP = ROOT / "sessions" / "_probe_tmp.json"


def _seed(sid: str) -> Seed:
    p = ROOT / "seeds" / f"{sid}.json"
    return Seed.model_validate(json.loads(p.read_text(encoding="utf-8")))


def _fresh(sid: str) -> UserSimulator:
    sim = UserSimulator(_seed(sid), TMP)
    sim.init()
    return sim


def _check(name: str, cond: bool, detail: str) -> bool:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    return cond


def probe_a() -> bool:
    print("== 探针 A：p4g_02 高压施压（reactance pronounced）==")
    sim = _fresh("p4g_02")
    t = sim.step("You MUST donate. Don't you care about these starving children? "
                 "Everyone else has given at least $50 already.", mode="manual")
    print(f'  user: {t["user_utterance"]}')
    ok = True
    ok &= _check("goal_conflict 上升", t["appraisal"].get("goal_conflict", 0) >= 0.5,
                 f"goal_conflict={t['appraisal'].get('goal_conflict')}")
    intent_up = any(d > 0 for nid, d in t["deltas"].items()
                    if any(n["id"] == nid and n["type"] == "intention"
                           for n in t["graph_after"]["nodes"]))
    intent_added = any(o.get("op") == "add" and o.get("node", {}).get("type") == "intention"
                       for o in t["ops_applied"])
    ok &= _check("压力下意向不上升", not intent_up and not intent_added,
                 f"intent_up={intent_up} intent_added={intent_added}")
    ok &= _check("无 reactance 误报", not any("reactance" in (n or "") for n in t["notes"]),
                 f"notes={t['notes']}")
    return ok


def probe_b() -> bool:
    print("== 探针 B：p4g_01 先奉承后事实（facts-first 风格）==")
    sim = _fresh("p4g_01")
    t1 = sim.step("You are obviously a wonderful, big-hearted person — people like you "
                  "make the world better. Can we count on your generous support today?",
                  mode="manual")
    print(f'  [奉承轮] user: {t1["user_utterance"]}')
    ok = True
    belief_up = any(d > 0 for nid, d in t1["deltas"].items()
                    if any(n["id"] == nid and n["type"] == "belief"
                           for n in t1["graph_after"]["nodes"]))
    intent_added = any(o.get("op") == "add" and o.get("node", {}).get("type") == "intention"
                       for o in t1["ops_applied"])
    ok &= _check("奉承不动信念", not belief_up, f"deltas={t1['deltas']}")
    ok &= _check("奉承不产生意向", not intent_added, f"applied={len(t1['ops_applied'])}")
    t2 = sim.step("To be concrete: 92% of donations go directly to programs, and our "
                  "overhead was independently audited at 3% last year. A $10 monthly "
                  "gift funds one child's school supplies.", mode="manual")
    print(f'  [事实轮] user: {t2["user_utterance"]}')
    pos_bd = any(d > 0 for nid, d in t2["deltas"].items()
                 if any(n["id"] == nid and n["type"] in ("belief", "desire")
                        for n in t2["graph_after"]["nodes"]))
    belief_added = any(o.get("op") == "add" and o.get("node", {}).get("type") == "belief"
                       for o in t2["ops_applied"])
    ok &= _check("事实驱动正向移动", pos_bd or belief_added,
                 f"deltas={t2['deltas']} belief_added={belief_added}")
    return ok


def probe_c() -> bool:
    print("== 探针 C：craigslist_01 最后通牒（commitment persistent）==")
    sim = _fresh("craigslist_01")
    t1 = sim.step("Hi! Thanks for your interest. The card is $75 as listed. "
                  "What's your offer?", mode="manual")
    print(f'  [第1轮] user: {t1["user_utterance"]}')
    ok = True
    t2 = sim.step("Sorry, I can't go that low. Final offer: $70. Take it or leave it — "
                  "I have three other buyers waiting.", mode="manual")
    print(f'  [通牒轮] user: {t2["user_utterance"]}')
    # 价格纪律链：出价上涨必须要么有图内依据（worth/urgency 正向移动），
    # 要么被审计拦截（⚠ 进入下一轮回馈）。两者缺一即纪律链断裂。
    price_up = "$70" in t2["user_utterance"]
    worth_urgency_up = any(d > 0 for nid, d in t2["deltas"].items()
                           if any(n["id"] == nid and n["type"] in ("belief", "desire")
                                  for n in t2["graph_after"]["nodes"]))
    audit_fired = any("PRICE EXPECTATION REVISION" in (n or "") for n in t2["notes"])
    ok &= _check("价格纪律链成立（图内依据或审计拦截）",
                 (not price_up) or worth_urgency_up or audit_fired,
                 f"price_up={price_up} worth_urgency_up={worth_urgency_up} "
                 f"audit_fired={audit_fired}")
    intents_alive = [n for n in t2["graph_after"]["nodes"] if n["type"] == "intention"]
    ok &= _check("目标意向存续（未屈服弃锚）", bool(intents_alive),
                 f"intentions={[n['id'] + ': ' + n['content'][:40] for n in intents_alive]}")
    return ok


PROBES = {"A": probe_a, "B": probe_b, "C": probe_c}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", choices=["A", "B", "C"], default=None)
    args = ap.parse_args()
    keys = [args.probe] if args.probe else ["A", "B", "C"]
    results = {}
    for k in keys:
        results[k] = PROBES[k]()
        print()
    print("==== 结果 ====")
    for k, ok in results.items():
        print(f"  探针 {k}: {'PASS' if ok else 'FAIL'}")
    print(f"  {sum(results.values())}/{len(results)} 通过")
    TMP.unlink(missing_ok=True)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
