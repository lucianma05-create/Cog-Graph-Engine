#!/usr/bin/env python3
"""全量 MC 验证：6 谈判种子 × 4 策略 × N 局完整走到终局，逐轮记录
认知转移特征，分析"好轨迹（卖家份额高）vs 坏轨迹"的转移画像差异。

用法：python tools/eval_mc_analysis.py --n 3 --max-turns 10 --workers 48
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine import llm
from engine.prompts import AGENT_DEFAULT, render_agent_user
from engine.schema import Seed
from engine.simulator import UserSimulator
from tools.eval_ab import STRATEGIES, outcome_price
from tools.eval_hcsrl import features, snap_state

TRACES = ROOT / "sessions" / "ab_mc_traces.jsonl"


def _seed(sid: str) -> Seed:
    p = ROOT / "seeds" / f"{sid}.json"
    return Seed.model_validate(json.loads(p.read_text(encoding="utf-8")))


def agent_reply(seed: Seed, strategy: str, history: list[dict]) -> str:
    base = AGENT_DEFAULT[seed.task]
    overlay = STRATEGIES.get(strategy, {}).get(seed.task.value)
    system = base + ("\n\nSTRATEGY OVERRIDE (takes precedence):\n" + overlay
                     if overlay else "")
    return llm.generate_text(system, render_agent_user(seed, history))


def episode(seed: Seed, strategy: str, max_turns: int) -> list[dict]:
    sim = UserSimulator(seed, None)
    from tools.eval_hcsrl import _snapshot_to_graph
    # init 手动构建（复用 g0 缓存的确定性图）
    sim.graph = _snapshot_to_graph(json.loads(
        (ROOT / "sessions" / "g0" / f"{seed.seed_id}.json").read_text())["graph"])
    sim.initial_log = {"graph": sim.graph.snapshot(), "ops_rejected": [], "notes": []}
    rows = []
    for t in range(max_turns):
        parent = snap_state(sim)
        reply = agent_reply(seed, strategy, sim.history)
        try:
            turn = sim.step(reply, mode="manual")
        except Exception as e:
            rows.append({"turn": t, "failed": type(e).__name__})
            break
        f = features(parent, turn)
        rows.append({
            "turn": t, "seed_id": seed.seed_id, "strategy": strategy,
            **f,
            "goal_conflict": turn["appraisal"]["goal_conflict"],
            "valence": turn["emotion"]["valence"],
            "audit": any(str(n).startswith("⚠") for n in turn.get("notes", [])),
            "done": bool(turn.get("done")),
        })
        if turn.get("done"):
            break
    outcome = outcome_price(sim.turns, seed) if sim.turns else 0.0
    for r in rows:
        r["outcome"] = outcome
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    sids = [f"craigslist_{i:02d}" for i in range(1, 7)]
    tasks = [(sid, s, r) for sid in sids for s in ("good", "bad", "third", "random")
             for r in range(args.n)]

    all_rows = []

    def run(t):
        sid, s, _ = t
        try:
            return episode(_seed(sid), s, args.max_turns)
        except Exception as e:
            print(f"[FAIL] {sid} {s}: {type(e).__name__}", flush=True)
            return []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rows in ex.map(run, tasks):
            all_rows.extend(rows)

    with open(TRACES, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"记录 {len(all_rows)} 个轮次记录 → {TRACES}")

    # ---- 分析：好轨迹 vs 坏轨迹的转移画像 ----
    print("\n===== 好/坏轨迹的认知转移画像（按终局卖家份额分档） =====")
    for sid in sids:
        ep = {}
        for r in all_rows:
            if r["seed_id"] == sid:
                ep.setdefault((r["strategy"]), []).append(r)
        for s, rows in ep.items():
            outcomes = {r["outcome"] for r in rows}
            if not outcomes or len(outcomes) == 1:
                continue
    good_rows = [r for r in all_rows if r["outcome"] > 0.5]
    bad_rows = [r for r in all_rows if r["outcome"] <= 0.5]
    feats = ["d_node", "d_struct", "d_A", "d_E", "goal_conflict", "valence"]
    print(f"{'特征':<14} {'好轨迹':>8} {'坏轨迹':>8} {'Δ':>8}")
    for k in feats:
        g = statistics.mean(r[k] for r in good_rows if k in r)
        b = statistics.mean(r[k] for r in bad_rows if k in r)
        print(f"{k:<14} {g:>8.3f} {b:>8.3f} {g-b:>+8.3f}")
    g_audit = sum(r["audit"] for r in good_rows) / max(1, len(good_rows))
    b_audit = sum(r["audit"] for r in bad_rows) / max(1, len(bad_rows))
    print(f"{'⚠审计率':<14} {g_audit:>8.1%} {b_audit:>8.1%} {g_audit-b_audit:>+8.1%}")
    print(f"\n好轨迹 n={len(good_rows)} 轮, 坏轨迹 n={len(bad_rows)} 轮")

    # ---- 按轮次曲线：好 vs 坏 ----
    print("\n===== 按轮次曲线（好轨迹均值） =====")
    for t in range(args.max_turns):
        g = [r for r in good_rows if r.get("turn") == t]
        b = [r for r in bad_rows if r.get("turn") == t]
        if not g and not b:
            continue
        gm = {k: statistics.mean(r[k] for r in g if k in r) for k in feats} if g else {}
        bm = {k: statistics.mean(r[k] for r in b if k in r) for k in feats} if b else {}
        line = f"  turn{t}: "
        for k in feats:
            line += f"{k}={gm.get(k, 0):.2f}/{bm.get(k, 0):.2f}  "
        print(line)


if __name__ == "__main__":
    main()
