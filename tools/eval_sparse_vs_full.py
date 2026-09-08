#!/usr/bin/env python3
"""方向信号稀疏采样 vs 全量 MC：6 谈判种子批量对比。

每种子：
  1. 主轨迹（good 策略）
  2. 全量 MC 基线：所有非终止主节点 × 4 策略 × reps 完整走到底
  3. 稀疏迭代（方向版 importance）：探测 B 节点 → 深化 M 节点/轮 × R 轮，
     子节点入池 + R̃ 回传
指标（跨种子聚合）：
  - 高回报节点命中率（稀疏选中的主节点 ∩ 全量 top-2）
  - 最优回报比（稀疏发现的最优 / 全量最优）
  - 成本比（稀疏 MC 局数 / 全量 MC 局数）
  - 单位成本价值（发现回报 / MC 局数）

用法：python tools/eval_sparse_vs_full.py --reps 2 --workers 48
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

from engine.schema import Seed
from engine.simulator import UserSimulator
from tools.eval_hicsrl import (MAIN, ALTS, _borda_std, agent_reply,
                               directional_features, mc_episode, one_step,
                               snap_state)
from tools.eval_hcsrl import features, _snapshot_to_graph

K = 4          # 候选策略数（含主边）
B = 2          # 探测节点数
M = 2          # 每轮深化节点数
R = 2          # 深化轮数


def _seed(sid: str) -> Seed:
    p = ROOT / "seeds" / f"{sid}.json"
    return Seed.model_validate(json.loads(p.read_text(encoding="utf-8")))


def run_seed(sid: str, reps: int, max_turns: int, workers: int) -> dict:
    seed = _seed(sid)
    tmp = ROOT / "sessions" / f"_svf_tmp_{sid}.json"
    sim = UserSimulator(seed, tmp)
    sim.init()
    sim.session_file = None
    tmp.unlink(missing_ok=True)
    (ROOT / "sessions" / "reports" / f"_svf_tmp_{sid}.md").unlink(missing_ok=True)
    nodes = []
    for t in range(max_turns):
        parent = snap_state(sim)
        reply = agent_reply(seed, MAIN, sim.history)
        turn = sim.step(reply, mode="manual")
        nodes.append({"t": t, "parent": parent, "turn": turn,
                      "f": features(parent, turn)})
        if turn.get("done"):
            break

    pool = [n for n in nodes if not n["turn"].get("done")]

    # ---- 全量 MC ----
    def run_mc(n):
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(mc_episode, n["parent"], seed, s, max_turns)
                    for s in [MAIN] + ALTS for _ in range(reps)]
            return [f.result() for f in futs]
    full = {}
    for n in pool:
        res = run_mc(n)
        by = {}
        for r_ in res:
            if r_["ok"]:
                by.setdefault(r_["strategy"], []).append(r_["return"])
        means = {s: sum(v) / len(v) for s, v in by.items() if v}
        full[n["t"]] = {"best": max(means.values()) if means else 0.0,
                        "means": means}
    full_best = max(v["best"] for v in full.values())
    top2 = sorted(full, key=lambda k: -full[k]["best"])[:2]
    full_cost = len(pool) * (K * reps)

    # ---- 稀疏迭代（方向版）----
    for n in pool:
        n["d_features"] = directional_features(n["parent"], n["turn"])
    step = max(1, len(pool) // B)
    probes = pool[::step][:B]
    for n in probes:
        n["branches"] = [one_step(n["parent"], seed, s) for s in ALTS]
        n["sigma_borda"] = _borda_std(n["branches"])
    med = sorted(n["sigma_borda"] for n in probes)[len(probes) // 2]
    for n in pool:
        n.setdefault("sigma_borda", med)
        n["node_id"] = f"main{n['t']}"
    verified, node_seq = {}, 0
    sparse_picks, sparse_cost = set(), 0
    for r in range(R):
        for n in pool:
            conf = verified.get(n["node_id"], 0.0)
            df = n["d_features"]
            sh = (df["pos_congruence"] * 0.4 + df["pos_valence"] * 0.2
                  + df["flip_congruence"] * 0.2 + df["flip_valence"] * 0.2)
            n["imp"] = (0.5 * sh + 0.5 * n["sigma_borda"]) * (1.0 - conf) \
                + 0.3 * n.get("rtilde", 0.0)
        ranked = sorted(pool, key=lambda n: -n["imp"])
        chosen = ranked[:M]
        for n in chosen:
            if n["node_id"].startswith("main"):
                sparse_picks.add(n["t"])
            if "branches" not in n:
                n["branches"] = [one_step(n["parent"], seed, s) for s in ALTS]
            sparse_cost += K * reps
            res = run_mc(n)
            by = {}
            for r_ in res:
                if r_["ok"]:
                    by.setdefault(r_["strategy"], []).append(r_["return"])
            means = {s: sum(v) / len(v) for s, v in by.items() if v}
            children = [r_ for r_ in res if r_["ok"] and r_.get("child")]
            if children:
                n["rtilde"] = sum(r_["return"] for r_ in children) / len(children)
            for r_ in children[:3]:
                node_seq += 1
                p = {"appraisal": n["parent"]["appraisal"],
                     "emotion": n["parent"]["emotion"]}
                t_ = {"appraisal": r_["child"]["appraisal"],
                      "emotion": r_["child"]["emotion"]}
                pool.append({"t": f"{n['t']}→{r_['strategy']}",
                             "parent": r_["child"], "turn": {},
                             "f": {}, "d_features": directional_features(p, t_),
                             "sigma_borda": n["sigma_borda"],
                             "rtilde": r_["return"], "node_id": f"c{node_seq}"})
            verified[n["node_id"]] = min(1.0, verified.get(n["node_id"], 0.0) + 0.5)
    sparse_best = max(full[t]["best"] for t in sparse_picks) if sparse_picks else 0.0
    hit = len(sparse_picks & set(top2))
    print(f"{sid}: 全量最优={full_best:.2f} 稀疏最优={sparse_best:.2f} "
          f"命中top2={hit}/2 成本 {sparse_cost}/{full_cost}")
    return {"full_best": full_best, "sparse_best": sparse_best,
            "hit": hit, "sparse_cost": sparse_cost, "full_cost": full_cost,
            "n_nodes": len(pool), "picks": sorted(sparse_picks)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    sids = [f"craigslist_{i:02d}" for i in range(1, 7)]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_seed, sid, args.reps, args.max_turns, 4): sid
                for sid in sids}
        for f in futs:
            try:
                results.append(f.result())
            except Exception as e:
                print(f"[FAIL] {futs[f]}: {type(e).__name__}: {str(e)[:80]}")

    print("\n===== 聚合（跨种子） =====")
    hits = [r["hit"] for r in results]
    br = [r["sparse_best"] / r["full_best"] for r in results if r["full_best"] > 0]
    cr = [r["sparse_cost"] / r["full_cost"] for r in results]
    vpc_s = [r["sparse_best"] / r["sparse_cost"] for r in results]
    vpc_f = [r["full_best"] / r["full_cost"] for r in results]
    print(f"  高回报节点命中率: {statistics.mean(hits):.2f}/2（{sum(hits)}/{2*len(results)}）")
    print(f"  最优回报比（稀疏/全量）: {statistics.mean(br):.2f}")
    print(f"  成本比（稀疏/全量）: {statistics.mean(cr):.2f}")
    print(f"  单位成本价值 稀疏={statistics.mean(vpc_s):.4f} vs 全量={statistics.mean(vpc_f):.4f} "
          f"（比值 {statistics.mean([s/f for s, f in zip(vpc_s, vpc_f) if f > 0]):.2f}×）")
    json.dump(results, open("sessions/ab_sparse_vs_full.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
