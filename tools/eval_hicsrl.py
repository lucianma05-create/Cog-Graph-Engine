#!/usr/bin/env python3
"""H-I-CSRL 原型：迭代深化 + 验证饱和前移（无 critic）。

测试目标：
  1. importance 排序（S̃_H + σ_Borda，验证后 ×(1−置信度) 前移）选出的重点
     节点，其"候选完整 MC 回报的排序分歧"是否高于均匀选中的节点；
  2. 重点节点上"一步 Borda 排名 vs 完整 MC 排名"的一致性（Spearman）
     ——H 路线免费的机制证据：认知信号是否预测长期结果；
  3. 验证饱和项的作用：已被 MC 验证的节点在下一轮 importance 中退出
     重点集（前沿前移）。

用法：python tools/eval_hicsrl.py --seed craigslist_01
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine import llm
from engine.prompts import AGENT_DEFAULT, render_agent_user
from engine.schema import Seed
from engine.simulator import UserSimulator
from tools.eval_ab import STRATEGIES
from tools.eval_hcsrl import (features, agent_reply, borda_score, snap_state,
                              _snapshot_to_graph)

PRICE_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)")
MAIN = "good"
ALTS = ["bad", "third", "random"]


def _seed(sid: str) -> Seed:
    p = ROOT / "seeds" / f"{sid}.json"
    return Seed.model_validate(json.loads(p.read_text(encoding="utf-8")))


def spearman(a: list[float], b: list[float]) -> float:
    """简单 Spearman（按秩的 Pearson，无 scipy 依赖）。"""
    n = len(a)
    if n < 2:
        return 0.0

    def ranks(x):
        order = sorted(range(n), key=lambda i: x[i])
        r = [0] * n
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((ra[i] - ma) ** 2 for i in range(n))
    vb = sum((rb[i] - mb) ** 2 for i in range(n))
    return cov / ((va * vb) ** 0.5 + 1e-9)


def one_step(parent: dict, seed: Seed, strategy: str) -> dict:
    sim = UserSimulator(seed, None)
    sim.graph = _snapshot_to_graph(parent["graph"])
    sim.appraisal = dict(parent["appraisal"])
    sim.emotion = dict(parent["emotion"])
    sim.history = list(parent["history"])
    reply = agent_reply(seed, strategy, sim.history)
    try:
        turn = sim.step(reply, mode="manual")
    except Exception as e:
        return {"strategy": strategy, "ok": False, "err": type(e).__name__}
    return {"strategy": strategy, "ok": True, "reply": reply, "turn": turn,
            "borda": borda_score(turn, seed)}


def mc_episode(parent: dict, seed: Seed, strategy: str, max_turns: int) -> dict:
    """从父快照出发：第一步=候选策略，之后=主策略续采样到终局。"""
    sim = UserSimulator(seed, None)
    sim.graph = _snapshot_to_graph(parent["graph"])
    sim.appraisal = dict(parent["appraisal"])
    sim.emotion = dict(parent["emotion"])
    sim.history = list(parent["history"])
    reply = agent_reply(seed, strategy, sim.history)
    try:
        turn = sim.step(reply, mode="manual")
    except Exception as e:
        return {"strategy": strategy, "ok": False, "err": type(e).__name__}
    if turn.get("done"):
        return {"strategy": strategy, "ok": True, "return": _terminal_return(turn, seed),
                "done": True}
    for _ in range(max_turns - 1):
        reply = agent_reply(seed, MAIN, sim.history)
        try:
            turn = sim.step(reply, mode="manual")
        except Exception:
            return {"strategy": strategy, "ok": False, "err": "schema"}
        if turn.get("done"):
            return {"strategy": strategy, "ok": True,
                    "return": _terminal_return(turn, seed), "done": True}
    return {"strategy": strategy, "ok": True, "return": 0.0, "done": False}


def _terminal_return(turn: dict, seed: Seed) -> float:
    """谈判的简单终局回报：成交=按卖家份额 0.5+0.5·share，走人=0。"""
    reason = (turn.get("done_reason") or "").lower()
    if "walk" in reason or "breakdown" in reason:
        return 0.0
    buyer = None
    for m in PRICE_RE.findall(seed.private_persona or ""):
        buyer = float(m.replace(",", ""))
    seller = None
    for m in PRICE_RE.findall(seed.agent_private or ""):
        seller = float(m.replace(",", ""))
    listed = float(seed.notes.get("listed_price") or 0) or 1.0
    amounts = [float(m.replace(",", "")) for m in PRICE_RE.findall(turn.get("user_utterance", ""))]
    amounts = [v for v in amounts if v <= listed * 1.05]
    if not amounts or buyer is None or seller is None or seller == buyer:
        return 0.0
    share = max(0.0, min(1.0, (max(amounts) - buyer) / (seller - buyer)))
    return 0.5 + 0.5 * share


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", default="craigslist_01")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--probe-b", type=int, default=2)
    ap.add_argument("--deepen-m", type=int, default=2)
    ap.add_argument("--mc-reps", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()
    seed = _seed(args.seed)

    # ---- 主轨迹 ----
    sim = UserSimulator(seed, ROOT / "sessions" / "_hicsrl_tmp.json")
    sim.init()
    sim.session_file = None
    nodes = []
    for t in range(args.turns):
        parent = snap_state(sim)
        reply = agent_reply(seed, MAIN, sim.history)
        turn = sim.step(reply, mode="manual")
        f = features(parent, turn)
        nodes.append({"t": t, "parent": parent, "turn": turn, "f": f})
        print(f"[主] turn{t+1} 特征={f} done={turn.get('done')}")
        if turn.get("done"):
            break
    (ROOT / "sessions" / "_hicsrl_tmp.json").unlink(missing_ok=True)
    (ROOT / "sessions" / "reports" / "_hicsrl_tmp.md").unlink(missing_ok=True)
    pool = [n for n in nodes if not n["turn"].get("done")]

    # ---- 探测层：均匀选 B 个节点，各跑 ALTS 一步分支 ----
    step = max(1, len(pool) // args.probe_b)
    probes = pool[::step][:args.probe_b]
    for n in probes:
        n["branches"] = [one_step(n["parent"], seed, s) for s in ALTS]
        n["sigma_borda"] = _borda_std(n["branches"])
        print(f"[探测] turn{n['t']+1} σ_Borda={n['sigma_borda']:.3f}")
    median_sigma = sorted(n["sigma_borda"] for n in probes)[len(probes) // 2]
    for n in pool:
        if "sigma_borda" not in n:
            n["sigma_borda"] = median_sigma  # 中位数插补

    # ---- 深化迭代 ----
    verified = {}   # node t -> R̃_confidence
    for r in range(1, args.rounds + 1):
        for n in pool:
            sh = sum(n["f"].values()) / 4
            conf = verified.get(n["t"], 0.0)
            n["importance"] = (0.5 * sh + 0.5 * n["sigma_borda"]) * (1.0 - conf)
        ranked = sorted(pool, key=lambda n: -n["importance"])
        chosen = ranked[:args.deepen_m]
        print(f"\n[深化轮 {r}] 重点节点: " + ", ".join(
            f"turn{n['t']+1}(imp={n['importance']:.3f})" for n in chosen))
        for n in chosen:
            if "branches" not in n:  # 重点节点若未被探测过，先补一步分支
                n["branches"] = [one_step(n["parent"], seed, s) for s in ALTS]
                n["sigma_borda"] = _borda_std(n["branches"])
                print(f"  [补探测] turn{n['t']+1} σ_Borda={n['sigma_borda']:.3f}")
            with ThreadPoolExecutor(max_workers=4) as ex:
                futs = [ex.submit(mc_episode, n["parent"], seed, s, args.turns)
                        for s in [MAIN] + ALTS for _ in range(args.mc_reps)]
                res = [f.result() for f in futs]
            by_strat = {}
            for r_ in res:
                if r_["ok"]:
                    by_strat.setdefault(r_["strategy"], []).append(r_["return"])
            mc_mean = {s: sum(v) / len(v) for s, v in by_strat.items() if v}
            print(f"  turn{n['t']+1} MC 均值: " + ", ".join(
                f"{s}={v:.2f}" for s, v in sorted(mc_mean.items())))
            # Borda vs MC 一致性
            borda_of = {b["strategy"]: b["borda"]["congruence"] + b["borda"]["progress"]
                        + b["borda"]["no_warning"] for b in n["branches"] if b["ok"]}
            common = [s for s in mc_mean if s in borda_of]
            if len(common) >= 2:
                rho = spearman([borda_of[s] for s in common],
                               [mc_mean[s] for s in common])
                print(f"  Borda-MC Spearman = {rho:+.2f}（{len(common)} 个共同候选）")
            # 验证饱和：该节点置信度上升
            spread = max(mc_mean.values()) - min(mc_mean.values()) if mc_mean else 0.0
            verified[n["t"]] = min(1.0, verified.get(n["t"], 0.0) + 0.5)
            verified["R" + str(n["t"])] = spread
            n["mc_spread"] = spread
        # 与均匀对照：均匀选同样数量节点的 mc_spread（用上一轮已测的 probes 作近似）
        if r == 1:
            # 对未选中的前两个节点也跑 MC 作对照（近似的均匀基线）
            rest = [n for n in pool if "mc_spread" not in n][:args.deepen_m]
            with ThreadPoolExecutor(max_workers=4) as ex:
                for n in rest:
                    futs = [ex.submit(mc_episode, n["parent"], seed, s, args.turns)
                            for s in [MAIN] + ALTS for _ in range(args.mc_reps)]
                    res = [f.result() for f in futs]
                    by = {}
                    for r_ in res:
                        if r_["ok"]:
                            by.setdefault(r_["strategy"], []).append(r_["return"])
                    means = {s: sum(v) / len(v) for s, v in by.items() if v}
                    n["mc_spread"] = max(means.values()) - min(means.values()) if means else 0.0
            chosen_spreads = [n["mc_spread"] for n in chosen if "mc_spread" in n]
            rest_spreads = [n["mc_spread"] for n in rest if "mc_spread" in n]
            if chosen_spreads and rest_spreads:
                print(f"\n[对照] 重点节点 MC 分歧均值 = {sum(chosen_spreads)/len(chosen_spreads):.3f} "
                      f"vs 均匀节点 = {sum(rest_spreads)/len(rest_spreads):.3f}")

    print("\n[验证饱和] 各节点验证置信度:", {f"turn{t+1}": v for t, v in verified.items()
                                            if not str(t).startswith("R")})


def _borda_std(branches: list[dict]) -> float:
    vals = [sum(b["borda"].values()) / len(b["borda"]) for b in branches if b["ok"]]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


if __name__ == "__main__":
    main()
