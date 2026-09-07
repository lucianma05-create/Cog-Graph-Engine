#!/usr/bin/env python3
"""H-CSRL 启发式原型（单树复现）：主轨迹 → 认知转移特征 → 选父节点 →
一步分支 → Borda 排序。未训练任何模型，全部使用引擎实际落地字段。

流程（对应 CSRL轻量树采样双路线设计 §4）：
  1. 用 good 策略走一条完整主轨迹，逐节点保存父快照
  2. 每个节点计算 d_node/d_struct/d_A/d_E → S_H 分数
  3. 选分数最高的 B 个非终止父节点
  4. 每个父节点从同一快照 fork，用其余策略各走一步（候选=替代动作）
  5. 按任务维度对同父候选做 Borda 排序
  6. 打印一棵可读的树

用法：python tools/eval_hcsrl.py --seed craigslist_01
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine import llm
from engine.prompts import AGENT_DEFAULT, render_agent_user
from engine.schema import Seed
from engine.simulator import UserSimulator
from tools.eval_ab import STRATEGIES

B = 2   # 每轨迹父节点数
PRICE_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)")


def _seed(sid: str) -> Seed:
    p = ROOT / "seeds" / f"{sid}.json"
    return Seed.model_validate(json.loads(p.read_text(encoding="utf-8")))


def agent_reply(seed: Seed, strategy: str, history: list[dict]) -> str:
    base = AGENT_DEFAULT[seed.task]
    overlay = STRATEGIES.get(strategy, {}).get(seed.task.value)
    system = base + ("\n\nSTRATEGY OVERRIDE (takes precedence):\n" + overlay
                     if overlay else "")
    return llm.generate_text(system, render_agent_user(seed, history))


def snap_state(sim: UserSimulator) -> dict:
    return {
        "graph": sim.graph.snapshot(),
        "appraisal": dict(sim.appraisal),
        "emotion": dict(sim.emotion),
        "history": list(sim.history),
    }


def features(parent: dict, turn: dict) -> dict:
    """四组实际落地特征（不读 llm_raw）。"""
    pn = {n["id"]: n for n in parent["graph"]["nodes"]}
    cn = {n["id"]: n for n in turn["graph_after"]["nodes"]}
    union = set(pn) | set(cn)
    d_node = sum(abs(cn.get(i, pn.get(i, {}))["strength"] if i in cn else 0.0
                      - (pn[i]["strength"] if i in pn else 0.0))
                 for i in union) / (4 * max(1, len(union)))
    n_struct = 0
    for o in turn.get("ops_applied", []):
        if o.get("auto"):
            continue  # 级联删除只计一次
        if o["op"] == "add":
            n_struct += 1
        elif o["op"] == "update" and o.get("content_changed"):
            n_struct += 1
        elif o["op"] == "deactivate" and not o.get("auto"):
            n_struct += 1
        elif o["op"] in ("add", "remove") and "edge" in o:
            n_struct += 1
    d_struct = n_struct / max(1, len(pn) + len(parent["graph"]["edges"]))
    pa, ca = parent["appraisal"], turn["appraisal"]
    d_A = abs(ca["goal_congruence"]) + abs(ca["controllability"] - pa["controllability"]) \
        + abs(ca["goal_conflict"] - pa["goal_conflict"])
    pe, ce = parent["emotion"], turn["emotion"]
    d_E = abs(ce["valence"] - pe["valence"]) / 2 + abs(ce["arousal"] - pe["arousal"]) \
        + (1.0 if ce["category"] != pe["category"] else 0.0)
    return {"d_node": round(d_node, 3), "d_struct": round(d_struct, 3),
            "d_A": round(d_A, 3), "d_E": round(d_E, 3)}


def borda_score(turn: dict, seed: Seed) -> float:
    """谈判任务的候选效用（单条边，不含排名——排名在父节点内进行）。"""
    a, e = turn["appraisal"], turn["emotion"]
    price_warn = any("price offer" in str(n) for n in turn.get("notes", []))
    progress = 0.0
    if turn.get("done"):
        reason = (turn.get("done_reason") or "").lower()
        if "walk" not in reason and "breakdown" not in reason:
            progress = 1.0
    return {
        "congruence": max(0.0, a["goal_congruence"]),
        "conflict": 1.0 - a["goal_conflict"],
        "progress": progress,
        "no_warning": 0.0 if price_warn else 1.0,
        "valence": (e["valence"] + 1) / 2,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", default="craigslist_01")
    ap.add_argument("--turns", type=int, default=6)
    args = ap.parse_args()

    seed = _seed(args.seed)
    sim = UserSimulator(seed, ROOT / "sessions" / "_hcsrl_tmp.json")
    sim.init()
    sim.session_file = None

    # ---- 主轨迹（good 策略）----
    print(f"===== 主轨迹（good 策略）{args.seed} =====")
    nodes = []
    for t in range(args.turns):
        parent = snap_state(sim)
        reply = agent_reply(seed, "good", sim.history)
        turn = sim.step(reply, mode="manual")
        f = features(parent, turn)
        nodes.append({"t": t, "parent": parent, "reply": reply, "turn": turn,
                      "f": f, "done": turn.get("done")})
        print(f"  turn{t+1} agent: {reply[:60]}")
        print(f"          user:  {turn['user_utterance'][:70]}")
        print(f"          特征: {f} | done={turn.get('done')}")
        if turn.get("done"):
            break

    # ---- 选父节点：S_H = mean(特征)（此处省去费用归一与分位数校准，原型简化）----
    candidates = [n for n in nodes if not n["done"] and not n["turn"].get("done")]
    scores = [(n, sum(n["f"].values()) / 4) for n in candidates]
    scores.sort(key=lambda x: -x[1])
    parents = scores[:B]
    print(f"\n===== 选中父节点（S_H 排序，B={B}）=====")
    for n, s in parents:
        print(f"  turn{n['t']+1} S_H={s:.3f}  特征={n['f']}")

    # ---- 一步分支：替代策略各走一步，Borda 排序 ----
    alts = [s for s in ("bad", "third", "random") if s != "good"]
    for n, s in parents:
        print(f"\n===== 父节点 turn{n['t']+1} 的一步分支 =====")
        cand = [{"name": "good(主边)", "turn": n["turn"],
                 "borda": borda_score(n["turn"], seed)}]
        for alt in alts:
            # 用父快照重建子环境（简化：失活节点集未恢复，原型可接受）
            child2 = UserSimulator(seed, None)
            child2.graph = _snapshot_to_graph(n["parent"]["graph"])
            child2.appraisal = dict(n["parent"]["appraisal"])
            child2.emotion = dict(n["parent"]["emotion"])
            child2.history = list(n["parent"]["history"])
            child2.initial_log = sim.initial_log
            reply = agent_reply(seed, alt, child2.history)
            try:
                t2 = child2.step(reply, mode="manual")
            except Exception as e:
                print(f"  [{alt}] 一步失败: {type(e).__name__}")
                continue
            cand.append({"name": alt, "turn": t2, "borda": borda_score(t2, seed)})
            print(f"  [{alt}] reply: {reply[:55]}")
            print(f"         user:  {t2['user_utterance'][:65]} | done={t2.get('done')}")
        # Borda 排名：每个维度排名 1→0
        dims = ["congruence", "conflict", "progress", "no_warning", "valence"]
        ranks = {d: {} for d in dims}
        for d in dims:
            vals = {c["name"]: c["borda"][d] for c in cand}
            order = sorted(vals, key=lambda k: -vals[k])
            for r, name in enumerate(order):
                ranks[d][name] = 1 - r / max(1, len(order) - 1)
        for c in cand:
            c["borda_total"] = round(sum(ranks[d][c["name"]] for d in dims) / len(dims), 3)
        print("  Borda 总分（同父排名）:")
        for c in sorted(cand, key=lambda x: -x["borda_total"]):
            print(f"    {c['name']:<14} {c['borda_total']}")

    (ROOT / "sessions" / "_hcsrl_tmp.json").unlink(missing_ok=True)
    (ROOT / "sessions" / "reports" / "_hcsrl_tmp.md").unlink(missing_ok=True)


def _snapshot_to_graph(snap: dict):
    from engine.schema import Node
    from engine.updater import CognitiveGraph, Edge
    nodes = {n["id"]: Node.model_validate(n) for n in snap["nodes"]}
    edges = { (e["from"], e["to"], e["relation"]): Edge.model_validate(e)
              for e in snap["edges"] }
    return CognitiveGraph(nodes=nodes, edges=edges, deactivated=set())


if __name__ == "__main__":
    main()
