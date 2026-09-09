#!/usr/bin/env python3
"""固定轮数稀疏树搜索（sh_dir 先验 + UCB） vs 束搜索 vs 全量 MC。

每种子三种方法，同一任务回报函数：
  1. 全量 MC：主轨迹所有非终止节点 × 4 策略 × 2 重复（oracle 基线）
  2. 束搜索：从对话起点，每层保留 top-b 节点（b=2, D=2），每节点展开
     全部 4 策略各 1 局；节点排序只用实测回报（无先验、无探索项）
  3. 树搜索（本方法）：sh_dir 先验 + UCB（分数+探索）+ ε 随机名额，
     每选中节点只花 1 局（策略轮转），L=3 轮 × M=2 节点；终止 = 跑满
     L 轮 或 实测均值 Q̄（N≥2）≥ 任务阈值 θ（ES 0.75 / 捐赠 1.4 / 谈判 0.7）

指标（跨种子聚合）：最优回报比（方法最优/全量最优）、成本比
（MC 局数）、树搜索早停率。日志直接写 sessions/eval_tree_fixed.log
（批跑铁律：不依赖管道缓冲）。

用法：python tools/eval_tree_fixed.py --task all
     python tools/eval_tree_fixed.py --smoke --seed esconv_01
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import random
import statistics
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.schema import Seed
from engine.simulator import UserSimulator
from tools.eval_hcsrl import agent_reply, features, snap_state, _snapshot_to_graph
from tools.eval_hicsrl import directional_features, _terminal_return
from tools.eval_ab import judge_emotional, outcome_donation

STRATS = ["good", "bad", "third", "random"]
MAIN = "good"
LOG_PATH = ROOT / "sessions" / "eval_tree_fixed.log"

# 任务阈值（CSRL固定轮数树搜索设计.md §3）
THETA = {"es": 0.75, "donation": 1.4, "price": 0.7}  # price 按实测 full_best 均值 0.686 校准
MAX_TURNS = {"es": 10, "donation": 12, "price": 8}
SEEDS = {
    "es": ["esconv_01", "esconv_02", "esconv_06", "esconv_08"],
    "donation": ["p4g_01", "p4g_03", "p4g_08", "p4g_10"],
    "price": ["craigslist_01", "craigslist_02", "craigslist_03", "craigslist_04"],
}


def log(msg: str) -> None:
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _seed(sid: str) -> Seed:
    p = ROOT / "seeds" / f"{sid}.json"
    return Seed.model_validate(json.loads(p.read_text(encoding="utf-8")))


def _norm_fn(return_mode: str):
    if return_mode == "es":
        return lambda r: (r + 0.5) / 1.5
    if return_mode == "donation":
        return lambda r: r / 1.5
    return lambda r: r


def run_episode(parent: dict, seed: Seed, strategy: str, max_turns: int,
                return_mode: str) -> dict:
    """一局：从 parent 快照出发，第一句=strategy，其后 good 走到终局。
    回报按任务计算；child = 第一句之后的快照（子节点）。"""
    sim = UserSimulator(seed, None)
    sim.graph = _snapshot_to_graph(parent["graph"])
    sim.appraisal = dict(parent["appraisal"])
    sim.emotion = dict(parent["emotion"])
    sim.history = list(parent["history"])
    conv = []

    def _step(reply: str):
        try:
            turn = sim.step(reply, mode="manual")
        except Exception:
            return None
        conv.append({"agent_reply": reply,
                     "user_utterance": turn.get("user_utterance", "")})
        return turn

    reply = agent_reply(seed, strategy, sim.history)
    turn = _step(reply)
    if turn is None:
        return {"ok": False, "err": "schema"}
    done1 = bool(turn.get("done"))   # 第一步就结束 → 子节点才是终止节点
    child = snap_state(sim)
    while not turn.get("done") and len(conv) < max_turns:
        reply = agent_reply(seed, MAIN, sim.history)
        turn = _step(reply)
        if turn is None:
            return {"ok": False, "err": "schema"}
    if return_mode == "price":
        r = _terminal_return(turn, seed)
    elif return_mode == "donation":
        r = outcome_donation([turn])
    else:
        r = judge_emotional(seed, conv, samples=3)
    return {"ok": True, "return": r, "child": child,
            "done": bool(turn.get("done")), "done1": done1}


def main_walk(seed: Seed, max_turns: int) -> list:
    """主轨迹（good 走到底）。返回节点列表：{t, parent, prior}。"""
    tmp = ROOT / "sessions" / f"_etf_tmp_{uuid.uuid4().hex[:8]}.json"
    sim = UserSimulator(seed, tmp)
    sim.init()
    sim.session_file = None
    nodes = []
    for t in range(max_turns):
        parent = snap_state(sim)
        reply = agent_reply(seed, MAIN, sim.history)
        try:
            turn = sim.step(reply, mode="manual")
        except Exception:  # flash 结构失败 ~20%，主轨迹第一步也重试一次
            try:
                turn = sim.step(reply, mode="manual")
            except Exception:
                break
        df = directional_features(parent, turn)
        sh = (df["pos_congruence"] * 0.4 + df["pos_valence"] * 0.2
              + df["flip_congruence"] * 0.2 + df["flip_valence"] * 0.2)
        nodes.append({"t": t, "parent": parent, "turn": turn,
                      "sh": sh, "f": features(parent, turn)})
        if turn.get("done"):
            break
    tmp.unlink(missing_ok=True)
    (ROOT / "sessions" / "reports" / tmp.name.replace(".json", ".md")).unlink(missing_ok=True)
    return nodes


def full_mc(nodes: list, seed: Seed, max_turns: int, return_mode: str,
            reps: int = 2) -> tuple:
    """全量 MC：每个非终止主节点 × 4 策略 × reps 局。"""
    pool = [n for n in nodes if not n["turn"].get("done")]
    per_node = {}
    cost = 0
    for n in pool:
        by = {}
        for s in STRATS:
            for _ in range(reps):
                r = run_episode(n["parent"], seed, s, max_turns, return_mode)
                cost += 1
                if not r["ok"]:  # episode 级重试
                    r = run_episode(n["parent"], seed, s, max_turns, return_mode)
                    cost += 1
                if r["ok"]:
                    by.setdefault(s, []).append(r["return"])
        means = {s: sum(v) / len(v) for s, v in by.items() if v}
        per_node[n["t"]] = {"best": max(means.values()) if means else 0.0,
                            "means": means}
    full_best = max(v["best"] for v in per_node.values())
    top2 = sorted(per_node, key=lambda k: -per_node[k]["best"])[:2]
    return full_best, top2, cost, per_node


def beam_search(nodes: list, seed: Seed, max_turns: int, return_mode: str,
                b: int = 2, D: int = 2) -> dict:
    """束搜索：根=对话起点，每层保留 top-b 节点（按单局回报），每节点
    展开全部 4 策略各 1 局。无先验、无探索项、无探测层。"""
    root = nodes[0]["parent"]  # G0 之后、第一句话之前的快照
    beam = [{"label": "root", "parent": root, "ret": None}]
    best = None
    cost = 0
    for d in range(D):
        kids = []
        for node in beam:
            for s in STRATS:
                r = run_episode(node["parent"], seed, s, max_turns, return_mode)
                cost += 1
                if not r["ok"]:
                    r = run_episode(node["parent"], seed, s, max_turns, return_mode)
                    cost += 1
                if not r["ok"]:
                    continue
                label = f"{node['label']}→{s}"
                if best is None or r["return"] > best["return"]:
                    best = {"label": label, "strategy": s, "return": r["return"]}
                if not r.get("done1") and r.get("child"):
                    kids.append({"label": label, "parent": r["child"],
                                 "ret": r["return"]})
        kids.sort(key=lambda x: -(x["ret"] or -1))
        beam = kids[:b]
        if not beam:
            break
    return {"best_return": best["return"] if best else 0.0,
            "best_label": best["label"] if best else "-",
            "cost": cost}


def tree_search(nodes: list, seed: Seed, max_turns: int, return_mode: str,
                L: int = 3, M: int = 2, alpha: float = 1.0,
                c: float = 0.3, eps: float = 0.25) -> dict:
    """固定轮数稀疏树搜索：sh_dir 先验 + UCB + ε 随机名额。每选中节点
    1 局，策略轮转。终止：跑满 L 轮 或 实测均值 Q̄（N≥2）≥ θ。"""
    norm = _norm_fn(return_mode)
    theta_norm = norm(THETA[return_mode])
    pool = []
    for n in nodes:
        if n["turn"].get("done"):
            continue
        pool.append({"label": f"t{n['t']}", "t": n["t"], "parent": n["parent"],
                     "Q": 0.0, "N": 0, "prior": n["sh"], "tried": set(),
                     "is_main": True})
    best_ep = None
    cost = 0
    early_stop = False
    rounds_used = 0
    for rnd in range(L):
        rounds_used = rnd + 1
        n_total = sum(x["N"] for x in pool)
        best_qbar = None
        for x in pool:
            qbar = x["Q"] / x["N"] if x["N"] else 0.0
            x["qhat"] = (alpha * x["prior"] + x["N"] * qbar) / (alpha + x["N"])
            if x["N"] >= 2:
                best_qbar = qbar if best_qbar is None else max(best_qbar, qbar)
        if best_qbar is not None and best_qbar >= theta_norm:  # 早停看实测均值，先验只参与选择
            early_stop = True
            break
        for x in pool:
            x["score"] = x["qhat"] + c * math.sqrt(
                math.log(n_total + 1) / (1 + x["N"]))
        ranked = sorted(pool, key=lambda x: -x["score"])
        chosen = []
        for _ in range(min(M, len(pool))):
            if random.random() < eps:  # ε 随机名额：显式探索平衡
                n = random.choice([x for x in pool if x not in chosen])
            else:
                n = next(x for x in ranked if x not in chosen)
            chosen.append(n)
        for x in chosen:
            s = next((s for s in STRATS if s not in x["tried"]), None)
            if s is None:  # 4 策略都试过 → 从头轮转
                s = STRATS[len(x["tried"]) % 4]
            r = run_episode(x["parent"], seed, s, max_turns, return_mode)
            cost += 1
            if not r["ok"]:
                r = run_episode(x["parent"], seed, s, max_turns, return_mode)
                cost += 1
            if r["ok"]:
                x["Q"] += norm(r["return"])
                x["N"] += 1
                x["tried"].add(s)
                if best_ep is None or r["return"] > best_ep["return"]:
                    best_ep = {"label": x["label"], "strategy": s,
                               "return": r["return"]}
                if r.get("child") and not r.get("done1"):
                    pool.append({"label": f"{x['label']}→{s}", "t": None,
                                 "parent": r["child"], "Q": norm(r["return"]),
                                 "N": 1, "prior": 0.0, "tried": set(),
                                 "is_main": False})
    return {"best_return": best_ep["return"] if best_ep else 0.0,
            "best_label": best_ep["label"] if best_ep else "-",
            "best_strategy": best_ep["strategy"] if best_ep else "-",
            "cost": cost, "rounds_used": rounds_used, "early_stop": early_stop}


def run_seed(sid: str, return_mode: str) -> dict:
    seed = _seed(sid)
    max_turns = MAX_TURNS[return_mode]
    nodes = main_walk(seed, max_turns)
    if not nodes:  # 首步连续 schema 失败 → 明确抛错，交给种子级重试
        raise ValueError(f"{sid}: empty main walk")
    full_best, top2, full_cost, per_node = full_mc(nodes, seed, max_turns,
                                                   return_mode)
    beam = beam_search(nodes, seed, max_turns, return_mode)
    tree = tree_search(nodes, seed, max_turns, return_mode)
    out = {
        "seed": sid, "task": return_mode,
        "full_best": full_best, "full_cost": full_cost, "top2": top2,
        "n_nodes": len([n for n in nodes if not n["turn"].get("done")]),
        "beam": beam, "tree": tree,
    }
    log(f"{sid} [{return_mode}] 全量最优={full_best:.3f}(成本{full_cost}) | "
        f"束搜索={beam['best_return']:.3f}(成本{beam['cost']}, {beam['best_label']}) | "
        f"树搜索={tree['best_return']:.3f}(成本{tree['cost']}, {tree['best_label']}, "
        f"{tree['rounds_used']}轮, 早停={'是' if tree['early_stop'] else '否'})")
    return out


def _aggregate(results: list, task: str) -> None:
    if not results:
        return
    log(f"\n===== [{task}] 聚合 {len(results)} 种子 =====")
    fb = [r["full_best"] for r in results]
    log(f"  全量最优均值 = {statistics.mean(fb):.3f}")
    for method in ("beam", "tree"):
        rs = [r[method] for r in results]
        br = [r["best_return"] / f for r, f in zip(rs, fb) if f > 0]
        cr = [r["cost"] / r2["full_cost"] for r, r2 in zip(rs, results)]
        log(f"  {method:<5} 最优回报比 = {statistics.mean(br):.2f} "
            f"成本比 = {statistics.mean(cr):.2f} "
            f"单位成本价值 = {statistics.mean([r['best_return']/r['cost'] for r in rs]):.4f}"
            f" vs 全量 = {statistics.mean([r['full_best']/r['full_cost'] for r in results]):.4f}")
    trees = [r["tree"] for r in results]
    es_rate = sum(1 for t in trees if t["early_stop"]) / len(trees)
    rounds = statistics.mean([t["rounds_used"] for t in trees])
    log(f"  树搜索早停率 = {es_rate:.0%}，平均用 {rounds:.1f}/3 轮")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="all",
                    choices=["price", "es", "donation", "all"])
    ap.add_argument("--seeds", default=None, help="逗号分隔（覆盖默认种子集）")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", action="store_true", help="单种子冒烟（跳过全量 MC）")
    ap.add_argument("--seed", default="esconv_01")
    args = ap.parse_args()

    tasks = ["price", "es", "donation"] if args.task == "all" else [args.task]

    if args.smoke:
        seed = _seed(args.seed)
        task = next(t for t in tasks if args.seed in SEEDS[t])
        nodes = main_walk(seed, 6)
        tree = tree_search(nodes, seed, 6, task, L=2, M=1)
        beam = beam_search(nodes, seed, 6, task, b=1, D=1)
        log(f"[SMOKE] {args.seed} tree={tree} beam={beam}")
        return

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for task in tasks:
            sids = ([s.strip() for s in args.seeds.split(",") if s.strip()]
                    if args.seeds else SEEDS[task])
            for sid in sids:
                futs[ex.submit(run_seed, sid, task)] = (sid, task)
        for f in futs:
            sid, task = futs[f]
            for attempt in range(3):  # seed 级重试：失败即重新提交
                try:
                    results.append(f.result())
                    break
                except Exception as e:
                    if attempt == 2:
                        log(f"[FAIL] ({sid}, {task}): {type(e).__name__}: {str(e)[:100]}")
                    else:
                        f = ex.submit(run_seed, sid, task)
    out_path = ROOT / "sessions" / "eval_tree_fixed_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1,
                                   default=str), encoding="utf-8")
    for task in tasks:
        _aggregate([r for r in results if r["task"] == task], task)
    log(f"结果已写入 {out_path}")


if __name__ == "__main__":
    main()
