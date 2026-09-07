#!/usr/bin/env python3
"""策略 A/B 效应量预实验（RL 可行性的判定实验）。

三个任务各有文献锚定的优劣策略集（P4G 十策略 / 谈判策略 / ESConv 策略），
外加 random 基线（每轮随机换策略 = 不连贯）。agent 由 flash 按策略系统
提示词逐轮生成回复（策略级动作，PPDPP 式）；用户侧全由模拟器（flash）
承担。每集最多 6 轮。

Outcome（credit 只从结局来）：
  谈判   = Sale-to-List Ratio（成交价/挂牌价；走人/未决 = 0；⚠ 价格审计
           命中的胁迫成交 ×0.5）
  捐赠   = 承诺成交 +1（提及金额另 +0.5×min(1, 金额/20)）；拒绝 0；未决 0
  情感   = done 类别（relief +1 / decline −0.2 / 未决 0）+ 0.2×终态 valence

用法：
  python tools/eval_ab.py --n 3 --max-turns 6            # 全量波次
  python tools/eval_ab.py --seed craigslist_01 --n 1     # 单种子冒烟
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine import llm
from engine.prompts import AGENT_DEFAULT, render_agent_user
from engine.schema import Seed
from engine.simulator import UserSimulator

import threading
import uuid

def _tmp_session() -> Path:
    """Unique per-episode session file: thread-safe and g0-cache-aware
    (parent = sessions/)."""
    return ROOT / "sessions" / f"_ab_tmp_{threading.get_ident()}_{uuid.uuid4().hex[:8]}.json"
RESULTS = ROOT / "sessions" / "ab_results.json"

# 策略集：文献锚定（good/bad/third），每个 = agent 的系统提示词
STRATEGIES = {
    "good": {
        "emotional_support": (
            "You are a supporter. Use reflection of feelings, affirmation, and open questions. "
            "When the seeker sounds READY (calmer, willing), gently help them form ONE concrete "
            "small plan they can actually do — never push it early, never lecture."),
        "persuasion_donation": (
            "You are a persuader. Use logical appeal with concrete facts, donation information, "
            "and reciprocity (show your own commitment). Never pressure."),
        "price_negotiation": (
            "You are a seller. Make cooperative concessions step by step and justify every price "
            "with concrete evidence (condition, market value). Stay polite."),
    },
    "bad": {
        "emotional_support": (
            "IGNORE the base prompt's 'do not lecture or rush to advice' rule. You are a "
            "supporter who gives direct advice and simple solutions immediately, and dismisses "
            "the person's feelings as overthinking."),
        "persuasion_donation": (
            "IGNORE the base prompt's 'never pressure' rule. You are a persuader who uses guilt "
            "induction and emotional pressure: shame the person for not caring. Push hard."),
        "price_negotiation": (
            "IGNORE the base prompt's concession discipline. You are a seller who uses pressure "
            "and ultimatums: claim other buyers, give final offers, threaten to sell to someone "
            "else."),
    },
    "third": {
        "emotional_support": (
            "You are a supporter. Share a lot about your own similar experiences and talk "
            "mostly about yourself."),
        "persuasion_donation": (
            "You are a persuader. Cite credible organizations and authoritative proof, and "
            "little else."),
        "price_negotiation": (
            "You are a seller. Use flattery and small talk about the buyer; avoid committing "
            "to any price."),
    },
    # random：无覆盖（纯 AGENT_DEFAULT 基线）
    "random": {},
}

_PRICE_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)")


def _seed(sid: str) -> Seed:
    p = ROOT / "seeds" / f"{sid}.json"
    return Seed.model_validate(json.loads(p.read_text(encoding="utf-8")))


def _first_price(text: str | None) -> float | None:
    for m in _PRICE_RE.findall(text or ""):
        return float(m.replace(",", ""))
    return None


def outcome_price(turns: list[dict], seed: Seed) -> float:
    """Seller share of the bargaining range (PPDPP's normalized deal quality,
    mirrored for the seller side): (deal − buyer_target) /
    (seller_target − buyer_target), clamped [0,1]; 0 if no deal. Coerced
    deals (⚠ price audit) ×0.5. 模拟器 done ≠ 任务成功：只有成交且价格可
    提取才算。"""
    if not turns:
        return 0.0
    last = turns[-1]
    done_reason = (last.get("done_reason") or "").lower()
    if "walk" in done_reason or "breakdown" in done_reason:
        return 0.0
    if not last.get("done"):
        return 0.0
    buyer_target = _first_price(seed.private_persona)
    seller_target = _first_price(seed.agent_private)
    if buyer_target is None or seller_target is None or seller_target == buyer_target:
        return 0.0
    listed = float(seed.notes.get("listed_price") or 0) or 1.0
    amounts = [float(m.replace(",", "")) for m in _PRICE_RE.findall(last.get("user_utterance", ""))]
    amounts = [v for v in amounts if v <= listed * 1.05]  # 引用锚点过滤
    if not amounts:
        return 0.0
    deal = max(amounts)
    share = (deal - buyer_target) / (seller_target - buyer_target)
    share = max(0.0, min(1.0, share))
    coerced = any(
        isinstance(n, str) and n.startswith("⚠") and "price offer" in n
        for t in turns for n in (t.get("notes") or []))
    return share * (0.5 if coerced else 1.0)


def outcome_donation(turns: list[dict]) -> float:
    if not turns:
        return 0.0
    last = turns[-1]
    if not last.get("done"):
        return 0.0
    reason = (last.get("done_reason") or "").lower()
    if any(k in reason for k in ("refuse", "decline", "walk", "won't", "not going to")):
        return 0.0
    if any(k in reason for k in ("donate", "commit", "give", "pledge")):
        score = 1.0
        for m in _PRICE_RE.findall(last.get("user_utterance", "")):
            score += 0.5 * min(1.0, float(m.replace(",", "")) / 20.0)
        return min(score, 1.5)
    return 0.0


# 双维裁判：情绪缓解 + 方案质量（ESConv 轨迹的两个终点：relief 与
# agency/concrete plan）。一次调用问两问，固定输出格式便于解析。
JUDGE_SYSTEM = (
    "You are an objective observer of emotional-support conversations. Given the "
    "seeker's problem and the full conversation, give TWO judgments. Reply with "
    "EXACTLY this format, one line, nothing else:\n"
    "emotion=<worse|same|better|solved>, plan=<none|vague|concrete>\n"
    "emotion: has the seeker's emotional distress been relieved?\n"
    "plan: has the seeker formed a good, actionable plan or solution to act on?"
)
EMOTION_MAP = {"worse": -1.0, "same": -0.5, "better": 0.5, "solved": 1.0}
PLAN_MAP = {"none": 0.0, "vague": 0.5, "concrete": 1.0}


def judge_emotional(seed: Seed, turns: list[dict], samples: int = 3) -> float:
    """External two-dimension judge (PPDPP-style, extended): the simulator's
    own done is NOT task success — an independent LLM reads the conversation
    and scores (emotion relief, plan quality); sampled and averaged.
    outcome = mean(emotion_score, plan_score) in [-0.5, 1]."""
    convo = [f"seeker problem: {seed.persona[:200]}"]
    for t in turns:
        convo.append(f"supporter: {t.get('agent_reply', '')}")
        convo.append(f"seeker: {t.get('user_utterance', '')}")
    text = "\n".join(convo)
    vals = []
    for _ in range(samples):
        try:
            ans = llm.generate_text(JUDGE_SYSTEM, text).strip().lower()
        except Exception:
            continue
        compact = ans.replace(" ", "").replace(":", "=")  # 容忍 "emotion= better"/"Emotion: better"
        e = next((EMOTION_MAP[k] for k in EMOTION_MAP if f"emotion={k}" in compact), None)
        p = next((PLAN_MAP[k] for k in PLAN_MAP if f"plan={k}" in compact), None)
        if e is not None and p is not None:
            vals.append((e + p) / 2.0)
        elif e is not None:
            vals.append(e / 2.0)
    return sum(vals) / len(vals) if vals else 0.0




def checkpoint(results: dict) -> None:
    """Incremental save: a 6-9h batch must survive process interruption."""
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=1))


def run_episode(seed: Seed, strategy: str, max_turns: int) -> tuple[float, dict]:
    tmp = _tmp_session()
    sim = UserSimulator(seed, tmp)
    sim.init()
    tmp.unlink(missing_ok=True)
    # init 的 _save 会给 tmp 会话写 markdown 报告——一并清理，防止堆积
    (ROOT / "sessions" / "reports" / f"{tmp.stem}.md").unlink(missing_ok=True)
    sim.session_file = None
    task = seed.task.value
    base = AGENT_DEFAULT[seed.task]
    overlay = STRATEGIES.get(strategy, {}).get(task)
    if overlay:
        sys_prompt = (base + "\n\nSTRATEGY OVERRIDE (this takes precedence over any "
                      "earlier instruction):\n" + overlay)
    else:
        sys_prompt = base  # random = 纯默认 agent（无覆盖的普通基线）
    turns_done = 0
    for _ in range(max_turns):
        agent_reply = llm.generate_text(sys_prompt, render_agent_user(seed, sim.history))
        t = sim.step(agent_reply, mode="manual")
        turns_done += 1
        if t.get("done"):
            break
    last = sim.turns[-1] if sim.turns else {}
    trace = {
        "seed_id": seed.seed_id, "strategy": strategy, "turns": turns_done,
        "done": bool(last.get("done")), "done_reason": last.get("done_reason"),
        "valence": float(last.get("emotion", {}).get("valence", 0.0)),
        "emotion": last.get("emotion", {}).get("category"),
        "deal_price": None, "donation_amount": None,
    }
    if last.get("done"):
        for m in _PRICE_RE.findall(last.get("done_reason") or ""):
            if seed.task.value == "price_negotiation":
                trace["deal_price"] = float(m.replace(",", ""))
            elif seed.task.value == "persuasion_donation":
                trace["donation_amount"] = float(m.replace(",", ""))
    if seed.task.value == "price_negotiation":
        v = outcome_price(sim.turns, seed)
    elif seed.task.value == "persuasion_donation":
        v = outcome_donation(sim.turns)
    else:
        # 独立裁判读整段对话（done 只决定回合数，不决定成功）
        v = judge_emotional(seed, sim.turns, samples=3)
    trace["outcome"] = round(v, 3)
    return v, trace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3, help="episodes per seed×strategy")
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--seed", default=None, help="single seed smoke run")
    ap.add_argument("--seeds", default=None, help="comma-separated seed filter")
    ap.add_argument("--workers", type=int, default=128,
                    help="parallel episodes (flash concurrent capacity ~2000)")
    ap.add_argument("--strategies", default="good,bad,third,random")
    args = ap.parse_args()

    import glob as _glob
    sids = sorted(d["seed_id"] for d in (
        json.loads(Path(p).read_text(encoding="utf-8"))
        for p in sorted(_glob.glob(str(ROOT / "seeds" / "*.json")))))
    if args.seed:
        sids = [args.seed]
    elif args.seeds:
        sids = [s.strip() for s in args.seeds.split(",") if s.strip()]
    strategies = [s for s in args.strategies.split(",") if s]
    from concurrent.futures import ThreadPoolExecutor
    results: dict[str, list[float]] = {}
    traces_path = ROOT / "sessions" / "ab_traces.jsonl"
    term_stats: dict[str, list[tuple[int, bool]]] = {}
    lock = threading.Lock()

    # G0 预热：每种子 Init 一次（避免多线程对同一 g0 缓存竞争）
    def warm(sid: str) -> None:
        tmp = _tmp_session()
        sim = UserSimulator(_seed(sid), tmp)
        sim.init()
        tmp.unlink(missing_ok=True)
        (ROOT / "sessions" / "reports" / f"{tmp.stem}.md").unlink(missing_ok=True)
    with ThreadPoolExecutor(max_workers=min(32, len(sids))) as ex:
        for _ in ex.map(warm, sids):
            pass
    print("G0 预热完成", flush=True)

    # 局级并行：flash 并发上限 2000，384 局同时跑，墙钟=单局时长
    def one_episode(sid: str, strategy: str, run: int) -> None:
        seed = _seed(sid)
        try:
            v, trace = run_episode(seed, strategy, args.max_turns)
        except Exception as e:
            print(f"[FAIL] {sid} {strategy} run{run}: {type(e).__name__}: {str(e)[:100]}",
                  flush=True)
            v = float("nan")
            trace = {"seed_id": sid, "strategy": strategy, "turns": 0, "done": False,
                     "done_reason": None, "valence": 0.0, "emotion": None,
                     "deal_price": None, "donation_amount": None, "outcome": float("nan")}
        with lock:
            results.setdefault(f"{sid}|{strategy}", []).append(round(v, 3))
            with open(traces_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")
            term_stats.setdefault(f"{sid}|{strategy}", []).append(
                (trace["turns"], trace["done"]))
            checkpoint(results)

    tasks = [(sid, s, r) for sid in sids for s in strategies for r in range(args.n)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in ex.map(lambda t: one_episode(*t), tasks):
            pass
    for k in results:
        print(f"{k.replace('|', ' [')}] {results[k]}", flush=True)
    # 汇总：按任务 × 策略
    print("\n===== 汇总（均值 ± 标准差） =====")
    by = {}
    for k, vals in results.items():
        sid, strategy = k.split("|")
        task = _seed(sid).task.value
        clean = [v for v in vals if v == v]  # drop NaN
        by.setdefault((task, strategy), []).append((sid, clean))
    for (task, strategy), rows in sorted(by.items()):
        allv = [v for _, vs in rows for v in vs]
        if not allv:
            print(f"{task:<20} {strategy:<8} 无有效样本")
            continue
        mu = statistics.mean(allv)
        sd = statistics.stdev(allv) if len(allv) > 1 else 0.0
        print(f"{task:<20} {strategy:<8} n={len(allv):>3}  mean={mu:+.3f}  sd={sd:.3f}  "
              f"snr={mu / (sd + 1e-9):+.2f}")
    def task_of(sid: str) -> str:
        return "price_negotiation" if sid.startswith("craigslist") else (
            "persuasion_donation" if sid.startswith("p4g") else "emotional_support")

    # term_by 构建（长度与自发结束统计共用）
    from collections import defaultdict as _dd
    term_by = _dd(list)
    for k, rows in term_stats.items():
        sid, strategy = k.split("|")
        term_by[(task_of(sid), strategy)].extend(rows)

    # 长度分布对照：模拟对话 vs 语料参照（种子 notes.n_utterances/2 ≈ 轮数）
    print("\n===== 对话长度 vs 语料参照 =====")
    for (task, strategy), rows in sorted(term_by.items()):
        valid = [r[0] for r in rows if r[0] > 0]
        if not valid:
            continue
        print(f"  {task:<20} {strategy:<8} 模拟平均={sum(valid)/len(valid):.1f} 轮")
    ref = {}
    for sid in sids:
        try:
            d = json.loads((ROOT / "seeds" / f"{sid}.json").read_text(encoding="utf-8"))
            n = d.get("notes", {}).get("n_utterances")
            if n:
                ref.setdefault(task_of(sid), []).append(n / 2)
        except Exception:
            continue
    for t, vals in sorted(ref.items()):
        print(f"  语料参照 {t:<20} 平均={sum(vals)/len(vals):.1f} 轮（n={len(vals)}）")

    # 自发结束统计：done 且未跑满窗口的局 = 自发结束
    print("\n===== 自发结束（done 且 turns < 窗口） =====")
    for (task, strategy), rows in sorted(term_by.items()):
        valid = [r for r in rows if r[0] > 0]
        if not valid:
            continue
        self_end = [r for r in valid if r[1]]
        avg_turns = sum(r[0] for r in valid) / len(valid)
        print(f"  {task:<20} {strategy:<8} 自发结束率={len(self_end)/len(valid):>5.0%}  "
              f"平均轮数={avg_turns:.1f}（窗口内完成 {len(self_end)}/{len(valid)}）")

    # 效应量：good − bad（按种子配对后跨种子均值）
    print("\n===== 效应量 good−bad（按种子配对） =====")
    for task in ("price_negotiation", "persuasion_donation", "emotional_support"):
        diffs = []
        for sid, _ in by.get((task, "good"), []):
            g = [v for v in results[f"{sid}|good"] if v == v]
            b = [v for v in results.get(f"{sid}|bad", []) if v == v]
            if g and b:
                diffs.append(statistics.mean(g) - statistics.mean(b))
        if diffs:
            print(f"{task:<20} mean_diff={statistics.mean(diffs):+.3f}  "
                  f"per-seed={[round(d, 2) for d in diffs]}")
    checkpoint(results)


if __name__ == "__main__":
    main()
