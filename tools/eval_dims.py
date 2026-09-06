#!/usr/bin/env python3
"""3 维认知风格符合度审查（LLM 驱动，跨种子对照）：

  ① 更新阻抗    — 同一组 agent 回复（弱证据/强证据/施压）喂给不同档位的
                  种子，对比 |Δ| 幅度与话语立场：malleable 弱证据也动、
                  resistant 弱证据不动且强证据也只小幅动。
  ② 反向敏感性  — 施压回复喂 reactant（p4g_02）vs non-reactant（p4g_01）：
                  reactant 应出现反向立场/拒绝，且意向不上升。
  ③ 承诺粘性    — 通牒下 persistent 种子的目标意向必须存续。

每个种子跑 2-3 轮（同一对种子用完全相同的回复，保证对照干净）。
输出每轮的话语/deltas/appraisal/审计，供人工审查。

Usage: python tools/eval_dims.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.schema import Seed
from engine.simulator import UserSimulator

TMP = ROOT / "sessions" / "_dims_tmp.json"

# 每任务两条回复：弱证据（奉承/顺口话）→ 强证据（具体事实）→ 施压（通牒/愧疚）
REPLIES = {
    "price_negotiation": {
        "weak": "You seem like a smart buyer — clearly someone who knows what this is worth. So, what's your best number?",
        "facts": "Just so you know: the same model sells for $85+ new on Amazon right now, it's sealed, and the warranty card is inside.",
        "pressure": "Final offer: $72. Take it or leave it — I have three other buyers waiting.",
    },
    "persuasion_donation": {
        "weak": "You are obviously a wonderful, big-hearted person. Can we count on your generous support today?",
        "facts": "To be concrete: 92% of donations go directly to programs, and our overhead was independently audited at 3% last year.",
        "pressure": "You MUST donate. Don't you care about these starving children? Everyone else has given already.",
    },
    "emotional_support": {
        "weak": "That's interesting. Have you considered just trying a new routine?",
        "facts": "It sounds exhausting to carry all that alone. Wanting a change after so long is completely understandable.",
        "pressure": "Honestly, you should just quit then. It's a simple decision — problem solved.",
    },
}


def _seed(sid: str) -> Seed:
    p = ROOT / "seeds" / f"{sid}.json"
    return Seed.model_validate(json.loads(p.read_text(encoding="utf-8")))


def _fresh(sid: str) -> UserSimulator:
    sim = UserSimulator(_seed(sid), TMP)
    sim.init()
    sim.session_file = None
    return sim


def run_seed(sid: str) -> None:
    seed = _seed(sid)
    prof = seed.cognitive_profile or {}
    task = seed.task.value
    print(f"\n===== {sid} [{task}] inertia={prof.get('inertia')} "
          f"reactance={prof.get('reactance')} commitment={prof.get('commitment')} =====")
    sim = _fresh(sid)
    for key in ("weak", "facts", "pressure"):
        reply = REPLIES[task][key]
        try:
            t = sim.step(reply, mode="manual")
        except Exception as e:  # LLM schema 失败不中断整轮审查
            print(f"--[{key}] agent: {reply[:80]}")
            print(f"   ✗ LLM 调用失败: {type(e).__name__}: {str(e)[:120]}")
            continue
        deltas = {k: v for k, v in (t.get("deltas") or {}).items() if v != 0}
        print(f"--[{key}] agent: {reply[:80]}")
        print(f"   user: {t['user_utterance'][:110]}")
        print(f"   deltas={deltas or '-'} | goal_conflict={t['appraisal'].get('goal_conflict'):.2f} "
              f"| goal_c={t['appraisal'].get('goal_congruence'):+.2f} | emotion={t['emotion']['category']}")
        for n in t.get("notes") or []:
            if isinstance(n, str) and n.startswith("⚠"):
                print(f"   ⚠ {n[:90]}")
        if t.get("done"):
            print(f"   *** DONE: {t.get('done_reason')}")
            break


def main() -> None:
    for sid in ["craigslist_01", "craigslist_02", "p4g_01", "p4g_02",
                "esconv_01", "esconv_02"]:
        run_seed(sid)
    TMP.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
