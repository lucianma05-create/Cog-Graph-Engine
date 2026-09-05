"""Human-readable trajectory report (markdown) written after every turn.

This is the audit record: the full interaction (agent + simulated user), the
per-turn graph state, applied/rejected ops, deltas, appraisal/emotion, and the
completion flag. Raw machine logs stay in sessions/<seed_id>.json.
"""

from __future__ import annotations

from engine import llm
from engine.schema import Seed

TASK_LABELS = {"emotional_support": "情感支持", "persuasion_donation": "劝说捐赠",
               "price_negotiation": "讨价还价"}


def _node_line(n: dict) -> str:
    return f"| {n['id']} | {n['type']} | {n['strength']} | {n['content']} |"


def _edge_line(e: dict) -> str:
    return f"- {e['from']} —{e['relation']}→ {e['to']}"


def _op_line(o: dict) -> str:
    if o.get("node_id"):
        desc = o["node_id"]
        if o.get("node"):
            desc += f' "{o["node"]["content"][:40]}"'
        if o.get("op") == "update":
            desc += f" → s={o.get('strength_after')}"
    elif o.get("edge"):
        e = o["edge"]
        desc = f"{e['from']} —{e['relation']}→ {e['to']}"
    else:
        desc = str(o)
    suffix = ""
    if o.get("auto"):
        suffix = "（引擎自动）"
    if o.get("reason") and o["op"] == "deactivate":
        suffix = f"（{o['reason']}）"
    return f"- {o['op']} {desc} {suffix}".strip()


def render_report(seed: Seed, log: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Cog-Graph-Engine 交互轨迹报告")
    lines.append("")
    lines.append(f"- 种子：{seed.seed_id}（{TASK_LABELS.get(seed.task.value, seed.task.value)}）")
    lines.append(f"- 生成时间：{log.get('created_at', '')}")
    lines.append(f"- 模型：{llm.MODEL}")
    lines.append("")
    lines.append("## 用户与场景")
    lines.append("")
    lines.append("### Persona（固有状态）")
    lines.append("")
    lines.append(seed.simulator_persona())
    lines.append("")
    lines.append("### 场景")
    lines.append("")
    lines.append(seed.scenario)
    if seed.agent_private:
        lines.append("")
        lines.append("### Agent 私有信息（仅进入 agent 上下文，模拟器不可见）")
        lines.append("")
        lines.append(seed.agent_private)
    lines.append("")
    lines.append("### 无干预前缀（G0 的证据来源）")
    lines.append("")
    for u in seed.pre_context or [{"role": "user", "text": seed.u0}]:
        text = u.text if hasattr(u, "text") else u["text"]
        role = u.role if hasattr(u, "role") else u["role"]
        lines.append(f"> {role}: {text}")
    lines.append("")

    init = log.get("initial") or {}
    graph = init.get("graph") or {"nodes": [], "edges": []}
    lines.append(f"## G0 基础图（{len(graph['nodes'])} 节点 / {len(graph['edges'])} 边）")
    lines.append("")
    lines.append("| 节点 | 类型 | 强度 | 内容 |")
    lines.append("|---|---|---|---|")
    for n in graph["nodes"]:
        lines.append(_node_line(n))
    for e in graph["edges"]:
        lines.append(_edge_line(e))
    if init.get("ops_rejected"):
        lines.append("")
        lines.append("Init 被拒绝操作：" + "; ".join(f"{o['op']}({o['reason']})" for o in init["ops_rejected"]))
    lines.append("")

    for t in log.get("turns", []):
        lines.append(f"## 第 {t['turn_index']} 轮（{t['mode']}）")
        lines.append("")
        lines.append(f"- agent 回复：{t['agent_reply']}")
        lines.append(f"- 模拟用户回复：{t['user_utterance']}")
        lines.append(f"- appraisal：{t['appraisal']}")
        lines.append(f"- emotion：{t['emotion']}")
        lines.append(f"- 强度变化：{t.get('deltas') or '无'}")
        done = t.get("done")
        if done:
            lines.append(f"- **完成 ✓**：{t.get('done_reason') or '（无说明）'}")
        if t.get("ops_applied"):
            lines.append("- 已应用操作：")
            for o in t["ops_applied"]:
                lines.append(f"  {_op_line(o)}")
        if t.get("ops_rejected"):
            lines.append("- 被拒绝操作：")
            for o in t["ops_rejected"]:
                lines.append(f"  - {o['op']}（{o['reason']}）")
        if t.get("notes"):
            lines.append("- 备注：" + "；".join(t["notes"]))
        g = t.get("graph_after") or {"nodes": [], "edges": []}
        lines.append(f"- 当前图（{len(g['nodes'])} 节点 / {len(g['edges'])} 边）：")
        for n in g["nodes"]:
            lines.append(f"  {n['id']} [{n['type']} s={n['strength']}] {n['content'][:60]}")
        if g["edges"]:
            for e in g["edges"]:
                lines.append(f"  {e['from']} —{e['relation']}→ {e['to']}")
        lines.append("")
    return "\n".join(lines)
