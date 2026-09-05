/* Cog-Graph-Engine viewer: BDI graph, per-turn diffs, three driving modes. */
"use strict";

let state = null;          // full /api/state payload
let currentIdx = 0;        // 0 = initial G0; k = turn k (snapshot = turns[k-1].graph_after)
let network = null;
let editedPrompt = null;   // last user-edited agent prompt (auto_editable mode)
let selectedNodeId = null; // node whose full content is shown in the detail panel

const $ = (id) => document.getElementById(id);

/* ---------------------------------------------------------------- api */

async function api(method, path, body) {
  const resp = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!resp.ok) {
    let msg = `HTTP ${resp.status}`;
    try { const j = await resp.json(); msg = j.error || msg; } catch (_) {}
    throw new Error(msg);
  }
  return resp.json();
}

/* ------------------------------------------------------------- helpers */

function status(text, isError = false) {
  const el = $("statusLine");
  el.textContent = text;
  el.className = isError ? "error" : "";
}

function setBusy(busy) {
  for (const id of ["sendBtn", "resetBtn", "seedSelect", "prevBtn", "nextBtn", "jumpInput", "fitBtn"])
    $(id).disabled = busy;
  document.querySelectorAll('input[name="mode"]').forEach(r => (r.disabled = busy));
  $("sendBtn").textContent = busy ? "处理中…（LLM 调用）" : "发送 → 更新用户状态";
}

function snapshot(i) {
  if (i === 0) return {
    graph: state.initial.graph,
    deactivated: state.initial.deactivated || [],
    appraisal: state.initial.appraisal,
    emotion: state.initial.emotion,
    deltas: {},
  };
  const t = state.turns[i - 1];
  return {
    graph: t.graph_after,
    deactivated: t.deactivated_after || [],
    appraisal: t.appraisal,
    emotion: t.emotion,
    deltas: t.deltas || {},
    turn: t,
  };
}

function computeDiff(prev, cur) {
  const prevNodes = new Map(prev.graph.nodes.map(n => [n.id, n]));
  const curNodes = new Map(cur.graph.nodes.map(n => [n.id, n]));
  const prevDeact = new Set((prev.deactivated || []).map(d => d.id));
  const curDeact = new Set((cur.deactivated || []).map(d => d.id));
  const added = [], revived = [], changed = [], deactivated = [];
  for (const [id, n] of curNodes) {
    if (!prevNodes.has(id)) (prevDeact.has(id) ? revived : added).push(id);
    else if (prevNodes.get(id).strength !== n.strength) changed.push(id);
  }
  for (const id of curDeact) if (!prevDeact.has(id)) deactivated.push(id);
  const ekey = e => `${e.from}->${e.to}:${e.relation}`;
  const prevE = new Set(prev.graph.edges.map(ekey));
  const curE = new Set(cur.graph.edges.map(ekey));
  return {
    added, revived, changed, deactivated,
    addedEdges: [...curE].filter(k => !prevE.has(k)),
    removedEdges: [...prevE].filter(k => !curE.has(k)),
  };
}

function short(text, n) {
  return text.length <= n ? text : text.slice(0, n) + "…";
}

/* Node label: full content, wrapped by vis via widthConstraint; the full text
   is always available in the hover tooltip and the node-detail panel. */
function nodeLabel(n, suffix, cap) {
  const content = n.content.length <= cap ? n.content : n.content.slice(0, cap) + "…";
  let label = `${n.id} · ${n.strength}`;
  if (suffix) label += `\n${suffix}`;
  label += `\n${content}`;
  return label;
}

/* -------------------------------------------------------------- graph */

function renderGraph(i) {
  const snap = snapshot(i);
  const prev = i > 0 ? snapshot(i - 1) : null;
  const diff = prev ? computeDiff(prev, snap) : null;
  const oldPositions = network ? network.getPositions() : {};
  const used = new Set();

  const LABEL_CAP = 160;
  const nodes = [];
  for (const n of snap.graph.nodes) {
    used.add(n.id);
    let suffix = null, border = null, borderWidth = 2;
    const delta = snap.deltas[n.id];
    if (diff && diff.added.includes(n.id)) { border = "#27ae60"; borderWidth = 4; suffix = "＋ 新增"; }
    else if (diff && diff.revived.includes(n.id)) { border = "#16a085"; borderWidth = 4; suffix = "↻ 复活"; }
    else if (diff && diff.changed.includes(n.id)) { border = "#e67e22"; borderWidth = 3; }
    if (delta !== undefined && delta !== 0) suffix = (suffix ? suffix + " " : "") + `Δ ${delta > 0 ? "+" : ""}${delta}`;
    const node = {
      id: n.id, group: n.type, borderWidth,
      label: nodeLabel(n, suffix, LABEL_CAP),
      title: `[${n.type}] ${n.id} · strength ${n.strength}\n\n${n.content}`,
      widthConstraint: { maximum: 240 },
      font: { size: 12, multi: true, bold: { color: "#333" } },
    };
    if (border) node.color = { border, highlight: { border }, hover: { border } };
    if (oldPositions[n.id]) { node.x = oldPositions[n.id].x; node.y = oldPositions[n.id].y; }
    nodes.push(node);
  }
  for (const d of snap.deactivated) {
    if (used.has(d.id)) continue;   // never overlaps active
    let border = null, borderWidth = 2;
    if (diff && diff.deactivated.includes(d.id)) { border = "#c0392b"; borderWidth = 3; }
    const node = {
      id: d.id, group: "inactive", borderWidth,
      label: nodeLabel({ content: d.content, id: d.id, strength: 0 }, "− 失活", LABEL_CAP),
      title: `[${d.type}] ${d.id} · deactivated\n\n${d.content}`,
      widthConstraint: { maximum: 240 },
    };
    if (border) node.color = { border };
    nodes.push(node);
  }

  const REL_COLORS = { facilitates: "#3c7a2e", inhibits: "#c0392b", means_for: "#2e6da4", conflicts_with: "#b03a2e" };
  const edgeKey = e => `${e.from}->${e.to}:${e.relation}`;
  const addedEdgeSet = new Set(diff ? diff.addedEdges : []);
  const edges = snap.graph.edges.map(e => {
    const rel = e.relation;
    const isNew = addedEdgeSet.has(edgeKey(e));
    const edge = {
      from: e.from, to: e.to,
      label: rel,
      color: { color: isNew ? "#27ae60" : REL_COLORS[rel], highlight: isNew ? "#27ae60" : REL_COLORS[rel] },
      width: (isNew || rel === "means_for") ? 3 : 1.5,
      arrows: rel === "conflicts_with" ? { to: { enabled: true }, from: { enabled: true, scaleFactor: 0.7 } }
                                       : { to: { enabled: true, scaleFactor: 0.7 } },
      font: { size: 11, align: "middle", color: isNew ? "#27ae60" : "#666" },
    };
    if (rel === "inhibits") edge.dashes = [6, 4];
    return edge;
  });

  const data = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
  if (!network) {
    network = new vis.Network($("network"), data, {
      groups: {
        belief:    { color: { background: "#cfe2f3", border: "#2e6da4", highlight: { background: "#d9e9f7", border: "#2e6da4" } } },
        desire:    { color: { background: "#fde2c3", border: "#e07b00", highlight: { background: "#ffe9cf", border: "#e07b00" } } },
        intention: { color: { background: "#d4e8cf", border: "#3c7a2e", highlight: { background: "#e0f0da", border: "#3c7a2e" } } },
        inactive:  { color: { background: "#e8e8e8", border: "#999", highlight: { background: "#e8e8e8", border: "#999" } },
                     font: { color: "#888" }, shape: "box", dashes: [4, 3] },
      },
      layout: { improvedLayout: true },
      physics: {
        enabled: true, solver: "forceAtlas2Based",
        forceAtlas2Based: { gravitationalConstant: -80, springLength: 130, springConstant: 0.05 },
        stabilization: { enabled: true, iterations: 200, fit: true },
      },
      edges: { smooth: { type: "continuous", roundness: 0.5 } },
      interaction: { hover: true, tooltipDelay: 150, dragNodes: true },
    });
    network.on("click", (params) => {
      selectedNodeId = params.nodes.length ? params.nodes[0] : null;
      renderNodeDetail(selectedNodeId);
    });
    network.once("stabilizationIterationsDone", () => {
      network.setOptions({ physics: { enabled: false } });
      network.fit({ animation: true });
    });
  } else {
    network.setOptions({ physics: { enabled: true } });
    network.setData(data);
    network.once("stabilizationIterationsDone", () => {
      network.setOptions({ physics: { enabled: false } });
      network.fit({ animation: true });
    });
  }
}

/* ------------------------------------------------------------ sidebar */

function renderNodeDetail(nodeId) {
  const box = $("nodeDetailBox");
  box.textContent = "";
  if (!nodeId) {
    const hint = document.createElement("div");
    hint.className = "empty";
    hint.textContent = "点击图上的节点查看完整内容与关联边";
    box.appendChild(hint);
    return;
  }
  const snap = snapshot(currentIdx);
  const node = (snap.graph.nodes || []).find(n => n.id === nodeId)
            || (snap.deactivated || []).find(d => d.id === nodeId);
  if (!node) return;
  const deact = (snap.deactivated || []).some(d => d.id === nodeId);

  const head = document.createElement("div");
  head.style.fontWeight = "700";
  head.textContent = `${node.id} [${node.type}] · strength ${node.strength ?? 0}${deact ? " · 已失活" : ""}`;
  box.appendChild(head);

  const content = document.createElement("div");
  content.style.margin = "4px 0 6px";
  content.style.whiteSpace = "pre-wrap";   // full content, no truncation
  content.textContent = node.content;
  box.appendChild(content);

  const edges = (snap.graph.edges || []).filter(e => e.from === nodeId || e.to === nodeId);
  if (edges.length) {
    const eh = document.createElement("div");
    eh.className = "note";
    eh.textContent = "关联边：";
    box.appendChild(eh);
    for (const e of edges) {
      const row = document.createElement("div");
      row.className = "op " + e.relation;
      row.textContent = `${e.from} —${e.relation}→ ${e.to}`;
      box.appendChild(row);
    }
  }
}

function barRow(key, value, lo, hi, hue) {
  const row = document.createElement("div");
  row.className = "kv";
  const k = document.createElement("span"); k.className = "k"; k.textContent = key;
  const bar = document.createElement("div"); bar.className = "bar";
  const fill = document.createElement("i");
  const frac = Math.max(0, Math.min(1, (value - lo) / (hi - lo)));
  fill.style.width = (frac * 100).toFixed(1) + "%";
  fill.style.background = hue;
  bar.appendChild(fill);
  const v = document.createElement("span"); v.className = "v"; v.textContent = value;
  row.append(k, bar, v);
  return row;
}

function msgEl(kind, who, text, isCurrent) {
  const m = document.createElement("div");
  m.className = "msg " + kind + (isCurrent ? " current" : "");
  const w = document.createElement("span"); w.className = "who"; w.textContent = who;
  m.appendChild(w); m.appendChild(document.createTextNode(text));
  return m;
}

function renderSidebar(i) {
  const snap = snapshot(i);

  // 完整对话历史：前缀 + 全部轮次（当前轮高亮），最新轮自动滚到底部
  const dialog = $("dialogBox");
  dialog.textContent = "";
  const prefix = state.seed.pre_context && state.seed.pre_context.length
    ? state.seed.pre_context : [{ role: "user", text: state.seed.u0 }];
  const userRole = { emotional_support: "seeker", persuasion_donation: "persuadee",
                     price_negotiation: "buyer" }[state.task] || "user";
  const label = document.createElement("div");
  label.className = "note";
  label.textContent = "无干预前缀（G0 的证据来源）：";
  dialog.appendChild(label);
  for (const u of prefix) {
    dialog.appendChild(msgEl(u.role === userRole ? "user" : "agent", u.role, u.text, false));
  }
  for (const t of state.turns) {
    const cur = t.turn_index === currentIdx;
    const modeName = { manual: "手动", auto: "LLM 自动", auto_editable: "LLM 自定义" }[t.mode] || t.mode;
    dialog.appendChild(msgEl("agent", `agent 回复 · ${modeName}`, t.agent_reply, cur));
    dialog.appendChild(msgEl("user", "模拟用户回复", t.user_utterance, cur));
  }
  if (currentIdx === state.turns.length) dialog.scrollTop = dialog.scrollHeight;

  const st = $("stateBox");
  st.textContent = "";
  st.appendChild(barRow("goal_congruence", snap.appraisal.goal_congruence, -1, 1,
    snap.appraisal.goal_congruence >= 0 ? "#3c7a2e" : "#c0392b"));
  st.appendChild(barRow("controllability", snap.appraisal.controllability, 0, 1, "#2e6da4"));
  st.appendChild(barRow("goal_conflict", snap.appraisal.goal_conflict, 0, 1, "#b03a2e"));
  const em = document.createElement("div"); em.style.marginTop = "6px";
  em.textContent = `情绪: ${snap.emotion.category}　目标: ${snap.emotion.appraisal_target}`;
  st.appendChild(em);
  st.appendChild(barRow("valence", snap.emotion.valence, -1, 1,
    snap.emotion.valence >= 0 ? "#3c7a2e" : "#c0392b"));
  st.appendChild(barRow("arousal", snap.emotion.arousal, 0, 1, "#7d5ba6"));

  const ops = $("opsBox");
  ops.textContent = "";
  if (i === 0) {
    const note = document.createElement("div"); note.className = "note";
    note.textContent = `G0 由 Init 生成：${state.initial.graph.nodes.length} 个节点、${state.initial.graph.edges.length} 条边`;
    ops.appendChild(note);
    return;
  }
  const t = snap.turn;
  const items = [...(t.ops_applied || []), ...(t.ops_rejected || []).map(o => ({ ...o, rejected: true }))];
  if (!items.length) {
    const d = document.createElement("div"); d.className = "empty"; d.textContent = "本轮无更新操作（中性轮）";
    ops.appendChild(d);
  }
  for (const o of items) {
    const row = document.createElement("div");
    row.className = "op " + (o.rejected ? "reject" : o.op);
    const name = document.createElement("span"); name.className = "opname";
    name.textContent = (o.rejected ? "拒绝: " : "") + o.op + (o.auto ? " · 自动" : "");
    row.appendChild(name);
    if (o.node_id) row.appendChild(document.createTextNode(o.node_id + (o.node ? ` "${short(o.node.content, 24)}"` : "")));
    if (o.edge) row.appendChild(document.createTextNode(`${o.edge.from} -${o.edge.relation}-> ${o.edge.to}`));
    if (o.rejected && o.reason) row.appendChild(document.createTextNode(`（${o.reason}）`));
    ops.appendChild(row);
  }
  for (const n of t.notes || []) {
    const d = document.createElement("div"); d.className = "note"; d.textContent = "· " + n;
    ops.appendChild(d);
  }
}

function renderStyle() {
  const box = $("styleBox");
  box.textContent = "";
  const style = state.seed.cognitive_style;
  if (!style) {
    const d = document.createElement("div"); d.className = "empty";
    d.textContent = "（该种子未配置认知风格）";
    box.appendChild(d);
    return;
  }
  const p = document.createElement("div");
  p.style.whiteSpace = "pre-wrap";
  p.textContent = style;
  box.appendChild(p);
  if (state.seed.cognitive_profile) {
    const pr = document.createElement("div");
    pr.className = "note";
    pr.textContent = "引擎守卫：" + JSON.stringify(state.seed.cognitive_profile);
    box.appendChild(pr);
  }
}

function renderReference() {
  const box = $("refBox");
  box.textContent = "";
  for (const item of state.seed.reference_transcript || []) {
    const d = document.createElement("div"); d.className = "refitem";
    const who = document.createElement("span"); who.className = "rwho"; who.textContent = item.role;
    d.appendChild(who); d.appendChild(document.createTextNode(short(item.text, 300)));
    box.appendChild(d);
  }
}

function deltaSummary(deltas) {
  const entries = Object.entries(deltas || {});
  return entries.length
    ? entries.map(([id, d]) => `${id}${d > 0 ? "+" : ""}${d}`).join(" ")
    : "Δ 无";
}

function renderTrajectory() {
  const box = $("trajectoryBox");
  box.textContent = "";
  if (!state) return;
  const prefix = state.seed.pre_context && state.seed.pre_context.length
    ? state.seed.pre_context : [{ role: "user", text: state.seed.u0 }];
  const p0 = document.createElement("div");
  p0.className = "trajrow" + (currentIdx === 0 ? " active" : "");
  p0.textContent = `G0 前缀（${prefix.length} 句）· 点击回到初始状态`;
  p0.addEventListener("click", () => { currentIdx = 0; render(); });
  box.appendChild(p0);
  for (const t of state.turns) {
    const row = document.createElement("div");
    row.className = "trajrow" + (t.done ? " done" : "") + (currentIdx === t.turn_index ? " active" : "");
    const tn = document.createElement("span"); tn.className = "tn"; tn.textContent = `T${t.turn_index}`;
    row.appendChild(tn);
    if (t.done) {
      const badge = document.createElement("span"); badge.className = "donebadge"; badge.textContent = "✓ 完成";
      row.appendChild(badge);
    }
    row.appendChild(document.createTextNode(
      ` ${t.mode} | agent: ${short(t.agent_reply, 26)} | user: ${short(t.user_utterance, 26)}`));
    const dl = document.createElement("div"); dl.className = "tdelta";
    dl.textContent = "  " + deltaSummary(t.deltas);
    row.appendChild(dl);
    row.addEventListener("click", () => { currentIdx = t.turn_index; render(); });
    box.appendChild(row);
  }
}

/* ----------------------------------------------------------- top-level */

function render() {
  const maxIdx = state.turns.length;
  if (currentIdx > maxIdx) currentIdx = maxIdx;
  $("turnLabel").textContent = currentIdx === 0
    ? `初始状态 (G0) · 共 ${maxIdx} 轮`
    : `第 ${currentIdx} 轮 / 共 ${maxIdx} 轮`;
  $("prevBtn").disabled = currentIdx === 0;
  $("nextBtn").disabled = currentIdx >= maxIdx;
  updateModeUI();
  renderSidebar(currentIdx);
  renderGraph(currentIdx);
  renderNodeDetail(selectedNodeId);  // refresh detail after snapshot changes
  renderStyle();
  renderTrajectory();
  // completion flag
  const done = state.done && state.done.done;
  const banner = $("doneBanner");
  banner.classList.toggle("hidden", !done);
  if (done) {
    banner.textContent = `✓ 模拟器已判定会话结束：${state.done.done_reason || "（无说明）"}。无需继续交互，可重置会话重新开始。`;
  }
  $("sendBtn").disabled = !!done;
  // 仅由 done 状态控制禁用；此前 `!!done || r.disabled` 会在 done→重置后把单选钮永久锁死
  document.querySelectorAll('input[name="mode"]').forEach(r => (r.disabled = !!done));
  $("reportLink").classList.toggle("hidden", false);
  $("reportLink").href = `/api/report?seed=${encodeURIComponent(state.seed_id)}`;
  const persona = state.seed.persona;
  status(`模型 ${state.llm.model} · 种子 ${state.seed_id} · ${short(persona, 80)}`);
}

function currentMode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

function updateModeUI() {
  const mode = currentMode();
  const replyInput = $("replyInput"), promptArea = $("promptArea");
  replyInput.classList.toggle("hidden", mode !== "manual");
  promptArea.classList.toggle("hidden", mode === "manual");
  if (mode === "auto") {
    promptArea.value = state.agent_prompt_default || "";
    promptArea.readOnly = true;
  } else if (mode === "auto_editable") {
    if (editedPrompt === null) editedPrompt = state.agent_prompt_default || "";
    promptArea.value = editedPrompt;
    promptArea.readOnly = false;
  }
  $("sendBtn").textContent = mode === "manual" ? "发送 → 更新用户状态" : "生成 agent 回复 → 更新用户状态";
}

async function fetchState(seedId) {
  setBusy(true);
  try {
    state = await api("GET", `/api/state?seed=${encodeURIComponent(seedId)}`);
    currentIdx = state.turns.length;
    editedPrompt = null;
    renderReference();
    render();
  } catch (e) {
    status(`加载失败: ${e.message}`, true);
  } finally {
    setBusy(false);
  }
}

async function send() {
  const mode = currentMode();
  const body = { seed_id: state.seed_id, mode };
  if (mode === "manual") {
    body.agent_reply = $("replyInput").value.trim();
    if (!body.agent_reply) { status("请输入 agent 回复", true); return; }
  } else if (mode === "auto_editable") {
    editedPrompt = $("promptArea").value;
    body.system_prompt = editedPrompt.trim();
  }
  setBusy(true);
  status("LLM 处理中…");
  try {
    const payload = await api("POST", "/api/step", body);
    state = payload;
    currentIdx = state.turns.length;
    render();
    const last = payload.last_turn;
    if (last) status(`第 ${last.turn_index} 轮完成：${last.user_utterance.slice(0, 60)}`);
  } catch (e) {
    status(`失败: ${e.message}`, true);
  } finally {
    setBusy(false);
  }
}

async function reset() {
  if (!confirm("重置会话将丢弃当前所有轮次，并重新初始化认知图（一次 LLM 调用）。确定？")) return;
  setBusy(true);
  try {
    state = await api("POST", "/api/reset", { seed_id: state.seed_id });
    currentIdx = 0;
    editedPrompt = null;
    render();
    status("已重置，G0 重新生成");
  } catch (e) {
    status(`重置失败: ${e.message}`, true);
  } finally {
    setBusy(false);
  }
}

/* ------------------------------------------------------------ wiring */

async function init() {
  const resp = await api("GET", "/api/seeds");
  const sel = $("seedSelect");
  sel.textContent = "";
  for (const s of resp.seeds) {
    const opt = document.createElement("option");
    opt.value = s.seed_id;
    opt.textContent = `${s.seed_id} (${s.task})`;
    sel.appendChild(opt);
  }
  if (!resp.seeds.length) { status("未找到种子文件，请先运行 tools/extract_*_seeds.py", true); return; }

  document.querySelectorAll('input[name="mode"]').forEach(r =>
    r.addEventListener("change", updateModeUI));
  $("promptArea").addEventListener("input", () => { editedPrompt = $("promptArea").value; });
  $("sendBtn").addEventListener("click", send);
  $("replyInput").addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") send();
  });
  $("resetBtn").addEventListener("click", reset);
  $("prevBtn").addEventListener("click", () => { if (currentIdx > 0) { currentIdx--; render(); } });
  $("nextBtn").addEventListener("click", () => { if (currentIdx < state.turns.length) { currentIdx++; render(); } });
  $("jumpInput").addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    const v = parseInt($("jumpInput").value, 10);
    if (!isNaN(v) && v >= 0 && v <= state.turns.length) { currentIdx = v; render(); }
  });
  $("fitBtn").addEventListener("click", () => network && network.fit({ animation: true }));
  $("seedSelect").addEventListener("change", (e) => fetchState(e.target.value));

  await fetchState(resp.seeds[0].seed_id);
}

window.addEventListener("DOMContentLoaded", init);
