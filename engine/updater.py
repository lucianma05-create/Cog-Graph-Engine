"""Deterministic graph updater — the ONLY code that writes graph state.

G_t = Apply(G_{t-1}, Updates_t). Pure functions: same input => same output.
The LLM proposes ops; this module validates, renames, normalizes, and merges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.schema import Edge, EdgeUpdate, InitOutput, Node, NodeDraft, NodeUpdate, NODE_ID_RE
from engine.strength import normalize5, strength_of

# Significance threshold: strength jitters below this magnitude are rejected as
# noise (evidence: 34% of deltas were < 0.25 in live testing). Reactivation,
# auto-deactivate and content changes bypass it.
MIN_ABS_DELTA = 0.4


# ---------------------------------------------------------------------------
# Cognitive graph storage
# ---------------------------------------------------------------------------

@dataclass
class CognitiveGraph:
    """All nodes ever created (active + deactivated), the edge set, and the
    normalized 5-bin distribution behind every strength value."""

    nodes: dict[str, Node] = field(default_factory=dict)          # id -> node (deactivated keep content, strength 0)
    edges: dict[tuple[str, str, str], Edge] = field(default_factory=dict)
    dist: dict[str, list[float]] = field(default_factory=dict)    # id -> normalized probs
    deactivated: set[str] = field(default_factory=set)

    def active_nodes(self) -> list[Node]:
        return sorted((n for i, n in self.nodes.items() if i not in self.deactivated), key=lambda n: n.id)

    def edge_list(self) -> list[Edge]:
        return sorted(self.edges.values(), key=lambda e: e.key)

    def deactivated_info(self) -> list[dict]:
        return [
            {"id": i, "type": self.nodes[i].type.value, "content": self.nodes[i].content}
            for i in sorted(self.deactivated)
        ]

    def snapshot(self) -> dict:
        """JSON shape of the ACTIVE graph, per insight.md contract."""
        return {
            "nodes": [
                {"id": n.id, "type": n.type.value, "content": n.content, "strength": n.strength}
                for n in self.active_nodes()
            ],
            "edges": [e.as_json() for e in self.edge_list()],
        }


@dataclass
class ApplyResult:
    graph: CognitiveGraph
    deltas: dict[str, float] = field(default_factory=dict)       # id -> strength delta (nonzero only)
    ops_applied: list[dict] = field(default_factory=list)        # normalized op records
    ops_rejected: list[dict] = field(default_factory=list)       # {"op": ..., "reason": ...}
    notes: list[str] = field(default_factory=list)

    @property
    def rejected(self) -> list[dict]:
        return self.ops_rejected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_free_id(letter: str, used: set[str]) -> str:
    i = 1
    while f"{letter}{i}" in used:
        i += 1
    return f"{letter}{i}"


def _valid_probs(probs: list[float] | None) -> str | None:
    if probs is None:
        return "missing level_probs"
    if len(probs) != 5:
        return "level_probs must have exactly 5 entries"
    if any(p < 0 for p in probs):
        return "level_probs contain negative values"
    if sum(probs) <= 1e-9:
        return "level_probs sum to ~0"
    return None


def _relation_legal(relation: str, src: Node, tgt: Node) -> bool:
    """Directionality rule: cognition flows forward B -> D -> I only.

    facilitates/inhibits: B->D, D->I, B->I are legal; same-level (B->B, D->D,
    I->I) and backward (D->B, I->B, I->D) pairs are rejected. means_for stays
    I->D; conflicts_with stays D<->D (symmetric, a special relation).
    """
    st, tt = src.type.value, tgt.type.value
    if relation in ("facilitates", "inhibits"):
        return (st, tt) in (("belief", "desire"), ("desire", "intention"),
                            ("belief", "intention"))
    if relation == "means_for":
        return st == "intention" and tt == "desire"
    if relation == "conflicts_with":
        return st == "desire" and tt == "desire"
    return False


def _commitment_blocks(node_id: str, nodes: dict[str, Node],
                       edges: dict[tuple[str, str, str], Edge],
                       deactivated: set[str], killed_this_turn: set[str],
                       means_for_removed: set[tuple[str, str]],
                       replacement_pairs: set[tuple[str, str]]) -> tuple[bool, str]:
    """Bratman commitment persistence, enforced deterministically: an intention
    that still serves an active desire (live means_for edge) may not be dropped
    unless the desire dies this same turn, the purpose link is severed this same
    turn, a NEW intention serving the SAME desire is adopted this turn, or the
    interaction concludes (done). Returns (blocked, reason)."""
    if _type_value(nodes[node_id]) != "intention":
        return False, ""
    for (frm, to, rel) in edges:
        if rel != "means_for" or frm != node_id:
            continue
        if (frm, to) in means_for_removed:
            continue  # purpose link severed by an edge op this turn
        if to in deactivated or to in killed_this_turn:
            continue  # the desire is gone or dies this same turn
        if any(x for (x, y) in replacement_pairs if y == to):
            continue  # a new intention adopted this turn serves this desire
        return True, (f"commitment guard: {node_id} still serves active desire {to} "
                      f"(means_for) — deactivate the desire first, replace the "
                      f"intention, or conclude the interaction")
    return False, ""


def _edge_key(frm: str, to: str, relation: str) -> tuple[str, str, str]:
    if relation == "conflicts_with":  # undirected: canonical order
        frm, to = (to, frm) if frm > to else (frm, to)
    return (frm, to, relation)


def _type_value(draft: NodeDraft):
    """Type as plain string; tolerant of unvalidated (model_construct) drafts."""
    return draft.type.value if hasattr(draft.type, "value") else str(draft.type)


def _draft_to_node(draft: NodeDraft, probs: list[float]) -> Node:
    return Node(id=draft.id, type=_type_value(draft), content=draft.content.strip(),
                strength=strength_of(probs))


def _deactivate_node(
    node_id: str,
    nodes: dict[str, Node],
    edges: dict[tuple[str, str, str], Edge],
    dist: dict[str, list[float]],
    deactivated: set[str],
    deltas: dict[str, float],
    applied: list[dict],
    auto: bool,
    reason: str = "",
) -> None:
    """Set a node inactive and cascade-remove every edge touching it."""
    old = nodes[node_id].strength
    nodes[node_id] = nodes[node_id].model_copy(update={"strength": 0.0})
    dist[node_id] = [1.0, 0.0, 0.0, 0.0, 0.0]
    deactivated.add(node_id)
    if old > 0:
        deltas[node_id] = round(-old, 3)
    applied.append(
        {"op": "deactivate", "node_id": node_id, "auto": auto,
         "reason": reason or f"strength {old} -> 0"}
    )
    for key in [k for k in edges if k[0] == node_id or k[1] == node_id]:
        e = edges.pop(key)
        applied.append({"op": "remove", "edge": e.as_json(), "auto": True,
                        "reason": f"cascade: {node_id} deactivated"})


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def apply_updates(
    graph: CognitiveGraph,
    node_updates: list[NodeUpdate],
    edge_updates: list[EdgeUpdate],
    done: bool = False,
    commitment: str = "persistent",
) -> ApplyResult:
    nodes = dict(graph.nodes)
    edges = dict(graph.edges)
    dist = dict(graph.dist)
    deactivated = set(graph.deactivated)
    deltas: dict[str, float] = {}
    applied: list[dict] = []
    rejected: list[dict] = []
    notes: list[str] = []
    commit_on = commitment == "persistent" and not done

    # --- batch pre-scan for the commitment guard: deaths and severed links ---
    killed_this_turn: set[str] = set()
    for op in node_updates:
        if op.op == "deactivate" and op.node_id:
            killed_this_turn.add(op.node_id)
        elif op.op == "update" and op.level_probs is not None:
            if _valid_probs(op.level_probs) is None:
                probs, _ = normalize5(op.level_probs)
                if strength_of(probs) <= 1e-9:
                    killed_this_turn.add(op.node_id or "")
    means_for_removed: set[tuple[str, str]] = set()
    means_for_added: set[tuple[str, str]] = set()
    for op in edge_updates:
        e = op.edge
        if e.relation.value == "means_for":
            (means_for_added if op.op == "add" else means_for_removed).add((e.frm, e.to))

    # --- node ops: all adds first (so same-turn edges can reference them) ---
    applied_intents: set[str] = set()
    for op in node_updates:
        if op.op != "add":
            continue
        draft = op.node
        if draft is None:  # guarded by pydantic; defensive
            rejected.append({"op": {"op": "add"}, "reason": "missing node"})
            continue
        err = _valid_probs(draft.level_probs)
        if err:
            rejected.append({"op": {"op": "add", "id": draft.id}, "reason": err})
            continue
        type_val = _type_value(draft)
        if not NODE_ID_RE.match(draft.id) or draft.id[0] != type_val[0].upper():
            rejected.append({"op": {"op": "add", "id": draft.id},
                             "reason": "id pattern or type-prefix mismatch"})
            continue
        if not draft.content.strip():
            rejected.append({"op": {"op": "add", "id": draft.id}, "reason": "blank content"})
            continue

        final_id = draft.id
        if final_id in nodes:  # ids are never recycled; rename deterministically
            new_id = _next_free_id(final_id[0], set(nodes))
            notes.append(f"id_renamed {final_id} -> {new_id}")
            final_id = new_id
        probs, note = normalize5(draft.level_probs)
        if note:
            notes.append(f"{final_id}: {note}")
        nodes[final_id] = _draft_to_node(draft.model_copy(update={"id": final_id}), probs)
        dist[final_id] = probs
        if type_val == "intention":
            applied_intents.add(final_id)
        applied.append({
            "op": "add", "node_id": final_id,
            "node": {"id": final_id, "type": type_val,
                     "content": draft.content.strip(), "strength": nodes[final_id].strength},
            "auto": False,
        })

    # replacement = a NEW intention adopted this turn WITH a proposed means_for
    # link to the same desire (only these unblock the commitment guard)
    replacement_pairs = {(frm, to) for (frm, to) in means_for_added
                         if frm in applied_intents}

    # --- node ops: update / deactivate, in LLM order ---
    for op in node_updates:
        if op.op not in ("update", "deactivate"):
            continue
        node_id = op.node_id or ""
        if node_id not in nodes:
            rejected.append({"op": {"op": op.op, "node_id": node_id}, "reason": "unknown node_id"})
            continue

        if op.op == "deactivate":
            if node_id in deactivated:
                rejected.append({"op": {"op": "deactivate", "node_id": node_id},
                                 "reason": "already deactivated"})
                continue
            if commit_on and node_id in nodes:
                blocked, why = _commitment_blocks(node_id, nodes, edges, deactivated,
                                                  killed_this_turn, means_for_removed,
                                                  replacement_pairs)
                if blocked:
                    rejected.append({"op": {"op": "deactivate", "node_id": node_id},
                                     "reason": why})
                    continue
            _deactivate_node(node_id, nodes, edges, dist, deactivated, deltas, applied, auto=False)
            continue

        # op == "update" (may revive a deactivated node)
        was_deactivated = node_id in deactivated
        old_strength = 0.0 if was_deactivated else nodes[node_id].strength
        new_content = op.content.strip() if op.content is not None and op.content.strip() else None

        if op.level_probs is not None:
            err = _valid_probs(op.level_probs)
            if err:
                rejected.append({"op": {"op": "update", "node_id": node_id}, "reason": err})
                continue
            probs, note = normalize5(op.level_probs)
            if note:
                notes.append(f"{node_id}: {note}")
            new_strength = strength_of(probs)
        else:
            probs = None
            new_strength = old_strength

        if probs is not None and new_strength <= 1e-9:
            # mass at level 0 => node effectively gone
            if commit_on:
                blocked, why = _commitment_blocks(node_id, nodes, edges, deactivated,
                                                  killed_this_turn, means_for_removed,
                                                  replacement_pairs)
                if blocked:
                    rejected.append({"op": {"op": "update", "node_id": node_id},
                                     "reason": why})
                    continue
            _deactivate_node(node_id, nodes, edges, dist, deactivated, deltas, applied,
                             auto=True, reason="update drove strength to 0")
            continue

        # noise guard: strength jitters below the significance threshold.
        # Pure jitters (no real content change) are REJECTED. A genuine content
        # edit is evidence of a real cognitive change — its sub-threshold
        # strength move is APPLIED (facts-driven small moves, 2026-09-05 user
        # decision). Re-sending the same content is not a bypass. Reactivation
        # bypasses the guard entirely.
        node = nodes[node_id]
        content_actually_changes = new_content is not None and new_content != node.content
        if probs is not None and not was_deactivated:
            raw_delta = round(new_strength - old_strength, 3)
            if raw_delta != 0 and abs(raw_delta) < MIN_ABS_DELTA and not content_actually_changes:
                rejected.append({
                    "op": {"op": "update", "node_id": node_id},
                    "reason": f"below significance threshold (|Δ|={abs(raw_delta):.2f} < {MIN_ABS_DELTA})",
                })
                continue

        if probs is not None:
            nodes[node_id] = node.model_copy(update={"strength": new_strength})
            dist[node_id] = probs
        if content_actually_changes:
            nodes[node_id] = nodes[node_id].model_copy(update={"content": new_content})
        if was_deactivated:
            deactivated.discard(node_id)
            notes.append(f"reactivated {node_id}")

        delta = round(new_strength - old_strength, 3)
        if delta != 0:
            deltas[node_id] = delta
        applied.append({
            "op": "update", "node_id": node_id, "auto": False,
            "strength_after": new_strength,
            "content_changed": content_actually_changes,
            **({"level_probs": probs} if probs is not None else {}),
        })

    # --- edge ops, in LLM order ---
    for op in edge_updates:
        e = op.edge
        key = _edge_key(e.frm, e.to, e.relation.value)
        frm_node, tgt_node = nodes.get(key[0]), nodes.get(key[1])

        if op.op == "add":
            if frm_node is None or tgt_node is None or key[0] in deactivated or key[1] in deactivated:
                rejected.append({"op": {"op": "add", "edge": e.as_json()},
                                 "reason": "edge endpoint missing or inactive"})
                continue
            if not _relation_legal(key[2], frm_node, tgt_node):
                rejected.append({"op": {"op": "add", "edge": e.as_json()},
                                 "reason": f"illegal relation {key[2]} for endpoint types"})
                continue
            if key in edges:
                rejected.append({"op": {"op": "add", "edge": e.as_json()}, "reason": "duplicate edge"})
                continue
            edges[key] = Edge(frm=key[0], to=key[1], relation=key[2])
            applied.append({"op": "add", "edge": {"from": key[0], "to": key[1], "relation": key[2]},
                            "auto": False})
        else:  # remove
            if key not in edges:
                if key[0] in killed_this_turn or key[1] in killed_this_turn:
                    # structural no-op: THIS turn's deactivation cascade already
                    # removed the edge — accepting keeps same-turn proposals
                    # idempotent. Edges touching historically-deactivated nodes
                    # fall through to the rejection below (they are phantom
                    # removes and must feed back to the LLM).
                    notes.append(f"edge {key[0]}-{key[2]}->{key[1]} already gone "
                                 f"(deactivation cascade)")
                    continue
                rejected.append({"op": {"op": "remove", "edge": e.as_json()}, "reason": "edge not found"})
                continue
            edges.pop(key)
            applied.append({"op": "remove", "edge": {"from": key[0], "to": key[1], "relation": key[2]},
                            "auto": False})

    new_graph = CognitiveGraph(nodes=nodes, edges=edges, dist=dist, deactivated=deactivated)
    return ApplyResult(graph=new_graph, deltas=deltas, ops_applied=applied,
                       ops_rejected=rejected, notes=notes)


def build_initial_graph(init: InitOutput) -> ApplyResult:
    """G0 = Init(P, S, u0): reuse the same deterministic path as per-turn ops."""
    node_ops = [NodeUpdate(op="add", node=d) for d in init.nodes]
    edge_ops = [EdgeUpdate(op="add", edge=e) for e in init.edges]
    return apply_updates(CognitiveGraph(), node_ops, edge_ops)
