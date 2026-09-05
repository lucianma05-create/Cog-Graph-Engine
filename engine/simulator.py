"""UserSimulator session: Init -> [step], transactional, session-logged.

The simulator holds the PRIVATE cognitive graph (agent never sees it) and
observes only utterances, exactly like a real interactive environment.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from engine import llm, prompts, report
from engine.schema import Appraisal, Edge, Emotion, Node, Seed
from engine.strength import clamp
from engine.updater import CognitiveGraph, apply_updates, build_initial_graph

SCHEMA_VERSION = 1

NEUTRAL_APPRAISAL = {"goal_congruence": 0.0, "controllability": 0.5,
                     "certainty": 0.5, "goal_conflict": 0.0}
NEUTRAL_EMOTION = {"category": "neutral", "valence": 0.0, "arousal": 0.2,
                   "intensity": 0.1, "appraisal_target": "#session_start"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_PRICE_RE = re.compile(r"\$\s?(\d+(?:\.\d+)?)")


def price_audit_note(task: str, user_utterance: str, prev_max_offer: float | None,
                     positive_worth_or_urgency: bool, listed_price: float | None = None) -> str | None:
    """Heuristic audit for the PRICE EXPECTATION REVISION rule: if the buyer's
    new offer exceeds the previous max offer but NO worth-belief or urgency
    desire moved positively this turn, flag the violation (the graph is the
    audit trail of price movement; a rising acceptance intention alone does
    not justify a rising price — that would be circular).

    Amounts above the listed price are ignored: they are the user QUOTING the
    seller's anchor, not making an offer."""
    if task != "price_negotiation":
        return None
    if prev_max_offer is None:
        return None  # first offer: no baseline to audit against
    amounts = [float(m) for m in _PRICE_RE.findall(user_utterance)]
    if listed_price is not None:
        amounts = [v for v in amounts if v <= listed_price * 1.05]
    if not amounts:
        return None
    new_max = max(amounts)
    if new_max <= prev_max_offer:
        return None
    if positive_worth_or_urgency:
        return None
    return (f"⚠ price offer moved up (prev max ${prev_max_offer} -> ${new_max}) "
            "with no worth-belief/urgency rise — PRICE EXPECTATION REVISION rule possibly violated")


def _prev_max_offer(task: str, history: list[dict]) -> float | None:
    """Highest $ amount the simulated user has mentioned so far."""
    if task != "price_negotiation":
        return None
    best = None
    for h in history:
        if h["role"] != "user":
            continue
        for m in _PRICE_RE.findall(h["text"]):
            v = float(m)
            best = v if best is None or v > best else best
    return best


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _clamp_appraisal(a: Appraisal) -> tuple[dict, list[str]]:
    ranges = [("goal_congruence", -1.0, 1.0), ("controllability", 0.0, 1.0),
              ("certainty", 0.0, 1.0), ("goal_conflict", 0.0, 1.0)]
    out, notes = {}, []
    for key, lo, hi in ranges:
        v, note = clamp(getattr(a, key), lo, hi)
        if note:
            notes.append(f"appraisal.{key}: {note}")
        out[key] = v
    return out, notes


def _clamp_emotion(e: Emotion) -> tuple[dict, list[str]]:
    out, notes = {}, []
    for key, lo, hi in [("valence", -1.0, 1.0), ("arousal", 0.0, 1.0), ("intensity", 0.0, 1.0)]:
        v, note = clamp(getattr(e, key), lo, hi)
        if note:
            notes.append(f"emotion.{key}: {note}")
        out[key] = v
    out["category"] = e.category.strip()
    out["appraisal_target"] = e.appraisal_target.strip()
    return out, notes


def graph_from_snapshot(snapshot: dict, dist: dict | None,
                        deactivated: list[dict] | None) -> CognitiveGraph:
    nodes: dict[str, Node] = {}
    for nd in snapshot.get("nodes", []):
        n = Node.model_validate(nd)
        nodes[n.id] = n
    deact: set[str] = set()
    for d in deactivated or []:
        nid = d["id"]
        deact.add(nid)
        if nid not in nodes:
            nodes[nid] = Node(id=nid, type=d["type"], content=d["content"], strength=0.0)
    edges: dict[tuple[str, str, str], Edge] = {}
    for e in snapshot.get("edges", []):
        edge = Edge.model_validate(e)
        key = (edge.frm, edge.to, edge.relation.value)
        if edge.relation.value == "conflicts_with":
            key = (min(key[0], key[1]), max(key[0], key[1]), key[2])
        edges[key] = edge
    graph = CognitiveGraph(nodes=nodes, edges=edges, deactivated=deact)
    for nid, probs in (dist or {}).items():
        graph.dist[nid] = list(probs)
    return graph


class UserSimulator:
    def __init__(self, seed: Seed, session_file: Path | None = None):
        self.seed = seed
        self.session_file = session_file
        self.graph = CognitiveGraph()
        self.appraisal: dict = dict(NEUTRAL_APPRAISAL)
        self.emotion: dict = dict(NEUTRAL_EMOTION)
        self.history: list[dict] = []   # {"role": "agent"|"user", "text": ...}
        self.initial_log: dict = {}
        self.turns: list[dict] = []
        self.created_at = _now()

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, seed: Seed, session_file: Path | None) -> tuple["UserSimulator", bool]:
        """Restore from a session log when present; (sim, fresh) otherwise."""
        sim = cls(seed, session_file)
        if session_file is not None and session_file.exists():
            sim._restore(json.loads(session_file.read_text(encoding="utf-8")))
            return sim, False
        return sim, True

    def _restore(self, log: dict) -> None:
        self.created_at = log.get("created_at", self.created_at)
        self.initial_log = log.get("initial", {})
        self.turns = log.get("turns", [])
        for t in self.turns:
            self.history.append({"role": "agent", "text": t["agent_reply"]})
            self.history.append({"role": "user", "text": t["user_utterance"]})
        if self.turns:
            last = self.turns[-1]
            self.graph = graph_from_snapshot(last["graph_after"],
                                             last.get("node_distributions_after"),
                                             last.get("deactivated_after"))
            self.appraisal = last["appraisal"]
            self.emotion = last["emotion"]
        elif self.initial_log:
            self.graph = graph_from_snapshot(self.initial_log["graph"],
                                             self.initial_log.get("node_distributions"),
                                             self.initial_log.get("deactivated"))
            self.appraisal = self.initial_log.get("appraisal", dict(NEUTRAL_APPRAISAL))
            self.emotion = self.initial_log.get("emotion", dict(NEUTRAL_EMOTION))

    # ---------------------------------------------------------------- actions

    def g0_cache_path(self) -> Path | None:
        """The seed graph is a stable artifact: the FIRST Init result per seed
        is persisted (sessions/g0/<seed_id>.json) and reused, so the same seed
        always starts from the same base graph. Delete that file (or pass
        regenerate=True) to force a fresh Init."""
        if self.session_file is None:
            return None
        return self.session_file.parent / "g0" / f"{self.seed.seed_id}.json"

    def init(self, regenerate: bool = False) -> dict:
        """G0 = Init(P, S, H0); resets history/turns. Uses the persisted G0
        when present."""
        cache = self.g0_cache_path()
        if cache is not None and cache.exists() and not regenerate:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            self.initial_log = cached
            self.graph = graph_from_snapshot(
                cached["graph"], cached.get("node_distributions"),
                cached.get("deactivated"))
            self.appraisal = dict(NEUTRAL_APPRAISAL)
            self.emotion = dict(NEUTRAL_EMOTION)
            self.history = []
            self.turns = []
            self._save()
            return self.initial_log

        system = prompts.build_init_system(self.seed.task)
        user_text = prompts.render_init_user(self.seed)
        out, raw = llm.generate_init(system, user_text)
        result = build_initial_graph(out)
        self.graph = result.graph
        self.initial_log = {
            "seed_id": self.seed.seed_id,
            "created_at": _now(),
            "graph": self.graph.snapshot(),
            "node_distributions": {i: p for i, p in sorted(self.graph.dist.items())},
            "deactivated": self.graph.deactivated_info(),
            "appraisal": dict(NEUTRAL_APPRAISAL),
            "emotion": dict(NEUTRAL_EMOTION),
            "llm_raw": raw,
            "ops_rejected": result.ops_rejected,
            "notes": result.notes,
        }
        if cache is not None:
            atomic_write_json(cache, self.initial_log)
        self.appraisal = dict(NEUTRAL_APPRAISAL)
        self.emotion = dict(NEUTRAL_EMOTION)
        self.history = []
        self.turns = []
        self._save()
        return self.initial_log

    def step(self, agent_reply: str, mode: str = "manual",
             system_prompt_used: str | None = None) -> dict:
        """One turn: LLM transition (1 call) -> deterministic apply. Transactional:
        on any error the in-memory state and log are untouched."""
        system = prompts.build_turn_system(self.seed.task)
        user_text = prompts.render_turn_user(
            self.seed, self.graph, self.appraisal, self.emotion, self.history, agent_reply
        )
        out, raw = llm.generate_turn(system, user_text)
        appraisal, a_notes = _clamp_appraisal(out.appraisal)
        emotion, e_notes = _clamp_emotion(out.emotion)
        # normalize the completion flag: empty reason => None when not done
        done_reason = (out.done_reason or "").strip() or None
        result = apply_updates(self.graph, out.node_updates, out.edge_updates)
        node_types = {n["id"]: n["type"] for n in result.graph.snapshot()["nodes"]}
        positive_worth_or_urgency = any(
            node_types.get(nid) in ("belief", "desire")
            for nid, d in result.deltas.items() if d > 0
        )
        audit_note = price_audit_note(
            self.seed.task.value, out.user_utterance,
            _prev_max_offer(self.seed.task.value, self.history), positive_worth_or_urgency,
            listed_price=self.seed.notes.get("listed_price"))

        turn = {
            "turn_index": len(self.turns) + 1,
            "mode": mode,
            "agent_reply": agent_reply,
            "system_prompt_used": system_prompt_used,
            "llm_raw": raw,
            "appraisal": appraisal,
            "emotion": emotion,
            "user_utterance": out.user_utterance,
            "done": bool(out.done),
            "done_reason": done_reason,
            "graph_after": result.graph.snapshot(),   # graph_before = previous turn's graph_after (redundant)
            "node_distributions_after": {i: p for i, p in sorted(result.graph.dist.items())},
            "deactivated_after": result.graph.deactivated_info(),
            "deltas": result.deltas,
            "ops_applied": result.ops_applied,
            "ops_rejected": result.ops_rejected,
            "notes": result.notes + a_notes + e_notes + ([audit_note] if audit_note else []),
            "validation": {"schema_ok": True, "retries": 0},
        }
        # commit (nothing above mutated self)
        self.graph = result.graph
        self.appraisal = appraisal
        self.emotion = emotion
        self.history.append({"role": "agent", "text": agent_reply})
        self.history.append({"role": "user", "text": out.user_utterance})
        self.turns.append(turn)
        self._save()
        return turn

    # ---------------------------------------------------------------- logging

    @property
    def done(self) -> tuple[bool, str | None]:
        """(done, reason) of the LAST turn; (False, None) before any turn."""
        if not self.turns:
            return False, None
        return bool(self.turns[-1].get("done")), self.turns[-1].get("done_reason")

    def log(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "seed_id": self.seed.seed_id,
            "task": self.seed.task.value,
            "created_at": self.created_at,
            "initial": self.initial_log,
            "turns": self.turns,
        }

    def _save(self) -> None:
        if self.session_file is None:
            return
        log = self.log()
        atomic_write_json(self.session_file, log)
        # human-readable trajectory record alongside the machine log
        report_path = self.session_file.parent / "reports" / f"{self.session_file.stem}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = report_path.with_suffix(report_path.suffix + ".tmp")
        tmp.write_text(report.render_report(self.seed, log), encoding="utf-8")
        os.replace(tmp, report_path)
