"""Pydantic models and Anthropic tool input-schemas.

Schema authority: shared_work_space/insight.md — nodes carry only
{id, type, content, strength}; the LLM emits strength directly as a 0-4
float (the 5-bin level_probs machinery was deleted 2026-09-06: only
strength was ever consumed). All models use extra="forbid" so any drift
(e.g. the rejected 0904 fields confidence/observability/importance/
goal_impact, or based_on/elicits edges) fails validation loudly.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

NODE_ID_RE = re.compile(r"^(B|D|I)[1-9]\d*$")


class NodeType(str, Enum):
    belief = "belief"
    desire = "desire"
    intention = "intention"


class Relation(str, Enum):
    facilitates = "facilitates"
    inhibits = "inhibits"
    means_for = "means_for"
    conflicts_with = "conflicts_with"


class Task(str, Enum):
    emotional_support = "emotional_support"
    persuasion_donation = "persuasion_donation"
    price_negotiation = "price_negotiation"


class Edge(BaseModel):
    # Field is named `frm` because `from` is a Python keyword; it is serialized
    # back to "from" via alias, matching the insight.md / JSON contract.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    frm: str = Field(alias="from")
    to: str
    relation: Relation

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.frm, self.to, self.relation.value)

    def as_json(self) -> dict[str, str]:
        return {"from": self.frm, "to": self.to, "relation": self.relation.value}


class NodeDraft(BaseModel):
    """LLM-emitted node: strength is emitted directly (0-4 float)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^(B|D|I)[1-9]\d*$")
    type: NodeType
    content: str = Field(min_length=1)
    strength: float = Field(ge=0.0, le=4.0)

    @model_validator(mode="after")
    def _check_consistency(self) -> "NodeDraft":
        self.content = self.content.strip()
        if self.id[0] != self.type.value[0].upper():
            raise ValueError("node id prefix must match type (B/D/I)")
        return self


class Node(BaseModel):
    """Engine-stored node in the persistent graph."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^(B|D|I)[1-9]\d*$")
    type: NodeType
    content: str = Field(min_length=1)
    strength: float = Field(ge=0.0, le=4.0)


class Appraisal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Minimal appraisal set (2026-09-05): certainty dropped — belief strength
    # already carries the confidence signal and the model never used it.
    goal_congruence: float
    controllability: float
    goal_conflict: float


# Closed emotion label set: CogWM's 12 categories + neutral/irritation/distrust/
# warmth (observed in live runs). Closed set => trajectories are comparable.
EMOTION_CATEGORIES = (
    "neutral", "anxiety", "sadness", "shame", "guilt", "anger", "fear",
    "loneliness", "helplessness", "confusion", "frustration", "irritation",
    "distrust", "relief", "hope", "warmth", "surprise",
)


class Emotion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Minimal emotion set (2026-09-05): intensity dropped — the circumplex
    # model (valence x arousal) already carries felt strength; intensity was
    # near-duplicate of arousal.
    category: str = Field(min_length=1)
    valence: float
    arousal: float
    appraisal_target: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_category(self) -> "Emotion":
        self.category = self.category.strip().lower()
        if self.category not in EMOTION_CATEGORIES:
            raise ValueError(
                f"emotion category {self.category!r} not in the closed set "
                f"{EMOTION_CATEGORIES}"
            )
        return self


class NodeUpdate(BaseModel):
    """One delta op the LLM proposes; the deterministic updater applies it."""

    model_config = ConfigDict(extra="forbid")

    op: str = Field(pattern=r"^(add|update|deactivate)$")
    node_id: Optional[str] = None
    node: Optional[NodeDraft] = None
    strength: Optional[float] = Field(default=None, ge=0.0, le=4.0)
    content: Optional[str] = None

    @model_validator(mode="after")
    def _check_op_requirements(self) -> "NodeUpdate":
        if self.op == "add" and self.node is None:
            raise ValueError("op=add requires 'node'")
        if self.op in ("update", "deactivate") and not self.node_id:
            raise ValueError(f"op={self.op} requires 'node_id'")
        if self.op == "update" and self.strength is None and self.content is None:
            raise ValueError("op=update requires 'strength' and/or 'content'")
        return self


class EdgeUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: str = Field(pattern=r"^(add|remove)$")
    edge: Edge


class TurnOutput(BaseModel):
    """The full per-turn LLM contract (single structured call)."""

    model_config = ConfigDict(extra="forbid")

    node_updates: list[NodeUpdate] = Field(default_factory=list)
    edge_updates: list[EdgeUpdate] = Field(default_factory=list)
    appraisal: Appraisal
    emotion: Emotion
    user_utterance: str = Field(min_length=1)
    done: bool = False
    done_reason: Optional[str] = None

    @model_validator(mode="after")
    def _strip_utterance(self) -> "TurnOutput":
        self.user_utterance = self.user_utterance.strip()
        if self.done_reason is not None:
            self.done_reason = self.done_reason.strip()
        return self


class InitOutput(BaseModel):
    """LLM contract for G0 = Init(P, S, u0)."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[NodeDraft] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)


class RewriteOutput(BaseModel):
    """Conditional-retry contract: when the engine rejected substantive ops or
    fired audit warnings, a second call rewrites ONLY the utterance (and the
    completion flag) against the REAL post-apply graph. Graph changes are
    structurally impossible here — no delta fields exist."""
    model_config = ConfigDict(extra="forbid")

    user_utterance: str = Field(min_length=1)
    done: bool = False
    done_reason: Optional[str] = None

    @model_validator(mode="after")
    def _strip(self) -> "RewriteOutput":
        self.user_utterance = self.user_utterance.strip()
        if self.done_reason is not None:
            self.done_reason = self.done_reason.strip()
        return self


class Utterance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    text: str


class Seed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed_id: str
    task: Task
    persona: str                       # public pre-dialogue facts (visible to the agent LLM)
    private_persona: Optional[str] = None  # simulator-only private state (e.g. buyer's reservation price)
    agent_private: Optional[str] = None    # agent-side private state (e.g. seller's own reservation
                                           # values) — goes ONLY into the agent context, never the simulator
    scenario: str
    u0: str                            # the user's first utterance (first user line of pre_context)
    pre_context: list[Utterance] = Field(default_factory=list)
    # ^ conversation prefix BEFORE the agent's first intervention — the
    #   user's spontaneous expressions plus neutral greetings/questions.
    #   G0 = Init(P, S, pre_context). The simulation continues from here.
    reference_transcript: list[Utterance] = Field(default_factory=list)
    notes: dict[str, Any] = Field(default_factory=dict)
    cognitive_style: Optional[str] = None
    # ^ first-person CONDITIONED update habits ("I update X only when Y"),
    #   simulator-only (never shown to the agent LLM). Reviewed by hand per seed.
    cognitive_profile: Optional[dict[str, str]] = None
    # ^ engine-only guard switches, e.g. {"commitment": "persistent"|"flexible",
    #   "reactance": "normal"|"pronounced"}. The LLM never sees these values;
    #   it only sees the NL block above.

    def simulator_persona(self) -> str:
        """Persona block for the simulator (Init + turn transitions)."""
        if self.private_persona:
            return f"{self.persona}\n\nPRIVATE (known only to the user): {self.private_persona}"
        return self.persona


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed_id: str
    mode: str = Field(pattern=r"^(manual|auto|auto_editable)$")
    agent_reply: Optional[str] = None
    system_prompt: Optional[str] = None

    @model_validator(mode="after")
    def _check_mode_requirements(self) -> "StepRequest":
        if self.mode == "manual":
            if not self.agent_reply or not self.agent_reply.strip():
                raise ValueError("mode=manual requires non-empty agent_reply")
        return self


# ---------------------------------------------------------------------------
# Anthropic tool input_schemas (hand-written, flat, no $ref).
# Kept in sync with the pydantic models above; locked by tests/test_schema.py.
# ---------------------------------------------------------------------------

_STRENGTH_DESC = (
    "Strength 0-4 directly: 0-1 weak/tentative, 2-3 clearly present, "
    ">=3.5 strong/core. The engine applies it as given."
)

_NODE_PROPS = {
    "id": {"type": "string", "pattern": "^(B|D|I)[1-9][0-9]*$",
           "description": "Fresh node id; prefix letter must match type (B/D/I). Never reuse a retired id."},
    "type": {"type": "string", "enum": ["belief", "desire", "intention"]},
    "content": {"type": "string", "minLength": 1,
                "description": "The thought AS THE USER HOLDS IT, in the user's first-person voice: beliefs as the proposition itself or 'I think/believe ...', desires as 'I want ... / I want to avoid ...', intentions as 'I will ... / I am ready to ...'. Never 'The user believes/wants/...' wrappers."},
    "strength": {"type": "number", "minimum": 0, "maximum": 4,
                 "description": _STRENGTH_DESC},
}

_EDGE_PROPS = {
    "from": {"type": "string", "description": "Source node id."},
    "to": {"type": "string", "description": "Target node id."},
    "relation": {"type": "string",
                 "enum": ["facilitates", "inhibits", "means_for", "conflicts_with"],
                 "description": "Cognition flows forward B->D->I. facilitates/inhibits: B->D (desirability), D->I (deliberation), B->I (means evaluation) ONLY — same-level and backward pairs are illegal, I->B is wishful thinking; means_for: intention->desire only (the intention's purpose — keep it while the intention persists); conflicts_with: desire<->desire only."},
}

SIMULATE_USER_TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "node_updates": {
            "type": "array",
            "description": "Deltas to persistent BDI nodes. Empty array = no cognitive change this turn (allowed).",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["add", "update", "deactivate"]},
                    "node_id": {"type": "string",
                                "description": "Required for update/deactivate. Must reference an existing ACTIVE node (update may revive a deactivated node)."},
                    "node": {
                        "type": "object",
                        "description": "Required for op=add.",
                        "properties": _NODE_PROPS,
                        "required": ["id", "type", "content", "strength"],
                        "additionalProperties": False,
                    },
                    "strength": {"type": "number", "minimum": 0, "maximum": 4,
                                 "description": "For op=update: the node's new strength."},
                    "content": {"type": "string",
                                "description": "For op=update: optional revised content; omit to keep it unchanged."},
                },
                "required": ["op"],
                "additionalProperties": False,
            },
        },
        "edge_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["add", "remove"]},
                    "edge": {
                        "type": "object",
                        "properties": _EDGE_PROPS,
                        "required": ["from", "to", "relation"],
                        "additionalProperties": False,
                    },
                },
                "required": ["op", "edge"],
                "additionalProperties": False,
            },
        },
        "appraisal": {
            "type": "object",
            "description": "Event appraisal of THIS agent reply relative to the user's active desires.",
            "properties": {
                "goal_congruence": {"type": "number", "description": "[-1,1]; NET effect on the user's DOMINANT (strongest) active desire — when the reply splits the user's desires into winners and losers, follow the dominant one."},
                "controllability": {"type": "number",
                                    "description": "[0,1]; how much agency the user feels over THEIR OWN SITUATION (job/money/relationship) — not their reaction to this reply."},
                "goal_conflict": {"type": "number", "description": "how much the reply pits the user's active desires against each other (winners vs losers; activates conflicts_with)."},
            },
            "required": ["goal_congruence", "controllability", "goal_conflict"],
            "additionalProperties": False,
        },
        "emotion": {
            "type": "object",
            "properties": {
                "category": {"type": "string",
                             "enum": list(EMOTION_CATEGORIES),
                             "description": "Closed label set; pick the single closest one."},
                "valence": {"type": "number", "description": "[-1,1]"},
                "arousal": {"type": "number", "description": "[0,1]"},
                "appraisal_target": {"type": "string", "minLength": 1,
                                     "description": "Node id most affected, or '#agent_reply', or a short phrase."},
            },
            "required": ["category", "valence", "arousal", "appraisal_target"],
            "additionalProperties": False,
        },
        "user_utterance": {
            "type": "string",
            "minLength": 1,
            "description": "The next user utterance: 1-3 natural first-person sentences, explainable by the post-update graph. When done=true this is the user's closing line.",
        },
        "done": {
            "type": "boolean",
            "description": "true ONLY when the interaction has reached a terminal outcome and no further interaction is needed: the user's core issue is resolved (emotional relief / donation decision made / deal or breakdown). Otherwise false.",
        },
        "done_reason": {
            "type": "string",
            "description": "When done=true: the user's terminal stance in one short first-person sentence ('I will take it at $80', 'I have decided to walk away'). Omit or leave empty when done=false.",
        },
    },
    "required": ["node_updates", "edge_updates", "appraisal", "emotion", "user_utterance", "done"],
    "additionalProperties": False,
}

REWRITE_TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "user_utterance": {
            "type": "string",
            "minLength": 1,
            "description": "The rewritten user utterance: 1-3 natural first-person sentences, explainable by the CURRENT GRAPH shown in the prompt. When done=true this is the user's closing line.",
        },
        "done": {"type": "boolean"},
        "done_reason": {
            "type": "string",
            "description": "When done=true: the user's terminal stance in one short first-person sentence.",
        },
    },
    "required": ["user_utterance", "done"],
    "additionalProperties": False,
}

INIT_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": _NODE_PROPS,
                "required": ["id", "type", "content", "strength"],
                "additionalProperties": False,
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": _EDGE_PROPS,
                "required": ["from", "to", "relation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["nodes", "edges"],
    "additionalProperties": False,
}

TOOL_DEFS = {
    "simulate_user_turn": {
        "name": "simulate_user_turn",
        "description": (
            "Emit the user simulator's single-turn cognitive transition: graph update ops, "
            "appraisal, emotion, and the next user utterance. The deterministic engine applies "
            "strength is a 0-4 float you emit directly."
        ),
        "input_schema": SIMULATE_USER_TURN_SCHEMA,
    },
    "rewrite_user_turn": {
        "name": "rewrite_user_turn",
        "description": (
            "Conditional retry: the previous turn's graph proposal was partially rejected or "
            "audit-flagged by the engine. The CURRENT GRAPH in the prompt is the actual state. "
            "Rewrite ONLY the user utterance (and completion flag) to be consistent with it — "
            "do not re-propose graph changes (this tool has no fields for them)."
        ),
        "input_schema": REWRITE_TURN_SCHEMA,
    },
    "initialize_cognitive_state": {
        "name": "initialize_cognitive_state",
        "description": (
            "Build the initial cognitive graph G0 from persona, scenario, and the first user "
            "utterance only. Minimal and evidence-supported; an empty graph is valid."
        ),
        "input_schema": INIT_SCHEMA,
    },
}
