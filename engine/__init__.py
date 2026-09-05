"""Cog-Graph-Engine: lightweight BDI-graph user simulator for proactive dialogue.

Submodules are imported directly (engine.simulator, engine.updater, ...);
this package init stays minimal to keep import order trivial.
"""

from engine.schema import (
    Appraisal,
    Edge,
    Emotion,
    GraphState,
    InitOutput,
    Node,
    Seed,
    TurnOutput,
)
from engine.updater import ApplyResult, CognitiveGraph, apply_updates, build_initial_graph

__all__ = [
    "Appraisal",
    "Edge",
    "Emotion",
    "GraphState",
    "InitOutput",
    "Node",
    "Seed",
    "TurnOutput",
    "ApplyResult",
    "CognitiveGraph",
    "apply_updates",
    "build_initial_graph",
]
