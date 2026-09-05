"""The persuasive/supportive agent role (auto / auto_editable modes).

The agent NEVER sees the cognitive graph — the graph is the simulator's
private state; the agent only observes utterances, like in reality.
"""

from __future__ import annotations

from engine import llm, prompts
from engine.schema import Seed


def default_system_prompt(task) -> str:
    return prompts.AGENT_DEFAULT[task]


def generate_agent_reply(seed: Seed, history: list[dict], system_prompt: str) -> str:
    user_text = prompts.render_agent_user(seed, history)
    return llm.generate_text(system=system_prompt, user_text=user_text)
