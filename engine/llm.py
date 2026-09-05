"""Thin anthropic wrapper: forced tool_choice, pydantic validation, 1 retry.

Backend: Claude via the environment's proxy (ANTHROPIC_BASE_URL +
ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY); model from ANTHROPIC_MODEL.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel, ValidationError

from engine.schema import InitOutput, TOOL_DEFS, TurnOutput

# ---------------------------------------------------------------------------
# Config: env vars, optionally overridden by a project-local .env file
# (KEY=VALUE lines; missing keys fall back to the process environment).
# ---------------------------------------------------------------------------

_ENV_FILE = Path(__file__).parent.parent / ".env"


def _load_dotenv(path: Path) -> None:
    """Project .env is THE documented config file and WINS over inherited
    shell env (a stale shell export of ANTHROPIC_MODEL shadowed .env in
    practice — 2026-09-05). Keys missing in .env fall back to the process
    environment."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


_load_dotenv(_ENV_FILE)

MODEL = (
    os.environ.get("ANTHROPIC_MODEL")
    or os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL")
    or "claude-sonnet-4-6"
)
# G0 construction is one-shot and cached forever -> strongest model.
# Per-turn transitions + agent replies are the high-frequency path -> MODEL.
INIT_MODEL = os.environ.get("ANTHROPIC_INIT_MODEL") or MODEL

MAX_TOKENS_STRUCT = 4000
MAX_TOKENS_TEXT = 600

_RETRY_SUFFIX = (
    "\n\nNOTE: your previous output failed JSON-schema validation with the error below. "
    "Re-emit it strictly matching the tool's input_schema (no extra fields, no "
    "explanations):\n{err}"
)


class ConfigError(RuntimeError):
    pass


class SchemaError(RuntimeError):
    """Both structured-call attempts failed validation."""


def _client() -> anthropic.Anthropic:
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not base_url or not api_key:
        raise ConfigError(
            "ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN (or ANTHROPIC_API_KEY) "
            "must be set in the environment."
        )
    return anthropic.Anthropic(api_key=api_key, base_url=base_url, max_retries=2)


def _forced_tool_call(tool_name: str, system: str, user_text: str, model: str | None = None,
                      temperature: float | None = None) -> dict[str, Any]:
    client = _client()
    kwargs: dict[str, Any] = {
        "model": model or MODEL,
        "max_tokens": MAX_TOKENS_STRUCT,
        "system": system,
        # The proxy model defaults to thinking mode, which rejects forced
        # tool_choice; structured transitions do not need thinking.
        "thinking": {"type": "disabled"},
        "tools": [TOOL_DEFS[tool_name]],
        "tool_choice": {"type": "tool", "name": tool_name},
        "messages": [{"role": "user", "content": user_text}],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = client.messages.create(**kwargs)
    for block in resp.content:
        if block.type == "tool_use" and block.name == tool_name:
            return block.input
    raise SchemaError(f"model did not emit the {tool_name} tool call")


def call_structured(
    tool_name: str,
    model_cls: type[BaseModel],
    system: str,
    user_text: str,
    temperature: float | None = None,
    model: str | None = None,
) -> tuple[BaseModel, dict[str, Any]]:
    """One structured call with a single validation retry.

    Returns (validated_model, llm_raw) where llm_raw is the raw tool input.
    Raises SchemaError after the second failure.
    """
    errors: list[str] = []
    for attempt in range(2):
        try:
            raw = _forced_tool_call(tool_name, system, user_text,
                                    temperature=temperature, model=model)
            return model_cls.model_validate(raw), raw
        except ValidationError as e:
            errors.append(str(e)[:600])
        except anthropic.APIError as e:
            errors.append(str(e)[:600])
        if attempt == 0:
            user_text = user_text + _RETRY_SUFFIX.format(err=errors[-1])
    raise SchemaError("; ".join(errors))


def generate_turn(system: str, user_text: str) -> tuple[TurnOutput, dict[str, Any]]:
    out, raw = call_structured("simulate_user_turn", TurnOutput, system, user_text)
    return out, raw  # type: ignore[return-value]


def generate_init(system: str, user_text: str) -> tuple[InitOutput, dict[str, Any]]:
    """G0 construction is a deterministic task: temperature 0 + the strongest
    model (ANTHROPIC_INIT_MODEL), plus the engine persists the first result per
    seed so the seed graph is stable."""
    out, raw = call_structured("initialize_cognitive_state", InitOutput, system, user_text,
                               temperature=0.0, model=INIT_MODEL)
    return out, raw  # type: ignore[return-value]


def generate_text(system: str, user_text: str) -> str:
    """Free-text completion (agent auto replies)."""
    client = _client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS_TEXT,
        system=system,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": user_text}],
    )
    parts = [b.text for b in resp.content if b.type == "text" and b.text]
    if not parts:
        raise SchemaError("model returned no text")
    return "".join(parts).strip()
