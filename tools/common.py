"""Shared helpers for the three seed extractors."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SEEDS_DIR = PROJECT_ROOT / "seeds"
RAW_DIR = SEEDS_DIR / "raw"

COGWM_OUTPUT = Path("/data/user21300120/mmh/CogWM/bdi-annotation/output")


def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def contentful_turns(rec: dict) -> list[dict]:
    return [t for t in rec.get("turns", [])
            if t.get("beliefs") or t.get("desires") or t.get("intentions")]


def first_contentful(rec: dict) -> dict | None:
    for t in rec.get("turns", []):
        if t.get("beliefs") or t.get("desires") or t.get("intentions"):
            return t
    return None


def transcript(messages: list[dict], role_map: dict[str, str]) -> list[dict]:
    out = []
    for m in messages:
        text = (m.get("content") or "").strip()
        if text:
            out.append({"role": role_map.get(m["role"], m["role"]), "text": text})
    return out


def write_seed(seed: dict, dry_run: bool = False) -> None:
    path = SEEDS_DIR / f"{seed['seed_id']}.json"
    if dry_run:
        print(f"  [dry-run] would write {path}")
        return
    path.write_text(json.dumps(seed, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  wrote {path}")


def print_row(fields: list[tuple[str, str]]) -> None:
    print("  " + " | ".join(f"{k}={v}" for k, v in fields))
