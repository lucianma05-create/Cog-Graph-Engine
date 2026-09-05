"""Session persistence + per-seed locks shared by the HTTP server."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from engine.schema import Seed
from engine.simulator import UserSimulator


class SessionStore:
    def __init__(self, seeds_dir: Path, sessions_dir: Path):
        self.seeds_dir = Path(seeds_dir)
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._sims: dict[str, UserSimulator] = {}

    # ------------------------------------------------------------ helpers

    def _lock(self, seed_id: str) -> threading.Lock:
        return self._locks.setdefault(seed_id, threading.Lock())

    def session_path(self, seed_id: str) -> Path:
        return self.sessions_dir / f"{seed_id}.json"

    def load_seed(self, seed_id: str) -> Seed:
        p = self.seeds_dir / f"{seed_id}.json"
        if not p.exists():
            raise KeyError(seed_id)
        return Seed.model_validate(json.loads(p.read_text(encoding="utf-8")))

    def list_seeds(self) -> list[dict]:
        out = []
        for p in sorted(self.seeds_dir.glob("*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            out.append({
                "seed_id": data["seed_id"],
                "task": data["task"],
                "notes": data.get("notes", {}),
            })
        return out

    # ------------------------------------------------------------ actions

    def acquire(self, seed_id: str) -> threading.Lock | None:
        """Non-blocking per-seed lock; None means the seed is busy."""
        lock = self._lock(seed_id)
        return lock if lock.acquire(blocking=False) else None

    def get_sim(self, seed_id: str, auto_init: bool = True) -> UserSimulator:
        """Load or create the simulator for a seed; runs Init when there is no
        session log yet (one LLM call). Call under the seed lock."""
        if seed_id in self._sims:
            return self._sims[seed_id]
        seed = self.load_seed(seed_id)
        sim, fresh = UserSimulator.load(seed, self.session_path(seed_id))
        if fresh and auto_init:
            sim.init()
        self._sims[seed_id] = sim
        return sim

    def reset_sim(self, seed_id: str, regenerate: bool = False) -> UserSimulator:
        """Discard the session and re-Init. The seed graph (G0) is reused from
        its persisted artifact unless regenerate=True. Call under the seed lock."""
        seed = self.load_seed(seed_id)
        sim = UserSimulator(seed, self.session_path(seed_id))
        sim.init(regenerate=regenerate)
        self._sims[seed_id] = sim
        return sim
