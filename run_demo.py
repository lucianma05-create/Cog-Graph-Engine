#!/usr/bin/env python3
"""Start the BDI user-simulator demo: viewer + API on localhost.

Usage:
  python run_demo.py                       # first seed, port 8644
  python run_demo.py --seed p4g_01         # specific seed
  python run_demo.py --port 9000 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from server.app import make_server
from server.session_store import SessionStore


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", default=None, help="seed_id (default: first available)")
    ap.add_argument("--port", type=int, default=8644)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--reinit", action="store_true",
                    help="discard the persisted seed graph and regenerate G0")
    args = ap.parse_args()

    store = SessionStore(ROOT / "seeds", ROOT / "sessions")
    seeds = store.list_seeds()
    if not seeds:
        sys.exit("no seeds found — run tools/extract_*_seeds.py first")
    seed_id = args.seed or seeds[0]["seed_id"]
    if args.seed and not any(s["seed_id"] == args.seed for s in seeds):
        sys.exit(f"unknown seed {args.seed!r}; available: {[s['seed_id'] for s in seeds]}")

    # Eager Init for the selected seed so the first page load is instant.
    print(f"[init] initializing session for {seed_id} (one LLM call, cached G0) ...")
    sim = store.get_sim(seed_id)
    if args.reinit:
        print("[init] --reinit: regenerating the seed graph G0")
        sim.init(regenerate=True)
    print(f"[init] G0 ready: {len(sim.graph.active_nodes())} active node(s), "
          f"{len(sim.graph.edges)} edge(s)")

    httpd = make_server(args.host, args.port, store)
    print(f"[server] http://{args.host}:{args.port}/   (seed: {seed_id})")
    print("[server] Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] bye")


if __name__ == "__main__":
    main()
