#!/usr/bin/env python3
"""Deterministic session-log analyzer for the BDI-E simulator's structural
quality metrics. No LLM calls — pure bookkeeping over session JSON logs.

Metrics (see insight.md §8 and the 2026-09-05 evaluation discussion):
  1. 结构健康度  — schema-class rejections per turn (illegal relation, bad
     ids, missing endpoints...). Target: 0.
  2. 因果纪律    — ⚠ audit flags per turn: price (PRICE EXPECTATION REVISION),
     reactance, edge-contradiction. These are the deterministic proxy for
     "stance shifts traceable to graph changes".
  3. 噪声纪律    — benign rejections per turn (sub-threshold jitter, duplicate
     edge, cascade no-op). High counts = LLM drift, not schema breakage.
  4. 结局        — done rate and first-person done_reason.

Usage:
  python tools/eval_session.py                    # sessions/*.json only
  python tools/eval_session.py sessions/*.replay.json
  python tools/eval_session.py --all              # main + .smoke + .replay
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

SCHEMA_REASONS = ("illegal relation", "schema", "pattern", "missing", "unknown node_id",
                  "endpoint", "id pattern", "id_renamed", "level_probs", "blank content")
BENIGN_REASONS = ("below significance threshold", "duplicate", "already deactivated",
                  "edge not found", "cascade", "commitment guard")

AUDIT_KINDS = {
    "price": "PRICE EXPECTATION REVISION",
    "reactance": "reactance expected",
    "propagation": "contradict edge",
}


def classify_rejection(reason: str) -> str:
    if reason.startswith("commitment guard"):
        return "commitment-guard"
    if any(r in reason for r in BENIGN_REASONS):
        return "benign"
    if any(r in reason for r in SCHEMA_REASONS):
        return "schema"
    return "other"


def analyze(path: Path) -> dict | None:
    try:
        log = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    stats = Counter()
    turns = log.get("turns") or []
    n_audit = Counter()
    for t in turns:
        for r in t.get("ops_rejected") or []:
            stats[classify_rejection(str(r.get("reason", "")))] += 1
        for n in t.get("notes") or []:
            if isinstance(n, str) and n.startswith("⚠"):
                for kind, marker in AUDIT_KINDS.items():
                    if marker in n:
                        n_audit[kind] += 1
                        break
                else:
                    n_audit["other"] += 1
    init_rej = [classify_rejection(str(r.get("reason", "")))
                for r in (log.get("initial") or {}).get("ops_rejected") or []]
    for c in init_rej:
        stats[c] += 1
    done_turns = [t for t in turns if t.get("done")]
    last_done_reason = done_turns[-1].get("done_reason") if done_turns else None
    return {
        "file": path.name,
        "task": log.get("task", "?"),
        "turns": len(turns),
        "nodes_g0": len((log.get("initial") or {}).get("graph", {}).get("nodes", [])),
        "rej_schema": stats["schema"],
        "rej_benign": stats["benign"],
        "rej_commit": stats["commitment-guard"],
        "rej_other": stats["other"],
        "audit": dict(n_audit),
        "done": len(done_turns),
        "done_reason": last_done_reason,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("patterns", nargs="*", default=["sessions/*.json"])
    ap.add_argument("--all", action="store_true",
                    help="main + .smoke + .replay logs")
    args = ap.parse_args()
    patterns = list(args.patterns)
    if args.all:
        patterns = ["sessions/*.json", "sessions/*.smoke.json", "sessions/*.replay.json"]

    files = sorted({p for pat in patterns for p in glob.glob(pat)})
    rows = [r for r in (analyze(Path(f)) for f in files) if r]
    if not rows:
        sys.exit("no session logs found")

    print(f"{'file':<26} {'task':<20} {'turns':>5} {'G0':>3} {'schema':>6} "
          f"{'benign':>6} {'commit':>6} {'⚠audit':>7} {'done':>4}")
    tot = Counter()
    for r in sorted(rows, key=lambda r: r["file"]):
        print(f"{r['file']:<26} {r['task']:<20} {r['turns']:>5} {r['nodes_g0']:>3} "
              f"{r['rej_schema']:>6} {r['rej_benign']:>6} {r['rej_commit']:>6} "
              f"{sum(r['audit'].values()):>7} {r['done']:>4}")
        tot["turns"] += r["turns"]
        tot["schema"] += r["rej_schema"]
        tot["benign"] += r["rej_benign"]
        tot["commit"] += r["rej_commit"]
        tot["audit"] += sum(r["audit"].values())
        tot["done"] += r["done"]
    print("-" * 86)
    n = max(1, tot["turns"])
    print(f"{'TOTAL':<26} {'':<20} {tot['turns']:>5} {'':>3} {tot['schema']:>6} "
          f"{tot['benign']:>6} {tot['commit']:>6} {tot['audit']:>7} {tot['done']:>4}")
    print(f"  schema rejections/turn: {tot['schema'] / n:.3f}   "
          f"⚠ audit flags/turn: {tot['audit'] / n:.3f}")
    audit_detail = Counter()
    for r in rows:
        for k, v in r["audit"].items():
            audit_detail[k] += v
    if audit_detail:
        print(f"  ⚠ breakdown: {dict(audit_detail)}")
    for r in rows:
        if r["done_reason"]:
            print(f"  done[{r['file']}]: {r['done_reason']}")


if __name__ == "__main__":
    main()
