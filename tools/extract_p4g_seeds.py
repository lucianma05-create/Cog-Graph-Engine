"""Extract 2 P4G (persuasionforgood) seeds from CogWM's annotated jsonl:
one case with a committed donation, one without.

The seed captures the user's INHERENT pre-dialogue state:
- persona = pre-dialogue survey values ONLY (Big Five + moral foundations +
  demographics); B2/B3/B4/B6/B7 are anonymization/round metadata — excluded;
- u0 = the persuadee's TRUE first utterance (even a plain greeting) — NOT the
  later stance expressed after the persuader's pitch.
outcome and global_bdi are hindsight — never enter the seed.

Usage:
  python tools/extract_p4g_seeds.py --dry-run --limit 30
  python tools/extract_p4g_seeds.py --pick <id1>,<id2>
"""

from __future__ import annotations

import argparse
import re
import sys

from common import COGWM_OUTPUT, contentful_turns, first_contentful, load_jsonl, print_row, transcript, write_seed

SOURCE = COGWM_OUTPUT / "persuasionforgood.jsonl"

MSG_LO, MSG_HI = 10, 30
MIN_CONTENTFUL = 4

# Human-judged intervention boundary: leading transcript utterances that are
# UNINTERVENED. The persuader's first substantive pitch/steering is the cut.
PREFIX_CUTS = {"persuasionforgood_20180904-151613_546_live": 2,
               "persuasionforgood_20180825-071757_571_live": 2}
BIG5 = ["open", "conscientious", "extrovert", "agreeable", "neurotic"]
MORALS = ["care", "fairness", "loyalty", "authority", "purity", "freedom"]
DEMO = [("age", "age"), ("sex", "sex"), ("edu", "education"), ("ideology", "political ideology")]
EXCLUDE_KEYS = {"B2", "B3", "B4", "B6", "B7"}
CHARITY_RE = re.compile(r"(Save the Children|UNICEF|children[^,.]{0,60})", re.IGNORECASE)


def first_user_utterance(rec: dict) -> tuple[str, int] | None:
    """The persuadee's genuine first message (role == user)."""
    for i, m in enumerate(rec.get("messages") or []):
        if m.get("role") == "user":
            return m["content"].strip(), i
    return None


def build_persona(p: dict) -> str:
    parts = ["The persuadee's INHERENT pre-dialogue state (self-report survey):"]
    b5 = ", ".join(f"{name} ~{p.get(name + '.x')} (1-5)" for name in BIG5 if p.get(name + ".x") is not None)
    if b5:
        parts.append(f"Big Five: {b5}.")
    morals = ", ".join(f"{name} {p.get(name + '.x')}" for name in MORALS if p.get(name + ".x") is not None)
    if morals:
        parts.append(f"Moral foundations (1-6 scale): {morals}.")
    demo = ", ".join(f"{label} {p.get(key + '.x')}" for key, label in DEMO if p.get(key + ".x") not in (None, ""))
    if demo:
        parts.append(f"Demographics: {demo}.")
    return " ".join(parts)


def build_scenario(rec: dict) -> str:
    charity = "a children-focused charity"
    for m in rec.get("messages", [])[:3]:
        if m.get("role") != "assistant":
            continue
        match = CHARITY_RE.search(m.get("content") or "")
        if match:
            charity = match.group(1).strip()
            break
    return (
        "A one-on-one text chat in which the persuader asks the persuadee to donate "
        f"to {charity}. The persuadee starts out neutral — not committed to donating."
    )


def build_seed(rec: dict, index: int) -> dict:
    fc = first_contentful(rec)
    u0, u0_idx = first_user_utterance(rec)
    persuadee = (rec["metadata"].get("participants") or {}).get("persuadee") or {}
    committed = (rec.get("outcome") or {}).get("dialogue_donation_commitment") or {}
    full = transcript(rec["messages"], {"user": "persuadee", "assistant": "persuader"})
    cut = PREFIX_CUTS.get(rec["id"], 1)
    return {
        "seed_id": f"p4g_{index:02d}",
        "task": "persuasion_donation",
        "persona": build_persona(persuadee),
        "private_persona": None,
        "scenario": build_scenario(rec),
        "u0": u0,
        "pre_context": full[:cut],
        "reference_transcript": full,
        "notes": {
            "source_file": str(SOURCE),
            "source_record_id": rec["id"],
            "split": rec["metadata"].get("split"),
            "n_messages": len(rec["messages"]),
            "contentful_turn_count": len(contentful_turns(rec)),
            "u0_message_index": u0_idx,
            "first_contentful_turn_id": fc["turn_id"] if fc else None,
            "outcome_committed": committed.get("committed"),
            "outcome_amount": committed.get("amount"),
            "prefix_end_index": cut,
        },
    }


def candidates():
    for rec in load_jsonl(SOURCE):
        if (rec.get("metadata") or {}).get("split") != "train":
            continue
        n = len(rec.get("messages") or [])
        if not (MSG_LO <= n <= MSG_HI):
            continue
        if len(contentful_turns(rec)) < MIN_CONTENTFUL:
            continue
        if first_user_utterance(rec) is None:
            continue
        yield rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--pick", default=None, help="comma-separated source record ids")
    args = ap.parse_args()

    picked = {p.strip() for p in (args.pick or "").split(",")} - {""}
    shown = {"committed": 0, "not_committed": 0}
    idx = 0
    for rec in candidates():
        ddc = (rec.get("outcome") or {}).get("dialogue_donation_commitment") or {}
        committed = bool(ddc.get("committed"))
        if args.pick is not None:
            if rec["id"] not in picked:
                continue
            idx += 1
            write_seed(build_seed(rec, idx), args.dry_run)
            continue
        bucket = "committed" if committed else "not_committed"
        if shown[bucket] >= args.limit // 2:
            continue
        shown[bucket] += 1
        print_row([
            ("id", rec["id"]),
            ("msgs", str(len(rec["messages"]))),
            ("contentful", str(len(contentful_turns(rec)))),
            ("committed", str(committed)),
            ("u0", first_user_utterance(rec)[0][:60].replace("\n", " ")),
        ])

    if args.pick is None:
        print(f"\nshown: {shown} (target: one committed + one not-committed). "
              "Re-run with --pick <id1>,<id2> to write seeds.")


if __name__ == "__main__":
    main()
