"""Extract 2 ESConv seeds from CogWM's annotated jsonl.

The seed captures the user's INHERENT pre-dialogue state:
- persona = pre-dialogue self-reports ONLY (metadata.situation / problem_type /
  emotion_type / pre-survey initial_emotion_intensity);
- u0 = the user's TRUE first utterance in the conversation (even a plain
  greeting) — NOT the later disclosure elicited by the supporter's questions.
The hindsight-annotated global_bdi never enters the seed.

Usage:
  python tools/extract_esconv_seeds.py --dry-run --limit 30
  python tools/extract_esconv_seeds.py --pick esconv_00000,esconv_00007
"""

from __future__ import annotations

import argparse
import sys

from common import COGWM_OUTPUT, SEEDS_DIR, contentful_turns, first_contentful, load_jsonl, print_row, transcript, write_seed

SOURCE = COGWM_OUTPUT / "esconv.jsonl"

MSG_LO, MSG_HI = 14, 34        # conversation length window
MIN_CONTENTFUL = 5             # nonzero cognitive trajectory (for later comparison)

# Human-judged intervention boundary: number of leading transcript utterances
# that are UNINTERVENED (neutral greetings/questions + the user's spontaneous
# self-disclosures). The supporter's first substantive strategy move (reflection
# / reframing / advice) starts right after this cut.
PREFIX_CUTS = {"esconv_00000": 5, "esconv_00012": 4}


def first_user_utterance(rec: dict) -> tuple[str, int] | None:
    """The user's genuine first message (target_role == seeker -> role user)."""
    for i, m in enumerate(rec.get("messages") or []):
        if m.get("role") == "user":
            return m["content"].strip(), i
    return None


def build_persona(rec: dict) -> str:
    meta = rec["metadata"]
    parts: list[str] = ["The user's INHERENT pre-conversation state (self-reported before the dialogue):"]
    situation = (meta.get("situation") or "").strip()
    if situation:
        parts.append(f"The user's underlying problem, in their own words: {situation}")
    problem = (meta.get("problem_type") or "").strip()
    emotion = (meta.get("emotion_type") or "").strip()
    if problem or emotion:
        parts.append(f"Presenting problem: {problem or 'n/a'} (predominant emotion: {emotion or 'n/a'}).")
    survey = (meta.get("survey_score") or {}).get("seeker") or {}
    intensity = survey.get("initial_emotion_intensity")
    if intensity is not None:
        parts.append(f"The seeker self-reported an initial emotional intensity of {intensity} out of 5 (pre-dialogue).")
    return " ".join(parts)


def build_seed(rec: dict, index: int) -> dict:
    meta = rec["metadata"]
    u0, u0_idx = first_user_utterance(rec)
    fc = first_contentful(rec)
    problem = (meta.get("problem_type") or "").strip()
    situation = (meta.get("situation") or "").strip()
    full = transcript(rec["messages"], {"user": "seeker", "assistant": "supporter"})
    cut = PREFIX_CUTS.get(rec["id"], 1)
    return {
        "seed_id": f"esconv_{index:02d}",
        "task": "emotional_support",
        "persona": build_persona(rec),
        "private_persona": None,
        "scenario": (
            "A text-chat emotional support session: the supporter listens, empathizes, "
            f"and helps the seeker explore and reappraise their problem. "
            f"The seeker's stated issue: {problem or situation or 'a personal distress'}."
        ),
        "u0": u0,
        "pre_context": full[:cut],
        "reference_transcript": full,
        "notes": {
            "source_file": str(SOURCE),
            "source_record_id": rec["id"],
            "split": meta.get("split"),
            "problem_type": problem,
            "emotion_type": meta.get("emotion_type"),
            "n_messages": len(rec["messages"]),
            "contentful_turn_count": len(contentful_turns(rec)),
            "u0_message_index": u0_idx,
            "first_contentful_turn_id": fc["turn_id"] if fc else None,
            "prefix_end_index": cut,
        },
    }


def candidates():
    for rec in load_jsonl(SOURCE):
        meta = rec.get("metadata") or {}
        if meta.get("sample_type") != "esconv" or meta.get("split") != "train":
            continue
        if not (meta.get("situation") or "").strip():   # inherent state must exist
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

    found = 0
    picked = {p.strip() for p in (args.pick or "").split(",")} - {""}
    pick_idx = 0
    for rec in candidates():
        if args.pick is not None:
            if rec["id"] not in picked:
                continue
            pick_idx += 1
            write_seed(build_seed(rec, pick_idx), args.dry_run)
            continue
        u0, _ = first_user_utterance(rec)
        print_row([
            ("id", rec["id"]),
            ("msgs", str(len(rec["messages"]))),
            ("contentful", str(len(contentful_turns(rec)))),
            ("problem", str(rec["metadata"].get("problem_type"))),
            ("emotion", str(rec["metadata"].get("emotion_type"))),
            ("situation", (rec["metadata"].get("situation") or "")[:50].replace("\n", " ")),
            ("u0", u0[:50].replace("\n", " ")),
        ])
        found += 1
        if found >= args.limit:
            break

    if args.pick is None:
        print(f"\n{found} candidates shown. Re-run with --pick <id1>,<id2> to write seeds.")
        if found == 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
