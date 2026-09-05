"""Extract 2 CraigslistBargain seeds (simulated user = buyer, agent = seller).

Data source: the official CraigslistBargain train split (Codalab bundle,
cached at seeds/raw/craigslist_train_parsed.json; downloaded automatically).

The buyer's private Target/Bottomline go into private_persona (simulator-only,
never shown to the agent LLM).

Usage:
  python tools/extract_craigslist_seeds.py --dry-run --limit 30
  python tools/extract_craigslist_seeds.py --pick <uuid1>,<uuid2>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import RAW_DIR, print_row, write_seed

URL = "https://worksheets.codalab.org/rest/bundles/0xd34bbbc5fb3b4fccbd19e10756ca8dd7/contents/blob/parsed.json"
CACHE = RAW_DIR / "craigslist_train_parsed.json"

UTT_LO, UTT_HI = 10, 26
MIN_PER_ROLE = 3
MIN_U0_WORDS = 6
MIN_PRICE_CHANGES = 2


def ensure_data() -> Path:
    if CACHE.exists():
        return CACHE
    print(f"downloading CraigslistBargain train split ({URL}) ...")
    import requests
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL, timeout=300)
    resp.raise_for_status()
    CACHE.write_bytes(resp.content)
    print(f"cached {CACHE} ({len(resp.content)} bytes)")
    return CACHE


def iter_rows():
    data = json.loads(ensure_data().read_text(encoding="utf-8"))
    for j in data:
        try:
            yield parse_row(j)
        except Exception as e:  # one bad row must not kill the scan
            print(f"  [warn] skipping row: {e}")
            continue


def parse_row(j: dict) -> dict:
    kbs = (j.get("scenario") or {}).get("kbs") or []
    roles = {i: (kb.get("personal") or {}).get("Role") for i, kb in enumerate(kbs)}
    buyer = next((i for i, r in roles.items() if r == "buyer"), None)
    seller = next((i for i, r in roles.items() if r == "seller"), None)
    if buyer is None or seller is None:
        raise ValueError("missing buyer/seller role")
    item = (kbs[seller].get("item") or {}) or {}
    buyer_personal = (kbs[buyer].get("personal") or {}) or {}
    seller_personal = (kbs[seller].get("personal") or {}) or {}

    events = [e for e in (j.get("events") or [])
              if e.get("action") == "message" and str(e.get("data") or "").strip()]
    utterances = [(int(e.get("agent")), str(e.get("data")).strip()) for e in events]

    price = item.get("Price")
    price_changes = sum(
        1 for e in events
        if (e.get("metadata") or {}).get("price") is not None
        and (e.get("metadata") or {}).get("price") != price
    )
    return {
        "uuid": j.get("uuid"),
        "item": item,
        "buyer": buyer_personal,
        "seller": seller_personal,
        "buyer_idx": buyer,
        "seller_idx": seller,
        "utterances": utterances,
        "price_changes": price_changes,
    }


def build_transcript(row: dict) -> list[dict]:
    out = []
    for agent, text in row["utterances"]:
        out.append({"role": "buyer" if agent == row["buyer_idx"] else "seller", "text": text})
    return out


def build_seed(row: dict, index: int) -> dict:
    # The buyer's opening message is their spontaneous, unintervened state;
    # negotiation (the intervention) starts with the seller's first reply.
    cut = 1
    item = row["item"]
    buyer = row["buyer"]
    seller = row["seller"]
    desc = " ".join(item.get("Description") or []) if isinstance(item.get("Description"), list) else str(item.get("Description") or "")
    target = buyer.get("Target")
    bottomline = buyer.get("Bottomline")
    private = (
        f"The buyer's private negotiation position: target price ~${target}; "
        f"bottom line (lowest acceptable): "
        f"${bottomline}." if bottomline is not None else
        f"The buyer's private negotiation position: target price ~${target}; no hard bottom line set."
    )
    s_target = seller.get("Target")
    s_bottomline = seller.get("Bottomline")
    agent_private = (
        f"Your listed price is ${item.get('Price')}; your target price is ~${s_target}; "
        f"your bottom line (below which you walk away): "
        f"${s_bottomline}." if s_bottomline is not None else
        f"Your listed price is ${item.get('Price')}; your target price is ~${s_target}; no hard bottom line set."
    )
    category = str(item.get("Category") or "item")
    title = str(item.get("Title") or "").strip()
    price = item.get("Price")
    return {
        "seed_id": f"craigslist_{index:02d}",
        "task": "price_negotiation",
        "persona": "The user is the BUYER, negotiating to buy a second-hand item from a private seller.",
        "private_persona": private,
        "agent_private": agent_private,
        "scenario": (
            f"The SELLER has listed a {category} item"
            + (f": \"{title}\"" if title else "")
            + f" for ${price}. Listing description: {desc[:300] or '(no description)'} "
            "The buyer opens the negotiation over the price."
        ),
        "u0": next(text for agent, text in row["utterances"] if agent == row["buyer_idx"]),
        "pre_context": build_transcript(row)[:cut],
        "reference_transcript": build_transcript(row),
        "notes": {
            "source_file": str(CACHE),
            "source_record_id": row["uuid"],
            "category": category,
            "title": title,
            "listed_price": price,
            "buyer_target": target,
            "buyer_bottomline": bottomline,
            "seller_target": s_target,
            "seller_bottomline": s_bottomline,
            "n_utterances": len(row["utterances"]),
            "price_changes": row["price_changes"],
            "prefix_end_index": cut,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--pick", default=None, help="comma-separated uuids")
    args = ap.parse_args()

    picked = {p.strip() for p in (args.pick or "").split(",")} - {""}
    seen_categories: set[str] = set()
    shown = 0
    idx = 0
    for row in iter_rows():
        utts = row["utterances"]
        if not (UTT_LO <= len(utts) <= UTT_HI):
            continue
        buyer_utts = [t for a, t in utts if a == row["buyer_idx"]]
        seller_utts = [t for a, t in utts if a == row["seller_idx"]]
        if len(buyer_utts) < MIN_PER_ROLE or len(seller_utts) < MIN_PER_ROLE:
            continue
        if len(buyer_utts[0].split()) < MIN_U0_WORDS:
            continue
        if row["price_changes"] < MIN_PRICE_CHANGES:
            continue
        if row["item"].get("Price") is None or float(row["item"].get("Price")) <= 0:
            continue

        if args.pick is not None:
            if row["uuid"] not in picked:
                continue
            idx += 1
            write_seed(build_seed(row, idx), args.dry_run)
            continue

        category = str(row["item"].get("Category") or "item")
        if category in seen_categories:  # prefer category diversity in the pair
            continue
        seen_categories.add(category)
        shown += 1
        print_row([
            ("uuid", row["uuid"][:22]),
            ("cat", category),
            ("price", str(row["item"].get("Price"))),
            ("utts", str(len(utts))),
            ("buyer", str(len(buyer_utts))),
            ("seller", str(len(seller_utts))),
            ("price_changes", str(row["price_changes"])),
            ("buyer_target", str(row["buyer"].get("Target"))),
            ("bottomline", str(row["buyer"].get("Bottomline"))),
            ("u0", buyer_utts[0][:60]),
        ])
        if shown >= args.limit:
            break

    if args.pick is None:
        print(f"\n{shown} candidates shown. Re-run with --pick <uuid1>,<uuid2> to write seeds.")
        if shown == 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
