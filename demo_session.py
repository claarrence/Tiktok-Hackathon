"""Turn-by-turn walkthrough of one dev session — for the demo video.

Usage:
    python3 demo_session.py               # default showcase session
    python3 demo_session.py public_0006   # any public_set.jsonl sample_id

Drives the real Agent through the official evaluator's customer simulator and
prints, for every turn: the shopper's message, the agent's clarifying question,
the top-5 recommendations, and where the hidden target currently ranks.
"""

from __future__ import annotations

import sys

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent

DEFAULT_SAMPLE = "public_0042"


def short(product: dict, width: int = 58) -> str:
    title = str(product.get("title") or "").strip()
    return title[:width] + ("…" if len(title) > width else "")


def main() -> None:
    sample_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SAMPLE
    samples = {s["sample_id"]: s for s in load_jsonl("data/public_set.jsonl")}
    sample = samples[sample_id]
    catalog_ids, categories, products = catalog_index("data/catalog.jsonl")

    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    target = str(sample["ground_truth"]["parent_asin"])

    agent = Agent("data/catalog.jsonl")
    agent.reset(sample_id, sample["user_profile"])

    print("=" * 78)
    print(f"SESSION {sample_id}   scenario: {sample['scenario_type']}")
    print(f"shopper profile: {sample['user_profile'].get('summary', '')}")
    print(f"hidden target : {target}  —  {short(products[target])}")
    print("=" * 78)

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

    for turn in range(1, MAX_TURNS + 1):
        response = agent.respond(sample_id, message, turn, TOP_K)
        ranked = normalize_recommendations(response["recommendations"], catalog_ids)
        rank = ranked.index(target) + 1 if target in ranked else None

        print(f"\n── turn {turn} " + "─" * 64)
        print(f"  shopper : {message}")
        ask = response["ask_attribute"]
        print(f"  agent   : {response['message']}" + (f"   [asks: {ask}]" if ask else ""))
        print(f"  top 5   :")
        for i, asin in enumerate(ranked[:5], 1):
            mark = "  <<< TARGET" if asin == target else ""
            print(f"     {i}. {asin}  {short(products[asin], 46)}{mark}")
        print(f"  target rank: {rank if rank else 'outside top 10'}")

        if rank is not None:
            print("\n" + "=" * 78)
            print(f"HIT — target reached #{rank} on turn {turn}")
            print("=" * 78)
            return
        if turn == MAX_TURNS:
            break

        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            if override.get("new_value"):
                disclosed.add(str(override["new_value"]))
            message = str(override.get("message", "Actually, ignore my earlier preference."))
        else:
            message, boundary_used = customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )

    print("\n" + "=" * 78)
    print("MISS — target did not reach the top 10 within 10 turns")
    print("=" * 78)


if __name__ == "__main__":
    main()
