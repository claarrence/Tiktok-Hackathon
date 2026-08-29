"""Adaptive route-weight shaping — the applied half of Pillar III.

``adaptive_weights`` takes the distilled :class:`~context_engine.distiller.ContextVector`
and returns the per-route weight vector the ranker fuses with. The first block
reproduces the original intent-only rule table exactly; the second block layers
on the session trajectory (``precision_bias``) and the buyer's profile
(``demandingness``, ``tag_focus``, ``price_sensitivity``). With every ``PARAMS``
coefficient at 0 — and the distiller emitting a neutral vector — the second block
is a no-op and this function is byte-identical to the pre-pillar behaviour.
"""

from __future__ import annotations

from context_engine.params import PARAMS

BASE_WEIGHTS = {
    "keyword": 0.25,
    "category": 0.15,
    "vector": 0.15,
    "slot": 0.15,
    "tag": 0.05,
    "phrase": 0.17,
    "price": 0.08,
}


def _legacy_shape(weights: dict[str, float], intent: str, broaden: bool, critical: bool) -> None:
    """The original static rule table (kept verbatim for a safe fallback)."""
    if intent == "buying":
        weights["slot"] += 0.15
        weights["category"] += 0.05
        weights["vector"] -= 0.10
        weights["keyword"] -= 0.10
    else:
        weights["vector"] += 0.10
        weights["category"] -= 0.05
        weights["slot"] -= 0.05

    if broaden:
        weights["category"] = max(0.0, weights["category"] - 0.15)
        weights["vector"] += 0.10
        weights["keyword"] += 0.05

    if critical:
        weights["slot"] += 0.05


def adaptive_weights(context) -> dict[str, float]:
    """Runtime workflow re-orchestration: reshapes route weights per turn from
    the intent track, the session trajectory distilled so far, and the buyer's
    long-term profile — the 'self-evolution' pillar."""
    weights = dict(BASE_WEIGHTS)
    p = PARAMS

    _legacy_shape(weights, context.intent, context.broaden, context.rating_style_critical)

    # --- trajectory: earned shift from recall routes toward precision routes ---
    pb = context.precision_bias
    if pb:
        weights["slot"] += p["w_slot"] * pb
        weights["phrase"] += p["w_phrase"] * pb
        weights["vector"] -= p["w_vector"] * pb
        weights["keyword"] -= p["w_keyword"] * pb
        weights["category"] -= p["w_category"] * pb

    # --- profile: a more exacting buyer leans harder on exact matches ---
    demand = context.demandingness
    if demand > 0.5:
        weights["slot"] += p["w_demand_slot"] * (demand - 0.5)
        weights["phrase"] += p["w_demand_phrase"] * (demand - 0.5)

    # --- personalization modulation, centred so 0.5 == no change ---
    if context.tag_focus != 0.5:
        weights["tag"] = max(0.0, weights["tag"] + p["w_tag"] * (context.tag_focus - 0.5))
    if context.has_budget and context.price_sensitivity != 0.5:
        weights["price"] = max(
            0.0, weights["price"] + p["w_price"] * (context.price_sensitivity - 0.5)
        )

    total = sum(max(0.0, value) for value in weights.values()) or 1.0
    return {key: max(0.0, value) / total for key, value in weights.items()}
